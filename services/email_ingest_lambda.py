"""Event-driven processing of inbound supplier emails stored in S3.

This module implements the Lambda handler described in the Option A design
notes.  SES writes raw ``.eml`` payloads into ``s3://procwisemvp/emails/`` with
opaque filenames.  Each put event fans out via SQS and triggers the handler
below which:

* Downloads the raw email object and parses RFC-2047 headers safely.
* Extracts the ``rfq_id`` from a dedicated header, subject token, thread map, or
  as a last resort from the textual body of the message.
* Tags the original object and copies it into ``emails/{rfq_id}/ingest/`` so
  downstream processors have a deterministic prefix to scan.
* Moves unmatched emails into ``emails/_unmatched/`` for manual triage.

The implementation intentionally mirrors the pseudo-code shared with the
product team so ops can deploy the Lambda with minimal glue code.
"""

from __future__ import annotations

import json
import logging
import os
import re
from email import message_from_bytes
from email.header import decode_header, make_header
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote_plus

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


DEFAULT_BUCKET = os.getenv("EMAIL_INGEST_BUCKET", os.getenv("S3_BUCKET", "procwisemvp"))
DEFAULT_PREFIX = os.getenv("EMAIL_INGEST_PREFIX", "emails/")
THREAD_TABLE = os.getenv("EMAIL_THREAD_TABLE", os.getenv("DDB_TABLE", "procwise_outbound_emails"))

_BOTO_CONFIG = Config(retries={"max_attempts": 5, "mode": "standard"})
_S3_CLIENT = None
_DDB_TABLE = None

RFQ_SUBJECT_RE = re.compile(r"\bRFQ[-_](\d{8})[-_]?([A-Za-z0-9\-]+)", re.IGNORECASE)
RFQ_HTML_COMMENT_RE = re.compile(r"<!--\s*RFQ-ID\s*:\s*([A-Za-z0-9_-]+)\s*-->", re.IGNORECASE)


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3", config=_BOTO_CONFIG)
    return _S3_CLIENT


def _get_thread_table():
    global _DDB_TABLE
    if _DDB_TABLE is None:
        resource = boto3.resource("dynamodb", config=_BOTO_CONFIG)
        _DDB_TABLE = resource.Table(THREAD_TABLE)
    return _DDB_TABLE


def _decode_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        header = make_header(decode_header(value))
    except Exception:  # pragma: no cover - extremely defensive
        return value.strip()
    return str(header).strip()


def _normalise_rfq(date: str, token: str) -> str:
    return f"RFQ-{date}-{token}".upper()


def _extract_rfq_from_subject(subject: str) -> Optional[str]:
    if not subject:
        return None
    match = RFQ_SUBJECT_RE.search(subject)
    if not match:
        return None
    date, token = match.groups()
    return _normalise_rfq(date, token)


def _lookup_rfq_from_thread(in_reply_to: Optional[str], references: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    for value in (in_reply_to, references):
        if not value:
            continue
        candidates.extend(re.findall(r"<[^>]+>", value))

    if not candidates:
        return None

    table = _get_thread_table()
    for candidate in candidates:
        key = candidate.strip()
        if not key:
            continue
        try:
            result = table.get_item(Key={"message_id": key})
        except ClientError:  # pragma: no cover - propagation would retry via SQS
            logger.exception("Failed to query thread map for message %s", key)
            continue
        item = result.get("Item") if isinstance(result, dict) else None
        if item and item.get("rfq_id"):
            return str(item["rfq_id"]).upper()
    return None


def _strip_html(value: str) -> str:
    if not value:
        return ""

    comment_tokens = RFQ_HTML_COMMENT_RE.findall(value)
    cleaned = re.sub(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", " ", value, flags=re.I | re.S)
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if comment_tokens:
        return " ".join(comment_tokens + [cleaned]).strip()
    return cleaned


def _iter_text_parts(message) -> Iterable[str]:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True) or b""
                yield payload.decode(part.get_content_charset("utf-8"), errors="ignore")
            elif content_type == "text/html":
                payload = part.get_payload(decode=True) or b""
                html = payload.decode(part.get_content_charset("utf-8"), errors="ignore")
                yield _strip_html(html)
    else:
        if message.get_content_type() == "text/plain":
            payload = message.get_payload(decode=True) or b""
            yield payload.decode(message.get_content_charset("utf-8"), errors="ignore")
        elif message.get_content_type() == "text/html":
            payload = message.get_payload(decode=True) or b""
            html = payload.decode(message.get_content_charset("utf-8"), errors="ignore")
            yield _strip_html(html)


def _extract_rfq_from_body(message) -> Optional[str]:
    for text in _iter_text_parts(message):
        match = RFQ_SUBJECT_RE.search(text)
        if match:
            return _normalise_rfq(*match.groups())
    return None


def _tag_object(s3_client, bucket: str, key: str, rfq_id: str) -> None:
    s3_client.put_object_tagging(
        Bucket=bucket,
        Key=key,
        Tagging={"TagSet": [{"Key": "rfq-id", "Value": rfq_id}]},
    )


def _copy_with_tags(s3_client, bucket: str, source_key: str, rfq_id: str) -> str:
    basename = source_key.rsplit("/", 1)[-1]
    dest_key = f"{DEFAULT_PREFIX.rstrip('/')}/{rfq_id}/ingest/{basename}"
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=dest_key,
        TaggingDirective="REPLACE",
        Tagging=f"rfq-id={rfq_id}",
    )
    return dest_key


def _move_to_unmatched(s3_client, bucket: str, source_key: str) -> str:
    basename = source_key.rsplit("/", 1)[-1]
    dest_key = f"{DEFAULT_PREFIX.rstrip('/')}/_unmatched/{basename}"
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=dest_key,
        TaggingDirective="REPLACE",
        Tagging="needs-review=true",
    )
    return dest_key


def process_record(record: Dict[str, object]) -> Dict[str, object]:
    s3_client = _get_s3_client()
    bucket = record.get("s3", {}).get("bucket", {}).get("name") if isinstance(record.get("s3"), dict) else None
    key = record.get("s3", {}).get("object", {}).get("key") if isinstance(record.get("s3"), dict) else None
    if not bucket or not key:
        raise ValueError("Invalid S3 event record: missing bucket or key")

    decoded_key = unquote_plus(str(key))
    response = s3_client.get_object(Bucket=bucket, Key=decoded_key)
    body = response["Body"].read()
    message = message_from_bytes(body)

    subject = _decode_header(message.get("Subject"))
    rfq_id = (message.get("X-Procwise-RFQ-ID") or "").strip() or _extract_rfq_from_subject(subject)
    if not rfq_id:
        rfq_id = _lookup_rfq_from_thread(message.get("In-Reply-To"), message.get("References"))
    if not rfq_id:
        rfq_id = _extract_rfq_from_body(message)

    if rfq_id:
        rfq_id = rfq_id.upper()
        _tag_object(s3_client, bucket, decoded_key, rfq_id)
        dest_key = _copy_with_tags(s3_client, bucket, decoded_key, rfq_id)
        logger.info("Tagged S3 object %s/%s with RFQ %s", bucket, decoded_key, rfq_id)
        return {"rfq_id": rfq_id, "s3_key": dest_key, "status": "ok"}

    dest_key = _move_to_unmatched(s3_client, bucket, decoded_key)
    logger.warning("Unable to resolve RFQ for %s/%s; moved to %s", bucket, decoded_key, dest_key)
    return {"rfq_id": None, "s3_key": dest_key, "status": "needs_review"}


def _extract_s3_records(event_payload: Dict[str, object]) -> Iterable[Dict[str, object]]:
    records = event_payload.get("Records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and record.get("eventSource") == "aws:s3":
                yield record


def lambda_handler(event: Dict[str, object], context=None) -> Dict[str, object]:
    results: List[Dict[str, object]] = []

    for record in event.get("Records", []):
        body = record.get("body") if isinstance(record, dict) else None
        payload: Optional[Dict[str, object]] = None

        if isinstance(body, str):
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                logger.warning("Skipping record with invalid JSON body: %s", body[:128])
                continue

            if "Message" in payload and isinstance(payload["Message"], str):
                try:
                    payload = json.loads(payload["Message"])
                except json.JSONDecodeError:
                    logger.warning("Skipping SNS-wrapped record with invalid message: %s", payload["Message"][:128])
                    continue
        elif isinstance(record, dict) and record.get("eventSource") == "aws:s3":
            payload = record

        if not payload:
            continue

        for s3_record in _extract_s3_records(payload):
            try:
                results.append(process_record(s3_record))
            except Exception:  # pragma: no cover - retries handled by SQS
                logger.exception("Failed to process S3 record: %s", s3_record)
                raise

    return {"processed": results}

