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
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote_plus

import boto3
import psycopg2
from botocore.config import Config
from psycopg2 import extras

from services.email_dispatch_chain_store import mark_response as mark_dispatch_response
from services.email_thread_store import (
    DEFAULT_THREAD_TABLE,
    ensure_thread_table,
    lookup_thread_metadata,
    sanitise_thread_table_name,
)


logger = logging.getLogger(__name__)


DEFAULT_BUCKET = os.getenv("EMAIL_INGEST_BUCKET", os.getenv("S3_BUCKET", "procwisemvp"))
DEFAULT_PREFIX = os.getenv("EMAIL_INGEST_PREFIX", "emails/")
THREAD_TABLE = sanitise_thread_table_name(
    os.getenv("EMAIL_THREAD_TABLE", os.getenv("DDB_TABLE", DEFAULT_THREAD_TABLE)),
    logger=logger,
)
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")
_RAW_SUPPLIER_REPLY_TABLE = os.getenv("SUPPLIER_REPLY_TABLE", "proc.supplier_replies")
_RAW_SUPPLIER_OBJECT_TABLE = os.getenv(
    "SUPPLIER_REPLY_OBJECT_TABLE", "proc.supplier_reply_objects"
)
_BOTO_CONFIG = Config(retries={"max_attempts": 5, "mode": "standard"})
_S3_CLIENT = None
_THREAD_TABLE_READY = False
_DB_CONNECTION = None
_SUPPLIER_TABLE_INITIALISED = False
_SUPPLIER_OBJECT_TABLE_INITIALISED = False

RFQ_SUBJECT_RE = re.compile(r"\bRFQ[-_](\d{8})[-_]?([A-Za-z0-9\-]+)", re.IGNORECASE)
RFQ_HTML_COMMENT_RE = re.compile(r"<!--\s*RFQ-ID\s*:\s*([A-Za-z0-9_-]+)\s*-->", re.IGNORECASE)

_VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _sanitise_table_name(name: Optional[str]) -> str:
    candidate = (name or "").strip()
    if not candidate or not _VALID_TABLE_NAME.match(candidate):
        if candidate:
            logger.warning(
                "Invalid supplier reply table name %r; using default proc.supplier_replies",
                candidate,
            )
        return "proc.supplier_replies"
    return candidate


SUPPLIER_REPLY_TABLE = _sanitise_table_name(_RAW_SUPPLIER_REPLY_TABLE)
SUPPLIER_REPLY_OBJECT_TABLE = _sanitise_table_name(_RAW_SUPPLIER_OBJECT_TABLE)


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3", config=_BOTO_CONFIG)
    return _S3_CLIENT
def _get_db_connection():
    global _DB_CONNECTION
    if _DB_CONNECTION is not None:
        try:
            if getattr(_DB_CONNECTION, "closed", 1) == 0:
                return _DB_CONNECTION
        except Exception:
            _DB_CONNECTION = None

    if not (DB_HOST and DB_NAME and DB_USER):
        logger.debug("Database credentials not fully configured; skipping supplier reply persistence")
        return None

    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=int(DB_PORT or 5432),
            connect_timeout=5,
        )
        connection.autocommit = True
        _DB_CONNECTION = connection
    except Exception:
        logger.exception("Unable to connect to Postgres for supplier reply persistence")
        _DB_CONNECTION = None
    return _DB_CONNECTION


def _ensure_supplier_reply_table(connection) -> None:
    global _SUPPLIER_TABLE_INITIALISED
    if _SUPPLIER_TABLE_INITIALISED or connection is None:
        return

    ddl = f"""
        CREATE TABLE IF NOT EXISTS {SUPPLIER_REPLY_TABLE} (
            rfq_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            mailbox TEXT,
            subject TEXT,
            from_address TEXT,
            reply_body TEXT,
            s3_key TEXT,
            received_at TIMESTAMPTZ,
            headers JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (rfq_id, message_id)
        )
    """
    index_name = SUPPLIER_REPLY_TABLE.replace(".", "_") + "_rfq_idx"
    index_sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
            ON {SUPPLIER_REPLY_TABLE} (rfq_id)
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            cursor.execute(index_sql)
        _SUPPLIER_TABLE_INITIALISED = True
    except Exception:
        logger.exception("Failed to ensure supplier reply table %s", SUPPLIER_REPLY_TABLE)


def _ensure_supplier_reply_object_table(connection) -> None:
    global _SUPPLIER_OBJECT_TABLE_INITIALISED
    if _SUPPLIER_OBJECT_TABLE_INITIALISED or connection is None:
        return

    ddl = f"""
        CREATE TABLE IF NOT EXISTS {SUPPLIER_REPLY_OBJECT_TABLE} (
            bucket TEXT NOT NULL,
            object_key TEXT NOT NULL,
            etag TEXT NOT NULL,
            rfq_id TEXT,
            message_id TEXT,
            processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bucket, object_key, etag)
        )
    """
    index_sql = (
        f"CREATE INDEX IF NOT EXISTS {SUPPLIER_REPLY_OBJECT_TABLE.replace('.', '_')}_key_idx "
        f"ON {SUPPLIER_REPLY_OBJECT_TABLE} (object_key)"
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            cursor.execute(index_sql)
        _SUPPLIER_OBJECT_TABLE_INITIALISED = True
    except Exception:
        logger.exception(
            "Failed to ensure supplier reply object table %s",
            SUPPLIER_REPLY_OBJECT_TABLE,
        )


def _parse_received_at(date_header: Optional[str]) -> Optional[datetime]:
    if not date_header:
        return None
    try:
        parsed = parsedate_to_datetime(date_header)
    except (TypeError, ValueError, IndexError):
        logger.debug("Unable to parse Date header %r", date_header)
        return None

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_primary_body(message) -> str:
    for text in _iter_text_parts(message):
        cleaned = text.strip()
        if cleaned:
            return cleaned
    return ""


def _collect_headers(message) -> Dict[str, List[str]]:
    headers: Dict[str, List[str]] = {}
    for key, value in message.raw_items():
        decoded = _decode_header(value)
        headers.setdefault(key, []).append(decoded)
    return headers

def _ensure_thread_table(conn) -> bool:
    global _THREAD_TABLE_READY
    if conn is None or _THREAD_TABLE_READY:
        return conn is not None and _THREAD_TABLE_READY

    try:
        ensure_thread_table(conn, THREAD_TABLE, logger=logger)
        _THREAD_TABLE_READY = True
        return True
    except Exception:  # pragma: no cover - logging only
        logger.exception("Failed to ensure email thread mapping table %s", THREAD_TABLE)
        return False


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

    conn = _get_db_connection()
    if conn is None:
        return None

    if not _ensure_thread_table(conn):
        return None

    keys = [candidate.strip() for candidate in candidates if candidate.strip()]
    if not keys:
        return None

    metadata = lookup_thread_metadata(conn, THREAD_TABLE, keys, logger=logger)
    if metadata:
        return metadata[0]
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


def _build_reply_metadata(message, bucket: str, key: str) -> Dict[str, object]:
    subject = _decode_header(message.get("Subject"))
    from_address = _decode_header(message.get("From"))
    to_addresses = [_decode_header(value) for value in message.get_all("To", [])]
    cc_addresses = [_decode_header(value) for value in message.get_all("Cc", [])]
    mailbox = _decode_header(message.get("X-Procwise-Mailbox"))
    if not mailbox and to_addresses:
        mailbox = to_addresses[0]

    raw_message_id = message.get("Message-ID") or message.get("Message-Id")
    message_id = raw_message_id.strip() if isinstance(raw_message_id, str) else None

    in_reply_to = _decode_header(message.get("In-Reply-To"))
    reference_headers = message.get_all("References", [])
    references = [_decode_header(value) for value in reference_headers]

    metadata = {
        "subject": subject,
        "from_address": from_address,
        "mailbox": mailbox,
        "to_addresses": to_addresses,
        "cc_addresses": cc_addresses,
        "message_id": message_id,
        "body": _extract_primary_body(message),
        "received_at": _parse_received_at(message.get("Date")),
        "headers": _collect_headers(message),
        "s3_bucket": bucket,
        "s3_key": key,
        "original_s3_key": key,
        "in_reply_to": in_reply_to,
        "references": references,
    }
    return metadata


def _upsert_supplier_reply(rfq_id: str, metadata: Dict[str, object]) -> None:
    connection = _get_db_connection()
    if connection is None:
        return

    _ensure_supplier_reply_table(connection)

    headers = metadata.get("headers") or {}
    message_id = metadata.get("message_id")
    if not message_id and isinstance(headers, dict):
        candidates = headers.get("Message-ID") or headers.get("Message-Id")
        if isinstance(candidates, list) and candidates:
            message_id = candidates[0]
    if not message_id:
        message_id = metadata.get("s3_key")

    params = (
        rfq_id,
        message_id,
        metadata.get("mailbox"),
        metadata.get("subject"),
        metadata.get("from_address"),
        metadata.get("body"),
        metadata.get("s3_key"),
        metadata.get("received_at"),
        extras.Json(headers) if headers else None,
    )

    sql = f"""
        INSERT INTO {SUPPLIER_REPLY_TABLE} (
            rfq_id,
            message_id,
            mailbox,
            subject,
            from_address,
            reply_body,
            s3_key,
            received_at,
            headers
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (rfq_id, message_id) DO UPDATE
        SET
            mailbox = EXCLUDED.mailbox,
            subject = EXCLUDED.subject,
            from_address = EXCLUDED.from_address,
            reply_body = EXCLUDED.reply_body,
            s3_key = EXCLUDED.s3_key,
            received_at = EXCLUDED.received_at,
            headers = EXCLUDED.headers,
            updated_at = now()
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
    except Exception:
        logger.exception("Failed to upsert supplier reply for RFQ %s", rfq_id)


def _register_processed_object(
    connection,
    bucket: str,
    key: str,
    etag: Optional[str],
    *,
    rfq_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> None:
    if connection is None or not (bucket and key and etag):
        return

    _ensure_supplier_reply_object_table(connection)

    sql = f"""
        INSERT INTO {SUPPLIER_REPLY_OBJECT_TABLE} (
            bucket, object_key, etag, rfq_id, message_id
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (bucket, key, etag, rfq_id, message_id))
    except Exception:
        logger.exception(
            "Failed to register processed object %s/%s (etag=%s)", bucket, key, etag
        )


def _lookup_processed_object(connection, bucket: str, key: str, etag: Optional[str]) -> Optional[Dict[str, object]]:
    if connection is None or not (bucket and key and etag):
        return None

    _ensure_supplier_reply_object_table(connection)

    sql = f"""
        SELECT rfq_id, message_id
        FROM {SUPPLIER_REPLY_OBJECT_TABLE}
        WHERE bucket = %s AND object_key = %s AND etag = %s
        LIMIT 1
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, (bucket, key, etag))
            row = cursor.fetchone()
    except Exception:
        logger.exception(
            "Failed to check processed state for %s/%s (etag=%s)", bucket, key, etag
        )
        return None

    if not row:
        return None

    rfq_id, message_id = row
    return {"rfq_id": rfq_id, "message_id": message_id}


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
    raw_etag = record.get("s3", {}).get("object", {}).get("eTag")
    if not raw_etag:
        raw_etag = record.get("s3", {}).get("object", {}).get("etag")
    etag = str(raw_etag).strip('"') if raw_etag else None

    connection = _get_db_connection()
    existing = _lookup_processed_object(connection, bucket, decoded_key, etag)
    if existing:
        logger.info(
            "Skipping already processed object %s/%s (etag=%s)", bucket, decoded_key, etag
        )
        return {
            "rfq_id": existing.get("rfq_id"),
            "s3_key": decoded_key,
            "status": "duplicate",
            "metadata": {"message_id": existing.get("message_id")},
        }

    response = s3_client.get_object(Bucket=bucket, Key=decoded_key)
    body = response["Body"].read()
    message = message_from_bytes(body)
    if not etag:
        response_etag = response.get("ETag")
        etag = str(response_etag).strip('"') if response_etag else None

    metadata = _build_reply_metadata(message, bucket, decoded_key)
    subject = metadata.get("subject") or ""
    rfq_id = (message.get("X-Procwise-RFQ-ID") or "").strip() or _extract_rfq_from_subject(subject)
    if not rfq_id:
        rfq_id = _lookup_rfq_from_thread(message.get("In-Reply-To"), message.get("References"))
    if not rfq_id:
        rfq_id = _extract_rfq_from_body(message)

    if rfq_id:
        rfq_id = rfq_id.upper()
        metadata["rfq_id"] = rfq_id
        _tag_object(s3_client, bucket, decoded_key, rfq_id)
        dest_key = _copy_with_tags(s3_client, bucket, decoded_key, rfq_id)
        metadata["s3_key"] = dest_key
        logger.info("Tagged S3 object %s/%s with RFQ %s", bucket, decoded_key, rfq_id)
        _upsert_supplier_reply(rfq_id, metadata)
        try:
            mark_dispatch_response(
                connection,
                rfq_id=rfq_id,
                in_reply_to=metadata.get("in_reply_to"),
                references=metadata.get("references"),
                response_message_id=metadata.get("message_id"),
                response_metadata={
                    "subject": metadata.get("subject"),
                    "from_address": metadata.get("from_address"),
                },
            )
        except Exception:  # pragma: no cover - best effort logging
            logger.exception(
                "Failed to mark dispatch chain response for RFQ %s", rfq_id
            )
        _register_processed_object(
            connection,
            bucket,
            decoded_key,
            etag,
            rfq_id=rfq_id,
            message_id=metadata.get("message_id"),
        )
        return {
            "rfq_id": rfq_id,
            "s3_key": dest_key,
            "status": "ok",
            "metadata": {
                "mailbox": metadata.get("mailbox"),
                "subject": metadata.get("subject"),
                "from_address": metadata.get("from_address"),
                "received_at": metadata.get("received_at").isoformat()
                if isinstance(metadata.get("received_at"), datetime)
                else None,
                "message_id": metadata.get("message_id"),
            },
        }

    dest_key = _move_to_unmatched(s3_client, bucket, decoded_key)
    logger.warning("Unable to resolve RFQ for %s/%s; moved to %s", bucket, decoded_key, dest_key)
    _register_processed_object(
        connection,
        bucket,
        decoded_key,
        etag,
        rfq_id=None,
        message_id=metadata.get("message_id"),
    )
    return {
        "rfq_id": None,
        "s3_key": dest_key,
        "status": "needs_review",
        "metadata": {
            "mailbox": metadata.get("mailbox"),
            "subject": metadata.get("subject"),
            "from_address": metadata.get("from_address"),
            "message_id": metadata.get("message_id"),
        },
    }


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

