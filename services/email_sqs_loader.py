"""SQS-backed loader for SES inbound email notifications."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

__all__ = ["sqs_email_loader"]


def _parse_event_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Normalise ``Z`` suffix to maintain compatibility with ``fromisoformat``
        cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        logger.debug("Unable to parse event time %s", value, exc_info=True)
        return None


def _extract_records(payload: Dict[str, object]) -> Iterable[object]:
    records = payload.get("Records")
    if isinstance(records, list) and records:
        return records

    detail = payload.get("detail")
    if isinstance(detail, dict):
        detail_records: List[object] = []
        embedded_records = detail.get("records")
        if isinstance(embedded_records, list):
            detail_records.extend(embedded_records)
        bucket = detail.get("bucket") if isinstance(detail.get("bucket"), dict) else None
        obj = detail.get("object") if isinstance(detail.get("object"), dict) else None
        if bucket and obj and bucket.get("name") and obj.get("key"):
            detail_records.append(
                {
                    "s3": {
                        "bucket": {"name": bucket["name"]},
                        "object": {"key": obj["key"]},
                    },
                    "eventTime": detail.get("eventTime") or detail.get("time"),
                }
            )
        resources = detail.get("resources")
        if isinstance(resources, list):
            detail_records.extend(resources)
        if detail_records:
            return detail_records

    resources = payload.get("resources")
    if isinstance(resources, list) and resources:
        return resources

    return []


def _extract_bucket_key(record: object) -> Optional[Tuple[str, str, Optional[datetime]]]:
    if isinstance(record, dict):
        if "s3" in record and isinstance(record["s3"], dict):
            s3_block = record["s3"]
            bucket = None
            key = None
            if isinstance(s3_block.get("bucket"), dict):
                bucket = s3_block["bucket"].get("name")
            if isinstance(s3_block.get("object"), dict):
                key = s3_block["object"].get("key")
            timestamp = _parse_event_time(record.get("eventTime") or record.get("event_time"))
            if bucket and key:
                return bucket, unquote_plus(key), timestamp
        bucket_hint = None
        key_hint = None
        if isinstance(record.get("bucket"), dict):
            bucket_hint = record["bucket"].get("name")
        if isinstance(record.get("object"), dict):
            key_hint = record["object"].get("key")
        timestamp = _parse_event_time(record.get("eventTime") or record.get("time"))
        if bucket_hint and key_hint:
            return bucket_hint, unquote_plus(key_hint), timestamp
        resource = record.get("resource")
        if isinstance(resource, str):
            parsed = _parse_s3_arn(resource)
            if parsed:
                bucket, key = parsed
                return bucket, key, timestamp
    elif isinstance(record, str):
        parsed = _parse_s3_arn(record)
        if parsed:
            bucket, key = parsed
            return bucket, key, None
    return None


def _parse_s3_arn(arn: str) -> Optional[Tuple[str, str]]:
    if not arn or not arn.startswith("arn:aws:s3:::"):
        return None
    remainder = arn.split(":::", 1)[-1]
    if not remainder:
        return None
    if "/" in remainder:
        bucket, key = remainder.split("/", 1)
        return bucket, unquote_plus(key)
    return remainder, ""


def sqs_email_loader(
    *,
    queue_url: str,
    max_batch: int = 10,
    wait_seconds: int = 10,
    visibility_timeout: int = 30,
    sqs_client=None,
):
    """Return a loader that surfaces S3 notification messages from an SQS queue."""

    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    if wait_seconds < 0:
        raise ValueError("wait_seconds cannot be negative")
    if visibility_timeout <= 0:
        raise ValueError("visibility_timeout must be positive")

    sqs = sqs_client or boto3.client("sqs")

    def _load(limit: Optional[int] = None) -> List[Dict[str, object]]:
        if limit is not None:
            try:
                effective_limit = max(int(limit), 0)
            except Exception:
                effective_limit = 0
            if effective_limit == 0:
                return []
        else:
            effective_limit = None

        to_read = max_batch if effective_limit is None else min(max_batch, effective_limit)
        to_read = max(1, min(10, to_read))

        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=to_read,
                WaitTimeSeconds=wait_seconds,
                VisibilityTimeout=visibility_timeout,
                MessageAttributeNames=["All"],
            )
        except (BotoCoreError, ClientError):
            logger.exception("Failed to receive messages from %s", queue_url)
            return []

        sqs_messages = response.get("Messages", [])
        batch: List[Dict[str, object]] = []

        for sqs_message in sqs_messages:
            receipt = sqs_message.get("ReceiptHandle")
            body = sqs_message.get("Body")
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                logger.debug(
                    "Skipping non-JSON SQS payload for queue %s: %s",
                    queue_url,
                    body,
                )
                payload = {}

            for record in _extract_records(payload):
                parsed = _extract_bucket_key(record)
                if not parsed:
                    continue
                bucket, key, timestamp = parsed
                if not bucket or not key:
                    continue
                entry: Dict[str, object] = {
                    "id": key,
                    "s3_key": key,
                    "_bucket": bucket,
                }
                if timestamp is not None:
                    entry["_last_modified"] = timestamp
                batch.append(entry)

            if receipt:
                try:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                except (BotoCoreError, ClientError):
                    logger.exception("Failed to delete SQS message %s", receipt)

        return batch

    return _load

