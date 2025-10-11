from __future__ import annotations

import os
import json
import re
import base64
import quopri
import logging
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from typing import Dict, Optional, Set, Tuple

import boto3
import psycopg2
from psycopg2 import sql
from botocore.exceptions import ClientError

# ---------- Config ----------
REGION = os.getenv("AWS_REGION", "ap-south-1")
QUEUE_URL = os.environ["SQS_QUEUE_URL"]
PROCESSED_TABLE = os.getenv("PROCESSED_TABLE", "proc.processed_emails")
WATERMARK_TABLE = os.getenv("S3_WATERMARK_TABLE", "proc.email_s3_watermarks")
DELETE_ON_SUCCESS = os.getenv("DELETE_ON_SUCCESS", "true").lower() == "true"
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "15"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# Compile RFQ patterns
_default_patterns = [r"[A-Z]{3}\d{5,}", r"RFQ-\d{4}-\d{3,}[A-Z]?"]
PATTERNS_STR = os.getenv("RFQ_ID_PATTERNS")
RFQ_PATTERNS = json.loads(PATTERNS_STR) if PATTERNS_STR else _default_patterns
RFQ_ID_RE = re.compile(r"\\b(?:" + "|".join(RFQ_PATTERNS) + r")\\b", re.IGNORECASE)

# Subject noise (Re:, Fwd:, [EXTERNAL], etc.)
NOISE = re.compile(r"^(?:(re|fwd|fw)\\s*:\\s*)+|\\[[^\\]]+\\]\\s*", re.IGNORECASE)

# ---------- AWS ----------
sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# ---------- Logging ----------
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
log = logging.getLogger("email_watcher")

MESSAGE_SENTINEL_BUCKET = "sentinel::message"
ETAG_SENTINEL_BUCKET = "sentinel::etag"

_watermark_cache: Dict[str, Tuple[Optional[datetime], str]] = {}
_db_connection = None
_processed_table_ready = False
_watermark_table_ready = False


def _table_identifier(table_name: str) -> sql.SQL:
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    return sql.Identifier(table_name)


def _reset_connection_state() -> None:
    global _processed_table_ready, _watermark_table_ready
    _processed_table_ready = False
    _watermark_table_ready = False


def _get_db_connection():
    global _db_connection
    if _db_connection is not None:
        try:
            if getattr(_db_connection, "closed", 1) == 0:
                return _db_connection
        except Exception:
            _db_connection = None

    if not (DB_HOST and DB_NAME and DB_USER):
        log.error(
            "Database credentials not configured; unable to persist inbound email registry state",
        )
        return None

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=5,
        )
        conn.autocommit = True
        _db_connection = conn
        _reset_connection_state()
    except Exception:
        log.exception("Unable to connect to Postgres for email watcher registry")
        _db_connection = None
    return _db_connection


def _ensure_processed_table(connection) -> None:
    global _processed_table_ready
    if _processed_table_ready:
        return

    table_sql = _table_identifier(PROCESSED_TABLE)
    index_prefix = PROCESSED_TABLE.replace(".", "_")

    try:
        with connection.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        bucket TEXT NOT NULL,
                        key TEXT NOT NULL,
                        etag TEXT NOT NULL DEFAULT '',
                        rfq_id TEXT,
                        processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        mailbox TEXT,
                        message_id TEXT,
                        source_last_modified TIMESTAMPTZ
                    )
                    """
                ).format(table=table_sql)
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                        ALTER COLUMN etag SET DEFAULT ''
                    """
                ).format(table=table_sql)
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS mailbox TEXT
                    """
                ).format(table=table_sql)
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS message_id TEXT
                    """
                ).format(table=table_sql)
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS source_last_modified TIMESTAMPTZ
                    """
                ).format(table=table_sql)
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS {idx}
                        ON {table} (bucket, key, etag)
                    """
                ).format(
                    idx=sql.Identifier(f"{index_prefix}_bucket_key_etag_uidx"),
                    table=table_sql,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {idx}
                        ON {table} (rfq_id, processed_at DESC)
                    """
                ).format(
                    idx=sql.Identifier(f"{index_prefix}_rfq_ts_idx"),
                    table=table_sql,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {idx}
                        ON {table} (rfq_id, key)
                    """
                ).format(
                    idx=sql.Identifier(f"{index_prefix}_rfq_key_idx"),
                    table=table_sql,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {idx}
                        ON {table} (message_id)
                    """
                ).format(
                    idx=sql.Identifier(f"{index_prefix}_message_id_idx"),
                    table=table_sql,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS {idx}
                        ON {table} (COALESCE(mailbox, ''), message_id)
                        WHERE message_id IS NOT NULL AND message_id <> ''
                    """
                ).format(
                    idx=sql.Identifier(f"{index_prefix}_mailbox_message_uidx"),
                    table=table_sql,
                )
            )
    except Exception:
        log.exception("Failed to ensure processed email registry table exists")
        raise

    _processed_table_ready = True


def _ensure_watermark_table(connection) -> None:
    global _watermark_table_ready
    if _watermark_table_ready:
        return

    table_sql = _table_identifier(WATERMARK_TABLE)
    index_prefix = WATERMARK_TABLE.replace(".", "_")

    try:
        with connection.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        bucket TEXT PRIMARY KEY,
                        last_processed_key TEXT,
                        last_processed_ts TIMESTAMPTZ NOT NULL,
                        processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(table=table_sql)
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {idx}
                        ON {table} (last_processed_ts DESC)
                    """
                ).format(
                    idx=sql.Identifier(f"{index_prefix}_ts_idx"),
                    table=table_sql,
                )
            )
    except Exception:
        log.exception("Failed to ensure S3 watermark table exists")
        raise

    _watermark_table_ready = True

# ---------- Helpers ----------
def clean_subject(s: Optional[str]) -> str:
    s = (s or "").strip()
    while True:
        s2 = NOISE.sub("", s).strip()
        if s2 == s:
            return s
        s = s2

def extract_text_from_message(msg) -> Tuple[str, str]:
    """
    Returns (text_plain, text_html_as_text_fallback)
    """
    text_plain = None
    html = None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and text_plain is None:
                try:
                    text_plain = part.get_content()
                except:  # charset quirks
                    text_plain = _decode_bytes(part)
            elif ctype == "text/html" and html is None:
                try:
                    html = part.get_content()
                except:
                    html = _decode_bytes(part)
    else:
        ctype = msg.get_content_type()
        try:
            if ctype == "text/plain":
                text_plain = msg.get_content()
            elif ctype == "text/html":
                html = msg.get_content()
        except:
            body = _decode_bytes(msg)
            if ctype == "text/plain": text_plain = body
            elif ctype == "text/html": html = body

    if (not text_plain) and html:
        # basic HTML -> text fallback (no external deps)
        text_plain = _html_to_text(html)
    return (text_plain or ""), (html or "")

def _decode_bytes(part) -> str:
    payload = part.get_payload(decode=True) or b""
    # Try quoted-printable and base64 fallbacks
    try:
        return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    except Exception:
        try:
            return quopri.decodestring(payload).decode("utf-8", "replace")
        except Exception:
            try:
                return base64.b64decode(payload).decode("utf-8", "replace")
            except Exception:
                return payload.decode("utf-8", "replace")

def _html_to_text(html: str) -> str:
    # very minimal strip; preserve RFQ tokens
    txt = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    txt = re.sub(r"(?is)<style.*?>.*?</style>", " ", txt)
    txt = re.sub(r"(?is)<br\\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</p>", "\n", txt)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    return re.sub(r"\\s+", " ", txt).strip()

def extract_rfq_ids(text: str) -> Set[str]:
    return {m.group(0) for m in RFQ_ID_RE.finditer(text or "")}

def body_tag_extract(body_text: str, body_html: str) -> Optional[str]:
    # Look for hidden tag <!-- PROCWISE:RFQ_ID=... -->
    m = re.search(r"PROCWISE:RFQ_ID=([A-Za-z0-9\-]+)", body_html or "", re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"PROCWISE:RFQ_ID=([A-Za-z0-9\-]+)", body_text or "", re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def header_extract(msg) -> Optional[str]:
    val = msg.get("X-Procwise-RFQ-Id")
    if val:
        return val.strip()
    return None

def match_email_to_rfq(known_ids: Set[str], subject: str, body_text: str, body_html: str, msg) -> Tuple[Optional[str], str]:
    """
    Returns (rfq_id, matched_via)
    matched_via ∈ {'header','body','subject_id', 'none'}
    """
    # 1) Header
    hdr = header_extract(msg)
    if hdr and hdr in known_ids:
        return hdr, "header"

    # 2) Body tag
    tag = body_tag_extract(body_text, body_html)
    if tag and tag in known_ids:
        return tag, "body"

    # 3) Subject by exact ID
    ids = extract_rfq_ids(subject)
    for tok in ids:
        if tok in known_ids:
            return tok, "subject_id"
    return None, "none"

def _normalise_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        log.warning("Unable to parse watermark timestamp '%s'", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def load_watermark(bucket: Optional[str]) -> Tuple[Optional[datetime], str]:
    if not bucket:
        return None, ""
    cached = _watermark_cache.get(bucket)
    if cached is not None:
        return cached

    conn = _get_db_connection()
    if conn is None:
        return None, ""

    try:
        _ensure_watermark_table(conn)
    except Exception:
        return None, ""

    table_sql = _table_identifier(WATERMARK_TABLE)
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT last_processed_ts, last_processed_key FROM {table} WHERE bucket = %s"
                ).format(table=table_sql),
                (bucket,),
            )
            row = cur.fetchone()
    except Exception:
        log.exception("Failed to load watermark for bucket %s", bucket)
        return None, ""

    if not row:
        return None, ""

    ts, key = row
    if isinstance(ts, datetime):
        ts_value = ts.astimezone(timezone.utc)
    else:
        ts_value = _normalise_timestamp(str(ts))
    key_value = str(key or "")
    result = (ts_value, key_value)
    _watermark_cache[bucket] = result
    return result


def _store_watermark(bucket: Optional[str], key: str, last_modified: Optional[datetime]) -> None:
    if not bucket or last_modified is None:
        return

    conn = _get_db_connection()
    if conn is None:
        return

    try:
        _ensure_watermark_table(conn)
    except Exception:
        return

    table_sql = _table_identifier(WATERMARK_TABLE)
    ts_utc = last_modified.astimezone(timezone.utc)

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (bucket, last_processed_key, last_processed_ts, processed_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (bucket) DO UPDATE SET
                        last_processed_key = EXCLUDED.last_processed_key,
                        last_processed_ts = EXCLUDED.last_processed_ts,
                        processed_at = NOW()
                    """
                ).format(table=table_sql),
                (bucket, key or "", ts_utc),
            )
    except Exception:
        log.exception("Failed to store watermark for bucket %s", bucket)
        return

    _watermark_cache[bucket] = (ts_utc, key or "")


def _is_newer_than_watermark(
    candidate_ts: Optional[datetime],
    candidate_key: str,
    watermark_ts: Optional[datetime],
    watermark_key: str,
) -> bool:
    if candidate_ts is None:
        return True
    cand = candidate_ts.astimezone(timezone.utc)
    if watermark_ts is None:
        return True
    reference = watermark_ts.astimezone(timezone.utc)
    if cand > reference:
        return True
    if cand < reference:
        return False
    if not watermark_key:
        return True
    return candidate_key > watermark_key


def already_processed(
    s3_key: str,
    *,
    bucket: Optional[str] = None,
    message_id: Optional[str] = None,
    provider_id: Optional[str] = None,
) -> bool:
    if not s3_key:
        return False

    conn = _get_db_connection()
    if conn is None:
        return False

    try:
        _ensure_processed_table(conn)
    except Exception:
        return False

    table_sql = _table_identifier(PROCESSED_TABLE)

    try:
        with conn.cursor() as cur:
            if bucket:
                cur.execute(
                    sql.SQL(
                        "SELECT 1 FROM {table} WHERE bucket = %s AND key = %s LIMIT 1"
                    ).format(table=table_sql),
                    (bucket, s3_key),
                )
                if cur.fetchone():
                    return True

            if message_id:
                cur.execute(
                    sql.SQL(
                        "SELECT 1 FROM {table} WHERE bucket = %s AND key = %s LIMIT 1"
                    ).format(table=table_sql),
                    (MESSAGE_SENTINEL_BUCKET, message_id),
                )
                if cur.fetchone():
                    return True
                cur.execute(
                    sql.SQL(
                        "SELECT 1 FROM {table} WHERE message_id = %s LIMIT 1"
                    ).format(table=table_sql),
                    (message_id,),
                )
                if cur.fetchone():
                    return True

            if provider_id:
                cur.execute(
                    sql.SQL(
                        "SELECT 1 FROM {table} WHERE bucket = %s AND key = %s LIMIT 1"
                    ).format(table=table_sql),
                    (ETAG_SENTINEL_BUCKET, provider_id),
                )
                if cur.fetchone():
                    return True
                cur.execute(
                    sql.SQL(
                        "SELECT 1 FROM {table} WHERE etag = %s LIMIT 1"
                    ).format(table=table_sql),
                    (provider_id,),
                )
                if cur.fetchone():
                    return True
    except Exception:
        log.exception("Failed to check processed registry for %s", s3_key)
        return False

    return False


def _record_sentinel(
    *,
    kind: str,
    identifier: str,
    base_bucket: Optional[str],
    rfq_id: Optional[str],
) -> None:
    conn = _get_db_connection()
    if conn is None:
        return

    try:
        _ensure_processed_table(conn)
    except Exception:
        return

    table_sql = _table_identifier(PROCESSED_TABLE)
    bucket_value = MESSAGE_SENTINEL_BUCKET if kind == "message" else ETAG_SENTINEL_BUCKET
    mailbox_value = base_bucket or bucket_value
    message_id_value = identifier if kind == "message" else None

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (bucket, key, etag, rfq_id, processed_at, mailbox, message_id)
                    VALUES (%s, %s, '', %s, NOW(), %s, %s)
                    ON CONFLICT (bucket, key, etag) DO UPDATE SET
                        rfq_id = EXCLUDED.rfq_id,
                        mailbox = COALESCE(EXCLUDED.mailbox, {table}.mailbox),
                        message_id = COALESCE(EXCLUDED.message_id, {table}.message_id),
                        processed_at = NOW()
                    """
                ).format(table=table_sql),
                (bucket_value, identifier, rfq_id or "", mailbox_value, message_id_value),
            )
    except Exception:
        log.exception("Failed to record %s sentinel for %s", kind, identifier)


def mark_processed(
    bucket: Optional[str],
    s3_key: str,
    message_id: str,
    provider_id: str,
    rfq_id: str,
    last_modified: Optional[datetime],
) -> None:
    conn = _get_db_connection()
    if conn is None:
        return

    try:
        _ensure_processed_table(conn)
    except Exception:
        return

    table_sql = _table_identifier(PROCESSED_TABLE)
    last_modified_ts = (
        last_modified.astimezone(timezone.utc) if last_modified is not None else None
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        bucket,
                        key,
                        etag,
                        rfq_id,
                        processed_at,
                        mailbox,
                        message_id,
                        source_last_modified
                    )
                    VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s)
                    ON CONFLICT (bucket, key, etag) DO UPDATE SET
                        rfq_id = EXCLUDED.rfq_id,
                        mailbox = COALESCE(EXCLUDED.mailbox, {table}.mailbox),
                        message_id = COALESCE(EXCLUDED.message_id, {table}.message_id),
                        source_last_modified = COALESCE(EXCLUDED.source_last_modified, {table}.source_last_modified),
                        processed_at = NOW()
                    """
                ).format(table=table_sql),
                (
                    bucket or "",
                    s3_key,
                    provider_id or "",
                    rfq_id or "",
                    bucket or None,
                    message_id or None,
                    last_modified_ts,
                ),
            )
    except Exception:
        log.exception("Failed to record processed email %s", s3_key)
        return

    if message_id:
        _record_sentinel(
            kind="message",
            identifier=message_id,
            base_bucket=bucket,
            rfq_id=rfq_id or None,
        )
    if provider_id:
        _record_sentinel(
            kind="etag",
            identifier=provider_id,
            base_bucket=bucket,
            rfq_id=rfq_id or None,
        )

    _store_watermark(bucket, s3_key, last_modified)

def dead_letter(reason: str, details: dict):
    log.warning(f"DEAD-LETTER reason={reason} details={json.dumps(details)[:1000]}")

def load_known_rfq_ids() -> Set[str]:
    # TODO: Replace with your data source (DB query, API call). For now, assume all tokens are acceptable.
    # If you want strict matching: fetch active/open RFQ IDs here.
    return set()

# ---------- Core processing ----------
def process_s3_record(bucket: str, key: str) -> bool:
    """
    Returns True if processed successfully (or safely skipped), False if should retry.
    """
    watermark_ts, watermark_key = load_watermark(bucket)

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        log.error(f"Cannot fetch s3://{bucket}/{key}: {e}")
        return False

    raw = obj["Body"].read()
    # In SES->S3 pipeline, object is the raw RFC822 email
    last_modified = obj.get("LastModified")
    if last_modified and not _is_newer_than_watermark(last_modified, key, watermark_ts, watermark_key):
        log.info(
            "Skipping s3://%s/%s; last_modified=%s is not newer than watermark=%s::%s",
            bucket,
            key,
            last_modified,
            watermark_ts.isoformat() if watermark_ts else "<none>",
            watermark_key or "<none>",
        )
        return True

    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        provider_id = (obj.get("ETag") or "").strip('"')
        log.exception("Failed to parse email object s3://%s/%s", bucket, key)
        dead_letter(
            "parse_failure",
            {
                "s3_key": key,
                "bucket": bucket,
            },
        )
        mark_processed(bucket, key, "", provider_id, "", last_modified)
        return True

    message_id = (msg.get("Message-Id") or "").strip()
    provider_id = (obj.get("ETag") or "").strip('"')

    if already_processed(
        key,
        bucket=bucket,
        message_id=message_id or None,
        provider_id=provider_id or None,
    ):
        log.info(
            "Skipping duplicate email s3://%s/%s (message_id=%s, etag=%s)",
            bucket,
            key,
            message_id or "<none>",
            provider_id or "<none>",
        )
        return True

    subj = clean_subject(msg.get("Subject"))
    body_text, body_html = extract_text_from_message(msg)

    known = load_known_rfq_ids()  # empty ⇒ accept any token that matches the regex shapes
    rfq_id, matched_via = match_email_to_rfq(known or extract_rfq_ids(subj) | extract_rfq_ids(body_text), subj, body_text, body_html, msg)

    if not rfq_id:
        dead_letter("no_rfq_id_found", {
            "s3_key": key,
            "message_id": message_id,
            "subject": subj[:300],
        })
        # Mark processed to avoid infinite retries; adjust if you want to reprocess later
        mark_processed(bucket, key, message_id, provider_id, rfq_id or "", last_modified)
        return True

    # >>> Hand off to your pipeline here (NegotiationAgent → DraftingAgent) <<<
    log.info(f"Matched RFQ {rfq_id} via {matched_via} | subject='{subj}'")

    # Example hand-off stub:
    # strategy = negotiation_agent.analyze_email(rfq_id=rfq_id, email_raw=raw)
    # draft = drafting_agent.compose_reply(rfq_id=rfq_id, strategy=strategy)
    # send_email_with_anchors(draft, rfq_id)

    mark_processed(bucket, key, message_id, provider_id, rfq_id, last_modified)
    return True

def handle_sqs_message(m: dict) -> bool:
    """
    Returns True on success (safe to delete). False to retry later.
    Expects S3 event notification format in the message body (direct or SNS-wrapped).
    """
    try:
        body = m.get("Body") or "{}"
        payload = json.loads(body)

        # SNS envelope support (if S3→SNS→SQS)
        if "Type" in payload and payload.get("TopicArn") and payload.get("Message"):
            payload = json.loads(payload["Message"])

        # S3 event
        recs = payload.get("Records") or []
        ok_all = True
        for r in recs:
            s3info = r.get("s3", {})
            bucket = s3info.get("bucket", {}).get("name")
            key = s3info.get("object", {}).get("key")
            if not bucket or not key:
                log.error(f"Malformed S3 record: {r}")
                ok_all = False
                continue
            # URL-decoding is handled by boto3, but keys in event are URL-encoded
            from urllib.parse import unquote_plus
            key = unquote_plus(key)

            ok = process_s3_record(bucket, key)
            ok_all = ok_all and ok

        return ok_all
    except Exception as e:
        log.exception(f"Failed to handle SQS message: {e}")
        return False

def run_loop():
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,       # long polling
            VisibilityTimeout=max(30, LOOKBACK_MIN * 60 // 3),
        )
        msgs = resp.get("Messages", [])
        if not msgs:
            continue

        to_delete = []
        for m in msgs:
            ok = handle_sqs_message(m)
            if ok and DELETE_ON_SUCCESS:
                to_delete.append({"Id": m["MessageId"], "ReceiptHandle": m["ReceiptHandle"]})

        if to_delete:
            try:
                sqs.delete_message_batch(QueueUrl=QUEUE_URL, Entries=to_delete)
            except ClientError as e:
                log.error(f"Delete batch failed: {e}")

if __name__ == "__main__":
    run_loop()
