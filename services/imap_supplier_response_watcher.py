"""IMAP based supplier response watcher.

This module implements the hard requirements for reconciling dispatched RFQs
with inbound supplier responses collected via IMAP.  Messages are persisted to
``proc.supplier_responses`` (or a SQLite development fallback) and then
processed concurrently via the Supplier Interaction agent or a supplied
callback.

The watcher enforces the following invariants:

* All dispatched RFQs for a (workflow_id, action_id, run_id) triplet must have
  a corresponding inbound email before processing begins.  Duplicates in either
  collection are ignored, and only RFQ identifiers derived from explicit
  markers are considered (thread-only matches are rejected for the gate).
* After the final dispatch is recorded in ``proc.email_dispatch_chains`` the
  watcher waits at least 90 seconds before inspecting the mailbox.  This
  mirrors the production behaviour of the SES watcher which provides a
  post-dispatch settling period.
* Supplier responses are deduplicated per RFQ (latest message wins) prior to
  invoking downstream processing.  Each message is processed exactly once and
  flagged via ``processed_at`` to ensure idempotency.

The implementation intentionally avoids coupling to the heavier SES watcher so
tests can exercise the behaviour with lightweight SQLite fixtures.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import hashlib
import imaplib
import json
import logging
import os
import time
import re
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from email.policy import default as default_policy
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency in pure-SQLite environments
    import psycopg2
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - psycopg2 may be missing for tests
    psycopg2 = None  # type: ignore[assignment]
    Json = None  # type: ignore[assignment]

import sqlite3

from agents.base_agent import AgentContext
from agents.supplier_interaction_agent import SupplierInteractionAgent
from services.email_dispatch_chain_store import mark_response as mark_dispatch_response
from utils.email_markers import extract_rfq_id, extract_run_id, extract_supplier_id, split_hidden_marker


logger = logging.getLogger(__name__)

TABLE_NAME = "proc.supplier_responses"
DISPATCH_TABLE = "proc.email_dispatch_chains"
_MANDATORY_WAIT_SECONDS = 90


@dataclasses.dataclass(frozen=True)
class SupplierResponseRecord:
    """Representation of a supplier email pulled from IMAP."""

    workflow_id: Optional[str]
    action_id: Optional[str]
    run_id: Optional[str]
    rfq_id: str
    supplier_id: Optional[str]
    message_id: Optional[str]
    subject: str
    body: str
    from_address: Optional[str]
    received_at: datetime
    headers: Dict[str, str]
    mailbox: Optional[str]

    def normalised_rfq(self) -> str:
        return (self.rfq_id or "").strip().upper()


class DatabaseBackend:
    """Simple abstraction that hides the PostgreSQL/SQLite differences."""

    def __init__(self) -> None:
        self.dialect = "postgres"
        self._conn = None
        if not os.getenv("PGHOST") and psycopg2 is None:
            self.dialect = "sqlite"
            self._conn = sqlite3.connect("./procwise_dev.sqlite")
        elif not os.getenv("PGHOST"):
            self.dialect = "sqlite"
            self._conn = sqlite3.connect("./procwise_dev.sqlite")
        else:
            if psycopg2 is None:  # pragma: no cover - installation guard
                raise RuntimeError("psycopg2 is required for PostgreSQL connections")
            self._conn = psycopg2.connect(
                host=os.getenv("PGHOST"),
                port=int(os.getenv("PGPORT", "5432")),
                dbname=os.getenv("PGDATABASE"),
                user=os.getenv("PGUSER"),
                password=os.getenv("PGPASSWORD"),
                sslmode=os.getenv("PGSSLMODE") or None,
            )
        if self.dialect == "sqlite":
            self._conn.row_factory = sqlite3.Row

    @property
    def connection(self):  # pragma: no cover - trivial accessor
        return self._conn

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._conn = None

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        if self.dialect == "postgres":
            self._ensure_postgres_schema()
        else:
            self._ensure_sqlite_schema()

    def _ensure_postgres_schema(self) -> None:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    workflow_id TEXT,
                    action_id TEXT,
                    run_id TEXT,
                    rfq_id TEXT NOT NULL,
                    supplier_id TEXT,
                    message_id TEXT,
                    message_hash TEXT,
                    from_address TEXT,
                    subject TEXT,
                    body TEXT,
                    raw_headers JSONB,
                    mailbox TEXT,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    processed_at TIMESTAMPTZ,
                    dispatch_run_id TEXT,
                    submitted_at TIMESTAMPTZ,
                    response_text TEXT,
                    price NUMERIC,
                    lead_time TEXT,
                    context_summary TEXT
                )
                """
            )
            # Backfill columns for existing deployments.
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS workflow_id TEXT"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS action_id TEXT"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS run_id TEXT"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS message_hash TEXT"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS dispatch_run_id TEXT"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS raw_headers JSONB"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS mailbox TEXT"
            )
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS supplier_responses_msg_hash"
                f" ON {TABLE_NAME}(message_hash)"
            )
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS supplier_responses_rfq_supplier"
                f" ON {TABLE_NAME}(rfq_id, supplier_id)"
            )
        self._conn.commit()

    def _ensure_sqlite_schema(self) -> None:
        assert self._conn is not None
        table = f'"{TABLE_NAME}"'
        with self._conn:  # sqlite auto-commits inside context
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT,
                    action_id TEXT,
                    run_id TEXT,
                    rfq_id TEXT NOT NULL,
                    supplier_id TEXT,
                    message_id TEXT,
                    message_hash TEXT,
                    from_address TEXT,
                    subject TEXT,
                    body TEXT,
                    raw_headers TEXT,
                    mailbox TEXT,
                    received_at TEXT NOT NULL,
                    processed_at TEXT,
                    dispatch_run_id TEXT,
                    submitted_at TEXT,
                    response_text TEXT,
                    price REAL,
                    lead_time TEXT,
                    context_summary TEXT
                )
                """
            )
            self._maybe_add_sqlite_column(table, "workflow_id TEXT")
            self._maybe_add_sqlite_column(table, "action_id TEXT")
            self._maybe_add_sqlite_column(table, "run_id TEXT")
            self._maybe_add_sqlite_column(table, "message_hash TEXT")
            self._maybe_add_sqlite_column(table, "processed_at TEXT")
            self._maybe_add_sqlite_column(table, "dispatch_run_id TEXT")
            self._maybe_add_sqlite_column(table, "raw_headers TEXT")
            self._maybe_add_sqlite_column(table, "mailbox TEXT")
            self._conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS supplier_responses_msg_hash"
                f" ON {table}(message_hash)"
            )
            self._conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS supplier_responses_rfq_supplier"
                f" ON {table}(rfq_id, supplier_id)"
            )

    def _maybe_add_sqlite_column(self, table: str, definition: str) -> None:
        assert self._conn is not None
        column_name = definition.split()[0].strip('"')
        with self._conn:
            existing = {
                row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")
            }
            if column_name not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def upsert_response(self, record: SupplierResponseRecord) -> None:
        assert self._conn is not None
        message_hash = _hash_message(record.message_id, record.body)
        headers_payload: Optional[str | Dict[str, str]]
        if self.dialect == "postgres":
            headers_payload = Json(record.headers) if Json is not None else json.dumps(record.headers)
        else:
            headers_payload = json.dumps(record.headers)

        if self.dialect == "postgres":
            query = f"""
                INSERT INTO {TABLE_NAME} (
                    workflow_id, action_id, run_id, rfq_id, supplier_id,
                    message_id, message_hash, from_address, subject, body,
                    raw_headers, mailbox, received_at, dispatch_run_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_hash) DO UPDATE SET
                    workflow_id = EXCLUDED.workflow_id,
                    action_id = EXCLUDED.action_id,
                    run_id = EXCLUDED.run_id,
                    rfq_id = EXCLUDED.rfq_id,
                    supplier_id = COALESCE(EXCLUDED.supplier_id, {TABLE_NAME}.supplier_id),
                    from_address = EXCLUDED.from_address,
                    subject = EXCLUDED.subject,
                    body = EXCLUDED.body,
                    raw_headers = EXCLUDED.raw_headers,
                    mailbox = EXCLUDED.mailbox,
                    received_at = EXCLUDED.received_at,
                    dispatch_run_id = EXCLUDED.dispatch_run_id,
                    processed_at = NULL
            """
        else:
            query = f"""
                INSERT INTO "{TABLE_NAME}" (
                    workflow_id, action_id, run_id, rfq_id, supplier_id,
                    message_id, message_hash, from_address, subject, body,
                    raw_headers, mailbox, received_at, dispatch_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_hash) DO UPDATE SET
                    workflow_id = excluded.workflow_id,
                    action_id = excluded.action_id,
                    run_id = excluded.run_id,
                    rfq_id = excluded.rfq_id,
                    supplier_id = COALESCE(excluded.supplier_id, supplier_id),
                    from_address = excluded.from_address,
                    subject = excluded.subject,
                    body = excluded.body,
                    raw_headers = excluded.raw_headers,
                    mailbox = excluded.mailbox,
                    received_at = excluded.received_at,
                    dispatch_run_id = excluded.dispatch_run_id,
                    processed_at = NULL
            """

        params: Tuple
        params = (
            record.workflow_id,
            record.action_id,
            record.run_id,
            record.rfq_id,
            record.supplier_id,
            record.message_id,
            message_hash,
            record.from_address,
            record.subject,
            record.body,
            headers_payload,
            record.mailbox,
            record.received_at if self.dialect == "postgres" else record.received_at.isoformat(),
            record.run_id,
        )

        with self._conn:  # type: ignore[arg-type]
            cursor = self._conn.cursor()
            try:
                cursor.execute(query, params)
            finally:
                cursor.close()

    def mark_processed(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat() if self.dialect == "sqlite" else datetime.now(timezone.utc)
        placeholder = (
            ",".join(["%s"] * len(ids)) if self.dialect == "postgres" else ",".join(["?"] * len(ids))
        )
        if self.dialect == "postgres":
            query = f"UPDATE {TABLE_NAME} SET processed_at = %s WHERE id IN ({placeholder})"
            params: Tuple = (now, *ids)
        else:
            query = f'UPDATE "{TABLE_NAME}" SET processed_at = ? WHERE id IN ({placeholder})'
            params = (now, *ids)
        with self._conn:  # type: ignore[arg-type]
            cursor = self._conn.cursor()
            try:
                cursor.execute(query, params)
            finally:
                cursor.close()

    def fetch_latest_per_rfq(
        self,
        *,
        workflow_id: Optional[str],
        run_id: Optional[str],
    ) -> List[Dict[str, object]]:
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            if self.dialect == "postgres":
                cursor.execute(
                    f"""
                    SELECT id, workflow_id, action_id, run_id, rfq_id, supplier_id,
                           message_id, subject, body, from_address, received_at,
                           mailbox
                    FROM {TABLE_NAME}
                    WHERE (%s IS NULL OR workflow_id = %s)
                      AND (%s IS NULL OR run_id = %s)
                      AND (processed_at IS NULL OR processed_at < received_at)
                    ORDER BY rfq_id, received_at DESC, id DESC
                    """,
                    (workflow_id, workflow_id, run_id, run_id),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT id, workflow_id, action_id, run_id, rfq_id, supplier_id,
                           message_id, subject, body, from_address, received_at,
                           mailbox
                    FROM "{TABLE_NAME}"
                    WHERE (? IS NULL OR workflow_id = ?)
                      AND (? IS NULL OR run_id = ?)
                      AND (processed_at IS NULL OR processed_at < received_at)
                    ORDER BY rfq_id, received_at DESC, id DESC
                    """,
                    (workflow_id, workflow_id, run_id, run_id),
                )
            rows = cursor.fetchall()
        finally:
            cursor.close()

        latest: Dict[str, Dict[str, object]] = {}
        for row in rows:
            if isinstance(row, sqlite3.Row):
                record = dict(row)
                received_at_value = row["received_at"]
            else:
                record = {
                    "id": row[0],
                    "workflow_id": row[1],
                    "action_id": row[2],
                    "run_id": row[3],
                    "rfq_id": row[4],
                    "supplier_id": row[5],
                    "message_id": row[6],
                    "subject": row[7],
                    "body": row[8],
                    "from_address": row[9],
                    "received_at": row[10],
                    "mailbox": row[11],
                }
                received_at_value = row[10]

            rfq_key = (record.get("rfq_id") or "").upper()
            if not rfq_key or rfq_key not in latest:
                record["received_at"] = _coerce_datetime(received_at_value)
                latest[rfq_key] = record
        return list(latest.values())

    def fetch_dispatch_lookup(
        self,
        *,
        workflow_id: Optional[str],
        action_id: Optional[str],
        run_id: Optional[str],
    ) -> Dict[str, Dict[str, Optional[str]]]:
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            if self.dialect == "postgres":
                cursor.execute(
                    f"""
                    SELECT message_id, rfq_id, workflow_ref, dispatch_metadata
                    FROM {DISPATCH_TABLE}
                    WHERE (%s IS NULL OR dispatch_metadata ->> 'run_id' = %s)
                      AND (%s IS NULL OR dispatch_metadata ->> 'workflow_id' = %s)
                      AND (%s IS NULL OR workflow_ref = %s OR dispatch_metadata ->> 'action_id' = %s)
                """,
                    (
                        run_id,
                        run_id,
                        workflow_id,
                        workflow_id,
                        action_id,
                        action_id,
                        action_id,
                    ),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT message_id, rfq_id, workflow_ref, dispatch_metadata
                    FROM "{DISPATCH_TABLE}"
                    WHERE (? IS NULL OR json_extract(dispatch_metadata, '$.run_id') = ?)
                      AND (? IS NULL OR json_extract(dispatch_metadata, '$.workflow_id') = ?)
                      AND (? IS NULL OR workflow_ref = ? OR json_extract(dispatch_metadata, '$.action_id') = ?)
                """,
                    (
                        run_id,
                        run_id,
                        workflow_id,
                        workflow_id,
                        action_id,
                        action_id,
                        action_id,
                    ),
                )
            rows = cursor.fetchall()
        except Exception as exc:
            message = str(exc).lower()
            if "does not exist" in message or "no such table" in message:
                return {}
            raise
        finally:
            cursor.close()

        lookup: Dict[str, Dict[str, Optional[str]]] = {}
        for row in rows:
            message_id = row[0]
            rfq = row[1]
            if not message_id or not rfq:
                continue
            metadata_raw = row[3]
            metadata: Dict[str, Optional[str]] = {}
            if isinstance(metadata_raw, dict):
                metadata = {key: metadata_raw.get(key) for key in metadata_raw}
            elif metadata_raw:
                try:
                    parsed = json.loads(metadata_raw)
                    if isinstance(parsed, dict):
                        metadata = {key: parsed.get(key) for key in parsed}
                except Exception:  # pragma: no cover - tolerate malformed JSON
                    metadata = {}

            key = str(message_id).strip()
            if key.startswith("<") and key.endswith(">"):
                key = key[1:-1].strip()
            if not key:
                continue

            lookup[key.lower()] = {
                "rfq_id": str(rfq).strip() if rfq else None,
                "workflow_id": _normalise_identifier(
                    metadata.get("workflow_id") or metadata.get("WORKFLOW_ID")
                ),
                "action_id": _normalise_identifier(
                    metadata.get("action_id") or metadata.get("ACTION_ID") or row[2]
                ),
                "run_id": _normalise_identifier(
                    metadata.get("run_id") or metadata.get("RUN_ID")
                ),
                "supplier_id": _normalise_identifier(
                    metadata.get("supplier_id") or metadata.get("SUPPLIER_ID")
                ),
            }
        return lookup

    def fetch_dispatched_rfqs(
        self,
        *,
        workflow_id: Optional[str],
        action_id: Optional[str],
        run_id: Optional[str],
    ) -> Tuple[Sequence[str], Optional[datetime]]:
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            try:
                if self.dialect == "postgres":
                    cursor.execute(
                        f"""
                        SELECT rfq_id, MAX(created_at)
                        FROM {DISPATCH_TABLE}
                        WHERE (%s IS NULL OR workflow_ref = %s)
                          AND (%s IS NULL OR dispatch_metadata ->> 'run_id' = %s)
                          AND (%s IS NULL OR dispatch_metadata ->> 'workflow_id' = %s)
                        GROUP BY rfq_id
                        """,
                        (action_id, action_id, run_id, run_id, workflow_id, workflow_id),
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT rfq_id, MAX(created_at)
                        FROM "{DISPATCH_TABLE}"
                        WHERE (? IS NULL OR workflow_ref = ?)
                          AND (? IS NULL OR json_extract(dispatch_metadata, '$.run_id') = ?)
                          AND (? IS NULL OR json_extract(dispatch_metadata, '$.workflow_id') = ?)
                        GROUP BY rfq_id
                        """,
                        (action_id, action_id, run_id, run_id, workflow_id, workflow_id),
                    )
                rows = cursor.fetchall()
            except Exception as exc:
                message = str(exc).lower()
                if "does not exist" in message or "no such table" in message:
                    return [], None
                raise
        finally:
            cursor.close()

        rfqs: List[str] = []
        last_dispatch: Optional[datetime] = None
        for row in rows:
            rfq_id = row[0]
            if not rfq_id:
                continue
            rfqs.append(str(rfq_id).strip().upper())
            candidate_time = row[1]
            if isinstance(candidate_time, str):
                try:
                    candidate_dt = parsedate_to_datetime(candidate_time)
                except Exception:  # pragma: no cover - defensive
                    candidate_dt = None
            else:
                candidate_dt = candidate_time
            if candidate_dt is not None:
                candidate_aware = (
                    candidate_dt if candidate_dt.tzinfo else candidate_dt.replace(tzinfo=timezone.utc)
                )
                if last_dispatch is None or candidate_aware > last_dispatch:
                    last_dispatch = candidate_aware
        return rfqs, last_dispatch

    def fetch_unique_inbound_rfqs(
        self,
        *,
        workflow_id: Optional[str],
        action_id: Optional[str],
        run_id: Optional[str],
    ) -> List[str]:
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            if self.dialect == "postgres":
                cursor.execute(
                    f"""
                    SELECT DISTINCT rfq_id
                    FROM {TABLE_NAME}
                    WHERE (%s IS NULL OR workflow_id = %s)
                      AND (%s IS NULL OR action_id = %s)
                      AND (%s IS NULL OR run_id = %s)
                """,
                    (workflow_id, workflow_id, action_id, action_id, run_id, run_id),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT DISTINCT rfq_id
                    FROM "{TABLE_NAME}"
                    WHERE (? IS NULL OR workflow_id = ?)
                      AND (? IS NULL OR action_id = ?)
                      AND (? IS NULL OR run_id = ?)
                """,
                    (workflow_id, workflow_id, action_id, action_id, run_id, run_id),
                )
            rows = cursor.fetchall()
        finally:
            cursor.close()

        rfqs = []
        for row in rows:
            if isinstance(row, sqlite3.Row):
                value = row[0]
            elif isinstance(row, (tuple, list)):
                value = row[0]
            else:
                value = row
            if not value:
                continue
            cleaned = str(value).strip().upper()
            if cleaned:
                rfqs.append(cleaned)
        return sorted(set(rfqs))


def _hash_message(message_id: Optional[str], body: str) -> str:
    base = f"{message_id or ''}|{body or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _coerce_datetime(value: Optional[object]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except Exception:
                parsed = None
        if parsed is None:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _decode_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        header = make_header(decode_header(value))
        return str(header)
    except Exception:  # pragma: no cover - defensive
        return value


def _message_received_at(msg: Message) -> datetime:
    date_value = msg.get("Date")
    parsed = None
    if date_value:
        try:
            parsed = parsedate_to_datetime(date_value)
        except Exception:  # pragma: no cover - resilience against malformed dates
            parsed = None
    if parsed is None:
        parsed = datetime.now(timezone.utc)
    elif parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _message_headers(msg: Message) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for key, value in msg.items():
        headers[key] = value
    return headers


def _message_body(msg: Message) -> str:
    if msg.is_multipart():
        parts: List[str] = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                try:
                    payload = part.get_payload(decode=True) or b""
                    parts.append(payload.decode(part.get_content_charset() or "utf-8", errors="ignore"))
                except Exception:  # pragma: no cover - best effort
                    continue
        if parts:
            return "\n".join(parts)
    try:
        payload = msg.get_payload(decode=True) or b""
        return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
    except Exception:  # pragma: no cover - fallback to str
        return str(msg.get_payload())


def _normalise_identifier(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        cleaned = str(value).strip()
    return cleaned or None


def _dispatch_header_identifiers(msg: Message) -> List[str]:
    raw_values: List[str] = []
    for header in ("In-Reply-To", "References"):
        value = msg.get(header)
        if not value:
            continue
        if isinstance(value, (list, tuple, set)):
            raw_values.extend(str(item) for item in value if item)
        else:
            raw_values.append(str(value))

    pattern = re.compile(r"<([^>]+)>")
    collected: List[str] = []
    for item in raw_values:
        stripped = item.strip()
        if not stripped:
            continue
        if stripped.startswith("<") and stripped.endswith(">"):
            candidate = stripped[1:-1].strip()
            if candidate:
                collected.append(candidate)
            continue
        matches = pattern.findall(stripped)
        if matches:
            collected.extend(match.strip() for match in matches if match.strip())
            continue
        collected.append(stripped)

    unique: List[str] = []
    seen: set[str] = set()
    for candidate in collected:
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(candidate)
    return unique


def _extract_rfq_payload(
    msg: Message,
    *,
    workflow_id: Optional[str],
    action_id: Optional[str],
    run_id: Optional[str],
    mailbox: Optional[str],
    dispatch_lookup: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> Optional[SupplierResponseRecord]:
    body = _message_body(msg)
    comment, _ = split_hidden_marker(body)
    rfq_id = extract_rfq_id(comment)
    run_identifier = extract_run_id(comment)
    supplier = extract_supplier_id(comment)

    matched_dispatch: Optional[Dict[str, Optional[str]]] = None
    if not rfq_id and dispatch_lookup:
        for identifier in _dispatch_header_identifiers(msg):
            lowered = identifier.strip().lower()
            if not lowered:
                continue
            dispatch = dispatch_lookup.get(lowered)
            if not dispatch:
                continue
            matched_dispatch = dispatch
            rfq_id = dispatch.get("rfq_id") or rfq_id
            if not supplier:
                supplier = dispatch.get("supplier_id")
            if not run_identifier:
                run_identifier = dispatch.get("run_id")
            if not workflow_id:
                workflow_id = dispatch.get("workflow_id")
            if not action_id:
                action_id = dispatch.get("action_id")
            break

    if run_id:
        if not run_identifier:
            return None
        if run_identifier.strip().lower() != run_id.strip().lower():
            return None
    elif not run_identifier:
        return None

    rfq_clean = (rfq_id or "").strip()
    if not rfq_clean:
        return None

    if matched_dispatch:
        dispatch_workflow = matched_dispatch.get("workflow_id")
        if dispatch_workflow and workflow_id:
            if dispatch_workflow.strip().lower() != workflow_id.strip().lower():
                return None

    headers = _message_headers(msg)
    subject = _decode_header(msg.get("Subject"))
    from_address = msg.get("From")
    received_at = _message_received_at(msg)
    message_id = msg.get("Message-ID")

    return SupplierResponseRecord(
        workflow_id=workflow_id,
        action_id=action_id,
        run_id=run_identifier,
        rfq_id=rfq_clean,
        supplier_id=supplier,
        message_id=message_id.strip() if isinstance(message_id, str) else message_id,
        subject=subject,
        body=body,
        from_address=from_address,
        received_at=received_at,
        headers=headers,
        mailbox=mailbox,
    )


def _imap_connection() -> imaplib.IMAP4:
    host = os.getenv("IMAP_HOST")
    username = os.getenv("IMAP_USERNAME")
    password = os.getenv("IMAP_PASSWORD")
    if not (host and username and password):
        raise RuntimeError("IMAP configuration is incomplete")

    port = int(os.getenv("IMAP_PORT", "993"))
    use_ssl = os.getenv("IMAP_SSL", "true").lower() != "false"
    if use_ssl:
        return imaplib.IMAP4_SSL(host, port)
    return imaplib.IMAP4(host, port)


def _collect_imap_messages(
    *,
    workflow_id: Optional[str],
    action_id: Optional[str],
    run_id: Optional[str],
    limit: Optional[int] = None,
    dispatch_lookup: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> List[SupplierResponseRecord]:
    mailbox = os.getenv("IMAP_FOLDER", "INBOX")
    connection = _imap_connection()
    try:
        connection.login(os.getenv("IMAP_USERNAME"), os.getenv("IMAP_PASSWORD"))
        status, _ = connection.select(mailbox)
        if status != "OK":
            raise RuntimeError(f"Unable to select IMAP folder {mailbox}")
        criteria = "ALL"
        status, data = connection.search(None, criteria)
        if status != "OK":  # pragma: no cover - network error
            raise RuntimeError("IMAP search failed")
        ids = data[0].split()
        if limit is not None and limit > 0:
            ids = ids[-limit:]
        responses: List[SupplierResponseRecord] = []
        for msg_id in ids:
            status, msg_data = connection.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue
            for part in msg_data:
                if not isinstance(part, tuple):
                    continue
                msg = BytesParser(policy=default_policy).parsebytes(part[1])
                record = _extract_rfq_payload(
                    msg,
                    workflow_id=workflow_id,
                    action_id=action_id,
                    run_id=run_id,
                    mailbox=mailbox,
                    dispatch_lookup=dispatch_lookup,
                )
                if record is None:
                    continue
                responses.append(record)
                break
        return responses
    finally:
        with contextlib.suppress(Exception):
            connection.logout()


def _wait_for_dispatch_completion(last_dispatch: Optional[datetime]) -> None:
    if last_dispatch is None:
        return
    target = last_dispatch + timedelta(seconds=_MANDATORY_WAIT_SECONDS)
    now = datetime.now(timezone.utc)
    remaining = (target - now).total_seconds()
    if remaining > 0:
        logger.debug("Waiting %.1fs after final dispatch before polling IMAP", remaining)
        time.sleep(remaining)


def _log_gate_mismatch(
    *,
    workflow_id: Optional[str],
    action_id: Optional[str],
    run_id: Optional[str],
    dispatched: Sequence[str],
    inbound: Sequence[str],
) -> None:
    dispatched_set = {rfq for rfq in dispatched if rfq}
    inbound_set = {rfq for rfq in inbound if rfq}
    missing = sorted(dispatched_set - inbound_set)
    extra = sorted(inbound_set - dispatched_set)
    logger.warning(
        "Supplier response gate mismatch workflow=%s action=%s run=%s dispatched=%d inbound=%d missing=%s extra=%s",
        workflow_id,
        action_id,
        run_id,
        len(dispatched_set),
        len(inbound_set),
        missing,
        extra,
    )


def _process_record_with_agent(
    agent_factory: Callable[[], SupplierInteractionAgent],
    record: Dict[str, object],
) -> None:
    agent = agent_factory()
    workflow_id = record.get("workflow_id") or os.getenv("PROCWISE_RUN_ID") or "workflow"
    context = AgentContext(
        workflow_id=str(workflow_id),
        agent_id="SupplierInteractionAgent",
        user_id="system",
        input_data={
            "subject": record.get("subject"),
            "message": record.get("body"),
            "from_address": record.get("from_address"),
            "rfq_id": record.get("rfq_id"),
            "supplier_id": record.get("supplier_id"),
            "message_id": record.get("message_id"),
            "workflow_id": record.get("workflow_id"),
            "dispatch_run_id": record.get("run_id"),
        },
    )
    agent.run(context)


def _default_agent_factory(agent_nick) -> Callable[[], SupplierInteractionAgent]:
    def _factory() -> SupplierInteractionAgent:
        return SupplierInteractionAgent(agent_nick)

    return _factory


def run_imap_supplier_response_watcher(
    *,
    agent_nick,
    workflow_id: Optional[str],
    action_id: Optional[str],
    run_id: Optional[str],
    process_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    supplier_agent_factory: Optional[Callable[[], SupplierInteractionAgent]] = None,
    mailbox_limit: Optional[int] = None,
    max_workers: int = 4,
) -> Dict[str, object]:
    """Execute the IMAP watcher end-to-end for the supplied workflow.

    Returns a summary dictionary containing counts for persisted and processed
    messages.  The function is idempotent; previously processed RFQs are skipped
    unless a newer message has been received.
    """

    db = DatabaseBackend()
    try:
        db.ensure_schema()
        dispatched_rfqs, last_dispatch = db.fetch_dispatched_rfqs(
            workflow_id=workflow_id, action_id=action_id, run_id=run_id
        )
        _wait_for_dispatch_completion(last_dispatch)

        dispatch_lookup = db.fetch_dispatch_lookup(
            workflow_id=workflow_id, action_id=action_id, run_id=run_id
        )
        records = _collect_imap_messages(
            workflow_id=workflow_id,
            action_id=action_id,
            run_id=run_id,
            limit=mailbox_limit,
            dispatch_lookup=dispatch_lookup,
        )
        for record in records:
            db.upsert_response(record)
            with contextlib.suppress(Exception):
                conn = db.connection
                mark_dispatch_response(
                    conn,
                    rfq_id=record.rfq_id,
                    in_reply_to=record.headers.get("In-Reply-To"),
                    references=record.headers.get("References"),
                    response_message_id=record.message_id,
                    response_metadata={
                        "workflow_id": workflow_id,
                        "action_id": action_id,
                        "run_id": record.run_id,
                        "mailbox": record.mailbox,
                    },
                )

        inbound_rfqs = db.fetch_unique_inbound_rfqs(
            workflow_id=workflow_id, action_id=action_id, run_id=run_id
        )
        dispatched_set = [rfq.strip().upper() for rfq in dispatched_rfqs]
        if set(dispatched_set) != set(inbound_rfqs):
            _log_gate_mismatch(
                workflow_id=workflow_id,
                action_id=action_id,
                run_id=run_id,
                dispatched=dispatched_set,
                inbound=inbound_rfqs,
            )
            return {
                "persisted": len(records),
                "processed": 0,
                "dispatched_rfqs": sorted(set(dispatched_set)),
                "inbound_rfqs": inbound_rfqs,
            }

        latest_records = db.fetch_latest_per_rfq(workflow_id=workflow_id, run_id=run_id)
        if not latest_records:
            return {
                "persisted": len(records),
                "processed": 0,
                "dispatched_rfqs": sorted(set(dispatched_set)),
                "inbound_rfqs": inbound_rfqs,
            }

        if process_callback is None:
            factory = supplier_agent_factory or _default_agent_factory(agent_nick)

            def callback(payload: Dict[str, object]) -> None:
                _process_record_with_agent(factory, payload)

            process_callback = callback

        processed_ids: List[int] = []
        futures: Dict[concurrent.futures.Future[None], Dict[str, object]] = {}
        workers = max(1, min(max_workers, len(latest_records)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for record in latest_records:
                future = executor.submit(process_callback, record)
                futures[future] = record
            for future in concurrent.futures.as_completed(futures):
                record = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "Supplier response processing failed for rfq=%s", record.get("rfq_id")
                    )
                    continue
                record_id = record.get("id")
                if isinstance(record_id, int):
                    processed_ids.append(record_id)

        db.mark_processed(processed_ids)

        return {
            "persisted": len(records),
            "processed": len(processed_ids),
            "dispatched_rfqs": sorted(set(dispatched_set)),
            "inbound_rfqs": inbound_rfqs,
        }
    finally:
        db.close()


__all__ = ["run_imap_supplier_response_watcher", "DatabaseBackend", "SupplierResponseRecord"]
