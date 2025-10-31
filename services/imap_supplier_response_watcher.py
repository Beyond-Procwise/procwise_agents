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
from utils.email_markers import (
    extract_marker_token,
    extract_rfq_id,
    extract_run_id,
    extract_supplier_id,
    split_hidden_marker,
)


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
    attachments: Sequence[Dict[str, Any]] = dataclasses.field(default_factory=tuple)
    matched_sent_email_id: Optional[str] = None
    round_number: Optional[int] = None
    matched_dispatch_subject: Optional[str] = None
    match_score: float = 0.0
    match_method: str = "unknown"
    matched_on: Sequence[str] = dataclasses.field(default_factory=tuple)
    match_confidence: float = 0.0

    def normalised_rfq(self) -> str:
        return (self.rfq_id or "").strip().upper()


@dataclasses.dataclass(frozen=True)
class DispatchRecord:
    """Representation of a dispatched email awaiting a supplier reply."""

    rfq_id: Optional[str]
    supplier_id: Optional[str]
    workflow_id: Optional[str]
    action_id: Optional[str]
    token: Optional[str]
    message_id: Optional[str]
    mailbox: Optional[str]
    recipient: Optional[str] = None
    subject: Optional[str] = None
    dispatched_at: Optional[datetime] = None
    round_number: Optional[int] = None

    def normalised_token(self) -> Optional[str]:
        token = _normalise_identifier(self.token)
        if token:
            return token.lower()
        return None

    def normalised_supplier(self) -> Optional[str]:
        supplier = _normalise_identifier(self.supplier_id)
        if supplier:
            return supplier.lower()
        return None


@dataclasses.dataclass
class DispatchContext:
    """Container providing lookup indexes for dispatched emails."""

    records: List[DispatchRecord]
    by_token: Dict[str, DispatchRecord]
    by_message_id: Dict[str, DispatchRecord]
    by_supplier: Dict[str, List[DispatchRecord]]
    metadata: Dict[str, Dict[str, Any]]

    @classmethod
    def build(cls, records: Sequence[DispatchRecord]) -> "DispatchContext":
        token_index: Dict[str, DispatchRecord] = {}
        message_index: Dict[str, DispatchRecord] = {}
        supplier_index: Dict[str, List[DispatchRecord]] = {}
        metadata: Dict[str, Dict[str, Any]] = {}
        for record in records:
            token_norm = record.normalised_token()
            if token_norm and token_norm not in token_index:
                token_index[token_norm] = record
            message_id = _normalise_identifier(record.message_id)
            if message_id:
                lowered = message_id.lower()
                if lowered not in message_index:
                    message_index[lowered] = record
                if message_id.startswith("<") and message_id.endswith(">"):
                    inner = message_id[1:-1].strip().lower()
                    if inner and inner not in message_index:
                        message_index[inner] = record
            supplier_norm = record.normalised_supplier()
            if supplier_norm:
                supplier_index.setdefault(supplier_norm, []).append(record)
            metadata[record.message_id or f"{record.token}:{record.supplier_id}"] = {
                "recipient": record.recipient,
                "subject": record.subject,
                "dispatched_at": record.dispatched_at.isoformat()
                if isinstance(record.dispatched_at, datetime)
                else None,
                "workflow_id": record.workflow_id,
                "round_number": record.round_number,
            }
        return cls(list(records), token_index, message_index, supplier_index, metadata)

    def match_token(self, token: Optional[str]) -> Optional[DispatchRecord]:
        if not token:
            return None
        return self.by_token.get(token.strip().lower())

    def match_message(self, identifiers: Sequence[str]) -> Optional[DispatchRecord]:
        for identifier in identifiers:
            lowered = identifier.strip().lower()
            if not lowered:
                continue
            match = self.by_message_id.get(lowered)
            if match:
                return match
        return None

    def match_supplier(self, supplier_id: Optional[str]) -> Optional[DispatchRecord]:
        if not supplier_id:
            return None
        supplier_norm = supplier_id.strip().lower()
        if not supplier_norm:
            return None
        matches = self.by_supplier.get(supplier_norm)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        return None


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
                    context_summary TEXT,
                    attachments JSONB,
                    matched_sent_email_id TEXT,
                    round_number INTEGER,
                    matched_dispatch_subject TEXT
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
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS attachments JSONB"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS matched_sent_email_id TEXT"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS round_number INTEGER"
            )
            cur.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS matched_dispatch_subject TEXT"
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
                    context_summary TEXT,
                    attachments TEXT,
                    matched_sent_email_id TEXT,
                    round_number INTEGER,
                    matched_dispatch_subject TEXT
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
            self._maybe_add_sqlite_column(table, "attachments TEXT")
            self._maybe_add_sqlite_column(table, "matched_sent_email_id TEXT")
            self._maybe_add_sqlite_column(table, "round_number INTEGER")
            self._maybe_add_sqlite_column(table, "matched_dispatch_subject TEXT")
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

        attachments_serialised: Any
        attachments_list = list(record.attachments)
        if self.dialect == "postgres":
            attachments_serialised = (
                Json(attachments_list)
                if Json is not None
                else json.dumps(attachments_list)
            )
        else:
            attachments_serialised = json.dumps(attachments_list)

        if self.dialect == "postgres":
            query = f"""
                INSERT INTO {TABLE_NAME} (
                    workflow_id, action_id, run_id, rfq_id, supplier_id,
                    message_id, message_hash, from_address, subject, body,
                    raw_headers, mailbox, received_at, dispatch_run_id,
                    attachments, matched_sent_email_id, round_number, matched_dispatch_subject
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    attachments = EXCLUDED.attachments,
                    matched_sent_email_id = EXCLUDED.matched_sent_email_id,
                    round_number = EXCLUDED.round_number,
                    matched_dispatch_subject = EXCLUDED.matched_dispatch_subject,
                    processed_at = NULL
            """
        else:
            query = f"""
                INSERT INTO "{TABLE_NAME}" (
                    workflow_id, action_id, run_id, rfq_id, supplier_id,
                    message_id, message_hash, from_address, subject, body,
                    raw_headers, mailbox, received_at, dispatch_run_id,
                    attachments, matched_sent_email_id, round_number, matched_dispatch_subject
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    attachments = excluded.attachments,
                    matched_sent_email_id = excluded.matched_sent_email_id,
                    round_number = excluded.round_number,
                    matched_dispatch_subject = excluded.matched_dispatch_subject,
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
            record.received_at
            if self.dialect == "postgres"
            else record.received_at.isoformat(),
            record.run_id,
            attachments_serialised,
            record.matched_sent_email_id,
            record.round_number,
            record.matched_dispatch_subject,
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
                           mailbox, dispatch_run_id
                    FROM {TABLE_NAME}
                    WHERE (%s IS NULL OR workflow_id = %s)
                      AND (%s IS NULL OR dispatch_run_id = %s)
                      AND (processed_at IS NULL OR processed_at < received_at)
                    ORDER BY received_at DESC, id DESC
                    """,
                    (workflow_id, workflow_id, run_id, run_id),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT id, workflow_id, action_id, run_id, rfq_id, supplier_id,
                           message_id, subject, body, from_address, received_at,
                           mailbox, dispatch_run_id
                    FROM "{TABLE_NAME}"
                    WHERE (? IS NULL OR workflow_id = ?)
                      AND (? IS NULL OR dispatch_run_id = ?)
                      AND (processed_at IS NULL OR processed_at < received_at)
                    ORDER BY received_at DESC, id DESC
                    """,
                    (workflow_id, workflow_id, run_id, run_id),
                )
            rows = cursor.fetchall()
        finally:
            cursor.close()

        latest: Dict[Tuple[str, str, str], Dict[str, object]] = {}
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
                    "received_at_raw": row[10],
                    "mailbox": row[11],
                    "dispatch_run_id": row[12],
                }
                received_at_value = row[10]

            token_value = record.get("dispatch_run_id") or record.get("run_id")
            supplier_value = record.get("supplier_id") or ""
            rfq_value = record.get("rfq_id") or ""
            key = (
                (str(token_value).strip().lower() if token_value else ""),
                str(supplier_value).strip().lower(),
                str(rfq_value).strip().upper(),
            )
            if key in latest:
                continue
            record["received_at"] = _coerce_datetime(received_at_value)
            if token_value:
                record["run_id"] = str(token_value)
            latest[key] = record
        return list(latest.values())

    def mark_dispatch_by_token(
        self,
        *,
        token: Optional[str],
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if not token:
            return
        assert self._conn is not None
        payload = json.dumps(metadata) if metadata else None
        token_norm = token.strip()
        if not token_norm:
            return
        if self.dialect == "postgres":
            query = f"""
                UPDATE {DISPATCH_TABLE}
                SET awaiting_response = FALSE,
                    responded_at = NOW(),
                    response_metadata = COALESCE(%s, response_metadata),
                    updated_at = NOW()
                WHERE dispatch_metadata ->> 'run_id' = %s
                  AND awaiting_response = TRUE
            """
            params = (payload, token_norm)
        else:
            query = f"""
                UPDATE "{DISPATCH_TABLE}"
                SET awaiting_response = 0,
                    responded_at = CURRENT_TIMESTAMP,
                    response_metadata = COALESCE(?, response_metadata),
                    updated_at = CURRENT_TIMESTAMP
                WHERE json_extract(dispatch_metadata, '$.run_id') = ?
                  AND awaiting_response = 1
            """
            params = (payload, token_norm)
        with self._conn:  # type: ignore[arg-type]
            cursor = self._conn.cursor()
            try:
                cursor.execute(query, params)
            finally:
                cursor.close()

    def fetch_dispatch_context(
        self,
        *,
        workflow_id: Optional[str],
        action_id: Optional[str],
        run_id: Optional[str],
    ) -> Tuple[List[DispatchRecord], Optional[datetime]]:
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            try:
                if self.dialect == "postgres":
                    cursor.execute(
                        f"""
                        SELECT message_id, rfq_id, workflow_ref, dispatch_metadata,
                               created_at, awaiting_response
                        FROM {DISPATCH_TABLE}
                        WHERE (%s IS NULL OR dispatch_metadata ->> 'workflow_id' = %s)
                          AND (%s IS NULL OR workflow_ref = %s OR dispatch_metadata ->> 'action_id' = %s)
                          AND (%s IS NULL OR dispatch_metadata ->> 'run_id' = %s)
                        """,
                        (
                            workflow_id,
                            workflow_id,
                            action_id,
                            action_id,
                            action_id,
                            run_id,
                            run_id,
                        ),
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT message_id, rfq_id, workflow_ref, dispatch_metadata,
                               created_at, awaiting_response
                        FROM "{DISPATCH_TABLE}"
                        WHERE (? IS NULL OR json_extract(dispatch_metadata, '$.workflow_id') = ?)
                          AND (? IS NULL OR workflow_ref = ? OR json_extract(dispatch_metadata, '$.action_id') = ?)
                          AND (? IS NULL OR json_extract(dispatch_metadata, '$.run_id') = ?)
                        """,
                        (
                            workflow_id,
                            workflow_id,
                            action_id,
                            action_id,
                            action_id,
                            run_id,
                            run_id,
                        ),
                    )
                rows = cursor.fetchall()
            except Exception as exc:
                message = str(exc).lower()
                if "does not exist" in message or "no such table" in message:
                    return [], None
                raise
        finally:
            cursor.close()

        records: List[DispatchRecord] = []
        last_dispatch: Optional[datetime] = None
        for row in rows:
            message_id = row[0]
            rfq_id = row[1]
            workflow_ref = row[2]
            metadata_raw = row[3]
            created_at = row[4]
            awaiting = row[5]
            if awaiting in (False, 0):
                continue
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

            token = metadata.get("run_id") or metadata.get("dispatch_token")
            supplier = metadata.get("supplier_id") or metadata.get("SUPPLIER_ID")
            workflow_meta = metadata.get("workflow_id") or metadata.get("WORKFLOW_ID")
            action_meta = metadata.get("action_id") or metadata.get("ACTION_ID")
            mailbox_meta = metadata.get("mailbox") or metadata.get("MAILBOX")
            recipient_meta = metadata.get("recipient") or metadata.get("to")
            subject_meta = metadata.get("subject") or metadata.get("SUBJECT")
            round_meta = metadata.get("round") or metadata.get("round_number")

            dispatched_at: Optional[datetime] = None
            if isinstance(created_at, str):
                try:
                    dispatched_at = parsedate_to_datetime(created_at)
                except Exception:  # pragma: no cover - defensive
                    dispatched_at = None
            elif isinstance(created_at, datetime):
                dispatched_at = created_at if created_at.tzinfo else created_at.replace(
                    tzinfo=timezone.utc
                )

            round_number: Optional[int] = None
            if round_meta is not None:
                try:
                    round_number = int(round_meta)
                except (TypeError, ValueError):
                    round_number = None

            record = DispatchRecord(
                rfq_id=_normalise_identifier(rfq_id),
                supplier_id=_normalise_identifier(supplier) or None,
                workflow_id=_normalise_identifier(workflow_meta) or _normalise_identifier(workflow_id),
                action_id=_normalise_identifier(action_meta) or _normalise_identifier(workflow_ref),
                token=_normalise_identifier(token),
                message_id=_normalise_identifier(message_id),
                mailbox=_normalise_identifier(mailbox_meta),
                recipient=_normalise_identifier(recipient_meta) or metadata.get("recipient"),
                subject=subject_meta,
                dispatched_at=dispatched_at,
                round_number=round_number,
            )
            records.append(record)

            if dispatched_at is not None:
                if last_dispatch is None or dispatched_at > last_dispatch:
                    last_dispatch = dispatched_at

        return records, last_dispatch

    def fetch_inbound_tokens(
        self,
        *,
        workflow_id: Optional[str],
        action_id: Optional[str],
    ) -> Dict[str, Dict[str, Optional[str]]]:
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            if self.dialect == "postgres":
                cursor.execute(
                    f"""
                    SELECT DISTINCT COALESCE(dispatch_run_id, run_id) AS token,
                                    rfq_id,
                                    supplier_id
                    FROM {TABLE_NAME}
                    WHERE (%s IS NULL OR workflow_id = %s)
                      AND (%s IS NULL OR action_id = %s)
                """,
                    (workflow_id, workflow_id, action_id, action_id),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT DISTINCT COALESCE(dispatch_run_id, run_id) AS token,
                                    rfq_id,
                                    supplier_id
                    FROM "{TABLE_NAME}"
                    WHERE (? IS NULL OR workflow_id = ?)
                      AND (? IS NULL OR action_id = ?)
                """,
                    (workflow_id, workflow_id, action_id, action_id),
                )
            rows = cursor.fetchall()
        finally:
            cursor.close()

        tokens: Dict[str, Dict[str, Optional[str]]] = {}
        for row in rows:
            if isinstance(row, sqlite3.Row):
                token_value = row[0]
                rfq_value = row[1]
                supplier_value = row[2]
            elif isinstance(row, (tuple, list)):
                token_value, rfq_value, supplier_value = row[0], row[1], row[2]
            else:
                token_value = getattr(row, "token", None)
                rfq_value = getattr(row, "rfq_id", None)
                supplier_value = getattr(row, "supplier_id", None)

            token_norm = _normalise_identifier(token_value)
            if not token_norm:
                continue
            tokens[token_norm.lower()] = {
                "token": token_norm,
                "rfq_id": _normalise_identifier(rfq_value),
                "supplier_id": _normalise_identifier(supplier_value),
            }
        return tokens


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
    dispatch_context: Optional[DispatchContext] = None,
) -> Optional[SupplierResponseRecord]:
    body = _message_body(msg)
    comment, _ = split_hidden_marker(body)
    rfq_id = extract_rfq_id(comment)
    run_identifier = extract_run_id(comment)
    token = extract_marker_token(comment)
    supplier = extract_supplier_id(comment)

    matched_dispatch: Optional[DispatchRecord] = None
    if dispatch_context:
        dispatch_token = token or run_identifier
        matched_dispatch = dispatch_context.match_token(dispatch_token)
        if not matched_dispatch:
            matched_dispatch = dispatch_context.match_message(_dispatch_header_identifiers(msg))
        if not matched_dispatch and supplier:
            matched_dispatch = dispatch_context.match_supplier(supplier)

    if matched_dispatch:
        rfq_id = rfq_id or matched_dispatch.rfq_id
        supplier = supplier or matched_dispatch.supplier_id
        workflow_id = workflow_id or matched_dispatch.workflow_id
        action_id = action_id or matched_dispatch.action_id
        token = token or matched_dispatch.token
        run_identifier = run_identifier or matched_dispatch.token

    run_candidate = token or run_identifier
    if run_candidate:
        run_identifier = run_candidate

    if run_id and run_identifier:
        if run_identifier.strip().lower() != run_id.strip().lower():
            if not dispatch_context or dispatch_context.match_token(run_identifier) is None:
                return None
    elif not run_identifier:
        return None

    rfq_clean = (rfq_id or "").strip()
    if not rfq_clean and matched_dispatch:
        rfq_clean = matched_dispatch.rfq_id or ""
    if not rfq_clean:
        return None

    if matched_dispatch:
        dispatch_workflow = matched_dispatch.workflow_id
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
    dispatch_context: Optional[DispatchContext] = None,
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
                    dispatch_context=dispatch_context,
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
        dispatch_records, last_dispatch = db.fetch_dispatch_context(
            workflow_id=workflow_id, action_id=action_id, run_id=run_id
        )
        dispatch_context = DispatchContext.build(dispatch_records)
        _wait_for_dispatch_completion(last_dispatch)

        records = _collect_imap_messages(
            workflow_id=workflow_id,
            action_id=action_id,
            run_id=run_id,
            limit=mailbox_limit,
            dispatch_context=dispatch_context,
        )
        for record in records:
            db.upsert_response(record)
            matched = dispatch_context.match_token(record.run_id)
            metadata = {
                "workflow_id": workflow_id,
                "action_id": action_id,
                "run_id": record.run_id,
                "mailbox": record.mailbox,
            }
            with contextlib.suppress(Exception):
                conn = db.connection
                updated = mark_dispatch_response(
                    conn,
                    rfq_id=record.rfq_id,
                    in_reply_to=record.headers.get("In-Reply-To"),
                    references=record.headers.get("References"),
                    response_message_id=record.message_id,
                    response_metadata=metadata,
                )
                if not updated and matched and matched.token:
                    db.mark_dispatch_by_token(token=matched.token, metadata=metadata)

        expected_tokens = sorted(
            {rec.normalised_token() for rec in dispatch_records if rec.normalised_token()}
        )
        inbound_tokens = db.fetch_inbound_tokens(workflow_id=workflow_id, action_id=action_id)
        inbound_token_keys = sorted(inbound_tokens.keys())

        if expected_tokens and set(expected_tokens) != set(inbound_token_keys):
            missing = sorted(set(expected_tokens) - set(inbound_token_keys))
            extra = sorted(set(inbound_token_keys) - set(expected_tokens))
            logger.warning(
                "Supplier response gate mismatch workflow=%s action=%s run=%s expected=%d inbound=%d missing=%s extra=%s",
                workflow_id,
                action_id,
                run_id,
                len(expected_tokens),
                len(inbound_token_keys),
                missing,
                extra,
            )
            return {
                "persisted": len(records),
                "processed": 0,
                "expected_tokens": expected_tokens,
                "inbound_tokens": inbound_token_keys,
            }

        latest_records = db.fetch_latest_per_rfq(workflow_id=workflow_id, run_id=None)
        if not latest_records:
            return {
                "persisted": len(records),
                "processed": 0,
                "expected_tokens": expected_tokens,
                "inbound_tokens": inbound_token_keys,
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
            "expected_tokens": expected_tokens,
            "inbound_tokens": inbound_token_keys,
        }
    finally:
        db.close()


__all__ = ["run_imap_supplier_response_watcher", "DatabaseBackend", "SupplierResponseRecord"]
