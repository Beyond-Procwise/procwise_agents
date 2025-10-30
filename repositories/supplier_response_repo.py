from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence

from services.db import get_conn
from services.supplier_response_coordinator import notify_response_received

logger = logging.getLogger(__name__)

DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE SEQUENCE IF NOT EXISTS proc.supplier_response_id_seq;

CREATE TABLE IF NOT EXISTS proc.supplier_response (
    id BIGINT PRIMARY KEY DEFAULT nextval('proc.supplier_response_id_seq'),
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    response_message_id TEXT,
    response_subject TEXT,
    response_text TEXT,
    response_body TEXT,
    response_from TEXT,
    response_date TIMESTAMPTZ,
    original_message_id TEXT,
    original_subject TEXT,
    match_confidence NUMERIC(4, 2),
    match_score NUMERIC(5, 3),
    price NUMERIC(18, 4),
    lead_time INTEGER,
    response_time NUMERIC(18, 6),
    received_time TIMESTAMPTZ,
    dispatch_id TEXT,
    matched_on JSONB,
    raw_headers JSONB,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT supplier_response_unique UNIQUE (workflow_id, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_supplier_response_wf
ON proc.supplier_response (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_response_supplier
ON proc.supplier_response (supplier_id);

CREATE INDEX IF NOT EXISTS idx_supplier_response_unique
ON proc.supplier_response (unique_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_supplier_response_workflow_msg
ON proc.supplier_response (workflow_id, response_message_id);

ALTER TABLE proc.supplier_response
    ALTER COLUMN id SET DEFAULT nextval('proc.supplier_response_id_seq');
"""

@dataclass
class SupplierResponseRow:
    workflow_id: str
    unique_id: str
    supplier_id: Optional[str]
    response_text: str
    received_time: datetime
    response_time: Optional[Decimal] = None
    price: Optional[Decimal] = None
    lead_time: Optional[int] = None
    supplier_email: Optional[str] = None
    response_message_id: Optional[str] = None
    response_subject: Optional[str] = None
    response_from: Optional[str] = None
    original_message_id: Optional[str] = None
    original_subject: Optional[str] = None
    match_confidence: Optional[Decimal] = None
    response_body: Optional[str] = None
    match_score: Optional[float] = None
    matched_on: Optional[Sequence[str]] = None
    dispatch_id: Optional[str] = None
    raw_headers: Optional[Dict[str, Sequence[str]]] = None
    processed: bool = False


def _normalise_dt(value: datetime) -> datetime:
    if value.tzinfo:
        return value
    return value.replace(tzinfo=timezone.utc)


def _serialise_decimal(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _coerce_bool(value: Optional[bool]) -> bool:
    return bool(value)


def _ensure_postgres_column(cur, schema: str, table: str, column: str, definition: str) -> None:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        LIMIT 1
        """,
        (schema, table, column),
    )
    if cur.fetchone():
        return

    cur.execute(
        f'ALTER TABLE "{schema}"."{table}" ADD COLUMN "{column}" {definition}'
    )


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL_PG.split(";"))):
            cur.execute(statement)
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_text", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_body", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "supplier_email", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_message_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_subject", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_from", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_date", "TIMESTAMPTZ")
        _ensure_postgres_column(cur, "proc", "supplier_response", "original_message_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "original_subject", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "match_confidence", "NUMERIC(4, 2)")
        _ensure_postgres_column(cur, "proc", "supplier_response", "match_score", "NUMERIC(5, 3)")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_time", "NUMERIC(18, 6)")
        _ensure_postgres_column(cur, "proc", "supplier_response", "dispatch_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "matched_on", "JSONB")
        _ensure_postgres_column(cur, "proc", "supplier_response", "raw_headers", "JSONB")
        _ensure_postgres_column(cur, "proc", "supplier_response", "processed", "BOOLEAN DEFAULT FALSE")
        cur.close()


def insert_response(row: SupplierResponseRow) -> None:
    response_text = row.response_text or ""
    received_time = _normalise_dt(row.received_time)
    price_value = _serialise_decimal(row.price)
    match_confidence = _serialise_decimal(row.match_confidence)
    response_time = _serialise_decimal(row.response_time)
    supplier_email = row.supplier_email or None
    response_message_id = row.response_message_id or None
    response_subject = row.response_subject or None
    response_from = row.response_from or None
    original_message_id = row.original_message_id or None
    original_subject = row.original_subject or None
    processed = _coerce_bool(row.processed)
    response_date = received_time
    response_body = row.response_body or response_text
    dispatch_id = row.dispatch_id or None
    match_score_value = None
    if row.match_score is not None:
        try:
            match_score_value = str(Decimal(str(row.match_score)))
        except Exception:
            match_score_value = str(row.match_score)
    matched_on_json = None
    if row.matched_on:
        try:
            matched_on_json = json.dumps(list(row.matched_on))
        except Exception:
            matched_on_json = json.dumps([str(item) for item in row.matched_on])
    raw_headers_json = None
    if row.raw_headers:
        try:
            serialisable = {key: list(value) for key, value in row.raw_headers.items()}
        except Exception:
            serialisable = None
        if serialisable is not None:
            raw_headers_json = json.dumps(serialisable)

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "INSERT INTO proc.supplier_response "
            "(workflow_id, unique_id, supplier_id, supplier_email, response_message_id, response_subject, "
            "response_text, response_body, response_from, response_date, original_message_id, original_subject, "
            "match_confidence, match_score, price, lead_time, response_time, received_time, processed, dispatch_id, matched_on, raw_headers) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT(workflow_id, response_message_id) DO UPDATE SET "
            "supplier_id=COALESCE(EXCLUDED.supplier_id, proc.supplier_response.supplier_id), "
            "supplier_email=COALESCE(EXCLUDED.supplier_email, proc.supplier_response.supplier_email), "
            "unique_id=EXCLUDED.unique_id, "
            "response_text=EXCLUDED.response_text, "
            "response_body=EXCLUDED.response_body, "
            "response_subject=COALESCE(EXCLUDED.response_subject, proc.supplier_response.response_subject), "
            "response_from=COALESCE(EXCLUDED.response_from, proc.supplier_response.response_from), "
            "response_date=COALESCE(EXCLUDED.response_date, proc.supplier_response.response_date), "
            "original_message_id=COALESCE(EXCLUDED.original_message_id, proc.supplier_response.original_message_id), "
            "original_subject=COALESCE(EXCLUDED.original_subject, proc.supplier_response.original_subject), "
            "match_confidence=COALESCE(EXCLUDED.match_confidence, proc.supplier_response.match_confidence), "
            "match_score=COALESCE(EXCLUDED.match_score, proc.supplier_response.match_score), "
            "price=COALESCE(EXCLUDED.price, proc.supplier_response.price), "
            "lead_time=COALESCE(EXCLUDED.lead_time, proc.supplier_response.lead_time), "
            "response_time=COALESCE(EXCLUDED.response_time, proc.supplier_response.response_time), "
            "received_time=COALESCE(EXCLUDED.received_time, proc.supplier_response.received_time), "
            "dispatch_id=COALESCE(EXCLUDED.dispatch_id, proc.supplier_response.dispatch_id), "
            "matched_on=COALESCE(EXCLUDED.matched_on, proc.supplier_response.matched_on), "
            "raw_headers=COALESCE(EXCLUDED.raw_headers, proc.supplier_response.raw_headers), "
            "processed=EXCLUDED.processed"
        )
        cur.execute(
            q,
            (
                row.workflow_id,
                row.unique_id,
                row.supplier_id,
                supplier_email,
                response_message_id,
                response_subject,
                response_text,
                response_body,
                response_from,
                response_date,
                original_message_id,
                original_subject,
                match_confidence,
                match_score_value,
                price_value,
                row.lead_time,
                response_time,
                received_time,
                processed,
                dispatch_id,
                matched_on_json,
                raw_headers_json,
            ),
        )
        cur.close()

    try:
        notify_response_received(workflow_id=row.workflow_id, unique_id=row.unique_id)
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "Failed to broadcast response event for workflow=%s unique_id=%s",
            row.workflow_id,
            row.unique_id,
        )


def delete_responses(*, workflow_id: str, unique_ids: Iterable[str]) -> None:
    ids = [uid for uid in unique_ids if uid]
    if not ids:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "UPDATE proc.supplier_response SET processed=TRUE "
            "WHERE workflow_id=%s AND unique_id = ANY(%s)"
        )
        cur.execute(q, (workflow_id, ids))
        cur.close()


def align_workflow_assignments(*, workflow_id: str, unique_ids: Iterable[str]) -> None:
    """Ensure the provided unique identifiers are attached to the workflow."""

    identifiers: List[str] = []
    for uid in unique_ids:
        if uid in (None, ""):
            continue
        identifiers.append(str(uid))
    if not identifiers:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "UPDATE proc.supplier_response "
            "SET workflow_id=%s "
            "WHERE unique_id = ANY(%s) AND workflow_id <> %s"
        )
        cur.execute(q, (workflow_id, identifiers, workflow_id))
        cur.close()


def count_pending(
    *,
    workflow_id: str,
    unique_ids: Optional[Sequence[str]] = None,
    supplier_ids: Optional[Sequence[str]] = None,
    include_processed: bool = False,
) -> int:
    ids = [uid for uid in (unique_ids or []) if uid]
    supplier_filter = [sid for sid in (supplier_ids or []) if sid]

    with get_conn() as conn:
        cur = conn.cursor()
        filters = ["workflow_id=%s"]
        params: List[Any] = [workflow_id]

        if not include_processed:
            filters.append("COALESCE(processed, FALSE)=FALSE")

        if ids:
            filters.append("unique_id = ANY(%s)")
            params.append(ids)

        if supplier_filter:
            filters.append("supplier_id = ANY(%s)")
            params.append(supplier_filter)

        q = (
            "SELECT COUNT(*) FROM proc.supplier_response "
            f"WHERE {' AND '.join(filters)}"
        )
        cur.execute(q, tuple(params))
        row = cur.fetchone()
        cur.close()

        if not row:
            return 0
        try:
            return int(row[0])
        except Exception:
            return 0


def fetch_pending(
    *, workflow_id: str, include_processed: bool = False
) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        filters = ["workflow_id=%s"]
        params: List[Any] = [workflow_id]

        if not include_processed:
            filters.append("COALESCE(processed, FALSE)=FALSE")

        q = (
            "SELECT workflow_id, supplier_id, supplier_email, unique_id, response_text, response_body, "
            "response_message_id, response_subject, response_from, response_date, original_message_id, "
            "original_subject, match_confidence, match_score, price, lead_time, received_time, processed, "
            "dispatch_id, matched_on, raw_headers "
            "FROM proc.supplier_response "
            f"WHERE {' AND '.join(filters)}"
        )
        cur.execute(q, tuple(params))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, rec)) for rec in cur.fetchall()]
        cur.close()
        return [_normalise_row(row) for row in rows]


def fetch_all(*, workflow_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "SELECT workflow_id, supplier_id, supplier_email, unique_id, response_text, response_body, "
            "response_message_id, response_subject, response_from, response_date, original_message_id, "
            "original_subject, match_confidence, match_score, price, lead_time, received_time, processed, "
            "dispatch_id, matched_on, raw_headers "
            "FROM proc.supplier_response WHERE workflow_id=%s"
        )
        cur.execute(q, (workflow_id,))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, rec)) for rec in cur.fetchall()]
        cur.close()
        return [_normalise_row(row) for row in rows]


def reset_workflow(*, workflow_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.supplier_response WHERE workflow_id=%s", (workflow_id,))
        cur.close()


def lookup_workflow_for_unique(*, unique_id: str) -> Optional[str]:
    if not unique_id:
        return None

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "SELECT workflow_id "
            "FROM proc.supplier_response "
            "WHERE unique_id=%s "
            "LIMIT 1"
        )
        cur.execute(q, (unique_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        workflow_id = row[0]
        return workflow_id or None


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    payload.setdefault("response_body", payload.get("response_text"))
    payload.setdefault("subject", payload.get("response_subject"))
    payload.setdefault("from_addr", payload.get("response_from"))
    payload.setdefault("message_id", payload.get("response_message_id"))
    payload.setdefault("response_time", payload.get("response_time"))
    payload.setdefault("received_time", payload.get("response_date") or payload.get("received_time"))
    payload.setdefault("processed", _coerce_bool(payload.get("processed")))
    matched_on = payload.get("matched_on")
    if isinstance(matched_on, str):
        try:
            payload["matched_on"] = json.loads(matched_on)
        except Exception:
            payload["matched_on"] = [matched_on]
    raw_headers = payload.get("raw_headers")
    if isinstance(raw_headers, str):
        try:
            payload["raw_headers"] = json.loads(raw_headers)
        except Exception:
            payload["raw_headers"] = None
    return payload
