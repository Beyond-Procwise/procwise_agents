from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence

from services.db import get_conn
from services.supplier_response_coordinator import notify_response_received

logger = logging.getLogger(__name__)

DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.supplier_response (
    id SERIAL PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    rfq_id TEXT,
    response_message_id TEXT,
    response_subject TEXT,
    response_text TEXT,
    response_body TEXT,
    response_from TEXT,
    response_date TIMESTAMPTZ,
    original_message_id TEXT,
    original_subject TEXT,
    match_confidence NUMERIC(4, 2),
    match_evidence JSONB,
    raw_headers JSONB,
    price NUMERIC(18, 4),
    lead_time INTEGER,
    response_time NUMERIC(18, 6),
    received_time TIMESTAMPTZ,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT supplier_response_unique UNIQUE (workflow_id, unique_id),
    CONSTRAINT supplier_response_message_unique UNIQUE (workflow_id, response_message_id)
);

CREATE INDEX IF NOT EXISTS idx_supplier_response_wf
ON proc.supplier_response (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_response_supplier
ON proc.supplier_response (supplier_id);

CREATE INDEX IF NOT EXISTS idx_supplier_response_unique
ON proc.supplier_response (unique_id);
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
    rfq_id: Optional[str] = None
    response_body: Optional[str] = None
    response_message_id: Optional[str] = None
    response_subject: Optional[str] = None
    response_from: Optional[str] = None
    original_message_id: Optional[str] = None
    original_subject: Optional[str] = None
    match_confidence: Optional[Decimal] = None
    match_evidence: Optional[Sequence[str]] = None
    raw_headers: Optional[Dict[str, Any]] = None
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


def _serialise_match_evidence(value: Optional[Sequence[str]]) -> Optional[str]:
    if not value:
        return None
    items = [str(entry).strip() for entry in value if str(entry).strip()]
    if not items:
        return None
    return json.dumps(items)


def _serialise_headers(value: Optional[Dict[str, Any]]) -> Optional[str]:
    if not value:
        return None
    serialised: Dict[str, List[str]] = {}
    for key, raw in value.items():
        if raw in (None, ""):
            continue
        if isinstance(raw, (list, tuple, set)):
            entries = [str(item).strip() for item in raw if str(item).strip()]
        else:
            text = str(raw).strip()
            entries = [text] if text else []
        if entries:
            serialised[str(key)] = entries
    if not serialised:
        return None
    return json.dumps(serialised)


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


def _remove_duplicate_response_message_ids(cur) -> None:
    """Remove duplicate rows that share the same workflow/message identifier."""

    cur.execute(
        """
        WITH duplicates AS (
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY workflow_id, response_message_id
                        ORDER BY
                            COALESCE(processed, FALSE) DESC,
                            created_at DESC,
                            id DESC
                    ) AS rn
                FROM proc.supplier_response
                WHERE response_message_id IS NOT NULL
            ) ranked
            WHERE rn > 1
        )
        DELETE FROM proc.supplier_response sr
        USING duplicates d
        WHERE sr.id = d.id
        RETURNING sr.workflow_id, sr.response_message_id
        """
    )

    removed = cur.fetchall()
    if removed:
        logger.warning(
            "Removed %d duplicate supplier responses prior to creating message id index", len(removed)
        )


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL_PG.split(";"))):
            cur.execute(statement)
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_text", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_body", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "supplier_email", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "rfq_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_message_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_subject", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_from", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_date", "TIMESTAMPTZ")
        _ensure_postgres_column(cur, "proc", "supplier_response", "original_message_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "original_subject", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "match_confidence", "NUMERIC(4, 2)")
        _ensure_postgres_column(cur, "proc", "supplier_response", "match_evidence", "JSONB")
        _ensure_postgres_column(cur, "proc", "supplier_response", "raw_headers", "JSONB")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_time", "NUMERIC(18, 6)")
        _ensure_postgres_column(cur, "proc", "supplier_response", "processed", "BOOLEAN DEFAULT FALSE")
        _remove_duplicate_response_message_ids(cur)
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_supplier_response_message_unique
            ON proc.supplier_response (workflow_id, response_message_id)
            WHERE response_message_id IS NOT NULL
            """
        )
        cur.close()


def insert_response(row: SupplierResponseRow) -> None:
    response_text = row.response_text or ""
    response_body = row.response_body if row.response_body not in (None, "") else None
    if response_body is None:
        response_body = response_text
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
    rfq_id = row.rfq_id or None
    match_evidence = _serialise_match_evidence(row.match_evidence)
    raw_headers = _serialise_headers(row.raw_headers)

    with get_conn() as conn:
        cur = conn.cursor()
        if response_message_id:
            cur.execute(
                "SELECT unique_id FROM proc.supplier_response WHERE workflow_id=%s AND response_message_id=%s",
                (row.workflow_id, response_message_id),
            )
            existing = cur.fetchone()
            if existing:
                update_q = (
                    "UPDATE proc.supplier_response SET "
                    "supplier_id=COALESCE(%s, supplier_id), "
                    "supplier_email=COALESCE(%s, supplier_email), "
                    "rfq_id=COALESCE(%s, rfq_id), "
                    "response_text=%s, "
                    "response_body=%s, "
                    "response_subject=COALESCE(%s, response_subject), "
                    "response_from=COALESCE(%s, response_from), "
                    "response_date=COALESCE(%s, response_date), "
                    "original_message_id=COALESCE(%s, original_message_id), "
                    "original_subject=COALESCE(%s, original_subject), "
                    "match_confidence=COALESCE(%s, match_confidence), "
                    "price=COALESCE(%s, price), "
                    "lead_time=COALESCE(%s, lead_time), "
                    "response_time=COALESCE(%s, response_time), "
                    "received_time=COALESCE(%s, received_time), "
                    "match_evidence=COALESCE(%s, match_evidence), "
                    "raw_headers=COALESCE(%s, raw_headers), "
                    "processed=COALESCE(processed, FALSE) OR %s "
                    "WHERE workflow_id=%s AND response_message_id=%s"
                )
                cur.execute(
                    update_q,
                    (
                        row.supplier_id,
                        supplier_email,
                        rfq_id,
                        response_text,
                        response_body,
                        response_subject,
                        response_from,
                        response_date,
                        original_message_id,
                        original_subject,
                        match_confidence,
                        price_value,
                        row.lead_time,
                        response_time,
                        received_time,
                        match_evidence,
                        raw_headers,
                        processed,
                        row.workflow_id,
                        response_message_id,
                    ),
                )
                cur.close()
                try:
                    notify_response_received(workflow_id=row.workflow_id, unique_id=row.unique_id)
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Failed to notify response coordinator for workflow=%s unique_id=%s",
                        row.workflow_id,
                        row.unique_id,
                    )
                return
        q = (
            "INSERT INTO proc.supplier_response "
            "(workflow_id, supplier_id, supplier_email, rfq_id, unique_id, response_text, response_body, "
            "response_message_id, response_subject, response_from, response_date, original_message_id, "
            "original_subject, match_confidence, price, lead_time, response_time, received_time, match_evidence, raw_headers, processed) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT(workflow_id, unique_id) DO UPDATE SET "
            "supplier_id=COALESCE(EXCLUDED.supplier_id, proc.supplier_response.supplier_id), "
            "supplier_email=COALESCE(EXCLUDED.supplier_email, proc.supplier_response.supplier_email), "
            "rfq_id=COALESCE(EXCLUDED.rfq_id, proc.supplier_response.rfq_id), "
            "response_text=EXCLUDED.response_text, "
            "response_body=EXCLUDED.response_body, "
            "response_message_id=COALESCE(EXCLUDED.response_message_id, proc.supplier_response.response_message_id), "
            "response_subject=COALESCE(EXCLUDED.response_subject, proc.supplier_response.response_subject), "
            "response_from=COALESCE(EXCLUDED.response_from, proc.supplier_response.response_from), "
            "response_date=COALESCE(EXCLUDED.response_date, proc.supplier_response.response_date), "
            "original_message_id=COALESCE(EXCLUDED.original_message_id, proc.supplier_response.original_message_id), "
            "original_subject=COALESCE(EXCLUDED.original_subject, proc.supplier_response.original_subject), "
            "match_confidence=COALESCE(EXCLUDED.match_confidence, proc.supplier_response.match_confidence), "
            "price=COALESCE(EXCLUDED.price, proc.supplier_response.price), "
            "lead_time=COALESCE(EXCLUDED.lead_time, proc.supplier_response.lead_time), "
            "response_time=COALESCE(EXCLUDED.response_time, proc.supplier_response.response_time), "
            "received_time=COALESCE(EXCLUDED.received_time, proc.supplier_response.received_time), "
            "match_evidence=COALESCE(EXCLUDED.match_evidence, proc.supplier_response.match_evidence), "
            "raw_headers=COALESCE(EXCLUDED.raw_headers, proc.supplier_response.raw_headers), "
            "processed=proc.supplier_response.processed OR EXCLUDED.processed"
        )
        cur.execute(
            q,
            (
                row.workflow_id,
                row.supplier_id,
                supplier_email,
                rfq_id,
                row.unique_id,
                response_text,
                response_body,
                response_message_id,
                response_subject,
                response_from,
                response_date,
                original_message_id,
                original_subject,
                match_confidence,
                price_value,
                row.lead_time,
                response_time,
                received_time,
                match_evidence,
                raw_headers,
                processed,
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
            "SELECT workflow_id, supplier_id, supplier_email, rfq_id, unique_id, response_text, response_body, "
            "response_message_id, response_subject, response_from, response_date, original_message_id, "
            "original_subject, match_confidence, match_evidence, raw_headers, price, lead_time, response_time, received_time, processed "
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
            "SELECT workflow_id, supplier_id, supplier_email, rfq_id, unique_id, response_text, response_body, "
            "response_message_id, response_subject, response_from, response_date, original_message_id, "
            "original_subject, match_confidence, match_evidence, raw_headers, price, lead_time, response_time, received_time, processed "
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
    payload.setdefault("body_text", payload.get("response_text"))
    payload.setdefault("body_html", payload.get("response_body"))
    evidence = payload.get("match_evidence")
    if isinstance(evidence, str):
        try:
            payload["match_evidence"] = json.loads(evidence)
        except Exception:
            pass
    headers = payload.get("raw_headers")
    if isinstance(headers, str):
        try:
            payload["raw_headers"] = json.loads(headers)
        except Exception:
            pass
    return payload
