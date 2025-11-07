from __future__ import annotations

import logging
import json
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

CREATE SEQUENCE IF NOT EXISTS proc.supplier_response_id_seq;

CREATE TABLE IF NOT EXISTS proc.supplier_response (
    id BIGINT PRIMARY KEY DEFAULT nextval('proc.supplier_response_id_seq'),
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    rfq_id TEXT,
    response_message_id TEXT,
    response_subject TEXT,
    response_text TEXT,
    response_body TEXT,
    body_html TEXT,
    response_from TEXT,
    round_number INTEGER,
    in_reply_to JSONB,
    reference_ids JSONB,
    response_date TIMESTAMPTZ,
    original_message_id TEXT,
    original_subject TEXT,
    match_confidence NUMERIC(4, 2),
    match_score NUMERIC(5, 3),
    price NUMERIC(18, 4),
    currency TEXT,
    payment_terms TEXT,
    warranty TEXT,
    validity TEXT,
    exceptions TEXT,
    lead_time INTEGER,
    response_time NUMERIC(18, 6),
    tables JSONB,
    attachments JSONB,
    received_time TIMESTAMPTZ,
    dispatch_id TEXT,
    matched_on JSONB,
    raw_headers JSONB,
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
    rfq_id: Optional[str] = None
    response_body: Optional[str] = None
    body_html: Optional[str] = None
    response_message_id: Optional[str] = None
    response_subject: Optional[str] = None
    response_from: Optional[str] = None
    round_number: Optional[int] = None
    in_reply_to: Optional[Sequence[str]] = None
    references: Optional[Sequence[str]] = None
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


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, Decimal)):
        return str(value)
    return value


def _serialise_optional_json(value: Optional[object]) -> Optional[str]:
    if value in (None, "", [], {}, ()):  # pragma: no cover - simple guard
        return None
    try:
        return json.dumps(value, default=_json_default)
    except TypeError:
        return json.dumps(str(value))


def _normalise_message_identifier(value: Optional[object]) -> Optional[str]:
    if value in (None, "", False):
        return None
    if isinstance(value, (list, tuple, set)):
        for candidate in value:
            formatted = _normalise_message_identifier(candidate)
            if formatted:
                return formatted
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)) and parsed:
            return _normalise_message_identifier(parsed[0])
    text = text.strip("<>").strip()
    if not text:
        return None
    return f"<{text}>"


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


def _table_has_column(cur, schema: str, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        LIMIT 1
        """,
        (schema, table, column),
    )
    return cur.fetchone() is not None


def _remove_duplicate_response_message_ids(cur) -> None:
    """Remove duplicate rows that share the same workflow/message identifier."""

    if _table_has_column(cur, "proc", "supplier_response", "id"):
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
    else:
        cur.execute(
            """
            WITH duplicates AS (
                SELECT ctid
                FROM (
                    SELECT
                        ctid,
                        ROW_NUMBER() OVER (
                            PARTITION BY workflow_id, response_message_id
                            ORDER BY
                                COALESCE(processed, FALSE) DESC,
                                created_at DESC,
                                unique_id DESC
                        ) AS rn
                    FROM proc.supplier_response
                    WHERE response_message_id IS NOT NULL
                ) ranked
                WHERE rn > 1
            )
            DELETE FROM proc.supplier_response sr
            USING duplicates d
            WHERE sr.ctid = d.ctid
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
        _ensure_postgres_column(cur, "proc", "supplier_response", "body_html", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "supplier_email", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "rfq_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_message_id", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_subject", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_from", "TEXT")
        _ensure_postgres_column(cur, "proc", "supplier_response", "round_number", "INTEGER")
        _ensure_postgres_column(cur, "proc", "supplier_response", "in_reply_to", "JSONB")
        _ensure_postgres_column(cur, "proc", "supplier_response", "reference_ids", "JSONB")
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
    base_text = row.response_text or row.response_body or ""
    response_text = base_text or ""
    response_body = row.response_body if row.response_body not in (None, "") else None
    if response_body is None:
        response_body = response_text
    body_html = row.body_html if row.body_html not in (None, "") else None
    received_time = _normalise_dt(row.received_time)
    response_date = received_time
    price_value = _serialise_decimal(row.price)
    lead_time = row.lead_time
    match_confidence = _serialise_decimal(row.match_confidence)
    match_score = _serialise_decimal(row.match_score)
    response_time = _serialise_decimal(row.response_time)
    supplier_email = row.supplier_email or None
    rfq_id = row.rfq_id or None
    response_message_id = row.response_message_id or None
    response_subject = row.response_subject or None
    response_from = row.response_from or None
    try:
        round_number_value = int(row.round_number) if row.round_number is not None else None
    except Exception:
        round_number_value = None
    original_message_id = _normalise_message_identifier(row.original_message_id)
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

    in_reply_to_json = _serialise_optional_json(row.in_reply_to)
    references_json = _serialise_optional_json(row.references)
    def _coerce_text_field(value: Optional[object]) -> Optional[str]:
        if value in (None, ""):
            return None
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(items) if items else None
        if isinstance(value, dict):
            try:
                return json.dumps(value, default=_json_default)
            except Exception:
                return str(value)
        return str(value)

    currency = _coerce_text_field(getattr(row, "currency", None))
    payment_terms = _coerce_text_field(getattr(row, "payment_terms", None))
    warranty = _coerce_text_field(getattr(row, "warranty", None))
    validity = _coerce_text_field(getattr(row, "validity", None))
    exceptions = _coerce_text_field(getattr(row, "exceptions", None))
    match_evidence = _serialise_match_evidence(getattr(row, "match_evidence", None))
    tables_json = _serialise_optional_json(getattr(row, "tables", None))
    attachments_json = _serialise_optional_json(getattr(row, "attachments", None))
    matched_on = matched_on_json
    raw_headers = raw_headers_json

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
                    "body_html=COALESCE(%s, body_html), "
                    "response_subject=COALESCE(%s, response_subject), "
                    "response_from=COALESCE(%s, response_from), "
                    "round_number=COALESCE(%s, round_number), "
                    "in_reply_to=COALESCE(%s, in_reply_to), "
                    "reference_ids=COALESCE(%s, reference_ids), "
                    "response_date=COALESCE(%s, response_date), "
                    "original_message_id=COALESCE(%s, original_message_id), "
                    "original_subject=COALESCE(%s, original_subject), "
                    "match_confidence=COALESCE(%s, match_confidence), "
                    "match_score=COALESCE(%s, match_score), "
                    "price=COALESCE(%s, price), "
                    "currency=COALESCE(%s, currency), "
                    "payment_terms=COALESCE(%s, payment_terms), "
                    "warranty=COALESCE(%s, warranty), "
                    "validity=COALESCE(%s, validity), "
                    "exceptions=COALESCE(%s, exceptions), "
                    "lead_time=COALESCE(%s, lead_time), "
                    "response_time=COALESCE(%s, response_time), "
                    "received_time=COALESCE(%s, received_time), "
                    "match_evidence=COALESCE(%s, match_evidence), "
                    "matched_on=COALESCE(%s, matched_on), "
                    "raw_headers=COALESCE(%s, raw_headers), "
                    "tables=COALESCE(%s, tables), "
                    "attachments=COALESCE(%s, attachments), "
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
                        body_html,
                        response_subject,
                        response_from,
                        round_number_value,
                        in_reply_to_json,
                        references_json,
                        response_date,
                        original_message_id,
                        original_subject,
                        match_confidence,
                        match_score,
                        price_value,
                        currency,
                        payment_terms,
                        warranty,
                        validity,
                        exceptions,
                        lead_time,
                        response_time,
                        received_time,
                        match_evidence,
                        matched_on,
                        raw_headers,
                        tables_json,
                        attachments_json,
                        processed,
                        row.workflow_id,
                        response_message_id,
                    ),
                )
                cur.close()
                try:
                    notify_response_received(
                        workflow_id=row.workflow_id,
                        unique_id=row.unique_id,
                        round_number=row.round_number,
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Failed to notify response coordinator for workflow=%s unique_id=%s",
                        row.workflow_id,
                        row.unique_id,
                    )
                return

        insert_query = (
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
            "round_number=COALESCE(EXCLUDED.round_number, proc.supplier_response.round_number), "
            "in_reply_to=COALESCE(EXCLUDED.in_reply_to, proc.supplier_response.in_reply_to), "
            "reference_ids=COALESCE(EXCLUDED.reference_ids, proc.supplier_response.reference_ids), "
            "response_date=COALESCE(EXCLUDED.response_date, proc.supplier_response.response_date), "
            "original_message_id=COALESCE(EXCLUDED.original_message_id, proc.supplier_response.original_message_id), "
            "original_subject=COALESCE(EXCLUDED.original_subject, proc.supplier_response.original_subject), "
            "match_confidence=COALESCE(EXCLUDED.match_confidence, proc.supplier_response.match_confidence), "
            "match_score=COALESCE(EXCLUDED.match_score, proc.supplier_response.match_score), "
            "price=COALESCE(EXCLUDED.price, proc.supplier_response.price), "
            "currency=COALESCE(EXCLUDED.currency, proc.supplier_response.currency), "
            "payment_terms=COALESCE(EXCLUDED.payment_terms, proc.supplier_response.payment_terms), "
            "warranty=COALESCE(EXCLUDED.warranty, proc.supplier_response.warranty), "
            "validity=COALESCE(EXCLUDED.validity, proc.supplier_response.validity), "
            "exceptions=COALESCE(EXCLUDED.exceptions, proc.supplier_response.exceptions), "
            "lead_time=COALESCE(EXCLUDED.lead_time, proc.supplier_response.lead_time), "
            "response_time=COALESCE(EXCLUDED.response_time, proc.supplier_response.response_time), "
            "received_time=COALESCE(EXCLUDED.received_time, proc.supplier_response.received_time), "
            "dispatch_id=COALESCE(EXCLUDED.dispatch_id, proc.supplier_response.dispatch_id), "
            "matched_on=COALESCE(EXCLUDED.matched_on, proc.supplier_response.matched_on), "
            "raw_headers=COALESCE(EXCLUDED.raw_headers, proc.supplier_response.raw_headers), "
            "processed=EXCLUDED.processed"
        )
        cur.execute(
            insert_query,
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
                round_number_value,
                in_reply_to_json,
                references_json,
                response_date,
                original_message_id,
                original_subject,
                match_confidence,
                match_score_value,
                price_value,
                currency,
                payment_terms,
                warranty,
                validity,
                exceptions,
                lead_time,
                response_time,
                received_time,
                match_evidence,
                matched_on,
                raw_headers,
                tables_json,
                attachments_json,
                processed,
                dispatch_id,
                matched_on_json,
                raw_headers_json,
            ),
        )
        cur.close()

    try:
        notify_response_received(
            workflow_id=row.workflow_id,
            unique_id=row.unique_id,
            round_number=row.round_number,
        )
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
            "SELECT workflow_id, supplier_id, supplier_email, rfq_id, unique_id, response_text, response_body, body_html, "
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


def fetch_for_unique_ids(
    *,
    workflow_id: str,
    unique_ids: Sequence[str],
    supplier_ids: Optional[Sequence[str]] = None,
    include_processed: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch supplier responses filtered by unique and supplier identifiers."""

    identifiers = [uid for uid in unique_ids if uid]
    supplier_filter = [sid for sid in (supplier_ids or []) if sid]

    if not identifiers and not supplier_filter:
        return []

    filters = ["workflow_id=%s"]
    params: List[Any] = [workflow_id]

    if not include_processed:
        filters.append("COALESCE(processed, FALSE)=FALSE")

    if identifiers:
        filters.append("unique_id = ANY(%s)")
        params.append(identifiers)

    if supplier_filter:
        filters.append("supplier_id = ANY(%s)")
        params.append(supplier_filter)

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "SELECT workflow_id, supplier_id, supplier_email, rfq_id, unique_id, response_text, response_body, body_html, "
            "response_message_id, response_subject, response_from, response_date, original_message_id, "
            "original_subject, match_confidence, match_score, match_evidence, matched_on, raw_headers, price, currency, "
            "payment_terms, warranty, validity, exceptions, lead_time, response_time, received_time, tables, attachments, processed "
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
            "SELECT workflow_id, supplier_id, supplier_email, rfq_id, unique_id, response_text, response_body, body_html, "
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
