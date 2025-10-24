"""Persistence helpers for workflow-level email dispatch tracking."""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from services.db import get_conn


DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.workflow_email_tracking (
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    message_id TEXT,
    subject TEXT,
    dispatched_at TIMESTAMPTZ NOT NULL,
    responded_at TIMESTAMPTZ,
    response_message_id TEXT,
    matched BOOLEAN DEFAULT FALSE,
    thread_headers JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workflow_id, unique_id)
);

ALTER TABLE proc.workflow_email_tracking
    ADD COLUMN IF NOT EXISTS thread_headers JSONB;

CREATE INDEX IF NOT EXISTS idx_workflow_email_tracking_wf
ON proc.workflow_email_tracking (workflow_id);

CREATE INDEX IF NOT EXISTS idx_workflow_email_tracking_unique
ON proc.workflow_email_tracking (unique_id);
"""

@dataclass
class WorkflowDispatchRow:
    workflow_id: str
    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    dispatched_at: datetime
    responded_at: Optional[datetime]
    response_message_id: Optional[str]
    matched: bool
    thread_headers: Optional[Dict[str, Sequence[str]]] = None


def _normalise_dt(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raise TypeError("expected datetime for timestamp fields")


def _dedupe_workflow_unique_pairs(cur) -> None:
    """Remove duplicate (workflow_id, unique_id) rows before creating unique index."""

    cur.execute("SELECT to_regclass('proc.workflow_email_tracking')")
    table_name = cur.fetchone()[0]
    if not table_name:
        return

    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'proc'
          AND table_name = 'workflow_email_tracking'
        """
    )
    available_columns = {row[0] for row in cur.fetchall()}

    ordering_clauses = ["dispatched_at DESC NULLS LAST"]
    if "responded_at" in available_columns:
        ordering_clauses.append("responded_at DESC NULLS LAST")
    if "created_at" in available_columns:
        ordering_clauses.append("created_at DESC NULLS LAST")
    if "message_id" in available_columns:
        ordering_clauses.append("message_id DESC NULLS LAST")

    ordering = ", ".join(ordering_clauses) if ordering_clauses else "dispatched_at DESC"

    cur.execute(
        f"""
        WITH ranked AS (
            SELECT
                ctid,
                ROW_NUMBER() OVER (
                    PARTITION BY workflow_id, unique_id
                    ORDER BY {ordering}
                ) AS rn
            FROM proc.workflow_email_tracking
        )
        DELETE FROM proc.workflow_email_tracking t
        USING ranked
        WHERE t.ctid = ranked.ctid
        AND ranked.rn > 1;
        """
    )


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL_PG)
        _dedupe_workflow_unique_pairs(cur)
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_email_tracking_wf_unique
            ON proc.workflow_email_tracking (workflow_id, unique_id);
            """
        )
        cur.close()


def _parse_thread_headers(value: Optional[object]) -> Optional[Dict[str, Sequence[str]]]:
    if value in (None, "", b"", {}):
        return None
    raw = value
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode()
        except Exception:
            raw = bytes(raw).decode("utf-8", "ignore")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except Exception:
            return None
    elif isinstance(raw, dict):
        payload = raw
    else:
        return None

    result: Dict[str, Sequence[str]] = {}
    for key, items in payload.items():
        if items in (None, ""):
            continue
        if isinstance(items, (list, tuple, set)):
            values = tuple(str(item).strip("<> ") for item in items if str(item).strip())
        else:
            text = str(items).strip()
            values = (text,) if text else tuple()
        if values:
            result[str(key)] = values
    return result or None


def load_active_workflow_ids() -> List[str]:
    """Return workflow identifiers with dispatched emails awaiting responses."""

    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        query = (
            "SELECT DISTINCT workflow_id "
            "FROM proc.workflow_email_tracking "
            "WHERE dispatched_at IS NOT NULL "
            "AND (responded_at IS NULL OR matched = FALSE)"
        )
        cur.execute(query)
        rows = [row[0] for row in cur.fetchall() if row and row[0]]
        cur.close()
        return rows


def _serialise_thread_headers(headers: Optional[Dict[str, Sequence[str]]]) -> Optional[str]:
    if not headers:
        return None
    serialised: Dict[str, List[str]] = {}
    for key, value in headers.items():
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple, set)):
            entries = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            entries = [text] if text else []
        if entries:
            serialised[str(key)] = entries
    if not serialised:
        return None
    return json.dumps(serialised)


def record_dispatches(
    *,
    workflow_id: str,
    dispatches: Iterable[WorkflowDispatchRow],
) -> None:
    rows = list(dispatches)
    if not rows:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "INSERT INTO proc.workflow_email_tracking "
            "(workflow_id, unique_id, supplier_id, supplier_email, message_id, subject, "
            "dispatched_at, responded_at, response_message_id, matched, thread_headers) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (workflow_id, unique_id) DO UPDATE SET "
            "supplier_id=EXCLUDED.supplier_id, "
            "supplier_email=EXCLUDED.supplier_email, "
            "message_id=EXCLUDED.message_id, "
            "subject=EXCLUDED.subject, "
            "dispatched_at=EXCLUDED.dispatched_at, "
            "thread_headers=EXCLUDED.thread_headers"
        )
        params = [
            (
                row.workflow_id,
                row.unique_id,
                row.supplier_id,
                row.supplier_email,
                row.message_id,
                row.subject,
                _normalise_dt(row.dispatched_at),
                _normalise_dt(row.responded_at),
                row.response_message_id,
                row.matched,
                _serialise_thread_headers(row.thread_headers),
            )
            for row in rows
        ]
        cur.executemany(q, params)
        cur.close()


def load_workflow_rows(*, workflow_id: str) -> List[WorkflowDispatchRow]:
    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "SELECT workflow_id, unique_id, supplier_id, supplier_email, message_id, subject, "
            "dispatched_at, responded_at, response_message_id, matched, thread_headers "
            "FROM proc.workflow_email_tracking WHERE workflow_id=%s"
        )
        cur.execute(q, (workflow_id,))
        fetched = cur.fetchall()
        cols = [c.name for c in cur.description]
        cur.close()
        rows = []
        for record in fetched:
            data = dict(zip(cols, record))
            rows.append(
                WorkflowDispatchRow(
                    workflow_id=data["workflow_id"],
                    unique_id=data["unique_id"],
                    supplier_id=data.get("supplier_id"),
                    supplier_email=data.get("supplier_email"),
                    message_id=data.get("message_id"),
                    subject=data.get("subject"),
                    dispatched_at=_normalise_dt(data.get("dispatched_at")),
                    responded_at=_normalise_dt(data.get("responded_at")) if data.get("responded_at") else None,
                    response_message_id=data.get("response_message_id"),
                    matched=bool(data.get("matched")),
                    thread_headers=_parse_thread_headers(data.get("thread_headers")),
                )
            )
        return rows


def lookup_dispatch_row(*, workflow_id: str, unique_id: str) -> Optional[WorkflowDispatchRow]:
    if not workflow_id or not unique_id:
        return None

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "SELECT workflow_id, unique_id, supplier_id, supplier_email, message_id, subject, "
            "dispatched_at, responded_at, response_message_id, matched, thread_headers "
            "FROM proc.workflow_email_tracking "
            "WHERE workflow_id=%s AND unique_id=%s "
            "ORDER BY dispatched_at DESC NULLS LAST "
            "LIMIT 1"
        )
        cur.execute(q, (workflow_id, unique_id))
        record = cur.fetchone()
        cur.close()

    if not record:
        return None

    cols = (
        "workflow_id",
        "unique_id",
        "supplier_id",
        "supplier_email",
        "message_id",
        "subject",
        "dispatched_at",
        "responded_at",
        "response_message_id",
        "matched",
        "thread_headers",
    )
    data = dict(zip(cols, record))
    return WorkflowDispatchRow(
        workflow_id=data["workflow_id"],
        unique_id=data["unique_id"],
        supplier_id=data.get("supplier_id"),
        supplier_email=data.get("supplier_email"),
        message_id=data.get("message_id"),
        subject=data.get("subject"),
        dispatched_at=_normalise_dt(data.get("dispatched_at")),
        responded_at=_normalise_dt(data.get("responded_at")) if data.get("responded_at") else None,
        response_message_id=data.get("response_message_id"),
        matched=bool(data.get("matched")),
        thread_headers=_parse_thread_headers(data.get("thread_headers")),
    )


def load_workflow_unique_ids(*, workflow_id: str) -> List[str]:
    """Return unique identifiers associated with the workflow's dispatches."""

    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT unique_id FROM proc.workflow_email_tracking WHERE workflow_id=%s",
            (workflow_id,),
        )
        rows = [row[0] for row in cur.fetchall() if row and row[0]]
        cur.close()
        return rows


def mark_response(
    *,
    workflow_id: str,
    unique_id: str,
    responded_at: datetime,
    response_message_id: Optional[str],
) -> None:
    responded = _normalise_dt(responded_at)

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "UPDATE proc.workflow_email_tracking SET responded_at=%s, response_message_id=%s, matched=TRUE "
            "WHERE workflow_id=%s AND unique_id=%s"
        )
        cur.execute(q, (responded, response_message_id, workflow_id, unique_id))
        cur.close()


def reset_workflow(*, workflow_id: str) -> None:
    """Utility helper for tests to remove workflow rows."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.workflow_email_tracking WHERE workflow_id=%s", (workflow_id,))
        cur.close()


def lookup_workflow_for_unique(*, unique_id: str) -> Optional[str]:
    if not unique_id:
        return None

    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        q = (
            "SELECT workflow_id "
            "FROM proc.workflow_email_tracking "
            "WHERE unique_id=%s "
            "ORDER BY dispatched_at DESC NULLS LAST "
            "LIMIT 1"
        )
        cur.execute(q, (unique_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        workflow_id = row[0]
        return workflow_id or None
