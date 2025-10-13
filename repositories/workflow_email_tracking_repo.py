"""Persistence helpers for workflow-level email dispatch tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workflow_id, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_email_tracking_wf
ON proc.workflow_email_tracking (workflow_id);
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


def _normalise_dt(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raise TypeError("expected datetime for timestamp fields")


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL_PG)
        cur.close()


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
            "dispatched_at, responded_at, response_message_id, matched) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (workflow_id, unique_id) DO UPDATE SET "
            "supplier_id=EXCLUDED.supplier_id, "
            "supplier_email=EXCLUDED.supplier_email, "
            "message_id=EXCLUDED.message_id, "
            "subject=EXCLUDED.subject, "
            "dispatched_at=EXCLUDED.dispatched_at"
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
            "dispatched_at, responded_at, response_message_id, matched "
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
                )
            )
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
