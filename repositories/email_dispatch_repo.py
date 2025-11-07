"""Repository utilities for persisting outbound email dispatch metadata."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.email_dispatch (
    workflow_id TEXT NOT NULL,
    supplier_id TEXT,
    unique_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    dispatched_at TIMESTAMPTZ NOT NULL,
    subject TEXT,
    status TEXT NOT NULL DEFAULT 'sent',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (workflow_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_email_dispatch_workflow
    ON proc.email_dispatch (workflow_id);

CREATE INDEX IF NOT EXISTS idx_email_dispatch_unique
    ON proc.email_dispatch (unique_id);
"""


def init_schema() -> None:
    """Ensure the ``proc.email_dispatch`` table exists."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def record_dispatch(
    *,
    workflow_id: str,
    supplier_id: Optional[str],
    unique_id: str,
    round_number: int,
    message_id: str,
    dispatched_at: Optional[datetime],
    subject: Optional[str],
    status: str = "completed",
) -> None:
    """Upsert an outbound dispatch record."""

    if not workflow_id or not unique_id or not message_id:
        logger.error(
            "Cannot record dispatch without workflow_id=%s unique_id=%s message_id=%s",
            workflow_id,
            unique_id,
            message_id,
        )
        raise ValueError("workflow_id, unique_id, and message_id are required")

    dispatched_at = dispatched_at or datetime.now(timezone.utc)
    round_value = max(0, int(round_number))
    status_text = status or "sent"

    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO proc.email_dispatch (
                workflow_id, supplier_id, unique_id, round, message_id,
                dispatched_at, subject, status, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (workflow_id, message_id) DO UPDATE SET
                supplier_id = EXCLUDED.supplier_id,
                unique_id = EXCLUDED.unique_id,
                round = EXCLUDED.round,
                dispatched_at = EXCLUDED.dispatched_at,
                subject = EXCLUDED.subject,
                status = EXCLUDED.status,
                updated_at = NOW()
            """,
            (
                workflow_id,
                supplier_id,
                unique_id,
                round_value,
                message_id,
                dispatched_at,
                subject,
                status_text,
            ),
        )
        cur.close()


def count_completed_supplier_dispatches(workflow_id: str) -> int:
    """Return the number of suppliers with completed dispatches for a workflow."""

    if not workflow_id:
        return 0

    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(DISTINCT supplier_id)
            FROM proc.email_dispatch
            WHERE workflow_id = %s
              AND status = 'completed'
              AND supplier_id IS NOT NULL
            """,
            (workflow_id,),
        )
        row = cur.fetchone()
        cur.close()

    return int(row[0] or 0) if row else 0
