from __future__ import annotations
from datetime import datetime, timezone
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from services.db import get_conn

# DDL is defensive: if tables don't exist (dev), create minimalist schemas
DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.draft_rfq_emails (
    id BIGSERIAL PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    run_id TEXT,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- status fields (optional)
    sent BOOLEAN DEFAULT TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_rfq_emails_wf_uid
ON proc.draft_rfq_emails (workflow_id, unique_id);

CREATE INDEX IF NOT EXISTS idx_draft_rfq_emails_wf
ON proc.draft_rfq_emails (workflow_id);
"""

logger = logging.getLogger(__name__)

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS draft_rfq_emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    run_id TEXT,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    dispatched_at TEXT NOT NULL,
    sent INTEGER DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_rfq_emails_wf_uid
ON draft_rfq_emails (workflow_id, unique_id);

CREATE INDEX IF NOT EXISTS idx_draft_rfq_emails_wf
ON draft_rfq_emails (workflow_id);
"""

def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cur.executescript(DDL_SQLITE)
        else:
            cur.execute(DDL_PG)
        cur.close()

def _parse_dt(dt) -> datetime:
    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    # assume ISO string
    try:
        d = datetime.fromisoformat(dt)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def _expected_unique_id_rows(
    *, workflow_id: str, run_id: Optional[str] = None
) -> List[Tuple[str, Optional[str], datetime, Optional[bool]]]:
    with get_conn() as conn:
        if isinstance(conn, sqlite3.Connection):
            q = (
                "SELECT unique_id, supplier_id, dispatched_at, sent "
                "FROM draft_rfq_emails WHERE workflow_id=?"
            )
            params = [workflow_id]
            if run_id is not None:
                q += " AND run_id=?"
                params.append(run_id)
            cur = conn.cursor()
            cur.execute(q, params)
            rows = cur.fetchall()
            cur.close()
        else:
            q = (
                "SELECT unique_id, supplier_id, dispatched_at, sent "
                "FROM proc.draft_rfq_emails WHERE workflow_id=%s"
            )
            params = [workflow_id]
            if run_id is not None:
                q += " AND run_id=%s"
                params.append(run_id)
            cur = conn.cursor()
            cur.execute(q, params)
            rows = cur.fetchall()
            cur.close()

    return rows


def expected_unique_ids_and_last_dispatch(
    *, workflow_id: str, run_id: Optional[str] = None
) -> Tuple[Set[str], Dict[str, str], Optional[datetime]]:
    """
    Return:
      - set of unique_ids expected for this workflow (optionally filtered by run_id),
      - mapping unique_id -> supplier_id (may be None),
      - last dispatch time (max dispatched_at)
    """
    rows = _expected_unique_id_rows(workflow_id=workflow_id, run_id=run_id)

    ids: Set[str] = set()
    map_uid_to_supplier: Dict[str, str] = {}
    last_dt: Optional[datetime] = None

    for row in rows:
        uid = row[0]
        sid = row[1]
        dt = _parse_dt(row[2])
        if uid:
            ids.add(uid)
            if sid:
                map_uid_to_supplier[uid] = sid
        if last_dt is None or dt > last_dt:
            last_dt = dt
    return ids, map_uid_to_supplier, last_dt


def expected_unique_ids_and_last_dispatch_with_pending(
    *, workflow_id: str, run_id: Optional[str] = None
) -> Tuple[Set[str], Dict[str, str], Optional[datetime], Set[str]]:
    """
    Extended variant that also reports unsent (pending) unique identifiers.
    """

    rows = _expected_unique_id_rows(workflow_id=workflow_id, run_id=run_id)

    ids: Set[str] = set()
    map_uid_to_supplier: Dict[str, str] = {}
    last_dt: Optional[datetime] = None
    pending: Set[str] = set()

    for row in rows:
        uid = row[0]
        sid = row[1]
        dt = _parse_dt(row[2])
        sent_flag = row[3] if len(row) > 3 else True
        if uid:
            ids.add(uid)
            if sid:
                map_uid_to_supplier[uid] = sid
            if not sent_flag:
                pending.add(uid)
        if last_dt is None or dt > last_dt:
            last_dt = dt

    return ids, map_uid_to_supplier, last_dt, pending


def load_by_unique_id(unique_id: str) -> Optional[Dict[str, Any]]:
    """Load the most recent draft record for the given unique_id."""

    if not unique_id:
        return None

    try:
        with get_conn() as conn:
            if isinstance(conn, sqlite3.Connection):
                query = (
                    "SELECT id, NULL, supplier_id, NULL, NULL, NULL, dispatched_at, sent, NULL, NULL, "
                    "workflow_id, run_id, unique_id, NULL FROM draft_rfq_emails "
                    "WHERE unique_id = ? ORDER BY dispatched_at DESC LIMIT 1"
                )
                params = (unique_id,)
            else:
                query = (
                    "SELECT id, rfq_id, supplier_id, supplier_name, subject, body, created_on, sent, sender, payload, "
                    "workflow_id, run_id, unique_id, mailbox FROM proc.draft_rfq_emails "
                    "WHERE unique_id = %s ORDER BY created_on DESC LIMIT 1"
                )
                params = (unique_id,)

            cur = conn.cursor()
            cur.execute(query, params)
            row = cur.fetchone()
            cur.close()
    except Exception:
        logger.exception("Failed to load draft by unique_id=%s", unique_id)
        return None

    if not row:
        return None

    return {
        "id": row[0],
        "rfq_id": row[1],
        "supplier_id": row[2],
        "supplier_name": row[3],
        "subject": row[4],
        "body": row[5],
        "created_on": row[6],
        "sent": row[7],
        "sender": row[8],
        "payload": row[9],
        "workflow_id": row[10],
        "run_id": row[11],
        "unique_id": row[12],
        "mailbox": row[13],
    }
