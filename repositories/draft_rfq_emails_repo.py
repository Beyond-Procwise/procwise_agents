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
    rfq_id TEXT,
    supplier_id TEXT,
    supplier_name TEXT,
    subject TEXT,
    body TEXT,
    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent BOOLEAN NOT NULL DEFAULT FALSE,
    review_status TEXT NOT NULL DEFAULT 'PENDING',
    sent_on TIMESTAMPTZ,
    recipient_email TEXT,
    contact_level INTEGER NOT NULL DEFAULT 0,
    thread_index INTEGER NOT NULL DEFAULT 1,
    sender TEXT,
    sender_email TEXT,
    payload JSONB,
    updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    workflow_id TEXT NOT NULL,
    run_id TEXT,
    unique_id TEXT NOT NULL,
    mailbox TEXT,
    dispatched_at TIMESTAMPTZ,
    dispatch_run_id TEXT
);

ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS rfq_id TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS supplier_name TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS subject TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS body TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS created_on TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS sent BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'PENDING';
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS sent_on TIMESTAMPTZ;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS recipient_email TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS contact_level INTEGER NOT NULL DEFAULT 0;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS thread_index INTEGER NOT NULL DEFAULT 1;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS sender TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS sender_email TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS payload JSONB;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS workflow_id TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS unique_id TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS mailbox TEXT;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS dispatched_at TIMESTAMPTZ;
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS dispatch_run_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_rfq_emails_wf_uid
ON proc.draft_rfq_emails (workflow_id, unique_id);

CREATE INDEX IF NOT EXISTS idx_draft_rfq_emails_wf
ON proc.draft_rfq_emails (workflow_id);
"""

logger = logging.getLogger(__name__)

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS draft_rfq_emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rfq_id TEXT,
    supplier_id TEXT,
    supplier_name TEXT,
    subject TEXT,
    body TEXT,
    created_on TEXT NOT NULL DEFAULT (datetime('now')),
    sent INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'PENDING',
    sent_on TEXT,
    recipient_email TEXT,
    contact_level INTEGER NOT NULL DEFAULT 0,
    thread_index INTEGER NOT NULL DEFAULT 1,
    sender TEXT,
    sender_email TEXT,
    payload TEXT,
    updated_on TEXT NOT NULL DEFAULT (datetime('now')),
    workflow_id TEXT NOT NULL,
    run_id TEXT,
    unique_id TEXT NOT NULL,
    mailbox TEXT,
    dispatched_at TEXT,
    dispatch_run_id TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_rfq_emails_wf_uid
ON draft_rfq_emails (workflow_id, unique_id);

CREATE INDEX IF NOT EXISTS idx_draft_rfq_emails_wf
ON draft_rfq_emails (workflow_id);
"""

SQLITE_COLUMN_DEFINITIONS: Dict[str, str] = {
    "rfq_id": "TEXT",
    "supplier_id": "TEXT",
    "supplier_name": "TEXT",
    "subject": "TEXT",
    "body": "TEXT",
    "created_on": "TEXT NOT NULL DEFAULT (datetime('now'))",
    "sent": "INTEGER NOT NULL DEFAULT 0",
    "review_status": "TEXT NOT NULL DEFAULT 'PENDING'",
    "sent_on": "TEXT",
    "recipient_email": "TEXT",
    "contact_level": "INTEGER NOT NULL DEFAULT 0",
    "thread_index": "INTEGER NOT NULL DEFAULT 1",
    "sender": "TEXT",
    "sender_email": "TEXT",
    "payload": "TEXT",
    "updated_on": "TEXT NOT NULL DEFAULT (datetime('now'))",
    "workflow_id": "TEXT NOT NULL",
    "run_id": "TEXT",
    "unique_id": "TEXT NOT NULL",
    "mailbox": "TEXT",
    "dispatched_at": "TEXT",
    "dispatch_run_id": "TEXT",
}


def _ensure_sqlite_columns(cur: sqlite3.Cursor, table: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall() if row and len(row) > 1}
    for column, definition in SQLITE_COLUMN_DEFINITIONS.items():
        if column in existing:
            continue
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cur.executescript(DDL_SQLITE)
            _ensure_sqlite_columns(cur, "draft_rfq_emails")
        else:
            for statement in filter(None, (stmt.strip() for stmt in DDL_PG.split(";"))):
                cur.execute(statement)
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

def expected_unique_ids_and_last_dispatch(*, workflow_id: str, run_id: Optional[str] = None) -> Tuple[Set[str], Dict[str, str], Optional[datetime]]:
    """
    Return:
      - set of unique_ids expected for this workflow (optionally filtered by run_id),
      - mapping unique_id -> supplier_id (may be None),
      - last dispatch time (max dispatched_at)
    """
    with get_conn() as conn:
        if isinstance(conn, sqlite3.Connection):
            q = "SELECT unique_id, supplier_id, dispatched_at FROM draft_rfq_emails WHERE workflow_id=?"
            params = [workflow_id]
            if run_id is not None:
                q += " AND run_id=?"
                params.append(run_id)
            cur = conn.cursor()
            cur.execute(q, params)
            rows = cur.fetchall()
            cur.close()
        else:
            q = "SELECT unique_id, supplier_id, dispatched_at FROM proc.draft_rfq_emails WHERE workflow_id=%s"
            params = [workflow_id]
            if run_id is not None:
                q += " AND run_id=%s"
                params.append(run_id)
            cur = conn.cursor()
            cur.execute(q, params)
            rows = cur.fetchall()
            cur.close()

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


def load_by_unique_id(unique_id: str) -> Optional[Dict[str, Any]]:
    """Load the most recent draft record for the given unique_id."""

    if not unique_id:
        return None

    columns = (
        "id",
        "rfq_id",
        "supplier_id",
        "supplier_name",
        "subject",
        "body",
        "created_on",
        "sent",
        "review_status",
        "sender",
        "sender_email",
        "payload",
        "workflow_id",
        "run_id",
        "unique_id",
        "mailbox",
        "thread_index",
        "dispatched_at",
        "recipient_email",
        "contact_level",
        "sent_on",
        "updated_on",
        "dispatch_run_id",
    )

    try:
        with get_conn() as conn:
            if isinstance(conn, sqlite3.Connection):
                query = (
                    f"SELECT {', '.join(columns)} FROM draft_rfq_emails "
                    "WHERE unique_id = ? ORDER BY dispatched_at DESC, created_on DESC LIMIT 1"
                )
                params = (unique_id,)
            else:
                query = (
                    f"SELECT {', '.join(columns)} FROM proc.draft_rfq_emails "
                    "WHERE unique_id = %s ORDER BY dispatched_at DESC NULLS LAST, created_on DESC LIMIT 1"
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
        "review_status": row[8],
        "sender": row[9],
        "sender_email": row[10],
        "payload": row[11],
        "workflow_id": row[12],
        "run_id": row[13],
        "unique_id": row[14],
        "mailbox": row[15],
        "thread_index": row[16],
        "dispatched_at": row[17],
        "recipient_email": row[18],
        "contact_level": row[19],
        "sent_on": row[20],
        "updated_on": row[21],
        "dispatch_run_id": row[22],
    }
