from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime, timezone
import json

from services.db import get_conn, USING_SQLITE

DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.supplier_responses (
    id BIGSERIAL PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    run_id TEXT,
    unique_id TEXT,           -- primary matcher
    supplier_id TEXT,         -- secondary matcher
    message_id TEXT,
    mailbox TEXT,
    imap_uid TEXT,
    from_addr TEXT,
    to_addrs TEXT,
    subject TEXT,
    body_text TEXT,
    headers_json JSONB,
    received_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_status TEXT DEFAULT 'pending',
    processed_at TIMESTAMPTZ,
    extra JSONB
);

-- one row per (workflow, unique_id); replies without unique_id fall back to (workflow, supplier_id, message_id)
CREATE UNIQUE INDEX IF NOT EXISTS uq_supplier_responses_wf_uid
ON proc.supplier_responses (workflow_id, unique_id);

CREATE INDEX IF NOT EXISTS idx_supplier_responses_wf
ON proc.supplier_responses (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_responses_supplier
ON proc.supplier_responses (supplier_id);
"""

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS supplier_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    run_id TEXT,
    unique_id TEXT,
    supplier_id TEXT,
    message_id TEXT,
    mailbox TEXT,
    imap_uid TEXT,
    from_addr TEXT,
    to_addrs TEXT,
    subject TEXT,
    body_text TEXT,
    headers_json TEXT,
    received_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    processed_status TEXT DEFAULT 'pending',
    processed_at TEXT,
    extra TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_supplier_responses_wf_uid
ON supplier_responses (workflow_id, unique_id);

CREATE INDEX IF NOT EXISTS idx_supplier_responses_wf
ON supplier_responses (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_responses_supplier
ON supplier_responses (supplier_id);
"""

def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if USING_SQLITE:
            cur.executescript(DDL_SQLITE)
        else:
            cur.execute(DDL_PG)
        cur.close()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def upsert_response(
    *,
    workflow_id: str,
    run_id: Optional[str],
    unique_id: Optional[str],
    supplier_id: Optional[str],
    message_id: Optional[str],
    mailbox: Optional[str],
    imap_uid: Optional[str],
    from_addr: Optional[str],
    to_addrs: Iterable[str],
    subject: Optional[str],
    body_text: str,
    headers: Dict[str, Any],
    received_at: datetime,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    headers_serial = json.dumps(headers or {}, ensure_ascii=False)
    extra_serial = json.dumps(extra or {}, ensure_ascii=False)
    to_serial = ",".join([t for t in (to_addrs or [])])

    with get_conn() as conn:
        if USING_SQLITE:
            q = """
INSERT INTO supplier_responses
(workflow_id, run_id, unique_id, supplier_id, message_id, mailbox, imap_uid, from_addr, to_addrs, subject,
 body_text, headers_json, received_at, created_at, processed_status, extra)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
ON CONFLICT(workflow_id, unique_id) DO UPDATE SET
  supplier_id=COALESCE(excluded.supplier_id, supplier_responses.supplier_id),
  message_id=COALESCE(excluded.message_id, supplier_responses.message_id),
  mailbox=excluded.mailbox,
  imap_uid=excluded.imap_uid,
  from_addr=excluded.from_addr,
  to_addrs=excluded.to_addrs,
  subject=excluded.subject,
  body_text=excluded.body_text,
  headers_json=excluded.headers_json,
  received_at=excluded.received_at,
  processed_status='pending',
  processed_at=NULL
"""
            cur = conn.cursor()
            cur.execute(q, (
                workflow_id, run_id, unique_id, supplier_id, message_id, mailbox, imap_uid, from_addr, to_serial,
                subject, body_text, headers_serial, received_at.isoformat(), _now_iso(), extra_serial
            ))
            conn.commit()
            cur.close()
        else:
            q = """
INSERT INTO proc.supplier_responses
(workflow_id, run_id, unique_id, supplier_id, message_id, mailbox, imap_uid, from_addr, to_addrs, subject,
 body_text, headers_json, received_at, processed_status, extra)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, 'pending', %s::jsonb)
ON CONFLICT (workflow_id, unique_id) DO UPDATE SET
  supplier_id=COALESCE(EXCLUDED.supplier_id, proc.supplier_responses.supplier_id),
  message_id=COALESCE(EXCLUDED.message_id, proc.supplier_responses.message_id),
  mailbox=EXCLUDED.mailbox,
  imap_uid=EXCLUDED.imap_uid,
  from_addr=EXCLUDED.from_addr,
  to_addrs=EXCLUDED.to_addrs,
  subject=EXCLUDED.subject,
  body_text=EXCLUDED.body_text,
  headers_json=EXCLUDED.headers_json,
  received_at=EXCLUDED.received_at,
  processed_status='pending',
  processed_at=NULL
"""
            cur = conn.cursor()
            cur.execute(q, (
                workflow_id, run_id, unique_id, supplier_id, message_id, mailbox, imap_uid, from_addr, to_serial,
                subject, body_text, headers_serial, received_at, extra_serial
            ))
            cur.close()

def fetch_latest_for_workflow(*, workflow_id: str) -> List[Dict[str, Any]]:
    """
    Return all rows for this workflow with latest received_at, dedup by unique_id
    (since unique constraint enforces one per unique_id, this returns all current matches).
    """
    with get_conn() as conn:
        if USING_SQLITE:
            q = """
SELECT *
FROM supplier_responses
WHERE workflow_id=? AND processed_status='pending'
ORDER BY received_at ASC
"""
            cur = conn.cursor()
            cur.execute(q, (workflow_id,))
            rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
            cur.close()
            return rows
        else:
            q = """
SELECT *
FROM proc.supplier_responses
WHERE workflow_id=%s AND processed_status='pending'
ORDER BY received_at ASC
"""
            cur = conn.cursor()
            cur.execute(q, (workflow_id,))
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.close()
            return rows

def mark_processed_by_ids(ids: Iterable[int]) -> None:
    ids = list(ids)
    if not ids:
        return
    with get_conn() as conn:
        if USING_SQLITE:
            q = "UPDATE supplier_responses SET processed_status='processed', processed_at=? WHERE id IN ({})".format(
                ",".join(["?"] * len(ids))
            )
            cur = conn.cursor()
            cur.execute(q, [datetime.now(timezone.utc).isoformat()] + ids)
            conn.commit()
            cur.close()
        else:
            q = "UPDATE proc.supplier_responses SET processed_status='processed', processed_at=NOW() WHERE id = ANY(%s)"
            cur = conn.cursor()
            cur.execute(q, (ids,))
            cur.close()
