"""Helpers for persisting supplier responses captured by EmailWatcherV2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import sqlite3

from services.db import get_conn


DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.supplier_response (
    id BIGSERIAL PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    response_message_id TEXT,
    response_subject TEXT,
    response_body TEXT,
    response_from TEXT,
    response_date TIMESTAMPTZ,
    original_message_id TEXT,
    original_subject TEXT,
    match_confidence NUMERIC(3,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    UNIQUE (workflow_id, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_supplier_response_wf
ON proc.supplier_response (workflow_id);
"""

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS supplier_response (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    response_message_id TEXT,
    response_subject TEXT,
    response_body TEXT,
    response_from TEXT,
    response_date TEXT,
    original_message_id TEXT,
    original_subject TEXT,
    match_confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    processed INTEGER DEFAULT 0,
    UNIQUE (workflow_id, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_supplier_response_wf
ON supplier_response (workflow_id);
"""


@dataclass
class SupplierResponseRow:
    workflow_id: str
    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    response_message_id: Optional[str]
    response_subject: Optional[str]
    response_body: str
    response_from: Optional[str]
    response_date: Optional[datetime]
    original_message_id: Optional[str]
    original_subject: Optional[str]
    match_confidence: float
    processed: bool = False


def _normalise_dt(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raise TypeError("expected datetime for response_date")


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cur.executescript(DDL_SQLITE)
        else:
            cur.execute(DDL_PG)
        cur.close()


def insert_response(row: SupplierResponseRow) -> None:
    body = row.response_body or ""
    params_common = (
        row.workflow_id,
        row.unique_id,
        row.supplier_id,
        row.supplier_email,
        row.response_message_id,
        row.response_subject,
        body,
        row.response_from,
        _normalise_dt(row.response_date) if row.response_date else None,
        row.original_message_id,
        row.original_subject,
        float(row.match_confidence),
        1 if row.processed else 0,
    )

    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            q = (
                "INSERT INTO supplier_response "
                "(workflow_id, unique_id, supplier_id, supplier_email, response_message_id, response_subject, "
                "response_body, response_from, response_date, original_message_id, original_subject, match_confidence, processed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(workflow_id, unique_id) DO UPDATE SET "
                "supplier_id=excluded.supplier_id, "
                "supplier_email=excluded.supplier_email, "
                "response_message_id=excluded.response_message_id, "
                "response_subject=excluded.response_subject, "
                "response_body=excluded.response_body, "
                "response_from=excluded.response_from, "
                "response_date=excluded.response_date, "
                "original_message_id=excluded.original_message_id, "
                "original_subject=excluded.original_subject, "
                "match_confidence=excluded.match_confidence, "
                "processed=excluded.processed"
            )
            cur.execute(q, params_common)
            conn.commit()
        else:
            q = (
                "INSERT INTO proc.supplier_response "
                "(workflow_id, unique_id, supplier_id, supplier_email, response_message_id, response_subject, "
                "response_body, response_from, response_date, original_message_id, original_subject, match_confidence, processed) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(workflow_id, unique_id) DO UPDATE SET "
                "supplier_id=EXCLUDED.supplier_id, "
                "supplier_email=EXCLUDED.supplier_email, "
                "response_message_id=EXCLUDED.response_message_id, "
                "response_subject=EXCLUDED.response_subject, "
                "response_body=EXCLUDED.response_body, "
                "response_from=EXCLUDED.response_from, "
                "response_date=EXCLUDED.response_date, "
                "original_message_id=EXCLUDED.original_message_id, "
                "original_subject=EXCLUDED.original_subject, "
                "match_confidence=EXCLUDED.match_confidence, "
                "processed=EXCLUDED.processed"
            )
            cur.execute(q, params_common)
        cur.close()


def mark_processed(*, workflow_id: str, unique_ids: Iterable[str]) -> None:
    ids = [uid for uid in unique_ids if uid]
    if not ids:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            placeholders = ",".join(["?"] * len(ids))
            q = f"UPDATE supplier_response SET processed=1 WHERE workflow_id=? AND unique_id IN ({placeholders})"
            cur.execute(q, [workflow_id, *ids])
            conn.commit()
        else:
            q = (
                "UPDATE proc.supplier_response SET processed=TRUE WHERE workflow_id=%s AND unique_id = ANY(%s)"
            )
            cur.execute(q, (workflow_id, ids))
        cur.close()


def fetch_pending(*, workflow_id: str, include_processed: bool = False) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            base_query = (
                "SELECT workflow_id, unique_id, supplier_id, supplier_email, response_message_id, response_subject, "
                "response_body, response_from, response_date, original_message_id, original_subject, match_confidence "
                "FROM supplier_response WHERE workflow_id=?"
            )
            q = base_query if include_processed else f"{base_query} AND processed=0"
            cur.execute(q, (workflow_id,))
            rows = [dict(zip([c[0] for c in cur.description], rec)) for rec in cur.fetchall()]
            cur.close()
            return rows
        else:
            base_query = (
                "SELECT workflow_id, unique_id, supplier_id, supplier_email, response_message_id, response_subject, "
                "response_body, response_from, response_date, original_message_id, original_subject, match_confidence "
                "FROM proc.supplier_response WHERE workflow_id=%s"
            )
            q = base_query if include_processed else f"{base_query} AND processed=FALSE"
            cur.execute(q, (workflow_id,))
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, rec)) for rec in cur.fetchall()]
            cur.close()
            return rows


def reset_workflow(*, workflow_id: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cur.execute("DELETE FROM supplier_response WHERE workflow_id=?", (workflow_id,))
            conn.commit()
        else:
            cur.execute("DELETE FROM proc.supplier_response WHERE workflow_id=%s", (workflow_id,))
        cur.close()
