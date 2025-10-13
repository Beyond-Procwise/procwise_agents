from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

import sqlite3

from services.db import get_conn

DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.supplier_response (
    workflow_id TEXT NOT NULL,
    supplier_id TEXT,
    unique_id TEXT NOT NULL,
    response_text TEXT,
    price NUMERIC(18, 4),
    lead_time INTEGER,
    received_time TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (workflow_id, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_supplier_response_wf
ON proc.supplier_response (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_response_supplier
ON proc.supplier_response (supplier_id);
"""

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS supplier_response (
    workflow_id TEXT NOT NULL,
    supplier_id TEXT,
    unique_id TEXT NOT NULL,
    response_text TEXT,
    price REAL,
    lead_time INTEGER,
    received_time TEXT NOT NULL,
    PRIMARY KEY (workflow_id, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_supplier_response_wf
ON supplier_response (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_response_supplier
ON supplier_response (supplier_id);
"""


@dataclass
class SupplierResponseRow:
    workflow_id: str
    unique_id: str
    supplier_id: Optional[str]
    response_text: str
    received_time: datetime
    price: Optional[Decimal] = None
    lead_time: Optional[int] = None


def _normalise_dt(value: datetime) -> datetime:
    if value.tzinfo:
        return value
    return value.replace(tzinfo=timezone.utc)


def _serialise_decimal(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            cur.executescript(DDL_SQLITE)
        else:
            cur.execute(DDL_PG)
        cur.close()


def insert_response(row: SupplierResponseRow) -> None:
    response_text = row.response_text or ""
    received_time = _normalise_dt(row.received_time)
    price_value = _serialise_decimal(row.price)

    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            q = (
                "INSERT INTO supplier_response "
                "(workflow_id, supplier_id, unique_id, response_text, price, lead_time, received_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(workflow_id, unique_id) DO UPDATE SET "
                "supplier_id=excluded.supplier_id, "
                "response_text=excluded.response_text, "
                "price=excluded.price, "
                "lead_time=excluded.lead_time, "
                "received_time=excluded.received_time"
            )
            cur.execute(
                q,
                (
                    row.workflow_id,
                    row.supplier_id,
                    row.unique_id,
                    response_text,
                    price_value,
                    row.lead_time,
                    received_time.isoformat(),
                ),
            )
            conn.commit()
        else:
            q = (
                "INSERT INTO proc.supplier_response "
                "(workflow_id, supplier_id, unique_id, response_text, price, lead_time, received_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(workflow_id, unique_id) DO UPDATE SET "
                "supplier_id=EXCLUDED.supplier_id, "
                "response_text=EXCLUDED.response_text, "
                "price=EXCLUDED.price, "
                "lead_time=EXCLUDED.lead_time, "
                "received_time=EXCLUDED.received_time"
            )
            cur.execute(
                q,
                (
                    row.workflow_id,
                    row.supplier_id,
                    row.unique_id,
                    response_text,
                    price_value,
                    row.lead_time,
                    received_time,
                ),
            )
        cur.close()


def delete_responses(*, workflow_id: str, unique_ids: Iterable[str]) -> None:
    ids = [uid for uid in unique_ids if uid]
    if not ids:
        return

    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            placeholders = ",".join(["?"] * len(ids))
            q = f"DELETE FROM supplier_response WHERE workflow_id=? AND unique_id IN ({placeholders})"
            cur.execute(q, [workflow_id, *ids])
            conn.commit()
        else:
            q = "DELETE FROM proc.supplier_response WHERE workflow_id=%s AND unique_id = ANY(%s)"
            cur.execute(q, (workflow_id, ids))
        cur.close()


def fetch_pending(*, workflow_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        if isinstance(conn, sqlite3.Connection):
            q = (
                "SELECT workflow_id, supplier_id, unique_id, response_text, price, lead_time, received_time "
                "FROM supplier_response WHERE workflow_id=?"
            )
            cur.execute(q, (workflow_id,))
            rows = [dict(zip([c[0] for c in cur.description], rec)) for rec in cur.fetchall()]
            cur.close()
            return rows
        else:
            q = (
                "SELECT workflow_id, supplier_id, unique_id, response_text, price, lead_time, received_time "
                "FROM proc.supplier_response WHERE workflow_id=%s"
            )
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
