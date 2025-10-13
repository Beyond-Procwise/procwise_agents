from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

from services.db import USING_SQLITE, get_conn

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


def _assert_postgres_backend() -> None:
    if USING_SQLITE:
        raise RuntimeError(
            "Supplier responses repository requires a PostgreSQL connection."
        )


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


def init_schema() -> None:
    with get_conn() as conn:
        _assert_postgres_backend()
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL_PG.split(";"))):
            cur.execute(statement)
        _ensure_postgres_column(cur, "proc", "supplier_response", "response_text", "TEXT")
        cur.close()


def insert_response(row: SupplierResponseRow) -> None:
    response_text = row.response_text or ""
    received_time = _normalise_dt(row.received_time)
    price_value = _serialise_decimal(row.price)

    with get_conn() as conn:
        _assert_postgres_backend()
        cur = conn.cursor()
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
        _assert_postgres_backend()
        cur = conn.cursor()
        q = "DELETE FROM proc.supplier_response WHERE workflow_id=%s AND unique_id = ANY(%s)"
        cur.execute(q, (workflow_id, ids))
        cur.close()


def fetch_pending(*, workflow_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        _assert_postgres_backend()
        cur = conn.cursor()
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
        _assert_postgres_backend()
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.supplier_response WHERE workflow_id=%s", (workflow_id,))
        cur.close()
