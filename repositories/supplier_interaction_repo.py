from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import sqlite3

from services.db import get_conn

_FAKE_INTERACTIONS: List[Dict[str, Any]] = []


def _is_fake_connection(conn) -> bool:
    return hasattr(conn, "_store")


def _fake_reset() -> None:
    _FAKE_INTERACTIONS.clear()


def _fake_store_row(row: SupplierInteractionRow) -> Dict[str, Any]:
    return {
        "workflow_id": row.workflow_id,
        "unique_id": row.unique_id,
        "supplier_id": row.supplier_id,
        "supplier_email": row.supplier_email,
        "round_number": row.round_number,
        "direction": row.direction,
        "interaction_type": row.interaction_type,
        "status": row.status,
        "subject": row.subject,
        "body": row.body,
        "message_id": row.message_id,
        "in_reply_to": list(row.in_reply_to or []),
        "references": list(row.references or []),
        "rfq_id": row.rfq_id,
        "received_at": row.received_at,
        "processed_at": row.processed_at,
        "metadata": dict(row.metadata or {}),
    }


def _fake_upsert(row: SupplierInteractionRow) -> None:
    data = _fake_store_row(row)
    for existing in _FAKE_INTERACTIONS:
        if (
            existing["unique_id"] == data["unique_id"]
            and existing["direction"] == data["direction"]
        ):
            existing.update(data)
            return
    _FAKE_INTERACTIONS.append(data)


def _fake_mark_status(unique_ids: Iterable[str], direction: str, status: str, processed_at: Optional[datetime]) -> None:
    ids = {uid for uid in unique_ids if uid}
    for row in _FAKE_INTERACTIONS:
        if row["direction"] != direction:
            continue
        if row["unique_id"] in ids:
            row["status"] = status
            row["processed_at"] = processed_at


def _fake_fetch_by_status(
    *,
    workflow_id: str,
    status: str,
    direction: str,
    interaction_type: Optional[str],
    round_number: Optional[int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in _FAKE_INTERACTIONS:
        if row["workflow_id"] != workflow_id or row["direction"] != direction:
            continue
        if row["status"] != status:
            continue
        if interaction_type and row.get("interaction_type") != interaction_type:
            continue
        if round_number is not None and row.get("round_number") != round_number:
            continue
        results.append({
            **row,
            "in_reply_to": list(row.get("in_reply_to") or []),
            "references": list(row.get("references") or []),
            "metadata": dict(row.get("metadata") or {}),
        })
    return results


def _fake_lookup_outbound(unique_id: str) -> Optional[Dict[str, Any]]:
    for row in _FAKE_INTERACTIONS:
        if row["unique_id"] == unique_id and row["direction"] == "outbound":
            return {
                **row,
                "in_reply_to": list(row.get("in_reply_to") or []),
                "references": list(row.get("references") or []),
                "metadata": dict(row.get("metadata") or {}),
            }
    return None


def _fake_find_pending_by_rfq(rfq_id: Optional[str]) -> List[Dict[str, Any]]:
    if not rfq_id:
        return []
    rows: List[Dict[str, Any]] = []
    for row in _FAKE_INTERACTIONS:
        if row["direction"] != "outbound":
            continue
        if row.get("rfq_id") != rfq_id:
            continue
        if row.get("status") not in {"pending", "sent"}:
            continue
        rows.append({
            **row,
            "in_reply_to": list(row.get("in_reply_to") or []),
            "references": list(row.get("references") or []),
            "metadata": dict(row.get("metadata") or {}),
        })
    return rows


DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.supplier_interaction (
    id BIGSERIAL PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    round_number INTEGER NOT NULL,
    direction TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    status TEXT NOT NULL,
    subject TEXT,
    body TEXT,
    message_id TEXT,
    in_reply_to JSONB DEFAULT '[]'::jsonb,
    reference_ids JSONB DEFAULT '[]'::jsonb,
    rfq_id TEXT,
    received_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE (unique_id, direction)
);

CREATE INDEX IF NOT EXISTS idx_supplier_interaction_workflow
ON proc.supplier_interaction (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_interaction_status
ON proc.supplier_interaction (status);

CREATE INDEX IF NOT EXISTS idx_supplier_interaction_rfq
ON proc.supplier_interaction (rfq_id);
"""

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS supplier_interaction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    supplier_email TEXT,
    round_number INTEGER NOT NULL,
    direction TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    status TEXT NOT NULL,
    subject TEXT,
    body TEXT,
    message_id TEXT,
    in_reply_to TEXT DEFAULT '[]',
    reference_ids TEXT DEFAULT '[]',
    rfq_id TEXT,
    received_at TEXT,
    processed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}',
    UNIQUE (unique_id, direction)
);

CREATE INDEX IF NOT EXISTS idx_supplier_interaction_workflow
ON supplier_interaction (workflow_id);

CREATE INDEX IF NOT EXISTS idx_supplier_interaction_status
ON supplier_interaction (status);

CREATE INDEX IF NOT EXISTS idx_supplier_interaction_rfq
ON supplier_interaction (rfq_id);
"""


@dataclass
class SupplierInteractionRow:
    workflow_id: str
    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    round_number: int
    direction: str
    interaction_type: str
    status: str
    subject: Optional[str] = None
    body: Optional[str] = None
    message_id: Optional[str] = None
    in_reply_to: Optional[List[str]] = None
    references: Optional[List[str]] = None
    rfq_id: Optional[str] = None
    received_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


def _normalise_dt(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raise TypeError("expected datetime instance for timestamp fields")


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            cur.close()
            return
        if isinstance(conn, sqlite3.Connection):
            cur.executescript(DDL_SQLITE)
        else:
            cur.execute(DDL_PG)
        cur.close()


def reset() -> None:
    """Remove all interaction records (intended for tests)."""

    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            _fake_reset()
            cur.close()
            return
        if isinstance(conn, sqlite3.Connection):
            cur.execute("DELETE FROM supplier_interaction")
            conn.commit()
        else:
            cur.execute("DELETE FROM proc.supplier_interaction")
        cur.close()


def _json_dump(value: Optional[Iterable[str]]) -> str:
    if not value:
        return "[]"
    return json.dumps([str(item) for item in value if item is not None])


def _json_load(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded if item is not None]


def register_outbound(row: SupplierInteractionRow) -> None:
    params_common = (
        row.workflow_id,
        row.unique_id,
        row.supplier_id,
        row.supplier_email,
        row.round_number,
        row.direction,
        row.interaction_type,
        row.status,
        row.subject,
        row.body,
        row.message_id,
        _json_dump(row.in_reply_to),
        _json_dump(row.references),
        row.rfq_id,
        _normalise_dt(row.received_at),
        _normalise_dt(row.processed_at),
        json.dumps(row.metadata or {}),
    )

    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            _fake_upsert(row)
            cur.close()
            return
        if isinstance(conn, sqlite3.Connection):
            q = (
                "INSERT INTO supplier_interaction (workflow_id, unique_id, supplier_id, supplier_email, "
                "round_number, direction, interaction_type, status, subject, body, message_id, in_reply_to, reference_ids, rfq_id, "
                "received_at, processed_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(unique_id, direction) DO UPDATE SET "
                "supplier_id=excluded.supplier_id, "
                "supplier_email=excluded.supplier_email, "
                "round_number=excluded.round_number, "
                "interaction_type=excluded.interaction_type, "
                "status=excluded.status, "
                "subject=excluded.subject, "
                "body=excluded.body, "
                "message_id=excluded.message_id, "
                "in_reply_to=excluded.in_reply_to, "
                "reference_ids=excluded.reference_ids, "
                "rfq_id=excluded.rfq_id, "
                "received_at=excluded.received_at, "
                "processed_at=excluded.processed_at, "
                "metadata=excluded.metadata"
            )
            cur.execute(q, params_common)
            conn.commit()
        else:
            q = (
                "INSERT INTO proc.supplier_interaction (workflow_id, unique_id, supplier_id, supplier_email, "
                "round_number, direction, interaction_type, status, subject, body, message_id, in_reply_to, reference_ids, rfq_id, "
                "received_at, processed_at, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(unique_id, direction) DO UPDATE SET "
                "supplier_id=EXCLUDED.supplier_id, "
                "supplier_email=EXCLUDED.supplier_email, "
                "round_number=EXCLUDED.round_number, "
                "interaction_type=EXCLUDED.interaction_type, "
                "status=EXCLUDED.status, "
                "subject=EXCLUDED.subject, "
                "body=EXCLUDED.body, "
                "message_id=EXCLUDED.message_id, "
                "in_reply_to=EXCLUDED.in_reply_to, "
                "reference_ids=EXCLUDED.reference_ids, "
                "rfq_id=EXCLUDED.rfq_id, "
                "received_at=EXCLUDED.received_at, "
                "processed_at=EXCLUDED.processed_at, "
                "metadata=EXCLUDED.metadata"
            )
            cur.execute(q, params_common)
        cur.close()


def mark_status(*, unique_ids: Iterable[str], direction: str, status: str, processed_at: Optional[datetime] = None) -> None:
    ids = [uid for uid in unique_ids if uid]
    if not ids:
        return

    processed = _normalise_dt(processed_at) if processed_at else None

    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            _fake_mark_status(ids, direction, status, processed)
            cur.close()
            return
        if isinstance(conn, sqlite3.Connection):
            placeholders = ",".join(["?"] * len(ids))
            q = (
                f"UPDATE supplier_interaction SET status=?, processed_at=?, updated_at=CURRENT_TIMESTAMP "
                f"WHERE direction=? AND unique_id IN ({placeholders})"
            )
            cur.execute(q, [status, processed.isoformat() if processed else None, direction, *ids])
            conn.commit()
        else:
            q = (
                "UPDATE proc.supplier_interaction SET status=%s, processed_at=%s, updated_at=NOW() "
                "WHERE direction=%s AND unique_id = ANY(%s)"
            )
            cur.execute(q, (status, processed, direction, ids))
        cur.close()


def fetch_by_status(
    *,
    workflow_id: str,
    status: str,
    direction: str = "inbound",
    interaction_type: Optional[str] = None,
    round_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            cur.close()
            return _fake_fetch_by_status(
                workflow_id=workflow_id,
                status=status,
                direction=direction,
                interaction_type=interaction_type,
                round_number=round_number,
            )
        params: List[Any] = [workflow_id, direction, status]
        filters: List[str] = []
        if interaction_type:
            filters.append("interaction_type=%s" if not isinstance(conn, sqlite3.Connection) else "interaction_type=?")
            params.append(interaction_type)
        if round_number is not None:
            filters.append("round_number=%s" if not isinstance(conn, sqlite3.Connection) else "round_number=?")
            params.append(round_number)

        if isinstance(conn, sqlite3.Connection):
            base_query = (
                "SELECT workflow_id, unique_id, supplier_id, supplier_email, round_number, direction, interaction_type, status, "
                "subject, body, message_id, in_reply_to, reference_ids, rfq_id, received_at, processed_at, metadata "
                "FROM supplier_interaction WHERE workflow_id=? AND direction=? AND status=?"
            )
            if filters:
                base_query += " AND " + " AND ".join(filters)
            cur.execute(base_query, params)
            rows = [dict(zip([c[0] for c in cur.description], rec)) for rec in cur.fetchall()]
            for row in rows:
                row["in_reply_to"] = _json_load(row.get("in_reply_to"))
                row["references"] = _json_load(row.get("reference_ids"))
                row["metadata"] = json.loads(row.get("metadata") or "{}")
            cur.close()
            return rows

        base_query = (
            "SELECT workflow_id, unique_id, supplier_id, supplier_email, round_number, direction, interaction_type, status, "
            "subject, body, message_id, in_reply_to::text, reference_ids::text, rfq_id, received_at, processed_at, metadata::text "
            "FROM proc.supplier_interaction WHERE workflow_id=%s AND direction=%s AND status=%s"
        )
        if filters:
            base_query += " AND " + " AND ".join(filters)
        cur.execute(base_query, params)
        cols = [desc[0] for desc in cur.description]
        rows = []
        for rec in cur.fetchall():
            data = dict(zip(cols, rec))
            data["in_reply_to"] = _json_load(data.get("in_reply_to"))
            data["references"] = _json_load(data.get("reference_ids"))
            data["metadata"] = json.loads(data.get("metadata") or "{}")
            rows.append(data)
        cur.close()
        return rows


def lookup_outbound(unique_id: str) -> Optional[Dict[str, Any]]:
    if not unique_id:
        return None

    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            cur.close()
            return _fake_lookup_outbound(unique_id)
        if isinstance(conn, sqlite3.Connection):
            q = (
                "SELECT workflow_id, unique_id, supplier_id, supplier_email, round_number, interaction_type, status, subject, body, "
                "message_id, in_reply_to, reference_ids, rfq_id, metadata FROM supplier_interaction "
                "WHERE unique_id=? AND direction='outbound'"
            )
            cur.execute(q, (unique_id,))
            rec = cur.fetchone()
            if rec is None:
                cur.close()
                return None
            data = dict(zip([c[0] for c in cur.description], rec))
            data["in_reply_to"] = _json_load(data.get("in_reply_to"))
            data["references"] = _json_load(data.get("reference_ids"))
            data["metadata"] = json.loads(data.get("metadata") or "{}")
            cur.close()
            return data

        q = (
            "SELECT workflow_id, unique_id, supplier_id, supplier_email, round_number, interaction_type, status, subject, body, "
            "message_id, in_reply_to::text, reference_ids::text, rfq_id, metadata::text FROM proc.supplier_interaction "
            "WHERE unique_id=%s AND direction='outbound'"
        )
        cur.execute(q, (unique_id,))
        rec = cur.fetchone()
        if rec is None:
            cur.close()
            return None
        cols = [d[0] for d in cur.description]
        data = dict(zip(cols, rec))
        data["in_reply_to"] = _json_load(data.get("in_reply_to"))
        data["references"] = _json_load(data.get("reference_ids"))
        data["metadata"] = json.loads(data.get("metadata") or "{}")
        cur.close()
        return data


def record_inbound_response(
    *,
    outbound: SupplierInteractionRow,
    message_id: Optional[str],
    subject: Optional[str],
    body: str,
    from_address: Optional[str],
    received_at: datetime,
    in_reply_to: Iterable[str],
    references: Iterable[str],
    rfq_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    inbound_row = SupplierInteractionRow(
        workflow_id=outbound.workflow_id,
        unique_id=outbound.unique_id,
        supplier_id=outbound.supplier_id,
        supplier_email=from_address or outbound.supplier_email,
        round_number=outbound.round_number,
        direction="inbound",
        interaction_type=outbound.interaction_type,
        status="received",
        subject=subject,
        body=body,
        message_id=message_id,
        in_reply_to=list(in_reply_to),
        references=list(references),
        rfq_id=rfq_id or outbound.rfq_id,
        received_at=_normalise_dt(received_at),
        metadata=metadata,
    )
    register_outbound(inbound_row)
    mark_status(unique_ids=[outbound.unique_id], direction="outbound", status="received")


def find_pending_by_rfq(rfq_id: Optional[str]) -> List[Dict[str, Any]]:
    if not rfq_id:
        return []

    with get_conn() as conn:
        cur = conn.cursor()
        if _is_fake_connection(conn):
            cur.close()
            return _fake_find_pending_by_rfq(rfq_id)
        if isinstance(conn, sqlite3.Connection):
            q = (
                "SELECT workflow_id, unique_id, supplier_id, supplier_email, round_number, interaction_type, status, subject, body, "
                "message_id, in_reply_to, reference_ids, metadata FROM supplier_interaction "
                "WHERE rfq_id=? AND direction='outbound' AND status IN ('pending', 'sent')"
            )
            cur.execute(q, (rfq_id,))
            rows = [dict(zip([c[0] for c in cur.description], rec)) for rec in cur.fetchall()]
            for row in rows:
                row["in_reply_to"] = _json_load(row.get("in_reply_to"))
                row["references"] = _json_load(row.get("reference_ids"))
                row["metadata"] = json.loads(row.get("metadata") or "{}")
            cur.close()
            return rows

        q = (
            "SELECT workflow_id, unique_id, supplier_id, supplier_email, round_number, interaction_type, status, subject, body, "
            "message_id, in_reply_to::text, reference_ids::text, metadata::text FROM proc.supplier_interaction "
            "WHERE rfq_id=%s AND direction='outbound' AND status IN ('pending', 'sent')"
        )
        cur.execute(q, (rfq_id,))
        cols = [d[0] for d in cur.description]
        rows = []
        for rec in cur.fetchall():
            data = dict(zip(cols, rec))
            data["in_reply_to"] = _json_load(data.get("in_reply_to"))
            data["references"] = _json_load(data.get("reference_ids"))
            data["metadata"] = json.loads(data.get("metadata") or "{}")
            rows.append(data)
        cur.close()
        return rows
