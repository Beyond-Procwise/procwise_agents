"""Persistence helpers for workflow-level email dispatch tracking."""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set

from services.db import get_conn
from repositories import draft_rfq_emails_repo


DDL_PG = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.workflow_email_tracking (
    workflow_id TEXT NOT NULL,
    unique_id TEXT NOT NULL,
    dispatch_key TEXT,
    supplier_id TEXT,
    supplier_email TEXT,
    recipient_emails JSONB,
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
    dispatched_at: datetime
    supplier_id: Optional[str] = None
    supplier_email: Optional[str] = None
    recipient_emails: Optional[Sequence[str]] = None
    message_id: Optional[str] = None
    subject: Optional[str] = None
    round_number: Optional[int] = None
    dispatch_key: Optional[str] = None
    responded_at: Optional[datetime] = None
    response_message_id: Optional[str] = None
    matched: bool = False
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


def _ensure_primary_key(cur) -> None:
    """Ensure the table primary key matches (workflow_id, unique_id)."""

    cur.execute("SELECT to_regclass('proc.workflow_email_tracking')")
    table_name = cur.fetchone()[0]
    if not table_name:
        return

    cur.execute(
        """
        SELECT
            tc.constraint_name,
            kcu.column_name,
            kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = 'proc'
          AND tc.table_name = 'workflow_email_tracking'
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
        """
    )
    rows = cur.fetchall()

    if not rows:
        cur.execute(
            """
            ALTER TABLE proc.workflow_email_tracking
                ADD PRIMARY KEY (workflow_id, unique_id)
            """
        )
        return

    constraint_name = rows[0][0]
    columns = [row[1] for row in rows]
    expected = ["workflow_id", "unique_id"]

    if columns == expected:
        return

    safe_name = constraint_name.replace('"', '""') if constraint_name else None

    if safe_name:
        cur.execute(
            f"""
            ALTER TABLE proc.workflow_email_tracking
                DROP CONSTRAINT IF EXISTS "{safe_name}"
            """
        )

    try:
        cur.execute(
            """
            ALTER TABLE proc.workflow_email_tracking
                ADD PRIMARY KEY (workflow_id, unique_id)
            """
        )
    except Exception as exc:  # pragma: no cover - defensive for race conditions
        pgcode = getattr(exc, "pgcode", None)
        # ``42P16`` == multiple primary keys (already corrected elsewhere).
        if pgcode != "42P16":
            raise


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL_PG)
        _dedupe_workflow_unique_pairs(cur)
        cur.execute(
            """
            ALTER TABLE proc.workflow_email_tracking
                ADD COLUMN IF NOT EXISTS recipient_emails JSONB
            """
        )
        cur.execute(
            """
            ALTER TABLE proc.workflow_email_tracking
                ADD COLUMN IF NOT EXISTS dispatch_key TEXT
            """
        )
        cur.execute(
            """
            ALTER TABLE proc.workflow_email_tracking
                ADD COLUMN IF NOT EXISTS round_number INTEGER
            """
        )
        _ensure_primary_key(cur)
        cur.execute(
            """
            ALTER TABLE proc.workflow_email_tracking
                ALTER COLUMN dispatch_key DROP NOT NULL
            """
        )
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


def _normalise_expected_ids(ids: Iterable[str]) -> Set[str]:
    return {uid.strip() for uid in ids if uid and uid.strip()}


def load_active_workflow_ids() -> List[str]:
    """Return workflow identifiers with dispatched emails awaiting responses."""

    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT workflow_id FROM proc.workflow_email_tracking"
        )
        workflow_rows = [row[0] for row in cur.fetchall() if row and row[0]]
        cur.close()

    active: List[str] = []
    for workflow_id in workflow_rows:
        workflow_key = str(workflow_id)
        expected_unique_ids, _, _ = draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
            workflow_id=workflow_key
        )
        normalised_expected = _normalise_expected_ids(expected_unique_ids)
        if not normalised_expected:
            continue

        rows = load_workflow_rows(workflow_id=workflow_key)
        if not rows:
            continue

        rows_by_uid: Dict[str, WorkflowDispatchRow] = {
            (row.unique_id or "").strip(): row
            for row in rows
            if (row.unique_id or "").strip()
        }

        if not rows_by_uid:
            continue

        if any(uid not in rows_by_uid for uid in normalised_expected):
            continue

        has_pending = any(
            (rows_by_uid[uid].responded_at is None) or (not bool(rows_by_uid[uid].matched))
            for uid in normalised_expected
        )
        if not has_pending:
            continue

        all_dispatched = all(
            rows_by_uid[uid].dispatched_at is not None for uid in normalised_expected
        )
        if not all_dispatched:
            continue

        all_message_ids = all(
            (rows_by_uid[uid].message_id or "").strip() for uid in normalised_expected
        )
        if not all_message_ids:
            continue

        active.append(workflow_key)

    return active


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


def _serialise_emails(values: Optional[Sequence[str]]) -> Optional[str]:
    if not values:
        return None
    cleaned = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        return None
    return json.dumps(cleaned)


def _parse_email_list(payload: Optional[str]) -> Optional[Sequence[str]]:
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    if isinstance(parsed, list):
        entries = [str(item).strip() for item in parsed if str(item).strip()]
        return tuple(entries) if entries else None
    return None


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
            "(workflow_id, unique_id, dispatch_key, supplier_id, supplier_email, message_id, subject, round_number, "
            "dispatched_at, responded_at, response_message_id, matched, thread_headers) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (workflow_id, unique_id) DO UPDATE SET "
            "dispatch_key=EXCLUDED.dispatch_key, "
            "supplier_id=EXCLUDED.supplier_id, "
            "supplier_email=EXCLUDED.supplier_email, "
            "recipient_emails=EXCLUDED.recipient_emails, "
            "message_id=EXCLUDED.message_id, "
            "subject=EXCLUDED.subject, "
            "round_number=EXCLUDED.round_number, "
            "dispatched_at=EXCLUDED.dispatched_at, "
            "thread_headers=EXCLUDED.thread_headers"
        )
        params = [
            (
                row.workflow_id,
                row.unique_id,
                row.dispatch_key,
                row.supplier_id,
                row.supplier_email,
                _serialise_emails(row.recipient_emails),
                row.message_id,
                row.subject,
                row.round_number,
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
            "SELECT workflow_id, unique_id, dispatch_key, supplier_id, supplier_email, message_id, subject, round_number, "
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
                    dispatch_key=data.get("dispatch_key"),
                    supplier_id=data.get("supplier_id"),
                    supplier_email=data.get("supplier_email"),
                    recipient_emails=_parse_email_list(data.get("recipient_emails")),
                    message_id=data.get("message_id"),
                    subject=data.get("subject"),
                    round_number=data.get("round_number"),
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
            "SELECT workflow_id, unique_id, dispatch_key, supplier_id, supplier_email, message_id, subject, round_number, "
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
        "dispatch_key",
        "supplier_id",
        "supplier_email",
        "message_id",
        "subject",
        "round_number",
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
        dispatch_key=data.get("dispatch_key"),
        supplier_id=data.get("supplier_id"),
        supplier_email=data.get("supplier_email"),
        message_id=data.get("message_id"),
        subject=data.get("subject"),
        round_number=data.get("round_number"),
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
    round_number: Optional[int] = None,
) -> None:
    responded = _normalise_dt(responded_at)

    with get_conn() as conn:
        cur = conn.cursor()
        if round_number is not None:
            try:
                round_value = int(round_number)
            except Exception:
                round_value = None
        else:
            round_value = None

        if round_value is not None:
            q = (
                "UPDATE proc.workflow_email_tracking "
                "SET responded_at=%s, response_message_id=%s, matched=TRUE, round_number=COALESCE(%s, round_number) "
                "WHERE workflow_id=%s AND unique_id=%s"
            )
            cur.execute(q, (responded, response_message_id, round_value, workflow_id, unique_id))
        else:
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
