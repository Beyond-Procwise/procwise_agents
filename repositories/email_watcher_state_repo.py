"""Persistence helpers for per-round email watcher state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from services.db import get_conn


DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.workflow_round_watcher (
    workflow_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    status TEXT NOT NULL,
    expected_count INTEGER DEFAULT 0,
    responses_received INTEGER DEFAULT 0,
    pending_suppliers JSONB,
    pending_unique_ids JSONB,
    uid_cursor BIGINT,
    start_ts TIMESTAMPTZ,
    last_capture_ts TIMESTAMPTZ,
    timeout_deadline TIMESTAMPTZ,
    reason TEXT,
    last_error TEXT,
    heartbeat_ts TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workflow_id, round)
);
"""


def init_schema() -> None:
    """Initialise the workflow watcher state schema."""

    with get_conn() as conn:
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL.split(";"))):
            cur.execute(statement)
        cur.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_workflow_id(workflow_id: str) -> str:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        raise ValueError("workflow_id is required for watcher state mutations")
    return workflow_key


def _normalise_round(round_number: Optional[int]) -> int:
    if round_number is None:
        return 0
    try:
        value = int(round_number)
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise ValueError("round must be coercible to int") from exc
    return value


def _normalise_ts(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialise_sequence(values: Optional[Iterable[Any]]) -> Optional[str]:
    if values is None:
        return None
    normalised: List[str] = []
    for item in values:
        if item in (None, ""):
            continue
        text = str(item).strip()
        if not text:
            continue
        if text not in normalised:
            normalised.append(text)
    return json.dumps(normalised)


def _deserialize_sequence(value: Any) -> List[str]:
    if value in (None, "", [], {}, ()):  # pragma: no cover - trivial
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode()
        except Exception:  # pragma: no cover - defensive
            value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except Exception:  # pragma: no cover - defensive
            return []
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item)]
    return []


def get_state(workflow_id: str, round_number: Optional[int]) -> Optional[Dict[str, Any]]:
    """Fetch the watcher state for ``(workflow_id, round)`` when present."""

    workflow_key = _normalise_workflow_id(workflow_id)
    round_key = _normalise_round(round_number)
    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT workflow_id, round, status, expected_count, responses_received,
                   pending_suppliers, pending_unique_ids, uid_cursor, start_ts,
                   last_capture_ts, timeout_deadline, reason, last_error,
                   heartbeat_ts, created_at, updated_at
            FROM proc.workflow_round_watcher
            WHERE workflow_id=%s AND round=%s
            """,
            (workflow_key, round_key),
        )
        row = cur.fetchone()
        cur.close()

    if not row:
        return None

    (
        wf,
        rnd,
        status,
        expected_count,
        responses_received,
        pending_suppliers,
        pending_unique_ids,
        uid_cursor,
        start_ts,
        last_capture_ts,
        timeout_deadline,
        reason,
        last_error,
        heartbeat_ts,
        created_at,
        updated_at,
    ) = row

    return {
        "workflow_id": wf,
        "round": rnd,
        "status": status,
        "expected_count": expected_count or 0,
        "responses_received": responses_received or 0,
        "pending_suppliers": _deserialize_sequence(pending_suppliers),
        "pending_unique_ids": _deserialize_sequence(pending_unique_ids),
        "uid_cursor": uid_cursor,
        "start_ts": start_ts,
        "last_capture_ts": last_capture_ts,
        "timeout_deadline": timeout_deadline,
        "reason": reason,
        "last_error": last_error,
        "heartbeat_ts": heartbeat_ts,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def transition_state(
    workflow_id: str,
    round_number: Optional[int],
    *,
    status: str,
    expected_count: Optional[int] = None,
    responses_received: Optional[int] = None,
    pending_suppliers: Optional[Iterable[Any]] = None,
    pending_unique_ids: Optional[Iterable[Any]] = None,
    uid_cursor: Optional[int] = None,
    start_ts: Optional[datetime] = None,
    last_capture_ts: Optional[datetime] = None,
    timeout_deadline: Optional[datetime] = None,
    reason: Optional[str] = None,
    last_error: Optional[str] = None,
    heartbeat_ts: Optional[datetime] = None,
) -> None:
    """Insert or update watcher state with the supplied information."""

    workflow_key = _normalise_workflow_id(workflow_id)
    round_key = _normalise_round(round_number)
    init_schema()

    existing = get_state(workflow_key, round_key) or {}

    payload: Dict[str, Any] = dict(existing)
    payload.update(
        {
            "workflow_id": workflow_key,
            "round": round_key,
            "status": (status or "none").strip() or "none",
            "expected_count": int(expected_count)
            if expected_count is not None
            else payload.get("expected_count", 0),
            "responses_received": int(responses_received)
            if responses_received is not None
            else payload.get("responses_received", 0),
            "uid_cursor": int(uid_cursor) if uid_cursor is not None else payload.get("uid_cursor"),
            "reason": reason if reason is not None else payload.get("reason"),
            "last_error": last_error if last_error is not None else payload.get("last_error"),
        }
    )

    if pending_suppliers is not None:
        payload["pending_suppliers"] = list(pending_suppliers)
    if pending_unique_ids is not None:
        payload["pending_unique_ids"] = list(pending_unique_ids)

    if start_ts is not None or not existing:
        payload["start_ts"] = _normalise_ts(start_ts) or payload.get("start_ts") or _now()
    if last_capture_ts is not None:
        payload["last_capture_ts"] = _normalise_ts(last_capture_ts)
    if timeout_deadline is not None:
        payload["timeout_deadline"] = _normalise_ts(timeout_deadline)

    payload["heartbeat_ts"] = _normalise_ts(heartbeat_ts) or _now()

    _persist_state(payload)


def _persist_state(payload: Dict[str, Any]) -> None:
    pending_suppliers_json = _serialise_sequence(payload.get("pending_suppliers"))
    pending_unique_json = _serialise_sequence(payload.get("pending_unique_ids"))
    start_ts = _normalise_ts(payload.get("start_ts"))
    last_capture_ts = _normalise_ts(payload.get("last_capture_ts"))
    timeout_deadline = _normalise_ts(payload.get("timeout_deadline"))
    heartbeat_ts = _normalise_ts(payload.get("heartbeat_ts"))

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO proc.workflow_round_watcher (
                workflow_id, round, status, expected_count, responses_received,
                pending_suppliers, pending_unique_ids, uid_cursor, start_ts,
                last_capture_ts, timeout_deadline, reason, last_error,
                heartbeat_ts, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (workflow_id, round)
            DO UPDATE SET
                status = EXCLUDED.status,
                expected_count = EXCLUDED.expected_count,
                responses_received = EXCLUDED.responses_received,
                pending_suppliers = EXCLUDED.pending_suppliers,
                pending_unique_ids = EXCLUDED.pending_unique_ids,
                uid_cursor = EXCLUDED.uid_cursor,
                start_ts = COALESCE(EXCLUDED.start_ts, proc.workflow_round_watcher.start_ts),
                last_capture_ts = EXCLUDED.last_capture_ts,
                timeout_deadline = EXCLUDED.timeout_deadline,
                reason = EXCLUDED.reason,
                last_error = EXCLUDED.last_error,
                heartbeat_ts = COALESCE(EXCLUDED.heartbeat_ts, proc.workflow_round_watcher.heartbeat_ts),
                updated_at = NOW()
            """,
            (
                payload["workflow_id"],
                payload["round"],
                payload["status"],
                payload.get("expected_count", 0),
                payload.get("responses_received", 0),
                pending_suppliers_json,
                pending_unique_json,
                payload.get("uid_cursor"),
                start_ts,
                last_capture_ts,
                timeout_deadline,
                payload.get("reason"),
                payload.get("last_error"),
                heartbeat_ts,
            ),
        )
        cur.close()


def archive_other_rounds(workflow_id: str, round_number: Optional[int]) -> None:
    """Mark other rounds for the workflow as completed due to supersession."""

    workflow_key = _normalise_workflow_id(workflow_id)
    round_key = _normalise_round(round_number)
    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE proc.workflow_round_watcher
            SET status = CASE
                    WHEN status IN ('completed', 'partial_timeout', 'failed') THEN status
                    ELSE 'completed'
                END,
                reason = COALESCE(reason, 'superseded_by_round'),
                updated_at = NOW()
            WHERE workflow_id=%s AND round <> %s
            """,
            (workflow_key, round_key),
        )
        cur.close()


def clear_state(workflow_id: str, round_number: Optional[int]) -> None:
    """Remove watcher state for a specific workflow round."""

    workflow_key = _normalise_workflow_id(workflow_id)
    round_key = _normalise_round(round_number)
    init_schema()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM proc.workflow_round_watcher WHERE workflow_id=%s AND round=%s",
            (workflow_key, round_key),
        )
        cur.close()
