"""Repository helpers for workflow-level lifecycle tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.workflow_lifecycle (
    workflow_id TEXT PRIMARY KEY,
    supplier_agent_status TEXT,
    supplier_agent_updated_at TIMESTAMPTZ,
    negotiation_status TEXT,
    negotiation_updated_at TIMESTAMPTZ,
    watcher_status TEXT,
    watcher_started_at TIMESTAMPTZ,
    watcher_stopped_at TIMESTAMPTZ,
    watcher_runtime_seconds NUMERIC(18, 6),
    expected_responses INTEGER,
    received_responses INTEGER,
    last_event TEXT,
    last_event_at TIMESTAMPTZ,
    metadata JSONB
);
"""

_ALLOWED_SUPPLIER_STATUSES = {
    "started",
    "invoked",
    "running",
    "awaiting_responses",
    "waiting_for_responses",
    "responses_completed",
    "round_failed",
}
_ALLOWED_NEGOTIATION_STATUSES = {"started", "running", "completed", "finalized", "failed"}
_WATCHER_EVENTS = {"watcher_started", "watcher_active", "watcher_stopped"}


def init_schema() -> None:
    """Ensure the lifecycle table exists before performing mutations."""

    with get_conn() as conn:
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL.split(";"))):
            cur.execute(statement)
        cur.close()


def _normalise_timestamp(value: Optional[datetime]) -> datetime:
    if value is None:
        value = datetime.now(timezone.utc)
    elif value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _serialise_metadata(value: Any) -> Optional[str]:
    if value in (None, "", {}, []):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return json.dumps(value)
    except TypeError:
        try:
            sanitised = {key: str(val) for key, val in dict(value).items()}  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive
            return None
        return json.dumps(sanitised)


def _load_metadata(raw: Any) -> Optional[Dict[str, Any]]:
    if raw in (None, "", {}, []):
        return None
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return json.loads(raw)
    except Exception:  # pragma: no cover - defensive
        return None


def _select_row(workflow_id: str) -> Optional[Dict[str, Any]]:
    if not workflow_id:
        return None
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT workflow_id, supplier_agent_status, supplier_agent_updated_at,
                   negotiation_status, negotiation_updated_at,
                   watcher_status, watcher_started_at, watcher_stopped_at,
                   watcher_runtime_seconds, expected_responses, received_responses,
                   last_event, last_event_at, metadata
            FROM proc.workflow_lifecycle
            WHERE workflow_id=%s
            """,
            (workflow_id,),
        )
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    columns = [
        "workflow_id",
        "supplier_agent_status",
        "supplier_agent_updated_at",
        "negotiation_status",
        "negotiation_updated_at",
        "watcher_status",
        "watcher_started_at",
        "watcher_stopped_at",
        "watcher_runtime_seconds",
        "expected_responses",
        "received_responses",
        "last_event",
        "last_event_at",
        "metadata",
    ]
    data = dict(zip(columns, row))
    data["metadata"] = _load_metadata(data.get("metadata"))
    return data


def get_lifecycle(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Return the lifecycle record for ``workflow_id`` when present."""

    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return None
    return _select_row(workflow_key)


def reset_workflow(workflow_id: str) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM proc.workflow_lifecycle WHERE workflow_id=%s",
            (workflow_key,),
        )
        cur.close()


def record_supplier_agent_status(
    workflow_id: str,
    status: str,
    *,
    timestamp: Optional[datetime] = None,
) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return
    status_text = (status or "").strip().lower()
    if not status_text:
        return
    if status_text not in _ALLOWED_SUPPLIER_STATUSES:
        logger.debug(
            "workflow_lifecycle_repo ignoring supplier status %s for workflow %s",
            status_text,
            workflow_key,
        )
    init_schema()
    current = _select_row(workflow_key) or {"workflow_id": workflow_key}
    moment = _normalise_timestamp(timestamp)
    current.update(
        {
            "supplier_agent_status": status_text,
            "supplier_agent_updated_at": moment,
            "last_event": f"supplier_agent:{status_text}",
            "last_event_at": moment,
        }
    )
    _persist(workflow_key, current)


def record_negotiation_status(
    workflow_id: str,
    status: str,
    *,
    timestamp: Optional[datetime] = None,
) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return
    status_text = (status or "").strip().lower()
    if not status_text:
        return
    if status_text not in _ALLOWED_NEGOTIATION_STATUSES:
        logger.debug(
            "workflow_lifecycle_repo ignoring negotiation status %s for workflow %s",
            status_text,
            workflow_key,
        )
    init_schema()
    current = _select_row(workflow_key) or {"workflow_id": workflow_key}
    moment = _normalise_timestamp(timestamp)
    current.update(
        {
            "negotiation_status": status_text,
            "negotiation_updated_at": moment,
            "last_event": f"negotiation:{status_text}",
            "last_event_at": moment,
        }
    )
    _persist(workflow_key, current)


def record_watcher_event(
    workflow_id: str,
    event: str,
    *,
    timestamp: Optional[datetime] = None,
    expected_responses: Optional[int] = None,
    received_responses: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    runtime_seconds: Optional[float] = None,
) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return
    event_key = (event or "").strip().lower()
    if event_key not in _WATCHER_EVENTS:
        logger.debug(
            "workflow_lifecycle_repo ignoring watcher event %s for workflow %s",
            event,
            workflow_key,
        )
        return
    init_schema()
    current = _select_row(workflow_key) or {"workflow_id": workflow_key}
    moment = _normalise_timestamp(timestamp)
    updates: Dict[str, Any] = {
        "last_event": event_key,
        "last_event_at": moment,
    }
    if event_key == "watcher_started":
        updates.update(
            {
                "watcher_status": "started",
                "watcher_started_at": moment,
                "watcher_stopped_at": None,
                "watcher_runtime_seconds": None,
            }
        )
    elif event_key == "watcher_active":
        updates["watcher_status"] = "active"
    elif event_key == "watcher_stopped":
        updates.update(
            {
                "watcher_status": "stopped",
                "watcher_stopped_at": moment,
                "watcher_runtime_seconds": runtime_seconds,
            }
        )
    if expected_responses is not None:
        try:
            updates["expected_responses"] = int(expected_responses)
        except Exception:  # pragma: no cover - defensive conversion
            updates["expected_responses"] = expected_responses
    if received_responses is not None:
        try:
            updates["received_responses"] = int(received_responses)
        except Exception:  # pragma: no cover - defensive conversion
            updates["received_responses"] = received_responses
    serialised_metadata = _serialise_metadata(metadata)
    if serialised_metadata is not None:
        updates["metadata"] = serialised_metadata
    current.update(updates)
    _persist(workflow_key, current)


def _persist(workflow_id: str, payload: Dict[str, Any]) -> None:
    columns = [
        "supplier_agent_status",
        "supplier_agent_updated_at",
        "negotiation_status",
        "negotiation_updated_at",
        "watcher_status",
        "watcher_started_at",
        "watcher_stopped_at",
        "watcher_runtime_seconds",
        "expected_responses",
        "received_responses",
        "last_event",
        "last_event_at",
        "metadata",
    ]
    existing = _select_row(workflow_id)
    with get_conn() as conn:
        cur = conn.cursor()
        if existing:
            set_columns = [col for col in columns if col in payload]
            if not set_columns:
                cur.close()
                return
            assignments = ", ".join(f"{col}=%s" for col in set_columns)
            params = [
                _serialise_metadata(payload.get(col)) if col == "metadata" else payload.get(col)
                for col in set_columns
            ] + [workflow_id]
            cur.execute(
                f"UPDATE proc.workflow_lifecycle SET {assignments} WHERE workflow_id=%s",
                tuple(params),
            )
        else:
            insert_cols = ["workflow_id"] + [col for col in columns if col in payload]
            placeholders = ", ".join(["%s"] * len(insert_cols))
            values = [workflow_id]
            for col in insert_cols[1:]:
                value = payload.get(col)
                if col == "metadata":
                    value = _serialise_metadata(value)
                values.append(value)
            cur.execute(
                f"INSERT INTO proc.workflow_lifecycle ({', '.join(insert_cols)}) VALUES ({placeholders})",
                tuple(values),
            )
        cur.close()
