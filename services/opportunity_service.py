"""Utility helpers for persisting opportunity feedback state."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _ensure_feedback_table(conn) -> None:
    """Ensure the feedback table exists with the expected schema."""

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS proc.opportunity_feedback (
                    opportunity_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason TEXT,
                    user_id TEXT,
                    metadata JSONB,
                    created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "ALTER TABLE proc.opportunity_feedback ADD COLUMN IF NOT EXISTS metadata JSONB"
            )
            cur.execute(
                """
                ALTER TABLE proc.opportunity_feedback
                ADD COLUMN IF NOT EXISTS updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
                """
            )
        conn.commit()
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to ensure opportunity feedback table")
        conn.rollback()


def record_opportunity_feedback(
    agent_nick,
    opportunity_id: str,
    *,
    status: str,
    reason: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert or update feedback for a given opportunity."""

    payload: Dict[str, Any]
    try:
        with agent_nick.get_db_connection() as conn:
            _ensure_feedback_table(conn)
            serialised_meta = json.dumps(metadata) if metadata is not None else None
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO proc.opportunity_feedback
                        (opportunity_id, status, reason, user_id, metadata, updated_on)
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (opportunity_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        reason = EXCLUDED.reason,
                        user_id = EXCLUDED.user_id,
                        metadata = EXCLUDED.metadata,
                        updated_on = NOW()
                    RETURNING opportunity_id, status, reason, user_id, metadata, updated_on
                    """,
                    (
                        opportunity_id,
                        status,
                        reason,
                        user_id,
                        serialised_meta,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        if not row:
            raise RuntimeError("No feedback row returned")

        payload = {
            "opportunity_id": row[0],
            "status": row[1],
            "reason": row[2],
            "user_id": row[3],
            "metadata": _deserialize_metadata(row[4]),
            "updated_on": _coerce_datetime(row[5]),
        }
    except Exception:
        logger.exception("Failed to persist opportunity feedback")
        raise
    return payload


def load_opportunity_feedback(agent_nick) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of opportunity ids to recorded feedback."""

    feedback: Dict[str, Dict[str, Any]] = {}
    try:
        with agent_nick.get_db_connection() as conn:
            _ensure_feedback_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT opportunity_id, status, reason, user_id, metadata, updated_on
                    FROM proc.opportunity_feedback
                    """
                )
                rows = cur.fetchall() or []
        for row in rows:
            opportunity_id = row[0]
            if not opportunity_id:
                continue
            feedback[opportunity_id] = {
                "status": row[1],
                "reason": row[2],
                "user_id": row[3],
                "metadata": _deserialize_metadata(row[4]),
                "updated_on": _coerce_datetime(row[5]),
            }
    except Exception:  # pragma: no cover - best effort
        logger.exception("Failed to load opportunity feedback")
    return feedback


def _deserialize_metadata(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Unable to decode feedback metadata: %s", value)
        return None


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Unable to parse feedback timestamp: %s", value)
        return None
