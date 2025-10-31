"""Persistence helpers for round-level supplier response tracking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.workflow_round_responses (
    workflow_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    unique_id TEXT NOT NULL,
    supplier_id TEXT,
    response_received BOOLEAN DEFAULT FALSE,
    responded_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'waiting',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    PRIMARY KEY (workflow_id, round_number, unique_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_round_responses_lookup
    ON proc.workflow_round_responses (workflow_id, round_number, supplier_id);
"""


@dataclass
class RoundStatus:
    """Summary of supplier response completion for a round."""

    workflow_id: str
    round_number: int
    total: int
    completed_unique_ids: List[str]
    pending_unique_ids: List[str]
    failed_unique_ids: List[str]
    supplier_map: Dict[str, str]
    status_map: Dict[str, str]

    @property
    def complete(self) -> bool:
        return self.total > 0 and not self.pending_unique_ids and not self.failed_unique_ids

    def pending_suppliers(self) -> List[str]:
        return sorted({self.supplier_map.get(uid, "") for uid in self.pending_unique_ids if self.supplier_map.get(uid)})

    def completed_suppliers(self) -> List[str]:
        return sorted({self.supplier_map.get(uid, "") for uid in self.completed_unique_ids if self.supplier_map.get(uid)})

    def failed_suppliers(self) -> List[str]:
        return sorted({self.supplier_map.get(uid, "") for uid in self.failed_unique_ids if self.supplier_map.get(uid)})


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL.split(";"))):
            cur.execute(statement)
        cur.close()


def _normalise_round(round_number: Optional[int]) -> Optional[int]:
    if round_number is None:
        return None
    try:
        value = int(round_number)
    except Exception:
        return None
    return value if value >= 0 else None


def _normalise_timestamp(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def register_expected(
    *,
    workflow_id: str,
    expectations: Iterable[Tuple[Optional[int], Optional[str], Optional[str]]],
) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return

    rows: List[Tuple[str, int, str, Optional[str]]] = []
    for round_number, unique_id, supplier_id in expectations:
        round_value = _normalise_round(round_number)
        token = (unique_id or "").strip()
        if round_value is None or not token:
            continue
        supplier_token = (supplier_id or "").strip() or None
        rows.append((workflow_key, round_value, token, supplier_token))

    if not rows:
        return

    init_schema()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO proc.workflow_round_responses (
                workflow_id, round_number, unique_id, supplier_id,
                response_received, responded_at, status, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, FALSE, NULL, 'waiting', NOW(), NOW())
            ON CONFLICT (workflow_id, round_number, unique_id)
            DO UPDATE SET
                supplier_id = COALESCE(EXCLUDED.supplier_id, proc.workflow_round_responses.supplier_id),
                status = CASE
                    WHEN proc.workflow_round_responses.response_received THEN proc.workflow_round_responses.status
                    ELSE 'waiting'
                END,
                updated_at = NOW()
            """,
            rows,
        )
        cur.close()


def mark_response_received(
    *,
    workflow_id: str,
    round_number: Optional[int],
    unique_id: str,
    supplier_id: Optional[str] = None,
    responded_at: Optional[datetime] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return
    token = (unique_id or "").strip()
    if not token:
        return
    round_value = _normalise_round(round_number)
    if round_value is None:
        return

    init_schema()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE proc.workflow_round_responses
            SET response_received = TRUE,
                responded_at = COALESCE(responded_at, %s),
                status = 'completed',
                supplier_id = COALESCE(%s, supplier_id),
                metadata = CASE
                    WHEN %s IS NULL THEN metadata
                    ELSE %s
                END,
                updated_at = NOW()
            WHERE workflow_id = %s AND round_number = %s AND unique_id = %s
            """,
            (
                _normalise_timestamp(responded_at),
                (supplier_id or "").strip() or None,
                json.dumps(metadata) if metadata is not None else None,
                json.dumps(metadata) if metadata is not None else None,
                workflow_key,
                round_value,
                token,
            ),
        )
        cur.close()


def mark_round_failed(
    *,
    workflow_id: str,
    round_number: Optional[int],
    unique_ids: Sequence[str],
    reason: Optional[str] = None,
) -> None:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return
    round_value = _normalise_round(round_number)
    if round_value is None:
        return
    tokens = [token for token in (uid.strip() for uid in unique_ids if isinstance(uid, str)) if token]
    if not tokens:
        return

    init_schema()
    payload = json.dumps({"reason": reason or "timeout"})
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE proc.workflow_round_responses
            SET status = 'failed',
                updated_at = NOW(),
                metadata = CASE
                    WHEN metadata IS NULL THEN %s
                    ELSE metadata || %s::jsonb
                END
            WHERE workflow_id = %s AND round_number = %s AND unique_id = ANY(%s)
            """,
            (payload, payload, workflow_key, round_value, tokens),
        )
        cur.close()


def get_round_status(*, workflow_id: str, round_number: Optional[int]) -> Optional[RoundStatus]:
    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return None
    round_value = _normalise_round(round_number)
    if round_value is None:
        return None

    init_schema()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT unique_id, supplier_id, response_received, status
            FROM proc.workflow_round_responses
            WHERE workflow_id = %s AND round_number = %s
            """,
            (workflow_key, round_value),
        )
        rows = cur.fetchall()
        cur.close()

    if not rows:
        return None

    completed: List[str] = []
    pending: List[str] = []
    failed: List[str] = []
    supplier_map: Dict[str, str] = {}
    status_map: Dict[str, str] = {}

    for unique_id, supplier_id, response_received, status in rows:
        token = (unique_id or "").strip()
        if not token:
            continue
        supplier_token = (supplier_id or "").strip()
        if supplier_token:
            supplier_map[token] = supplier_token
        status_token = (status or "waiting").strip().lower()
        status_map[token] = status_token
        if response_received:
            completed.append(token)
        elif status_token == "failed":
            failed.append(token)
        else:
            pending.append(token)

    return RoundStatus(
        workflow_id=workflow_key,
        round_number=round_value,
        total=len(rows),
        completed_unique_ids=sorted(completed),
        pending_unique_ids=sorted(pending),
        failed_unique_ids=sorted(failed),
        supplier_map=supplier_map,
        status_map=status_map,
    )


def all_responses_received(*, workflow_id: str, round_number: Optional[int]) -> bool:
    status = get_round_status(workflow_id=workflow_id, round_number=round_number)
    return bool(status and status.complete)
