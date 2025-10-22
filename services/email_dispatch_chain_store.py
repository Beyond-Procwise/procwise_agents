"""Helpers to persist and reconcile email dispatch/response chains."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence

from repositories.workflow_email_tracking_repo import (
    WorkflowDispatchRow,
    init_schema as init_workflow_tracking_schema,
    record_dispatches as workflow_record_dispatches,
)

try:  # pragma: no cover - psycopg2 may be optional
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - fallback when psycopg2 missing
    Json = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


CHAIN_TABLE = "proc.email_dispatch_chains"


def _collect_message_ids(values: Iterable[str]) -> list[str]:
    collected: list[str] = []
    pattern = re.compile(r"<([^>]+)>")
    for value in values:
        if not value:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
        else:
            trimmed = str(value).strip()
        if not trimmed:
            continue
        # Accept raw message identifiers without angle brackets.
        if trimmed.startswith("<") and trimmed.endswith(">"):
            candidate = trimmed[1:-1].strip()
            if candidate:
                collected.append(candidate)
            continue
        matches = pattern.findall(trimmed)
        if matches:
            collected.extend(match.strip() for match in matches if match.strip())
            continue
        collected.append(trimmed)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in collected:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(item)
    return ordered


def record_dispatch(
    *,
    workflow_id: str,
    unique_id: str,
    supplier_id: Optional[str],
    supplier_email: str,
    message_id: str,
    subject: str,
    dispatched_at: datetime,
    dispatch_key: Optional[str] = None,
    **kwargs,
) -> None:
    """Record dispatch metadata in workflow tracking."""

    if not workflow_id:
        logger.error(
            "CRITICAL: Attempting to record dispatch without workflow_id! unique_id=%s supplier=%s. "
            "This will break response tracking.",
            unique_id,
            supplier_id,
        )
        raise ValueError("workflow_id is required for dispatch tracking")

    logger.info(
        "Recording dispatch: workflow=%s unique=%s supplier=%s message_id=%s",
        workflow_id,
        unique_id,
        supplier_id,
        message_id,
    )

    try:
        init_workflow_tracking_schema()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialise workflow email tracking schema")
        raise

    try:
        workflow_record_dispatches(
            workflow_id=workflow_id,
            dispatches=[
                WorkflowDispatchRow(
                    workflow_id=workflow_id,
                    unique_id=unique_id,
                    dispatch_key=str(dispatch_key or message_id or uuid.uuid4().hex),
                    supplier_id=supplier_id,
                    supplier_email=supplier_email,
                    message_id=message_id,
                    subject=subject,
                    dispatched_at=dispatched_at,
                    responded_at=None,
                    response_message_id=None,
                    matched=False,
                )
            ],
        )
    except Exception:
        logger.exception(
            "Failed to record dispatch for workflow=%s unique=%s",
            workflow_id,
            unique_id,
        )
        raise


def ensure_chain_table(connection) -> None:
    """Create the dispatch chain table if necessary."""

    with connection.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CHAIN_TABLE} (
                id SERIAL PRIMARY KEY,
                rfq_id TEXT NOT NULL,
                message_id TEXT NOT NULL UNIQUE,
                thread_index INTEGER NOT NULL DEFAULT 1,
                supplier_id TEXT,
                workflow_ref TEXT,
                recipients JSONB,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                dispatch_metadata JSONB,
                awaiting_response BOOLEAN NOT NULL DEFAULT TRUE,
                responded_at TIMESTAMPTZ,
                response_message_id TEXT,
                response_metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        # Backfill columns when upgrading existing deployments.
        cur.execute(
            f"ALTER TABLE {CHAIN_TABLE} ADD COLUMN IF NOT EXISTS dispatch_metadata JSONB"
        )
        cur.execute(
            f"ALTER TABLE {CHAIN_TABLE} ADD COLUMN IF NOT EXISTS response_metadata JSONB"
        )


def register_dispatch(
    connection,
    *,
    rfq_id: str,
    message_id: Optional[str],
    subject: str,
    body: str,
    thread_index: Optional[int] = None,
    supplier_id: Optional[str] = None,
    workflow_ref: Optional[str] = None,
    recipients: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Persist a dispatched email so replies can be chained."""

    if not message_id or not rfq_id:
        return

    ensure_chain_table(connection)

    payload = None
    if recipients:
        cleaned = [value for value in recipients if value]
        if cleaned:
            payload = Json(cleaned) if Json is not None else json.dumps(cleaned)
    metadata_json = Json(metadata) if metadata and Json is not None else (
        json.dumps(metadata) if metadata else None
    )

    with connection.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {CHAIN_TABLE} (
                rfq_id,
                message_id,
                thread_index,
                supplier_id,
                workflow_ref,
                recipients,
                subject,
                body,
                dispatch_metadata,
                awaiting_response,
                created_at,
                updated_at
            )
            VALUES (%s, %s, COALESCE(%s, 1), %s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW())
            ON CONFLICT (message_id) DO UPDATE SET
                thread_index = EXCLUDED.thread_index,
                supplier_id = EXCLUDED.supplier_id,
                workflow_ref = EXCLUDED.workflow_ref,
                recipients = EXCLUDED.recipients,
                subject = EXCLUDED.subject,
                body = EXCLUDED.body,
                dispatch_metadata = EXCLUDED.dispatch_metadata,
                awaiting_response = TRUE,
                responded_at = NULL,
                response_message_id = NULL,
                response_metadata = NULL,
                updated_at = NOW()
            """,
            (
                rfq_id,
                message_id,
                thread_index,
                supplier_id,
                workflow_ref,
                payload,
                subject,
                body,
                metadata_json,
            ),
        )


def mark_response(
    connection,
    *,
    rfq_id: Optional[str],
    in_reply_to: Optional[str] = None,
    references: Optional[Iterable[str] | str] = None,
    response_message_id: Optional[str] = None,
    response_metadata: Optional[dict] = None,
) -> bool:
    """Mark dispatch entries as responded when a supplier replies."""

    identifiers: list[str] = []
    if in_reply_to:
        identifiers.extend(_collect_message_ids([in_reply_to]))
    if references:
        if isinstance(references, (list, tuple, set)):
            identifiers.extend(_collect_message_ids(references))
        else:
            identifiers.extend(_collect_message_ids([references]))

    if not identifiers:
        return False

    ensure_chain_table(connection)

    payload = json.dumps(response_metadata) if response_metadata else None
    updated = False

    with connection.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {CHAIN_TABLE}
            SET awaiting_response = FALSE,
                responded_at = NOW(),
                response_message_id = COALESCE(%s, response_message_id),
                response_metadata = COALESCE(%s, response_metadata),
                updated_at = NOW()
            WHERE LOWER(message_id) = ANY(%s)
              AND awaiting_response = TRUE
              AND (%s IS NULL OR rfq_id = %s)
            RETURNING message_id
            """,
            (
                response_message_id,
                payload,
                [identifier.lower() for identifier in identifiers],
                rfq_id,
                rfq_id,
            ),
        )
        updated = bool(cur.rowcount)

    if updated:
        logger.debug("Marked dispatch chain responded for rfq=%s ids=%s", rfq_id, identifiers)
    return updated


def _collect_thread_identifiers_from_message(message: dict) -> list[str]:
    identifiers: list[str] = []

    def _gather(value: object) -> None:
        if value in (None, ""):
            return
        if isinstance(value, (list, tuple, set)):
            for entry in value:
                _gather(entry)
            return
        try:
            text = str(value)
        except Exception:
            return
        trimmed = text.strip()
        if trimmed:
            identifiers.append(trimmed)

    _gather(message.get("in_reply_to"))
    _gather(message.get("references"))

    headers = message.get("headers")
    if isinstance(headers, dict):
        _gather(headers.get("In-Reply-To") or headers.get("in-reply-to"))
        _gather(headers.get("References") or headers.get("references"))

    return identifiers


def _collect_body_message_ids(message: dict) -> list[str]:
    body = message.get("body")
    if not isinstance(body, str) or "<" not in body:
        return []

    pattern = re.compile(r"<([^>]+)>")
    collected: list[str] = []
    seen: set[str] = set()
    for match in pattern.findall(body):
        candidate = match.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        collected.append(candidate)
    return collected


def find_best_chain_match(
    connection,
    message: dict,
    *,
    mailbox: Optional[str] = None,
    lookback_days: int = 5,
    thread_identifiers: Optional[Sequence[str]] = None,
    recipients_hint: bool = False,
) -> Optional[dict]:
    """Return the most relevant dispatch-chain match for ``message``."""

    ensure_chain_table(connection)

    identifiers: list[str] = []
    if thread_identifiers:
        identifiers.extend(thread_identifiers)
    identifiers.extend(_collect_thread_identifiers_from_message(message))
    identifiers.extend(_collect_body_message_ids(message))

    message_ids = _collect_message_ids(identifiers)

    supplier_hint = message.get("supplier_id")
    supplier_norm: Optional[str] = None
    if supplier_hint not in (None, ""):
        try:
            supplier_norm = str(supplier_hint).strip().lower() or None
        except Exception:
            supplier_norm = None

    if lookback_days <= 0:
        lookback_days = 1

    mailbox_filter = mailbox or None

    with connection.cursor() as cur:
        direct_row = None
        if message_ids:
            cur.execute(
                f"""
                SELECT rfq_id, supplier_id, dispatch_metadata, thread_index, created_at, message_id
                FROM {CHAIN_TABLE}
                WHERE awaiting_response = TRUE
                  AND message_id = ANY(%s)
                  AND (%s IS NULL
                       OR dispatch_metadata ->> 'mailbox' IS NULL
                       OR dispatch_metadata ->> 'mailbox' = %s)
                ORDER BY thread_index DESC, created_at DESC
                LIMIT 1
                """,
                (message_ids, mailbox_filter, mailbox_filter),
            )
            direct_row = cur.fetchone()

        if direct_row:
            metadata = direct_row[2]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            return {
                "rfq_id": direct_row[0],
                "supplier_id": direct_row[1],
                "dispatch_metadata": metadata,
                "thread_index": direct_row[3],
                "created_at": direct_row[4],
                "message_id": direct_row[5],
                "matched_via": "message_id",
                "score": 0.75,
            }

        lookback_text = str(int(lookback_days))

        cur.execute(
            f"""
            SELECT rfq_id, supplier_id, dispatch_metadata, thread_index, created_at, message_id
            FROM {CHAIN_TABLE}
            WHERE awaiting_response = TRUE
              AND created_at >= NOW() - (%s || ' days')::INTERVAL
              AND (%s IS NULL
                   OR dispatch_metadata ->> 'mailbox' IS NULL
                   OR dispatch_metadata ->> 'mailbox' = %s)
              AND (%s IS NULL OR LOWER(supplier_id) = %s)
            ORDER BY thread_index DESC, created_at DESC
            LIMIT 1
            """,
            (
                lookback_text,
                mailbox_filter,
                mailbox_filter,
                supplier_norm,
                supplier_norm,
            ),
        )
        window_row = cur.fetchone()

    if not window_row:
        return None

    metadata = window_row[2]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = None

    score = 0.6
    if supplier_norm:
        score = max(score, 0.7)

    return {
        "rfq_id": window_row[0],
        "supplier_id": window_row[1],
        "dispatch_metadata": metadata,
        "thread_index": window_row[3],
        "created_at": window_row[4],
        "message_id": window_row[5],
        "matched_via": "time_window",
        "score": score,
    }


def pending_dispatch_count(connection, rfq_id: Optional[str]) -> int:
    if not rfq_id:
        return 0
    ensure_chain_table(connection)
    with connection.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {CHAIN_TABLE} WHERE rfq_id = %s AND awaiting_response = TRUE",
            (rfq_id,),
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0

