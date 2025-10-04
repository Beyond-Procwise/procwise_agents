"""Helpers to persist and reconcile email dispatch/response chains."""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Optional, Sequence

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

