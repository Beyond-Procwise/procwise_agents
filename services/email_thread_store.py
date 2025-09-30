"""Utilities for persisting outbound email thread mappings."""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Sequence, Tuple

try:  # pragma: no cover - psycopg2 may be optional in some environments
    from psycopg2 import sql
    from psycopg2.extras import Json
except Exception:  # pragma: no cover
    sql = None  # type: ignore[assignment]
    Json = None  # type: ignore[assignment]


DEFAULT_THREAD_TABLE = "proc.email_thread_map"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sanitise_thread_table_name(
    name: Optional[str], *, logger: Optional[logging.Logger] = None
) -> str:
    """Return a safe, schema-qualified table name for the thread map."""

    candidate = str(name or "").strip()
    if not candidate:
        return DEFAULT_THREAD_TABLE

    parts = candidate.split(".")
    if len(parts) > 2 or any(not _IDENTIFIER_RE.match(part) for part in parts):
        if logger:
            logger.warning(
                "Invalid email thread table name %r; using default %s",
                candidate,
                DEFAULT_THREAD_TABLE,
            )
        return DEFAULT_THREAD_TABLE

    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}.{parts[1]}"


def _split_table_name(table_name: str) -> Tuple[Optional[str], str]:
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        return schema, table
    return None, table_name


def _table_identifier(table_name: str):
    if sql is None:  # pragma: no cover - defensive fallback
        raise RuntimeError("psycopg2 is required for email thread store operations")

    schema, table = _split_table_name(table_name)
    if schema:
        return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    return sql.Identifier(table)


def ensure_thread_table(connection, table_name: str, *, logger: Optional[logging.Logger] = None) -> None:
    """Create the email thread table if it does not already exist."""

    if sql is None:  # pragma: no cover - defensive fallback
        if logger:
            logger.warning("Cannot ensure thread table because psycopg2 is unavailable")
        return

    table_sql = _table_identifier(table_name)
    try:
        with connection.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        message_id TEXT PRIMARY KEY,
                        rfq_id TEXT NOT NULL,
                        supplier_id TEXT,
                        recipients JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(table=table_sql)
            )
    except Exception:  # pragma: no cover - best effort logging
        if logger:
            logger.exception("Failed to ensure email thread mapping table %s", table_name)
        raise


def record_thread_mapping(
    connection,
    table_name: str,
    *,
    message_id: str,
    rfq_id: str,
    supplier_id: Optional[str] = None,
    recipients: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Insert or update a thread mapping for the dispatched email."""

    if sql is None or Json is None:  # pragma: no cover - defensive fallback
        if logger:
            logger.warning("Cannot record thread mapping because psycopg2 is unavailable")
        return

    if not message_id or not rfq_id:
        return

    table_sql = _table_identifier(table_name)
    recipient_list: Optional[Sequence[str]] = None
    if recipients:
        recipient_list = [value for value in recipients if isinstance(value, str) and value.strip()]

    payload = Json(recipient_list) if recipient_list is not None else None

    try:
        with connection.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (message_id, rfq_id, supplier_id, recipients, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (message_id) DO UPDATE SET
                        rfq_id = EXCLUDED.rfq_id,
                        supplier_id = EXCLUDED.supplier_id,
                        recipients = EXCLUDED.recipients,
                        updated_at = NOW()
                    """
                ).format(table=table_sql),
                (message_id, rfq_id, supplier_id, payload),
            )
    except Exception:  # pragma: no cover - best effort logging
        if logger:
            logger.exception("Failed to record thread mapping for message %s", message_id)
        raise


def lookup_rfq_from_threads(
    connection,
    table_name: str,
    message_ids: Sequence[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Return the first RFQ identifier associated with ``message_ids``."""

    if sql is None:  # pragma: no cover - defensive fallback
        if logger:
            logger.debug("psycopg2 unavailable; skipping thread lookup")
        return None

    candidates = [value for value in message_ids if isinstance(value, str) and value.strip()]
    if not candidates:
        return None

    table_sql = _table_identifier(table_name)
    try:
        with connection.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT rfq_id FROM {table} WHERE message_id = ANY(%s) LIMIT 1").format(
                    table=table_sql
                ),
                (candidates,),
            )
            row = cur.fetchone()
    except Exception:  # pragma: no cover - best effort logging
        if logger:
            logger.exception("Failed to lookup RFQ from thread map for %s", candidates)
        return None

    if row:
        rfq_id = row[0]
        if isinstance(rfq_id, str):
            return rfq_id.upper()
    return None
