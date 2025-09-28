"""Utilities for bootstrapping SES email watching workflows."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional


from agents.base_agent import AgentNick
from services.email_watcher import SESEmailWatcher
from utils.gpu import configure_gpu


logger = logging.getLogger(__name__)

# Ensure GPU-related environment flags mirror the broader application defaults.
configure_gpu()

__all__ = [
    "EmailWatcherStartupResult",
    "preview_recent_emails",
    "preview_then_watch",
]


@dataclass
class EmailWatcherStartupResult:
    """Container describing the initial mailbox state and watcher execution."""

    preview: List[Dict[str, object]]
    processed: Optional[int]
    watcher: SESEmailWatcher
    background_thread: Optional[threading.Thread] = None


def _log_preview(mailbox: str, preview: List[Dict[str, object]]) -> None:
    if not preview:
        logger.info("No recent inbound emails available for mailbox %s", mailbox)
        return

    logger.info(
        "Previewing %d recent inbound email(s) for mailbox %s", len(preview), mailbox
    )
    for index, message in enumerate(preview, start=1):
        subject = message.get("subject") or "<no subject>"
        received_at = message.get("received_at") or "<unknown>"
        logger.info("[%d] subject=%s | received_at=%s", index, subject, received_at)


def _parse_received_at(value: object) -> Optional[datetime]:
    """Best-effort parsing of RFC822 timestamp strings into datetimes."""

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        try:
            parsed = parsedate_to_datetime(value)
        except Exception:  # pragma: no cover - defensive against malformed headers
            parsed = None
        if parsed is not None and parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    return None


def _sort_preview(messages: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Order messages by their ``received_at`` timestamp (newest first)."""

    def sort_key(message: Dict[str, object]) -> datetime:
        parsed = _parse_received_at(message.get("received_at"))
        return parsed or datetime.min.replace(tzinfo=timezone.utc)

    return sorted(messages, key=sort_key, reverse=True)



def preview_recent_emails(
    watcher: SESEmailWatcher,
    *,
    limit: int = 5,
) -> List[Dict[str, object]]:
    """Return a preview of recent inbound emails without marking them as read."""

    try:
        limit_int = max(0, int(limit))
    except Exception:
        limit_int = 3

    if limit_int == 0:
        return []

    try:
        preview = watcher.peek_recent_messages(limit=limit_int)

    except Exception:  # pragma: no cover - defensive against unexpected loaders
        logger.exception("Failed to preview recent emails for mailbox %s", watcher.mailbox_address)
        return []

    sorted_preview = _sort_preview(preview)
    limited_preview = sorted_preview[:limit_int]

    _log_preview(watcher.mailbox_address, limited_preview)
    return limited_preview



def preview_then_watch(
    agent_nick: AgentNick,
    *,
    preview_limit: int = 5,
    watch_limit: Optional[int] = None,
    watch_interval: Optional[int] = None,
    stop_after: Optional[int] = None,
    watch_timeout: Optional[int] = None,
    watcher: Optional[SESEmailWatcher] = None,
    run_in_background: bool = False,
) -> EmailWatcherStartupResult:
    """Preview recent emails and kick off the SES email watcher."""

    email_watcher = watcher or SESEmailWatcher(agent_nick)

    preview = preview_recent_emails(email_watcher, limit=preview_limit)

    def _run_watch() -> int:
        return email_watcher.watch(
            interval=watch_interval,
            limit=watch_limit,
            stop_after=stop_after,
            timeout_seconds=watch_timeout,
        )

    if run_in_background:
        thread = threading.Thread(target=_run_watch, name="ses-email-watcher", daemon=True)
        thread.start()
        processed_count: Optional[int] = None
        background_thread = thread
    else:
        processed_count = _run_watch()
        background_thread = None

    return EmailWatcherStartupResult(
        preview=preview,
        processed=processed_count,
        watcher=email_watcher,
        background_thread=background_thread,
    )
