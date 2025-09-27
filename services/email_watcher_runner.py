"""Utilities for bootstrapping SES email watching workflows."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

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

    logger.info("Previewing %d recent inbound email(s) for mailbox %s", len(preview), mailbox)
    for index, message in enumerate(preview, start=1):
        logger.info(
            "[%d] %s → %s at %s | snippet=%s",
            index,
            message.get("from") or "<unknown>",
            message.get("subject") or "<no subject>",
            message.get("received_at") or "<unknown>",
            message.get("snippet") or "",
        )


def preview_recent_emails(
    watcher: SESEmailWatcher,
    *,
    limit: int = 3,
) -> List[Dict[str, object]]:
    """Return a preview of recent inbound emails without marking them as read."""

    try:
        preview = watcher.peek_recent_messages(limit=limit)
    except Exception:  # pragma: no cover - defensive against unexpected loaders
        logger.exception("Failed to preview recent emails for mailbox %s", watcher.mailbox_address)
        return []

    _log_preview(watcher.mailbox_address, preview)
    return preview


def preview_then_watch(
    agent_nick: AgentNick,
    *,
    preview_limit: int = 3,
    watch_limit: Optional[int] = None,
    watch_interval: Optional[int] = None,
    stop_after: Optional[int] = None,
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
