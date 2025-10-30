"""Compatibility layer exposing the legacy EmailWatcherV2 interface."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from services.email_watcher import (
    ContinuousEmailWatcher,
    EmailDispatchRecord,
    EmailResponse,
    EmailWatcher,
    EmailWatcherConfig,
    ImapEmailFetcher,
    WorkflowTracker,
    embed_unique_id_in_email_body,
    generate_unique_email_id,
    get_supplier_responses,
    mark_interaction_processed,
    parse_email_bytes,
    register_sent_email,
    wait_for_responses,
)


def _parse_email(raw: bytes) -> EmailResponse:
    """Backwards-compatible helper for tests importing the private parser."""

    return parse_email_bytes(raw)


class EmailWatcherV2(EmailWatcher):
    """Thin wrapper around :class:`services.email_watcher.EmailWatcher`."""

    def __init__(
        self,
        *,
        supplier_agent: Optional[SupplierInteractionAgent] = None,
        negotiation_agent: Optional[NegotiationAgent] = None,
        dispatch_wait_seconds: int = 90,
        poll_interval_seconds: int = 30,
        max_poll_attempts: int = 10,
        match_threshold: float = 0.45,
        email_fetcher: Optional[Callable[..., List[EmailResponse]]] = None,
        mailbox: Optional[str] = None,
        imap_host: Optional[str] = None,
        imap_username: Optional[str] = None,
        imap_password: Optional[str] = None,
        imap_port: Optional[int] = None,
        imap_use_ssl: Optional[bool] = None,
        imap_login: Optional[str] = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        lookback_minutes: int = 240,
        max_fetch: Optional[int] = None,
        poll_backoff_factor: Optional[float] = None,
        poll_jitter_seconds: Optional[float] = None,
        poll_max_interval_seconds: Optional[int] = None,
        poll_timeout_seconds: Optional[int] = None,
    ) -> None:
        config = EmailWatcherConfig(
            imap_host=imap_host,
            imap_username=imap_username,
            imap_password=imap_password,
            imap_port=imap_port or 993,
            imap_use_ssl=True if imap_use_ssl is None else imap_use_ssl,
            imap_login=imap_login,
            imap_mailbox=mailbox or "INBOX",
            dispatch_wait_seconds=max(0, dispatch_wait_seconds),
            poll_interval_seconds=max(1, poll_interval_seconds),
            max_poll_attempts=max(1, max_poll_attempts),
            match_threshold=match_threshold,
            lookback_minutes=max(1, lookback_minutes),
            max_fetch=max_fetch,
            poll_backoff_factor=poll_backoff_factor if poll_backoff_factor is not None else 1.8,
            poll_jitter_seconds=poll_jitter_seconds if poll_jitter_seconds is not None else 2.0,
            poll_max_interval_seconds=(
                poll_max_interval_seconds
                if poll_max_interval_seconds is not None
                else max(1, int((poll_interval_seconds or 1) * 6))
            ),
            poll_timeout_seconds=poll_timeout_seconds,
        )
        super().__init__(
            config=config,
            supplier_agent=supplier_agent,
            negotiation_agent=negotiation_agent,
            fetcher=email_fetcher,
            sleep=sleep,
            now=now,
        )

        # legacy attribute compatibility
        self.dispatch_wait_seconds = config.dispatch_wait_seconds
        self.poll_interval_seconds = config.poll_interval_seconds
        self.max_poll_attempts = config.max_poll_attempts
        self.match_threshold = config.match_threshold
        self._mailbox = config.imap_mailbox
        self._imap_host = config.imap_host
        self._imap_username = config.imap_username
        self._imap_password = config.imap_password
        self._imap_port = config.imap_port
        self._imap_use_ssl = config.imap_use_ssl
        self._imap_login = config.imap_login
        self._fetcher = email_fetcher
        self.poll_backoff_factor = config.poll_backoff_factor
        self.poll_jitter_seconds = config.poll_jitter_seconds
        self.poll_max_interval_seconds = config.poll_max_interval_seconds
        self.poll_timeout_seconds = config.poll_timeout_seconds

    def wait_and_collect_responses(self, workflow_id: str) -> Dict[str, object]:
        return self.wait_for_responses(workflow_id)


__all__ = [
    "EmailWatcherV2",
    "EmailWatcher",
    "EmailWatcherConfig",
    "WorkflowTracker",
    "EmailDispatchRecord",
    "EmailResponse",
    "ImapEmailFetcher",
    "ContinuousEmailWatcher",
    "generate_unique_email_id",
    "embed_unique_id_in_email_body",
    "register_sent_email",
    "get_supplier_responses",
    "mark_interaction_processed",
    "wait_for_responses",
    "_parse_email",
]
