"""Compatibility layer exposing the legacy EmailWatcherV2 interface."""

from __future__ import annotations

import imaplib
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from repositories import draft_rfq_emails_repo, email_dispatch_repo, workflow_lifecycle_repo
from services.email_watcher import (
    ContinuousEmailWatcher,
    EmailDispatchRecord,
    EmailWatcher,
    EmailWatcherConfig,
    ImapEmailFetcher,
    WorkflowTracker,
    _imap_client,
    embed_unique_id_in_email_body,
    generate_unique_email_id,
    get_supplier_responses,
    mark_interaction_processed,
    parse_email_bytes,
    register_sent_email,
    wait_for_responses,
)

from services.email_watcher import EmailResponse as _EmailResponse

logger = logging.getLogger(__name__)


def _parse_email(raw: bytes) -> EmailResponse:
    """Backwards-compatible helper for tests importing the private parser."""

    return parse_email_bytes(raw)


def EmailResponse(*args, body_text: Optional[str] = None, **kwargs) -> _EmailResponse:
    """Compatibility wrapper that accepts legacy ``body_text`` keyword arguments."""

    if body_text is not None and "body" not in kwargs and len(args) < 7:
        kwargs["body"] = body_text
    return _EmailResponse(*args, **kwargs)


def _default_fetcher(
    *,
    host: str,
    username: str,
    password: str,
    mailbox: str = "INBOX",
    port: int = 993,
    use_ssl: bool = True,
    login: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> Tuple[List[_EmailResponse], Optional[int]]:
    """Legacy synchronous IMAP fetcher returning parsed responses and the last UID."""

    client = _imap_client(host, username, password, port=port, use_ssl=use_ssl, login=login)
    try:
        client.select(mailbox, readonly=True)
        status, data = client.uid("SEARCH", None, "UNSEEN")
        if status != "OK":
            return [], None

        identifiers: List[int] = []
        for entry in data:
            if isinstance(entry, bytes):
                text = entry.decode("utf-8", errors="ignore").strip()
                if text:
                    identifiers.extend(int(token) for token in text.split() if token.isdigit())

        if not identifiers:
            return [], None

        responses: List[_EmailResponse] = []
        last_uid: Optional[int] = None
        for identifier in identifiers[-limit if limit else None :]:
            status, payload = client.uid("FETCH", str(identifier), "(RFC822)")
            if status != "OK":
                continue
            for part in payload:
                if not isinstance(part, tuple) or len(part) < 2:
                    continue
                raw_bytes = part[1]
                if not isinstance(raw_bytes, (bytes, bytearray)):
                    continue
                try:
                    responses.append(parse_email_bytes(bytes(raw_bytes)))
                    last_uid = identifier
                except Exception:
                    continue

        if since is not None:
            filtered: List[_EmailResponse] = []
            for resp in responses:
                received = getattr(resp, "received_at", None)
                if received is None or received < since:
                    continue
                filtered.append(resp)
            responses = filtered

        zero_point = datetime.min.replace(tzinfo=timezone.utc)
        responses.sort(key=lambda item: getattr(item, "received_at", zero_point))
        return responses, last_uid
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            client.logout()
        except Exception:
            pass


class _SupplierAgentProxy:
    """Inject legacy metadata into supplier agent contexts."""

    def __init__(self, delegate: SupplierInteractionAgent):
        self._delegate = delegate

    def __getattr__(self, item):
        return getattr(self._delegate, item)

    def execute(self, context):  # type: ignore[override]
        payload = getattr(context, "input_data", None)
        if isinstance(payload, dict):
            payload.setdefault("workflow_status", "responses_complete")
            payload.setdefault("status", "responses_complete")
            payload.setdefault("timeout_reached", False)
        return self._delegate.execute(context)


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
        match_threshold: float = 0.8,
        email_fetcher: Optional[Callable[..., List[_EmailResponse]]] = None,
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
        response_idle_timeout_seconds: Optional[int] = None,
        max_total_wait_seconds: Optional[int] = None,
    ) -> None:
        if poll_timeout_seconds is None and max_total_wait_seconds is not None:
            poll_timeout_seconds = max_total_wait_seconds

        if response_idle_timeout_seconds is None or response_idle_timeout_seconds < 0:
            response_grace_seconds = 0
        else:
            response_grace_seconds = int(response_idle_timeout_seconds)

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
            response_grace_seconds=response_grace_seconds,
        )
        wrapped_supplier = _SupplierAgentProxy(supplier_agent) if supplier_agent is not None else None
        wrapped_fetcher = self._wrap_fetcher(email_fetcher) if email_fetcher is not None else None

        super().__init__(
            config=config,
            supplier_agent=wrapped_supplier,
            negotiation_agent=negotiation_agent,
            fetcher=wrapped_fetcher,
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
        self._fetcher = wrapped_fetcher if email_fetcher is not None else None
        self._legacy_fetcher = email_fetcher
        self._legacy_supplier_agent = supplier_agent
        self.poll_backoff_factor = config.poll_backoff_factor
        self.poll_jitter_seconds = config.poll_jitter_seconds
        self.poll_max_interval_seconds = config.poll_max_interval_seconds
        self.poll_timeout_seconds = config.poll_timeout_seconds
        self.response_idle_timeout_seconds = config.response_grace_seconds
        self.max_total_wait_seconds = config.poll_timeout_seconds

    @staticmethod
    def _wrap_fetcher(
        fetcher: Callable[..., List[_EmailResponse]]
    ) -> Callable[..., List[_EmailResponse]]:
        def _wrapped(*args, **kwargs):
            try:
                return fetcher(*args, **kwargs)
            except (imaplib.IMAP4.abort, imaplib.IMAP4.error):
                return []

        return _wrapped

    def wait_and_collect_responses(self, workflow_id: str) -> Dict[str, object]:
        raw_result = self.wait_for_responses(workflow_id)
        retries = 0
        while (
            retries < 3
            and not raw_result.get("complete")
            and int(raw_result.get("responded_count") or 0) > 0
            and int(raw_result.get("dispatched_count") or 0) > 0
            and int(raw_result.get("responded_count") or 0)
            < int(raw_result.get("dispatched_count") or 0)
            and raw_result.get("timeout_reason") in (None, "max_attempts")
        ):
            previous_count = int(raw_result.get("responded_count") or 0)
            next_result = self.wait_for_responses(workflow_id)
            retries += 1
            if int(next_result.get("responded_count") or 0) <= previous_count:
                raw_result = next_result
                break
            raw_result = next_result

        result: Dict[str, object] = dict(raw_result)

        dispatched_count = int(result.get("dispatched_count") or 0)
        if dispatched_count == 0:
            try:
                expected_unique_ids, supplier_map, _ = draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
                    workflow_id=workflow_id
                )
            except Exception:
                expected_unique_ids, supplier_map = set(), {}
            try:
                dispatched_total = email_dispatch_repo.count_completed_supplier_dispatches(workflow_id)
            except Exception:
                dispatched_total = 0
            if expected_unique_ids:
                pending_unique_ids = sorted(uid for uid in expected_unique_ids if uid)
                if dispatched_total < len(expected_unique_ids):
                    result.update(
                        {
                            "workflow_id": workflow_id,
                            "complete": False,
                            "status": "max_attempts_exceeded",
                            "workflow_status": "partial_timeout",
                            "timeout_reached": True,
                            "dispatched_count": dispatched_total,
                            "responded_count": 0,
                            "expected_responses": len(expected_unique_ids),
                            "pending_unique_ids": pending_unique_ids,
                            "pending_suppliers": [
                                supplier_map.get(uid) for uid in pending_unique_ids if supplier_map.get(uid)
                            ],
                        }
                    )
                    try:
                        workflow_lifecycle_repo.record_watcher_event(
                            workflow_id,
                            "watcher_stopped",
                            expected_responses=result.get("expected_responses"),
                            received_responses=result.get("responded_count"),
                            metadata={
                                "status": result.get("status"),
                                "workflow_status": result.get("workflow_status"),
                                "timeout_reached": True,
                                "stop_reason": "partial_timeout",
                                "pending_unique_ids": pending_unique_ids,
                            },
                        )
                    except Exception:
                        logger.exception("Failed to record watcher lifecycle stop for workflow %s", workflow_id)
                    return result

        status = result.get("status")
        timeout_reason = result.get("timeout_reason")

        if status == "failed" and timeout_reason:
            if timeout_reason == "timeout":
                status = "timeout"
            elif timeout_reason == "max_attempts":
                status = "max_attempts_exceeded"
            result["status"] = status

        if status is None:
            status = "pending"

        expected_responses = int(result.get("dispatched_count") or 0)
        result.setdefault("expected_responses", expected_responses)

        complete = bool(result.get("complete"))
        timeout_reached = False
        if status in {"timeout", "max_attempts_exceeded"}:
            timeout_reached = True
        if timeout_reason in {"timeout", "max_attempts"}:
            timeout_reached = True
        if status == "failed" and not complete:
            timeout_reached = True
        result["timeout_reached"] = timeout_reached

        workflow_status = "responses_complete" if complete else "partial_timeout"
        if complete and status not in {"completed", "success"}:
            result.setdefault("status", "completed")
        if not complete and status not in {"timeout", "max_attempts_exceeded", "failed"}:
            workflow_status = status
        result["workflow_status"] = workflow_status

        result.setdefault("matched_responses", {})
        if not complete:
            result.setdefault("pending_unique_ids", result.get("pending_unique_ids", []))
            result.setdefault("pending_suppliers", result.get("pending_suppliers", []))
        else:
            result.pop("pending_unique_ids", None)
            result.pop("pending_suppliers", None)

        metadata = {
            "status": result.get("status"),
            "workflow_status": result.get("workflow_status"),
            "timeout_reached": result.get("timeout_reached"),
        }
        if result.get("timeout_reached"):
            metadata["stop_reason"] = result.get("workflow_status")
        try:
            workflow_lifecycle_repo.record_watcher_event(
                workflow_id,
                "watcher_stopped",
                expected_responses=result.get("expected_responses"),
                received_responses=result.get("responded_count"),
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to record watcher lifecycle stop for workflow %s", workflow_id)

        return result


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
    "_imap_client",
    "_default_fetcher",
]
