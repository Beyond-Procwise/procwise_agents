"""Integrated utilities that bridge outbound dispatch registration and
continuous email monitoring into a shared supplier interaction store."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from repositories import supplier_interaction_repo
from repositories.supplier_interaction_repo import SupplierInteractionRow
from services.email_watcher_v2 import EmailResponse, _default_fetcher
from utils import email_tracking

logger = logging.getLogger(__name__)


def generate_unique_email_id(workflow_id: str, supplier_id: Optional[str], *, round_number: int = 1) -> str:
    """Generate a unique identifier that encodes workflow/supplier round context."""

    supplier_hint = f"{supplier_id or 'anon'}-r{max(1, round_number)}"
    return email_tracking.generate_unique_email_id(workflow_id, supplier_hint)


def embed_unique_id_in_email_body(body: Optional[str], unique_id: str) -> str:
    """Expose :func:`utils.email_tracking.embed_unique_id_in_email_body`."""

    return email_tracking.embed_unique_id_in_email_body(body, unique_id)


@dataclass
class _DispatchContext:
    workflow_id: str
    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    round_number: int
    interaction_type: str
    rfq_id: Optional[str]
    metadata: Dict[str, object]


def register_sent_email(
    agent_nick: str,
    workflow_id: str,
    supplier_id: Optional[str],
    supplier_email: Optional[str],
    unique_id: str,
    *,
    round_number: int = 1,
    interaction_type: str = "initial",
    subject: Optional[str] = None,
    body: Optional[str] = None,
    message_id: Optional[str] = None,
    rfq_id: Optional[str] = None,
    thread_headers: Optional[Dict[str, Iterable[str]]] = None,
) -> None:
    """Persist a dispatched email so the watcher can reconcile inbound replies."""

    supplier_interaction_repo.init_schema()

    headers = thread_headers or {}
    record = SupplierInteractionRow(
        workflow_id=workflow_id,
        unique_id=unique_id,
        supplier_id=supplier_id,
        supplier_email=supplier_email,
        round_number=max(1, round_number),
        direction="outbound",
        interaction_type=interaction_type,
        status="pending",
        subject=subject,
        body=body,
        message_id=message_id,
        in_reply_to=list(headers.get("in_reply_to", [])) if headers else None,
        references=list(headers.get("references", [])) if headers else None,
        rfq_id=rfq_id,
        metadata={"agent": agent_nick},
    )
    supplier_interaction_repo.register_outbound(record)


def get_supplier_responses(
    agent_nick: str,
    workflow_id: str,
    *,
    interaction_type: Optional[str] = None,
    round_number: Optional[int] = None,
    status: str = "received",
) -> List[Dict[str, object]]:
    """Return inbound supplier responses filtered by status."""

    supplier_interaction_repo.init_schema()
    rows = supplier_interaction_repo.fetch_by_status(
        workflow_id=workflow_id,
        status=status,
        interaction_type=interaction_type,
        round_number=round_number,
    )
    logger.debug(
        "Loaded %d supplier interaction rows for workflow=%s status=%s agent=%s",
        len(rows),
        workflow_id,
        status,
        agent_nick,
    )
    return rows


def mark_interaction_processed(agent_nick: str, workflow_id: str, unique_id: str) -> None:
    """Mark a supplier response as processed by downstream agents."""

    if not unique_id:
        return
    logger.debug(
        "Marking supplier interaction processed agent=%s workflow=%s unique_id=%s",
        agent_nick,
        workflow_id,
        unique_id,
    )
    supplier_interaction_repo.mark_status(
        unique_ids=[unique_id],
        direction="inbound",
        status="processed",
        processed_at=datetime.now(timezone.utc),
    )


def wait_for_responses(
    workflow_id: str,
    *,
    round_number: Optional[int] = None,
    interaction_type: Optional[str] = None,
    timeout_minutes: int = 60,
    poll_interval_seconds: int = 30,
) -> List[Dict[str, object]]:
    """Block until responses arrive or timeout is reached."""

    deadline = datetime.now(timezone.utc) + timedelta(minutes=max(1, timeout_minutes))
    while datetime.now(timezone.utc) < deadline:
        rows = supplier_interaction_repo.fetch_by_status(
            workflow_id=workflow_id,
            status="received",
            round_number=round_number,
            interaction_type=interaction_type,
        )
        if rows:
            return rows
        time.sleep(max(1, poll_interval_seconds))
    return []


class ContinuousEmailWatcher:
    """Background IMAP poller that persists supplier responses immediately."""

    def __init__(
        self,
        agent_nick: str,
        *,
        poll_interval_seconds: int = 30,
        lookback_minutes: int = 10,
        email_fetcher=None,
        mailbox: Optional[str] = None,
        imap_host: Optional[str] = None,
        imap_username: Optional[str] = None,
        imap_password: Optional[str] = None,
        imap_port: Optional[int] = None,
        imap_use_ssl: Optional[bool] = None,
        imap_login: Optional[str] = None,
        sleep_fn=time.sleep,
        now_fn=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.agent_nick = agent_nick
        self.poll_interval_seconds = max(5, poll_interval_seconds)
        self.lookback_minutes = max(1, lookback_minutes)
        self._sleep = sleep_fn
        self._now = now_fn
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mailbox = mailbox or "INBOX"
        self._fetcher = email_fetcher
        self._fetcher_kwargs = dict(
            host=imap_host,
            username=imap_username,
            password=imap_password,
            mailbox=self._mailbox,
            port=imap_port or 993,
            use_ssl=True if imap_use_ssl is None else imap_use_ssl,
            login=imap_login,
        )
        self._since = self._now() - timedelta(minutes=self.lookback_minutes)
        supplier_interaction_repo.init_schema()

    def start_monitoring(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="ContinuousEmailWatcher", daemon=True)
        self._thread.start()
        logger.info("ContinuousEmailWatcher started agent=%s mailbox=%s", self.agent_nick, self._mailbox)

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None
        logger.info("ContinuousEmailWatcher stopped agent=%s", self.agent_nick)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                fetcher = self._fetcher or _default_fetcher
                responses = fetcher(since=self._since, **self._fetcher_kwargs)
                if responses:
                    newest = max(resp.received_at for resp in responses if resp.received_at)
                    if newest:
                        self._since = newest
                    for response in responses:
                        self._handle_response(response)
            except Exception:
                logger.exception("ContinuousEmailWatcher encountered an error")
            finally:
                self._sleep(self.poll_interval_seconds)

    def _handle_response(self, response: EmailResponse) -> None:
        outbound = None
        if response.unique_id:
            row = supplier_interaction_repo.lookup_outbound(response.unique_id)
            if row:
                outbound = SupplierInteractionRow(
                    workflow_id=row["workflow_id"],
                    unique_id=row["unique_id"],
                    supplier_id=row.get("supplier_id"),
                    supplier_email=row.get("supplier_email"),
                    round_number=row.get("round_number", 1),
                    direction="outbound",
                    interaction_type=row.get("interaction_type", "initial"),
                    status=row.get("status", "pending"),
                    subject=row.get("subject"),
                    body=row.get("body"),
                    message_id=row.get("message_id"),
                    in_reply_to=row.get("in_reply_to"),
                    references=row.get("references"),
                    rfq_id=row.get("rfq_id"),
                    metadata=row.get("metadata"),
                )
        if outbound is None and response.rfq_id:
            candidates = supplier_interaction_repo.find_pending_by_rfq(response.rfq_id.upper())
            if len(candidates) == 1:
                row = candidates[0]
                outbound = SupplierInteractionRow(
                    workflow_id=row["workflow_id"],
                    unique_id=row["unique_id"],
                    supplier_id=row.get("supplier_id"),
                    supplier_email=row.get("supplier_email"),
                    round_number=row.get("round_number", 1),
                    direction="outbound",
                    interaction_type=row.get("interaction_type", "initial"),
                    status=row.get("status", "pending"),
                    subject=row.get("subject"),
                    body=row.get("body"),
                    message_id=row.get("message_id"),
                    in_reply_to=row.get("in_reply_to"),
                    references=row.get("references"),
                    rfq_id=response.rfq_id,
                    metadata=row.get("metadata"),
                )
        if outbound is None:
            logger.warning(
                "Could not match supplier response to dispatch agent=%s unique_id=%s rfq_id=%s",
                self.agent_nick,
                response.unique_id,
                response.rfq_id,
            )
            return

        supplier_interaction_repo.record_inbound_response(
            outbound=outbound,
            message_id=response.message_id,
            subject=response.subject,
            body=response.body,
            from_address=response.from_address,
            received_at=response.received_at,
            in_reply_to=response.in_reply_to,
            references=response.references,
            rfq_id=response.rfq_id,
            metadata={"agent": self.agent_nick},
        )
        logger.info(
            "Recorded supplier response workflow=%s unique_id=%s agent=%s",
            outbound.workflow_id,
            outbound.unique_id,
            self.agent_nick,
        )
