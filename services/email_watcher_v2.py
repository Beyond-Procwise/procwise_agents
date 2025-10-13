"""Workflow-centric email watcher implementation (EmailWatcherV2)."""

from __future__ import annotations

import email
import imaplib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from repositories import workflow_email_tracking_repo as tracking_repo
from repositories import supplier_response_repo
from repositories.supplier_response_repo import SupplierResponseRow
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from utils.email_tracking import extract_tracking_metadata, extract_unique_id_from_body

logger = logging.getLogger(__name__)


@dataclass
class EmailDispatchRecord:
    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    thread_headers: Dict[str, str] = field(default_factory=dict)
    dispatched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EmailResponse:
    unique_id: Optional[str]
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    from_address: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    body: str
    received_at: datetime
    in_reply_to: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)


@dataclass
class WorkflowTracker:
    workflow_id: str
    dispatched_count: int = 0
    responded_count: int = 0
    email_records: Dict[str, EmailDispatchRecord] = field(default_factory=dict)
    matched_responses: Dict[str, EmailResponse] = field(default_factory=dict)
    all_dispatched: bool = False
    all_responded: bool = False
    last_dispatched_at: Optional[datetime] = None

    def register_dispatches(self, dispatches: Iterable[EmailDispatchRecord]) -> None:
        for dispatch in dispatches:
            self.email_records[dispatch.unique_id] = dispatch
            if dispatch.dispatched_at and (
                self.last_dispatched_at is None or dispatch.dispatched_at > self.last_dispatched_at
            ):
                self.last_dispatched_at = dispatch.dispatched_at
        self.dispatched_count = len(self.email_records)
        self.all_dispatched = True

    def record_response(self, unique_id: str, response: EmailResponse) -> None:
        if unique_id not in self.email_records:
            return
        if unique_id in self.matched_responses:
            return
        self.matched_responses[unique_id] = response
        self.responded_count = len(self.matched_responses)
        self.all_responded = self.responded_count >= self.dispatched_count > 0


def _imap_client(
    host: str,
    username: str,
    password: str,
    *,
    port: int = 993,
    use_ssl: bool = True,
    login: Optional[str] = None,
) -> imaplib.IMAP4:
    if use_ssl:
        client: imaplib.IMAP4 = imaplib.IMAP4_SSL(host, port)
    else:  # pragma: no cover - plain IMAP only used in limited environments
        client = imaplib.IMAP4(host, port)
    login_name = login or username
    client.login(login_name, password)
    return client


def _decode_message(raw: bytes) -> EmailMessage:
    return BytesParser(policy=policy.default).parsebytes(raw)


def _extract_thread_ids(message: EmailMessage) -> Dict[str, Sequence[str]]:
    refs: Sequence[str] = []
    in_reply: Sequence[str] = []
    references = message.get_all("References", failobj=[])
    if references:
        refs = tuple(
            ref.strip("<> ")
            for header in references
            for ref in header.split()
            if ref.strip()
        )
    reply = message.get_all("In-Reply-To", failobj=[])
    if reply:
        in_reply = tuple(ref.strip("<> ") for header in reply for ref in header.split() if ref.strip())
    return {"references": refs, "in_reply_to": in_reply}


def _extract_plain_text(message: EmailMessage) -> str:
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    return part.get_content().strip()
                except Exception:
                    continue
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/html" and "attachment" not in disp:
                try:
                    return part.get_content()
                except Exception:
                    continue
        return ""
    return message.get_content() if message.get_content() else ""


def _parse_email(raw: bytes) -> EmailResponse:
    message = _decode_message(raw)
    subject = message.get("Subject")
    message_id = (message.get("Message-ID") or "").strip("<> ") or None
    from_address = message.get("From")
    body = _extract_plain_text(message)
    unique_id = extract_unique_id_from_body(body)
    metadata = extract_tracking_metadata(body)
    if metadata and not unique_id:
        unique_id = metadata.unique_id
    supplier_id = metadata.supplier_id if metadata else None

    date_header = message.get("Date")
    try:
        received_at = email.utils.parsedate_to_datetime(date_header) if date_header else None
    except Exception:  # pragma: no cover - defensive
        received_at = None
    received_at = received_at or datetime.now(timezone.utc)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)

    thread_ids = _extract_thread_ids(message)

    return EmailResponse(
        unique_id=unique_id,
        supplier_id=supplier_id,
        supplier_email=None,
        from_address=from_address,
        message_id=message_id,
        subject=subject,
        body=body or "",
        received_at=received_at,
        in_reply_to=thread_ids.get("in_reply_to", ()),
        references=thread_ids.get("references", ()),
    )


def _normalise_thread_header(value) -> Sequence[str]:
    if value in (None, ""):
        return tuple()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v).strip("<> ") for v in value if v)
    return (str(value).strip("<> "),)


def _default_fetcher(
    *,
    host: str,
    username: str,
    password: str,
    mailbox: str,
    since: datetime,
    port: int = 993,
    use_ssl: bool = True,
    login: Optional[str] = None,
) -> List[EmailResponse]:
    client = _imap_client(host, username, password, port=port, use_ssl=use_ssl, login=login)
    try:
        client.select(mailbox, readonly=True)
        since_str = since.strftime("%d-%b-%Y")
        typ, data = client.search(None, f'(SINCE {since_str})')
        if typ != "OK":
            return []
        ids = (data[0] or b"").decode().split()
        responses: List[EmailResponse] = []
        for message_id in ids:
            typ, payload = client.fetch(message_id, "(RFC822)")
            if typ != "OK" or not payload or not isinstance(payload[0], tuple):
                continue
            raw = payload[0][1]
            responses.append(_parse_email(raw))
        return responses
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            client.logout()
        except Exception:
            pass


def _calculate_match_score(dispatch: EmailDispatchRecord, email: EmailResponse) -> float:
    score = 0.0

    if email.unique_id and email.unique_id == dispatch.unique_id:
        return 1.0

    thread_ids = set(dispatch.thread_headers.get("references", ())) | set(
        dispatch.thread_headers.get("in_reply_to", ())
    )
    if dispatch.message_id:
        thread_ids.add(dispatch.message_id)

    reply_headers = set(email.in_reply_to) | set(email.references)
    if dispatch.message_id and dispatch.message_id in reply_headers:
        score += 0.8
    elif thread_ids & reply_headers:
        score += 0.8

    if dispatch.supplier_email and email.from_address:
        if dispatch.supplier_email.lower() in email.from_address.lower():
            score += 0.6

    if dispatch.subject and email.subject:
        normalised_subject = dispatch.subject.lower()
        if normalised_subject in email.subject.lower():
            score += 0.5

    return score


class EmailWatcherV2:
    """Workflow-aware watcher coordinating dispatch/response tracking."""

    def __init__(
        self,
        *,
        supplier_agent: Optional[SupplierInteractionAgent] = None,
        negotiation_agent: Optional[NegotiationAgent] = None,
        dispatch_wait_seconds: int = 90,
        poll_interval_seconds: int = 30,
        max_poll_attempts: int = 10,
        match_threshold: float = 0.5,
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
    ) -> None:
        self.supplier_agent = supplier_agent
        self.negotiation_agent = negotiation_agent
        self.dispatch_wait_seconds = max(0, dispatch_wait_seconds)
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self.max_poll_attempts = max(1, max_poll_attempts)
        self.match_threshold = match_threshold
        self._fetcher = email_fetcher
        self._mailbox = mailbox or "INBOX"
        self._imap_host = imap_host
        self._imap_username = imap_username
        self._imap_password = imap_password
        self._imap_port = imap_port
        self._imap_use_ssl = imap_use_ssl
        self._imap_login = imap_login
        self._sleep = sleep
        self._now = now
        self._trackers: Dict[str, WorkflowTracker] = {}

        tracking_repo.init_schema()
        supplier_response_repo.init_schema()

    def _ensure_tracker(self, workflow_id: str) -> WorkflowTracker:
        tracker = self._trackers.get(workflow_id)
        if tracker is not None:
            return tracker
        tracker = WorkflowTracker(workflow_id=workflow_id)
        rows = tracking_repo.load_workflow_rows(workflow_id=workflow_id)
        if rows:
            dispatches = [
                EmailDispatchRecord(
                    unique_id=row.unique_id,
                    supplier_id=row.supplier_id,
                    supplier_email=row.supplier_email,
                    message_id=row.message_id,
                    subject=row.subject,
                    dispatched_at=row.dispatched_at,
                )
                for row in rows
            ]
            tracker.register_dispatches(dispatches)
            matched = [row for row in rows if row.matched]
            for row in matched:
                tracker.record_response(
                    row.unique_id,
                    EmailResponse(
                        unique_id=row.unique_id,
                        supplier_id=row.supplier_id,
                        supplier_email=row.supplier_email,
                        from_address=None,
                        message_id=row.response_message_id,
                        subject=None,
                        body="",
                        received_at=row.responded_at or row.dispatched_at,
                    ),
                )
        self._trackers[workflow_id] = tracker
        return tracker

    def register_workflow_dispatch(
        self,
        workflow_id: str,
        dispatches: Sequence[Dict[str, object]],
    ) -> WorkflowTracker:
        if not workflow_id:
            raise ValueError("workflow_id is required to register dispatches")

        tracker = self._ensure_tracker(workflow_id)
        records: List[EmailDispatchRecord] = []
        repo_rows: List[WorkflowDispatchRow] = []

        for payload in dispatches:
            unique_id = str(payload.get("unique_id") or uuid.uuid4().hex)
            supplier_id = payload.get("supplier_id")
            supplier_email = payload.get("supplier_email")
            message_id = payload.get("message_id")
            subject = payload.get("subject")
            dispatched_at = payload.get("dispatched_at")
            raw_thread_headers = (
                payload.get("thread_headers") if isinstance(payload.get("thread_headers"), dict) else {}
            )
            if isinstance(dispatched_at, datetime):
                dispatched_dt = dispatched_at if dispatched_at.tzinfo else dispatched_at.replace(tzinfo=timezone.utc)
            else:
                dispatched_dt = self._now()

            record = EmailDispatchRecord(
                unique_id=unique_id,
                supplier_id=str(supplier_id) if supplier_id else None,
                supplier_email=str(supplier_email) if supplier_email else None,
                message_id=str(message_id) if message_id else None,
                subject=str(subject) if subject else None,
                thread_headers={
                    str(k): _normalise_thread_header(v) for k, v in raw_thread_headers.items()
                },
                dispatched_at=dispatched_dt,
            )
            records.append(record)
            repo_rows.append(
                WorkflowDispatchRow(
                    workflow_id=workflow_id,
                    unique_id=unique_id,
                    supplier_id=record.supplier_id,
                    supplier_email=record.supplier_email,
                    message_id=record.message_id,
                    subject=record.subject,
                    dispatched_at=dispatched_dt,
                    responded_at=None,
                    response_message_id=None,
                    matched=False,
                )
            )

        tracker.register_dispatches(records)
        tracking_repo.record_dispatches(workflow_id=workflow_id, dispatches=repo_rows)
        return tracker

    def _fetch_emails(self, since: datetime) -> List[EmailResponse]:
        if self._fetcher:
            return self._fetcher(since=since)

        if not all([self._imap_host, self._imap_username, self._imap_password]):
            raise RuntimeError("IMAP credentials must be supplied when no custom fetcher is provided")

        return _default_fetcher(
            host=self._imap_host,
            username=self._imap_username,
            password=self._imap_password,
            mailbox=self._mailbox,
            since=since,
            port=self._imap_port or 993,
            use_ssl=True if self._imap_use_ssl is None else self._imap_use_ssl,
            login=self._imap_login,
        )

    def _match_responses(
        self, tracker: WorkflowTracker, responses: Iterable[EmailResponse]
    ) -> List[str]:
        matched: List[str] = []
        for email in responses:
            matched_id: Optional[str] = None
            best_score = 0.0
            for unique_id, dispatch in tracker.email_records.items():
                if unique_id in tracker.matched_responses:
                    continue
                score = _calculate_match_score(dispatch, email)
                if score > best_score:
                    matched_id = unique_id
                    best_score = score
            if matched_id and best_score >= self.match_threshold:
                logger.debug(
                    "Matched response for workflow=%s unique_id=%s score=%.2f",
                    tracker.workflow_id,
                    matched_id,
                    best_score,
                )
                tracker.record_response(matched_id, email)
                tracking_repo.mark_response(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    responded_at=email.received_at,
                    response_message_id=email.message_id,
                )
                supplier_response_repo.insert_response(
                    SupplierResponseRow(
                        workflow_id=tracker.workflow_id,
                        unique_id=matched_id,
                        supplier_id=email.supplier_id,
                        response_text=email.body,
                        received_time=email.received_at,
                    )
                )
                matched.append(matched_id)
        return matched

    def wait_and_collect_responses(self, workflow_id: str) -> Dict[str, object]:
        tracker = self._ensure_tracker(workflow_id)
        if tracker.dispatched_count == 0:
            return {
                "workflow_id": workflow_id,
                "complete": True,
                "dispatched_count": 0,
                "responded_count": 0,
                "matched_responses": {},
            }

        if tracker.last_dispatched_at:
            target = tracker.last_dispatched_at + timedelta(seconds=self.dispatch_wait_seconds)
            now = self._now()
            if target > now:
                wait_time = (target - now).total_seconds()
                logger.info(
                    "Waiting %.1f seconds before polling IMAP for workflow %s",
                    wait_time,
                    workflow_id,
                )
                self._sleep(wait_time)

        attempts = 0
        since = tracker.last_dispatched_at or (self._now() - timedelta(hours=4))
        while attempts < self.max_poll_attempts and not tracker.all_responded:
            responses = self._fetch_emails(since)
            matched_ids = self._match_responses(tracker, responses)
            if matched_ids:
                self._process_agents(tracker)
            if tracker.all_responded:
                break
            attempts += 1
            self._sleep(self.poll_interval_seconds)

        complete = tracker.all_responded
        result = {
            "workflow_id": workflow_id,
            "complete": complete,
            "dispatched_count": tracker.dispatched_count,
            "responded_count": tracker.responded_count,
            "matched_responses": tracker.matched_responses,
        }

        self._process_agents(tracker)

        return result

    def _process_agents(self, tracker: WorkflowTracker) -> None:
        if not self.supplier_agent:
            return

        pending_rows = supplier_response_repo.fetch_pending(workflow_id=tracker.workflow_id)
        processed_ids: List[str] = []
        for row in pending_rows:
            matched_response = tracker.matched_responses.get(row.get("unique_id"))
            context = AgentContext(
                workflow_id=tracker.workflow_id,
                agent_id="EmailWatcherV2",
                user_id="system",
                input_data={
                    "message": row.get("response_text", ""),
                    "email_headers": {
                        "message_id": matched_response.message_id if matched_response else None,
                        "subject": matched_response.subject if matched_response else None,
                        "from": matched_response.from_address if matched_response else None,
                        "workflow_id": tracker.workflow_id,
                        "unique_id": row.get("unique_id"),
                        "supplier_id": row.get("supplier_id"),
                        "price": row.get("price"),
                        "lead_time": row.get("lead_time"),
                        "received_time": row.get("received_time"),
                    },
                },
            )
            try:
                output: AgentOutput = self.supplier_agent.execute(context)
            except Exception:
                logger.exception("SupplierInteractionAgent failed for workflow %s", tracker.workflow_id)
                continue

            processed_ids.append(row.get("unique_id"))

            if (
                output
                and isinstance(output, AgentOutput)
                and output.status == AgentStatus.SUCCESS
                and output.next_agents
                and "NegotiationAgent" in output.next_agents
                and self.negotiation_agent is not None
            ):
                try:
                    neg_context = AgentContext(
                        workflow_id=tracker.workflow_id,
                        agent_id="NegotiationAgent",
                        user_id="system",
                        input_data=output.data,
                    )
                    self.negotiation_agent.execute(neg_context)
                except Exception:
                    logger.exception("NegotiationAgent failed for workflow %s", tracker.workflow_id)

        supplier_response_repo.delete_responses(workflow_id=tracker.workflow_id, unique_ids=processed_ids)
