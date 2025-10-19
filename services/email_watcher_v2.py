"""Workflow-centric email watcher implementation (EmailWatcherV2)."""

from __future__ import annotations

import email
import inspect
import imaplib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from repositories import workflow_email_tracking_repo as tracking_repo
from repositories import draft_rfq_emails_repo
from repositories import supplier_response_repo
from repositories.supplier_response_repo import SupplierResponseRow
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from utils.email_tracking import (
    extract_tracking_metadata,
    extract_unique_id_from_body,
    extract_unique_id_from_headers,
)

logger = logging.getLogger(__name__)


def _row_to_dispatch_record(row: WorkflowDispatchRow) -> Optional[EmailDispatchRecord]:
    unique_id = getattr(row, "unique_id", None)
    if unique_id in (None, ""):
        return None
    return EmailDispatchRecord(
        unique_id=str(unique_id),
        supplier_id=getattr(row, "supplier_id", None),
        supplier_email=getattr(row, "supplier_email", None),
        message_id=getattr(row, "message_id", None),
        subject=getattr(row, "subject", None),
        rfq_id=None,
        thread_headers=row.thread_headers or {},
        dispatched_at=row.dispatched_at,
    )


def _sync_tracker_from_rows(tracker: WorkflowTracker, rows: Sequence[WorkflowDispatchRow]) -> None:
    dispatches: List[EmailDispatchRecord] = []
    for row in rows:
        record = _row_to_dispatch_record(row)
        if record is not None:
            dispatches.append(record)
    if dispatches:
        tracker.register_dispatches(dispatches)

    for row in rows:
        if not getattr(row, "matched", False):
            continue
        unique_id = getattr(row, "unique_id", None)
        if not unique_id:
            continue
        tracker.record_response(
            str(unique_id),
            EmailResponse(
                unique_id=str(unique_id),
                supplier_id=getattr(row, "supplier_id", None),
                supplier_email=getattr(row, "supplier_email", None),
                from_address=None,
                message_id=getattr(row, "response_message_id", None),
                subject=None,
                body="",
                received_at=getattr(row, "responded_at", None)
                or getattr(row, "dispatched_at", datetime.now(timezone.utc)),
            ),
        )


@dataclass
class EmailDispatchRecord:
    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    rfq_id: Optional[str] = None
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
    workflow_id: Optional[str] = None
    rfq_id: Optional[str] = None


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
    expected_unique_ids: Set[str] = field(default_factory=set)
    expected_dispatch_count: Optional[int] = None
    pending_unique_ids: Set[str] = field(default_factory=set)

    def register_dispatches(self, dispatches: Iterable[EmailDispatchRecord]) -> None:
        for dispatch in dispatches:
            if not dispatch.unique_id:
                continue
            key = str(dispatch.unique_id).strip()
            if not key:
                continue
            self.email_records[key] = dispatch
            if dispatch.dispatched_at and (
                self.last_dispatched_at is None or dispatch.dispatched_at > self.last_dispatched_at
            ):
                self.last_dispatched_at = dispatch.dispatched_at
        self.refresh_dispatch_status()

    def record_response(self, unique_id: str, response: EmailResponse) -> None:
        if unique_id not in self.email_records:
            return
        if unique_id in self.matched_responses:
            return
        self.matched_responses[unique_id] = response
        self.responded_count = len(self.matched_responses)
        self.all_responded = self.responded_count >= self.dispatched_count > 0

    def set_expected_count(self, count: Optional[int]) -> None:
        if count is None:
            return
        try:
            numeric = int(count)
        except Exception:
            return
        if numeric < 0:
            numeric = 0
        if self.expected_dispatch_count is None or numeric > self.expected_dispatch_count:
            self.expected_dispatch_count = numeric
        self.refresh_dispatch_status()

    def set_expected_unique_ids(self, unique_ids: Iterable[str]) -> None:
        updated = False
        for raw in unique_ids:
            if raw in (None, ""):
                continue
            text = str(raw).strip()
            if not text:
                continue
            if text not in self.expected_unique_ids:
                self.expected_unique_ids.add(text)
                updated = True
        if updated:
            if (
                self.expected_dispatch_count is None
                or len(self.expected_unique_ids) > self.expected_dispatch_count
            ):
                self.expected_dispatch_count = len(self.expected_unique_ids)
        self.refresh_dispatch_status()

    def expected_dispatch_total(self) -> int:
        if self.expected_unique_ids:
            return len(self.expected_unique_ids)
        if self.expected_dispatch_count:
            return max(int(self.expected_dispatch_count), 0)
        return 0

    def missing_expected_dispatches(self) -> Set[str]:
        if not self.expected_unique_ids:
            return set()
        recorded = set(self.email_records.keys())
        return {uid for uid in self.expected_unique_ids if uid not in recorded}

    def set_pending_unique_ids(self, unique_ids: Iterable[str]) -> None:
        pending: Set[str] = set()
        for raw in unique_ids:
            if raw in (None, ""):
                continue
            text = str(raw).strip()
            if text:
                pending.add(text)
        self.pending_unique_ids = pending
        self.refresh_dispatch_status()

    def _refresh_dispatch_flags(self) -> None:
        expected_total = self.expected_dispatch_total()
        if self.pending_unique_ids:
            self.all_dispatched = False
            return
        if expected_total <= 0:
            self.all_dispatched = self.dispatched_count > 0
            return
        missing = self.missing_expected_dispatches()
        if missing:
            self.all_dispatched = False
            return
        self.all_dispatched = self.dispatched_count >= expected_total

    def refresh_dispatch_status(self) -> None:
        self.dispatched_count = len(self.email_records)
        self._refresh_dispatch_flags()


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
    header_map = {
        key: message.get_all(key, failobj=[])
        for key in ("X-Procwise-Unique-Id", "X-Procwise-Unique-ID", "X-Procwise-Uid")
    }
    unique_id = extract_unique_id_from_headers(header_map)
    if not unique_id:
        fallback_header = message.get("X-Procwise-Unique-Id")
        if fallback_header:
            unique_id = str(fallback_header).strip()
    body_unique = extract_unique_id_from_body(body)
    if body_unique and not unique_id:
        unique_id = body_unique
    metadata = extract_tracking_metadata(body)
    supplier_id = metadata.supplier_id if metadata else None
    if metadata and not unique_id:
        unique_id = metadata.unique_id

    header_unique_id = (message.get("X-Procwise-Unique-Id") or "").strip()
    if header_unique_id and not unique_id:
        unique_id = header_unique_id

    header_supplier_id = (message.get("X-Procwise-Supplier-Id") or "").strip()
    if header_supplier_id and not supplier_id:
        supplier_id = header_supplier_id

    workflow_id = metadata.workflow_id if metadata else None
    header_workflow_id = (message.get("X-Procwise-Workflow-Id") or "").strip()
    if header_workflow_id and not workflow_id:
        workflow_id = header_workflow_id

    rfq_id = (message.get("X-Procwise-RFQ-ID") or "").strip() or None

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
        workflow_id=workflow_id,
        rfq_id=rfq_id,
    )


def _normalise_thread_header(value) -> Sequence[str]:
    if value in (None, ""):
        return tuple()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v).strip("<> ") for v in value if v)
    return (str(value).strip("<> "),)


def _normalise_email_address(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    _, address = parseaddr(str(value))
    address = address.strip().lower()
    return address or None


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


def _score_to_confidence(score: float) -> Decimal:
    """Convert a floating point score into a quantised confidence value.

    The score produced by the matcher is normalised into the range 0-1 so it can
    be persisted as a numeric confidence value for downstream processing.  Any
    unexpected inputs are treated defensively and clamped into range before
    quantisation to two decimal places.
    """

    try:
        if score != score:  # NaN check
            score = 0.0
    except Exception:  # pragma: no cover - defensive
        score = 0.0

    normalised = max(0.0, min(float(score), 1.0))
    decimal_score = Decimal(str(normalised))
    return decimal_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _calculate_match_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> float:
    score = 0.0

    if email_response.unique_id and email_response.unique_id == dispatch.unique_id:
        return 1.0

    if (
        email_response.supplier_id
        and dispatch.supplier_id
        and email_response.supplier_id == dispatch.supplier_id
    ):
        score += 0.65

    thread_ids = set(dispatch.thread_headers.get("references", ())) | set(
        dispatch.thread_headers.get("in_reply_to", ())
    )
    if dispatch.message_id:
        thread_ids.add(dispatch.message_id)

    reply_headers = set(email_response.in_reply_to) | set(email_response.references)
    if dispatch.message_id and dispatch.message_id in reply_headers:
        score += 0.8
    elif thread_ids & reply_headers:
        score += 0.8

    if dispatch.supplier_email and email_response.from_address:
        dispatch_email = email.utils.parseaddr(str(dispatch.supplier_email))[1].lower()
        response_email = email.utils.parseaddr(str(email_response.from_address))[1].lower()
        if dispatch_email and response_email:
            if dispatch_email == response_email:
                score += 0.6
            else:
                dispatch_domain = dispatch_email.split("@")[-1]
                response_domain = response_email.split("@")[-1]
                if dispatch_domain and response_domain and dispatch_domain == response_domain:
                    score += 0.35

    if dispatch.subject and email_response.subject:
        normalised_subject = dispatch.subject.lower()
        if normalised_subject in email_response.subject.lower():
            score += 0.5

    return score


class EmailWatcherV2:
    """Workflow-aware watcher coordinating dispatch/response tracking."""

    def __init__(
        self,
        *,
        supplier_agent: Optional[SupplierInteractionAgent] = None,
        supplier_agent_factory: Optional[
            Callable[[str], SupplierInteractionAgent]
        ] = None,
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
    ) -> None:
        self._default_supplier_agent = supplier_agent
        self._supplier_agent_factory = supplier_agent_factory
        self._supplier_agent_signature = None
        if supplier_agent_factory is not None:
            try:
                self._supplier_agent_signature = inspect.signature(supplier_agent_factory)
            except (TypeError, ValueError):
                self._supplier_agent_signature = None
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
        self._workflow_supplier_agents: Dict[str, SupplierInteractionAgent] = {}

        tracking_repo.init_schema()
        supplier_response_repo.init_schema()

    def _ensure_tracker(self, workflow_id: str) -> WorkflowTracker:
        tracker = self._trackers.get(workflow_id)
        if tracker is not None:
            return tracker
        tracker = WorkflowTracker(workflow_id=workflow_id)
        rows = tracking_repo.load_workflow_rows(workflow_id=workflow_id)
        if rows:
            _sync_tracker_from_rows(tracker, rows)
        self._trackers[workflow_id] = tracker
        return tracker

    def attach_supplier_agent(
        self, workflow_id: str, agent: SupplierInteractionAgent
    ) -> None:
        key = (workflow_id or "").strip()
        if not key or agent is None:
            return
        self._workflow_supplier_agents[key] = agent

    def _create_supplier_agent(
        self, workflow_id: str
    ) -> Optional[SupplierInteractionAgent]:
        factory = self._supplier_agent_factory
        if factory is None:
            return None
        try:
            if self._supplier_agent_signature is not None:
                if len(self._supplier_agent_signature.parameters) == 0:
                    return factory()  # type: ignore[misc]
                return factory(workflow_id)
            return factory(workflow_id)
        except TypeError:
            try:
                return factory()  # type: ignore[misc]
            except Exception:
                logger.exception(
                    "Supplier agent factory failed for workflow %s", workflow_id
                )
                return None
        except Exception:
            logger.exception(
                "Supplier agent factory failed for workflow %s", workflow_id
            )
            return None

    def _resolve_supplier_agent(
        self, workflow_id: str, *, create: bool = True
    ) -> Optional[SupplierInteractionAgent]:
        key = (workflow_id or "").strip()
        if key and key in self._workflow_supplier_agents:
            return self._workflow_supplier_agents[key]
        agent: Optional[SupplierInteractionAgent] = None
        if create and key:
            agent = self._create_supplier_agent(key)
            if agent is not None:
                self._workflow_supplier_agents[key] = agent
                return agent
        if self._default_supplier_agent is not None:
            return self._default_supplier_agent
        return None

    def _refresh_tracker_from_repo(self, tracker: WorkflowTracker) -> None:
        try:
            rows = tracking_repo.load_workflow_rows(workflow_id=tracker.workflow_id)
        except Exception:
            logger.exception(
                "Failed to refresh dispatch records for workflow %s",
                tracker.workflow_id,
            )
            return
        if rows:
            _sync_tracker_from_rows(tracker, rows)

    def _load_dispatch_expectations(self, tracker: WorkflowTracker) -> None:
        try:
            (
                draft_ids,
                supplier_map,
                last_dispatched_at,
                pending_unique_ids,
            ) = draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch_with_pending(
                workflow_id=tracker.workflow_id
            )
        except AttributeError:
            try:
                draft_ids, supplier_map, last_dispatched_at = (
                    draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
                        workflow_id=tracker.workflow_id
                    )
                )
                pending_unique_ids = set()
            except Exception:
                logger.debug(
                    "Unable to resolve draft expectations for workflow=%s", tracker.workflow_id,
                    exc_info=True,
                )
                return
        except Exception:
            logger.debug(
                "Unable to resolve draft expectations for workflow=%s", tracker.workflow_id,
                exc_info=True,
            )
            return

        if draft_ids:
            tracker.set_expected_unique_ids(draft_ids)
        if supplier_map:
            for uid, supplier in supplier_map.items():
                key = str(uid).strip() if uid else ""
                if not key:
                    continue
                record = tracker.email_records.get(key)
                if record and not record.supplier_id and supplier:
                    record.supplier_id = str(supplier)
        if last_dispatched_at:
            if (
                tracker.last_dispatched_at is None
                or last_dispatched_at > tracker.last_dispatched_at
            ):
                tracker.last_dispatched_at = last_dispatched_at
        tracker.set_pending_unique_ids(pending_unique_ids or set())

    def _await_pending_dispatches(self, tracker: WorkflowTracker) -> None:
        attempts = 0
        while True:
            self._load_dispatch_expectations(tracker)
            expected_total = tracker.expected_dispatch_total()
            missing = tracker.missing_expected_dispatches()
            pending_unsent = set(tracker.pending_unique_ids)
            if expected_total <= 0 and not missing and not pending_unsent:
                tracker.refresh_dispatch_status()
                return

            if attempts >= self.max_poll_attempts:
                logger.warning(
                    "Timed out waiting for remaining email dispatches for workflow=%s (missing=%s pending_unsent=%s)",
                    tracker.workflow_id,
                    list(sorted(missing))[:5],
                    list(sorted(pending_unsent))[:5],
                )
                tracker.refresh_dispatch_status()
                return

            logger.info(
                "Deferring supplier response polling until all emails dispatched for workflow=%s (missing=%s pending_unsent=%s)",
                tracker.workflow_id,
                list(sorted(missing))[:5],
                list(sorted(pending_unsent))[:5],
            )
            attempts += 1
            self._sleep(self.poll_interval_seconds)
            self._refresh_tracker_from_repo(tracker)

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
            rfq_id = payload.get("rfq_id")
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
                rfq_id=str(rfq_id) if rfq_id else None,
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
                    thread_headers={
                        key: list(value) for key, value in record.thread_headers.items()
                    }
                    if record.thread_headers
                    else None,
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
    ) -> List[SupplierResponseRow]:
        matched_rows: List[SupplierResponseRow] = []
        for email in responses:
            matched_id: Optional[str] = None
            best_score = 0.0
            best_dispatch: Optional[EmailDispatchRecord] = None
            for unique_id, dispatch in tracker.email_records.items():
                if unique_id in tracker.matched_responses:
                    continue
                score = _calculate_match_score(dispatch, email)
                if score > best_score:
                    matched_id = unique_id
                    best_score = score
                    best_dispatch = dispatch
            if (not matched_id or best_score < self.match_threshold) and email.supplier_id:
                supplier_matches = [
                    uid
                    for uid, dispatch in tracker.email_records.items()
                    if uid not in tracker.matched_responses
                    and dispatch.supplier_id
                    and dispatch.supplier_id == email.supplier_id
                ]
                if len(supplier_matches) == 1:
                    matched_id = supplier_matches[0]
                    best_score = max(best_score, self.match_threshold)
                    best_dispatch = tracker.email_records.get(matched_id)
            if not matched_id and len([
                uid for uid in tracker.email_records.keys() if uid not in tracker.matched_responses
            ]) == 1:
                remaining_uid = next(
                    uid
                    for uid in tracker.email_records.keys()
                    if uid not in tracker.matched_responses
                )
                matched_id = remaining_uid
                logger.debug(
                    "Defaulting response assignment for workflow=%s to remaining unique_id=%s",
                    tracker.workflow_id,
                    matched_id,
                )
                best_score = max(best_score, self.match_threshold)
                best_dispatch = tracker.email_records.get(matched_id)

            if matched_id and best_score >= self.match_threshold:
                logger.debug(
                    "Matched response for workflow=%s unique_id=%s score=%.2f",
                    tracker.workflow_id,
                    matched_id,
                    best_score,
                )
                if best_dispatch is None:
                    best_dispatch = tracker.email_records.get(matched_id)
                if best_dispatch is None:
                    logger.warning(
                        "Matched response for workflow=%s unique_id=%s but no dispatch record found",
                        tracker.workflow_id,
                        matched_id,
                    )
                    continue
                tracker.record_response(matched_id, email)
                tracking_repo.mark_response(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    responded_at=email.received_at,
                    response_message_id=email.message_id,
                )
                supplier_email = (
                    email.supplier_email
                    or _normalise_email_address(email.from_address)
                    or best_dispatch.supplier_email
                )
                supplier_id = email.supplier_id or best_dispatch.supplier_id
                response_time: Optional[Decimal] = None
                if best_dispatch.dispatched_at and email.received_at:
                    try:
                        delta_seconds = (email.received_at - best_dispatch.dispatched_at).total_seconds()
                        if delta_seconds >= 0:
                            response_time = Decimal(str(delta_seconds))
                    except Exception:  # pragma: no cover - defensive conversion
                        response_time = None
                response_row = SupplierResponseRow(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    supplier_id=supplier_id,
                    supplier_email=supplier_email,
                    response_text=email.body,
                    received_time=email.received_at,
                    response_time=response_time,
                    response_message_id=email.message_id,
                    response_subject=email.subject,
                    response_from=email.from_address,
                    original_message_id=best_dispatch.message_id,
                    original_subject=best_dispatch.subject,
                    match_confidence=_score_to_confidence(best_score),
                    processed=False,
                )
                supplier_response_repo.insert_response(response_row)
                matched_rows.append(response_row)
            else:
                logger.warning(
                    "Unable to confidently match supplier email for workflow=%s message_id=%s (best_score=%.2f)",
                    tracker.workflow_id,
                    email.message_id,
                    best_score,
                )
        return matched_rows

    def wait_and_collect_responses(self, workflow_id: str) -> Dict[str, object]:
        tracker = self._ensure_tracker(workflow_id)
        self._load_dispatch_expectations(tracker)
        self._await_pending_dispatches(tracker)
        self._refresh_tracker_from_repo(tracker)

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
            self._refresh_tracker_from_repo(tracker)
            poll_since = tracker.last_dispatched_at or since
            responses = self._fetch_emails(poll_since)
            if responses:
                self._refresh_tracker_from_repo(tracker)
            matched_rows = self._match_responses(tracker, responses)
            if matched_rows:
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

        if (
            self._resolve_supplier_agent(workflow_id, create=False) is not None
            and tracker.matched_responses
        ):
            supplier_response_repo.delete_responses(
                workflow_id=workflow_id,
                unique_ids=list(tracker.matched_responses.keys()),
            )

        return result

    def _process_agents(self, tracker: WorkflowTracker) -> None:
        pending_rows = supplier_response_repo.fetch_pending(workflow_id=tracker.workflow_id)
        if not pending_rows:
            return

        supplier_agent = self._resolve_supplier_agent(tracker.workflow_id)
        if supplier_agent is None:
            logger.warning(
                "No SupplierInteractionAgent configured for workflow %s; leaving %d responses pending",
                tracker.workflow_id,
                len(pending_rows),
            )
            return

        processed_ids: List[str] = []
        for row in pending_rows:
            unique_id = row.get("unique_id")
            matched = tracker.matched_responses.get(unique_id) if unique_id else None
            supplier_id = row.get("supplier_id") or (matched.supplier_id if matched else None)
            subject = row.get("response_subject") or (matched.subject if matched else None)
            message_id = row.get("response_message_id") or (matched.message_id if matched else None)
            from_address = row.get("response_from") or (matched.from_address if matched else None)
            body_text = row.get("response_body") or (matched.body if matched else "")
            workflow_id = matched.workflow_id if matched and matched.workflow_id else tracker.workflow_id
            rfq_id = matched.rfq_id if matched and matched.rfq_id else None
            supplier_email = row.get("supplier_email") or (matched.supplier_email if matched else None)

            input_payload = {
                "message": body_text or "",
                "subject": subject,
                "message_id": message_id,
                "from_address": from_address,
                "workflow_id": workflow_id,
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "supplier_email": supplier_email,
            }
            if rfq_id:
                input_payload["rfq_id"] = rfq_id

            headers = {
                "message_id": message_id,
                "subject": subject,
                "from": from_address,
                "workflow_id": workflow_id,
                "unique_id": unique_id,
                "supplier_id": supplier_id,
            }
            if rfq_id:
                headers["rfq_id"] = rfq_id

            context = AgentContext(
                workflow_id=workflow_id,
                agent_id="EmailWatcherV2",
                user_id="system",
                input_data={**input_payload, "email_headers": headers},
            )

            try:
                wait_result: AgentOutput = supplier_agent.execute(context)
            except Exception:
                logger.exception("SupplierInteractionAgent failed for workflow %s", workflow_id)
                continue

            batch_ready = bool(wait_result.data.get("batch_ready"))
            if not batch_ready:
                logger.info(
                    "Supplier batch not complete for workflow %s; waiting for remaining responses",
                    tracker.workflow_id,
                )
                return

            responses = wait_result.data.get("supplier_responses") or []
            expected = wait_result.data.get("expected_responses") or 0
            collected = wait_result.data.get("collected_responses") or len(responses)
            if expected and collected != expected:
                logger.warning(
                    "Supplier batch size mismatch for workflow %s (expected=%s collected=%s)",
                    tracker.workflow_id,
                    expected,
                    collected,
                )
                return

            ids_from_agent = [
                uid
                for uid in wait_result.data.get("unique_ids", [])
                if isinstance(uid, str) and uid
            ]
            if not ids_from_agent:
                ids_from_agent = [
                    response.get("unique_id")
                    for response in responses
                    if isinstance(response, dict) and response.get("unique_id")
                ]
            if unique_id and unique_id not in ids_from_agent:
                ids_from_agent.append(unique_id)
            processed_ids.extend(ids_from_agent)

            negotiation_payload = dict(wait_result.data)
            negotiation_payload.setdefault("supplier_responses", responses)
            negotiation_payload.setdefault("supplier_responses_batch", responses)
            negotiation_payload.setdefault("supplier_responses_count", len(responses))
            negotiation_payload.setdefault("negotiation_batch", True)
            negotiation_payload.setdefault(
                "batch_metadata",
                {
                    "expected": expected,
                    "collected": collected,
                    "ready": True,
                },
            )

            if self.negotiation_agent is not None:
                try:
                    neg_context = AgentContext(
                        workflow_id=tracker.workflow_id,
                        agent_id="NegotiationAgent",
                        user_id="system",
                        input_data=negotiation_payload,
                    )
                    self.negotiation_agent.execute(neg_context)
                except Exception:
                    logger.exception("NegotiationAgent failed for workflow %s", tracker.workflow_id)

            if ids_from_agent:
                supplier_response_repo.delete_responses(
                    workflow_id=tracker.workflow_id, unique_ids=ids_from_agent
                )

        if processed_ids:
            supplier_response_repo.delete_responses(
                workflow_id=tracker.workflow_id, unique_ids=processed_ids
            )
        else:
            fallback_ids = [
                row.get("unique_id")
                for row in pending_rows
                if isinstance(row, dict) and row.get("unique_id")
            ]
            if fallback_ids:
                supplier_response_repo.delete_responses(
                    workflow_id=tracker.workflow_id, unique_ids=fallback_ids
                )
