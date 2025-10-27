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
from decimal import Decimal, ROUND_HALF_UP
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from repositories import (
    supplier_interaction_repo,
    supplier_response_repo,
    workflow_email_tracking_repo as tracking_repo,
)
from repositories.supplier_interaction_repo import SupplierInteractionRow
from repositories.supplier_response_repo import SupplierResponseRow
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from utils.email_tracking import (
    extract_tracking_metadata,
    extract_unique_id_from_body,
    extract_unique_id_from_headers,
)
from utils import email_tracking
from services.supplier_response_coordinator import get_supplier_response_coordinator

logger = logging.getLogger(__name__)


@dataclass
class EmailDispatchRecord:
    unique_id: str
    dispatch_key: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    rfq_id: Optional[str] = None
    thread_headers: Dict[str, str] = field(default_factory=dict)
    dispatched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    round_number: Optional[int] = None


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
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    in_reply_to: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)
    workflow_id: Optional[str] = None
    rfq_id: Optional[str] = None


@dataclass
class WorkflowTracker:
    workflow_id: str
    dispatched_count: int = 0
    responded_count: int = 0
    expected_responses: int = 0
    email_records: Dict[str, List[EmailDispatchRecord]] = field(default_factory=dict)
    matched_responses: Dict[str, EmailResponse] = field(default_factory=dict)
    response_history: Dict[str, List[EmailResponse]] = field(default_factory=dict)
    responded_unique_ids: Set[str] = field(default_factory=set)
    rfq_index: Dict[str, List[str]] = field(default_factory=dict)
    all_dispatched: bool = False
    all_responded: bool = False
    last_dispatched_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    last_response_at: Optional[datetime] = None
    round_index: Dict[str, Optional[int]] = field(default_factory=dict)
    completion_logged: bool = False

    def register_dispatches(self, dispatches: Iterable[EmailDispatchRecord]) -> None:
        for dispatch in dispatches:
            bucket = self.email_records.setdefault(dispatch.unique_id, [])
            bucket.append(dispatch)
            bucket.sort(key=lambda item: item.dispatched_at or datetime.min)
            if dispatch.round_number is not None:
                self.round_index[dispatch.unique_id] = dispatch.round_number
            if dispatch.rfq_id:
                normalised = _normalise_identifier(dispatch.rfq_id)
                if normalised:
                    self.rfq_index.setdefault(normalised, []).append(dispatch.unique_id)
            if dispatch.dispatched_at and (
                self.last_dispatched_at is None or dispatch.dispatched_at > self.last_dispatched_at
            ):
                self.last_dispatched_at = dispatch.dispatched_at
            if dispatch.dispatched_at and (
                self.started_at is None or dispatch.dispatched_at < self.started_at
            ):
                self.started_at = dispatch.dispatched_at
        self.dispatched_count = len(self.email_records)
        self.expected_responses = self.dispatched_count
        self.all_dispatched = True
        self.all_responded = self.responded_count >= self.expected_responses > 0

    def record_response(self, unique_id: str, response: EmailResponse) -> None:
        if unique_id not in self.email_records:
            return

        history = self.response_history.setdefault(unique_id, [])
        history.append(response)
        self.matched_responses[unique_id] = response

        if unique_id not in self.responded_unique_ids:
            self.responded_unique_ids.add(unique_id)
            self.responded_count = len(self.responded_unique_ids)
        else:
            # keep counts consistent even if new dispatches arrive later
            self.responded_count = len(self.responded_unique_ids)

        self.last_response_at = response.received_at or self.last_response_at
        self.all_responded = self.responded_count >= self.expected_responses > 0

    def latest_dispatch(self, unique_id: str) -> Optional[EmailDispatchRecord]:
        records = self.email_records.get(unique_id)
        if not records:
            return None
        return records[-1]

    def latest_response(self, unique_id: str) -> Optional[EmailResponse]:
        """Return the most recent response recorded for a dispatch."""
        history = self.response_history.get(unique_id)
        if history:
            return history[-1]
        return self.matched_responses.get(unique_id)

    def elapsed_seconds(self, now: Optional[datetime] = None) -> float:
        """Return elapsed seconds from the first dispatch to ``now``."""

        reference = self.started_at or self.last_dispatched_at
        if reference is None:
            return 0.0
        moment = now or datetime.now(timezone.utc)
        try:
            delta = (moment - reference).total_seconds()
        except Exception:  # pragma: no cover - defensive arithmetic
            return 0.0
        return float(delta) if delta >= 0 else 0.0

    def distinct_rounds(self) -> List[int]:
        rounds = {
            value
            for value in self.round_index.values()
            if isinstance(value, int) and value >= 0
        }
        return sorted(rounds)

    def round_for_unique(self, unique_id: Optional[str]) -> Optional[int]:
        if not unique_id:
            return None
        return self.round_index.get(unique_id)


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


def _extract_plain_text(message: EmailMessage) -> Tuple[str, Optional[str]]:
    plain_text: Optional[str] = None
    html_text: Optional[str] = None
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    content = part.get_content()
                    if content is not None:
                        plain_text = str(content).strip()
                        break
                except Exception:
                    continue
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/html" and "attachment" not in disp:
                try:
                    content = part.get_content()
                    if content is not None:
                        html_text = str(content)
                        break
                except Exception:
                    continue
    else:
        content = message.get_content()
        if content is not None:
            if (message.get_content_type() or "").lower() == "text/html":
                html_text = str(content)
            else:
                plain_text = str(content)
    if plain_text is None and html_text is not None:
        plain_text = html_text
    return (plain_text or "", html_text)


def _parse_email(raw: bytes) -> EmailResponse:
    message = _decode_message(raw)
    subject = message.get("Subject")
    message_id = (message.get("Message-ID") or "").strip("<> ") or None
    from_address = message.get("From")
    plain_body, html_body = _extract_plain_text(message)
    body = plain_body or html_body or ""
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
        body=body,
        body_text=plain_body or body,
        body_html=html_body,
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


def _normalise_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text.upper() or None


def _normalise_email_address(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    _, address = parseaddr(str(value))
    address = address.strip().lower()
    return address or None


def _coerce_round_number(value: Optional[object]) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


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
        max_total_wait_seconds: Optional[int] = None,
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
        baseline_timeout = self.poll_interval_seconds * self.max_poll_attempts * 3
        derived_timeout = max(self.dispatch_wait_seconds, baseline_timeout)
        if max_total_wait_seconds is not None:
            try:
                derived_timeout = max(0, int(max_total_wait_seconds))
            except Exception:  # pragma: no cover - defensive conversion
                derived_timeout = max(self.dispatch_wait_seconds, baseline_timeout)
        self.max_total_wait_seconds = derived_timeout if derived_timeout > 0 else None
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
                    dispatch_key=row.dispatch_key,
                    supplier_id=row.supplier_id,
                    supplier_email=row.supplier_email,
                    message_id=row.message_id,
                    subject=row.subject,
                    thread_headers=row.thread_headers or {},
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
        previous_unique_ids = set(tracker.email_records.keys())
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
            round_number = _coerce_round_number(
                payload.get("round_number")
                or payload.get("round")
                or payload.get("round_num")
            )
            dispatch_key = str(
                payload.get("dispatch_key")
                or payload.get("message_id")
                or uuid.uuid4().hex
            )
            raw_thread_headers = (
                payload.get("thread_headers") if isinstance(payload.get("thread_headers"), dict) else {}
            )
            if isinstance(dispatched_at, datetime):
                dispatched_dt = dispatched_at if dispatched_at.tzinfo else dispatched_at.replace(tzinfo=timezone.utc)
            else:
                dispatched_dt = self._now()

            record = EmailDispatchRecord(
                unique_id=unique_id,
                dispatch_key=dispatch_key,
                supplier_id=str(supplier_id) if supplier_id else None,
                supplier_email=str(supplier_email) if supplier_email else None,
                message_id=str(message_id) if message_id else None,
                subject=str(subject) if subject else None,
                rfq_id=str(rfq_id) if rfq_id else None,
                thread_headers={
                    str(k): _normalise_thread_header(v) for k, v in raw_thread_headers.items()
                },
                dispatched_at=dispatched_dt,
                round_number=round_number,
            )
            records.append(record)
            repo_rows.append(
                WorkflowDispatchRow(
                    workflow_id=workflow_id,
                    unique_id=unique_id,
                    dispatch_key=dispatch_key,
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
        if records:
            batch_size = len(records)
            self._register_expected_with_coordinator(tracker)
            self._log_dispatch_registration(
                tracker,
                newly_added=max(0, tracker.expected_responses - len(previous_unique_ids)),
                batch_size=batch_size,
            )
        return tracker

    def _register_expected_with_coordinator(self, tracker: WorkflowTracker) -> None:
        unique_ids = list(tracker.email_records.keys())
        if not unique_ids:
            return
        try:
            coordinator = get_supplier_response_coordinator()
        except Exception:
            logger.exception(
                "Failed to initialise supplier response coordinator for workflow %s",
                tracker.workflow_id,
            )
            return
        if not coordinator:
            return
        try:
            coordinator.register_expected_responses(
                tracker.workflow_id,
                unique_ids,
                max(len(unique_ids), tracker.expected_responses),
            )
        except Exception:
            logger.exception(
                "Failed to register expected responses with coordinator for workflow %s",
                tracker.workflow_id,
            )

    def _round_fragment(
        self, tracker: WorkflowTracker, unique_id: Optional[str] = None
    ) -> str:
        if unique_id:
            round_number = tracker.round_for_unique(unique_id)
            if round_number is not None:
                return f" round={round_number}"
            return ""
        rounds = tracker.distinct_rounds()
        if rounds:
            joined = ",".join(str(value) for value in rounds)
            return f" rounds={joined}"
        return ""

    def _log_dispatch_registration(
        self,
        tracker: WorkflowTracker,
        *,
        newly_added: int,
        batch_size: int,
    ) -> None:
        elapsed = tracker.elapsed_seconds(self._now())
        round_fragment = self._round_fragment(tracker)
        logger.info(
            "EmailWatcher registered dispatch batch workflow=%s%s batch_size=%d new_unique=%d "
            "expected_total=%d responded=%d elapsed=%.1fs",
            tracker.workflow_id,
            round_fragment,
            batch_size,
            newly_added,
            tracker.expected_responses,
            tracker.responded_count,
            elapsed,
        )

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
            for unique_id, dispatch_list in tracker.email_records.items():
                dispatch = dispatch_list[-1]
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
                    for uid, dispatch_list in tracker.email_records.items()
                    if uid not in tracker.matched_responses
                    and dispatch_list
                    and dispatch_list[-1].supplier_id
                    and dispatch_list[-1].supplier_id == email.supplier_id
                ]
                if len(supplier_matches) == 1:
                    matched_id = supplier_matches[0]
                    best_score = max(best_score, self.match_threshold)
                    best_dispatch = tracker.latest_dispatch(matched_id)
            if (not matched_id or best_score < self.match_threshold) and email.rfq_id:
                normalised_rfq = _normalise_identifier(email.rfq_id)
                if normalised_rfq:
                    rfq_candidates = [
                        uid
                        for uid, dispatch_list in tracker.email_records.items()
                        if uid not in tracker.matched_responses
                        and dispatch_list
                        and dispatch_list[-1].rfq_id
                        and _normalise_identifier(dispatch_list[-1].rfq_id)
                        == normalised_rfq
                    ]
                    if len(rfq_candidates) == 1:
                        matched_id = rfq_candidates[0]
                        best_score = max(best_score, self.match_threshold)
                        best_dispatch = tracker.latest_dispatch(matched_id)

            remaining_unresponded = [
                uid
                for uid in tracker.email_records.keys()
                if uid not in tracker.responded_unique_ids
            ]
            if not matched_id and len(remaining_unresponded) == 1:
                remaining_uid = remaining_unresponded[0]
                matched_id = remaining_uid
                logger.debug(
                    "Defaulting response assignment for workflow=%s to remaining unique_id=%s",
                    tracker.workflow_id,
                    matched_id,
                )
                best_score = max(best_score, self.match_threshold)
                best_dispatch = tracker.latest_dispatch(matched_id)

            if matched_id and best_score >= self.match_threshold:
                logger.debug(
                    "Matched response for workflow=%s unique_id=%s score=%.2f",
                    tracker.workflow_id,
                    matched_id,
                    best_score,
                )
                if best_dispatch is None:
                    best_dispatch = tracker.latest_dispatch(matched_id)
                if best_dispatch is None:
                    logger.warning(
                        "Matched response for workflow=%s unique_id=%s but no dispatch record found",
                        tracker.workflow_id,
                        matched_id,
                    )
                    continue
                tracker.record_response(matched_id, email)
                elapsed = tracker.elapsed_seconds(email.received_at)
                logger.info(
                    "EmailWatcher captured supplier response workflow=%s%s responded=%d/%d elapsed=%.1fs",
                    tracker.workflow_id,
                    self._round_fragment(tracker, matched_id),
                    tracker.responded_count,
                    tracker.expected_responses,
                    elapsed,
                )
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
                    rfq_id=email.rfq_id or best_dispatch.rfq_id,
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

        idle_attempts = 0
        timeout_reached = False
        poll_started_at = self._now()
        since = tracker.last_dispatched_at or (poll_started_at - timedelta(hours=4))
        while not tracker.all_responded:
            now = self._now()
            elapsed = tracker.elapsed_seconds(now)
            runtime_elapsed = max(0.0, (now - poll_started_at).total_seconds())
            if self.max_total_wait_seconds is not None and runtime_elapsed >= self.max_total_wait_seconds:
                timeout_reached = True
                logger.warning(
                    "EmailWatcher timed out waiting for responses workflow=%s%s expected=%d responded=%d elapsed=%.1fs runtime=%.1fs",
                    workflow_id,
                    self._round_fragment(tracker),
                    tracker.expected_responses,
                    tracker.responded_count,
                    elapsed,
                    runtime_elapsed,
                )
                break

            responses = self._fetch_emails(since)
            matched_rows = self._match_responses(tracker, responses)
            if matched_rows:
                self._process_agents(tracker)
            if tracker.all_responded:
                break

            if responses or matched_rows:
                idle_attempts = 0
            else:
                idle_attempts += 1

            logger.info(
                "EmailWatcher poll summary workflow=%s%s expected=%d responded=%d elapsed=%.1fs runtime=%.1fs idle_attempts=%d",
                workflow_id,
                self._round_fragment(tracker),
                tracker.expected_responses,
                tracker.responded_count,
                elapsed,
                runtime_elapsed,
                idle_attempts,
            )

            if idle_attempts >= self.max_poll_attempts:
                logger.info(
                    "EmailWatcher entering idle backoff for workflow=%s%s expected=%d responded=%d elapsed=%.1fs runtime=%.1fs",
                    workflow_id,
                    self._round_fragment(tracker),
                    tracker.expected_responses,
                    tracker.responded_count,
                    elapsed,
                    runtime_elapsed,
                )
                break

            self._sleep(self.poll_interval_seconds)

        complete = tracker.all_responded
        result = {
            "workflow_id": workflow_id,
            "complete": complete,
            "dispatched_count": tracker.dispatched_count,
            "responded_count": tracker.responded_count,
            "matched_responses": tracker.matched_responses,
            "response_history": tracker.response_history,
            "expected_responses": tracker.expected_responses,
            "elapsed_seconds": tracker.elapsed_seconds(self._now()),
            "timeout_reached": timeout_reached,
            "workflow_status": "responses_complete"
            if complete
            else "awaiting_responses",
        }

        self._process_agents(tracker)

        return result

    def _process_agents(self, tracker: WorkflowTracker) -> None:
        if not tracker.all_dispatched or not tracker.all_responded:
            logger.debug(
                "Skipping agent processing for workflow %s until all dispatches and responses are ready",
                tracker.workflow_id,
            )
            return
        if not tracker.completion_logged:
            logger.info(
                "EmailWatcher responses complete workflow=%s%s expected=%d responded=%d elapsed=%.1fs",
                tracker.workflow_id,
                self._round_fragment(tracker),
                tracker.expected_responses,
                tracker.responded_count,
                tracker.elapsed_seconds(self._now()),
            )
            tracker.completion_logged = True
        pending_rows = supplier_response_repo.fetch_pending(workflow_id=tracker.workflow_id)
        if not pending_rows:
            return

        if not tracker.all_dispatched:
            logger.info(
                "Deferring supplier interaction for workflow %s until all dispatches are recorded",
                tracker.workflow_id,
            )
            return

        processed_ids: List[str] = []
        for row in pending_rows:
            unique_id = row.get("unique_id")
            latest = tracker.latest_response(unique_id) if unique_id else None
            supplier_id = row.get("supplier_id") or (latest.supplier_id if latest else None)
            subject = row.get("response_subject") or (latest.subject if latest else None)
            message_id = row.get("response_message_id") or (latest.message_id if latest else None)
            from_address = row.get("response_from") or (latest.from_address if latest else None)
            body_text = row.get("response_body") or (latest.body if latest else "")
            workflow_id = latest.workflow_id if latest and latest.workflow_id else tracker.workflow_id
            rfq_id = latest.rfq_id if latest and latest.rfq_id else None
            supplier_email = row.get("supplier_email") or (latest.supplier_email if latest else None)
            history = tracker.response_history.get(unique_id, []) if unique_id else []

            input_payload = {
                "message": body_text or "",
                "body": body_text or "",
                "subject": subject,
                "message_id": message_id,
                "from_address": from_address,
                "workflow_id": workflow_id,
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "supplier_email": supplier_email,
                "action": "await_workflow_batch",
                "workflow_status": "responses_complete",
            }
            if rfq_id:
                input_payload["rfq_id"] = rfq_id
            if history:
                input_payload.setdefault(
                    "response_history",
                    [
                        {
                            "message_id": item.message_id,
                            "subject": item.subject,
                            "body": item.body,
                            "received_at": item.received_at,
                            "from_address": item.from_address,
                        }
                        for item in history
                    ],
                )

            expected_ids = list(tracker.email_records.keys())
            if expected_ids:
                input_payload.setdefault("expected_unique_ids", expected_ids)
            dispatch_total = getattr(tracker, "dispatched_count", None)
            if dispatch_total:
                input_payload.setdefault("expected_dispatch_count", dispatch_total)
                input_payload.setdefault("expected_email_count", dispatch_total)

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

            if self.supplier_agent is None:
                logger.warning(
                    "No SupplierInteractionAgent configured for workflow %s; skipping response processing",
                    tracker.workflow_id,
                )
                continue

            try:
                wait_result: AgentOutput = self.supplier_agent.execute(context)
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
            negotiation_payload.setdefault("workflow_status", "responses_complete")

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


def generate_unique_email_id(
    workflow_id: str,
    supplier_id: Optional[str],
    *,
    round_number: int = 1,
) -> str:
    """Generate a unique identifier encoding workflow, supplier, and round."""

    supplier_hint = f"{supplier_id or 'anon'}-r{max(1, round_number)}"
    return email_tracking.generate_unique_email_id(workflow_id, supplier_hint)


def embed_unique_id_in_email_body(body: Optional[str], unique_id: str) -> str:
    """Expose :func:`utils.email_tracking.embed_unique_id_in_email_body`."""

    return email_tracking.embed_unique_id_in_email_body(body, unique_id)


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
    """Persist a dispatched email so inbound replies can be reconciled."""

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
        poll_interval_seconds: int = 60,
        email_fetcher: Optional[Callable[..., List[EmailResponse]]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.agent_nick = agent_nick
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self._email_fetcher = email_fetcher
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn

    def _fetch(self, **kwargs) -> List[EmailResponse]:
        if self._email_fetcher is not None:
            return self._email_fetcher(**kwargs)
        return _default_fetcher(**kwargs)

    def _sleep(self) -> None:
        try:
            self._sleep_fn(self.poll_interval_seconds)
        except Exception:  # pragma: no cover - defensive
            time.sleep(self.poll_interval_seconds)

    def _row_to_outbound(self, row: Dict[str, Any]) -> SupplierInteractionRow:
        return SupplierInteractionRow(
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
            in_reply_to=list(row.get("in_reply_to") or []),
            references=list(row.get("references") or row.get("reference_ids") or []),
            rfq_id=row.get("rfq_id"),
            metadata=row.get("metadata"),
        )

    def _resolve_outbound(self, response: EmailResponse) -> Optional[SupplierInteractionRow]:
        row: Optional[Dict[str, Any]] = None
        if response.unique_id:
            row = supplier_interaction_repo.lookup_outbound(response.unique_id)

        if row is None and response.rfq_id:
            rfq_id = response.rfq_id
            candidates = supplier_interaction_repo.find_pending_by_rfq(rfq_id)
            if not candidates and rfq_id.upper() != rfq_id:
                candidates = supplier_interaction_repo.find_pending_by_rfq(rfq_id.upper())
            if len(candidates) == 1:
                row = candidates[0]

        if row is None:
            return None

        try:
            return self._row_to_outbound(row)
        except KeyError:
            logger.warning(
                "Malformed outbound interaction row encountered agent=%s row_keys=%s",
                self.agent_nick,
                sorted(row.keys()),
            )
            return None

    def run_once(self, workflow_id: str) -> List[Dict[str, object]]:
        responses = self._fetch(workflow_id=workflow_id)
        if not responses:
            return []

        supplier_interaction_repo.init_schema()
        stored: List[Dict[str, object]] = []
        for response in responses:
            stored_record = self._handle_response(response)
            if stored_record:
                stored.append(stored_record)
        return stored

    def _handle_response(self, response: EmailResponse) -> Optional[Dict[str, object]]:
        outbound = self._resolve_outbound(response)
        if outbound is None:
            logger.warning(
                "Could not match supplier response to dispatch agent=%s unique_id=%s rfq_id=%s",
                self.agent_nick,
                response.unique_id,
                response.rfq_id,
            )
            return None

        supplier_interaction_repo.record_inbound_response(
            outbound=outbound,
            message_id=response.message_id or uuid.uuid4().hex,
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

        return {
            "unique_id": outbound.unique_id,
            "supplier_id": outbound.supplier_id,
            "workflow_id": outbound.workflow_id,
            "status": "received",
            "subject": response.subject,
        }

    def run_continuously(
        self,
        workflow_id: str,
        *,
        until: Optional[datetime] = None,
        timeout_minutes: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        collected: List[Dict[str, object]] = []
        start = self._now_fn()
        while True:
            collected.extend(self.run_once(workflow_id))
            if collected:
                return collected
            now = self._now_fn()
            if until and now >= until:
                return collected
            if timeout_minutes and (now - start) >= timedelta(minutes=timeout_minutes):
                return collected
            self._sleep()