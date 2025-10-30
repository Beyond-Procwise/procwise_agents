"""Unified email watcher module for ProcWise."""
from __future__ import annotations

import asyncio
import imaplib
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from agents.base_agent import AgentContext, AgentOutput
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
from utils import email_tracking
from utils.email_tracking import (
    extract_tracking_metadata,
    extract_unique_id_from_body,
    extract_unique_id_from_headers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmailDispatchRecord:
    """Representation of an outbound message tracked by the watcher."""

    unique_id: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    rfq_id: Optional[str] = None
    dispatch_id: Optional[str] = None
    round_number: Optional[int] = None
    workflow_id: Optional[str] = None
    thread_headers: Dict[str, Sequence[str]] = field(default_factory=dict)
    dispatched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class EmailResponse:
    """Parsed inbound supplier response."""

    unique_id: Optional[str]
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    from_address: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    body: str
    received_at: datetime
    body_html: Optional[str] = None
    in_reply_to: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)
    workflow_id: Optional[str] = None
    rfq_id: Optional[str] = None
    round_number: Optional[int] = None
    raw_headers: Dict[str, Sequence[str]] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowTracker:
    """Tracks dispatches and matched responses for a workflow."""

    workflow_id: str
    dispatched_count: int = 0
    responded_count: int = 0
    email_records: Dict[str, EmailDispatchRecord] = field(default_factory=dict)
    matched_responses: Dict[str, EmailResponse] = field(default_factory=dict)
    rfq_index: Dict[str, List[str]] = field(default_factory=dict)
    all_dispatched: bool = False
    all_responded: bool = False
    last_dispatched_at: Optional[datetime] = None

    def register_dispatches(self, dispatches: Iterable[EmailDispatchRecord]) -> None:
        for dispatch in dispatches:
            self.email_records[dispatch.unique_id] = dispatch
            if dispatch.rfq_id:
                normalised = _normalise_identifier(dispatch.rfq_id)
                if normalised:
                    self.rfq_index.setdefault(normalised, []).append(dispatch.unique_id)
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


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


def _strip_html_tags(html: str) -> str:
    parser = _BodyHTMLStripper()
    parser.feed(html)
    parser.close()
    return parser.text


def _extract_bodies(message: EmailMessage) -> tuple[str, Optional[str]]:
    text_content: Optional[str] = None
    html_content: Optional[str] = None

    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    candidate = part.get_content()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Failed to extract text/plain content: %s", exc)
                    continue
                if isinstance(candidate, str) and candidate.strip():
                    text_content = candidate.strip()
                    break
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/html" and "attachment" not in disp:
                try:
                    candidate_html = part.get_content()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Failed to extract text/html content: %s", exc)
                    continue
                if isinstance(candidate_html, str) and candidate_html.strip():
                    html_content = candidate_html
                    if text_content:
                        break
        if text_content is None and html_content:
            text_content = _strip_html_tags(html_content)
    else:
        payload = message.get_content()
        if isinstance(payload, str):
            ctype = message.get_content_type()
            if ctype == "text/html":
                html_content = payload
                text_content = _strip_html_tags(payload)
            else:
                text_content = payload.strip()

    return text_content or "", html_content


class _BodyHTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - exercised in tests
        if data:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in self._parts if part.strip())


def parse_email_bytes(raw: bytes) -> EmailResponse:
    """Parse raw RFC822 bytes into :class:`EmailResponse`."""

    message = _decode_message(raw)
    body_text, body_html = _extract_bodies(message)
    thread_headers = _extract_thread_ids(message)

    raw_headers: Dict[str, Sequence[str]] = {}
    for key in message.keys():
        values = message.get_all(key, failobj=[])
        if not values:
            continue
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if cleaned:
            raw_headers[key] = tuple(cleaned)

    def _header_lookup(name: str) -> Optional[str]:
        lowered = name.lower()
        for header_name, values in raw_headers.items():
            if header_name.lower() == lowered and values:
                return values[-1]
        return None

    header_candidates = {
        key: raw_headers.get(key, tuple())
        for key in (
            "X-ProcWise-Unique-ID",
            "X-ProcWise-Unique-Id",
            "X-ProcWise-Uid",
            "X-Procwise-Unique-Id",
        )
    }

    unique_id = extract_unique_id_from_headers(header_candidates)
    if not unique_id:
        fallback_header = _header_lookup("X-ProcWise-Unique-ID")
        if fallback_header:
            unique_id = fallback_header.strip()
    body_unique = extract_unique_id_from_body(body_text)
    if body_unique and not unique_id:
        unique_id = body_unique

    metadata = extract_tracking_metadata(body_text)
    supplier_id = metadata.supplier_id if metadata else None
    if metadata and not unique_id:
        unique_id = metadata.unique_id

    header_supplier_id = _header_lookup("X-ProcWise-Supplier-ID")
    if header_supplier_id and not supplier_id:
        supplier_id = header_supplier_id.strip()

    workflow_id = metadata.workflow_id if metadata else None
    header_workflow_id = _header_lookup("X-ProcWise-Workflow-ID")
    if header_workflow_id:
        workflow_id = header_workflow_id.strip()

    round_number: Optional[int] = None
    header_round = _header_lookup("X-ProcWise-Round")
    if header_round:
        try:
            round_number = int(header_round)
        except Exception:  # pragma: no cover - defensive
            round_number = None
    if round_number is None and metadata and getattr(metadata, "round_number", None):
        try:
            round_number = int(getattr(metadata, "round_number"))
        except Exception:
            round_number = None

    rfq_id = metadata.rfq_id if metadata else None
    header_rfq = _header_lookup("X-ProcWise-RFQ-ID")
    if header_rfq:
        rfq_id = header_rfq.strip()

    subject = message.get("Subject")
    message_id = (message.get("Message-ID") or "").strip("<> ") or None
    from_address = message.get("From")

    date_header = message.get("Date")
    try:
        received_at = parsedate_to_datetime(date_header) if date_header else None
    except Exception:  # pragma: no cover - defensive
        received_at = None
    received_at = received_at or datetime.now(timezone.utc)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)

    supplier_email = metadata.supplier_email if metadata else None
    if not supplier_email and from_address:
        supplier_email = parseaddr(from_address)[1]

    return EmailResponse(
        unique_id=unique_id,
        supplier_id=supplier_id,
        supplier_email=supplier_email,
        from_address=from_address,
        message_id=message_id,
        subject=subject,
        body=body_text or "",
        received_at=received_at,
        body_html=body_html,
        in_reply_to=thread_headers.get("in_reply_to", ()),
        references=thread_headers.get("references", ()),
        workflow_id=workflow_id,
        rfq_id=rfq_id,
        round_number=round_number,
        raw_headers=raw_headers,
    )


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------


def _normalise_thread_header(value: Any) -> Sequence[str]:
    if value in (None, ""):
        return tuple()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip("<> ") for item in value if str(item).strip())
    return (str(value).strip("<> "),)


def _normalise_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text.upper() or None


def _normalise_email_address(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    address = parseaddr(value)[1]
    return address.lower() if address else None


def _extract_email_domain(value: Optional[str]) -> Optional[str]:
    address = _normalise_email_address(value)
    if not address or "@" not in address:
        return None
    return address.split("@", 1)[-1]


def _normalise_subject_line(subject: Optional[str]) -> Optional[str]:
    if not subject:
        return None
    subject = subject.strip()
    subject = re.sub(r"\s+", " ", subject)
    subject = subject.replace("Re:", "").replace("Fwd:", "").strip()
    if subject.lower().startswith("re: "):
        subject = subject[4:].strip()
    return subject.lower() or None


def _score_to_confidence(score: float) -> Decimal:
    return Decimal(score).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _thread_header_score(
    dispatch: EmailDispatchRecord,
    email_response: EmailResponse,
) -> Tuple[float, str]:
    if not email_response.in_reply_to and not email_response.references:
        return 0.0, "no-thread-headers"

    reply_headers = set(email_response.in_reply_to) | set(email_response.references)
    dispatch_headers = set()
    if dispatch.thread_headers:
        dispatch_headers.update(dispatch.thread_headers.get("references", ()))
        dispatch_headers.update(dispatch.thread_headers.get("in_reply_to", ()))
    if dispatch.message_id:
        dispatch_headers.add(dispatch.message_id.strip("<>"))
    dispatch_headers = {header.strip().lower() for header in dispatch_headers if header}
    reply_headers = {header.strip().lower() for header in reply_headers if header}

    if not dispatch_headers:
        return 0.0, "no-dispatch-headers"
    if reply_headers & dispatch_headers:
        return 1.0, "thread-header"
    return 0.0, "thread-miss"


def _supplier_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> Tuple[float, str]:
    dispatch_email = _normalise_email_address(dispatch.supplier_email)
    response_email = _normalise_email_address(email_response.from_address or email_response.supplier_email)
    if dispatch_email and response_email and dispatch_email == response_email:
        return 0.7, "supplier-email"
    dispatch_supplier = _normalise_identifier(dispatch.supplier_id)
    response_supplier = _normalise_identifier(email_response.supplier_id)
    if dispatch_supplier and response_supplier and dispatch_supplier == response_supplier:
        return 0.6, "supplier-id"
    return 0.0, "supplier-miss"


def _subject_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> Tuple[float, str]:
    if not dispatch.subject or not email_response.subject:
        return 0.0, "subject-missing"
    dispatch_subject = _normalise_subject_line(dispatch.subject)
    response_subject = _normalise_subject_line(email_response.subject)
    if dispatch_subject and response_subject and dispatch_subject == response_subject:
        return 0.4, "subject"
    return 0.0, "subject-miss"


def _temporal_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> Tuple[float, str]:
    if not dispatch.dispatched_at or not email_response.received_at:
        return 0.0, "temporal-missing"
    delta_seconds = abs((email_response.received_at - dispatch.dispatched_at).total_seconds())
    if delta_seconds <= 72 * 3600:
        return 0.15, "temporal"
    return 0.0, "temporal-far"


def _domain_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> Tuple[float, str]:
    dispatch_domain = _extract_email_domain(dispatch.supplier_email)
    response_domain = _extract_email_domain(email_response.from_address or email_response.supplier_email)
    if dispatch_domain and response_domain and dispatch_domain == response_domain:
        return 0.25, "email-domain"
    return 0.0, "domain-miss"


def _workflow_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> Tuple[float, str]:
    if not dispatch.workflow_id or not email_response.workflow_id:
        return 0.0, "workflow-missing"
    dispatch_id = _normalise_identifier(dispatch.workflow_id)
    response_id = _normalise_identifier(email_response.workflow_id)
    if dispatch_id and response_id and dispatch_id == response_id:
        return 0.3, "workflow"
    return 0.0, "workflow-miss"


def _round_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> Tuple[float, str]:
    if dispatch.round_number is None or email_response.round_number is None:
        return 0.0, "round-missing"
    if int(dispatch.round_number) == int(email_response.round_number):
        return 0.2, "round"
    return 0.0, "round-miss"


def score_dispatch_match(
    dispatch: EmailDispatchRecord,
    email_response: EmailResponse,
    *,
    match_threshold: float,
) -> Tuple[float, str]:
    """Compute a composite score comparing ``email_response`` to ``dispatch``."""

    if email_response.unique_id and email_response.unique_id == dispatch.unique_id:
        return 1.0, "unique-id"

    for thread_id in email_response.in_reply_to:
        if dispatch.message_id and thread_id.strip().lower() == dispatch.message_id.strip("<>").lower():
            return 1.0, "in-reply-to"
    for thread_id in email_response.references:
        if dispatch.message_id and thread_id.strip().lower() == dispatch.message_id.strip("<>").lower():
            return 0.9, "references"

    score = 0.0
    best_reason = "unknown"

    for component in (
        _thread_header_score,
        _supplier_score,
        _subject_score,
        _temporal_score,
        _workflow_score,
        _round_score,
        _domain_score,
    ):
        component_score, reason = component(dispatch, email_response)
        if component_score:
            score += component_score
            best_reason = reason
    score = min(score, 1.0)
    if score < match_threshold:
        return score, best_reason
    return score, best_reason


# ---------------------------------------------------------------------------
# IMAP fetching
# ---------------------------------------------------------------------------


def _parse_search_results(response: Sequence[bytes]) -> List[str]:
    identifiers: List[str] = []
    for chunk in response:
        if not chunk:
            continue
        identifiers.extend(chunk.decode().split())
    return identifiers


class ImapEmailFetcher:
    """Fetches email responses from an IMAP server."""

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        mailbox: str = "INBOX",
        port: int = 993,
        use_ssl: bool = True,
        login: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.mailbox = mailbox
        self.port = port
        self.use_ssl = use_ssl
        self.login = login
        self.limit = limit

    async def fetch(self, *, since: datetime) -> List[EmailResponse]:
        return await asyncio.to_thread(self._fetch_sync, since)

    def _fetch_sync(self, since: datetime) -> List[EmailResponse]:
        logger.debug("Fetching IMAP emails since %s", since.isoformat())
        client = _imap_client(
            self.host,
            self.username,
            self.password,
            port=self.port,
            use_ssl=self.use_ssl,
            login=self.login,
        )
        try:
            client.select(self.mailbox)
            search_key = since.strftime('%d-%b-%Y')
            status, data = client.search(None, "SINCE", search_key)
            if status != "OK":
                logger.warning("IMAP search failed: %s %s", status, data)
                return []
            identifiers = _parse_search_results(data)
            responses: List[EmailResponse] = []
            for identifier in identifiers[-self.limit if self.limit else None :]:
                status, payload = client.fetch(identifier, "(RFC822)")
                if status != "OK":
                    logger.warning("Failed to fetch message %s: %s", identifier, status)
                    continue
                for part in payload:
                    if not isinstance(part, tuple):
                        continue
                    raw = part[1]
                    try:
                        responses.append(parse_email_bytes(raw))
                    except Exception:
                        logger.exception("Failed to parse email %s", identifier)
            return responses
        finally:
            try:
                client.logout()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to logout IMAP client", exc_info=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmailWatcherConfig:
    imap_host: Optional[str] = None
    imap_username: Optional[str] = None
    imap_password: Optional[str] = None
    imap_port: int = 993
    imap_use_ssl: bool = True
    imap_login: Optional[str] = None
    imap_mailbox: str = "INBOX"
    dispatch_wait_seconds: int = 90
    poll_interval_seconds: int = 30
    max_poll_attempts: int = 10
    match_threshold: float = 0.45
    lookback_minutes: int = 240
    max_fetch: Optional[int] = None
    poll_backoff_factor: float = 1.8
    poll_jitter_seconds: float = 2.0
    poll_max_interval_seconds: int = 300
    poll_timeout_seconds: Optional[int] = None

    @classmethod
    def from_settings(cls, settings: Any) -> "EmailWatcherConfig":
        def _coerce_float(names: Sequence[str], default: float) -> float:
            for name in names:
                value = getattr(settings, name, None)
                if value in (None, ""):
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
            return default

        def _coerce_int(names: Sequence[str], default: int) -> int:
            for name in names:
                value = getattr(settings, name, None)
                if value in (None, ""):
                    continue
                try:
                    return int(value)
                except Exception:
                    continue
            return default

        def _coerce_optional_int(names: Sequence[str]) -> Optional[int]:
            for name in names:
                value = getattr(settings, name, None)
                if value in (None, ""):
                    continue
                try:
                    candidate = int(value)
                except Exception:
                    continue
                if candidate > 0:
                    return candidate
            return None

        backoff = _coerce_float(("poll_backoff_factor", "imap_poll_backoff_factor"), 1.8)
        jitter = _coerce_float(("poll_jitter_seconds", "imap_poll_jitter_seconds"), 2.0)
        max_interval = _coerce_int(("poll_max_interval_seconds", "imap_poll_max_interval", "imap_poll_max_interval_seconds"), 300)
        timeout_seconds = _coerce_optional_int(("poll_timeout_seconds", "imap_poll_timeout_seconds"))
        return cls(
            imap_host=getattr(settings, "imap_host", None),
            imap_username=getattr(settings, "imap_username", None),
            imap_password=getattr(settings, "imap_password", None),
            imap_port=int(getattr(settings, "imap_port", 993) or 993),
            imap_use_ssl=bool(getattr(settings, "imap_use_ssl", True)),
            imap_login=getattr(settings, "imap_login", None),
            imap_mailbox=getattr(settings, "imap_mailbox", "INBOX"),
            dispatch_wait_seconds=int(getattr(settings, "dispatch_wait_seconds", 90) or 90),
            poll_interval_seconds=int(getattr(settings, "poll_interval_seconds", 30) or 30),
            max_poll_attempts=int(getattr(settings, "max_poll_attempts", 10) or 10),
            match_threshold=float(getattr(settings, "match_threshold", 0.45) or 0.45),
            lookback_minutes=int(getattr(settings, "lookback_minutes", 240) or 240),
            max_fetch=getattr(settings, "max_fetch", None),
            poll_backoff_factor=backoff,
            poll_jitter_seconds=jitter,
            poll_max_interval_seconds=max_interval,
            poll_timeout_seconds=timeout_seconds,
        )


# ---------------------------------------------------------------------------
# Poll controller
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _AdaptivePollController:
    config: EmailWatcherConfig
    now: Callable[[], datetime]
    base_interval: float = field(init=False)
    _current_interval: float = field(init=False)
    max_interval: float = field(init=False)
    backoff_factor: float = field(init=False)
    jitter_seconds: float = field(init=False)
    max_empty_attempts: int = field(init=False)
    timeout_seconds: Optional[int] = field(init=False)
    start_time: datetime = field(init=False)
    last_activity: datetime = field(init=False)
    empty_attempts: int = field(init=False, default=0)
    total_polls: int = field(init=False, default=0)
    last_delay: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.base_interval = max(1.0, float(self.config.poll_interval_seconds))
        self._current_interval = self.base_interval
        self.max_interval = max(self.base_interval, float(self.config.poll_max_interval_seconds or self.base_interval))
        self.backoff_factor = max(1.0, float(self.config.poll_backoff_factor or 1.0))
        self.jitter_seconds = max(0.0, float(self.config.poll_jitter_seconds or 0.0))
        self.max_empty_attempts = max(1, int(self.config.max_poll_attempts or 1))
        self.timeout_seconds = (
            int(self.config.poll_timeout_seconds)
            if self.config.poll_timeout_seconds and self.config.poll_timeout_seconds > 0
            else None
        )
        self.start_time = self.now()
        self.last_activity = self.start_time
        self.empty_attempts = 0
        self.total_polls = 0
        self.last_delay = 0.0

    def _apply_jitter(self, value: float) -> float:
        if self.jitter_seconds <= 0.0:
            return value
        jitter = random.uniform(0.0, self.jitter_seconds)
        return max(0.0, value + jitter)

    def record_activity(self) -> None:
        self.total_polls += 1
        self.empty_attempts = 0
        self._current_interval = self.base_interval
        self.last_activity = self.now()
        self.last_delay = 0.0

    def record_empty(self) -> None:
        self.total_polls += 1
        self.empty_attempts += 1
        self.last_activity = self.now()

    def next_delay(self) -> float:
        delay = self._apply_jitter(self._current_interval)
        self.last_delay = delay
        self._current_interval = min(self._current_interval * self.backoff_factor, self.max_interval)
        return delay

    def check_limits(self) -> Tuple[bool, Optional[str]]:
        if self.timeout_seconds is not None:
            elapsed = (self.now() - self.start_time).total_seconds()
            if elapsed >= self.timeout_seconds:
                return True, "timeout"
        if self.empty_attempts >= self.max_empty_attempts:
            return True, "max_attempts"
        return False, None


# ---------------------------------------------------------------------------
# Core watcher implementation
# ---------------------------------------------------------------------------


class EmailWatcher:
    """Unified email watcher orchestrating dispatch + response handling."""

    def __init__(
        self,
        *,
        config: EmailWatcherConfig,
        supplier_agent: Optional[SupplierInteractionAgent] = None,
        negotiation_agent: Optional[NegotiationAgent] = None,
        fetcher: Optional[Union[ImapEmailFetcher, Callable[..., List[EmailResponse]], Callable[..., Awaitable[List[EmailResponse]]]]] = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config
        self.supplier_agent = supplier_agent
        self.negotiation_agent = negotiation_agent
        self._fetcher = fetcher
        self._sleep = sleep
        self._now = now
        self._trackers: Dict[str, WorkflowTracker] = {}

        tracking_repo.init_schema()
        supplier_response_repo.init_schema()

    # ------------------------------------------------------------------
    # Tracker management
    # ------------------------------------------------------------------

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
                    rfq_id=getattr(row, "rfq_id", None),
                    dispatch_id=row.dispatch_key,
                    round_number=row.round_number,
                    workflow_id=workflow_id,
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
                        received_at=row.responded_at or row.dispatched_at or self._now(),
                    ),
                )
        self._trackers[workflow_id] = tracker
        return tracker

    # ------------------------------------------------------------------
    # Dispatch registration
    # ------------------------------------------------------------------

    def register_workflow_dispatch(self, workflow_id: str, dispatches: Sequence[Dict[str, object]]) -> WorkflowTracker:
        if not workflow_id:
            raise ValueError("workflow_id is required to register dispatches")

        tracker = self._ensure_tracker(workflow_id)
        records: List[EmailDispatchRecord] = []
        repo_rows: List[WorkflowDispatchRow] = []

        for payload in dispatches:
            unique_id = str(payload.get("unique_id") or email_tracking.generate_unique_email_id(workflow_id))
            supplier_id = payload.get("supplier_id")
            supplier_email = payload.get("supplier_email")
            message_id = payload.get("message_id")
            subject = payload.get("subject")
            dispatched_at = payload.get("dispatched_at")
            rfq_id = payload.get("rfq_id")
            dispatch_key = payload.get("dispatch_key")
            raw_thread_headers = payload.get("thread_headers") if isinstance(payload.get("thread_headers"), dict) else {}
            round_value = payload.get("round_number") or payload.get("round") or payload.get("thread_index")
            try:
                round_number = int(round_value) if round_value is not None else None
            except Exception:
                round_number = None
            if round_number is not None and round_number < 1:
                round_number = None
            if isinstance(dispatched_at, datetime):
                dispatched_dt = dispatched_at if dispatched_at.tzinfo else dispatched_at.replace(tzinfo=timezone.utc)
            else:
                dispatched_dt = self._now()

            normalised_headers: Dict[str, Sequence[str]] = {}
            for key, value in raw_thread_headers.items():
                normalised = _normalise_thread_header(value)
                if normalised:
                    normalised_headers[str(key)] = normalised

            records.append(
                EmailDispatchRecord(
                    unique_id=unique_id,
                    supplier_id=str(supplier_id) if supplier_id else None,
                    supplier_email=str(supplier_email) if supplier_email else None,
                    message_id=str(message_id) if message_id else None,
                    subject=str(subject) if subject else None,
                    rfq_id=str(rfq_id) if rfq_id else None,
                    dispatch_id=str(dispatch_key).strip() if dispatch_key else None,
                    round_number=round_number,
                    workflow_id=workflow_id,
                    thread_headers=normalised_headers,
                    dispatched_at=dispatched_dt,
                )
            )

            repo_rows.append(
                WorkflowDispatchRow(
                    workflow_id=workflow_id,
                    unique_id=unique_id,
                    dispatch_key=str(dispatch_key).strip() if dispatch_key else None,
                    supplier_id=str(supplier_id) if supplier_id else None,
                    supplier_email=str(supplier_email) if supplier_email else None,
                    message_id=str(message_id) if message_id else None,
                    subject=str(subject) if subject else None,
                    round_number=round_number,
                    dispatched_at=dispatched_dt,
                    responded_at=None,
                    response_message_id=None,
                    matched=False,
                    thread_headers={
                        key: list(value) for key, value in normalised_headers.items()
                    }
                    if normalised_headers
                    else None,
                )
            )

        tracker.register_dispatches(records)
        tracking_repo.record_dispatches(workflow_id=workflow_id, dispatches=repo_rows)
        return tracker

    # ------------------------------------------------------------------
    # Fetching + matching
    # ------------------------------------------------------------------

    async def _fetch_emails(self, since: datetime) -> List[EmailResponse]:
        fetcher = self._fetcher
        if fetcher is None and self.config.imap_host and self.config.imap_username and self.config.imap_password:
            fetcher = ImapEmailFetcher(
                host=self.config.imap_host,
                username=self.config.imap_username,
                password=self.config.imap_password,
                mailbox=self.config.imap_mailbox,
                port=self.config.imap_port,
                use_ssl=self.config.imap_use_ssl,
                login=self.config.imap_login,
                limit=self.config.max_fetch,
            )
        if fetcher is None:
            logger.warning("Email fetcher not configured; returning no responses")
            return []

        if isinstance(fetcher, ImapEmailFetcher):
            return await fetcher.fetch(since=since)
        if asyncio.iscoroutinefunction(fetcher):
            return await fetcher(since=since)
        if callable(fetcher):
            return await asyncio.to_thread(fetcher, since=since)
        # Fallback for awaitable instances
        result = fetcher(since=since)  # type: ignore[operator]
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _match_responses(self, tracker: WorkflowTracker, responses: Sequence[EmailResponse]) -> List[SupplierResponseRow]:
        matched_rows: List[SupplierResponseRow] = []
        for email_response in responses:
            matched_id: Optional[str] = None
            best_dispatch: Optional[EmailDispatchRecord] = None
            best_score = 0.0
            best_reason = "unknown"

            if email_response.unique_id:
                uid = email_response.unique_id.strip()
                if uid in tracker.email_records:
                    matched_id = uid
                    best_dispatch = tracker.email_records[uid]
                    best_score = 1.0
                    best_reason = "unique-id"

            if not matched_id and email_response.rfq_id:
                normalised_rfq = _normalise_identifier(email_response.rfq_id)
                if normalised_rfq and normalised_rfq in tracker.rfq_index:
                    for uid in tracker.rfq_index[normalised_rfq]:
                        dispatch = tracker.email_records.get(uid)
                        if not dispatch:
                            continue
                        score, reason = score_dispatch_match(
                            dispatch,
                            email_response,
                            match_threshold=self.config.match_threshold,
                        )
                        if dispatch.rfq_id and _normalise_identifier(dispatch.rfq_id) == normalised_rfq:
                            if score < self.config.match_threshold:
                                score = max(score, 0.9)
                                reason = "rfq_id"
                        if score > best_score:
                            matched_id = uid
                            best_dispatch = dispatch
                            best_score = score
                            best_reason = reason

            if not matched_id:
                for uid, dispatch in tracker.email_records.items():
                    if uid in tracker.matched_responses:
                        continue
                    score, reason = score_dispatch_match(dispatch, email_response, match_threshold=self.config.match_threshold)
                    if score > best_score:
                        matched_id = uid
                        best_dispatch = dispatch
                        best_score = score
                        best_reason = reason

            if matched_id and best_dispatch and best_score >= self.config.match_threshold:
                tracker.record_response(matched_id, email_response)
                supplier_email = (
                    email_response.supplier_email
                    or _normalise_email_address(email_response.from_address)
                    or best_dispatch.supplier_email
                )
                supplier_id = email_response.supplier_id or best_dispatch.supplier_id
                response_time = None
                if best_dispatch.dispatched_at and email_response.received_at:
                    delta = email_response.received_at - best_dispatch.dispatched_at
                    response_time = max(delta.total_seconds(), 0.0)

                response_row = SupplierResponseRow(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    supplier_id=supplier_id,
                    supplier_email=supplier_email,
                    response_text=email_response.body,
                    response_body=email_response.body_html,
                    received_time=email_response.received_at,
                    response_time=response_time,
                    response_message_id=email_response.message_id,
                    response_subject=email_response.subject,
                    response_from=email_response.from_address,
                    original_message_id=best_dispatch.message_id,
                    original_subject=best_dispatch.subject,
                    match_confidence=_score_to_confidence(best_score),
                    match_score=best_score,
                    matched_on=best_reason,
                    dispatch_id=best_dispatch.dispatch_id or best_dispatch.unique_id,
                    raw_headers=email_response.raw_headers,
                    processed=False,
                )
                supplier_response_repo.insert_response(response_row)
                if tracker.workflow_id:
                    responded_at = email_response.received_at or datetime.now(timezone.utc)
                    try:
                        tracking_repo.mark_response(
                            workflow_id=tracker.workflow_id,
                            unique_id=matched_id,
                            responded_at=responded_at,
                            response_message_id=email_response.message_id,
                        )
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "Failed to mark workflow email tracking responded workflow=%s unique_id=%s",
                            tracker.workflow_id,
                            matched_id,
                        )
                matched_rows.append(response_row)
            else:
                logger.warning(
                    "MATCH FAILED: workflow=%s email_from=%s score=%.2f reason=%s",
                    tracker.workflow_id,
                    email_response.from_address,
                    best_score,
                    best_reason,
                )
        return matched_rows

    # ------------------------------------------------------------------
    # Agent processing
    # ------------------------------------------------------------------

    def _process_agents(self, tracker: WorkflowTracker) -> None:
        if not tracker.all_dispatched or not tracker.all_responded:
            logger.debug(
                "Skipping agent processing for workflow %s until all dispatches and responses are ready",
                tracker.workflow_id,
            )
            return

        pending_rows = supplier_response_repo.fetch_pending(workflow_id=tracker.workflow_id)
        if not pending_rows:
            return

        processed_ids: List[str] = []
        for row in pending_rows:
            unique_id = row.get("unique_id")
            matched = tracker.matched_responses.get(unique_id) if unique_id else None
            supplier_id = row.get("supplier_id") or (matched.supplier_id if matched else None)
            subject = row.get("response_subject") or (matched.subject if matched else None)
            message_id = row.get("response_message_id") or (matched.message_id if matched else None)
            from_address = row.get("response_from") or (matched.from_address if matched else None)
            body_text = (
                row.get("response_text")
                or row.get("response_body")
                or (matched.body if matched else "")
            )
            workflow_id = matched.workflow_id if matched and matched.workflow_id else tracker.workflow_id
            rfq_id = matched.rfq_id if matched and matched.rfq_id else None
            supplier_email = row.get("supplier_email") or (matched.supplier_email if matched else None)

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
            }
            if rfq_id:
                input_payload["rfq_id"] = rfq_id

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
                agent_id="EmailWatcher",
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def wait_for_responses_async(self, workflow_id: str) -> Dict[str, object]:
        tracker = self._ensure_tracker(workflow_id)
        if tracker.dispatched_count == 0:
            return {
                "workflow_id": workflow_id,
                "complete": True,
                "dispatched_count": 0,
                "responded_count": 0,
                "matched_responses": tracker.matched_responses,
                "status": "completed",
                "poll_attempts": 0,
            }

        start_time = self._now()
        if tracker.last_dispatched_at:
            target = tracker.last_dispatched_at + timedelta(seconds=self.config.dispatch_wait_seconds)
            now = self._now()
            if target > now:
                wait_time = (target - now).total_seconds()
                logger.info(
                    "Waiting %.1f seconds before polling IMAP for workflow %s",
                    wait_time,
                    workflow_id,
                )
                await asyncio.to_thread(self._sleep, wait_time)

        since = tracker.last_dispatched_at or (self._now() - timedelta(minutes=self.config.lookback_minutes))
        last_capture = tracker.last_dispatched_at or start_time
        status = "pending"
        timeout_reason: Optional[str] = None
        poll_controller = _AdaptivePollController(config=self.config, now=self._now)

        while not tracker.all_responded:
            exceeded, reason = poll_controller.check_limits()
            if exceeded:
                status = "timeout" if reason == "timeout" else "max_attempts_exceeded"
                timeout_reason = reason
                logger.warning(
                    "Watcher exiting for workflow=%s due to %s after %.1fs",
                    workflow_id,
                    reason,
                    (self._now() - start_time).total_seconds(),
                )
                break

            responses = await self._fetch_emails(since)
            matched_rows = self._match_responses(tracker, responses)
            if matched_rows:
                last_capture = self._now()
                since = min(since, last_capture)
                poll_controller.record_activity()
                self._process_agents(tracker)
            else:
                poll_controller.record_empty()

            pending_suppliers = [
                dispatch.supplier_id
                for uid, dispatch in tracker.email_records.items()
                if uid not in tracker.matched_responses and dispatch.supplier_id
            ]
            heartbeat = {
                "event": "watcher_heartbeat",
                "workflow_id": workflow_id,
                "expected_count": tracker.dispatched_count,
                "responses_received": tracker.responded_count,
                "pending_suppliers": pending_suppliers,
                "since_last_capture_s": int(max(0.0, (self._now() - last_capture).total_seconds())),
                "poll_attempts": poll_controller.total_polls,
                "last_delay_s": round(poll_controller.last_delay, 2),
            }
            logger.info(json.dumps(heartbeat))

            if tracker.all_responded:
                status = "completed"
                break

            exceeded, reason = poll_controller.check_limits()
            if exceeded:
                status = "timeout" if reason == "timeout" else "max_attempts_exceeded"
                timeout_reason = reason
                logger.warning(
                    "Watcher exiting for workflow=%s due to %s after %.1fs",
                    workflow_id,
                    reason,
                    (self._now() - start_time).total_seconds(),
                )
                break

            delay = poll_controller.next_delay()
            if delay > 0:
                logger.debug(
                    "Sleeping %.2fs before next poll workflow=%s attempts=%s",
                    delay,
                    workflow_id,
                    poll_controller.empty_attempts,
                )
                await asyncio.to_thread(self._sleep, delay)

        complete = tracker.all_responded
        if complete:
            status = "completed"
        result = {
            "workflow_id": workflow_id,
            "complete": complete,
            "dispatched_count": tracker.dispatched_count,
            "responded_count": tracker.responded_count,
            "matched_responses": tracker.matched_responses,
            "status": status,
            "poll_attempts": poll_controller.total_polls,
            "last_delay_s": poll_controller.last_delay,
        }
        if timeout_reason:
            result["timeout_reason"] = timeout_reason

        self._process_agents(tracker)
        logger.info(json.dumps({"event": "watcher_stopped", "workflow_id": workflow_id, "status": status}))
        return result

    def wait_for_responses(self, workflow_id: str) -> Dict[str, object]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "wait_for_responses cannot be called from an active event loop; "
                "use wait_for_responses_async instead",
            )
        return asyncio.run(self.wait_for_responses_async(workflow_id))


# ---------------------------------------------------------------------------
# Compatibility + utility helpers
# ---------------------------------------------------------------------------


def generate_unique_email_id(
    workflow_id: str,
    supplier_id: Optional[str] = None,
    *,
    round_number: Optional[int] = None,
) -> str:
    """Proxy to :func:`utils.email_tracking.generate_unique_email_id`."""

    return email_tracking.generate_unique_email_id(workflow_id, supplier_id, round_number=round_number)


def embed_unique_id_in_email_body(body: Optional[str], unique_id: str) -> str:
    """Proxy to :func:`utils.email_tracking.embed_unique_id_in_email_body`."""

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
    """Persist a dispatched email for downstream response reconciliation."""

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
        in_reply_to=list(headers.get("in_reply_to", [])) or None,
        references=list(headers.get("references", [])) or None,
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

    _ = agent_nick  # retained for backwards compatibility / logging
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

    supplier_interaction_repo.init_schema()
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
    """Background poller that persists supplier responses as they arrive."""

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

    def _fetch(self, **kwargs: Any) -> List[EmailResponse]:
        if self._email_fetcher is not None:
            return self._email_fetcher(**kwargs)
        raise RuntimeError("ContinuousEmailWatcher requires an email_fetcher when no IMAP config is supplied")

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
            in_reply_to=row.get("in_reply_to"),
            references=row.get("references") or row.get("reference_ids"),
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

    def _handle_response(self, response: EmailResponse) -> Optional[Dict[str, Any]]:
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

__all__ = [
    "EmailDispatchRecord",
    "EmailResponse",
    "WorkflowTracker",
    "ImapEmailFetcher",
    "EmailWatcherConfig",
    "EmailWatcher",
    "parse_email_bytes",
    "generate_unique_email_id",
    "embed_unique_id_in_email_body",
    "register_sent_email",
    "get_supplier_responses",
    "mark_interaction_processed",
    "wait_for_responses",
    "ContinuousEmailWatcher",
]
