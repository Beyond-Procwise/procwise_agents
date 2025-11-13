"""Unified email watcher module for ProcWise."""
from __future__ import annotations

import asyncio
import imaplib
import json
import logging
import os
import random
import re
import threading
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
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from agents.base_agent import AgentContext, AgentOutput
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from services.process_routing_service import ProcessRoutingService
from repositories import (
    supplier_interaction_repo,
    supplier_response_repo,
    workflow_email_tracking_repo as tracking_repo,
    workflow_round_response_repo,
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
    response_grace_seconds: int = 180

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
        grace_raw = getattr(settings, "response_grace_seconds", getattr(settings, "email_response_grace_seconds", 180))
        try:
            grace_value = int(grace_raw)
        except Exception:
            grace_value = 180
        grace_value = max(0, grace_value)

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
            response_grace_seconds=grace_value,
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
    grace_until: Optional[datetime] = field(init=False, default=None)
    grace_active: bool = field(init=False, default=False)
    grace_reason: Optional[str] = field(init=False, default=None)

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
        if self.grace_active:
            self.grace_active = False
            self.grace_reason = None

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
        now = self.now()
        reason: Optional[str] = None
        if self.timeout_seconds is not None:
            elapsed = (now - self.start_time).total_seconds()
            if elapsed >= self.timeout_seconds:
                reason = "timeout"
        if reason is None and self.empty_attempts >= self.max_empty_attempts:
            reason = "max_attempts"

        if reason and self.grace_until and now < self.grace_until:
            if not self.grace_active:
                self.grace_active = True
                self.grace_reason = reason
                self.empty_attempts = 0
                self.start_time = now
                self.last_activity = now
                self._current_interval = self.base_interval
            return False, None

        if reason:
            self.grace_active = False
            self.grace_reason = reason
            return True, reason
        return False, None

    def activate_grace(self, until: datetime, *, reason: Optional[str] = None) -> bool:
        now = self.now()
        if until <= now:
            return False
        self.grace_until = until
        self.grace_active = True
        self.grace_reason = reason
        self.empty_attempts = 0
        self.start_time = now
        self.last_activity = now
        self._current_interval = self.base_interval
        return True


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
        process_router: Optional[ProcessRoutingService] = None,
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
        self.process_router = process_router

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
        expectations: List[Tuple[Optional[int], Optional[str], Optional[str]]] = []
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
            for row in dispatches:
                expectations.append((row.round_number, row.unique_id, row.supplier_id))
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
        if expectations:
            try:
                workflow_round_response_repo.register_expected(
                    workflow_id=workflow_id,
                    expectations=expectations,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to register existing round expectations for workflow=%s",
                    workflow_id,
                    exc_info=True,
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

        round_expectations: List[Tuple[Optional[int], Optional[str], Optional[str]]] = []
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
            round_expectations.append((round_number, unique_id, str(supplier_id) if supplier_id else None))

        tracker.register_dispatches(records)
        tracking_repo.record_dispatches(workflow_id=workflow_id, dispatches=repo_rows)
        try:
            workflow_round_response_repo.register_expected(
                workflow_id=workflow_id,
                expectations=round_expectations,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to record round expectations for workflow=%s",
                workflow_id,
                exc_info=True,
            )
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
                    round_number=best_dispatch.round_number,
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
                try:
                    workflow_round_response_repo.mark_response_received(
                        workflow_id=tracker.workflow_id,
                        round_number=best_dispatch.round_number,
                        unique_id=matched_id,
                        supplier_id=supplier_id,
                        responded_at=email_response.received_at,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to mark round response received for workflow=%s unique_id=%s",
                        tracker.workflow_id,
                        matched_id,
                        exc_info=True,
                    )
                matched_rows.append(response_row)
            else:
                pending_candidates = [
                    (uid, dispatch)
                    for uid, dispatch in tracker.email_records.items()
                    if uid not in tracker.matched_responses
                ]
                resolved = False
                if not matched_id and len(pending_candidates) == 1:
                    uid, dispatch = pending_candidates[0]
                    supplier_match = False
                    response_supplier = _normalise_identifier(email_response.supplier_id)
                    dispatch_supplier = _normalise_identifier(dispatch.supplier_id)
                    if response_supplier and dispatch_supplier and response_supplier == dispatch_supplier:
                        supplier_match = True
                    else:
                        response_email = _normalise_email_address(
                            email_response.from_address or email_response.supplier_email
                        )
                        dispatch_email = _normalise_email_address(dispatch.supplier_email)
                        supplier_match = bool(response_email and dispatch_email and response_email == dispatch_email)

                    if supplier_match or not tracker.matched_responses:
                        matched_id = uid
                        best_dispatch = dispatch
                        best_score = max(best_score, 0.55)
                        best_reason = "sole-pending-dispatch"

                if matched_id and best_dispatch and best_score >= 0.5:
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
                        match_confidence=_score_to_confidence(max(best_score, self.config.match_threshold)),
                        match_score=max(best_score, self.config.match_threshold),
                        matched_on=best_reason,
                        dispatch_id=best_dispatch.dispatch_id or best_dispatch.unique_id,
                        raw_headers=email_response.raw_headers,
                        processed=False,
                        round_number=best_dispatch.round_number,
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
                    try:
                        workflow_round_response_repo.mark_response_received(
                            workflow_id=tracker.workflow_id,
                            round_number=best_dispatch.round_number,
                            unique_id=matched_id,
                            supplier_id=supplier_id,
                            responded_at=email_response.received_at,
                        )
                    except Exception:  # pragma: no cover - defensive logging
                        logger.debug(
                            "Failed to mark round response received for workflow=%s unique_id=%s",
                            tracker.workflow_id,
                            matched_id,
                            exc_info=True,
                        )
                    matched_rows.append(response_row)
                    resolved = True

                if not resolved:
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
        grace_deadline: Optional[datetime] = None
        if self.config.response_grace_seconds > 0:
            baseline = tracker.last_dispatched_at or start_time
            grace_deadline = baseline + timedelta(seconds=self.config.response_grace_seconds)

        while not tracker.all_responded:
            exceeded, reason = poll_controller.check_limits()
            if exceeded:
                status = "timeout" if reason == "timeout" else "max_attempts_exceeded"
                timeout_reason = reason
                if grace_deadline and self._now() < grace_deadline:
                    poll_controller.activate_grace(grace_deadline, reason=reason)
                    timeout_reason = None
                else:
                    status = "timeout" if reason == "timeout" else "max_attempts_exceeded"
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
            if grace_deadline:
                heartbeat["grace_remaining_s"] = max(
                    0,
                    int((grace_deadline - self._now()).total_seconds()),
                )
            logger.info(json.dumps(heartbeat))

            if tracker.all_responded:
                status = "completed"
                break

            exceeded, reason = poll_controller.check_limits()
            if exceeded:
                timeout_reason = reason
                if grace_deadline and self._now() < grace_deadline:
                    if poll_controller.activate_grace(grace_deadline, reason=reason):
                        logger.info(
                            json.dumps(
                                {
                                    "event": "watcher_grace_period",
                                    "workflow_id": workflow_id,
                                    "grace_until": grace_deadline.isoformat(),
                                    "pending_suppliers": pending_suppliers,
                                    "responses_received": tracker.responded_count,
                                }
                            )
                        )
                        timeout_reason = None
                        continue
                status = "timeout" if reason == "timeout" else "max_attempts_exceeded"
                timeout_reason = reason
                logger.warning(
                    "Watcher exiting for workflow=%s due to %s after %.1fs",
                    workflow_id,
                    reason,
                    (self._now() - start_time).total_seconds(),
                )
                break

            if not tracker.all_responded and grace_deadline and self._now() < grace_deadline:
                poll_controller.activate_grace(grace_deadline, reason=timeout_reason)
                timeout_reason = None

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
        pending_unique_ids = [
            uid
            for uid in tracker.email_records
            if uid not in tracker.matched_responses
        ]
        if timeout_reason:
            result["timeout_reason"] = timeout_reason

        if not complete and pending_unique_ids:
            status = "failed"
            result["status"] = status
            result["pending_unique_ids"] = pending_unique_ids
            result["pending_suppliers"] = [
                tracker.email_records[uid].supplier_id
                for uid in pending_unique_ids
                if tracker.email_records.get(uid)
            ]
            failure_payload = {
                "reason": timeout_reason or "responses_missing",
                "pending_unique_ids": pending_unique_ids,
                "pending_suppliers": [
                    tracker.email_records[uid].supplier_id
                    for uid in pending_unique_ids
                    if tracker.email_records.get(uid)
                ],
                "responded_count": tracker.responded_count,
                "dispatched_count": tracker.dispatched_count,
            }
            self._mark_process_failed(tracker.workflow_id, failure_payload)
        else:
            self._process_agents(tracker)

        logger.info(json.dumps({"event": "watcher_stopped", "workflow_id": workflow_id, "status": status}))
        return result

    def _mark_process_failed(self, workflow_id: Optional[str], details: Dict[str, Any]) -> None:
        router = getattr(self, "process_router", None)
        workflow_key = (workflow_id or "").strip()
        if not workflow_key or not router:
            return
        try:
            router.mark_workflow_failed(workflow_key, reason=details.get("reason"), details=details)
        except Exception:
            logger.exception(
                "Failed to mark workflow %s as failed after watcher timeout", workflow_key
            )

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

MAX_IMAP_AUTH_RETRIES = 3


def send_alert(alert_code: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish an alert event for operational visibility."""

    message = {"alert_code": alert_code}
    if payload:
        message.update(payload)
    try:
        bus = get_event_bus()
    except Exception:
        logger.exception("Failed to initialise event bus for alert %s", alert_code)
        return
    try:
        bus.publish("alerts", message)
    except Exception:
        logger.exception("Failed to publish alert %s", alert_code)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    if value in (None, ""):
        return default
    return value


def _setting(*attr_names: str) -> Optional[str]:
    if not app_settings:
        return None
    for name in attr_names:
        value = getattr(app_settings, name, None)
        if value not in (None, ""):
            return value
    return None


def _resolve_float(env_key: str, setting_keys: Sequence[str], default: float) -> float:
    raw_env = _env(env_key)
    if raw_env not in (None, ""):
        try:
            return float(raw_env)
        except Exception:
            logger.warning("Invalid %s value %s; falling back to %s", env_key, raw_env, default)
    for key in setting_keys:
        raw_setting = _setting(key)
        if raw_setting in (None, ""):
            continue
        try:
            return float(raw_setting)
        except Exception:
            logger.warning("Invalid setting %s=%s; falling back to %s", key, raw_setting, default)
    return default


def _resolve_optional_int(env_key: str, setting_keys: Sequence[str]) -> Optional[int]:
    raw_env = _env(env_key)
    if raw_env not in (None, ""):
        try:
            candidate = int(float(raw_env))
        except Exception:
            logger.warning("Invalid %s value %s; ignoring", env_key, raw_env)
        else:
            if candidate > 0:
                return candidate
    for key in setting_keys:
        raw_setting = _setting(key)
        if raw_setting in (None, ""):
            continue
        try:
            candidate = int(float(raw_setting))
        except Exception:
            logger.warning("Invalid setting %s=%s; ignoring", key, raw_setting)
            continue
        if candidate > 0:
            return candidate
    return None


def _load_dispatch_rows(workflow_id: str) -> List[WorkflowDispatchRow]:
    try:
        workflow_email_tracking_repo.init_schema()
        return workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to load workflow dispatch records for %s", workflow_id)
        return []


def _serialise_response(row: Dict[str, Any], mailbox: Optional[str]) -> Dict[str, Any]:
    return {
        "workflow_id": row.get("workflow_id"),
        "unique_id": row.get("unique_id"),
        "supplier_id": row.get("supplier_id"),
        "rfq_id": row.get("rfq_id"),
        "body_text": row.get("response_text", ""),
        "body_html": row.get("response_body"),
        "subject": row.get("subject"),
        "from_addr": row.get("from_addr"),
        "message_id": row.get("message_id"),
        "received_at": row.get("received_time"),
        "mailbox": mailbox,
        "imap_uid": None,
        "price": row.get("price"),
        "lead_time": row.get("lead_time"),
    }


def _resolve_agent_dependency(
    agent_registry: Optional[Any],
    orchestrator: Optional[Any],
    preferred_keys: Sequence[str],
) -> Optional[Any]:
    """Best-effort resolution of shared agent instances.

    The email watcher operates in long-running background threads where
    instantiating new agent instances is expensive (pulling vector stores,
    DB connections, etc.).  This helper attempts to reuse agents that have
    already been registered with either the provided ``agent_registry`` or
    an ``orchestrator`` instance.  ``preferred_keys`` accepts both snake_case
    and CamelCase identifiers so legacy aliases resolve correctly.
    """

    if not preferred_keys:
        return None

    registries: List[Any] = []
    if agent_registry:
        registries.append(agent_registry)
    if orchestrator:
        orch_registry = getattr(orchestrator, "agents", None)
        if orch_registry:
            registries.append(orch_registry)
        agent_nick = getattr(orchestrator, "agent_nick", None)
        if agent_nick:
            nick_registry = getattr(agent_nick, "agents", None)
            if nick_registry:
                registries.append(nick_registry)

    for registry in registries:
        if registry is None:
            continue
        getter = getattr(registry, "get", None)
        for key in preferred_keys:
            candidate = None
            if callable(getter):
                try:
                    candidate = getter(key)
                except Exception:  # pragma: no cover - defensive
                    candidate = None
            if candidate is None and hasattr(registry, "__getitem__"):
                try:
                    candidate = registry[key]
                except Exception:  # pragma: no cover - registry miss or KeyError
                    candidate = None
            if candidate:
                return candidate

    fallback_sources: List[Any] = []
    if agent_registry:
        fallback_sources.append(agent_registry)
    if orchestrator:
        fallback_sources.append(orchestrator)
        agent_nick = getattr(orchestrator, "agent_nick", None)
        if agent_nick:
            fallback_sources.append(agent_nick)

    for source in fallback_sources:
        if source is None:
            continue
        for key in preferred_keys:
            candidate = getattr(source, key, None)
            if candidate:
                return candidate
            alt = getattr(source, f"{key}_agent", None)
            if alt:
                return alt

    return None


def run_email_watcher_for_workflow(
    *,
    workflow_id: str,
    run_id: Optional[str],
    wait_seconds_after_last_dispatch: int = 90,
    lookback_minutes: int = 240,
    mailbox_name: Optional[str] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[Any] = None,
    supplier_agent: Optional[Any] = None,
    negotiation_agent: Optional[Any] = None,
    process_routing_service: Optional[Any] = None,
    max_workers: int = 8,
    workflow_memory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect supplier responses for ``workflow_id`` using the unified EmailWatcher."""

    _ = run_id  # maintained for backwards compatibility
    _ = agent_registry
    _ = orchestrator
    _ = max_workers
    _ = lookback_minutes
    _ = workflow_memory

    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return {
            "status": "failed",
            "reason": "workflow_id is required",
            "workflow_id": workflow_id,
        }

    dispatch_rows = _load_dispatch_rows(workflow_key)
    if not dispatch_rows:
        return {
            "status": "skipped",
            "reason": f"No recorded dispatches for workflow {workflow_key}",
            "workflow_id": workflow_key,
            "expected": 0,
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
        }

    expected_unique_ids, _, _ = draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
        workflow_id=workflow_key
    )
    expected_unique_ids = {
        (unique_id or "").strip()
        for unique_id in expected_unique_ids
        if unique_id and unique_id.strip()
    }

    rows_by_uid = {
        (row.unique_id or "").strip(): row
        for row in dispatch_rows
        if (row.unique_id or "").strip()
    }

    if not expected_unique_ids:
        logger.debug(
            "Workflow %s has no expected dispatch records from drafting; deferring watcher",
            workflow_key,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "No expected dispatches recorded",
            "workflow_id": workflow_key,
            "expected": len(dispatch_rows),
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": [],
            "missing_required_fields": {},
        }

    missing_expected_rows = sorted(
        uid for uid in expected_unique_ids if uid not in rows_by_uid
    )
    if missing_expected_rows:
        logger.debug(
            "Workflow %s awaiting dispatch rows for %s; deferring watcher",
            workflow_key,
            missing_expected_rows,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": len(expected_unique_ids),
            "found": len(expected_unique_ids) - len(missing_expected_rows),
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": missing_expected_rows,
            "missing_required_fields": {uid: ["dispatch_record"] for uid in missing_expected_rows},
        }

    expected_dispatch_total = len(expected_unique_ids)

    imap_host = _env("IMAP_HOST") or _setting("imap_host")
    imap_user = _env("IMAP_USER")
    imap_username = (
        _env("IMAP_USERNAME")
        or _setting("imap_username", "imap_user", "imap_login")
        or (imap_user.split("@")[0] if imap_user and "@" in imap_user else None)
    )
    imap_password = _env("IMAP_PASSWORD") or _setting("imap_password")
    imap_domain = _env("IMAP_DOMAIN") or _setting("imap_domain")
    imap_login = _env("IMAP_LOGIN") or _setting("imap_login")

    def _normalise_domain(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        domain = value.strip()
        if "@" in domain:
            domain = domain.split("@", 1)[-1]
        return domain or None

    def _pick_login() -> Optional[str]:
        if imap_login:
            return imap_login.strip()
        if imap_user:
            candidate = imap_user.strip()
            if candidate:
                return candidate
        username = (imap_username or "").strip()
        if not username:
            return None
        if "@" in username:
            return username
        domain = _normalise_domain(imap_domain)
        if domain:
            return f"{username}@{domain}"
        return username

    imap_login = _pick_login()
    if not imap_username and imap_login:
        imap_username = imap_login.strip() or None
    try:
        imap_port = int(_env("IMAP_PORT")) if _env("IMAP_PORT") else None
    except Exception:
        imap_port = None
    imap_use_ssl_raw = (
        _env("IMAP_USE_SSL")
        or _env("IMAP_ENCRYPTION")
        or _setting("imap_use_ssl", "imap_encryption")
    )
    imap_use_ssl: Optional[bool]
    if imap_use_ssl_raw is None:
        imap_use_ssl = None
    else:
        raw_value = str(imap_use_ssl_raw).strip().lower()
        if raw_value in {"0", "false", "no", "none", "off"}:
            imap_use_ssl = False
        elif raw_value in {"ssl", "imaps", "true", "1", "yes", "on"}:
            imap_use_ssl = True
        else:
            imap_use_ssl = True
    mailbox = (
        mailbox_name
        or _env("IMAP_MAILBOX")
        or _setting("imap_mailbox")
        or "INBOX"
    )

    if not imap_host or not imap_password or not (imap_username or imap_login):
        logger.warning(
            "IMAP credentials are not configured; skipping EmailWatcher (host=%s user=%s)",
            imap_host,
            imap_username,
        )
        return {
            "status": "skipped",
            "reason": "IMAP credentials not configured",
            "workflow_id": workflow_key,
            "expected": len(dispatch_rows),
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
        }

    missing_required_fields: Dict[str, List[str]] = {}
    missing_message_ids: List[str] = []

    for unique_id in sorted(expected_unique_ids):
        row = rows_by_uid.get(unique_id)
        missing_fields: List[str] = []
        if not row:
            missing_fields.append("dispatch_record")
        else:
            if getattr(row, "dispatched_at", None) is None:
                missing_fields.append("dispatched_at")
            message_id = (row.message_id or "").strip()
            if not message_id:
                missing_message_ids.append(unique_id)
                missing_fields.append("message_id")

        if missing_fields:
            missing_required_fields[unique_id] = missing_fields

        supplier_email = (row.supplier_email or "").strip()
        if not supplier_email:
            missing_fields.append("supplier_email")

        subject_value = (row.subject or "").strip()
        if not subject_value:
            missing_fields.append("subject")

        if missing_fields:
            missing_required_fields[unique_id] = missing_fields
            continue

    if missing_message_ids:
        logger.warning(
            "Workflow %s has dispatches missing message_id: %s",
            workflow_key,
            ", ".join(sorted(missing_message_ids)),
        )

    if missing_required_fields:
        logger.debug(
            "Workflow %s has pending dispatch metadata; deferring watcher start (missing=%s)",
            workflow_key,
            missing_required_fields,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": expected_dispatch_total,
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": sorted(missing_required_fields.keys()),
            "missing_required_fields": missing_required_fields,
        }

    completed_unique_ids = {
        unique_id
        for unique_id in expected_unique_ids
        if unique_id in rows_by_uid
        and (rows_by_uid[unique_id].message_id or "").strip()
        and getattr(rows_by_uid[unique_id], "dispatched_at", None) is not None
    }
    if expected_unique_ids and expected_unique_ids - completed_unique_ids:
        pending = sorted(expected_unique_ids - completed_unique_ids)
        logger.debug(
            "Workflow %s dispatches incomplete; waiting for %s to finish",
            workflow_key,
            pending,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": expected_dispatch_total,
            "found": len(completed_unique_ids),
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": pending,
            "missing_required_fields": missing_required_fields,
        }

    try:
        poll_interval = int(_env("IMAP_POLL_INTERVAL", "30"))
    except Exception:
        poll_interval = 30
    try:
        max_attempts = int(_env("IMAP_MAX_POLL_ATTEMPTS", "10"))
    except Exception:
        max_attempts = 10

    poll_backoff_factor = _resolve_float(
        "IMAP_POLL_BACKOFF_FACTOR",
        ("imap_poll_backoff_factor", "poll_backoff_factor"),
        1.8,
    )
    poll_jitter_seconds = max(
        0.0,
        _resolve_float(
            "IMAP_POLL_JITTER_SECONDS",
            ("imap_poll_jitter_seconds", "poll_jitter_seconds"),
            2.0,
        ),
    )
    poll_max_interval_default = float(max(poll_interval * 6, poll_interval))
    poll_max_interval = _resolve_float(
        "IMAP_POLL_MAX_INTERVAL",
        ("imap_poll_max_interval", "poll_max_interval_seconds"),
        poll_max_interval_default,
    )
    poll_max_interval_seconds = max(int(poll_max_interval), poll_interval)
    poll_timeout_seconds = _resolve_optional_int(
        "IMAP_POLL_TIMEOUT_SECONDS",
        ("imap_poll_timeout_seconds", "poll_timeout_seconds"),
    )

    supplier_agent = supplier_agent or _resolve_agent_dependency(
        agent_registry, orchestrator, ("supplier_interaction", "SupplierInteractionAgent")
    )
    negotiation_agent = negotiation_agent or _resolve_agent_dependency(
        agent_registry, orchestrator, ("negotiation", "NegotiationAgent")
    )
    process_routing_service = process_routing_service or _resolve_agent_dependency(
        agent_registry,
        orchestrator,
        ("process_routing_service", "process_router", "ProcessRoutingService"),
    )
    if process_routing_service is None and supplier_agent is not None:
        process_routing_service = getattr(supplier_agent, "process_routing_service", None)

    response_grace_seconds = _resolve_optional_int(
        "EMAIL_WATCHER_RESPONSE_GRACE_SECONDS",
        ("email_response_grace_seconds", "response_grace_seconds"),
    )
    default_grace_field = EmailWatcherConfig.__dataclass_fields__["response_grace_seconds"]
    default_grace_seconds = default_grace_field.default
    if default_grace_seconds is MISSING:
        default_grace_seconds = 180

    config = EmailWatcherConfig(
        imap_host=imap_host,
        imap_username=imap_username,
        imap_password=imap_password,
        imap_port=imap_port or 993,
        imap_use_ssl=True if imap_use_ssl is None else imap_use_ssl,
        imap_login=imap_login,
        imap_mailbox=mailbox or "INBOX",
        dispatch_wait_seconds=max(0, int(wait_seconds_after_last_dispatch)),
        poll_interval_seconds=max(1, poll_interval),
        max_poll_attempts=max(1, max_attempts),
        poll_backoff_factor=max(1.0, poll_backoff_factor),
        poll_jitter_seconds=poll_jitter_seconds,
        poll_max_interval_seconds=poll_max_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        response_grace_seconds=response_grace_seconds
        if response_grace_seconds is not None
        else int(default_grace_seconds),
    )
    watcher = EmailWatcher(
        config=config,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        process_router=process_routing_service,
        sleep=time.sleep,
    )
    logger.info(
        "watcher_started workflow=%s expected=%s status=active",
        workflow_key,
        expected_dispatch_total,
    )
    logger.info(
        "watcher_started workflow=%s expected=%s status=active",
        workflow_key,
        expected_dispatch_total,
    )
    try:
        result = watcher.wait_for_responses(workflow_key)
    except imaplib.IMAP4.error as exc:
        logger.error(
            "IMAP authentication failed for host=%s user=%s: %s",
            imap_host,
            imap_login or imap_username,
            exc,
        )
        return {
            "status": "failed",
            "reason": "IMAP authentication failed",
            "workflow_id": workflow_key,
            "expected": len(dispatch_rows),
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
        }
    expected = result.get("dispatched_count", expected_dispatch_total)
    responded = result.get("responded_count", 0)
    matched = result.get("matched_responses", {}) or {}
    matched_ids = sorted(matched.keys())

    pending_rows = supplier_response_repo.fetch_pending(workflow_id=workflow_key)
    responses = [_serialise_response(row, mailbox) for row in pending_rows]

    watcher_status = result.get("status") or (
        "completed" if result.get("complete") else "pending"
    )
    status = "processed" if watcher_status == "completed" else watcher_status

    response_payload = {
        "status": status,
        "watcher_status": watcher_status,
        "workflow_id": workflow_key,
        "expected": expected,
        "found": responded,
        "rows": responses,
        "matched_unique_ids": matched_ids,
    }
    if workflow_status:
        response_payload["workflow_status"] = workflow_status
    if "expected_responses" in result:
        response_payload["expected_responses"] = result.get("expected_responses")
    if "elapsed_seconds" in result:
        response_payload["elapsed_seconds"] = result.get("elapsed_seconds")
    if "timeout_reached" in result:
        response_payload["timeout_reached"] = bool(result.get("timeout_reached"))
    if "pending_suppliers" in result:
        response_payload["pending_suppliers"] = result.get("pending_suppliers")
    if "pending_unique_ids" in result:
        response_payload["pending_unique_ids"] = result.get("pending_unique_ids")
    if "last_capture_ts" in result:
        response_payload["last_capture_ts"] = result.get("last_capture_ts")
    if "timeout_deadline" in result:
        response_payload["timeout_deadline"] = result.get("timeout_deadline")

    if result.get("pending_unique_ids"):
        response_payload["pending_unique_ids"] = list(result["pending_unique_ids"])
    if result.get("pending_suppliers"):
        response_payload["pending_suppliers"] = list(result["pending_suppliers"])
    if result.get("timeout_reason"):
        response_payload["timeout_reason"] = result["timeout_reason"]

    expected_ids = expected_unique_ids or {
        row.unique_id for row in dispatch_rows if row.unique_id
    }
    if status not in {"processed", "completed"}:
        missing = sorted(expected_ids - set(matched_ids))
        response_payload["reason"] = (
            "Not all responses received"
            if missing
            else "Responses still pending"
        )
        response_payload["missing_unique_ids"] = missing
        return response_payload

    if matched_ids:
        try:
            supplier_response_repo.delete_responses(
                workflow_id=workflow_key,
                unique_ids=matched_ids,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to remove supplier responses for workflow %s",
                workflow_key,
            )

    return response_payload


class EmailWatcherService:
    """Run the unified :class:`EmailWatcher` as a background polling service."""

    def __init__(
        self,
        *,
        poll_interval_seconds: Optional[int] = None,
        post_dispatch_interval_seconds: Optional[int] = None,
        dispatch_wait_seconds: Optional[int] = None,
        watcher_runner: Optional[Callable[..., Dict[str, object]]] = None,
        agent_registry: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        supplier_agent: Optional[Any] = None,
        negotiation_agent: Optional[Any] = None,
        process_routing_service: Optional[Any] = None,
    ) -> None:
        if poll_interval_seconds is None:
            poll_interval_seconds = self._env_int("EMAIL_WATCHER_SERVICE_INTERVAL", fallback="90")
        if poll_interval_seconds <= 0:
            poll_interval_seconds = 90

        if post_dispatch_interval_seconds is None:
            post_dispatch_interval_seconds = self._env_int(
                "EMAIL_WATCHER_SERVICE_POST_DISPATCH_INTERVAL", fallback="30"
            )
        if post_dispatch_interval_seconds <= 0:
            post_dispatch_interval_seconds = 30

        if dispatch_wait_seconds is None:
            dispatch_wait_seconds = self._env_int("EMAIL_WATCHER_SERVICE_DISPATCH_WAIT", fallback="0")
        if dispatch_wait_seconds < 0:
            dispatch_wait_seconds = 0

        self._poll_interval = poll_interval_seconds
        self._post_dispatch_interval = post_dispatch_interval_seconds
        self._dispatch_wait = dispatch_wait_seconds
        self._runner: Callable[..., Dict[str, object]] = watcher_runner or run_email_watcher_for_workflow
        self._agent_registry = agent_registry
        self._orchestrator = orchestrator
        self._supplier_agent = supplier_agent
        self._negotiation_agent = negotiation_agent
        self._process_router = process_routing_service
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._forced_lock = threading.Lock()
        self._forced_workflows: Set[str] = set()

    def watch_workflow(
        self,
        *,
        workflow_id: str,
        run_id: Optional[str],
        wait_seconds_after_last_dispatch: int = 0,
        lookback_minutes: int = 240,
        mailbox_name: Optional[str] = None,
        agent_registry: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        supplier_agent: Optional[Any] = None,
        negotiation_agent: Optional[Any] = None,
        process_routing_service: Optional[Any] = None,
        max_workers: int = 8,
        workflow_memory: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the watcher synchronously for ``workflow_id``.

        This helper mirrors :func:`run_email_watcher_for_workflow` while
        defaulting to dependencies captured by the service instance.  It is
        primarily used by orchestration flows that need to force a watcher
        pass immediately after dispatch without waiting for the background
        loop.
        """

        return self._runner(
            workflow_id=workflow_id,
            run_id=run_id,
            wait_seconds_after_last_dispatch=wait_seconds_after_last_dispatch,
            lookback_minutes=lookback_minutes,
            mailbox_name=mailbox_name,
            agent_registry=agent_registry or self._agent_registry,
            orchestrator=orchestrator or self._orchestrator,
            supplier_agent=supplier_agent or self._supplier_agent,
            negotiation_agent=negotiation_agent or self._negotiation_agent,
            process_routing_service=(
                process_routing_service or self._process_router
            ),
            max_workers=max_workers,
            workflow_memory=workflow_memory,
        )

    @staticmethod
    def _env_int(name: str, *, fallback: str) -> int:
        try:
            return int(os.environ.get(name, fallback))
        except Exception:
            return int(fallback)

    def start(self) -> None:
        """Start the watcher loop if it is not already running."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="EmailWatcherService", daemon=True)
        self._thread.start()
        logger.info(
            "EmailWatcherService started (poll_interval=%ss post_dispatch_interval=%ss dispatch_wait=%ss)",
            self._poll_interval,
            self._post_dispatch_interval,
            self._dispatch_wait,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher loop to stop and wait for the thread."""

        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("EmailWatcherService stopped")

    def notify_workflow(self, workflow_id: str) -> None:
        """Wake the service to prioritise ``workflow_id`` in the next cycle."""

        workflow_key = (workflow_id or "").strip()
        if not workflow_key:
            return

        self._preempt_existing_workflows(workflow_key)
        if self._thread and self._thread.is_alive():
            self.stop()
            self.start()
        with self._forced_lock:
            self._forced_workflows.add(workflow_key)
        self._wake_event.set()

    def _consume_forced_workflows(self) -> List[str]:
        with self._forced_lock:
            items = list(self._forced_workflows)
            self._forced_workflows.clear()
        return items

    def _preempt_existing_workflows(self, new_workflow_id: str) -> None:
        try:
            active_workflows = workflow_email_tracking_repo.load_active_workflow_ids()
        except Exception:
            logger.exception(
                "EmailWatcherService failed to enumerate active workflows before preemption"
            )
            return

        seen: Set[str] = set()
        for workflow_id in active_workflows:
            workflow_key = (workflow_id or "").strip()
            if not workflow_key or workflow_key == new_workflow_id:
                continue
            if workflow_key in seen:
                continue
            seen.add(workflow_key)
            try:
                workflow_email_tracking_repo.reset_workflow(workflow_key)
                workflow_lifecycle_repo.record_watcher_event(
                    workflow_key,
                    "watcher_stopped",
                    expected_responses=0,
                    received_responses=0,
                    metadata={
                        "stop_reason": "preempted_by_new_workflow",
                        "preempted_by": new_workflow_id,
                    },
                )
                logger.info(
                    "EmailWatcherService preempted workflow=%s in favour of workflow=%s",
                    workflow_key,
                    new_workflow_id,
                )
            except Exception:
                logger.exception(
                    "EmailWatcherService failed to preempt workflow=%s before starting workflow=%s",
                    workflow_key,
                    new_workflow_id,
                )
        if seen:
            with self._forced_lock:
                self._forced_workflows = {
                    wf for wf in self._forced_workflows if wf == new_workflow_id
                }

    def _should_skip_workflow(self, workflow_id: str) -> bool:
        """Return ``True`` when ``workflow_id`` should not be processed."""

        workflow_key = (workflow_id or "").strip()
        if not workflow_key:
            return False

        try:
            lifecycle = workflow_lifecycle_repo.get_lifecycle(workflow_key)
        except Exception:
            logger.exception(
                "EmailWatcherService failed to load lifecycle for workflow=%s",
                workflow_key,
            )
            return False

        if not lifecycle:
            return False

        negotiation_status = str(
            lifecycle.get("negotiation_status") or ""
        ).strip().lower()
        if negotiation_status not in {"completed", "finalized"}:
            return False

        watcher_status = str(lifecycle.get("watcher_status") or "").strip().lower()
        if watcher_status != "stopped":
            return False

        metadata = lifecycle.get("metadata") or {}
        stop_reason = str(metadata.get("stop_reason") or "").strip().lower()
        if stop_reason in {
            "negotiation_completed",
            "negotiation_completed_pending_responses",
        }:
            logger.debug(
                "EmailWatcherService suppressing workflow=%s due to completed negotiation",
                workflow_key,
            )
            return True

        return False

    def _wait_for_next_cycle(self, seconds: float) -> None:
        if seconds <= 0:
            seconds = 0
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            awakened = self._wake_event.wait(timeout=remaining)
            if awakened:
                self._wake_event.clear()
                break

    def update_dependencies(
        self,
        *,
        agent_registry: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        supplier_agent: Optional[Any] = None,
        negotiation_agent: Optional[Any] = None,
    ) -> None:
        """Refresh shared dependency references used by the watcher."""

        if agent_registry is not None:
            self._agent_registry = agent_registry
        if orchestrator is not None:
            self._orchestrator = orchestrator
        if supplier_agent is not None:
            self._supplier_agent = supplier_agent
        if negotiation_agent is not None:
            self._negotiation_agent = negotiation_agent

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            waiting_for_dispatch = False
            processed_workflow = False
            try:
                workflow_ids = workflow_email_tracking_repo.load_active_workflow_ids()
            except Exception:
                logger.exception("Failed to load workflows for email watcher service")
                workflow_ids = []

            forced = self._consume_forced_workflows()
            if forced:
                seen = set(workflow_ids)
                for workflow_id in forced:
                    if workflow_id not in seen:
                        workflow_ids.append(workflow_id)
                        seen.add(workflow_id)

            for workflow_id in workflow_ids:
                if self._stop_event.is_set():
                    break

                if not workflow_id:
                    continue

                if self._should_skip_workflow(workflow_id):
                    processed_workflow = True
                    continue

                try:
                    result = self._runner(
                        workflow_id=workflow_id,
                        run_id=None,
                        wait_seconds_after_last_dispatch=self._dispatch_wait,
                        agent_registry=self._agent_registry,
                        orchestrator=self._orchestrator,
                        supplier_agent=self._supplier_agent,
                        negotiation_agent=self._negotiation_agent,
                        process_routing_service=self._process_router,
                    )
                    status = str(result.get("status") or "").lower()
                    processed_workflow = True
                    if status == "failed":
                        logger.error(
                            "Email watcher service failed for workflow=%s: %s",
                            workflow_id,
                            result.get("reason") or result.get("error"),
                        )
                    elif status == "waiting_for_dispatch":
                        waiting_for_dispatch = True
                except Exception:
                    logger.exception("Email watcher service encountered an error for workflow %s", workflow_id)

            if waiting_for_dispatch or not processed_workflow:
                sleep_seconds = self._poll_interval
            else:
                sleep_seconds = self._post_dispatch_interval

            self._wait_for_next_cycle(sleep_seconds)
            if self._stop_event.is_set():
                break

        logger.debug("EmailWatcherService loop terminated")

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
        match_threshold: float = 0.8,
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

__all__ = []
