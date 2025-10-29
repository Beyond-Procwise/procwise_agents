"""Workflow-centric email watcher implementation (EmailWatcherV2)."""

from __future__ import annotations

import calendar
import email
import hashlib
import imaplib
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from difflib import SequenceMatcher

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from repositories import (
    draft_rfq_emails_repo,
    email_dispatch_repo,
    email_watcher_state_repo,
    supplier_interaction_repo,
    supplier_response_repo,
    workflow_email_tracking_repo as tracking_repo,
    workflow_lifecycle_repo,
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
from services.event_bus import get_event_bus
from services.supplier_response_coordinator import get_supplier_response_coordinator

logger = logging.getLogger(__name__)


_PRICE_PATTERN = re.compile(
    r"(?P<currency>USD|EUR|GBP|AUD|CAD|INR|JPY|CHF|RMB|CNY|\$|€|£)?\s*" r"(?P<amount>\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_LEAD_TIME_PATTERN = re.compile(
    r"(?P<value>\d{1,3})\s*(?P<unit>day|days|business day|business days|week|weeks)",
    re.IGNORECASE,
)
_PAYMENT_TERMS_PATTERN = re.compile(
    r"net\s*\d{1,3}|payment\s*terms?:\s*[^\n\.]+", re.IGNORECASE
)
_WARRANTY_PATTERN = re.compile(r"warranty[^\n\.]*", re.IGNORECASE)
_VALIDITY_PATTERN = re.compile(r"valid(?:ity)?[^\n\.]*", re.IGNORECASE)
_EXCEPTION_PATTERN = re.compile(r"except(?:ion|ions)?[^\n\.]*", re.IGNORECASE)


def _extract_price_fields(text: str) -> Tuple[Optional[Decimal], Optional[str]]:
    match = _PRICE_PATTERN.search(text)
    if not match:
        return None, None
    amount = match.group("amount")
    currency = match.group("currency")
    if amount:
        normalised = amount.replace(",", "").replace(" ", "")
        try:
            price = Decimal(normalised)
        except (InvalidOperation, ValueError):  # pragma: no cover - defensive
            price = None
    else:
        price = None
    if currency:
        symbol = currency.upper()
        currency_map = {"$": "USD", "€": "EUR", "£": "GBP"}
        currency = currency_map.get(currency) or currency_map.get(symbol) or symbol
    return price, currency


def _extract_lead_time(text: str) -> Optional[int]:
    match = _LEAD_TIME_PATTERN.search(text)
    if not match:
        return None
    try:
        value = int(match.group("value"))
    except (TypeError, ValueError):
        return None
    unit = match.group("unit").lower()
    if "week" in unit:
        return value * 7
    return value


def _find_pattern(text: str, pattern: re.Pattern[str]) -> Optional[str]:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(0).strip()


def extract_structured_fields(email: EmailResponse) -> Dict[str, Any]:
    text_segments = [email.body_text or "", email.body_html or ""]
    combined = "\n".join(segment for segment in text_segments if segment)
    price, currency = _extract_price_fields(combined)
    lead_time_days = _extract_lead_time(combined)
    payment_terms = _find_pattern(combined, _PAYMENT_TERMS_PATTERN)
    warranty = _find_pattern(combined, _WARRANTY_PATTERN)
    validity = _find_pattern(combined, _VALIDITY_PATTERN)
    exceptions = _find_pattern(combined, _EXCEPTION_PATTERN)
    return {
        "price": price,
        "currency": currency,
        "lead_time_days": lead_time_days,
        "payment_terms": payment_terms,
        "warranty": warranty,
        "validity": validity,
        "exceptions": exceptions,
        "attachments": list(email.attachments),
        "tables": list(email.tables),
    }


@dataclass
class EmailDispatchRecord:
    workflow_id: Optional[str]
    unique_id: str
    dispatch_key: str
    supplier_id: Optional[str]
    supplier_email: Optional[str]
    message_id: Optional[str]
    subject: Optional[str]
    rfq_id: Optional[str] = None
    thread_headers: Dict[str, Sequence[str]] = field(default_factory=dict)
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
    headers: Dict[str, Sequence[str]] = field(default_factory=dict)
    attachments: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    tables: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    round_number: Optional[int] = None


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
    supplier_unique_map: Dict[str, Set[str]] = field(default_factory=dict)
    expected_supplier_ids: Set[str] = field(default_factory=set)
    expected_unique_ids: Set[str] = field(default_factory=set)
    placeholder_unique_ids: Set[str] = field(default_factory=set)
    all_dispatched: bool = False
    all_responded: bool = False
    last_dispatched_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    last_response_at: Optional[datetime] = None
    last_capture_at: Optional[datetime] = None
    timeout_deadline: Optional[datetime] = None
    round_index: Dict[str, Optional[int]] = field(default_factory=dict)
    completion_logged: bool = False
    seen_message_ids: Set[str] = field(default_factory=set)
    seen_fingerprints: Set[str] = field(default_factory=set)
    status: str = "initializing"

    def _refresh_dispatch_state(self) -> None:
        actual_unique_ids = {
            unique_id for unique_id, records in self.email_records.items() if records
        }
        self.dispatched_count = len(actual_unique_ids)
        if self.expected_unique_ids:
            self.all_dispatched = self.expected_unique_ids.issubset(actual_unique_ids)
        else:
            self.all_dispatched = bool(actual_unique_ids)

    def register_dispatches(self, dispatches: Iterable[EmailDispatchRecord]) -> None:
        for dispatch in dispatches:
            bucket = self.email_records.setdefault(dispatch.unique_id, [])
            if dispatch.unique_id in self.placeholder_unique_ids:
                self.placeholder_unique_ids.discard(dispatch.unique_id)
            bucket.append(dispatch)
            bucket.sort(key=lambda item: item.dispatched_at or datetime.min)
            supplier_key = dispatch.supplier_id or ""
            if supplier_key:
                self.expected_supplier_ids.add(supplier_key)
                self.supplier_unique_map.setdefault(supplier_key, set()).add(
                    dispatch.unique_id
                )
            self.expected_unique_ids.add(dispatch.unique_id)
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
        self._refresh_dispatch_state()
        supplier_total = len(self.expected_supplier_ids)
        if supplier_total > self.expected_responses:
            self.expected_responses = supplier_total
        self.all_responded = self.responded_count >= self.expected_responses > 0

    def register_expected_unique_ids(
        self,
        unique_ids: Iterable[str],
        supplier_index: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        supplier_index = supplier_index or {}
        updated = False
        for unique_id in unique_ids:
            uid = str(unique_id).strip()
            if not uid:
                continue
            if uid not in self.expected_unique_ids:
                self.expected_unique_ids.add(uid)
                updated = True
            supplier_hint = supplier_index.get(uid)
            if supplier_hint:
                supplier_key = str(supplier_hint).strip()
                if supplier_key:
                    self.expected_supplier_ids.add(supplier_key)
                    self.supplier_unique_map.setdefault(supplier_key, set()).add(uid)
            if uid not in self.email_records:
                self.email_records[uid] = []
                self.placeholder_unique_ids.add(uid)
        if updated:
            self.expected_responses = max(
                self.expected_responses,
                len(self.expected_unique_ids),
                len(self.expected_supplier_ids),
                self.responded_count,
            )
        self._refresh_dispatch_state()
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

    def set_expected_responses(self, expected: Optional[int]) -> None:
        if expected is None:
            return
        try:
            value = int(expected)
        except Exception:  # pragma: no cover - defensive conversion
            return
        if value < 0:
            value = 0
        minimum = max(
            len(self.expected_unique_ids),
            len(self.expected_supplier_ids),
        )
        self.expected_responses = max(value, self.responded_count, minimum)
        self.all_responded = self.responded_count >= self.expected_responses > 0

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


def _extract_plain_text(message: EmailMessage) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    plain_text: Optional[str] = None
    html_text: Optional[str] = None
    attachments: List[Dict[str, Any]] = []
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            filename = part.get_filename()
            is_attachment = "attachment" in disp or bool(filename)
            if is_attachment:
                payload = part.get_payload(decode=True) or b""
                size = len(payload) if isinstance(payload, (bytes, bytearray)) else None
                attachments.append(
                    {
                        "filename": filename or "",
                        "content_type": ctype,
                        "size": size,
                    }
                )
                continue
            try:
                content = part.get_content()
            except Exception:  # pragma: no cover - defensive
                continue
            if content is None:
                continue
            if ctype == "text/plain" and plain_text is None:
                plain_text = str(content).strip()
            elif ctype == "text/html" and html_text is None:
                html_text = str(content)
    else:
        try:
            content = message.get_content()
        except Exception:  # pragma: no cover - defensive
            content = None
        if content is not None:
            if (message.get_content_type() or "").lower() == "text/html":
                html_text = str(content)
            else:
                plain_text = str(content)
    if plain_text is None and html_text is not None:
        plain_text = html_text
    return (plain_text or "", html_text, attachments)


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._current_row: Optional[List[str]] = None
        self._current_cell: Optional[str] = None
        self._header_phase: bool = False

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag_lower = tag.lower()
        if tag_lower == "table":
            self._current = {"headers": [], "rows": []}
            self._current_row = None
            self._current_cell = None
            self._header_phase = True
        elif tag_lower == "tr" and self._current is not None:
            self._current_row = []
        elif tag_lower in {"th", "td"} and self._current is not None:
            self._current_cell = ""
            self._header_phase = self._header_phase and tag_lower == "th"

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._current_cell is not None:
            self._current_cell += data

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag_lower = tag.lower()
        if tag_lower in {"th", "td"} and self._current is not None:
            if self._current_cell is not None:
                text = self._current_cell.strip()
                if self._header_phase and tag_lower == "th":
                    self._current.setdefault("headers", []).append(text)
                elif self._current_row is not None:
                    self._current_row.append(text)
            self._current_cell = None
        elif tag_lower == "tr" and self._current is not None:
            if self._current_row:
                self._current.setdefault("rows", []).append(self._current_row)
            self._current_row = None
            self._header_phase = False
        elif tag_lower == "table" and self._current is not None:
            if self._current.get("headers") or self._current.get("rows"):
                self.tables.append(self._current)
            self._current = None
            self._current_row = None
            self._current_cell = None
            self._header_phase = False


def _extract_tables_from_html(html_text: Optional[str]) -> List[Dict[str, Any]]:
    if not html_text:
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(html_text)
    except Exception:  # pragma: no cover - defensive parsing
        return []
    return parser.tables


def _parse_email(raw: bytes) -> EmailResponse:
    message = _decode_message(raw)
    subject = message.get("Subject")
    message_id = (message.get("Message-ID") or "").strip("<> ") or None
    from_address = message.get("From")
    plain_body, html_body, attachments = _extract_plain_text(message)
    tables = _extract_tables_from_html(html_body)
    body = plain_body or html_body or ""
    header_aliases = (
        "X-ProcWise-Unique-ID",
        "X-Procwise-Unique-Id",
        "X-Procwise-Unique-ID",
        "X-Procwise-Uid",
    )
    header_map = {key: message.get_all(key, failobj=[]) for key in header_aliases}
    unique_id = extract_unique_id_from_headers(header_map)
    if not unique_id:
        for alias in header_aliases:
            fallback_header = message.get(alias)
            if fallback_header:
                unique_id = str(fallback_header).strip()
                if unique_id:
                    break
    body_unique = extract_unique_id_from_body(body)
    if body_unique and not unique_id:
        unique_id = body_unique
    metadata = extract_tracking_metadata(body)
    supplier_id = metadata.supplier_id if metadata else None
    if metadata and not unique_id:
        unique_id = metadata.unique_id

    header_unique_id = (
        message.get("X-ProcWise-Unique-ID")
        or message.get("X-Procwise-Unique-Id")
        or ""
    ).strip()
    if header_unique_id and not unique_id:
        unique_id = header_unique_id

    header_supplier_id = (
        message.get("X-ProcWise-Supplier-ID")
        or message.get("X-Procwise-Supplier-Id")
        or ""
    ).strip()
    if header_supplier_id and not supplier_id:
        supplier_id = header_supplier_id

    workflow_id = metadata.workflow_id if metadata else None
    header_workflow_id = (
        message.get("X-ProcWise-Workflow-ID")
        or message.get("X-Procwise-Workflow-Id")
        or ""
    ).strip()
    if header_workflow_id and not workflow_id:
        workflow_id = header_workflow_id

    rfq_id = (message.get("X-Procwise-RFQ-ID") or "").strip() or None
    round_header = (
        message.get("X-ProcWise-Round")
        or message.get("X-Procwise-Round")
        or ""
    ).strip()
    round_number: Optional[int] = None
    if round_header:
        try:
            parsed_round = int(round_header)
        except ValueError:
            parsed_round = None
        if parsed_round is not None and parsed_round >= 0:
            round_number = parsed_round

    date_header = message.get("Date")
    try:
        received_at = email.utils.parsedate_to_datetime(date_header) if date_header else None
    except Exception:  # pragma: no cover - defensive
        received_at = None
    received_at = received_at or datetime.now(timezone.utc)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)

    thread_ids = _extract_thread_ids(message)
    header_names = list(dict.fromkeys(message.keys()))
    raw_headers: Dict[str, Sequence[str]] = {}
    for header in header_names:
        values = [
            str(value).strip()
            for value in message.get_all(header, failobj=[])
            if str(value).strip()
        ]
        if values:
            raw_headers[str(header)] = tuple(values)

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
        headers=raw_headers,
        attachments=tuple(attachments),
        tables=tuple(tables),
        round_number=round_number,
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


def _normalise_subject_line(subject: Optional[str]) -> Optional[str]:
    """Normalise a subject line for deduplication and logging."""

    if not subject:
        return None
    value = str(subject).strip()
    if not value:
        return None
    value = re.sub(r"^(?:(?i:(re|fw|fwd))\s*:)+", "", value).strip()
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value or None


def _subject_similarity(left: Optional[str], right: Optional[str]) -> float:
    """Compute a similarity ratio between two subject lines."""

    left_norm = _normalise_subject_line(left)
    right_norm = _normalise_subject_line(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    try:
        return SequenceMatcher(None, left_norm, right_norm).ratio()
    except Exception:  # pragma: no cover - defensive guard
        return 0.0


def _subject_hash(subject: Optional[str]) -> Optional[str]:
    normalised = _normalise_subject_line(subject)
    if not normalised:
        return None
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def _format_message_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)) and parsed:
            first = parsed[0]
            if isinstance(first, str):
                text = first.strip()
    text = text.strip("<>").strip()
    if not text:
        return None
    return f"<{text}>"


def _normalise_message_token(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text.strip() or None


def _response_fingerprint(email: EmailResponse) -> Optional[str]:
    subject_norm = _normalise_subject_line(email.subject) or ""
    from_norm = _normalise_email_address(email.from_address) or ""
    received_at = email.received_at or datetime.now(timezone.utc)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)
    bucket = int(received_at.timestamp() // 120)
    payload = "|".join([subject_norm, from_norm, str(bucket)])
    if not payload.strip("|"):
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_round_number(value: Optional[object]) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _coerce_last_uid(value: Optional[object]) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _coerce_responses(candidate: Optional[object]) -> List[EmailResponse]:
    if candidate is None:
        return []
    if isinstance(candidate, EmailResponse):
        return [candidate]
    if isinstance(candidate, list):
        return [item for item in candidate if isinstance(item, EmailResponse)]
    if isinstance(candidate, tuple):
        return [item for item in candidate if isinstance(item, EmailResponse)]
    if isinstance(candidate, set):
        return [item for item in candidate if isinstance(item, EmailResponse)]
    if isinstance(candidate, Iterable):
        return [item for item in candidate if isinstance(item, EmailResponse)]
    return []


def _normalise_fetch_result(payload: Optional[object]) -> Tuple[List[EmailResponse], Optional[int]]:
    if payload is None:
        return [], None

    if isinstance(payload, tuple) and len(payload) == 2:
        responses = _coerce_responses(payload[0])
        last_uid = _coerce_last_uid(payload[1])
        if responses or last_uid is not None:
            return responses, last_uid

    if isinstance(payload, dict):
        responses = _coerce_responses(
            payload.get("responses")
            or payload.get("emails")
            or payload.get("messages")
        )
        last_uid = _coerce_last_uid(
            payload.get("last_uid")
            or payload.get("uid")
            or payload.get("last_seen_uid")
            or payload.get("max_uid")
        )
        return responses, last_uid

    if isinstance(payload, EmailResponse):
        return [payload], None

    if isinstance(payload, Iterable):
        return [item for item in payload if isinstance(item, EmailResponse)], None

    return [], None


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
    last_seen_uid: Optional[int] = None,
    backtrack_seconds: int = 180,
    max_uid_samples: int = 200,
    workflow_ids: Optional[Sequence[str]] = None,
    mailbox_filter: Optional[str] = None,
    **_: object,
) -> Tuple[List[EmailResponse], Optional[int]]:
    client = _imap_client(host, username, password, port=port, use_ssl=use_ssl, login=login)
    try:
        client.select(mailbox, readonly=True)
        candidate_uids: Set[int] = set()
        unseen_uids: Set[int] = set()
        workflow_tokens: List[str] = []
        if workflow_ids:
            for token in workflow_ids:
                if token in (None, ""):
                    continue
                text = str(token).strip()
                if text:
                    workflow_tokens.append(text)
        mailbox_header = str(mailbox_filter).strip() if mailbox_filter else ""

        def _search_uids(*terms: str, mark_unseen: bool = False) -> None:
            search_terms = list(terms)
            if mailbox_header:
                search_terms.extend(["HEADER", "X-ProcWise-Mailbox", mailbox_header])
            typ, data = client.uid("SEARCH", None, *search_terms)
            if typ != "OK":
                return
            payload = data[0] or b""
            if isinstance(payload, bytes):
                parts = payload.split()
            else:  # pragma: no cover - defensive
                parts = []
            for part in parts:
                if not part:
                    continue
                if isinstance(part, bytes):
                    token = part.decode(errors="ignore")
                else:
                    token = str(part)
                if token.isdigit():
                    value = int(token)
                    candidate_uids.add(value)
                    if mark_unseen:
                        unseen_uids.add(value)

        since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        since_floor = since_aware - timedelta(seconds=max(0, backtrack_seconds))
        since_str = since_floor.strftime("%d-%b-%Y")

        if workflow_tokens:
            for token in workflow_tokens:
                _search_uids(
                    "UNSEEN",
                    "SINCE",
                    since_str,
                    "HEADER",
                    "X-ProcWise-Workflow-ID",
                    token,
                    mark_unseen=True,
                )
        else:
            _search_uids("UNSEEN", "SINCE", since_str, mark_unseen=True)
        if last_seen_uid is not None:
            _search_uids("UID", f"{last_seen_uid + 1}:*")

        if workflow_tokens:
            for token in workflow_tokens:
                _search_uids("SINCE", since_str, "HEADER", "X-ProcWise-Workflow-ID", token)
        else:
            _search_uids("SINCE", since_str)

        if not candidate_uids:
            _search_uids("ALL")

        if not candidate_uids:
            return [], last_seen_uid

        candidate_list = sorted(candidate_uids)
        if max_uid_samples > 0 and len(candidate_list) > max_uid_samples:
            candidate_list = candidate_list[-max_uid_samples:]

        responses: List[EmailResponse] = []
        max_uid_value = last_seen_uid or 0
        lower_bound = since_aware - timedelta(seconds=max(backtrack_seconds, 86400))

        for uid_value in candidate_list:
            typ, payload = client.uid("FETCH", str(uid_value), "(RFC822 INTERNALDATE)")
            if typ != "OK" or not payload:
                continue

            raw_bytes: Optional[bytes] = None
            internal_dt: Optional[datetime] = None

            for part in payload:
                if not isinstance(part, tuple) or len(part) < 2:
                    continue
                meta, content = part[0], part[1]
                if isinstance(meta, (bytes, bytearray)):
                    try:
                        internal_tuple = imaplib.Internaldate2tuple(meta)
                    except Exception:
                        internal_tuple = None
                    if internal_tuple:
                        timestamp = calendar.timegm(internal_tuple)
                        internal_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                if isinstance(content, (bytes, bytearray)):
                    raw_bytes = bytes(content)

            if raw_bytes is None:
                continue

            email_response = _parse_email(raw_bytes)

            if internal_dt:
                email_response.received_at = internal_dt
            elif email_response.received_at is None:
                email_response.received_at = datetime.now(timezone.utc)

            received_at = email_response.received_at
            if received_at and received_at.tzinfo is None:
                received_at = received_at.replace(tzinfo=timezone.utc)
                email_response.received_at = received_at

            if (
                received_at
                and uid_value not in unseen_uids
                and received_at < lower_bound
            ):
                continue

            responses.append(email_response)
            if uid_value > max_uid_value:
                max_uid_value = uid_value

        responses.sort(key=lambda response: response.received_at or since_aware)

        return responses, (max_uid_value if max_uid_value else last_seen_uid)
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


def _calculate_match_score(
    dispatch: EmailDispatchRecord, email_response: EmailResponse
) -> Tuple[float, List[str]]:
    if email_response.unique_id and email_response.unique_id == dispatch.unique_id:
        return 1.0, ["unique_id"]

    matched_on: List[str] = []
    score = 0.0

    WORKFLOW_WEIGHT = 0.6
    UNIQUE_WEIGHT = 0.2
    SUPPLIER_WEIGHT = 0.1
    ROUND_WEIGHT = 0.1
    THREAD_WEIGHT = 0.5
    SUBJECT_WEIGHT = 0.3
    SUBJECT_SIMILARITY_THRESHOLD = 0.6

    def _normalise(value: Optional[object]) -> Optional[str]:
        if value in (None, ""):
            return None
        return str(value).strip().lower() or None

    header_map = {key.lower(): tuple(value) for key, value in email_response.headers.items()}

    def _collect_header_values(*keys: str) -> List[str]:
        collected: List[str] = []
        for key in keys:
            for entry in header_map.get(key.lower(), ()):  # type: ignore[assignment]
                text = str(entry).strip()
                if text:
                    collected.append(text)
        return collected

    dispatch_workflow = _normalise(dispatch.workflow_id)
    workflow_candidates = {
        value
        for value in (
            *(_normalise(email_response.workflow_id),),
            *(_normalise(candidate) for candidate in _collect_header_values("x-procwise-workflow-id")),
        )
        if value is not None
    }
    if dispatch_workflow and dispatch_workflow in workflow_candidates:
        score += WORKFLOW_WEIGHT
        matched_on.append("workflow_id")

    dispatch_unique = _normalise(dispatch.unique_id)
    unique_candidates = {
        value
        for value in (
            *(_normalise(email_response.unique_id),),
            *(
                _normalise(candidate)
                for candidate in _collect_header_values(
                    "x-procwise-unique-id",
                    "x-procwise-uid",
                )
            ),
        )
        if value is not None
    }
    if dispatch_unique and dispatch_unique in unique_candidates:
        score += UNIQUE_WEIGHT
        matched_on.append("unique_id")

    supplier_candidates = {
        value
        for value in (
            _normalise(email_response.supplier_id),
            *(_normalise(candidate) for candidate in _collect_header_values("x-procwise-supplier-id")),
        )
        if value is not None
    }
    dispatch_supplier = _normalise(dispatch.supplier_id)
    if dispatch_supplier and dispatch_supplier in supplier_candidates:
        score += SUPPLIER_WEIGHT
        matched_on.append("supplier_id")

    dispatch_round = _coerce_round_number(dispatch.round_number)
    response_round = _coerce_round_number(email_response.round_number)
    if response_round is None:
        for candidate in _collect_header_values("x-procwise-round"):
            candidate_round = _coerce_round_number(candidate)
            if candidate_round is not None:
                response_round = candidate_round
                break
    if (
        dispatch_round is not None
        and response_round is not None
        and dispatch_round == response_round
    ):
        score += ROUND_WEIGHT
        matched_on.append("round")

    subject_similarity = _subject_similarity(dispatch.subject, email_response.subject)
    if subject_similarity >= SUBJECT_SIMILARITY_THRESHOLD:
        score += SUBJECT_WEIGHT * subject_similarity
        matched_on.append("subject_similarity")

    thread_ids: Set[str] = set()
    for candidate in (
        *(dispatch.thread_headers.get("references", ())),
        *(dispatch.thread_headers.get("in_reply_to", ())),
    ):
        normalised = _normalise_message_token(candidate)
        if normalised:
            thread_ids.add(normalised)
    dispatch_message_token = _normalise_message_token(dispatch.message_id)
    if dispatch_message_token:
        thread_ids.add(dispatch_message_token)

    reply_headers: Set[str] = set()
    for candidate in (
        *(email_response.in_reply_to or ()),
        *(email_response.references or ()),
    ):
        normalised = _normalise_message_token(candidate)
        if normalised:
            reply_headers.add(normalised)
    for candidate in _collect_header_values("in-reply-to", "references"):
        normalised = _normalise_message_token(candidate)
        if normalised:
            reply_headers.add(normalised)
    if dispatch_message_token and dispatch_message_token in reply_headers:
        score = max(score, 1.0)
        if "in_reply_to" not in matched_on:
            matched_on.append("in_reply_to")
    elif thread_ids & reply_headers:
        score += THREAD_WEIGHT
        if "thread_reference" not in matched_on:
            matched_on.append("thread_reference")

    return min(score, 1.0), matched_on


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
        max_total_wait_seconds: Optional[int] = None,
        response_idle_timeout_seconds: int = 900,
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
        self.response_idle_timeout_seconds = max(1, int(response_idle_timeout_seconds))
        baseline_timeout = self.poll_interval_seconds * self.max_poll_attempts * 3
        derived_timeout = max(self.dispatch_wait_seconds, baseline_timeout)
        if max_total_wait_seconds is not None:
            try:
                derived_timeout = max(0, int(max_total_wait_seconds))
            except Exception:  # pragma: no cover - defensive conversion
                derived_timeout = max(self.dispatch_wait_seconds, baseline_timeout)
        self.max_total_wait_seconds = derived_timeout if derived_timeout > 0 else None
        self._trackers: Dict[str, WorkflowTracker] = {}
        self._imap_last_seen_uid: Optional[int] = None
        self._backtrack_window_seconds = 180
        self.agent_nick = "EmailWatcherV2"
        self._last_unmatched: List[Dict[str, Any]] = []

        tracking_repo.init_schema()
        supplier_response_repo.init_schema()
        try:
            workflow_lifecycle_repo.init_schema()
        except Exception:  # pragma: no cover - defensive initialisation
            logger.debug("Failed to initialise workflow lifecycle schema", exc_info=True)

    def _ensure_tracker(self, workflow_id: str) -> WorkflowTracker:
        tracker = self._trackers.get(workflow_id)
        if tracker is not None:
            return tracker
        tracker = WorkflowTracker(workflow_id=workflow_id)
        rows = tracking_repo.load_workflow_rows(workflow_id=workflow_id)
        if rows:
            dispatches = [
                EmailDispatchRecord(
                    workflow_id=row.workflow_id,
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

    def _state_round_value(self, tracker: WorkflowTracker) -> int:
        round_number = self._resolve_round(tracker)
        return round_number if isinstance(round_number, int) else 0

    def _archive_other_rounds(self, tracker: WorkflowTracker) -> None:
        try:
            email_watcher_state_repo.archive_other_rounds(
                tracker.workflow_id, self._state_round_value(tracker)
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to archive watcher state for workflow=%s", tracker.workflow_id
            )

    def _update_watcher_state(
        self,
        tracker: WorkflowTracker,
        status: str,
        **extra: Any,
    ) -> None:
        payload = {
            "expected_count": tracker.expected_responses,
            "responses_received": tracker.responded_count,
            "pending_suppliers": self._pending_suppliers(tracker),
            "pending_unique_ids": self._pending_unique_ids(tracker),
            "last_capture_ts": tracker.last_capture_at,
            "timeout_deadline": tracker.timeout_deadline,
        }
        payload.update(extra)
        try:
            email_watcher_state_repo.transition_state(
                tracker.workflow_id,
                self._state_round_value(tracker),
                status=status,
                **payload,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to persist watcher state for workflow=%s", tracker.workflow_id
            )

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
                workflow_id=workflow_id,
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
        unique_ids = sorted(
            set(tracker.email_records.keys()) | tracker.expected_unique_ids
        )
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
                round_number=self._resolve_round(tracker),
            )
        except Exception:
            logger.exception(
                "Failed to register expected responses with coordinator for workflow %s",
                tracker.workflow_id,
            )

    def _sync_expected_response_count(self, tracker: WorkflowTracker) -> bool:
        """Refresh the expected response count from the dispatch repository."""

        confirmed = False
        draft_unique_ids: Set[str] = set()
        supplier_index: Dict[str, Optional[str]] = {}
        try:
            draft_unique_ids, supplier_index, _ = (
                draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
                    workflow_id=tracker.workflow_id
                )
            )
        except Exception:
            logger.exception(
                "Failed to load drafted email expectations for workflow=%s",
                tracker.workflow_id,
            )
        if draft_unique_ids:
            tracker.register_expected_unique_ids(draft_unique_ids, supplier_index)

        expected_from_drafts = len(tracker.expected_unique_ids)

        try:
            dispatch_total = email_dispatch_repo.count_completed_supplier_dispatches(
                tracker.workflow_id
            )
        except Exception:
            logger.exception(
                "Failed to load completed dispatch count for workflow=%s",
                tracker.workflow_id,
            )
            dispatch_total = None

        fallback = max(len(tracker.expected_supplier_ids), tracker.dispatched_count)

        expected_total: Optional[int] = None
        confirm_total = False

        if dispatch_total is not None:
            if expected_from_drafts and dispatch_total < expected_from_drafts:
                expected_total = expected_from_drafts
            else:
                expected_total = max(dispatch_total, expected_from_drafts)
                confirm_total = expected_total > 0
        elif expected_from_drafts:
            expected_total = expected_from_drafts
        else:
            expected_total = None

        if expected_total is None:
            if fallback:
                tracker.set_expected_responses(fallback)
                if not expected_from_drafts:
                    confirm_total = True
        elif expected_total > 0:
            tracker.set_expected_responses(expected_total)
            if tracker.dispatched_count >= expected_total:
                confirm_total = True
        else:
            # ``count_completed_supplier_dispatches`` filters out ``NULL`` supplier ids,
            # so workflows that dispatch emails without a supplier identifier (e.g. when
            # the supplier agent is inactive) report ``0`` even though dispatches exist.
            # Fall back to the tracker counts in that scenario so the watcher can move
            # past the dispatch confirmation gate.
            if fallback:
                tracker.set_expected_responses(fallback)
                confirm_total = True
        if confirm_total:
            confirmed = True
        return confirmed

    def _resolve_round(
        self, tracker: WorkflowTracker, *, unique_id: Optional[str] = None
    ) -> Optional[int]:
        if unique_id:
            return tracker.round_for_unique(unique_id)
        rounds = tracker.distinct_rounds()
        if len(rounds) == 1:
            return rounds[0]
        return None

    def _log_event(
        self,
        event: str,
        tracker: WorkflowTracker,
        *,
        round_override: Optional[int] = None,
        **fields: Any,
    ) -> None:
        payload: Dict[str, Any] = {"event": event, "workflow_id": tracker.workflow_id}
        round_value = round_override
        if round_value is None:
            round_value = self._resolve_round(tracker)
        if round_value is not None:
            payload["round"] = round_value

        def _serialise(value: Any) -> Any:
            if isinstance(value, datetime):
                moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
                return moment.astimezone(timezone.utc).isoformat()
            if isinstance(value, set):
                return sorted(value)
            return value

        for key, value in fields.items():
            if value is None:
                continue
            payload[key] = _serialise(value)

        try:
            logger.info(json.dumps(payload))
        except TypeError:
            safe_payload = {key: str(val) for key, val in payload.items()}
            logger.info(json.dumps(safe_payload))

    def _pending_unique_ids(self, tracker: WorkflowTracker) -> List[str]:
        known_unique_ids = set(tracker.email_records.keys()) | set(
            tracker.expected_unique_ids
        )
        return sorted(
            uid for uid in known_unique_ids if uid not in tracker.responded_unique_ids
        )

    def _pending_suppliers(
        self, tracker: WorkflowTracker, pending_unique: Optional[Iterable[str]] = None
    ) -> List[str]:
        unique_ids = list(pending_unique) if pending_unique is not None else self._pending_unique_ids(tracker)
        seen: Set[str] = set()
        pending_suppliers: List[str] = []
        for uid in unique_ids:
            dispatch = tracker.latest_dispatch(uid)
            supplier_id = dispatch.supplier_id if dispatch else None
            if not supplier_id:
                continue
            if supplier_id in seen:
                continue
            seen.add(supplier_id)
            pending_suppliers.append(supplier_id)
        remaining = max(0, tracker.expected_responses - tracker.responded_count)
        if remaining and len(pending_suppliers) < remaining:
            for supplier_id in sorted(tracker.expected_supplier_ids):
                if supplier_id in seen:
                    continue
                unique_set = tracker.supplier_unique_map.get(supplier_id) or set()
                if unique_set and all(
                    uid in tracker.responded_unique_ids for uid in unique_set
                ):
                    continue
                seen.add(supplier_id)
                pending_suppliers.append(supplier_id)
                if len(pending_suppliers) >= remaining:
                    break
        return pending_suppliers

    def _update_idle_deadline(self, tracker: WorkflowTracker, capture_time: datetime) -> None:
        moment = capture_time if capture_time.tzinfo else capture_time.replace(tzinfo=timezone.utc)
        tracker.last_capture_at = moment
        tracker.timeout_deadline = moment + timedelta(seconds=self.response_idle_timeout_seconds)

    def _emit_timeout_event(
        self,
        tracker: WorkflowTracker,
        *,
        round_number: Optional[int],
        pending_suppliers: Sequence[str],
    ) -> None:
        try:
            bus = get_event_bus()
        except Exception:
            bus = None
        payload = {
            "workflow_id": tracker.workflow_id,
            "round": round_number,
            "pending_suppliers": list(pending_suppliers),
            "expected_count": tracker.expected_responses,
            "responses_received": tracker.responded_count,
        }
        if bus is not None:
            try:
                bus.publish("responses_timeout", payload)
                scoped = f"responses_timeout:{tracker.workflow_id}:{round_number}" if round_number is not None else None
                if scoped:
                    bus.publish(scoped, payload)
            except Exception:
                logger.exception(
                    "Failed to publish responses_timeout event workflow=%s", tracker.workflow_id
                )
        else:
            logger.debug(
                "Event bus unavailable when emitting responses_timeout workflow=%s",
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

    def _fetch_emails(
        self,
        since: datetime,
        *,
        workflow_ids: Optional[Sequence[str]] = None,
        mailbox_filter: Optional[str] = None,
    ) -> List[EmailResponse]:
        payload: Optional[object]

        workflow_filters: List[str] = []
        if workflow_ids:
            for value in workflow_ids:
                if value in (None, ""):
                    continue
                text = str(value).strip()
                if text:
                    workflow_filters.append(text)
        mailbox_filter_value = str(mailbox_filter).strip() if mailbox_filter else ""

        if self._fetcher:
            fetch_kwargs = {
                "since": since,
                "last_seen_uid": self._imap_last_seen_uid,
                "backtrack_seconds": self._backtrack_window_seconds,
            }
            if workflow_filters:
                fetch_kwargs["workflow_ids"] = list(dict.fromkeys(workflow_filters))
            if mailbox_filter_value:
                fetch_kwargs["mailbox_filter"] = mailbox_filter_value
            try:
                payload = self._fetcher(**fetch_kwargs)
            except TypeError:
                fetch_kwargs.pop("last_seen_uid", None)
                fetch_kwargs.pop("backtrack_seconds", None)
                try:
                    payload = self._fetcher(**fetch_kwargs)
                except TypeError:
                    fallback_kwargs = {"since": since}
                    if workflow_filters:
                        fallback_kwargs["workflow_ids"] = list(
                            dict.fromkeys(workflow_filters)
                        )
                    if mailbox_filter_value:
                        fallback_kwargs["mailbox_filter"] = mailbox_filter_value
                    payload = self._fetcher(**fallback_kwargs)
        else:
            if not all([self._imap_host, self._imap_username, self._imap_password]):
                raise RuntimeError("IMAP credentials must be supplied when no custom fetcher is provided")

            payload = _default_fetcher(
                host=self._imap_host,
                username=self._imap_username,
                password=self._imap_password,
                mailbox=self._mailbox,
                since=since,
                port=self._imap_port or 993,
                use_ssl=True if self._imap_use_ssl is None else self._imap_use_ssl,
                login=self._imap_login,
                last_seen_uid=self._imap_last_seen_uid,
                backtrack_seconds=self._backtrack_window_seconds,
                workflow_ids=list(dict.fromkeys(workflow_filters)) or None,
                mailbox_filter=mailbox_filter_value or None,
            )

        responses, last_uid = _normalise_fetch_result(payload)
        if last_uid is not None:
            if self._imap_last_seen_uid is None or last_uid > self._imap_last_seen_uid:
                self._imap_last_seen_uid = last_uid
        return responses

    def _match_responses(
        self, tracker: WorkflowTracker, responses: Iterable[EmailResponse]
    ) -> List[SupplierResponseRow]:
        matched_rows: List[SupplierResponseRow] = []
        threshold = max(self.match_threshold, 0.75)
        for email in responses:
            message_id_raw = email.message_id or ""
            message_id = message_id_raw.strip()
            fingerprint = _response_fingerprint(email)
            if message_id and message_id in tracker.seen_message_ids:
                logger.debug(
                    "EmailWatcher deduped message_id workflow=%s message_id=%s",
                    tracker.workflow_id,
                    message_id,
                )
                continue
            if (
                fingerprint
                and not message_id
                and fingerprint in tracker.seen_fingerprints
            ):
                logger.debug(
                    "EmailWatcher deduped fingerprint workflow=%s fingerprint=%s",
                    tracker.workflow_id,
                    fingerprint,
                )
                continue
            header_presence_map = {key.lower(): value for key, value in email.headers.items()}

            def _has_header_value(entries: Optional[object]) -> bool:
                if entries in (None, ""):
                    return False
                if isinstance(entries, (list, tuple, set)):
                    return any(str(item).strip() for item in entries if item is not None)
                return bool(str(entries).strip())

            workflow_header_present = bool(email.workflow_id) or _has_header_value(
                header_presence_map.get("x-procwise-workflow-id")
            )
            unique_header_present = bool(email.unique_id) or _has_header_value(
                header_presence_map.get("x-procwise-unique-id")
            )
            supplier_header_present = bool(email.supplier_id) or _has_header_value(
                header_presence_map.get("x-procwise-supplier-id")
            )
            round_header_present = (
                email.round_number is not None
                or _has_header_value(header_presence_map.get("x-procwise-round"))
            )
            def _merge_reasons(existing: List[str], *reasons: str) -> List[str]:
                ordered = list(existing)
                for reason in reasons:
                    if not reason:
                        continue
                    if reason not in ordered:
                        ordered.append(reason)
                return ordered
            matched_id: Optional[str] = None
            best_score = 0.0
            best_dispatch: Optional[EmailDispatchRecord] = None
            best_reasons: List[str] = []
            match_details: Dict[str, Any] = {}
            for unique_id, dispatch_list in tracker.email_records.items():
                dispatch = dispatch_list[-1]
                if unique_id in tracker.matched_responses:
                    continue
                score, reasons = _calculate_match_score(dispatch, email)
                if score > best_score:
                    matched_id = unique_id
                    best_score = score
                    best_dispatch = dispatch
                    best_reasons = _merge_reasons([], *reasons)
                    match_details = {"matcher": "header_score"}
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
                    best_reasons = _merge_reasons(best_reasons, "supplier_id")
                    match_details.setdefault("supplier_match_candidates", len(supplier_matches))
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
                        best_reasons = _merge_reasons(best_reasons, "rfq")
                        match_details.setdefault("rfq_match_candidates", len(rfq_candidates))

            if (
                (not matched_id or best_score < self.match_threshold)
                and email.received_at
            ):
                candidates: List[Tuple[str, EmailDispatchRecord]] = []
                for uid, dispatch_list in tracker.email_records.items():
                    if uid in tracker.matched_responses:
                        continue
                    if not dispatch_list:
                        continue
                    dispatch_candidate = dispatch_list[-1]
                    dispatched_at = dispatch_candidate.dispatched_at
                    if not dispatched_at:
                        continue
                    candidates.append((uid, dispatch_candidate))
                if candidates:
                    def _time_distance(item: Tuple[str, EmailDispatchRecord]) -> float:
                        dispatch_time = item[1].dispatched_at
                        received_at = email.received_at
                        if dispatch_time is None or received_at is None:
                            return float("inf")
                        if dispatch_time.tzinfo is None:
                            dispatch_dt = dispatch_time.replace(tzinfo=timezone.utc)
                        else:
                            dispatch_dt = dispatch_time
                        if received_at.tzinfo is None:
                            received_dt = received_at.replace(tzinfo=timezone.utc)
                        else:
                            received_dt = received_at
                        return abs((received_dt - dispatch_dt).total_seconds())

                    candidates.sort(key=_time_distance)
                    candidate_uid, candidate_dispatch = candidates[0]
                    time_delta_seconds = _time_distance((candidate_uid, candidate_dispatch))
                    subject_similarity = _subject_similarity(
                        candidate_dispatch.subject,
                        email.subject,
                    )
                    match_details.setdefault("temporal_candidate_count", len(candidates))
                    match_details.update(
                        {
                            "temporal_delta_seconds": time_delta_seconds,
                            "subject_similarity": round(subject_similarity, 3),
                        }
                    )
                    if subject_similarity >= 0.7:
                        matched_id = candidate_uid
                        best_dispatch = candidate_dispatch
                        best_score = max(best_score, threshold)
                        best_reasons = _merge_reasons(
                            best_reasons,
                            "temporal_proximity",
                            "subject_similarity",
                        )
                        match_details["matcher"] = "temporal_subject"

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
                best_score = max(best_score, threshold)
                best_dispatch = tracker.latest_dispatch(matched_id)

            if matched_id and best_score >= threshold:
                criteria_str = ", ".join(best_reasons) if best_reasons else "n/a"
                try:
                    detail_payload = json.dumps(match_details, sort_keys=True)
                except TypeError:
                    detail_payload = str(match_details)
                logger.info(
                    "MATCH SUCCESS: workflow=%s unique_id=%s score=%.2f criteria=[%s] details=%s",
                    tracker.workflow_id,
                    matched_id,
                    best_score,
                    criteria_str,
                    detail_payload,
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
                outbound_data = supplier_interaction_repo.lookup_outbound(matched_id)
                if outbound_data is None and email.unique_id and email.unique_id != matched_id:
                    outbound_data = supplier_interaction_repo.lookup_outbound(email.unique_id)
                outbound: Optional[SupplierInteractionRow] = None
                if outbound_data is not None:
                    outbound = SupplierInteractionRow(
                        workflow_id=outbound_data.get("workflow_id"),
                        unique_id=outbound_data.get("unique_id"),
                        supplier_id=outbound_data.get("supplier_id"),
                        supplier_email=outbound_data.get("supplier_email"),
                        round_number=int(outbound_data.get("round_number") or 1),
                        direction=str(outbound_data.get("direction") or "outbound"),
                        interaction_type=str(
                            outbound_data.get("interaction_type") or "initial"
                        ),
                        status=str(outbound_data.get("status") or "pending"),
                        subject=outbound_data.get("subject"),
                        body=outbound_data.get("body"),
                        message_id=outbound_data.get("message_id"),
                        in_reply_to=outbound_data.get("in_reply_to"),
                        references=outbound_data.get("references"),
                        rfq_id=outbound_data.get("rfq_id"),
                        received_at=outbound_data.get("received_at"),
                        processed_at=outbound_data.get("processed_at"),
                        metadata=outbound_data.get("metadata"),
                    )
                if outbound is None:
                    logger.warning(
                        "services.email_watcher_v2 missing outbound row agent=%s workflow=%s unique_id=%s",
                        self.agent_nick,
                        tracker.workflow_id,
                        matched_id,
                    )
                tracker.record_response(matched_id, email)
                elapsed = tracker.elapsed_seconds(email.received_at)
                tracker.seen_message_ids.add(message_id)
                if not message_id and fingerprint:
                    tracker.seen_fingerprints.add(fingerprint)
                capture_time = self._now()
                self._update_idle_deadline(tracker, capture_time)
                pending_suppliers_after = self._pending_suppliers(tracker)
                logger.info(
                    "EmailWatcher captured supplier response workflow=%s%s responded=%d/%d elapsed=%.1fs matched_on=%s best_score=%.2f",
                    tracker.workflow_id,
                    self._round_fragment(tracker, matched_id),
                    tracker.responded_count,
                    tracker.expected_responses,
                    elapsed,
                    best_reasons,
                    best_score,
                )
                supplier_id_value = email.supplier_id or best_dispatch.supplier_id
                self._log_event(
                    "response_captured",
                    tracker,
                    round_override=tracker.round_for_unique(matched_id),
                    supplier_id=supplier_id_value,
                    dispatch_id=matched_id,
                    message_id=email.message_id,
                    score=round(best_score, 4),
                    matched_on=list(best_reasons),
                    responded=tracker.responded_count,
                    expected=tracker.expected_responses,
                    pending_suppliers=pending_suppliers_after,
                    status=tracker.status,
                    elapsed_s=round(tracker.elapsed_seconds(self._now()), 2),
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
                    or (outbound.supplier_email if outbound else None)
                    or best_dispatch.supplier_email
                )
                supplier_id = supplier_id_value or (outbound.supplier_id if outbound else None)
                metadata_payload: Dict[str, Any] = {
                    "agent": self.agent_nick,
                    "matched_on": best_reasons,
                }
                if email.received_at:
                    received_ts = (
                        email.received_at
                        if email.received_at.tzinfo
                        else email.received_at.replace(tzinfo=timezone.utc)
                    )
                    metadata_payload["received_at"] = received_ts.isoformat()
                if best_dispatch.dispatched_at:
                    dispatch_ts = (
                        best_dispatch.dispatched_at
                        if best_dispatch.dispatched_at.tzinfo
                        else best_dispatch.dispatched_at.replace(tzinfo=timezone.utc)
                    )
                    metadata_payload.setdefault("dispatched_at", dispatch_ts.isoformat())
                if outbound is not None:
                    supplier_interaction_repo.record_inbound_response(
                        outbound=outbound,
                        message_id=email.message_id or uuid.uuid4().hex,
                        subject=email.subject,
                        body=email.body,
                        from_address=email.from_address,
                        received_at=email.received_at,
                        in_reply_to=email.in_reply_to,
                        references=email.references,
                        rfq_id=email.rfq_id,
                        metadata=metadata_payload,
                    )
                original_message_id_value = _format_message_identifier(
                    best_dispatch.message_id
                )
                if not original_message_id_value:
                    thread_candidates = list(email.in_reply_to) + list(email.references)
                    for thread_id in thread_candidates:
                        formatted = _format_message_identifier(thread_id)
                        if formatted:
                            original_message_id_value = formatted
                            break
                response_time: Optional[Decimal] = None
                if best_dispatch.dispatched_at and email.received_at:
                    try:
                        delta_seconds = (email.received_at - best_dispatch.dispatched_at).total_seconds()
                        if delta_seconds >= 0:
                            response_time = Decimal(str(delta_seconds))
                    except Exception:  # pragma: no cover - defensive conversion
                        response_time = None
                structured = extract_structured_fields(email)
                matched_decimal = _score_to_confidence(best_score)
                plain_text = email.body_text or email.body
                response_row = SupplierResponseRow(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    supplier_id=supplier_id,
                    supplier_email=supplier_email,
                    rfq_id=email.rfq_id or best_dispatch.rfq_id,
                    response_text=plain_text or "",
                    response_body=plain_text or "",
                    received_time=email.received_at,
                    response_time=response_time,
                    response_message_id=email.message_id,
                    response_subject=email.subject,
                    response_from=email.from_address,
                    round_number=tracker.round_for_unique(matched_id),
                    original_message_id=original_message_id_value,
                    original_subject=best_dispatch.subject,
                    match_confidence=matched_decimal,
                    match_score=matched_decimal,
                    match_evidence=best_reasons,
                    matched_on=best_reasons,
                    raw_headers=email.headers,
                    body_html=email.body_html,
                    in_reply_to=list(email.in_reply_to),
                    references=list(email.references),
                    price=structured.get("price"),
                    currency=structured.get("currency"),
                    lead_time=structured.get("lead_time_days"),
                    payment_terms=structured.get("payment_terms"),
                    warranty=structured.get("warranty"),
                    validity=structured.get("validity"),
                    exceptions=structured.get("exceptions"),
                    attachments=structured.get("attachments"),
                    tables=structured.get("tables"),
                    processed=False,
                )
                supplier_response_repo.insert_response(response_row)
                matched_rows.append(response_row)
            else:
                reasons = best_reasons or ["insufficient-evidence"]
                if message_id:
                    tracker.seen_message_ids.add(message_id)
                if not message_id and fingerprint:
                    tracker.seen_fingerprints.add(fingerprint)
                header_summary = {
                    "workflow": workflow_header_present,
                    "unique": unique_header_present,
                    "supplier": supplier_header_present,
                    "round": round_header_present,
                }
                attempted_unique = email.unique_id or "n/a"
                attempted_supplier = email.supplier_id or "n/a"
                attempted_threads = len(email.references) + len(email.in_reply_to)
                logger.warning(
                    "MATCH FAILED: workflow=%s email_from=%s best_score=%.2f threshold=%.2f attempted=[unique_id=%s, supplier=%s, thread_ids=%d] matched_on=%s headers=%s",
                    tracker.workflow_id,
                    email.from_address or email.supplier_email or "unknown",
                    best_score,
                    self.match_threshold,
                    attempted_unique,
                    attempted_supplier,
                    attempted_threads,
                    reasons,
                    header_summary,
                )
        return matched_rows

    def wait_and_collect_responses(self, workflow_id: str) -> Dict[str, object]:
        tracker = self._ensure_tracker(workflow_id)
        self._archive_other_rounds(tracker)
        initial_confirmed = self._sync_expected_response_count(tracker)
        if tracker.dispatched_count == 0 and not tracker.expected_unique_ids:
            self._update_watcher_state(
                tracker,
                "completed",
                reason="no_dispatches",
                responses_received=0,
                expected_count=0,
            )
            return {
                "workflow_id": workflow_id,
                "complete": True,
                "dispatched_count": 0,
                "responded_count": 0,
                "matched_responses": {},
            }

        lifecycle = workflow_lifecycle_repo.get_lifecycle(workflow_id) or {}
        supplier_status = (lifecycle.get("supplier_agent_status") or "").strip().lower()
        allowed_statuses = {"awaiting_responses", "running", "started", "active"}
        if supplier_status and supplier_status not in allowed_statuses:
            logger.info(
                "EmailWatcher deferring start until SupplierInteractionAgent is active workflow=%s status=%s",
                workflow_id,
                supplier_status or "unknown",
            )
            self._update_watcher_state(
                tracker,
                "none",
                reason="supplier_agent_inactive",
            )
            return {
                "workflow_id": workflow_id,
                "complete": False,
                "dispatched_count": tracker.dispatched_count,
                "responded_count": tracker.responded_count,
                "matched_responses": {},
                "expected_responses": tracker.expected_responses,
                "workflow_status": "awaiting_supplier_activation",
                "timeout_reached": False,
                "reason": "supplier_agent_inactive",
            }

        negotiation_status = (lifecycle.get("negotiation_status") or "").strip().lower()
        if negotiation_status in {"completed", "finalized"}:
            pending_unique_ids = self._pending_unique_ids(tracker)
            pending_suppliers = self._pending_suppliers(tracker, pending_unique_ids)
            outstanding_responses = bool(pending_unique_ids) or (
                tracker.expected_responses > 0
                and tracker.responded_count < tracker.expected_responses
            )
            if outstanding_responses:
                logger.warning(
                    "EmailWatcher negotiation already completed with outstanding responses workflow=%s status=%s expected=%d responded=%d pending_suppliers=%s",
                    workflow_id,
                    negotiation_status,
                    tracker.expected_responses,
                    tracker.responded_count,
                    ",".join(pending_suppliers) or "-",
                )
                self._update_watcher_state(
                    tracker,
                    "awaiting_responses",
                    reason="negotiation_completed_pending_responses",
                    pending_unique_ids=pending_unique_ids,
                    pending_suppliers=pending_suppliers,
                )
                return {
                    "workflow_id": workflow_id,
                    "complete": False,
                    "dispatched_count": tracker.dispatched_count,
                    "responded_count": tracker.responded_count,
                    "matched_responses": tracker.matched_responses,
                    "response_history": tracker.response_history,
                    "expected_responses": tracker.expected_responses,
                    "workflow_status": "negotiation_completed_pending_responses",
                    "timeout_reached": False,
                    "pending_unique_ids": pending_unique_ids,
                    "pending_suppliers": pending_suppliers,
                    "reason": "negotiation_completed_pending_responses",
                }

            logger.info(
                "EmailWatcher skipping run because negotiation already completed workflow=%s status=%s",
                workflow_id,
                negotiation_status,
            )
            self._update_watcher_state(
                tracker,
                "completed",
                reason="negotiation_completed",
            )
            return {
                "workflow_id": workflow_id,
                "complete": True,
                "dispatched_count": tracker.dispatched_count,
                "responded_count": tracker.responded_count,
                "matched_responses": tracker.matched_responses,
                "expected_responses": tracker.expected_responses,
                "response_history": tracker.response_history,
                "workflow_status": "negotiation_completed",
                "timeout_reached": False,
                "reason": "negotiation_completed",
            }

        watcher_started = False
        watcher_start_at: Optional[datetime] = None
        timeout_reached = False
        negotiation_stop = False
        stop_reason: Optional[str] = None
        last_pending_unique: List[str] = []
        last_pending_suppliers: List[str] = []
        final_status: Optional[str] = None

        try:
            watcher_start_at = self._now()
            self._update_idle_deadline(tracker, watcher_start_at)
            expected_confirmed = initial_confirmed
            if not expected_confirmed:
                expected_confirmed = self._sync_expected_response_count(tracker)
            pending_unique_ids = self._pending_unique_ids(tracker)
            pending_suppliers = self._pending_suppliers(tracker, pending_unique_ids)
            tracker.status = "initializing"
            self._update_watcher_state(
                tracker,
                "initializing",
                start_ts=watcher_start_at,
                last_capture_ts=watcher_start_at,
                timeout_deadline=tracker.timeout_deadline,
                uid_cursor=self._imap_last_seen_uid,
                pending_unique_ids=pending_unique_ids,
                pending_suppliers=pending_suppliers,
            )
            workflow_lifecycle_repo.record_watcher_event(
                workflow_id,
                "watcher_started",
                expected_responses=tracker.expected_responses,
                received_responses=tracker.responded_count,
                metadata={"supplier_status": supplier_status} if supplier_status else None,
            )
            watcher_started = True

            self._log_event(
                "watcher_started",
                tracker,
                start_ts=watcher_start_at,
                expected=tracker.expected_responses,
                responded=tracker.responded_count,
                pending_suppliers=pending_suppliers,
                status=tracker.status,
                elapsed_s=0.0,
                uid_cursor=self._imap_last_seen_uid,
            )

            if not expected_confirmed:
                stop_reason = "awaiting_dispatch_confirmation"
                final_status = "initializing"
                last_pending_unique = list(pending_unique_ids)
                last_pending_suppliers = list(pending_suppliers)
                return {
                    "workflow_id": workflow_id,
                    "complete": False,
                    "dispatched_count": tracker.dispatched_count,
                    "responded_count": tracker.responded_count,
                    "matched_responses": tracker.matched_responses,
                    "expected_responses": tracker.expected_responses,
                    "workflow_status": "awaiting_dispatch_confirmation",
                    "timeout_reached": False,
                    "pending_unique_ids": list(pending_unique_ids),
                    "pending_suppliers": list(pending_suppliers),
                }

            tracker.status = "active"
            last_pending_unique = list(pending_unique_ids)
            last_pending_suppliers = list(pending_suppliers)
            self._update_watcher_state(
                tracker,
                "active",
                pending_unique_ids=last_pending_unique,
                pending_suppliers=last_pending_suppliers,
                heartbeat_ts=watcher_start_at,
            )
            workflow_lifecycle_repo.record_watcher_event(
                workflow_id,
                "watcher_active",
                expected_responses=tracker.expected_responses,
                received_responses=tracker.responded_count,
            )

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

            base_sleep = float(self.poll_interval_seconds)
            adaptive_sleep = base_sleep
            idle_attempts = 0
            error_backoff = base_sleep
            error_backoff_cap = max(base_sleep, 300.0)
            poll_started_at = self._now()
            baseline_since = tracker.last_dispatched_at or (poll_started_at - timedelta(hours=4))
            since_cursor = tracker.last_response_at or baseline_since
            if since_cursor.tzinfo is None:
                since_cursor = since_cursor.replace(tzinfo=timezone.utc)
            if baseline_since.tzinfo is None:
                baseline_since = baseline_since.replace(tzinfo=timezone.utc)
            if since_cursor < baseline_since:
                since_cursor = baseline_since

            while not tracker.all_responded:
                now = self._now()
                elapsed = tracker.elapsed_seconds(now)
                runtime_elapsed = max(0.0, (now - poll_started_at).total_seconds())
                try:
                    lifecycle = workflow_lifecycle_repo.get_lifecycle(workflow_id) or {}
                    negotiation_status = (
                        lifecycle.get("negotiation_status") or ""
                    ).strip().lower()
                except Exception:  # pragma: no cover - defensive lifecycle lookup
                    negotiation_status = ""
                if negotiation_status in {"completed", "finalized"}:
                    logger.info(
                        "EmailWatcher detected negotiation completion; stopping workflow=%s status=%s",
                        workflow_id,
                        negotiation_status,
                    )
                    negotiation_stop = True
                    stop_reason = "negotiation_completed"
                    break
                if (
                    self.max_total_wait_seconds is not None
                    and runtime_elapsed >= self.max_total_wait_seconds
                ):
                    timeout_reached = True
                    stop_reason = "timeout"
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

                try:
                    responses = self._fetch_emails(
                        since_cursor,
                        workflow_ids=[workflow_id],
                    )
                except (imaplib.IMAP4.abort, imaplib.IMAP4.error) as exc:
                    logger.warning(
                        "EmailWatcher IMAP abort encountered workflow=%s error=%s",
                        workflow_id,
                        exc,
                    )
                    self._log_event(
                        "watcher_imap_abort",
                        tracker,
                        status=tracker.status,
                        error=str(exc),
                    )
                    self._sleep(base_sleep)
                    continue
                matched_rows = self._match_responses(tracker, responses)
                now_after_poll = self._now()
                reference_capture = tracker.last_capture_at or watcher_start_at
                if reference_capture is None:
                    reference_capture = now_after_poll
                if reference_capture.tzinfo is None:
                    reference_capture = reference_capture.replace(tzinfo=timezone.utc)
                since_last_capture = max(
                    0.0, (now_after_poll - reference_capture).total_seconds()
                )
                pending_unique_ids = self._pending_unique_ids(tracker)
                pending_suppliers = self._pending_suppliers(tracker, pending_unique_ids)
                last_pending_unique = pending_unique_ids
                last_pending_suppliers = pending_suppliers
                timeout_deadline = tracker.timeout_deadline
                self._log_event(
                    "watcher_heartbeat",
                    tracker,
                    responded=tracker.responded_count,
                    expected=tracker.expected_responses,
                    pending_suppliers=pending_suppliers,
                    since_last_capture_s=round(since_last_capture, 2),
                    timeout_deadline=timeout_deadline,
                    status=tracker.status,
                    elapsed_s=round(tracker.elapsed_seconds(self._now()), 2),
                )
                self._update_watcher_state(
                    tracker,
                    "active",
                    pending_unique_ids=pending_unique_ids,
                    pending_suppliers=pending_suppliers,
                    heartbeat_ts=now_after_poll,
                )
                if (
                    tracker.expected_responses > tracker.responded_count
                    and since_last_capture >= self.response_idle_timeout_seconds
                ):
                    timeout_reached = True
                    stop_reason = "partial_timeout"
                    tracker.status = "partial_timeout"
                    self._emit_timeout_event(
                        tracker,
                        round_number=self._resolve_round(tracker),
                        pending_suppliers=pending_suppliers,
                    )
                    break
                if matched_rows:
                    workflow_lifecycle_repo.record_watcher_event(
                        workflow_id,
                        "watcher_active",
                        expected_responses=tracker.expected_responses,
                        received_responses=tracker.responded_count,
                    )
                    self._process_agents(tracker)
                cursor_candidate = tracker.last_response_at
                if cursor_candidate is None and matched_rows:
                    cursor_candidate = max(
                        (
                            row.received_time
                            for row in matched_rows
                            if row.received_time is not None
                        ),
                        default=None,
                    )
                if cursor_candidate is not None and cursor_candidate.tzinfo is None:
                    cursor_candidate = cursor_candidate.replace(tzinfo=timezone.utc)
                if cursor_candidate is not None and cursor_candidate > since_cursor:
                    since_cursor = cursor_candidate
                if tracker.all_responded:
                    break

                outstanding = max(0, tracker.expected_responses - tracker.responded_count)
                if responses or matched_rows:
                    idle_attempts = 0
                    adaptive_sleep = base_sleep
                    idle_snapshot = 0
                else:
                    idle_attempts += 1
                    idle_snapshot = idle_attempts
                    if idle_attempts >= self.max_poll_attempts:
                        adaptive_sleep = min(adaptive_sleep * 2, max(base_sleep, 300.0))
                        logger.info(
                            "EmailWatcher continuing to monitor workflow=%s%s outstanding=%d runtime=%.1fs next_poll=%.1fs",
                            workflow_id,
                            self._round_fragment(tracker),
                            outstanding,
                            runtime_elapsed,
                            adaptive_sleep,
                        )
                        idle_attempts = 0
                        idle_snapshot = 0

                logger.info(
                    "EmailWatcher poll summary workflow=%s%s expected=%d responded=%d elapsed=%.1fs runtime=%.1fs idle_attempts=%d next_sleep=%.1fs outstanding=%d",
                    workflow_id,
                    self._round_fragment(tracker),
                    tracker.expected_responses,
                    tracker.responded_count,
                    elapsed,
                    runtime_elapsed,
                    idle_snapshot,
                    adaptive_sleep,
                    outstanding,
                )

                self._sleep(adaptive_sleep)

            if stop_reason is None and tracker.all_responded:
                stop_reason = "responses_complete"

            negotiation_completed = negotiation_stop
            outstanding_responses = bool(last_pending_unique) or (
                tracker.expected_responses > 0
                and tracker.responded_count < tracker.expected_responses
            )
            if negotiation_completed and outstanding_responses:
                logger.warning(
                    "EmailWatcher negotiation completed before responses were collected workflow=%s%s expected=%d responded=%d pending_suppliers=%s",
                    workflow_id,
                    self._round_fragment(tracker),
                    tracker.expected_responses,
                    tracker.responded_count,
                    ",".join(last_pending_suppliers) or "-",
                )

            complete = tracker.all_responded or (
                negotiation_completed and not outstanding_responses
            )
            if stop_reason in {"partial_timeout", "timeout"}:
                workflow_state = "partial_timeout"
                final_status = "partial_timeout"
            elif negotiation_completed and not outstanding_responses:
                workflow_state = "negotiation_completed"
                final_status = "completed"
            elif negotiation_completed and outstanding_responses:
                workflow_state = "negotiation_completed_pending_responses"
                final_status = "incomplete"
            elif tracker.all_responded:
                workflow_state = "responses_complete"
                final_status = "completed"
            else:
                workflow_state = "awaiting_responses"
                final_status = final_status or "active"

            if negotiation_completed and outstanding_responses:
                tracker.status = "incomplete"
            elif negotiation_completed or tracker.all_responded:
                tracker.status = "completed"
            elif stop_reason in {"partial_timeout", "timeout"}:
                tracker.status = "partial_timeout"
            else:
                tracker.status = "active"
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
                "workflow_status": workflow_state,
                "pending_unique_ids": list(last_pending_unique),
                "pending_suppliers": list(last_pending_suppliers),
            }
            if tracker.last_capture_at is not None:
                capture_ts = (
                    tracker.last_capture_at
                    if tracker.last_capture_at.tzinfo
                    else tracker.last_capture_at.replace(tzinfo=timezone.utc)
                )
                result["last_capture_ts"] = capture_ts.isoformat()
            if tracker.timeout_deadline is not None:
                deadline_ts = (
                    tracker.timeout_deadline
                    if tracker.timeout_deadline.tzinfo
                    else tracker.timeout_deadline.replace(tzinfo=timezone.utc)
                )
                result["timeout_deadline"] = deadline_ts.isoformat()

            if not negotiation_completed:
                self._process_agents(tracker)

            self._update_watcher_state(
                tracker,
                tracker.status
                or final_status
                or ("partial_timeout" if timeout_reached else "completed"),
                reason=stop_reason,
                pending_unique_ids=last_pending_unique,
                pending_suppliers=last_pending_suppliers,
            )

            return result
        except Exception as exc:
            final_status = "failed"
            self._update_watcher_state(
                tracker,
                "failed",
                last_error=str(exc),
                pending_unique_ids=last_pending_unique,
                pending_suppliers=last_pending_suppliers,
            )
            raise
        finally:
            if watcher_started:
                runtime = 0.0
                if watcher_start_at is not None:
                    try:
                        runtime = max(
                            0.0, (self._now() - watcher_start_at).total_seconds()
                        )
                    except Exception:  # pragma: no cover - defensive
                        runtime = 0.0
                stop_metadata: Dict[str, Any] = {}
                if stop_reason:
                    stop_metadata["stop_reason"] = stop_reason
                if timeout_reached:
                    stop_metadata["timeout_reached"] = True
                if negotiation_stop:
                    stop_metadata["negotiation_completed"] = True
                if last_pending_suppliers:
                    stop_metadata["pending_suppliers"] = list(last_pending_suppliers)
                if final_status:
                    stop_metadata["status"] = final_status
                resolved_status = (
                    tracker.status
                    or final_status
                    or (
                        "partial_timeout"
                        if stop_reason in {"partial_timeout", "timeout"}
                        else "complete"
                    )
                )
                self._log_event(
                    "watcher_stopped",
                    tracker,
                    status=resolved_status,
                    responded=tracker.responded_count,
                    expected=tracker.expected_responses,
                    pending_suppliers=last_pending_suppliers,
                    elapsed_s=runtime,
                )
                workflow_lifecycle_repo.record_watcher_event(
                    workflow_id,
                    "watcher_stopped",
                    expected_responses=tracker.expected_responses,
                    received_responses=tracker.responded_count,
                    runtime_seconds=runtime,
                    metadata=stop_metadata or None,
                )
            self._trackers.pop(workflow_id, None)

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
        ids_from_agent: List[str] = []
        negotiation_payload: Dict[str, Any] = {}
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
                workflow_lifecycle_repo.record_negotiation_status(
                    tracker.workflow_id, "started"
                )
                neg_context = AgentContext(
                    workflow_id=tracker.workflow_id,
                    agent_id="NegotiationAgent",
                    user_id="system",
                    input_data=negotiation_payload,
                )
                result = self.negotiation_agent.execute(neg_context)
                status = getattr(result, "status", None)
                if status == AgentStatus.SUCCESS or status is None:
                    workflow_lifecycle_repo.record_negotiation_status(
                        tracker.workflow_id, "completed"
                    )
                else:
                    workflow_lifecycle_repo.record_negotiation_status(
                        tracker.workflow_id, "failed"
                    )
            except Exception:
                logger.exception("NegotiationAgent failed for workflow %s", tracker.workflow_id)
                workflow_lifecycle_repo.record_negotiation_status(
                    tracker.workflow_id, "failed"
                )

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
    dispatched_at: Optional[datetime] = None,
) -> None:
    """Persist a dispatched email so inbound replies can be reconciled."""

    supplier_interaction_repo.init_schema()

    headers = thread_headers or {}
    if isinstance(dispatched_at, datetime):
        dispatched_dt = dispatched_at if dispatched_at.tzinfo else dispatched_at.replace(tzinfo=timezone.utc)
    else:
        dispatched_dt = datetime.now(timezone.utc)
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
        processed_at=dispatched_dt,
        metadata={
            "agent": agent_nick,
            "dispatched_at": dispatched_dt.isoformat(),
        },
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
        poll_jitter_seconds: float = 3.0,
        match_threshold: float = 0.8,
        soft_timeout_minutes: Optional[int] = 360,
        grace_period_minutes: int = 45,
    ) -> None:
        self.agent_nick = agent_nick
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self._email_fetcher = email_fetcher
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn
        self.poll_jitter_seconds = poll_jitter_seconds if poll_jitter_seconds >= 0 else 0.0
        self._last_unmatched: List[Dict[str, Any]] = []
        self.match_threshold = match_threshold
        self.soft_timeout_minutes = soft_timeout_minutes if soft_timeout_minutes and soft_timeout_minutes > 0 else None
        self.grace_period_minutes = max(0, grace_period_minutes)
        self._dispatch_cache: Dict[str, Dict[str, EmailDispatchRecord]] = {}
        self._response_coordinator = get_supplier_response_coordinator()
        tracking_repo.init_schema()
        supplier_response_repo.init_schema()

    def _fetch(self, **kwargs) -> List[EmailResponse]:
        if self._email_fetcher is not None:
            payload = self._email_fetcher(**kwargs)
        else:
            payload = _default_fetcher(**kwargs)
        responses, _ = _normalise_fetch_result(payload)
        return responses

    def _sleep(self) -> None:
        try:
            duration = float(self.poll_interval_seconds)
            if self.poll_jitter_seconds:
                duration += random.uniform(0, float(self.poll_jitter_seconds))
            self._sleep_fn(duration)
        except Exception:  # pragma: no cover - defensive
            duration = float(self.poll_interval_seconds)
            if self.poll_jitter_seconds:
                duration += random.uniform(0, float(self.poll_jitter_seconds))
            time.sleep(duration)

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

    def _resolve_outbound(
        self,
        response: EmailResponse,
        *,
        matched_unique_id: Optional[str] = None,
    ) -> Optional[SupplierInteractionRow]:
        row: Optional[Dict[str, Any]] = None
        if matched_unique_id:
            row = supplier_interaction_repo.lookup_outbound(matched_unique_id)

        if row is None and response.unique_id:
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

    def _response_fingerprint(self, response: EmailResponse) -> Optional[str]:
        subject_norm = _normalise_subject_line(response.subject) or ""
        from_norm = _normalise_email_address(response.from_address) or ""
        received_at = response.received_at or self._now_fn()
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
        timestamp = received_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        payload = "|".join([subject_norm, from_norm, timestamp])
        if not payload.strip("|"):
            return None
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _build_dispatch_record(
        self,
        workflow_id: str,
        tracking_row: Optional[WorkflowDispatchRow],
        outbound_row: Optional[Dict[str, Any]],
    ) -> EmailDispatchRecord:
        if tracking_row is None and not outbound_row:
            raise ValueError("dispatch record requires source data")
        unique_id = (
            tracking_row.unique_id
            if tracking_row is not None
            else str(outbound_row.get("unique_id") or "")
        )
        supplier_id = None
        supplier_email = None
        message_id = None
        subject = None
        dispatched_at = None
        rfq_id = None
        round_number = None

        if outbound_row:
            supplier_id = outbound_row.get("supplier_id") or supplier_id
            supplier_email = outbound_row.get("supplier_email") or supplier_email
            message_id = outbound_row.get("message_id") or message_id
            subject = outbound_row.get("subject") or subject
            dispatched_at = outbound_row.get("dispatched_at") or dispatched_at
            rfq_id = outbound_row.get("rfq_id") or rfq_id
            round_number = outbound_row.get("round_number")

        if tracking_row is not None:
            supplier_id = tracking_row.supplier_id or supplier_id
            supplier_email = tracking_row.supplier_email or supplier_email
            message_id = tracking_row.message_id or message_id
            subject = tracking_row.subject or subject
            dispatched_at = tracking_row.dispatched_at or dispatched_at

        if dispatched_at is None:
            dispatched_at = self._now_fn()

        thread_headers: Dict[str, Sequence[str]] = {}
        if tracking_row and tracking_row.thread_headers:
            thread_headers = {
                key: tuple(value)
                for key, value in tracking_row.thread_headers.items()
                if value
            }
        elif outbound_row:
            thread_headers = {
                "in_reply_to": tuple(outbound_row.get("in_reply_to") or []),
                "references": tuple(
                    outbound_row.get("references")
                    or outbound_row.get("reference_ids")
                    or []
                ),
            }

        dispatch_key: Optional[str] = None
        if tracking_row and tracking_row.dispatch_key:
            dispatch_key = str(tracking_row.dispatch_key)
        elif outbound_row and outbound_row.get("dispatch_key"):
            dispatch_key = str(outbound_row.get("dispatch_key"))
        elif message_id:
            dispatch_key = str(message_id)
        elif unique_id:
            dispatch_key = str(unique_id)
        else:
            dispatch_key = uuid.uuid4().hex

        if not unique_id:
            unique_id = str(dispatch_key)

        return EmailDispatchRecord(
            workflow_id=workflow_id,
            unique_id=unique_id,
            dispatch_key=str(dispatch_key),
            supplier_id=str(supplier_id) if supplier_id else None,
            supplier_email=str(supplier_email) if supplier_email else None,
            message_id=str(message_id) if message_id else None,
            subject=str(subject) if subject else None,
            rfq_id=str(rfq_id) if rfq_id else None,
            thread_headers={key: tuple(value) for key, value in thread_headers.items() if value},
            dispatched_at=dispatched_at,
            round_number=round_number if isinstance(round_number, int) else None,
        )

    def _build_dispatch_index(
        self, workflow_id: str, round_number: Optional[int]
    ) -> Tuple[Dict[str, EmailDispatchRecord], List[WorkflowDispatchRow]]:
        outbound_rows = self._load_outbound_snapshot(workflow_id, round_number)
        outbound_map = {
            row.get("unique_id"): row
            for row in outbound_rows
            if row.get("unique_id")
        }
        tracking_rows = tracking_repo.load_workflow_rows(workflow_id=workflow_id)
        index: Dict[str, EmailDispatchRecord] = {}
        for tracking_row in tracking_rows:
            unique_id = tracking_row.unique_id
            outbound_row = outbound_map.get(unique_id)
            record = self._build_dispatch_record(workflow_id, tracking_row, outbound_row)
            index[unique_id] = record

        for unique_id, outbound_row in outbound_map.items():
            if unique_id in index:
                continue
            index[unique_id] = self._build_dispatch_record(workflow_id, None, outbound_row)

        self._dispatch_cache[workflow_id] = index
        return index, tracking_rows

    def run_once(
        self,
        workflow_id: str,
        dispatch_index: Dict[str, EmailDispatchRecord],
        tracking_rows: Optional[Sequence[WorkflowDispatchRow]] = None,
    ) -> List[Dict[str, object]]:
        responses = self._fetch(workflow_id=workflow_id)
        if not responses:
            return []

        supplier_interaction_repo.init_schema()
        stored: List[Dict[str, object]] = []
        self._last_unmatched = []
        pending_unique_ids: Set[str] = {
            unique_id for unique_id in dispatch_index.keys() if unique_id
        }
        if tracking_rows:
            for row in tracking_rows:
                unique_id = getattr(row, "unique_id", None)
                if not unique_id:
                    continue
                if getattr(row, "matched", False) or getattr(row, "responded_at", None):
                    pending_unique_ids.discard(unique_id)
                elif getattr(row, "response_message_id", None):
                    pending_unique_ids.discard(unique_id)
        for response in responses:
            stored_record = self._handle_response(
                response, dispatch_index, pending_unique_ids
            )
            if stored_record:
                stored.append(stored_record)
        return stored

    def _handle_response(
        self,
        response: EmailResponse,
        dispatch_index: Dict[str, EmailDispatchRecord],
        pending_unique_ids: Set[str],
    ) -> Optional[Dict[str, object]]:
        matched_unique: Optional[str] = None
        best_score = 0.0
        best_reasons: List[str] = []
        best_dispatch: Optional[EmailDispatchRecord] = None
        for unique_id, dispatch in dispatch_index.items():
            if pending_unique_ids and unique_id not in pending_unique_ids:
                continue
            score, reasons = _calculate_match_score(dispatch, response)
            if score > best_score:
                best_score = score
                matched_unique = unique_id
                best_reasons = reasons
                best_dispatch = dispatch

        fingerprint = self._response_fingerprint(response)

        unmatched = (
            not matched_unique
            or best_score < self.match_threshold
            or best_dispatch is None
        )

        if unmatched and pending_unique_ids:
            remaining_pending = [uid for uid in pending_unique_ids]
            if matched_unique and matched_unique in remaining_pending:
                remaining_pending = [matched_unique]
            if len(remaining_pending) == 1:
                fallback_unique = remaining_pending[0]
                fallback_dispatch = dispatch_index.get(fallback_unique)
                if fallback_dispatch is not None:
                    matched_unique = fallback_unique
                    best_dispatch = fallback_dispatch
                    best_score = max(best_score, self.match_threshold)
                    best_reasons = list({*best_reasons, "sole_pending_dispatch"})
                    unmatched = False

        if unmatched:
            subject_norm = _normalise_subject_line(response.subject)
            logger.warning(
                "services.email_watcher_v2 unable to match response agent=%s workflow=%s message_id=%s best_score=%.2f matched_on=%s",
                self.agent_nick,
                getattr(response, "workflow_id", None),
                response.message_id,
                best_score,
                best_reasons,
            )
            self._last_unmatched.append(
                {
                    "message_id": response.message_id,
                    "subject": response.subject,
                    "subject_normalised": subject_norm,
                    "subject_hash": _subject_hash(response.subject),
                    "from_address": response.from_address,
                    "received_at": response.received_at,
                    "reason": "low_confidence_match",
                    "best_score": best_score,
                    "matched_on": best_reasons,
                    "fingerprint": fingerprint,
                }
            )
            return None

        outbound = self._resolve_outbound(response, matched_unique_id=matched_unique)
        if outbound is None:
            subject_norm = _normalise_subject_line(response.subject)
            logger.warning(
                "services.email_watcher_v2 missing outbound row agent=%s workflow=%s unique_id=%s",
                self.agent_nick,
                best_dispatch.workflow_id,
                matched_unique,
            )
            self._last_unmatched.append(
                {
                    "message_id": response.message_id,
                    "subject": response.subject,
                    "subject_normalised": subject_norm,
                    "subject_hash": _subject_hash(response.subject),
                    "from_address": response.from_address,
                    "received_at": response.received_at,
                    "reason": "no_outbound_record",
                    "best_score": best_score,
                    "matched_on": best_reasons,
                    "fingerprint": fingerprint,
                }
            )
            return None

        if matched_unique:
            pending_unique_ids.discard(matched_unique)

        metadata_payload: Dict[str, Any] = {"agent": self.agent_nick, "matched_on": best_reasons}
        if response.received_at:
            received_ts = (
                response.received_at
                if response.received_at.tzinfo
                else response.received_at.replace(tzinfo=timezone.utc)
            )
            metadata_payload["received_at"] = received_ts.isoformat()
        if best_dispatch and best_dispatch.dispatched_at:
            dispatch_ts = (
                best_dispatch.dispatched_at
                if best_dispatch.dispatched_at.tzinfo
                else best_dispatch.dispatched_at.replace(tzinfo=timezone.utc)
            )
            metadata_payload.setdefault("dispatched_at", dispatch_ts.isoformat())

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
            metadata=metadata_payload,
        )

        workflow_id = outbound.workflow_id
        supplier_email = (
            response.supplier_email
            or _normalise_email_address(response.from_address)
            or outbound.supplier_email
        )
        supplier_id = response.supplier_id or outbound.supplier_id
        response_time: Optional[Decimal] = None
        if best_dispatch.dispatched_at and response.received_at:
            try:
                delta_seconds = (
                    response.received_at - best_dispatch.dispatched_at
                ).total_seconds()
                if delta_seconds >= 0:
                    response_time = Decimal(str(delta_seconds))
            except Exception:  # pragma: no cover - defensive conversion
                response_time = None

        tracking_repo.mark_response(
            workflow_id=workflow_id,
            unique_id=matched_unique,
            responded_at=response.received_at,
            response_message_id=response.message_id,
        )

        supplier_response_repo.insert_response(
            SupplierResponseRow(
                workflow_id=workflow_id,
                unique_id=matched_unique,
                supplier_id=supplier_id,
                supplier_email=supplier_email,
                rfq_id=response.rfq_id or best_dispatch.rfq_id,
                response_text=response.body,
                response_body=response.body_html or response.body,
                response_message_id=response.message_id,
                response_subject=response.subject,
                response_from=response.from_address,
                received_time=response.received_at,
                round_number=best_dispatch.round_number,
                response_time=response_time,
                original_message_id=best_dispatch.message_id,
                original_subject=best_dispatch.subject,
                match_confidence=_score_to_confidence(best_score),
                match_evidence=best_reasons,
                raw_headers=response.headers,
                processed=False,
            )
        )

        try:
            self._response_coordinator.record_response(
                workflow_id,
                matched_unique,
                round_number=best_dispatch.round_number,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to register supplier response completion workflow=%s unique_id=%s",
                workflow_id,
                matched_unique,
            )

        logger.info(
            "services.email_watcher_v2 captured supplier response workflow=%s unique_id=%s best_score=%.2f matched_on=%s",
            workflow_id,
            matched_unique,
            best_score,
            best_reasons,
        )

        return {
            "unique_id": matched_unique,
            "supplier_id": supplier_id,
            "workflow_id": workflow_id,
            "status": "received",
            "subject": response.subject,
            "message_id": response.message_id,
            "received_at": response.received_at,
            "from_address": response.from_address,
            "subject_normalised": _normalise_subject_line(response.subject),
            "subject_hash": _subject_hash(response.subject),
            "match_score": best_score,
            "matched_on": best_reasons,
            "fingerprint": fingerprint,
        }

    def _load_outbound_snapshot(
        self, workflow_id: str, round_number: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Return outbound dispatch rows for the workflow."""

        statuses = ["pending", "sent", "received", "processed"]
        snapshot: Dict[str, Dict[str, Any]] = {}
        for status in statuses:
            rows = supplier_interaction_repo.fetch_by_status(
                workflow_id=workflow_id,
                status=status,
                direction="outbound",
                round_number=round_number,
            )
            for row in rows:
                snapshot[row["unique_id"]] = row
        return list(snapshot.values())

    def _load_existing_responses(
        self, workflow_id: str, round_number: Optional[int]
    ) -> Dict[str, Dict[str, Any]]:
        """Return already-recorded inbound responses for the workflow."""

        statuses = ["received", "processed"]
        existing: Dict[str, Dict[str, Any]] = {}
        for status in statuses:
            rows = supplier_interaction_repo.fetch_by_status(
                workflow_id=workflow_id,
                status=status,
                direction="inbound",
                round_number=round_number,
            )
            for row in rows:
                subject_norm = _normalise_subject_line(row.get("subject"))
                response_record: Dict[str, Any] = {
                    "unique_id": row.get("unique_id"),
                    "supplier_id": row.get("supplier_id"),
                    "workflow_id": row.get("workflow_id"),
                    "status": row.get("status", status),
                    "subject": row.get("subject"),
                    "message_id": row.get("message_id"),
                    "received_at": row.get("received_at"),
                    "from_address": row.get("supplier_email") or row.get("from_address"),
                    "subject_normalised": subject_norm,
                    "subject_hash": _subject_hash(row.get("subject")),
                }
                unique_id = row.get("unique_id")
                if unique_id:
                    existing[unique_id] = response_record
        return existing

    def run_continuously(
        self,
        workflow_id: str,
        *,
        round_number: Optional[int] = None,
        until: Optional[datetime] = None,
        timeout_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        collected: Dict[str, Dict[str, object]] = {}
        unmatched: Dict[str, Dict[str, Any]] = {}
        seen_message_ids: Set[str] = set()
        seen_subject_hashes: Set[str] = set()
        seen_fingerprints: Set[str] = set()
        start = self._now_fn()
        state: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "round": round_number,
            "started_at": start,
            "expected_responses": 0,
            "responses_received": 0,
            "pending_suppliers": [],
            "matched_dispatch_ids": [],
            "unmatched_inbound": [],
            "status": "awaiting_dispatch",
            "late_suppliers": [],
            "timeout_reason": None,
            "soft_timeout_triggered": False,
        }

        existing = self._load_existing_responses(workflow_id, round_number)
        for uid, record in existing.items():
            collected[uid] = record
            msg_id = record.get("message_id")
            subj_hash = record.get("subject_hash")
            if msg_id:
                seen_message_ids.add(msg_id)
            if subj_hash:
                seen_subject_hashes.add(subj_hash)

        poll_index = 0
        registered_expected = 0
        soft_deadline: Optional[datetime] = None
        if self.soft_timeout_minutes:
            soft_deadline = start + timedelta(minutes=self.soft_timeout_minutes)
        hard_deadline: Optional[datetime] = None
        if timeout_minutes:
            hard_deadline = start + timedelta(minutes=timeout_minutes)
        if until:
            hard_deadline = (
                min(hard_deadline, until) if hard_deadline is not None else until
            )

        while True:
            poll_index += 1
            now = self._now_fn()
            state["last_poll_at"] = now
            state["elapsed_seconds"] = max(0.0, (now - start).total_seconds())

            dispatch_index, tracking_rows = self._build_dispatch_index(
                workflow_id, round_number
            )
            expected_total = len(tracking_rows) if tracking_rows else len(dispatch_index)
            if expected_total > state["expected_responses"]:
                state["expected_responses"] = expected_total

            expected_unique_ids = [
                row.unique_id for row in tracking_rows if getattr(row, "unique_id", None)
            ]
            if not expected_unique_ids and dispatch_index:
                expected_unique_ids = list(dispatch_index.keys())
            if expected_total and expected_total != registered_expected:
                try:
                    self._response_coordinator.register_expected_responses(
                        workflow_id,
                        expected_unique_ids,
                        expected_total,
                        round_number=round_number,
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "Failed to register expected responses workflow=%s",
                        workflow_id,
                    )
                registered_expected = expected_total

            if tracking_rows:
                pending_suppliers = sorted(
                    {
                        row.supplier_id
                        or row.supplier_email
                        or row.unique_id
                        for row in tracking_rows
                        if not row.matched and not row.responded_at
                    }
                )
            else:
                pending_suppliers = sorted(
                    {
                        dispatch.supplier_id
                        or dispatch.supplier_email
                        or dispatch.unique_id
                        for dispatch in dispatch_index.values()
                    }
                )
            state["pending_suppliers"] = pending_suppliers

            stored_batch = self.run_once(
                workflow_id, dispatch_index, tracking_rows
            )
            for record in stored_batch:
                msg_id = record.get("message_id")
                subj_hash = record.get("subject_hash")
                fingerprint = record.get("fingerprint")
                subject_norm = record.get("subject_normalised")
                if msg_id and msg_id in seen_message_ids:
                    logger.info(
                        "services.email_watcher_v2 deduped inbound workflow=%s unique_id=%s action=deduped reason=message_id subject_norm=%s",
                        workflow_id,
                        record.get("unique_id"),
                        subject_norm,
                    )
                    continue
                if (
                    fingerprint
                    and not msg_id
                    and fingerprint in seen_fingerprints
                ):
                    logger.info(
                        "services.email_watcher_v2 deduped inbound workflow=%s unique_id=%s action=deduped reason=fingerprint subject_norm=%s",
                        workflow_id,
                        record.get("unique_id"),
                        subject_norm,
                    )
                    continue
                if subj_hash and subj_hash in seen_subject_hashes:
                    logger.info(
                        "services.email_watcher_v2 deduped inbound workflow=%s unique_id=%s action=deduped reason=subject_hash subject_hash=%s subject_norm=%s",
                        workflow_id,
                        record.get("unique_id"),
                        subj_hash,
                        subject_norm,
                    )
                    continue
                collected[record["unique_id"]] = record
                if msg_id:
                    seen_message_ids.add(msg_id)
                if subj_hash:
                    seen_subject_hashes.add(subj_hash)
                if not msg_id and fingerprint:
                    seen_fingerprints.add(fingerprint)

            for unmatched_record in getattr(self, "_last_unmatched", []):
                key = (
                    unmatched_record.get("message_id")
                    or unmatched_record.get("subject_hash")
                    or unmatched_record.get("fingerprint")
                )
                if not key:
                    key = uuid.uuid4().hex
                if key not in unmatched:
                    unmatched[key] = unmatched_record
                    logger.info(
                        "services.email_watcher_v2 unmatched inbound workflow=%s message_id=%s reason=%s subject_norm=%s best_score=%s",
                        workflow_id,
                        unmatched_record.get("message_id"),
                        unmatched_record.get("reason"),
                        unmatched_record.get("subject_normalised"),
                        unmatched_record.get("best_score"),
                    )

            state["responses_received"] = len(collected)
            state["matched_dispatch_ids"] = sorted(collected.keys())
            state["unmatched_inbound"] = list(unmatched.values())

            logger.info(
                "services.email_watcher_v2 poll workflow=%s poll_index=%d expected=%d responded=%d pending=%s elapsed=%.1fs mailbox=%s",
                workflow_id,
                poll_index,
                state["expected_responses"],
                state["responses_received"],
                ",".join(state["pending_suppliers"]) or "-",
                state["elapsed_seconds"],
                self.agent_nick,
            )

            expected = state["expected_responses"]
            responded = state["responses_received"]
            if expected and responded >= expected:
                state["status"] = "complete"
                break

            if (
                soft_deadline
                and not state["soft_timeout_triggered"]
                and now >= soft_deadline
                and expected
                and responded < expected
            ):
                state["status"] = "soft_timeout"
                state["timeout_reason"] = "soft"
                state["soft_timeout_triggered"] = True
                state["late_suppliers"] = pending_suppliers
                logger.warning(
                    "services.email_watcher_v2 soft timeout workflow=%s expected=%d responded=%d pending=%s",
                    workflow_id,
                    expected,
                    responded,
                    ",".join(pending_suppliers) or "-",
                )
                if hard_deadline is None and self.grace_period_minutes:
                    hard_deadline = now + timedelta(minutes=self.grace_period_minutes)

            if (
                hard_deadline
                and now >= hard_deadline
                and expected
                and responded < expected
            ):
                state["status"] = "timeout"
                state["timeout_reason"] = "hard"
                break

            self._sleep()

        state["responses"] = list(collected.values())
        if state["status"] == "complete":
            logger.info(
                "EmailWatcher responses complete workflow=%s expected=%d responded=%d elapsed=%.1fs",
                workflow_id,
                state["expected_responses"],
                state["responses_received"],
                state.get("elapsed_seconds", 0.0),
            )
            try:
                self._response_coordinator.clear(workflow_id)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to clear response coordinator for workflow=%s",
                    workflow_id,
                )
        else:
            if state["status"] == "timeout" and state["responses_received"] < state["expected_responses"]:
                state["status"] = "partial"
            logger.warning(
                "EmailWatcher responses incomplete workflow=%s status=%s expected=%d responded=%d pending=%s",
                workflow_id,
                state["status"],
                state["expected_responses"],
                state["responses_received"],
                ",".join(state["pending_suppliers"]) or "-",
            )

        state.pop("soft_timeout_triggered", None)
        return state