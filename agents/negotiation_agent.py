from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import binascii
import threading
import time
from copy import deepcopy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import uuid
import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

from html import escape
from email.utils import parsedate_to_datetime

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from agents.email_drafting_agent import EmailDraftingAgent, DEFAULT_NEGOTIATION_SUBJECT
from repositories import (
    supplier_response_repo,
    workflow_lifecycle_repo,
    workflow_round_response_repo,
)
from repositories.workflow_round_response_repo import RoundStatus
from services.supplier_response_coordinator import (
    get_supplier_response_coordinator,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from agents.supplier_interaction_agent import SupplierInteractionAgent
from utils.gpu import configure_gpu
from utils.email_markers import attach_hidden_marker

from pathlib import Path

from services.redis_client import get_workflow_redis_client
from services.lmstudio_client import get_lmstudio_client

logger = logging.getLogger(__name__)

# -----------------------------
# Tunables & feature toggles
# -----------------------------
MAX_SUPPLIER_REPLIES = int(os.getenv("NEG_MAX_SUPPLIER_REPLIES", "3"))
LLM_ENABLED = os.getenv("NEG_ENABLE_LLM", "1").strip() not in {"0", "false", "False"}
LLM_MODEL = os.getenv("NEG_LLM_MODEL", "llama3.2:latest")
COST_OF_CAPITAL_APR = float(os.getenv("NEG_COST_OF_CAPITAL_APR", "0.12"))
LEAD_TIME_VALUE_PCT_PER_WEEK = float(os.getenv("NEG_LT_VALUE_PCT_PER_WEEK", "0.01"))
def _resolve_thread_transcript_limit() -> Optional[int]:
    """Return the configured transcript limit or ``None`` for full history."""

    raw_limit = os.getenv("NEG_THREAD_TRANSCRIPT_LIMIT")
    if raw_limit is None:
        return None

    raw_limit = raw_limit.strip()
    if not raw_limit:
        return None

    try:
        parsed = int(raw_limit)
    except ValueError:
        logger.warning(
            "Invalid NEG_THREAD_TRANSCRIPT_LIMIT=%s; defaulting to full history",
            raw_limit,
        )
        return None

    if parsed <= 0:
        return None

    return parsed


THREAD_HISTORY_TRANSCRIPT_LIMIT = _resolve_thread_transcript_limit()
AGGRESSIVE_FIRST_COUNTER_PCT = float(os.getenv("NEG_FIRST_COUNTER_AGGR_PCT", "0.12"))
FINAL_OFFER_PATTERNS = (
    "best and final",
    "best final",
    "best offer we can",
    "best price we can",
    "cannot go lower",
    "final offer",
    "final price",
    "final quotation",
    "last price",
    "lowest we can do",
    "our best price",
    "rock bottom",
    "take it or leave it",
    "ultimatum",
)

LEVER_CATEGORIES = {
    "COMMERCIAL",
    "OPERATIONAL",
    "RISK",
    "STRATEGIC",
    "RELATIONAL",
}

TRADE_OFF_HINTS = {
    "Commercial": "May require volume commitments, stepped pricing, or altered cash flow.",
    "Operational": "May reduce supplier flexibility or need shared planning resources.",
    "Risk": "Could introduce legal negotiation overhead or stricter enforcement costs.",
    "Strategic": "Requires executive sponsorship and potential co-investment or exclusivity.",
    "Relational": "Demands governance time and tighter alignment of internal stakeholders.",
}


@dataclass(frozen=True)
class UniqueConstraintInfo:
    columns: Tuple[str, ...]
    constraint_name: Optional[str] = None
    predicate: Optional[str] = None
    index_name: Optional[str] = None


class NegotiationEmailHTMLShellBuilder:
    """Render negotiation drafts into a modern email-safe HTML shell."""

    BRAND_LABEL = "Beyond Procwise"
    MAX_PREHEADER = 160
    PARAGRAPH_STYLE = (
        "margin:0 0 16px 0;" "font-size:15px;" "line-height:1.6;" "color:#1f2937;"
    )
    LIST_STYLE = "margin:0 0 16px 24px;padding:0;"
    LIST_ITEM_STYLE = (
        "margin:0 0 8px 0;" "padding:0;" "font-size:15px;" "line-height:1.6;" "color:#1f2937;"
    )

    def __init__(self, *, brand_label: Optional[str] = None) -> None:
        self.brand_label = (brand_label or self.BRAND_LABEL).strip() or self.BRAND_LABEL

    @staticmethod
    def _parse_blocks(text: str) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        bullets: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if bullets:
                    blocks.append({"type": "list", "items": list(bullets)})
                    bullets.clear()
                continue
            bullet_match = re.match(r"^[-*•]\s+(.*)$", stripped)
            if bullet_match:
                bullets.append(bullet_match.group(1).strip())
                continue
            if bullets:
                blocks.append({"type": "list", "items": list(bullets)})
                bullets.clear()
            blocks.append({"type": "paragraph", "text": stripped})
        if bullets:
            blocks.append({"type": "list", "items": list(bullets)})
        return blocks

    @classmethod
    def _render_blocks(cls, blocks: List[Dict[str, Any]]) -> str:
        html_parts: List[str] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "list":
                items = block.get("items") or []
                if not isinstance(items, Sequence):
                    continue
                rendered_items = "".join(
                    f'<li style="{cls.LIST_ITEM_STYLE}">{escape(str(item))}</li>'
                    for item in items
                )
                if rendered_items:
                    html_parts.append(f'<ul style="{cls.LIST_STYLE}">{rendered_items}</ul>')
            elif block_type == "paragraph":
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                html_parts.append(f'<p style="{cls.PARAGRAPH_STYLE}">{escape(text)}</p>')
        return "".join(html_parts)

    @staticmethod
    def _build_preheader(text: str) -> str:
        if not text:
            return ""
        collapsed = re.sub(r"\s+", " ", text).strip()
        if not collapsed:
            return ""
        if len(collapsed) <= NegotiationEmailHTMLShellBuilder.MAX_PREHEADER:
            return collapsed
        truncated = collapsed[: NegotiationEmailHTMLShellBuilder.MAX_PREHEADER - 1].rstrip()
        return f"{truncated}…"

    def build(self, *, subject: Optional[str], body_text: str, preheader: Optional[str] = None) -> str:
        if not isinstance(body_text, str):
            return ""
        trimmed_body = body_text.strip()
        if not trimmed_body:
            return ""

        safe_subject = escape((subject or "Negotiation Update").strip() or "Negotiation Update")
        blocks = self._parse_blocks(trimmed_body)
        body_html = self._render_blocks(blocks)
        if not body_html:
            return ""

        preheader_source = preheader if isinstance(preheader, str) else trimmed_body
        preheader_text = escape(self._build_preheader(preheader_source))
        brand_html = escape(self.brand_label)

        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\"/>\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>\n"
            f"  <title>{safe_subject}</title>\n"
            "</head>\n"
            '<body style="margin:0;padding:0;background-color:#f5f6fa;">\n'
            f"  <div style=\"display:none;font-size:1px;color:#f5f6fa;line-height:1px;max-height:0;max-width:0;opacity:0;overflow:hidden;\">{preheader_text}</div>\n"
            '  <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color:#f5f6fa;">\n'
            "    <tr>\n"
            '      <td align="center" style="padding:24px 16px;">\n'
            '        <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:640px;background-color:#ffffff;border-radius:14px;border:1px solid #e2e8f0;overflow:hidden;">\n'
            "          <tr>\n"
            '            <td style="padding:24px 32px;background-color:#0f172a;color:#f8fafc;font-family:\'Segoe UI\',Arial,sans-serif;font-size:16px;font-weight:600;letter-spacing:0.02em;">\n'
            f"              {brand_html}\n"
            "            </td>\n"
            "          </tr>\n"
            "          <tr>\n"
            '            <td style="padding:24px 32px 8px 32px;font-family:\'Segoe UI\',Arial,sans-serif;font-size:22px;font-weight:600;color:#0f172a;">\n'
            f"              {safe_subject}\n"
            "            </td>\n"
            "          </tr>\n"
            "          <tr>\n"
            '            <td style="padding:0 32px 32px 32px;font-family:\'Segoe UI\',Arial,sans-serif;">\n'
            f"              {body_html}\n"
            "            </td>\n"
            "          </tr>\n"
            "          <tr>\n"
            '            <td style="padding:20px 32px 28px 32px;background-color:#f8fafc;font-family:\'Segoe UI\',Arial,sans-serif;font-size:12px;line-height:1.6;color:#64748b;border-top:1px solid #e2e8f0;">\n'
            "              <p style=\"margin:0 0 6px 0;font-weight:600;color:#0f172a;\">Human review required</p>\n"
            "              <p style=\"margin:0;\">This negotiation draft was prepared by Beyond Procwise’s agentic framework and must be validated by a procurement lead before sending to the supplier.</p>\n"
            "            </td>\n"
            "          </tr>\n"
            "        </table>\n"
            "      </td>\n"
            "    </tr>\n"
            "  </table>\n"
            "</body>\n"
            "</html>"
        )


@dataclass
class NegotiationContext:
    current_offer: float
    target_price: float
    round_index: int = 1
    currency: Optional[str] = None
    aggressiveness: float = 0.5
    leverage: float = 0.5
    urgency: float = 0.5
    risk_buffer_pct: float = 0.05
    min_abs_buffer: float = 0.0
    step_pct_of_gap: float = 0.1
    min_abs_step: float = 1.0
    max_rounds: int = 3
    walkaway_price: Optional[float] = None
    ask_early_pay_disc: Optional[float] = None
    ask_lead_time_keep: bool = True


@dataclass
class SupplierSignals:
    offer_prev: Optional[float] = None
    offer_new: Optional[float] = None
    message_text: str = ""


def _detect_finality(message: str) -> bool:
    lowered = (message or "").lower()
    return any(pattern in lowered for pattern in FINAL_OFFER_PATTERNS)


def _format_currency(amount: float, currency: Optional[str]) -> str:
    if currency:
        return f"{currency} {amount:,.2f}"
    return f"{amount:,.2f}"


def plan_counter(ctx: NegotiationContext, signals: SupplierSignals) -> Dict[str, object]:
    """Plan counter pricing and supporting asks for the negotiation agent."""

    log: List[str] = []
    asks: List[str] = []
    lead_time_request: Optional[str] = None

    if ctx.round_index > ctx.max_rounds:
        counter = min(ctx.current_offer, ctx.walkaway_price or ctx.current_offer)
        counter = round(counter, 2)
        message = (
            f"Round {ctx.round_index} exceeds configured max rounds; hold at "
            f"{_format_currency(counter, ctx.currency)} and focus on non-price levers."
        )
        log.append("Max rounds reached; maintaining prior position.")
        return {
            "decision": "hold",
            "counter_price": counter,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": False,
        }

    if ctx.current_offer <= 0 or ctx.target_price <= 0:
        message = "Offer or target missing/invalid; request structured pricing."
        asks.append("Confirm unit price, currency, tiered price @ 100/250/500.")
        log.append("Invalid numeric input detected while planning counter.")
        return {
            "decision": "clarify",
            "counter_price": None,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": False,
        }

    current_offer = float(ctx.current_offer)
    target_price = float(ctx.target_price)
    gap_value = current_offer - target_price
    gap_pct = gap_value / target_price if target_price else None
    finality = _detect_finality(signals.message_text)

    threshold = ctx.walkaway_price if ctx.walkaway_price is not None else target_price

    if finality:
        log.append("Supplier message contained final-offer language.")
        if threshold is not None and current_offer <= threshold:
            counter = round(min(current_offer, target_price), 2)
            message = (
                "Supplier signalled final offer within acceptable threshold; accept with a minor sweetener."
            )
            if ctx.ask_early_pay_disc:
                asks.append(
                    f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount on this final offer."
                )
            asks.append("Confirm net 30 terms and shipment schedule before closing.")
            if ctx.ask_lead_time_keep:
                lead_time_request = "Confirm committed lead time"
            return {
                "decision": "accept",
                "counter_price": counter,
                "asks": list(dict.fromkeys(asks)),
                "lead_time_request": lead_time_request,
                "message": message,
                "log": log,
                "finality": True,
            }

        message = (
            "Supplier marked this as a final offer but it remains above our no-deal threshold; pause and escalate."
        )
        log.append("Final offer rejected because it exceeds walk-away threshold.")
        return {
            "decision": "decline",
            "counter_price": None,
            "asks": [],
            "lead_time_request": None,
            "message": message,
            "log": log,
            "finality": True,
        }

    if gap_value <= 0:
        counter = round(min(current_offer, target_price), 2)
        message = "Supplier already at/below target; accept while requesting a minor sweetener."
        log.append("Offer meets target; recommending soft acceptance.")
        if ctx.ask_early_pay_disc:
            asks.append(
                f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount."
            )
        asks.append("Confirm lead time and packaging before sign-off.")
        if ctx.ask_lead_time_keep:
            lead_time_request = "Maintain committed lead time"
        return {
            "decision": "accept",
            "counter_price": counter,
            "asks": list(dict.fromkeys(asks)),
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": finality,
        }

    round_no = ctx.round_index
    counter_price: float

    if round_no <= 1:
        if gap_pct is not None and gap_pct > 0.10:
            counter_price = max(target_price, current_offer * 0.88)
            log.append("Round 1 anchor strategy applied with ~12% reduction from supplier offer.")
            asks.extend(
                [
                    "Volume-based discount for 250/500 unit tiers?",
                    "Improved payment terms (net 45 or early-pay option).",
                    "Explore alternative specs or components to lower cost.",
                ]
            )
        else:
            counter_price = (current_offer + target_price) / 2
            log.append("Round 1 gap within 10%; proposing midpoint counter.")
            asks.extend(
                [
                    "Hold quoted price for the full project timeline.",
                    "Include expedited production slot if volumes increase.",
                ]
            )
        message_intro = "Round 1 plan"
    elif round_no == 2:
        if gap_pct is not None and gap_pct <= 0.10:
            counter_price = (current_offer + target_price) / 2
            log.append("Round 2 midpoint strategy: splitting difference with supplier.")
        else:
            step = gap_value * 0.6
            counter_price = current_offer - step
            log.append(
                "Round 2 assertive push: capturing ~60% of remaining gap to accelerate convergence."
            )
        message_intro = "Round 2 plan"
    else:
        buffer_value = max(ctx.min_abs_buffer, target_price * ctx.risk_buffer_pct)
        threshold = target_price + buffer_value
        if threshold < current_offer:
            counter_price = threshold
            log.append("Round 3+ buffer enforcement: targeting risk-adjusted threshold.")
        else:
            counter_price = max(target_price, current_offer * (1 - ctx.step_pct_of_gap))
            log.append("Round 3+ soft landing: small decrement while reinforcing asks.")
        message_intro = f"Round {round_no} plan"

    counter_price = round(max(counter_price, target_price), 2)

    if ctx.ask_early_pay_disc and ctx.ask_early_pay_disc > 0:
        asks.append(
            f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount if counter accepted."
        )
    if ctx.ask_lead_time_keep:
        lead_time_request = "Hold quoted lead time"
    asks.append("Validate packaging, warranty, and compliance terms before sign-off.")

    message = (
        f"{message_intro}: counter at {_format_currency(counter_price, ctx.currency)}."
        " Reinforce commercial levers while positioning for collaborative win-win."
    )

    return {
        "decision": "counter",
        "counter_price": counter_price,
        "asks": list(dict.fromkeys(asks)),
        "lead_time_request": lead_time_request,
        "message": message,
        "log": log,
        "finality": finality,
    }

PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "resources" / "reference_data" / "negotiation_playbook.json"

MARKET_REVIEW_THRESHOLD = float(os.getenv("NEG_MARKET_REVIEW_PCT", "0.2"))
MARKET_ESCALATION_THRESHOLD = float(os.getenv("NEG_MARKET_ESCALATION_PCT", "0.4"))
MAX_VOLUME_LIMIT = float(os.getenv("NEG_MAX_VOLUME_LIMIT", "1000"))
MAX_TERM_DAYS = int(os.getenv("NEG_MAX_TERM_DAYS", "120"))

DEFAULT_NEGOTIATION_MESSAGE_TEMPLATE = "{header}\n{details}{context_sections}"

BATCH_INPUT_KEYS = ("negotiation_batch", "supplier_responses_batch", "batch_responses")
BATCH_SHARED_KEYS = (
    "shared_context",
    "shared_payload",
    "batch_defaults",
    "shared_fields",
    "defaults",
)
BATCH_EXCLUDE_KEYS = {
    "negotiation_batch",
    "supplier_responses_batch",
    "batch_responses",
    "shared_context",
    "shared_payload",
    "batch_defaults",
    "shared_fields",
    "defaults",
    "batch_metadata",
    "batch_results",
    "batch_summary",
    "agentic_plan",
    "pass_fields",
    "results",
    "drafts",
    "supplier_responses",
}


@dataclass
class NegotiationIdentifier:
    workflow_id: str
    session_reference: str
    supplier_id: str
    round_number: int = 1

    def __post_init__(self) -> None:
        self.workflow_id = self._normalise(self.workflow_id, fallback_prefix="WF")
        self.session_reference = self._normalise(
            self.session_reference, fallback_prefix="WF"
        )
        self.supplier_id = self._normalise(self.supplier_id, fallback_prefix="SUP")
        try:
            self.round_number = int(self.round_number) if self.round_number else 1
        except Exception:
            self.round_number = 1

    @staticmethod
    def _normalise(value: Optional[str], *, fallback_prefix: str = "") -> str:
        if isinstance(value, str):
            token = value.strip()
        elif value is None:
            token = ""
        else:
            token = str(value).strip()
        if not token:
            return f"{fallback_prefix}-{uuid.uuid4().hex[:12].upper()}" if fallback_prefix else ""
        return token

    @property
    def unique_key(self) -> str:
        return f"{self.workflow_id}:{self.supplier_id}:{self.round_number}"

    @property
    def thread_key(self) -> str:
        return f"{self.workflow_id}:{self.supplier_id}"


@dataclass
class EmailThreadState:
    thread_id: str
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    subject_base: str = ""

    def to_headers(self, round_number: int) -> Dict[str, Any]:
        message_id = f"<{uuid.uuid4()}@procwise.co.uk>"
        headers: Dict[str, Any] = {"Message-ID": message_id}
        if self.references:
            headers["References"] = " ".join(self.references[-10:])
        if self.in_reply_to:
            headers["In-Reply-To"] = self.in_reply_to
        subject = self.subject_base or DEFAULT_NEGOTIATION_SUBJECT
        if round_number > 1 and subject:
            if subject.lower().startswith("re:"):
                headers["Subject"] = subject
            else:
                headers["Subject"] = f"Re: {subject}".strip()
        elif subject:
            headers["Subject"] = subject
        return headers

    def update_after_send(self, message_id: Optional[str]) -> None:
        token = self._normalise_token(message_id)
        if not token:
            return
        if not self.thread_id:
            self.thread_id = token
        if token not in self.references:
            self.references.append(token)
        self.in_reply_to = token

    def update_after_receive(self, message_id: Optional[str]) -> None:
        token = self._normalise_token(message_id)
        if not token:
            return
        if token not in self.references:
            self.references.append(token)
        self.in_reply_to = token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "in_reply_to": self.in_reply_to,
            "references": list(self.references),
            "subject_base": self.subject_base,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], *, fallback_subject: str) -> "EmailThreadState":
        thread_id = str(data.get("thread_id") or f"<{uuid.uuid4()}@procwise.co.uk>")
        references = data.get("references") if isinstance(data.get("references"), list) else []
        return EmailThreadState(
            thread_id=thread_id,
            in_reply_to=data.get("in_reply_to"),
            references=[str(item) for item in references if item],
            subject_base=str(data.get("subject_base") or fallback_subject or DEFAULT_NEGOTIATION_SUBJECT),
        )

    @staticmethod
    def _normalise_token(token: Optional[str]) -> Optional[str]:
        if isinstance(token, str):
            value = token.strip()
        elif token is None:
            value = ""
        else:
            value = str(token).strip()
        return value or None


class ResponseMatcher:
    """Match supplier responses to the registered negotiation rounds."""

    def __init__(self, db_connection_factory: Optional[Any]) -> None:
        self.db = db_connection_factory
        self._pending_responses: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def register_expected_response(
        self,
        identifier: NegotiationIdentifier,
        email_action_id: Optional[str],
        expected_unique_id: str,
        timeout_seconds: int = 900,
    ) -> None:
        payload = {
            "round": identifier.round_number,
            "email_action_id": email_action_id,
            "unique_id": expected_unique_id,
            "registered_at": datetime.now(timezone.utc),
            "timeout_at": datetime.now(timezone.utc)
            + timedelta(seconds=max(timeout_seconds, 60)),
            "thread_key": identifier.thread_key,
        }
        with self._lock:
            bucket = self._pending_responses.setdefault(identifier.workflow_id, {})
            bucket[identifier.supplier_id] = payload

    def match_response(
        self,
        workflow_id: str,
        supplier_id: str,
        message_id: Optional[str],
        response_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not workflow_id or not supplier_id:
            return None
        with self._lock:
            workflow_bucket = self._pending_responses.get(workflow_id)
            if not workflow_bucket:
                return None
            expected = workflow_bucket.get(supplier_id)
            if not expected:
                return None
            if datetime.now(timezone.utc) > expected.get(
                "timeout_at", datetime.now(timezone.utc)
            ):
                workflow_bucket.pop(supplier_id, None)
                return None
            workflow_bucket.pop(supplier_id, None)
        payload = dict(expected)
        payload["response_data"] = response_data or {}
        payload["message_id"] = message_id
        payload["matched_at"] = datetime.now(timezone.utc)
        return payload

    def get_pending_count(self, workflow_id: str) -> int:
        with self._lock:
            bucket = self._pending_responses.get(workflow_id) or {}
            return len(bucket)


@dataclass
class EmailHistoryEntry:
    email_id: str
    round_number: int
    supplier_id: str
    supplier_name: Optional[str]
    subject: str
    body_text: str
    body_html: str
    sender: str
    recipients: List[str]
    sent_at: datetime
    message_id: Optional[str]
    thread_headers: Dict[str, Any]
    metadata: Dict[str, Any]
    decision: Dict[str, Any]
    negotiation_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_id": self.email_id,
            "round_number": self.round_number,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "subject": self.subject,
            "body_text": self.body_text,
            "body_html": self.body_html,
            "sender": self.sender,
            "recipients": list(self.recipients),
            "sent_at": self.sent_at.isoformat()
            if isinstance(self.sent_at, datetime)
            else self.sent_at,
            "message_id": self.message_id,
            "thread_headers": dict(self.thread_headers),
            "metadata": dict(self.metadata),
            "decision": dict(self.decision),
            "negotiation_context": dict(self.negotiation_context),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EmailHistoryEntry":
        sent_at = data.get("sent_at")
        if isinstance(sent_at, str):
            try:
                sent_at = datetime.fromisoformat(sent_at)
            except Exception:
                sent_at = datetime.now(timezone.utc)
        elif not isinstance(sent_at, datetime):
            sent_at = datetime.now(timezone.utc)

        return EmailHistoryEntry(
            email_id=data.get("email_id") or str(uuid.uuid4()),
            round_number=int(data.get("round_number", 1)),
            supplier_id=str(data.get("supplier_id") or ""),
            supplier_name=data.get("supplier_name"),
            subject=data.get("subject", ""),
            body_text=data.get("body_text", ""),
            body_html=data.get("body_html", ""),
            sender=data.get("sender", ""),
            recipients=list(data.get("recipients") or []),
            sent_at=sent_at,
            message_id=data.get("message_id"),
            thread_headers=dict(data.get("thread_headers") or {}),
            metadata=dict(data.get("metadata") or {}),
            decision=dict(data.get("decision") or {}),
            negotiation_context=dict(data.get("negotiation_context") or {}),
        )


class EmailThreadManager:
    """Manage negotiation email history keyed by workflow and supplier."""

    def __init__(self) -> None:
        self._threads: Dict[str, List[EmailHistoryEntry]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _thread_key(workflow_id: str, supplier_id: str) -> str:
        return f"{workflow_id}:{supplier_id}"

    def set_thread(
        self, workflow_id: str, supplier_id: str, entries: Sequence[EmailHistoryEntry]
    ) -> None:
        key = self._thread_key(workflow_id, supplier_id)
        with self._lock:
            ordered = sorted(entries, key=lambda entry: entry.sent_at)
            self._threads[key] = ordered

    def add_email(
        self,
        workflow_id: str,
        supplier_id: str,
        email_entry: EmailHistoryEntry,
    ) -> None:
        key = self._thread_key(workflow_id, supplier_id)
        with self._lock:
            bucket = self._threads.setdefault(key, [])
            bucket.append(email_entry)
            bucket.sort(key=lambda entry: entry.sent_at)

    def get_thread(
        self, workflow_id: str, supplier_id: str
    ) -> List[EmailHistoryEntry]:
        key = self._thread_key(workflow_id, supplier_id)
        with self._lock:
            entries = self._threads.get(key, [])
            return list(entries)

    def get_thread_summary(
        self, workflow_id: str, supplier_id: str
    ) -> Dict[str, Any]:
        thread = self.get_thread(workflow_id, supplier_id)
        if not thread:
            return {
                "total_emails": 0,
                "rounds": [],
                "first_sent": None,
                "last_sent": None,
                "thread_key": self._thread_key(workflow_id, supplier_id),
            }

        rounds = sorted({entry.round_number for entry in thread})
        first_sent = thread[0].sent_at.isoformat()
        last_sent = thread[-1].sent_at.isoformat()

        return {
            "total_emails": len(thread),
            "rounds": rounds,
            "first_sent": first_sent,
            "last_sent": last_sent,
            "thread_key": self._thread_key(workflow_id, supplier_id),
        }


class NegotiationEmailHTMLBuilder:
    """Build professional HTML emails for negotiation rounds."""

    BASE_STYLES = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333333;
            max-width: 650px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .email-container {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin: 20px auto;
        }
        .header {
            border-bottom: 3px solid #0066cc;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        .header h1 {
            color: #0066cc;
            font-size: 24px;
            margin: 0 0 5px 0;
            font-weight: 600;
        }
        .round-badge {
            display: inline-block;
            background-color: #0066cc;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .greeting {
            font-size: 16px;
            margin-bottom: 20px;
            color: #333333;
        }
        .section {
            margin: 25px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #0066cc;
            border-radius: 4px;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #0066cc;
            margin: 0 0 12px 0;
        }
        .pricing-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background-color: #ffffff;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .pricing-table th {
            background-color: #0066cc;
            color: #ffffff;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
        }
        .pricing-table td {
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
            font-size: 14px;
        }
        .pricing-table tr:last-child td {
            border-bottom: none;
        }
        .pricing-table .label {
            font-weight: 600;
            color: #495057;
        }
        .pricing-table .value {
            color: #212529;
        }
        .pricing-table .highlight {
            background-color: #fff3cd;
            font-weight: 600;
            color: #856404;
        }
        .asks-list {
            margin: 15px 0;
            padding: 0;
            list-style: none;
        }
        .asks-list li {
            padding: 10px 15px;
            margin: 8px 0;
            background-color: #ffffff;
            border-left: 3px solid #28a745;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .asks-list li:before {
            content: "✓";
            color: #28a745;
            font-weight: bold;
            margin-right: 10px;
        }
        .callout {
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 6px;
            font-size: 14px;
        }
        .callout-info {
            background-color: #d1ecf1;
            border-left: 4px solid #0c5460;
            color: #0c5460;
        }
        .callout-warning {
            background-color: #fff3cd;
            border-left: 4px solid #856404;
            color: #856404;
        }
        .callout-success {
            background-color: #d4edda;
            border-left: 4px solid #155724;
            color: #155724;
        }
        .playbook-recommendations {
            margin: 20px 0;
            padding: 0;
        }
        .recommendation-item {
            padding: 15px;
            margin: 10px 0;
            background-color: #e7f3ff;
            border-left: 4px solid #0066cc;
            border-radius: 4px;
        }
        .recommendation-item .lever {
            font-weight: 600;
            color: #0066cc;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .recommendation-item .description {
            color: #495057;
            font-size: 14px;
            line-height: 1.5;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
            font-size: 14px;
            color: #6c757d;
        }
        .signature {
            margin-top: 25px;
            font-size: 15px;
            color: #333333;
        }
        .signature .name {
            font-weight: 600;
            color: #0066cc;
        }
        .button {
            display: inline-block;
            padding: 12px 24px;
            background-color: #0066cc;
            color: #ffffff !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            text-align: center;
            margin: 15px 0;
        }
        .button:hover {
            background-color: #0052a3;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-card .label {
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .metric-card .value {
            font-size: 20px;
            font-weight: 600;
            color: #0066cc;
        }
        @media only screen and (max-width: 600px) {
            body {
                padding: 10px;
            }
            .email-container {
                padding: 20px;
            }
            .pricing-table {
                font-size: 12px;
            }
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """

    @staticmethod
    def build_negotiation_email(
        *,
        round_number: int,
        contact_name: Optional[str],
        supplier_name: Optional[str],
        decision: Dict[str, Any],
        positions: Optional[Dict[str, Any]],
        currency: Optional[str],
        playbook_recommendations: Optional[List[Dict[str, Any]]],
        negotiation_message: str,
        sender_name: Optional[str] = None,
        company_name: Optional[str] = "Procwise",
    ) -> str:
        strategy = decision.get("strategy", "counter")
        counter_price = decision.get("counter_price")
        asks = decision.get("asks", [])
        lead_time_request = decision.get("lead_time_request")

        greeting = NegotiationEmailHTMLBuilder._build_greeting(
            contact_name=contact_name,
            supplier_name=supplier_name,
            round_number=round_number,
        )

        header = NegotiationEmailHTMLBuilder._build_header(
            round_number=round_number,
            strategy=strategy,
        )

        pricing_section = ""
        if counter_price is not None and positions:
            pricing_section = NegotiationEmailHTMLBuilder._build_pricing_section(
                counter_price=counter_price,
                positions=positions,
                currency=currency,
            )

        message_section = NegotiationEmailHTMLBuilder._build_message_section(
            negotiation_message=negotiation_message,
            round_number=round_number,
        )

        asks_section = ""
        if asks or lead_time_request:
            asks_section = NegotiationEmailHTMLBuilder._build_asks_section(
                asks=asks,
                lead_time_request=lead_time_request,
            )

        playbook_section = ""
        if playbook_recommendations:
            playbook_section = NegotiationEmailHTMLBuilder._build_playbook_section(
                recommendations=playbook_recommendations[:3],
            )

        footer = NegotiationEmailHTMLBuilder._build_footer(
            sender_name=sender_name,
            company_name=company_name,
            round_number=round_number,
        )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Negotiation Round {round_number}</title>
    {NegotiationEmailHTMLBuilder.BASE_STYLES}
</head>
<body>
    <div class="email-container">
        {header}
        {greeting}
        {message_section}
        {pricing_section}
        {asks_section}
        {playbook_section}
        {footer}
    </div>
</body>
</html>
        """

        return html

    @staticmethod
    def _build_greeting(
        *,
        contact_name: Optional[str],
        supplier_name: Optional[str],
        round_number: int,
    ) -> str:
        name = contact_name or supplier_name or "there"

        if round_number == 1:
            greeting_text = f"Dear {name},"
        elif round_number == 2:
            greeting_text = f"Dear {name},<br><br>Thank you for your response."
        else:
            greeting_text = (
                f"Dear {name},<br><br>Thank you for continuing our discussion."
            )

        return f'<div class="greeting">{greeting_text}</div>'

    @staticmethod
    def _build_header(*, round_number: int, strategy: str) -> str:
        strategy_titles = {
            "counter": "Negotiation Proposal",
            "accept": "Agreement Confirmation",
            "decline": "Negotiation Status",
            "clarify": "Request for Information",
            "review": "Review Required",
        }

        title = strategy_titles.get(strategy, "Negotiation Update")

        return f"""
        <div class="header">
            <h1>{title}</h1>
            <span class="round-badge">Round {round_number}</span>
        </div>
        """

    @staticmethod
    def _build_pricing_section(
        *,
        counter_price: float,
        positions: Dict[str, Any],
        currency: Optional[str],
    ) -> str:
        def format_price(value: Optional[float]) -> str:
            if value is None:
                return "—"
            symbol = (
                "£"
                if currency == "GBP"
                else "$"
                if currency == "USD"
                else "€"
                if currency == "EUR"
                else ""
            )
            return f"{symbol}{value:,.2f}"

        current_offer = positions.get("supplier_offer")
        target = positions.get("desired")

        rows = []

        if current_offer is not None:
            rows.append(
                f"""
            <tr>
                <td class="label">Your Current Offer</td>
                <td class="value">{format_price(current_offer)}</td>
            </tr>
            """
            )

        rows.append(
            f"""
        <tr class="highlight">
            <td class="label">Our Counter Proposal</td>
            <td class="value">{format_price(counter_price)}</td>
        </tr>
        """
        )

        if target is not None:
            rows.append(
                f"""
            <tr>
                <td class="label">Target Price</td>
                <td class="value">{format_price(target)}</td>
            </tr>
            """
            )

        if current_offer is not None and counter_price is not None:
            gap = current_offer - counter_price
            gap_pct = (gap / current_offer) * 100 if current_offer else 0
            rows.append(
                f"""
            <tr>
                <td class="label">Price Adjustment</td>
                <td class="value">{format_price(gap)} ({gap_pct:.1f}%)</td>
            </tr>
            """
            )

        return f"""
        <div class="section">
            <div class="section-title">💰 Pricing Proposal</div>
            <table class="pricing-table">
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    @staticmethod
    def _build_message_section(
        *,
        negotiation_message: str,
        round_number: int,
    ) -> str:
        paragraphs = negotiation_message.split("\n\n")
        html_paragraphs = [
            f"<p>{escape(p.strip())}</p>" for p in paragraphs if p.strip()
        ]

        callout_type = "callout-info" if round_number <= 2 else "callout-warning"

        return f"""
        <div class="section">
            <div class="section-title">📋 Proposal Details</div>
            <div class="callout {callout_type}">
                {''.join(html_paragraphs)}
            </div>
        </div>
        """

    @staticmethod
    def _build_asks_section(
        *,
        asks: List[str],
        lead_time_request: Optional[str],
    ) -> str:
        items: List[str] = []

        if lead_time_request:
            items.append(
                f"<li><strong>Lead Time:</strong> {escape(lead_time_request)}</li>"
            )

        for ask in asks:
            if ask and isinstance(ask, str):
                items.append(f"<li>{escape(ask)}</li>")

        if not items:
            return ""

        return f"""
        <div class="section">
            <div class="section-title">✓ Key Requirements</div>
            <ul class="asks-list">
                {''.join(items)}
            </ul>
        </div>
        """

    @staticmethod
    def _build_playbook_section(
        *,
        recommendations: List[Dict[str, Any]],
    ) -> str:
        if not recommendations:
            return ""

        items: List[str] = []
        for rec in recommendations[:3]:
            if not isinstance(rec, dict):
                continue

            lever = escape(str(rec.get("lever", "")))
            description = escape(str(rec.get("play", "")))

            if lever and description:
                items.append(
                    f"""
                <div class="recommendation-item">
                    <div class="lever">{lever}</div>
                    <div class="description">{description}</div>
                </div>
                """
                )

        if not items:
            return ""

        return f"""
        <div class="section">
            <div class="section-title">💡 Value Creation Opportunities</div>
            <div class="playbook-recommendations">
                {''.join(items)}
            </div>
        </div>
        """

    @staticmethod
    def _build_footer(
        *,
        sender_name: Optional[str],
        company_name: str,
        round_number: int,
    ) -> str:
        sender = sender_name or "The Procurement Team"

        if round_number >= 3:
            next_steps = """
            <div class="callout callout-warning">
                <strong>⏰ Closing Round:</strong> We're working to finalize this agreement. \
                Please respond at your earliest convenience to keep the process moving forward.
            </div>
            """
        else:
            next_steps = """
            <div class="callout callout-info">
                <strong>Next Steps:</strong> We look forward to your response and are happy to \
                discuss any aspects of this proposal in more detail.
            </div>
            """

        return f"""
        {next_steps}

        <div class="signature">
            <p>Best regards,<br>
            <span class="name">{escape(sender)}</span><br>
            {escape(company_name)}</p>
        </div>

        <div class="footer">
            <p style="font-size: 12px; color: #6c757d;">
                This is an automated negotiation communication generated as part of our procurement workflow.
                For questions or concerns, please reply directly to this email.
            </p>
        </div>
        """


@dataclass
class NegotiationPositions:
    start: Optional[float]
    desired: Optional[float]
    no_deal: Optional[float]
    supplier_offer: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def serialise(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "desired": self.desired,
            "no_deal": self.no_deal,
            "supplier_offer": self.supplier_offer,
            "history": list(self.history),
        }

    def snapshot_for_next_round(
        self, counter_price: Optional[float], round_no: int
    ) -> Dict[str, Any]:
        history = list(self.history)

        def _append(entry_type: str, value: Optional[float]) -> None:
            if value is None:
                return
            record = {
                "round": round_no,
                "type": entry_type,
                "value": value,
            }
            if not any(
                existing.get("round") == record["round"]
                and existing.get("type") == record["type"]
                and self._is_close(existing.get("value"), record["value"])
                for existing in history
            ):
                history.append(record)

        _append("supplier_offer", self.supplier_offer)
        _append("counter", counter_price)

        next_start = counter_price if counter_price is not None else self.start

        return {
            "start": next_start,
            "desired": self.desired,
            "no_deal": self.no_deal,
            "supplier_offer": self.supplier_offer,
            "history": history,
            "last_counter": counter_price if counter_price is not None else next_start,
        }

    @staticmethod
    def _is_close(value_a: Any, value_b: Any, *, tolerance: float = 1e-6) -> bool:
        try:
            return abs(float(value_a) - float(value_b)) <= tolerance
        except (TypeError, ValueError):
            return False


def compute_decision(
    payload: Dict[str, Any],
    supplier_message_text: str = "",
    offer_prev: Optional[float] = None,
) -> Dict[str, Any]:
    ask_disc_raw = payload.get("ask_early_pay_disc", 0.02)
    try:
        ask_disc = float(ask_disc_raw) if ask_disc_raw is not None else None
    except (TypeError, ValueError):
        ask_disc = None

    walkaway_raw = payload.get("walkaway_price")
    try:
        walkaway = float(walkaway_raw) if walkaway_raw is not None else None
    except (TypeError, ValueError):
        walkaway = None

    current_offer = float(payload["current_offer"])
    target_price = float(payload["target_price"])
    round_idx = int(payload.get("round", 1))

    ctx = NegotiationContext(
        current_offer=current_offer,
        target_price=target_price,
        round_index=round_idx,
        currency=payload.get("currency"),
        aggressiveness=0.75,
        leverage=0.6,
        urgency=0.3,
        risk_buffer_pct=0.06,
        min_abs_buffer=3.0,
        step_pct_of_gap=0.12,
        min_abs_step=4.0,
        max_rounds=int(payload.get("max_rounds", 3)),
        walkaway_price=walkaway,
        ask_early_pay_disc=ask_disc,
        ask_lead_time_keep=bool(payload.get("ask_lead_time_keep", True)),
    )
    signals = SupplierSignals(
        offer_prev=offer_prev,
        offer_new=current_offer,
        message_text=supplier_message_text or "",
    )
    return plan_counter(ctx, signals)


def decide_strategy(
    payload: Dict[str, Any],
    *,
    lead_weeks: Optional[float] = None,
    constraints: Optional[List[str]] = None,
    supplier_message: Optional[str] = None,
    offer_prev: Optional[float] = None,
) -> Dict[str, Any]:
    """Plan price counter proposals using deterministic negotiation heuristics."""

    current_offer = payload.get("current_offer")
    target = payload.get("target_price")
    if current_offer is None or target is None:
        return {
            "strategy": "clarify",
            "counter_price": None,
            "asks": ["Confirm unit price, currency, tiered price @ 100/250/500."],
            "lead_time_request": None,
            "rationale": "Price missing; request structured quote.",
        }

    plan = compute_decision(payload, supplier_message_text=supplier_message or "", offer_prev=offer_prev)

    decision: Dict[str, Any] = {
        "strategy": plan.get("decision", "counter"),
        "counter_price": plan.get("counter_price"),
        "asks": plan.get("asks", []),
        "lead_time_request": plan.get("lead_time_request"),
        "rationale": plan.get("message", ""),
        "decision_origin": "plan_counter",
        "price_plan_locked": True,
    }

    if lead_weeks and lead_weeks > 3:
        lead_req = plan.get("lead_time_request") or "≤ 2 weeks or split shipment (20–30% now, balance later)"
        decision["lead_time_request"] = lead_req
        if plan.get("asks") is None:
            decision["asks"] = []
        decision["asks"] = list(dict.fromkeys((decision.get("asks") or []) + ["Split shipment or expedite option"]))

    if constraints:
        decision.setdefault("constraints", constraints)

    decision.setdefault("decision_log", plan.get("log", []))
    return decision


class NegotiationAgent(BaseAgent):
    """Generate deterministic negotiation decisions and coordinate counter drafting."""

    AGENTIC_PLAN_STEPS = (
        "Analyse supplier offers, historical spend, and negotiation constraints for the active RFQ session.",
        "Compute pricing and terms strategy including counter targets, concessions, and playbook actions.",
        "Coordinate email drafting and state persistence to deliver the counter proposal and next steps.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self._state_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._email_agent: Optional[EmailDraftingAgent] = None
        self._supplier_agent: Optional["SupplierInteractionAgent"] = None
        self._state_schema_checked = False
        self._sessions_schema_checked = False
        self._playbook_cache: Optional[Dict[str, Any]] = None
        self._state_lock = threading.RLock()
        self._email_agent_lock = threading.Lock()
        self._supplier_agent_lock = threading.Lock()
        self._negotiation_session_state_identifier_column: Optional[str] = None
        self._negotiation_sessions_identifier_column: Optional[str] = None
        self.response_matcher = ResponseMatcher(getattr(self.agent_nick, "get_db_connection", None))
        self._email_thread_manager = EmailThreadManager()

        # Feature flag for enhanced negotiation message composition
        self._use_enhanced_messages = (
            os.getenv("NEG_USE_ENHANCED_MESSAGES", "false").lower() == "true"
        )

        self._email_thread_manager = EmailThreadManager()
        self._html_builder = NegotiationEmailHTMLBuilder()
        self._redis_client = get_workflow_redis_client()

    def _session_key(self, workflow_id: str) -> str:
        return f"negotiation_session:{workflow_id}"

    def _load_session_state(self, workflow_id: str, max_rounds: int) -> NegotiationSession:
        session = NegotiationSession(session_id=workflow_id, max_rounds=max_rounds)
        if not self._redis_client:
            return session
        try:
            raw = self._redis_client.get(self._session_key(workflow_id))
        except Exception:
            logger.debug("Failed to load negotiation session from Redis", exc_info=True)
            return session
        if not raw:
            return session
        try:
            payload = json.loads(raw)
            loaded = NegotiationSession.from_dict(payload)
            loaded.max_rounds = max_rounds
            if not loaded.session_id:
                loaded.session_id = workflow_id
            return loaded
        except Exception:
            logger.exception("Unable to deserialize negotiation session state")
            return session

    def _save_session_state(self, workflow_id: str, session: NegotiationSession) -> None:
        if not self._redis_client:
            return
        try:
            self._redis_client.set(
                self._session_key(workflow_id),
                json.dumps(session.to_dict()),
            )
        except Exception:
            logger.debug("Failed to persist negotiation session to Redis", exc_info=True)

    def _clear_session_state(self, workflow_id: str) -> None:
        if not self._redis_client:
            return
        try:
            self._redis_client.delete(self._session_key(workflow_id))
        except Exception:
            logger.debug("Failed to clear negotiation session state", exc_info=True)

    def _log_round_event(
        self,
        *,
        workflow_id: Optional[str],
        round_number: Optional[int],
        supplier_id: Optional[str],
        status: str,
        **extra: Any,
    ) -> None:
        payload: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "round": round_number,
            "supplier_id": supplier_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
            elif isinstance(value, (list, tuple)):
                payload[key] = [
                    item
                    if isinstance(item, (str, int, float, bool)) or item is None
                    else str(item)
                    for item in value
                ]
            else:
                payload[key] = str(value)
        try:
            message = json.dumps(payload)
        except TypeError:
            sanitised = {key: str(value) for key, value in payload.items()}
            message = json.dumps(sanitised)
        logger.info("NEGOTIATION_ROUND_EVENT %s", message)

    @contextmanager
    def _session_lock(self, workflow_id: str, supplier_id: str, round_no: int):
        """Cross-process advisory lock to avoid duplicate negotiation runs."""

        key_parts = [workflow_id, supplier_id, str(round_no or 1)]
        token = ":".join(part or "" for part in key_parts)
        lock_id = binascii.crc32(token.encode("utf-8", "ignore")) & 0x7FFFFFFF

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            yield True
            return

        conn_manager = get_conn()
        owns_manager = hasattr(conn_manager, "__enter__") and hasattr(conn_manager, "__exit__")
        connection = None
        acquired = False
        try:
            connection = conn_manager.__enter__() if owns_manager else conn_manager
            if connection is None:
                yield True
                return

            with connection.cursor() as cur:
                try:
                    cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
                    row = cur.fetchone()
                    if row is None or row[0] is None:
                        acquired = True
                    else:
                        acquired = bool(row[0])
                except Exception:
                    logger.debug(
                        "Advisory lock attempt failed; continuing without distributed lock",
                        exc_info=True,
                    )
                    acquired = True
            connection.commit()

            if not acquired:
                yield False
                return

            try:
                yield True
            finally:
                try:
                    with connection.cursor() as cur:
                        cur.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                    connection.commit()
                except Exception:  # pragma: no cover - unlock best effort
                    logger.debug(
                        "Failed to release advisory lock for workflow=%s supplier=%s round=%s",
                        workflow_id,
                        supplier_id,
                        round_no,
                        exc_info=True,
                    )
        finally:
            if connection is not None:
                if owns_manager:
                    conn_manager.__exit__(None, None, None)
                else:
                    try:
                        connection.close()
                    except Exception:
                        pass

    def _resolve_session_id(self, context: AgentContext) -> Optional[str]:
        """Derive the canonical negotiation session identifier."""

        if not context:
            return None

        for key in ("workflow_id", "session_id", "process_id"):
            candidate = getattr(context, key, None)
            text = self._coerce_text(candidate)
            if text:
                return text

        payload = context.input_data if isinstance(context.input_data, dict) else {}
        for key in ("workflow_id", "session_id", "process_id"):
            candidate = payload.get(key)
            text = self._coerce_text(candidate)
            if text:
                return text

        responses = payload.get("supplier_responses")
        if isinstance(responses, list):
            for entry in responses:
                if not isinstance(entry, dict):
                    continue
                uid = self._coerce_text(entry.get("unique_id"))
                if uid:
                    return uid

        return None

    def _resolve_session_reference(self, context: AgentContext) -> Optional[str]:
        """Resolve a display/reference token for downstream compatibility."""

        payload = context.input_data if isinstance(context.input_data, dict) else {}
        responses = payload.get("supplier_responses")
        if isinstance(responses, list):
            for entry in responses:
                if not isinstance(entry, dict):
                    continue
                for key in ("session_reference", "unique_id", "workflow_id", "session_id", "process_id"):
                    candidate = self._coerce_text(entry.get(key))
                    if candidate:
                        return candidate

        for key in ("session_reference", "unique_id"):
            candidate = self._coerce_text(payload.get(key))
            if candidate:
                return candidate

        return self._coerce_text(getattr(context, "workflow_id", None))

    def _resolve_negotiation_identifier(self, context: AgentContext) -> NegotiationIdentifier:
        payload = context.input_data if isinstance(context.input_data, dict) else {}

        workflow_id = self._coerce_text(getattr(context, "workflow_id", None))
        if not workflow_id:
            raise ValueError(
                "NegotiationAgent requires workflow_id on context; none was provided"
            )

        payload_workflow_id = self._coerce_text(payload.get("workflow_id")) if isinstance(payload, dict) else None
        if payload_workflow_id and payload_workflow_id != workflow_id:
            logger.warning(
                "Workflow ID mismatch between context (%s) and payload (%s); using context value",
                workflow_id,
                payload_workflow_id,
            )

        supplier_id: Optional[str] = None
        for key in ("supplier_id", "supplier", "supplier_name"):
            candidate = payload.get(key)
            text = self._coerce_text(candidate)
            if text:
                supplier_id = text
                break
        if not supplier_id:
            responses = payload.get("supplier_responses")
            if isinstance(responses, list):
                for entry in responses:
                    if not isinstance(entry, dict):
                        continue
                    candidate = self._coerce_text(
                        entry.get("supplier_id") or entry.get("supplier") or entry.get("supplier_name")
                    )
                    if candidate:
                        supplier_id = candidate
                        break
        if not supplier_id:
            supplier_id = f"SUP-{uuid.uuid4().hex[:10].upper()}"

        session_reference: Optional[str] = None
        for key in ("session_reference", "unique_id", "conversation_id", "thread_id"):
            candidate = payload.get(key)
            text = self._coerce_text(candidate)
            if text:
                session_reference = text
                break
        if not session_reference:
            session_reference = f"{workflow_id}-{supplier_id}"

        round_number_raw = payload.get("round")
        try:
            round_number = int(round_number_raw) if round_number_raw is not None else 1
        except Exception:
            round_number = 1

        identifier = NegotiationIdentifier(
            workflow_id=workflow_id,
            session_reference=session_reference,
            supplier_id=supplier_id,
            round_number=round_number,
        )

        if isinstance(payload, dict):
            payload.setdefault("workflow_id", identifier.workflow_id)
            payload.setdefault("session_reference", identifier.session_reference)
            payload.setdefault("unique_id", identifier.session_reference)
            payload.setdefault("supplier_id", identifier.supplier_id)
            payload.setdefault("supplier", identifier.supplier_id)
            payload.setdefault("round", identifier.round_number)

        return identifier

    def execute(self, context: AgentContext) -> AgentOutput:
        """Execute the negotiation agent and persist consolidated email drafts."""

        result = super().execute(context)
        try:
            self._flush_email_draft_actions(context, result)
        except Exception:  # pragma: no cover - defensive logging of email actions
            logger.exception("Failed to persist negotiation email draft actions")
        finally:
            if hasattr(context, "_pending_email_actions"):
                delattr(context, "_pending_email_actions")
            if hasattr(context, "_pending_email_round_tasks"):
                delattr(context, "_pending_email_round_tasks")
            if hasattr(context, "_pending_email_round_lock"):
                delattr(context, "_pending_email_round_lock")
        return result

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("NegotiationAgent starting")

        batch_entries, shared_context = self._extract_batch_inputs(context.input_data)

        if batch_entries:
            # Determine if multi-round orchestration is enabled
            input_payload = context.input_data if isinstance(context.input_data, dict) else {}
            multi_round_enabled = bool(
                shared_context.get("multi_round_enabled")
                or input_payload.get("multi_round_enabled")
                or getattr(self.agent_nick.settings, "negotiation_multi_round_enabled", False)
            )

            if multi_round_enabled:
                max_rounds = (
                    shared_context.get("max_rounds")
                    or input_payload.get("max_rounds")
                    or getattr(self.agent_nick.settings, "negotiation_max_rounds", 3)
                    or 3
                )
                try:
                    max_rounds = int(max_rounds)
                except Exception:
                    logger.warning("Unable to coerce max_rounds=%s; defaulting to 3", max_rounds)
                    max_rounds = 3

                return self._run_multi_round_negotiation(
                    context,
                    batch_entries,
                    shared_context,
                    max_rounds=max(1, max_rounds),
                )

            return self._run_batch_negotiations(context, batch_entries, shared_context)

        return self._run_single_negotiation(context)

    def _extract_batch_inputs(
        self, payload: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not isinstance(payload, dict):
            return [], {}

        responses = payload.get("supplier_responses")
        if isinstance(responses, list):
            response_batch = [entry for entry in responses if isinstance(entry, dict)]
        else:
            response_batch = []

        if response_batch:
            shared: Dict[str, Any] = {"negotiation_batch": True}
            for key in BATCH_SHARED_KEYS:
                candidate = payload.get(key)
                if isinstance(candidate, dict):
                    shared.update(candidate)
            return response_batch, shared

        batch: List[Dict[str, Any]] = []
        for key in BATCH_INPUT_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                batch = [entry for entry in value if isinstance(entry, dict)]
                if batch:
                    break

        if not batch:
            return [], {}

        shared: Dict[str, Any] = {}
        for key in BATCH_SHARED_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                shared.update(candidate)

        return batch, shared

    def _coerce_batch_defaults(
        self, base_payload: Dict[str, Any], shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        if isinstance(base_payload, dict):
            for key, value in base_payload.items():
                if key in BATCH_EXCLUDE_KEYS:
                    continue
                defaults[key] = value
        if isinstance(shared_context, dict):
            defaults.update(shared_context)
        return defaults

    def _prepare_batch_payload(
        self, defaults: Dict[str, Any], entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if isinstance(defaults, dict):
            payload.update(defaults)
        payload.update(entry)
        for key in BATCH_EXCLUDE_KEYS:
            payload.pop(key, None)
        supplier_id = payload.get("supplier_id") or payload.get("supplier")
        if supplier_id is not None:
            payload.setdefault("supplier_id", supplier_id)
            payload.setdefault("supplier", supplier_id)
        payload["_batch_execution"] = True
        return payload

    def _bucket_entries_by_round(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group negotiation batch entries by round to enforce sequential execution."""

        buckets: Dict[Tuple[Optional[int], Optional[Any]], Dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_round = None
            for key in ("round", "round_number", "round_no"):
                candidate = entry.get(key)
                if candidate is not None:
                    raw_round = candidate
                    break

            numeric_sort: Optional[int] = None
            display_round: Optional[Any] = None
            if raw_round is not None:
                try:
                    numeric_sort = int(float(raw_round))
                    display_round = numeric_sort
                except Exception:
                    display_round = raw_round
            else:
                numeric_sort = 1

            bucket_key = (numeric_sort, display_round)
            bucket = buckets.get(bucket_key)
            if not bucket:
                bucket = {
                    "round": display_round,
                    "numeric_sort": numeric_sort,
                    "entries": [],
                }
                buckets[bucket_key] = bucket
            bucket["entries"].append(entry)

        ordered = sorted(
            buckets.values(),
            key=lambda bucket: (
                0,
                bucket["numeric_sort"],
            )
            if bucket.get("numeric_sort") is not None
            else (
                1,
                str(bucket.get("round") or "").lower(),
            ),
        )
        return ordered

    def _compute_batch_workers(self, entries: Sequence[Dict[str, Any]]) -> int:
        """Determine worker pool size for the provided batch entries."""

        try:
            configured_workers = getattr(self.agent_nick.settings, "negotiation_parallel_workers", None)
            if configured_workers is not None:
                try:
                    configured_workers = int(configured_workers)
                except Exception:
                    configured_workers = None
            max_workers: Optional[int] = (
                configured_workers if configured_workers and configured_workers > 0 else None
            )
        except Exception:
            max_workers = None

        if max_workers is None:
            cpu_default = os.cpu_count() or 4
            max_workers = max(1, cpu_default)

        unique_suppliers: Set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            supplier_id = entry.get("supplier_id") or entry.get("supplier")
            if supplier_id is None:
                continue
            text = self._coerce_text(supplier_id)
            if text:
                unique_suppliers.add(text)

        desired_workers = max(1, len(unique_suppliers)) if unique_suppliers else len(entries)
        if desired_workers <= 0:
            desired_workers = len(entries) if entries else 1

        workers = max(desired_workers, max_workers)
        workers = min(workers, max(1, len(entries)))
        return max(workers, 1)

    def _resolve_batch_entry_output(
        self, context: AgentContext, payload: Dict[str, Any]
    ):
        """Execute a batch entry and capture deferred email tasks for later bundling."""

        result = self._execute_batch_entry(context, payload)
        if isinstance(result, AgentOutput) and isinstance(result.data, dict):
            pending_task = result.data.get("pending_email")
            if result.data.get("deferred_email") and isinstance(pending_task, dict):
                identifier_info = pending_task.get("identifier") or {}
                identifier = NegotiationIdentifier(
                    workflow_id=(
                        identifier_info.get("workflow_id")
                        or payload.get("workflow_id")
                        or context.workflow_id
                    ),
                    supplier_id=(
                        identifier_info.get("supplier_id")
                        or payload.get("supplier_id")
                        or payload.get("supplier")
                    ),
                    session_reference=(
                        identifier_info.get("session_reference")
                        or payload.get("session_reference")
                        or payload.get("unique_id")
                    ),
                    round_number=int(
                        identifier_info.get("round_number")
                        or payload.get("round")
                        or payload.get("round_number")
                        or 1
                    ),
                )
                self._register_pending_email_task(
                    context=context,
                    identifier=identifier,
                    task=pending_task,
                    result=result,
                )
        return result

    def _register_pending_email_task(
        self,
        *,
        context: AgentContext,
        identifier: NegotiationIdentifier,
        task: Dict[str, Any],
        result: AgentOutput,
    ) -> None:
        """Record deferred email finalisation tasks for later round bundling."""

        try:
            lock = getattr(context, "_pending_email_round_lock")
        except AttributeError:
            lock = threading.Lock()
            setattr(context, "_pending_email_round_lock", lock)

        try:
            task_map = getattr(context, "_pending_email_round_tasks")
        except AttributeError:
            task_map = defaultdict(list)
            setattr(context, "_pending_email_round_tasks", task_map)

        try:
            round_key = int(identifier.round_number)
        except Exception:
            round_key = 1

        entry = {
            "identifier": identifier,
            "task": deepcopy(task),
            "result": result,
        }

        if lock:
            with lock:
                task_map[round_key].append(entry)
        else:  # pragma: no cover - defensive fallback
            task_map[round_key].append(entry)

    def _drain_pending_email_tasks(
        self, context: AgentContext, round_number: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Retrieve and clear pending email tasks for a negotiation round."""

        try:
            task_map = getattr(context, "_pending_email_round_tasks")
        except AttributeError:
            return []

        if not task_map:
            return []

        try:
            lock = getattr(context, "_pending_email_round_lock")
        except AttributeError:
            lock = None

        try:
            round_key = int(round_number) if round_number is not None else None
        except Exception:
            round_key = None

        if round_key is None:
            return []

        if lock:
            with lock:
                return list(task_map.pop(round_key, []))
        return list(task_map.pop(round_key, []))

    def _combine_supplier_responses(
        self, responses: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Coalesce supplier responses, de-duplicating by message identifier."""

        combined: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for response in responses:
            if not isinstance(response, dict):
                continue
            response_copy = {key: value for key, value in response.items() if value is not None}
            message_id = self._coerce_text(
                response_copy.get("message_id")
                or response_copy.get("Message-ID")
                or response_copy.get("id")
                or response_copy.get("unique_id")
            )
            try:
                unique_key = message_id or json.dumps(response_copy, sort_keys=True)
            except Exception:
                unique_key = message_id or repr(sorted(response_copy.items()))
            if unique_key in seen:
                continue
            seen.add(unique_key)
            combined.append(response_copy)
        return combined

    def _ingest_batch_entry_result(
        self,
        *,
        payload: Dict[str, Any],
        result: AgentOutput,
        aggregated_results: List[Dict[str, Any]],
        drafts: List[Dict[str, Any]],
        failed_records: List[Dict[str, Any]],
        success_ids: List[str],
        next_agents: Set[str],
        supplier_history: Dict[str, Dict[str, Any]],
    ) -> None:
        supplier_id = payload.get("supplier_id") or payload.get("supplier")
        session_reference = (
            payload.get("session_reference")
            or payload.get("unique_id")
        )

        record_reference = (
            result.data.get("session_reference")
            or result.data.get("unique_id")
            or session_reference
        ) if isinstance(result.data, dict) else session_reference

        record_supplier = None
        if isinstance(result.data, dict):
            record_supplier = result.data.get("supplier")
        record = {
            "supplier_id": record_supplier
            or supplier_id
            or payload.get("supplier_id"),
            "session_reference": record_reference,
            "unique_id": record_reference,
            "status": result.status.value,
            "output": result.data if isinstance(result.data, dict) else {},
            "pass_fields": result.pass_fields if isinstance(result.pass_fields, dict) else {},
            "next_agents": list(result.next_agents),
        }
        if result.error:
            record["error"] = result.error
        if result.action_id:
            record["action_id"] = result.action_id

        aggregated_results.append(record)

        supplier_key = self._coerce_text(record.get("supplier_id"))
        if supplier_key:
            history = supplier_history.setdefault(
                supplier_key,
                {"records": [], "responses": []},
            )
            history["records"].append(record)
        else:
            history = None

        if result.status == AgentStatus.SUCCESS:
            if supplier_key:
                success_ids.append(supplier_key)
            result_drafts = (
                result.data.get("drafts")
                if isinstance(result.data, dict)
                else None
            )
            if isinstance(result_drafts, list):
                for draft in result_drafts:
                    if isinstance(draft, dict):
                        drafts.append(draft)
            next_agents.update(result.next_agents)
        else:
            failed_records.append(
                {
                    "supplier_id": record.get("supplier_id"),
                    "session_reference": record.get("session_reference"),
                    "status": record.get("status"),
                    "error": record.get("error"),
                }
            )

        if history is not None and isinstance(record.get("output"), dict):
            responses = record["output"].get("supplier_responses")
            collected = history.get("responses") or []
            if isinstance(responses, list):
                for entry in responses:
                    if isinstance(entry, dict):
                        collected.append(dict(entry))
            history["responses"] = self._combine_supplier_responses(collected)

            combined_responses = history["responses"]
            if combined_responses:
                record_output = dict(record["output"])
                record_output["supplier_responses"] = combined_responses
                draft_payload = record_output.get("draft_payload")
                if isinstance(draft_payload, dict):
                    payload_copy = dict(draft_payload)
                    payload_copy["supplier_responses"] = combined_responses
                    record_output["draft_payload"] = payload_copy
                record["output"] = record_output

                pass_fields = record.get("pass_fields")
                if isinstance(pass_fields, dict):
                    pass_fields = dict(pass_fields)
                else:
                    pass_fields = {}
                pass_fields["supplier_responses"] = combined_responses
                record["pass_fields"] = pass_fields

    def _process_round_entries(
        self,
        *,
        context: AgentContext,
        entries: Sequence[Dict[str, Any]],
        aggregated_results: List[Dict[str, Any]],
        drafts: List[Dict[str, Any]],
        draft_bundles: List[Dict[str, Any]],
        failed_records: List[Dict[str, Any]],
        success_ids: List[str],
        next_agents: Set[str],
        supplier_history: Dict[str, Dict[str, Any]],
    ) -> None:
        if not entries:
            return

        if len(entries) == 1:
            payload = entries[0]
            try:
                result = self._resolve_batch_entry_output(context, payload)
            except Exception as exc:  # pragma: no cover - defensive
                supplier_id = payload.get("supplier_id") or payload.get("supplier")
                session_reference = (
                    payload.get("session_reference")
                    or payload.get("unique_id")
                )
                record = {
                    "supplier_id": supplier_id,
                    "session_reference": session_reference,
                    "unique_id": session_reference,
                    "status": AgentStatus.FAILED.value,
                    "error": str(exc),
                }
                aggregated_results.append(record)
                failed_records.append(record)
                return

            self._ingest_batch_entry_result(
                payload=payload,
                result=result,
                aggregated_results=aggregated_results,
                drafts=drafts,
                failed_records=failed_records,
                success_ids=success_ids,
                next_agents=next_agents,
                supplier_history=supplier_history,
            )
            self._queue_round_action_for_immediate_drafts(
                context=context,
                payload=payload,
                result=result,
            )
            round_number = entries[0].get("round") if entries else None
            self._finalize_round_email_bundle(
                context=context,
                round_number=round_number,
                drafts=drafts,
                draft_bundles=draft_bundles,
            )
            return

        max_workers = self._compute_batch_workers(entries)

        futures: List[Tuple[Dict[str, Any], Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for payload in entries:
                futures.append(
                    (payload, executor.submit(self._resolve_batch_entry_output, context, payload))
                )

            for payload, future in futures:
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive aggregation
                    supplier_id = payload.get("supplier_id") or payload.get("supplier")
                    session_reference = (
                        payload.get("session_reference")
                        or payload.get("unique_id")
                    )
                    record = {
                        "supplier_id": supplier_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "status": AgentStatus.FAILED.value,
                        "error": str(exc),
                    }
                    aggregated_results.append(record)
                    failed_records.append(record)
                    continue

                self._ingest_batch_entry_result(
                    payload=payload,
                    result=result,
                    aggregated_results=aggregated_results,
                    drafts=drafts,
                    failed_records=failed_records,
                    success_ids=success_ids,
                    next_agents=next_agents,
                    supplier_history=supplier_history,
                )
                self._queue_round_action_for_immediate_drafts(
                    context=context,
                    payload=payload,
                    result=result,
                )

        round_number = None
        for entry in entries:
            round_value = entry.get("round")
            if round_value is not None:
                round_number = round_value
                break

        self._finalize_round_email_bundle(
            context=context,
            round_number=round_number,
            drafts=drafts,
            draft_bundles=draft_bundles,
        )

    def _execute_batch_entry(
        self, parent_context: AgentContext, payload: Dict[str, Any]
    ) -> AgentOutput:
        workflow_id = payload.get("workflow_id") or parent_context.workflow_id or "negotiation-batch"
        user_id = payload.get("user_id") or parent_context.user_id or "system"
        sub_context = AgentContext(
            workflow_id=str(workflow_id),
            agent_id=parent_context.agent_id,
            user_id=str(user_id),
            input_data=dict(payload),
            parent_agent=parent_context.agent_id,
            routing_history=list(parent_context.routing_history),
        )
        return self.execute(sub_context)

    def _finalize_round_email_bundle(
        self,
        *,
        context: AgentContext,
        round_number: Optional[int],
        drafts: List[Dict[str, Any]],
        draft_bundles: List[Dict[str, Any]],
        negotiation_state: Optional[Dict[str, Any]] = None,
        require_hitl: bool = False,
    ) -> None:
        """Finalize deferred emails for a negotiation round after all suppliers complete."""

        if round_number is None:
            return

        pending_entries = self._drain_pending_email_tasks(context, round_number)
        if not pending_entries:
            return

        if require_hitl:
            if isinstance(negotiation_state, dict):
                storage = negotiation_state.setdefault("hitl_email_tasks", {})
                if isinstance(storage, dict):
                    storage[int(round_number)] = deepcopy(pending_entries)
            return

        primary_identifier = None
        for entry in pending_entries:
            candidate = entry.get("identifier")
            if isinstance(candidate, NegotiationIdentifier):
                primary_identifier = candidate
                break

        if primary_identifier is None:
            sample_task = pending_entries[0].get("task", {}) if pending_entries else {}
            workflow_id = None
            supplier_id = None
            session_reference = None
            if isinstance(sample_task, dict):
                identifier_data = sample_task.get("identifier") or {}
                if isinstance(identifier_data, dict):
                    workflow_id = identifier_data.get("workflow_id")
                    supplier_id = identifier_data.get("supplier_id") or identifier_data.get("supplier")
                    session_reference = identifier_data.get("session_reference")
            primary_identifier = NegotiationIdentifier(
                workflow_id=workflow_id or context.workflow_id,
                supplier_id=supplier_id or "SUP-BUNDLE",
                session_reference=session_reference or f"bundle-{round_number}",
                round_number=round_number,
            )

        bundle_task = {"bundle": pending_entries, "round_no": round_number}
        bundle_output = self._finalize_email_round(context, primary_identifier, bundle_task)

        if not isinstance(bundle_output, AgentOutput):
            return

        bundle_data = bundle_output.data if isinstance(bundle_output.data, dict) else {}
        bundle_drafts = bundle_data.get("drafts") if isinstance(bundle_data, dict) else None
        if isinstance(bundle_drafts, list):
            for draft in bundle_drafts:
                if isinstance(draft, dict):
                    drafts.append(draft)

        if isinstance(bundle_data, dict) and bundle_data:
            draft_bundles.append(deepcopy(bundle_data))

    def _queue_round_action_for_immediate_drafts(
        self,
        *,
        context: AgentContext,
        payload: Dict[str, Any],
        result: AgentOutput,
    ) -> None:
        """Queue email draft actions for non-deferred negotiation outputs."""

        if not isinstance(result, AgentOutput):
            return
        if result.status != AgentStatus.SUCCESS:
            return
        data = result.data if isinstance(result.data, dict) else {}
        if not data or data.get("deferred_email") or data.get("pending_email"):
            return

        drafts_payload = data.get("drafts")
        if not isinstance(drafts_payload, list):
            return

        draft_records = [draft for draft in drafts_payload if isinstance(draft, dict)]
        if not draft_records:
            return

        supplier_id = (
            data.get("supplier")
            or payload.get("supplier_id")
            or payload.get("supplier")
        )
        supplier_name = data.get("supplier_name") or payload.get("supplier_name")
        round_number = (
            data.get("round")
            or data.get("round_no")
            or payload.get("round")
            or payload.get("round_number")
        )

        subject = None
        draft_payload = data.get("draft_payload")
        if isinstance(draft_payload, dict):
            subject = draft_payload.get("subject")

        body = data.get("negotiation_message")
        if not body and draft_records:
            primary = draft_records[0]
            body = (
                primary.get("body")
                or primary.get("text")
                or primary.get("html")
            )

        decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
        negotiation_message = data.get("negotiation_message")

        try:
            self._queue_email_draft_action(
                context,
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                round_number=round_number,
                subject=subject,
                body=body,
                drafts=draft_records,
                decision=decision,
                negotiation_message=negotiation_message,
                agentic_plan=result.agentic_plan,
                context_snapshot=self._build_email_context_snapshot(context),
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to queue immediate negotiation draft action")
            return

        if getattr(context, "process_id", None) is None:
            routing = getattr(self.agent_nick, "process_routing_service", None)
            if routing and hasattr(routing, "log_action"):
                pending: List[Dict[str, Any]] = getattr(
                    context, "_pending_email_actions", []
                )
                entry = pending.pop() if pending else None
                if entry and entry.get("process_output"):
                    try:
                        routing.log_action(
                            process_id=None,
                            agent_type="EmailDraftingAgent",
                            action_desc=entry.get("description"),
                            process_output=entry.get("process_output"),
                            status="completed",
                            run_id=None,
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception(
                            "Failed to log immediate negotiation draft action"
                        )

    def _run_batch_negotiations(
        self,
        context: AgentContext,
        batch_entries: List[Dict[str, Any]],
        shared_context: Dict[str, Any],
    ) -> AgentOutput:
        workflow_id = (
            shared_context.get("workflow_id")
            if isinstance(shared_context, dict)
            else None
        ) or getattr(context, "workflow_id", None)

        defaults = self._coerce_batch_defaults(context.input_data, shared_context)
        prepared_entries = [
            self._prepare_batch_payload(defaults, entry) for entry in batch_entries if isinstance(entry, dict)
        ]

        if not prepared_entries:
            empty_data = {
                "negotiation_batch": True,
                "results": [],
                "drafts": [],
                "successful_suppliers": [],
                "failed_suppliers": [],
            }
            if workflow_id:
                self._clear_session_state(workflow_id)
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=empty_data,
                    pass_fields={"negotiation_batch": True, "batch_results": []},
                ),
            )

        expected_total_raw = (
            shared_context.get("expected_count")
            or shared_context.get("expected_responses")
            or shared_context.get("expected_email_count")
        )
        expected_total: Optional[int] = None
        if expected_total_raw is not None:
            try:
                expected_total = int(expected_total_raw)
            except Exception:
                logger.debug(
                    "Unable to coerce expected_total=%s for workflow_id=%s", expected_total_raw, workflow_id
                )
                expected_total = None
        if expected_total is not None and expected_total > 0 and len(prepared_entries) < expected_total:
            logger.error(
                "Batch negotiation received incomplete entries for workflow=%s (got=%s, expected=%s)",
                workflow_id,
                len(prepared_entries),
                expected_total,
            )
            error_payload = {
                "negotiation_batch": True,
                "workflow_id": workflow_id,
                "expected_count": expected_total,
                "received_count": len(prepared_entries),
                "results": [],
                "error": f"Incomplete batch: received {len(prepared_entries)}/{expected_total} entries",
            }
            if workflow_id:
                self._clear_session_state(workflow_id)
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data=error_payload,
                    pass_fields=error_payload,
                    error="incomplete_batch",
                ),
            )

        if len(prepared_entries) == 1:
            try:
                return self._execute_batch_entry(context, prepared_entries[0])
            except Exception as exc:  # pragma: no cover - propagate as failure
                logger.exception("NegotiationAgent batch execution failed", exc_info=True)
                if workflow_id:
                    self._clear_session_state(workflow_id)
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={
                            "negotiation_batch": True,
                            "results": [
                                {
                                    "supplier_id": prepared_entries[0].get("supplier_id"),
                                    "session_reference": prepared_entries[0].get("session_reference")
                                    or prepared_entries[0].get("unique_id"),
                                    "status": AgentStatus.FAILED.value,
                                    "error": str(exc),
                                }
                            ],
                        },
                        error=str(exc),
                    ),
                )

        round_batches = self._bucket_entries_by_round(prepared_entries)

        aggregated_results: List[Dict[str, Any]] = []
        drafts: List[Dict[str, Any]] = []
        draft_bundles: List[Dict[str, Any]] = []
        failed_records: List[Dict[str, Any]] = []
        success_ids: List[str] = []
        supplier_history: Dict[str, Dict[str, Any]] = {}
        next_agents: Set[str] = set()

        for batch in round_batches:
            entries = batch.get("entries") or []
            if not entries:
                continue
            self._process_round_entries(
                context=context,
                entries=entries,
                aggregated_results=aggregated_results,
                drafts=drafts,
                draft_bundles=draft_bundles,
                failed_records=failed_records,
                success_ids=success_ids,
                next_agents=next_agents,
                supplier_history=supplier_history,
            )

        if aggregated_results:
            deduped_results: List[Dict[str, Any]] = []
            seen_keys: Set[Tuple[Optional[str], Optional[str], Optional[Any]]] = set()
            for record in aggregated_results:
                supplier_key = self._coerce_text(record.get("supplier_id"))
                session_reference = self._coerce_text(record.get("session_reference"))
                output_payload = record.get("output") or {}
                round_marker = None
                if isinstance(output_payload, dict):
                    round_marker = output_payload.get("round")
                key = (supplier_key, session_reference, round_marker)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped_results.append(record)
            aggregated_results = deduped_results

        supplier_map: Dict[str, Dict[str, Any]] = {}
        for supplier_key, history in supplier_history.items():
            records = history.get("records") or []
            if not records:
                continue
            supplier_map[supplier_key] = records[-1]

        success_ids = list(dict.fromkeys(success_ids))

        any_success = any(record["status"] == AgentStatus.SUCCESS.value for record in aggregated_results)
        any_failure = any(record["status"] != AgentStatus.SUCCESS.value for record in aggregated_results)

        round_groups: Dict[str, Dict[str, Any]] = {}
        for record in aggregated_results:
            output_payload = record.get("output") or {}
            round_value_raw = (
                output_payload.get("round")
                or output_payload.get("round_number")
                or output_payload.get("round_no")
            )
            numeric_round: Optional[int] = None
            if round_value_raw is not None:
                try:
                    numeric_round = int(float(round_value_raw))
                except Exception:
                    numeric_round = None
            group_key = (
                str(numeric_round)
                if numeric_round is not None
                else str(round_value_raw)
                if round_value_raw is not None
                else "unknown"
            )
            if group_key not in round_groups:
                round_groups[group_key] = {
                    "round": numeric_round if numeric_round is not None else round_value_raw,
                    "suppliers": [],
                    "results": [],
                    "drafts": [],
                }
            group_entry = round_groups[group_key]
            supplier_ref = record.get("supplier_id") or output_payload.get("supplier")
            if supplier_ref:
                try:
                    supplier_text = str(supplier_ref).strip()
                except Exception:
                    supplier_text = None
                if supplier_text:
                    group_entry["suppliers"].append(supplier_text)
            group_entry["results"].append(record)
            draft_entries = output_payload.get("drafts")
            if isinstance(draft_entries, list):
                group_entry["drafts"].extend(
                    [draft for draft in draft_entries if isinstance(draft, dict)]
                )

        round_summaries: List[Dict[str, Any]] = []
        for group_entry in round_groups.values():
            seen_suppliers: Set[str] = set()
            ordered_suppliers: List[str] = []
            for supplier_ref in group_entry["suppliers"]:
                if supplier_ref not in seen_suppliers:
                    seen_suppliers.add(supplier_ref)
                    ordered_suppliers.append(supplier_ref)
            round_summaries.append(
                {
                    "round": group_entry["round"],
                    "suppliers": ordered_suppliers,
                    "results": list(group_entry["results"]),
                    "drafts": list(group_entry["drafts"]),
                }
            )

        if round_summaries:
            round_summaries.sort(
                key=lambda entry: (
                    0,
                    entry["round"],
                )
                if isinstance(entry.get("round"), int)
                else (
                    1,
                    str(entry.get("round") or ""),
                )
            )

        overall_status = AgentStatus.SUCCESS
        error_text = None
        if any_failure and not any_success:
            overall_status = AgentStatus.FAILED
            error_text = "All negotiation rounds failed"

        if success_ids:
            try:
                success_ids = list(dict.fromkeys(success_ids))
            except Exception:
                success_ids = [sid for sid in success_ids if sid]

        data = {
            "negotiation_batch": True,
            "results": aggregated_results,
            "drafts": draft_bundles,
            "draft_records": drafts,
            "successful_suppliers": success_ids,
            "failed_suppliers": failed_records,
            "results_by_supplier": supplier_map,
            "batch_size": len(aggregated_results),
        }
        if round_summaries:
            data["round_summaries"] = round_summaries

        pass_fields = {
            "negotiation_batch": True,
            "batch_results": aggregated_results,
        }
        if draft_bundles:
            pass_fields["drafts"] = draft_bundles
        if drafts:
            pass_fields["draft_records"] = drafts
        if round_summaries:
            pass_fields["round_summaries"] = deepcopy(round_summaries)

        return self._with_plan(
            context,
            AgentOutput(
                status=overall_status,
                data=data,
                pass_fields=pass_fields,
                next_agents=sorted(next_agents),
                error=error_text,
            ),
        )

    def _hitl_enforced(self) -> bool:
        """Return whether HITL checkpoints are enforced."""

        try:
            return bool(getattr(self.agent_nick.settings, "hitl_enabled", True))
        except Exception:
            return True

    def _extract_hitl_decisions(
        self,
        context: AgentContext,
        shared_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect explicit HITL decisions provided in the inbound payload."""

        decisions: Dict[str, Any] = {}

        def _merge(source: Optional[Dict[str, Any]]) -> None:
            if not isinstance(source, dict):
                return
            for key in ("hitl_decisions", "hitl_approvals", "hitl_review", "hitl"):
                value = source.get(key)
                if isinstance(value, dict):
                    for round_key, decision_value in value.items():
                        decisions[str(round_key)] = decision_value

        if isinstance(context.input_data, dict):
            _merge(context.input_data)
            nested_shared = context.input_data.get("shared_context")
            if isinstance(nested_shared, dict):
                _merge(nested_shared)
        _merge(shared_context)
        return decisions

    @staticmethod
    def _normalise_hitl_value(value: Any) -> Tuple[str, Optional[str]]:
        """Normalise arbitrary decision tokens into approved/pending/rejected."""

        reason: Optional[str] = None
        candidate = value
        if isinstance(candidate, dict):
            reason = cast(Optional[str], candidate.get("reason") or candidate.get("notes"))
            candidate = candidate.get("status") or candidate.get("decision")

        if isinstance(candidate, bool):
            return ("approved" if candidate else "rejected", reason)

        token = str(candidate).strip().lower() if candidate is not None else ""
        if token in {"approved", "approve", "ok", "okay", "yes", "true", "allow", "proceed"}:
            return "approved", reason
        if token in {"rejected", "reject", "no", "false", "deny", "denied", "blocked"}:
            return "rejected", reason
        if token in {"pending", "awaiting", "hold", "review"}:
            return "pending", reason
        return "pending", reason

    def _resolve_hitl_decision(
        self,
        context: AgentContext,
        shared_context: Dict[str, Any],
        negotiation_state: Dict[str, Any],
        round_num: int,
    ) -> Dict[str, Any]:
        """Determine the HITL decision for the given round."""

        if not self._hitl_enforced():
            return {"status": "approved", "source": "hitl_disabled"}

        decisions = negotiation_state.setdefault(
            "hitl_decisions", self._extract_hitl_decisions(context, shared_context)
        )

        raw_value: Any = None
        for key in (str(round_num), round_num):
            if key in decisions:
                raw_value = decisions[key]
                break

        auto_flag: Optional[bool] = None
        if raw_value is None:
            if isinstance(shared_context, dict):
                auto_flag = shared_context.get("hitl_auto_approve")
            if isinstance(context.input_data, dict):
                inherited = context.input_data.get("hitl_auto_approve")
                if inherited is not None:
                    auto_flag = bool(inherited)

        if raw_value is None:
            if isinstance(auto_flag, bool) and auto_flag:
                return {"status": "approved", "source": "auto_approved"}
            return {"status": "pending", "source": "awaiting_review"}

        status, reason = self._normalise_hitl_value(raw_value)
        decision_info: Dict[str, Any] = {
            "status": status,
            "source": "provided",
            "raw": raw_value,
        }
        if reason:
            decision_info["reason"] = reason
        return decision_info

    def _log_hitl_checkpoint(
        self,
        context: AgentContext,
        review_payload: Dict[str, Any],
    ) -> None:
        """Persist a HITL checkpoint for auditing purposes."""

        routing = getattr(self.agent_nick, "process_routing_service", None)
        if routing is None or not hasattr(routing, "log_action"):
            return

        try:
            serialisable = json.loads(json.dumps(review_payload, default=str))
        except Exception:
            serialisable = {key: str(value) for key, value in review_payload.items()}

        status_token = review_payload.get("status")
        if status_token == "approved":
            status_value = "completed"
        elif status_token == "rejected":
            status_value = "failed"
        else:
            status_value = "pending"

        description = (
            f"HITL review for negotiation round {review_payload.get('round')}"
        )

        try:
            routing.log_action(
                process_id=getattr(context, "process_id", None),
                agent_type=self.__class__.__name__,
                action_desc=description,
                process_output=serialisable,
                status=status_value,
                run_id=None,
            )
        except Exception:  # pragma: no cover - defensive log guard
            logger.exception("Failed to log HITL checkpoint")

    def _record_hitl_review(
        self,
        context: AgentContext,
        negotiation_state: Dict[str, Any],
        round_result: Dict[str, Any],
        decision_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach HITL review metadata to the round result."""

        round_no = int(round_result.get("round", 0) or 0)
        review_payload: Dict[str, Any] = {
            "round": round_no,
            "status": decision_info.get("status", "pending"),
            "decision_source": decision_info.get("source"),
            "reason": decision_info.get("reason"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suppliers": [],
        }

        for record in round_result.get("results", []):
            if not isinstance(record, dict):
                continue
            decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
            review_payload["suppliers"].append(
                {
                    "supplier_id": record.get("supplier_id"),
                    "status": record.get("status"),
                    "strategy": decision.get("strategy"),
                    "counter_price": decision.get("counter_price"),
                    "draft_count": len(record.get("drafts") or []),
                }
            )

        negotiation_state.setdefault("hitl_reviews", []).append(review_payload)
        if review_payload["status"] != "approved":
            pending_rounds = negotiation_state.setdefault("hitl_pending_rounds", set())
            pending_rounds.add(round_no)

        round_result["hitl_review"] = review_payload
        self._log_hitl_checkpoint(context, review_payload)
        return review_payload

    def _compile_final_quotes(self, negotiation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile supplier offers and counters for downstream evaluation."""

        quotes: List[Dict[str, Any]] = []
        active_suppliers = negotiation_state.get("active_suppliers", {})
        if not isinstance(active_suppliers, dict):
            return quotes

        for supplier_id, supplier_state in active_suppliers.items():
            if not isinstance(supplier_state, dict):
                continue
            entry_payload = supplier_state.get("entry") or {}
            decisions = supplier_state.get("decisions") or []
            latest_decision = decisions[-1] if decisions else {}
            responses = supplier_state.get("responses") or []
            latest_response = responses[-1] if responses else {}

            currency = (
                latest_decision.get("currency")
                or entry_payload.get("currency")
                or entry_payload.get("currency_code")
            )

            quotes.append(
                {
                    "supplier_id": supplier_id,
                    "supplier_offer": entry_payload.get("current_offer")
                    or entry_payload.get("price"),
                    "counter_offer": latest_decision.get("counter_price"),
                    "strategy": latest_decision.get("strategy"),
                    "currency": currency,
                    "rounds_completed": max(0, int(supplier_state.get("current_round", 1)) - 1),
                    "latest_response": latest_response,
                }
            )

        return quotes

    def _run_multi_round_negotiation(
        self,
        context: AgentContext,
        batch_entries: List[Dict[str, Any]],
        shared_context: Dict[str, Any],
        max_rounds: int = 3,
    ) -> AgentOutput:
        """Coordinate multi-round negotiations with supplier response waiting."""

        workflow_id = shared_context.get("workflow_id") or context.workflow_id

        max_rounds = max(1, min(int(max_rounds), 3))

        shared_context = dict(shared_context or {})

        negotiation_state: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "total_suppliers": len(batch_entries),
            "completed_suppliers": set(),
            "active_suppliers": {},
            "failed_suppliers": {},
            "round_history": [],
            "current_round": 1,
            "max_rounds": max_rounds,
            "hitl_reviews": [],
            "hitl_pending_rounds": set(),
        }

        session = self._load_session_state(workflow_id, max_rounds)
        session.negotiation_parameters.update(shared_context.get("negotiation_parameters", {}))
        negotiation_state["session"] = session
        shared_context["session"] = session

        for entry in batch_entries:
            if not isinstance(entry, dict):
                continue
            supplier_id = (
                entry.get("supplier_id")
                or entry.get("supplier")
                or entry.get("supplier_name")
            )
            supplier_key = self._coerce_text(supplier_id)
            supplier_entry = dict(entry)
            if not supplier_key:
                supplier_key = f"SUP-{uuid.uuid4().hex[:10].upper()}"
                supplier_entry.setdefault("supplier_id", supplier_key)
                supplier_entry.setdefault("supplier", supplier_key)
            negotiation_state["active_suppliers"][supplier_key] = {
                "entry": supplier_entry,
                "current_round": 1,
                "responses": [],
                "decisions": [],
                "status": "PENDING",
            }
            session.register_supplier(supplier_key, supplier_entry.get("parameters"))

        all_round_results: List[Dict[str, Any]] = []

        for round_num in range(1, max_rounds + 1):
            logger.info(
                "Starting negotiation round %s/%s for workflow_id=%s",
                round_num,
                max_rounds,
                workflow_id,
            )

            decision_info = self._resolve_hitl_decision(
                context,
                shared_context,
                negotiation_state,
                round_num,
            )
            require_hitl_hold = decision_info.get("status") != "approved"

            round_entries = self._prepare_round_entries(
                negotiation_state["active_suppliers"],
                round_num,
                shared_context,
            )

            if not round_entries:
                logger.info(
                    "No active suppliers for round %s, ending negotiations", round_num
                )
                break

            round_result = self._execute_negotiation_round(
                context=context,
                round_entries=round_entries,
                round_num=round_num,
                workflow_id=workflow_id,
                negotiation_state=negotiation_state,
                require_hitl=require_hitl_hold,
            )

            session.update_round(round_num)
            round_records = round_result.get("results", [])
            if isinstance(round_records, list):
                for record in round_records:
                    if not isinstance(record, dict):
                        continue
                    supplier_key = self._coerce_text(record.get("supplier_id"))
                    if not supplier_key:
                        continue
                    session.register_supplier(supplier_key)
                    supplier_state = session.supplier_negotiations.get(supplier_key)
                    if supplier_state:
                        supplier_state.round_history.append(record)

            session.pending_responses = [
                supplier_id
                for supplier_id, info in negotiation_state["active_suppliers"].items()
                if info.get("status") not in {"ACCEPTED", "DECLINED", "FAILED"}
            ]
            self._save_session_state(workflow_id, session)

            all_round_results.append(round_result)

            review_info = self._record_hitl_review(
                context,
                negotiation_state,
                round_result,
                decision_info,
            )

            if require_hitl_hold:
                logger.info(
                    "Round %s awaiting HITL approval; pausing multi-round flow",
                    round_num,
                )
                negotiation_state["override_status"] = (
                    "HITL_REJECTED"
                    if review_info.get("status") == "rejected"
                    else "AWAITING_HITL"
                )
                break

            if round_num < max_rounds:
                responses_received, all_received = self._wait_for_round_responses(
                    context=context,
                    round_result=round_result,
                    round_num=round_num,
                    negotiation_state=negotiation_state,
                )

                if responses_received:
                    for supplier_id, responses in responses_received.items():
                        supplier_key = self._coerce_text(supplier_id)
                        if not supplier_key:
                            continue
                        session_state = session.supplier_negotiations.get(supplier_key)
                        if session_state is None:
                            session_state = SupplierNegotiationState(
                                supplier_id=supplier_key
                            )
                            session.supplier_negotiations[supplier_key] = session_state
                        session_state.received_responses.extend(responses)
                    session.received_responses = sorted(
                        set(session.received_responses)
                        .union(set(responses_received.keys()))
                    )
                    self._save_session_state(workflow_id, session)

                if not responses_received:
                    logger.warning(
                        "No responses received for round %s, stopping negotiations",
                        round_num,
                    )
                    break

                self._process_round_responses(
                    responses_received,
                    negotiation_state,
                    round_num,
                )

                if not all_received:
                    logger.warning(
                        "Incomplete supplier responses for round %s; halting progression",
                        round_num,
                    )
                    break

            remaining_active = [
                supplier_id
                for supplier_id, supplier_state in negotiation_state["active_suppliers"].items()
                if supplier_state.get("status")
                not in {"COMPLETED", "ACCEPTED", "DECLINED", "FAILED"}
            ]

            if not remaining_active:
                logger.info("All supplier negotiations completed")
                break

        else:
            if negotiation_state.get("override_status") == "AWAITING_HITL":
                negotiation_state.pop("override_status", None)

        if (
            all_round_results
            and not negotiation_state.get("hitl_pending_rounds")
        ):
            last_round_result = all_round_results[-1]
            last_round_no = last_round_result.get("round")
            if (
                isinstance(last_round_no, int)
                and last_round_no >= negotiation_state.get("current_round", 1)
                and last_round_no >= max_rounds
            ):
                responses_received, all_received = self._wait_for_round_responses(
                    context=context,
                    round_result=last_round_result,
                    round_num=last_round_no,
                    negotiation_state=negotiation_state,
                )
                if responses_received:
                    self._process_round_responses(
                        responses_received,
                        negotiation_state,
                        last_round_no,
                    )
                negotiation_state["final_round_responses"] = responses_received
                negotiation_state["final_round_all_received"] = all_received

        final_output = self._consolidate_multi_round_results(
            context=context,
            negotiation_state=negotiation_state,
            all_round_results=all_round_results,
        )

        final_output.data.setdefault("hitl_reviews", negotiation_state.get("hitl_reviews", []))
        pending_rounds = negotiation_state.get("hitl_pending_rounds", set())
        if isinstance(pending_rounds, set):
            final_output.data["hitl_pending_rounds"] = sorted(pending_rounds)
        else:
            final_output.data["hitl_pending_rounds"] = []

        override_status = negotiation_state.get("override_status")
        if override_status:
            final_output.data["final_status"] = override_status

        final_output.data["final_round_all_responses_received"] = negotiation_state.get(
            "final_round_all_received"
        )
        final_output.data["final_round_responses"] = negotiation_state.get(
            "final_round_responses", {}
        )
        final_output.data["final_quotes"] = self._compile_final_quotes(negotiation_state)
        final_output.data["ready_for_quote_evaluation"] = (
            not final_output.data.get("hitl_pending_rounds")
            and bool(all_round_results)
            and negotiation_state.get("final_round_all_received") is True
        )
        if negotiation_state.get("hitl_email_tasks"):
            try:
                final_output.data["hitl_email_tasks"] = json.loads(
                    json.dumps(negotiation_state.get("hitl_email_tasks"), default=str)
                )
            except Exception:
                final_output.data["hitl_email_tasks"] = negotiation_state.get(
                    "hitl_email_tasks"
                )

        if final_output.data.get("ready_for_quote_evaluation"):
            final_output.next_agents = sorted(
                set(final_output.next_agents or []) | {"QuoteEvaluationAgent"}
            )

        return final_output

    def _prepare_round_entries(
        self,
        active_suppliers: Dict[str, Dict[str, Any]],
        round_num: int,
        shared_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare negotiation entries for execution in the specified round."""

        round_entries: List[Dict[str, Any]] = []
        session: Optional[NegotiationSession] = None
        potential_session = shared_context.get("session") if isinstance(shared_context, dict) else None
        if isinstance(potential_session, NegotiationSession):
            session = potential_session

        for supplier_id, supplier_state in active_suppliers.items():
            if not isinstance(supplier_state, dict):
                continue
            if supplier_state.get("status") in {"COMPLETED", "ACCEPTED", "DECLINED"}:
                continue

            base_entry = dict(supplier_state.get("entry") or {})
            base_entry["round"] = round_num

            responses = supplier_state.get("responses") or []
            if responses:
                latest_response = responses[-1]
                if isinstance(latest_response, dict):
                    base_entry["supplier_response"] = latest_response
                    message_text = latest_response.get("message_text")
                    if message_text:
                        base_entry["supplier_message"] = message_text
                    base_entry["previous_offer"] = latest_response.get("price")
                    snippets = latest_response.get("snippets")
                    if isinstance(snippets, list) and snippets:
                        base_entry["supplier_snippets"] = snippets

            decisions = supplier_state.get("decisions") or []
            if decisions:
                prev_decision = decisions[-1]
                if isinstance(prev_decision, dict):
                    counter_price = prev_decision.get("counter_price")
                    if counter_price is not None:
                        base_entry["previous_counter_price"] = counter_price
                        base_entry.setdefault("agent_previous_offer", counter_price)

            for key, value in shared_context.items():
                if key not in base_entry:
                    base_entry[key] = value

            base_entry.setdefault("supplier_id", supplier_id)
            base_entry.setdefault("supplier", supplier_id)

            if session is not None:
                supplier_key = self._coerce_text(supplier_id)
                if supplier_key and supplier_key in session.supplier_negotiations:
                    session_state = session.supplier_negotiations[supplier_key]
                    metrics: Dict[str, Any] = {}
                    previous_responses = session_state.received_responses
                    latest_response = previous_responses[-1] if previous_responses else None
                    previous_price = None
                    if isinstance(latest_response, dict):
                        previous_price = latest_response.get("price") or latest_response.get("counter_price")
                    target_price = base_entry.get("target_price") or session.negotiation_parameters.get(
                        "price_target"
                    )
                    if previous_price and target_price:
                        try:
                            previous_price_val = float(previous_price)
                            target_price_val = float(target_price)
                            gap_value = previous_price_val - target_price_val
                            if previous_price_val:
                                metrics["gap_percentage"] = round(
                                    (gap_value / previous_price_val) * 100, 2
                                )
                            metrics["gap_value"] = gap_value
                        except (TypeError, ValueError):
                            metrics["gap_value"] = None
                    if metrics:
                        base_entry.setdefault("negotiation_metrics", metrics)

            round_entries.append(base_entry)

        return round_entries

    def _execute_round_entry_with_lock(
        self, context: AgentContext, entry: Dict[str, Any]
    ) -> Tuple[Optional[str], AgentOutput, float]:
        workflow_id = (
            self._coerce_text(entry.get("workflow_id"))
            or self._coerce_text(context.workflow_id)
        )
        supplier_id = self._coerce_text(
            entry.get("supplier_id") or entry.get("supplier")
        )
        if not supplier_id:
            supplier_id = f"SUP-{uuid.uuid4().hex[:10].upper()}"
            entry.setdefault("supplier_id", supplier_id)
            entry.setdefault("supplier", supplier_id)
        else:
            entry.setdefault("supplier_id", supplier_id)
            entry.setdefault("supplier", supplier_id)

        round_raw = entry.get("round") or entry.get("round_number")
        try:
            round_number = int(round_raw) if round_raw is not None else 1
        except Exception:
            round_number = 1

        start_time = time.time()

        with self._session_lock(workflow_id or "", supplier_id, round_number) as acquired:
            if not acquired:
                self._log_round_event(
                    workflow_id=workflow_id,
                    round_number=round_number,
                    supplier_id=supplier_id,
                    status="LockBusy",
                )
                failure_output = AgentOutput(
                    status=AgentStatus.FAILED,
                    data={
                        "workflow_id": workflow_id,
                        "supplier_id": supplier_id,
                        "supplier": supplier_id,
                        "round": round_number,
                        "message": "Another negotiation instance is already processing this supplier round.",
                    },
                    error="negotiation_session_locked",
                )
                return supplier_id, self._with_plan(context, failure_output), start_time

            self._log_round_event(
                workflow_id=workflow_id,
                round_number=round_number,
                supplier_id=supplier_id,
                status="Drafting",
            )
            result = self._resolve_batch_entry_output(context, entry)
            return supplier_id, result, start_time

    def _execute_negotiation_round(
        self,
        context: AgentContext,
        round_entries: List[Dict[str, Any]],
        round_num: int,
        workflow_id: str,
        negotiation_state: Dict[str, Any],
        *,
        require_hitl: bool = False,
    ) -> Dict[str, Any]:
        """Execute a negotiation round across all active suppliers."""

        logger.info(
            "Executing round %s for %s suppliers (workflow_id=%s)",
            round_num,
            len(round_entries),
            workflow_id,
        )

        aggregated_results: List[Dict[str, Any]] = []
        drafts: List[Dict[str, Any]] = []
        draft_bundles: List[Dict[str, Any]] = []
        failed_records: List[Dict[str, Any]] = []

        max_workers = self._compute_batch_workers(round_entries)

        start_times: Dict[str, float] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: List[Tuple[Dict[str, Any], Any]] = []
            for entry in round_entries:
                supplier_id = (
                    entry.get("supplier_id")
                    or entry.get("supplier")
                    or entry.get("supplier_name")
                )
                supplier_key = self._coerce_text(supplier_id)
                if supplier_key and supplier_key not in start_times:
                    start_times[supplier_key] = time.time()
                futures.append(
                    (entry, executor.submit(self._execute_round_entry_with_lock, context, entry))
                )

            for entry, future in futures:
                supplier_id = (
                    entry.get("supplier_id")
                    or entry.get("supplier")
                    or entry.get("supplier_name")
                )
                supplier_key = self._coerce_text(supplier_id)

                try:
                    future_supplier, result, lock_start = future.result()
                    if future_supplier:
                        supplier_key = self._coerce_text(future_supplier) or supplier_key
                        if supplier_key and supplier_key not in start_times:
                            start_times[supplier_key] = lock_start
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "Failed to execute negotiation for supplier %s", supplier_key
                    )
                    if supplier_key:
                        duration_ms = int(
                            (time.time() - start_times.get(supplier_key, time.time())) * 1000
                        )
                        self._log_round_event(
                            workflow_id=workflow_id,
                            round_number=round_num,
                            supplier_id=supplier_key,
                            status="DraftFailed",
                            error=str(exc),
                            duration_ms=duration_ms,
                        )
                    failed_records.append(
                        {
                            "supplier_id": supplier_key,
                            "round": round_num,
                            "status": "FAILED",
                            "error": str(exc),
                        }
                    )
                    negotiation_state["failed_suppliers"][supplier_key] = {
                        "round": round_num,
                        "error": str(exc),
                    }
                    supplier_state = negotiation_state["active_suppliers"].get(
                        supplier_key
                    )
                    if supplier_state is not None:
                        supplier_state["status"] = "FAILED"
                    continue

                result = cast(AgentOutput, result)

                if not supplier_key:
                    supplier_key = self._coerce_text(future_supplier)
                if not supplier_key:
                    supplier_key = f"SUP-{uuid.uuid4().hex[:8].upper()}"

                duration_ms = int(
                    (time.time() - start_times.get(supplier_key, lock_start)) * 1000
                )

                if isinstance(result, AgentOutput) and result.status == AgentStatus.SUCCESS:
                    result_data = result.data or {}

                    aggregated_results.append(
                        {
                            "supplier_id": supplier_key,
                            "round": round_num,
                            "status": "SUCCESS",
                            "output": result_data,
                            "decision": result_data.get("decision"),
                            "drafts": result_data.get("drafts", []),
                        }
                    )

                    decision = result_data.get("decision") or {}
                    strategy = None
                    if isinstance(decision, dict):
                        strategy = decision.get("strategy")
                    self._log_round_event(
                        workflow_id=workflow_id,
                        round_number=round_num,
                        supplier_id=supplier_key,
                        status="Drafted",
                        duration_ms=duration_ms,
                        strategy=strategy,
                    )

                    drafts_payload = result_data.get("drafts")
                    if isinstance(drafts_payload, list):
                        drafts.extend(
                            [draft for draft in drafts_payload if isinstance(draft, dict)]
                        )

                    supplier_state = negotiation_state["active_suppliers"].get(
                        supplier_key
                    )
                    if supplier_state is not None:
                        supplier_state.setdefault("decisions", []).append(
                            result_data.get("decision", {})
                        )
                        supplier_state["current_round"] = round_num

                        strategy_lower = (strategy or "").lower()
                        if strategy_lower in {"accept", "decline"}:
                            supplier_state["status"] = (
                                "ACCEPTED" if strategy_lower == "accept" else "DECLINED"
                            )
                            negotiation_state["completed_suppliers"].add(supplier_key)

                else:
                    error_text = None
                    if isinstance(result, AgentOutput):
                        error_text = result.error
                    self._log_round_event(
                        workflow_id=workflow_id,
                        round_number=round_num,
                        supplier_id=supplier_key,
                        status="DraftFailed",
                        duration_ms=duration_ms,
                        error=error_text,
                    )
                    failed_records.append(
                        {
                            "supplier_id": supplier_key,
                            "round": round_num,
                            "status": "FAILED",
                            "error": error_text,
                        }
                    )
                    negotiation_state["failed_suppliers"][supplier_key] = {
                        "round": round_num,
                        "error": error_text,
                    }
                    supplier_state = negotiation_state["active_suppliers"].get(
                        supplier_key
                    )
                    if supplier_state is not None:
                        supplier_state["status"] = "FAILED"

        self._finalize_round_email_bundle(
            context=context,
            round_number=round_num,
            drafts=drafts,
            draft_bundles=draft_bundles,
            negotiation_state=negotiation_state,
            require_hitl=require_hitl,
        )

        round_result = {
            "round": round_num,
            "workflow_id": workflow_id,
            "results": aggregated_results,
            "drafts": drafts,
            "draft_bundles": draft_bundles,
            "failed": failed_records,
            "suppliers_processed": len(round_entries),
            "suppliers_succeeded": len(
                [record for record in aggregated_results if record["status"] == "SUCCESS"]
            ),
            "suppliers_failed": len(failed_records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        negotiation_state.setdefault("round_history", []).append(round_result)
        negotiation_state["current_round"] = round_num

        if round_num >= max_rounds or len(negotiation_state["completed_suppliers"]) >= len(
            negotiation_state.get("active_suppliers", {})
        ):
            self._clear_session_state(workflow_id)
        else:
            self._save_session_state(workflow_id, session)

        return round_result

    def _load_round_supplier_responses(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[str],
        supplier_ids: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not workflow_id:
            return {}

        unique_tokens = [
            self._coerce_text(token) for token in unique_ids if self._coerce_text(token)
        ]
        supplier_tokens = [
            self._coerce_text(supplier) for supplier in supplier_ids if self._coerce_text(supplier)
        ]

        if not unique_tokens and not supplier_tokens:
            return {}

        try:
            rows = supplier_response_repo.fetch_for_unique_ids(
                workflow_id=workflow_id,
                unique_ids=unique_tokens,
                supplier_ids=supplier_tokens,
                include_processed=False,
            )
        except Exception:
            logger.exception(
                "Failed to load supplier responses from repository",
                extra={"workflow_id": workflow_id},
            )
            return {}

        mapped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            supplier_id = self._coerce_text(row.get("supplier_id"))
            if supplier_id:
                mapped[supplier_id].append(row)
        return mapped

    @staticmethod
    def _run_async_task(coro: Awaitable[Any]) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" not in str(exc):
                raise
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _record_round_status(
        self,
        context: AgentContext,
        workflow_id: Optional[str],
        status: str,
    ) -> None:
        workflow_key = self._coerce_text(workflow_id)
        if workflow_key:
            try:
                workflow_lifecycle_repo.record_negotiation_status(workflow_key, status)
            except Exception:
                logger.debug(
                    "Failed to persist negotiation status %s for workflow=%s",
                    status,
                    workflow_key,
                    exc_info=True,
                )
        routing = getattr(self.agent_nick, "process_routing_service", None)
        process_id = getattr(context, "process_id", None)
        if routing and process_id:
            try:
                routing.update_agent_status(
                    process_id,
                    self.__class__.__name__,
                    status,
                    modified_by=getattr(context, "user_id", None),
                )
            except Exception:
                logger.debug(
                    "Failed to update process routing status to %s for process_id=%s",
                    status,
                    process_id,
                    exc_info=True,
                )

    def _await_responses_with_coordinator(
        self,
        workflow_id: Optional[str],
        round_number: int,
        unique_ids: Sequence[str],
        timeout: float,
        poll_interval: float,
    ) -> Optional[Tuple[Any, Optional[RoundStatus]]]:
        workflow_key = self._coerce_text(workflow_id)
        if not workflow_key:
            return None
        identifiers = [self._coerce_text(uid) for uid in unique_ids if self._coerce_text(uid)]
        if not identifiers:
            return None

        try:
            coordinator = get_supplier_response_coordinator()
        except Exception:
            logger.exception("Failed to initialise supplier response coordinator")
            return None

        if coordinator is None:
            return None

        try:
            workflow_round_response_repo.register_expected(
                workflow_id=workflow_key,
                expectations=[(round_number, uid, None) for uid in identifiers],
            )
        except Exception:
            logger.debug(
                "Failed to register round expectations prior to coordinator wait",
                exc_info=True,
            )

        try:
            coordinator.register_expected_responses(
                workflow_key,
                identifiers,
                max(len(identifiers), 1),
                round_number=round_number,
            )
        except Exception:
            logger.debug(
                "Coordinator registration failed for workflow=%s", workflow_key, exc_info=True
            )

        loop_timeout = max(0.0, float(timeout))
        poll_delay = max(1.0, float(poll_interval))

        async def _wait_async() -> Any:
            start = time.monotonic()

            def _await_block(wait_value: float) -> Any:
                return coordinator.await_completion(
                    workflow_key,
                    wait_value,
                    round_number=round_number,
                )

            state = await asyncio.to_thread(_await_block, 0.0)
            while True:
                pending = getattr(state, "pending_unique_ids", [])
                complete = bool(getattr(state, "complete", False)) and not pending
                if complete:
                    return state
                elapsed = time.monotonic() - start
                if loop_timeout and elapsed >= loop_timeout:
                    return state
                remaining = loop_timeout - elapsed if loop_timeout else poll_delay
                wait_value = poll_delay if not loop_timeout else max(0.0, min(poll_delay, remaining))
                if wait_value <= 0:
                    return state
                state = await asyncio.to_thread(_await_block, wait_value)
                await asyncio.sleep(0)

        state = self._run_async_task(_wait_async())
        round_status = workflow_round_response_repo.get_round_status(
            workflow_id=workflow_key, round_number=round_number
        )
        return state, round_status

    def _wait_for_round_responses(
        self,
        context: AgentContext,
        round_result: Dict[str, Any],
        round_num: int,
        negotiation_state: Dict[str, Any],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], bool]:
        """Wait for supplier responses before commencing the next round."""

        drafts = round_result.get("drafts") or []
        workflow_id = round_result.get("workflow_id")

        aggregated: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        expected_suppliers: Set[str] = set()
        pending_suppliers: Set[str] = set()
        responded_suppliers: Set[str] = set()
        timed_out_suppliers: Set[str] = set()

        max_rounds_raw = (
            negotiation_state.get("max_rounds")
            or round_result.get("max_rounds")
            or self._coerce_text(negotiation_state.get("max_rounds"))
        )
        try:
            max_rounds_value = int(max_rounds_raw) if max_rounds_raw is not None else 3
        except Exception:
            max_rounds_value = 3

        session_workflow_id = (
            self._coerce_text(workflow_id)
            or self._coerce_text(negotiation_state.get("workflow_id"))
            or self._coerce_text(context.workflow_id)
        )
        session: Optional[NegotiationSession] = None
        session_candidate = negotiation_state.get("session")
        if isinstance(session_candidate, NegotiationSession):
            session = session_candidate
        elif isinstance(session_candidate, dict):
            try:
                session = NegotiationSession.from_dict(session_candidate)
                negotiation_state["session"] = session
            except Exception:
                logger.debug("Failed to hydrate negotiation session from dict", exc_info=True)
                session = None
        if session is None and session_workflow_id:
            session = self._load_session_state(session_workflow_id, max_rounds_value)
            negotiation_state["session"] = session
        if session:
            session.update_round(round_num)
            session.max_rounds = max_rounds_value

        def _persist_session_snapshot() -> None:
            if not session or not session_workflow_id:
                return
            for supplier_id in expected_suppliers:
                session.register_supplier(supplier_id)
                supplier_state = session.supplier_negotiations.get(supplier_id)
                if supplier_state is None:
                    continue
                if supplier_id in responded_suppliers:
                    supplier_state.status = "RESPONDED"
                elif supplier_id in timed_out_suppliers:
                    supplier_state.status = "TIMEOUT"
                elif supplier_id in pending_suppliers:
                    supplier_state.status = "AWAITING_RESPONSE"
                elif not supplier_state.status:
                    supplier_state.status = "PENDING"
            if responded_suppliers:
                session.received_responses = sorted(
                    set(session.received_responses).union(responded_suppliers)
                )
            session.pending_responses = sorted(pending_suppliers)
            session.negotiation_parameters.update(
                negotiation_state.get("negotiation_parameters", {})
            )
            self._save_session_state(session_workflow_id, session)

        result: Tuple[Dict[str, List[Dict[str, Any]]], bool] = ({}, False)

        try:
            if not drafts:
                logger.warning("No drafts to wait for in round %s", round_num)
                pending_suppliers.clear()
                self._record_round_status(
                    context, session_workflow_id, "responses_completed"
                )
                return result

            supplier_by_id: Dict[str, Dict[str, Any]] = {}
            round_unique_ids: Set[str] = set()
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                supplier_id = self._coerce_text(
                    draft.get("supplier_id") or draft.get("supplier")
                )
                if supplier_id:
                    expected_suppliers.add(supplier_id)
                    supplier_by_id[supplier_id] = draft
                token = self._coerce_text(
                    draft.get("unique_id") or draft.get("session_reference")
                )
                if token:
                    round_unique_ids.add(token)

            if not expected_suppliers:
                logger.warning(
                    "Unable to determine suppliers for round %s drafts; aborting wait",
                    round_num,
                )
                pending_suppliers.clear()
                self._record_round_status(
                    context, session_workflow_id, "responses_completed"
                )
                return result

            timeout = self._calculate_round_timeout(len(expected_suppliers), round_num)
            poll_interval = getattr(
                self.agent_nick.settings, "email_response_poll_seconds", 60
            )

            self._record_round_status(
                context, session_workflow_id, "waiting_for_responses"
            )

            logger.info(
                "Waiting for %s supplier responses for round %s (workflow_id=%s)",
                len(expected_suppliers),
                round_num,
                workflow_id,
            )

            self._build_round_watch_payload(
                drafts=drafts,
                workflow_id=workflow_id,
                round_num=round_num,
                negotiation_state=negotiation_state,
            )

            coordinator_outcome = self._await_responses_with_coordinator(
                session_workflow_id,
                round_number=round_num,
                unique_ids=list(round_unique_ids),
                timeout=timeout,
                poll_interval=poll_interval,
            )

            if coordinator_outcome is not None:
                state, round_status = coordinator_outcome
                pending_unique_ids = list(
                    getattr(state, "pending_unique_ids", []) or []
                )
                if pending_unique_ids and session_workflow_id:
                    try:
                        workflow_round_response_repo.mark_round_failed(
                            workflow_id=session_workflow_id,
                            round_number=round_num,
                            unique_ids=pending_unique_ids,
                            reason=getattr(state, "status", None),
                        )
                    except Exception:
                        logger.debug(
                            "Failed to flag pending responses as failed for workflow=%s",
                            session_workflow_id,
                            exc_info=True,
                        )

                repo_responses = self._load_round_supplier_responses(
                    workflow_id=workflow_id,
                    unique_ids=list(round_unique_ids),
                    supplier_ids=list(expected_suppliers),
                )
                for supplier_id, responses in repo_responses.items():
                    if not responses:
                        continue
                    aggregated[supplier_id].extend(responses)
                    responded_suppliers.add(supplier_id)
                    pending_suppliers.discard(supplier_id)

                if round_status:
                    pending_suppliers.update(round_status.pending_suppliers())
                    responded_suppliers.update(round_status.completed_suppliers())
                    timed_out_suppliers.update(round_status.failed_suppliers())

                if pending_suppliers:
                    timed_out_suppliers.update(pending_suppliers)

                if timed_out_suppliers:
                    for supplier_id in sorted(timed_out_suppliers):
                        self._log_round_event(
                            workflow_id=workflow_id,
                            round_number=round_num,
                            supplier_id=supplier_id,
                            status="ResponseTimeout",
                        )

                for supplier_id, responses in aggregated.items():
                    self._log_round_event(
                        workflow_id=workflow_id,
                        round_number=round_num,
                        supplier_id=supplier_id,
                        status="ResponseReceived",
                        response_count=len(responses),
                        source="watcher",
                    )

                _persist_session_snapshot()

                all_received_flag = bool(
                    round_status and round_status.complete and not pending_unique_ids
                )

                if all_received_flag and not pending_suppliers:
                    self._record_round_status(
                        context, session_workflow_id, "responses_completed"
                    )
                else:
                    self._record_round_status(
                        context, session_workflow_id, "round_failed"
                    )

                return dict(aggregated), all_received_flag and not pending_suppliers

            start_times: Dict[str, float] = {
                supplier_id: time.time() for supplier_id in expected_suppliers
            }

            pending_suppliers.update(expected_suppliers)
            _persist_session_snapshot()

            for supplier_id in pending_suppliers:
                self._log_round_event(
                    workflow_id=workflow_id,
                    round_number=round_num,
                    supplier_id=supplier_id,
                    status="AwaitingResponse",
                )

            preloaded = self._load_round_supplier_responses(
                workflow_id=workflow_id,
                unique_ids=list(round_unique_ids),
                supplier_ids=list(pending_suppliers),
            )
            for supplier_id, responses in preloaded.items():
                if not responses:
                    continue
                aggregated[supplier_id].extend(responses)
                responded_suppliers.add(supplier_id)
                if supplier_id in pending_suppliers:
                    pending_suppliers.discard(supplier_id)
                    elapsed_ms = int(
                        (time.time() - start_times.get(supplier_id, time.time())) * 1000
                    )
                    self._log_round_event(
                        workflow_id=workflow_id,
                        round_number=round_num,
                        supplier_id=supplier_id,
                        status="ResponseReceived",
                        duration_ms=elapsed_ms,
                        response_count=len(responses),
                        source="database",
                    )
            _persist_session_snapshot()

            if not pending_suppliers:
                received_total = sum(len(values) for values in aggregated.values())
                logger.info(
                    "Received %s supplier responses for round %s",
                    received_total,
                    round_num,
                )
                self._record_round_status(
                    context, session_workflow_id, "responses_completed"
                )
                return dict(aggregated), True

            supplier_agent = self._get_supplier_agent()
            if supplier_agent is None:
                logger.error(
                    "SupplierInteractionAgent unavailable; cannot await round %s responses",
                    round_num,
                )
                timed_out_suppliers.update(pending_suppliers)
                _persist_session_snapshot()
                self._record_round_status(
                    context, session_workflow_id, "round_failed"
                )
                return result

            timeout = self._calculate_round_timeout(len(expected_suppliers), round_num)
            poll_interval = getattr(
                self.agent_nick.settings, "email_response_poll_seconds", 60
            )

            deadline = time.time() + timeout

            logger.info(
                "Initiating response wait: timeout=%ss, poll_interval=%ss, expected_count=%s",
                timeout,
                poll_interval,
                len(expected_suppliers),
            )

            while pending_suppliers and time.time() < deadline:
                wait_drafts: List[Dict[str, Any]] = []
                for supplier_id in list(pending_suppliers):
                    draft = supplier_by_id.get(supplier_id)
                    if isinstance(draft, dict):
                        wait_drafts.append(draft)

                if not wait_drafts:
                    break

                remaining = max(1, int(deadline - time.time()))

                wait_started = time.time()
                responses = supplier_agent.wait_for_multiple_responses(
                    wait_drafts,
                    timeout=remaining,
                    poll_interval=poll_interval,
                    limit=len(wait_drafts),
                    enable_negotiation=False,
                )

                if not responses:
                    logger.debug(
                        "No supplier responses returned for round %s iteration; remaining suppliers=%s",
                        round_num,
                        sorted(pending_suppliers),
                    )
                    continue

                mapped = self._map_responses_to_suppliers(
                    [resp for resp in responses if isinstance(resp, dict)],
                    wait_drafts,
                    negotiation_state,
                )

                for supplier_id, supplier_responses in mapped.items():
                    if not supplier_responses:
                        continue
                    aggregated[supplier_id].extend(supplier_responses)
                    responded_suppliers.add(supplier_id)
                    if supplier_id in pending_suppliers:
                        pending_suppliers.discard(supplier_id)
                        elapsed_ms = int(
                            (time.time() - start_times.get(supplier_id, wait_started)) * 1000
                        )
                        self._log_round_event(
                            workflow_id=workflow_id,
                            round_number=round_num,
                            supplier_id=supplier_id,
                            status="ResponseReceived",
                            duration_ms=elapsed_ms,
                            response_count=len(supplier_responses),
                            source="watcher",
                        )
                    _persist_session_snapshot()

                if pending_suppliers and time.time() < deadline:
                    logger.info(
                        "Still awaiting %s supplier responses in round %s: %s",
                        len(pending_suppliers),
                        round_num,
                        sorted(pending_suppliers),
                    )

            if pending_suppliers:
                follow_up = self._load_round_supplier_responses(
                    workflow_id=workflow_id,
                    unique_ids=list(round_unique_ids),
                    supplier_ids=list(pending_suppliers),
                )
                for supplier_id, responses in follow_up.items():
                    if not responses:
                        continue
                    aggregated[supplier_id].extend(responses)
                    responded_suppliers.add(supplier_id)
                    if supplier_id in pending_suppliers:
                        pending_suppliers.discard(supplier_id)
                        elapsed_ms = int(
                            (time.time() - start_times.get(supplier_id, time.time())) * 1000
                        )
                        self._log_round_event(
                            workflow_id=workflow_id,
                            round_number=round_num,
                            supplier_id=supplier_id,
                            status="ResponseReceived",
                            duration_ms=elapsed_ms,
                            response_count=len(responses),
                            source="database",
                        )
                _persist_session_snapshot()

            if pending_suppliers:
                logger.warning(
                    "Timed out waiting for suppliers %s in round %s",
                    sorted(pending_suppliers),
                    round_num,
                )
                for supplier_id in sorted(pending_suppliers):
                    elapsed_ms = int(
                        (time.time() - start_times.get(supplier_id, time.time())) * 1000
                    )
                    self._log_round_event(
                        workflow_id=workflow_id,
                        round_number=round_num,
                        supplier_id=supplier_id,
                        status="ResponseTimeout",
                        duration_ms=elapsed_ms,
                    )
                timed_out_suppliers.update(pending_suppliers)
                _persist_session_snapshot()

            received_total = sum(len(responses) for responses in aggregated.values())
            logger.info(
                "Received %s supplier responses for round %s",
                received_total,
                round_num,
            )

            all_received = not pending_suppliers
            if not all_received and session_workflow_id:
                try:
                    round_status = workflow_round_response_repo.get_round_status(
                        workflow_id=session_workflow_id, round_number=round_num
                    )
                except Exception:
                    logger.debug(
                        "Failed to load round status for workflow=%s",
                        session_workflow_id,
                        exc_info=True,
                    )
                    round_status = None
                if round_status and round_status.pending_unique_ids:
                    try:
                        workflow_round_response_repo.mark_round_failed(
                            workflow_id=session_workflow_id,
                            round_number=round_num,
                            unique_ids=round_status.pending_unique_ids,
                            reason="timeout",
                        )
                    except Exception:
                        logger.debug(
                            "Failed to mark pending round responses as failed for workflow=%s",
                            session_workflow_id,
                            exc_info=True,
                        )

            self._record_round_status(
                context,
                session_workflow_id,
                "responses_completed" if all_received else "round_failed",
            )

            return dict(aggregated), all_received

        except Exception:
            logger.exception("Failed to wait for round %s responses", round_num)
            return result
        finally:
            _persist_session_snapshot()

    def _calculate_round_timeout(self, supplier_count: int, round_num: int) -> int:
        """Compute the timeout for waiting on supplier responses for a round."""

        base_timeout = getattr(
            self.agent_nick.settings, "negotiation_round_base_timeout", 900
        )
        per_supplier_time = getattr(
            self.agent_nick.settings, "negotiation_per_supplier_timeout", 300
        )
        round_multiplier = 1.0 + max(0, round_num - 1) * 0.2

        timeout = int((base_timeout + (supplier_count * per_supplier_time)) * round_multiplier)

        max_timeout = getattr(
            self.agent_nick.settings, "negotiation_max_round_timeout", 3600
        )

        return min(timeout, max_timeout)

    def _build_round_watch_payload(
        self,
        drafts: List[Dict[str, Any]],
        workflow_id: Optional[str],
        round_num: int,
        negotiation_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct a watch payload for supplier response waiting."""

        unique_ids: List[str] = []
        supplier_ids: List[str] = []

        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            unique_id = self._coerce_text(draft.get("unique_id"))
            supplier_id = self._coerce_text(
                draft.get("supplier_id") or draft.get("supplier")
            )
            if unique_id:
                unique_ids.append(unique_id)
            if supplier_id:
                supplier_ids.append(supplier_id)

        poll_interval = getattr(
            self.agent_nick.settings, "email_response_poll_seconds", 60
        )

        return {
            "await_all_responses": True,
            "await_response": True,
            "drafts": drafts,
            "workflow_id": workflow_id,
            "round": round_num,
            "expected_dispatch_count": len(drafts),
            "expected_email_count": len(drafts),
            "expected_unique_ids": unique_ids,
            "unique_ids": unique_ids,
            "supplier_ids": supplier_ids,
            "response_poll_interval": poll_interval,
            "response_batch_limit": len(drafts),
        }

    def _map_responses_to_suppliers(
        self,
        responses: List[Dict[str, Any]],
        drafts: List[Dict[str, Any]],
        negotiation_state: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Map responses back to supplier identifiers."""

        supplier_responses: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        unique_id_to_supplier: Dict[str, str] = {}
        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            unique_id = self._coerce_text(draft.get("unique_id"))
            supplier_id = self._coerce_text(
                draft.get("supplier_id") or draft.get("supplier")
            )
            if unique_id and supplier_id:
                unique_id_to_supplier[unique_id] = supplier_id

        for response in responses:
            if not isinstance(response, dict):
                continue

            supplier_id = self._coerce_text(
                response.get("supplier_id") or response.get("supplier")
            )

            if not supplier_id:
                unique_id = self._coerce_text(response.get("unique_id"))
                if unique_id:
                    supplier_id = unique_id_to_supplier.get(unique_id)

            if not supplier_id:
                metadata = response.get("metadata")
                if isinstance(metadata, dict):
                    supplier_id = self._coerce_text(
                        metadata.get("supplier_id") or metadata.get("supplier")
                    )

            if supplier_id:
                supplier_responses[supplier_id].append(response)
            else:
                logger.warning(
                    "Could not identify supplier for response %s",
                    response.get("message_id"),
                )

        return dict(supplier_responses)

    def _process_round_responses(
        self,
        supplier_responses: Dict[str, List[Dict[str, Any]]],
        negotiation_state: Dict[str, Any],
        round_num: int,
    ) -> None:
        """Update negotiation state with supplier responses."""

        workflow_hint = self._coerce_text(negotiation_state.get("workflow_id"))

        for supplier_id, responses in supplier_responses.items():
            supplier_state = negotiation_state["active_suppliers"].get(supplier_id)
            if not supplier_state:
                logger.warning(
                    "Received response for unknown supplier %s in round %s",
                    supplier_id,
                    round_num,
                )
                continue

            entry_payload = supplier_state.get("entry")
            if not isinstance(entry_payload, dict):
                entry_payload = {}
                supplier_state["entry"] = entry_payload

            supplier_workflow = (
                workflow_hint
                or self._coerce_text(entry_payload.get("workflow_id"))
                or self._coerce_text(entry_payload.get("workflow"))
            )

            supplier_state.setdefault("responses", []).extend(
                [response for response in responses if isinstance(response, dict)]
            )

            for response in responses:
                if not isinstance(response, dict):
                    continue

                latest_price = self._extract_price_from_response(response)
                if latest_price is not None:
                    supplier_state["entry"]["current_offer"] = latest_price
                    supplier_state["entry"]["price"] = latest_price

                message_text = self._extract_message_from_response(response)
                if message_text:
                    supplier_state["entry"]["supplier_message"] = message_text

                if self._detect_final_offer(message_text, []):
                    supplier_state["entry"]["final_offer_signaled"] = True

                response_workflow = supplier_workflow or self._coerce_text(
                    response.get("workflow_id")
                )
                if response_workflow:
                    if not supplier_workflow:
                        supplier_workflow = response_workflow
                    if not workflow_hint:
                        workflow_hint = response_workflow
                        negotiation_state.setdefault("workflow_id", response_workflow)
                    if "workflow_id" not in entry_payload:
                        entry_payload["workflow_id"] = response_workflow
                self._capture_supplier_response(
                    workflow_id=response_workflow,
                    supplier_id=supplier_id,
                    round_number=round_num,
                    response=response,
                )

            if responses:
                try:
                    current_round_value = int(
                        supplier_state.get("current_round", round_num)
                    )
                except (TypeError, ValueError):
                    current_round_value = round_num
                if current_round_value <= round_num:
                    supplier_state["current_round"] = round_num + 1

            logger.info(
                "Processed %s response(s) for supplier %s in round %s",
                len(responses),
                supplier_id,
                round_num,
            )

    def _extract_price_from_response(self, response: Dict[str, Any]) -> Optional[float]:
        """Extract a numeric price from a supplier response payload."""

        for key in ("price", "quoted_price", "unit_price", "offer_price", "current_offer"):
            value = response.get(key)
            parsed = self._parse_money(value)
            if parsed is not None:
                return parsed

        message_text = self._extract_message_from_response(response)
        if message_text:
            patterns = [
                r"[£$€₹]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
                r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:GBP|USD|EUR|INR)",
            ]
            for pattern in patterns:
                match = re.search(pattern, message_text)
                if match:
                    try:
                        price_str = match.group(1).replace(",", "")
                        return float(price_str)
                    except (ValueError, IndexError):
                        continue

        return None

    def _extract_message_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract plain text content from a supplier response."""

        for key in ("message", "message_text", "body", "text", "content"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _consolidate_multi_round_results(
        self,
        context: AgentContext,
        negotiation_state: Dict[str, Any],
        all_round_results: List[Dict[str, Any]],
    ) -> AgentOutput:
        """Aggregate results and produce the final agent output for multi-round runs."""

        supplier_summaries: Dict[str, Dict[str, Any]] = {}

        active_suppliers = negotiation_state.get("active_suppliers", {})
        for supplier_id, supplier_state in active_suppliers.items():
            if not isinstance(supplier_state, dict):
                continue
            supplier_summaries[supplier_id] = {
                "supplier_id": supplier_id,
                "final_status": supplier_state.get("status"),
                "total_rounds": len(supplier_state.get("decisions", [])),
                "responses_received": len(supplier_state.get("responses", [])),
                "decisions": list(supplier_state.get("decisions", [])),
                "latest_decision": (
                    supplier_state.get("decisions", [])[-1]
                    if supplier_state.get("decisions")
                    else None
                ),
            }

        for supplier_id in negotiation_state.get("completed_suppliers", set()):
            supplier_summary = supplier_summaries.setdefault(
                supplier_id,
                {
                    "supplier_id": supplier_id,
                    "final_status": "COMPLETED",
                    "total_rounds": 0,
                    "responses_received": 0,
                    "decisions": [],
                    "latest_decision": None,
                },
            )
            supplier_summary["completed"] = True

        for supplier_id, failure_info in negotiation_state.get(
            "failed_suppliers", {}
        ).items():
            supplier_summaries.setdefault(
                supplier_id,
                {
                    "supplier_id": supplier_id,
                    "final_status": "FAILED",
                    "failure_info": failure_info,
                },
            )

        all_drafts: List[Dict[str, Any]] = []
        for round_result in all_round_results:
            drafts_payload = round_result.get("drafts")
            if isinstance(drafts_payload, list):
                all_drafts.extend(
                    [draft for draft in drafts_payload if isinstance(draft, dict)]
                )

        completed_suppliers = sorted(
            list(negotiation_state.get("completed_suppliers", set()))
        )

        failed_suppliers = sorted(
            list(negotiation_state.get("failed_suppliers", {}).keys())
        )

        final_positions: List[Dict[str, Any]] = []
        for supplier_id, summary in supplier_summaries.items():
            supplier_state = (
                active_suppliers.get(supplier_id)
                if isinstance(active_suppliers, dict)
                else None
            )
            entry_payload = (
                supplier_state.get("entry")
                if isinstance(supplier_state, dict)
                else {}
            )
            latest_decision = summary.get("latest_decision")
            if isinstance(latest_decision, dict):
                decision_copy = dict(latest_decision)
            else:
                decision_copy = {}
            currency = (
                decision_copy.get("currency")
                or entry_payload.get("currency")
                or entry_payload.get("currency_code")
            )
            final_positions.append(
                {
                    "supplier_id": supplier_id,
                    "final_status": summary.get("final_status"),
                    "rounds_completed": summary.get("total_rounds"),
                    "final_supplier_offer": entry_payload.get("current_offer")
                    or entry_payload.get("price"),
                    "final_counter_offer": decision_copy.get("counter_price"),
                    "currency": currency,
                    "final_strategy": decision_copy.get("strategy"),
                }
            )

        data = {
            "workflow_id": negotiation_state.get("workflow_id"),
            "negotiation_type": "multi_round",
            "total_rounds_executed": len(all_round_results),
            "max_rounds": negotiation_state.get("max_rounds", 3),
            "total_suppliers": negotiation_state.get("total_suppliers", 0),
            "completed_suppliers": completed_suppliers,
            "failed_suppliers": failed_suppliers,
            "supplier_summaries": supplier_summaries,
            "round_history": negotiation_state.get("round_history", []),
            "all_drafts": all_drafts,
            "final_positions": final_positions,
            "final_status": self._determine_overall_status(negotiation_state),
        }

        self._log_final_negotiation_outcome(context, data)

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=data,
            ),
        )

    def _determine_overall_status(self, negotiation_state: Dict[str, Any]) -> str:
        """Summarise the overall negotiation status for reporting."""

        total = negotiation_state.get("total_suppliers", 0) or 0
        completed = len(negotiation_state.get("completed_suppliers", set()))
        failed = len(negotiation_state.get("failed_suppliers", {}))

        if total > 0 and completed == total:
            return "ALL_COMPLETED"
        if total > 0 and failed == total:
            return "ALL_FAILED"
        if total > 0 and completed + failed == total:
            return "PARTIALLY_COMPLETED"
        return "IN_PROGRESS"

    def _run_single_negotiation(self, context: AgentContext) -> AgentOutput:

        payload = context.input_data if isinstance(context.input_data, dict) else {}
        identifier = self._resolve_negotiation_identifier(context)
        supplier = identifier.supplier_id
        session_reference = identifier.session_reference
        workflow_id = identifier.workflow_id
        session_id = workflow_id

        round_hint = None
        if isinstance(payload, dict):
            round_hint = payload.get("round")
        try:
            lock_round = int(round_hint) if round_hint is not None else identifier.round_number
        except Exception:
            lock_round = identifier.round_number
        if not isinstance(lock_round, int) or lock_round < 1:
            lock_round = max(int(identifier.round_number or 1), 1)

        lock_context = self._session_lock(workflow_id, supplier, lock_round)
        lock_acquired = lock_context.__enter__()
        if not lock_acquired:
            lock_context.__exit__(None, None, None)
            logger.warning(
                "NegotiationAgent concurrency guard prevented duplicate run",
                extra={
                    "workflow_id": workflow_id,
                    "supplier_id": supplier,
                    "round": lock_round,
                },
            )
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={
                        "workflow_id": workflow_id,
                        "supplier_id": supplier,
                        "round": lock_round,
                        "locked": True,
                        "message": "Another negotiation instance is already processing this supplier round.",
                    },
                    error="negotiation_session_locked",
                ),
            )

        try:
            return self._run_single_negotiation_locked(
                context,
                identifier,
                payload,
                session_reference,
                supplier,
                workflow_id,
                session_id,
                round_hint,
            )
        finally:
            lock_context.__exit__(None, None, None)

    def _run_single_negotiation_locked(
        self,
        context: AgentContext,
        identifier: NegotiationIdentifier,
        payload: Dict[str, Any],
        session_reference: str,
        supplier: str,
        workflow_id: str,
        session_id: Optional[str],
        raw_round_hint: Any,
    ) -> AgentOutput:

        rfq_id: Optional[str] = None
        for key in ("rfq_id", "rfqId", "rfq", "rfq_reference", "rfqReference"):
            candidate = payload.get(key) if isinstance(payload, dict) else None
            text = self._coerce_text(candidate)
            if text:
                rfq_id = text
                break

        state_identifier: Optional[str] = self._coerce_text(workflow_id)
        if not state_identifier:
            for candidate in (rfq_id, session_reference, session_id):
                text = self._coerce_text(candidate)
                if text:
                    state_identifier = text
                    break
        if not state_identifier:
            state_identifier = identifier.workflow_id
        else:
            identifier.workflow_id = state_identifier

        if rfq_id and isinstance(context.input_data, dict):
            context.input_data.setdefault("rfq_id", rfq_id)
        rfq_value = rfq_id or state_identifier
        raw_round = raw_round_hint if raw_round_hint is not None else identifier.round_number

        state_identifier = identifier.workflow_id or state_identifier

        price_raw = (
            context.input_data.get("current_offer")
            if context.input_data.get("current_offer") is not None
            else context.input_data.get("price")
        )

        normalised_inputs, validation_issues = self._normalise_negotiation_inputs(
            context.input_data
        )

        price = normalised_inputs.get("current_offer")
        target_price = normalised_inputs.get("target_price")
        walkaway_price = normalised_inputs.get("walkaway_price")
        currency = normalised_inputs.get("currency")
        lead_weeks = normalised_inputs.get("lead_time_weeks")
        volume_units = normalised_inputs.get("volume_units")
        term_days = normalised_inputs.get("term_days")
        valid_until = normalised_inputs.get("valid_until")
        market_floor = normalised_inputs.get("market_floor_price")

        currency_conf = self._coerce_float(
            context.input_data.get("currency_confidence")
            or context.input_data.get("current_offer_currency_confidence")
        )
        if currency_conf is not None and currency_conf < 0.5:
            logger.warning(
                "Currency confidence %.2f below threshold; withholding price data", currency_conf
            )
            currency = None
            price = None

        constraints = self._ensure_list(context.input_data.get("constraints"))
        supplier_snippets = self._collect_supplier_snippets(context.input_data)
        supplier_message = self._coerce_text(
            context.input_data.get("supplier_message")
            or context.input_data.get("response_text")
            or context.input_data.get("message")
        )

        # Load state before subject normalization (bug fix)
        state, _ = self._load_session_state(state_identifier, supplier)
        state["workflow_id"] = identifier.workflow_id
        state["session_reference"] = session_reference
        state["supplier_id"] = supplier

        round_no = int(state.get("current_round", 1))
        if isinstance(raw_round, (int, float)):
            try:
                round_no = max(round_no, int(raw_round))
            except (TypeError, ValueError):
                round_no = max(round_no, 1)
        round_no = max(1, min(round_no, 3))
        state["current_round"] = round_no
        identifier.round_number = round_no

        previous_positions = state.get("positions") if isinstance(state.get("positions"), dict) else None
        previous_counter_hint = (
            context.input_data.get("agent_previous_offer")
            or context.input_data.get("previous_counter_price")
            or (
                previous_positions.get("last_counter")
                if isinstance(previous_positions, dict)
                else None
            )
            or state.get("last_counter_price")
        )
        previous_counter = self._coerce_float(previous_counter_hint)
        positions = self._build_positions(
            supplier_offer=price,
            target_price=target_price,
            walkaway_price=walkaway_price,
            previous_counter=previous_counter,
            previous_positions=previous_positions,
            round_no=round_no,
        )
        target_price = positions.desired
        walkaway_price = positions.no_deal
        price = (
            positions.supplier_offer
            if positions.supplier_offer is not None
            else positions.start
        )

        incoming_subject = self._coerce_text(context.input_data.get("subject"))
        base_candidate = self._normalise_base_subject(incoming_subject)
        if base_candidate and not state.get("base_subject"):
            state["base_subject"] = base_candidate

        thread_state = self._load_thread_state(
            identifier, state, subject_hint=incoming_subject
        )

        signals = self._extract_negotiation_signals(
            supplier_message=supplier_message,
            snippets=supplier_snippets,
            market=context.input_data.get("market_context") or {},
            performance=context.input_data.get("supplier_performance") or {},
        )

        zopa = self._estimate_zopa(
            price=price,
            target=target_price,
            history=context.input_data.get("history") or {},
            benchmarks=context.input_data.get("benchmarks") or {},
            should_cost=context.input_data.get("should_cost"),
            signals=signals,
        )

        pricing_payload: Dict[str, Any] = {
            "current_offer": price,
            "target_price": target_price,
            "round": raw_round,
            "currency": currency,
            "max_rounds": context.input_data.get("max_rounds") or 3,
            "walkaway_price": walkaway_price,
            "ask_early_pay_disc": context.input_data.get("ask_early_pay_disc", 0.02),
            "ask_lead_time_keep": context.input_data.get("ask_lead_time_keep", True),
        }
        pricing_payload["start_position"] = positions.start
        pricing_payload["desired_position"] = positions.desired
        pricing_payload["no_deal_position"] = positions.no_deal

        offer_prev = normalised_inputs.get("previous_offer")
        if offer_prev is None and positions.history:
            for entry in reversed(positions.history):
                if entry.get("type") != "supplier_offer":
                    continue
                value = self._coerce_float(entry.get("value"))
                if value is None:
                    continue
                if positions.supplier_offer is not None and NegotiationPositions._is_close(
                    value, positions.supplier_offer
                ):
                    continue
                offer_prev = value
                break

        decision = decide_strategy(
            pricing_payload,
            lead_weeks=lead_weeks,
            constraints=constraints,
            supplier_message=supplier_message,
            offer_prev=offer_prev,
        )
        decision = self._adaptive_strategy(
            base_decision=decision,
            zopa=zopa,
            signals=signals,
            round_hint=raw_round,
            lead_weeks=lead_weeks,
            target_price=target_price,
            price=price,
        )
        base_plan_message = decision.get("rationale")
        decision.setdefault("strategy", "clarify")
        decision.setdefault("counter_price", None)
        decision.setdefault("asks", [])
        decision.setdefault("lead_time_request", None)
        decision.setdefault("rationale", "")
        decision["closing_round"] = round_no >= 3
        decision["counter_price"] = self._respect_positions(
            decision.get("counter_price"), positions
        )
        decision["positions"] = positions.serialise()
        if validation_issues:
            decision["validation_issues"] = validation_issues
        if volume_units is not None:
            decision["volume_units"] = volume_units
        if term_days is not None:
            decision["term_days"] = term_days
        if valid_until:
            decision["valid_until"] = valid_until
        if market_floor is not None:
            decision["market_floor_price"] = market_floor
        decision["plan_counter_message"] = base_plan_message

        outlier_result = self._detect_outliers(
            supplier_offer=positions.supplier_offer,
            target_price=positions.desired,
            walkaway_price=positions.no_deal,
            market_floor=market_floor,
            volume_units=volume_units,
            term_days=term_days,
        )
        if outlier_result.get("alerts"):
            decision.setdefault("alerts", [])
            for alert in outlier_result["alerts"]:
                if alert not in decision["alerts"]:
                    decision["alerts"].append(alert)
            decision["outlier_alerts"] = outlier_result["alerts"]
        if outlier_result.get("requires_review"):
            decision["strategy"] = "review"
            decision.setdefault("flags", {})
            decision["flags"]["review_recommended"] = True
            decision["recommendation"] = (
                outlier_result.get("recommendation") or "query_for_human_review"
            )
            asks = list(decision.get("asks", []))
            asks.append("Please justify the requested pricing/terms for internal review.")
            decision["asks"] = list(dict.fromkeys(asks))
            if not outlier_result.get("human_override"):
                decision["counter_price"] = self._respect_positions(
                    decision.get("counter_price"), positions
                )
        if outlier_result.get("human_override"):
            decision.setdefault("flags", {})
            decision["flags"]["human_override_required"] = True
            decision["human_override_required"] = True
            decision["counter_price"] = None

        structured_rationale = self._compose_rationale(
            positions=positions,
            decision=decision,
            currency=currency,
            lead_weeks=lead_weeks,
            volume_units=volume_units,
            term_days=term_days,
            outlier_message=outlier_result.get("message"),
            validation_issues=validation_issues,
        )
        if base_plan_message and base_plan_message not in structured_rationale:
            decision["rationale"] = f"{structured_rationale} {base_plan_message}".strip()
        else:
            decision["rationale"] = structured_rationale

        playbook_context = self._resolve_playbook_context(context, decision)
        play_recommendations = playbook_context.get("plays", [])
        if play_recommendations:
            decision["play_recommendations"] = play_recommendations
            descriptor = playbook_context.get("descriptor")
            if descriptor:
                decision["playbook_descriptor"] = descriptor
            style_used = playbook_context.get("style")
            if style_used:
                decision["playbook_style"] = style_used
            lever_focus = playbook_context.get("lever_priorities")
            if lever_focus:
                decision.setdefault("lever_priorities", lever_focus)
            examples = playbook_context.get("examples")
            if examples:
                decision["playbook_examples"] = examples
        else:
            decision.setdefault("play_recommendations", [])

        message_id = self._coerce_text(context.input_data.get("message_id"))
        supplier_reply_registered = False
        if message_id and state.get("last_supplier_msg_id") != message_id:
            previously_awaiting = bool(state.get("awaiting_response", False))
            if previously_awaiting:
                state["supplier_reply_count"] = int(state.get("supplier_reply_count", 0)) + 1
            else:
                state["supplier_reply_count"] = int(state.get("supplier_reply_count", 0))
            state["last_supplier_msg_id"] = message_id
            state["awaiting_response"] = False
            try:
                current_round_value = int(state.get("current_round", round_no))
            except (TypeError, ValueError):
                current_round_value = round_no
            if current_round_value <= round_no:
                state["current_round"] = round_no + 1
            supplier_reply_registered = True
            if thread_state:
                thread_state.update_after_receive(message_id)
            if self.response_matcher and identifier.workflow_id and supplier:
                matched = self.response_matcher.match_response(
                    identifier.workflow_id,
                    supplier,
                    message_id,
                    context.input_data if isinstance(context.input_data, dict) else {},
                )
                if matched:
                    matched_list = state.setdefault("matched_responses", [])
                    if isinstance(matched_list, list):
                        matched_list.append(matched)

        decision["round"] = round_no
        state["positions"] = positions.snapshot_for_next_round(
            decision.get("counter_price"), round_no
        )
        state["last_counter_price"] = decision.get("counter_price")

        final_offer_reason = self._detect_final_offer(supplier_message, supplier_snippets)
        should_continue, new_status, halt_reason = self._should_continue(
            state, supplier_reply_registered, final_offer_reason
        )

        if outlier_result.get("human_override"):
            should_continue = False
            new_status = "PAUSED"
            halt_reason = outlier_result.get("message") or "Human override required."
            decision.setdefault("status_reason", halt_reason)
        elif outlier_result.get("requires_review") and outlier_result.get("message"):
            decision.setdefault("status_reason", outlier_result.get("message"))

        savings_score = 0.0
        if price and price > 0 and target_price is not None:
            try:
                savings_score = (price - target_price) / float(price)
            except ZeroDivisionError:
                savings_score = 0.0

        decision_log = self._build_decision_log(
            supplier,
            session_reference,
            price,
            target_price,
            decision,
        )

        draft_records: List[Dict[str, Any]] = []

        counter_options: List[Dict[str, Any]] = []

        sent_message_id: Optional[str] = None

        if not should_continue:
            supplier_name = context.input_data.get("supplier_name") or supplier
            state["status"] = new_status
            state["awaiting_response"] = halt_reason == "Awaiting supplier response."
            stop_message = self._build_stop_message(new_status, halt_reason, round_no)
            decision.setdefault("status_reason", halt_reason)
            self._persist_thread_state(state, thread_state)
            email_thread_history = self._get_email_thread_history(
                identifier.workflow_id, supplier
            )
            email_thread_summary = self._get_email_thread_summary(
                identifier.workflow_id, supplier
            )
            state["email_history"] = email_thread_history
            self._save_session_state(identifier.workflow_id, supplier, state)
            self._record_learning_snapshot(
                context,
                identifier.workflow_id or state_identifier or session_id,
                supplier,
                decision,
                state,
                bool(state.get("awaiting_response", False)),
                supplier_reply_registered,
                rfq_id=rfq_value,
            )
            awaiting_now = bool(state.get("awaiting_response", False))
            negotiation_open = (
                new_status not in {"COMPLETED", "EXHAUSTED"}
                and not awaiting_now
                and decision.get("strategy") != "review"
                and not decision.get("human_override_required")
            )
            data = {
                "supplier": supplier,
                "rfq_id": rfq_value,
                "session_reference": session_reference,
                "unique_id": session_reference,
                "round": round_no,
                "counter_proposals": counter_options,
                "decision": decision,
                "savings_score": savings_score,
                "decision_log": decision_log,
                "message": stop_message,
                "email_subject": None,
                "email_body": None,
                "supplier_snippets": supplier_snippets,
                "negotiation_allowed": negotiation_open,
                "interaction_type": "negotiation",
                "session_state": self._public_state(state),
                "currency": currency,
                "current_offer": price,
                "target_price": target_price,
                "supplier_message": supplier_message,
                "drafts": draft_records,
                "sent_status": False,
                "awaiting_response": awaiting_now,
                "play_recommendations": play_recommendations,
                "playbook_descriptor": playbook_context.get("descriptor"),
                "playbook_examples": playbook_context.get("examples"),
                "positions": decision.get("positions"),
                "validation_issues": decision.get("validation_issues"),
                "outlier_alerts": decision.get("outlier_alerts"),
                "volume_units": volume_units,
                "term_days": term_days,
                "valid_until": valid_until,
                "flags": decision.get("flags"),
                "market_floor_price": market_floor,
                "normalised_inputs": normalised_inputs,
                "email_history": email_thread_history,
                "email_thread_summary": email_thread_summary,
                "current_email": draft_records[0] if draft_records else None,
                "all_emails_sent": len(email_thread_history),
            }
            history_payload, history_summary, all_sent = self._compose_email_history_payload(
                identifier.workflow_id,
                supplier,
                state,
            )
            data["email_history"] = history_payload
            data["email_thread_summary"] = history_summary
            data["all_emails_sent"] = all_sent
            logger.info(
                "NegotiationAgent halted negotiation for supplier=%s session_id=%s reason=%s",
                supplier,
                session_id,
                halt_reason,
            )
            pass_fields = dict(data)
            next_agents: List[str] = []
            if awaiting_now:
                watch_fields = self._build_supplier_watch_fields(
                    context=context,
                    session_reference=session_reference,
                    supplier=supplier,
                    drafts=draft_records,
                    state=state,
                    workflow_id=session_id,
                    rfq_id=rfq_value,
                )
                if watch_fields:
                    pass_fields.update(watch_fields)
                    next_agents.append("SupplierInteractionAgent")
            output = AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=pass_fields,
                next_agents=next_agents,
            )
            output = self._with_plan(context, output)
            subject_hint = None
            if draft_records:
                primary_draft = draft_records[0]
                if isinstance(primary_draft, dict):
                    subject_hint = primary_draft.get("subject")

            self._queue_email_draft_action(
                context,
                supplier_id=supplier,
                supplier_name=supplier_name,
                round_number=round_no,
                subject=subject_hint,
                body=stop_message,
                drafts=draft_records,
                decision=decision,
                negotiation_message=stop_message,
                agentic_plan=output.agentic_plan,
                context_snapshot=self._build_email_context_snapshot(context),
            )
            return output

        optimized = self._optimize_multi_issue(
            price=price,
            currency=currency,
            target=target_price,
            lead_weeks=lead_weeks,
            weights=context.input_data.get("weights") or {},
            policy=context.input_data.get("policy_guidance") or {},
            constraints=context.input_data.get("hard_constraints") or {},
            zopa=zopa,
            signals=signals,
            round_no=round_no,
        )
        overrides = optimized.get("decision_overrides", {}) or {}
        if decision.get("price_plan_locked"):
            overrides = {k: v for k, v in overrides.items() if k != "counter_price"}
        decision.update(overrides)
        counter_options = optimized.get("counter_options") or []
        if not counter_options and decision.get("counter_price") is not None:
            counter_options = [{"price": decision["counter_price"], "terms": None, "bundle": None}]

        supplier_name = context.input_data.get("supplier_name") or supplier
        decision["supplier_id"] = supplier
        decision.setdefault("supplier_name", supplier_name)
        decision.setdefault("supplier", supplier_name)
        decision.setdefault("workflow_id", identifier.workflow_id)
        decision.setdefault("workflow_ref", identifier.workflow_id)
        decision.setdefault("session_reference", session_reference)
        decision.setdefault("unique_id", session_reference)
        decision.setdefault("round", round_no)
        decision.setdefault("rfq_id", rfq_value)

        contact_name = self._resolve_contact_name(
            context.input_data if isinstance(context.input_data, dict) else {},
            fallback=supplier_name,
        )
        supplier_identifier = context.input_data.get("supplier_id") or supplier
        procurement_summary = self._retrieve_procurement_summary(
            supplier_id=supplier_identifier,
            supplier_name=supplier_name,
        )
        rag_snippets = self._collect_vector_snippets(
            context=context,
            supplier_name=supplier_name or supplier_identifier,
            workflow_reference=session_reference,
        )

        negotiation_message = self._build_summary(
            context,
            session_reference,
            decision,
            price,
            target_price,
            currency,
            round_no,
            supplier=supplier,
            supplier_name=supplier_name,
            supplier_message=supplier_message,
            supplier_snippets=supplier_snippets,
            playbook_context=playbook_context,
            signals=signals,
            zopa=zopa,
            procurement_summary=procurement_summary,
            rag_snippets=rag_snippets,
            contact_name=contact_name,
        )

        if play_recommendations:
            negotiation_message = self._append_playbook_recommendations(
                negotiation_message,
                play_recommendations,
                playbook_context,
            )

        if state_identifier and supplier:
            self._store_session(
                state_identifier, supplier, round_no, decision.get("counter_price")
            )

        try:
            supplier_reply_count = int(state.get("supplier_reply_count", 0))
        except Exception:
            supplier_reply_count = 0

        draft_payload = {
            "intent": "NEGOTIATION_COUNTER",
            "session_reference": session_reference,
            "unique_id": session_reference,
            "supplier_id": supplier,
            "workflow_id": identifier.workflow_id,
            "supplier_name": supplier_name,
            "rfq_id": rfq_value,
            "current_offer": price_raw,
            "current_offer_numeric": price,
            "target_price": target_price,
            "counter_price": decision.get("counter_price"),
            "currency": currency,
            "decision": decision,
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "round": round_no,
            "closing_round": round_no >= 3,
            "supplier_reply_count": supplier_reply_count,
            "supplier_message": supplier_message,
            "supplier_snippets": supplier_snippets,
            "from_address": context.input_data.get("from_address"),
            "negotiation_message": negotiation_message,
            "play_recommendations": play_recommendations,
            "playbook_descriptor": playbook_context.get("descriptor"),
            "playbook_examples": playbook_context.get("examples"),
            "counter_options": counter_options,
            "positions": decision.get("positions"),
            "validation_issues": decision.get("validation_issues"),
            "outlier_alerts": decision.get("outlier_alerts"),
            "flags": decision.get("flags"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
            "normalised_inputs": normalised_inputs,
            "human_override_required": decision.get("human_override_required", False),
            "contact_name": contact_name,
            "supplier_contact": contact_name,
        }

        created_at = datetime.now(timezone.utc)
        draft_payload.setdefault("draft_created_at", created_at.isoformat())
        if session_reference:
            freshness_payload = (
                draft_payload.get("dispatch_freshness")
                if isinstance(draft_payload.get("dispatch_freshness"), dict)
                else {}
            )
            existing_entry = freshness_payload.get(session_reference)
            if isinstance(existing_entry, dict):
                entry = dict(existing_entry)
            else:
                entry = {}
            entry.setdefault("thread_index", round_no)
            entry.setdefault("min_dispatched_at", draft_payload["draft_created_at"])
            freshness_payload[session_reference] = entry
            draft_payload["dispatch_freshness"] = freshness_payload

        draft_payload.setdefault("subject", self._format_negotiation_subject(state))

        recipients = self._collect_recipient_candidates(context)
        if recipients:
            draft_payload.setdefault("recipients", recipients)

        computed_thread_headers: Optional[Dict[str, Any]] = None
        if thread_state:
            computed_thread_headers = thread_state.to_headers(round_no)

        thread_headers: Optional[Any] = None
        if isinstance(context.input_data, dict):
            incoming_thread_headers = context.input_data.get("thread_headers")
            if isinstance(incoming_thread_headers, dict):
                thread_headers = dict(incoming_thread_headers)
            else:
                thread_headers = incoming_thread_headers
        if not thread_headers:
            cached_thread_headers = state.get("last_thread_headers")
            if isinstance(cached_thread_headers, dict):
                thread_headers = dict(cached_thread_headers)
            else:
                thread_headers = cached_thread_headers
        if not thread_headers and computed_thread_headers:
            thread_headers = dict(computed_thread_headers)
        if isinstance(thread_headers, dict) and thread_headers:
            draft_payload.setdefault("thread_headers", thread_headers)
            state["last_thread_headers"] = dict(thread_headers)
        elif thread_headers:
            draft_payload.setdefault("thread_headers", thread_headers)
            state["last_thread_headers"] = thread_headers
        elif computed_thread_headers:
            draft_payload.setdefault("thread_headers", dict(computed_thread_headers))
            state["last_thread_headers"] = dict(computed_thread_headers)

        draft_metadata = {
            "counter_price": decision.get("counter_price"),
            "target_price": target_price,
            "current_offer": price,
            "round": round_no,
            "closing_round": round_no >= 3,
            "supplier_reply_count": supplier_reply_count,
            "strategy": decision.get("strategy"),
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "intent": "NEGOTIATION_COUNTER",
            "play_recommendations": play_recommendations,
            "positions": decision.get("positions"),
            "validation_issues": decision.get("validation_issues"),
            "outlier_alerts": decision.get("outlier_alerts"),
            "flags": decision.get("flags"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
            "contact_name": contact_name,
            "supplier_contact": contact_name,
        }

        email_action_id: Optional[str] = None
        email_subject: Optional[str] = None
        email_body: Optional[str] = None
        draft_records: List[Dict[str, Any]] = []
        next_agents: List[str] = []

        draft_stub = self._build_email_draft_stub(
            context=context,
            draft_payload=draft_payload,
            metadata=draft_metadata,
            negotiation_message=negotiation_message,
            supplier_id=supplier,
            supplier_name=supplier_name,
            contact_name=contact_name,
            session_reference=session_reference,
            rfq_id=rfq_value,
            recipients=recipients,
            thread_headers=thread_headers if isinstance(thread_headers, dict) else None,
            round_number=round_no,
            decision=decision,
            currency=currency,
            playbook_context=playbook_context,
        )
        draft_stub.setdefault("intent", "NEGOTIATION_COUNTER")
        draft_stub["negotiation_message"] = negotiation_message
        draft_stub["counter_proposals"] = counter_options
        draft_stub["thread_index"] = round_no
        draft_stub["closing_round"] = round_no >= 3
        if currency:
            draft_stub["currency"] = currency
        if supplier_snippets:
            draft_stub["supplier_snippets"] = supplier_snippets
        if play_recommendations:
            draft_stub["play_recommendations"] = play_recommendations
        descriptor = playbook_context.get("descriptor")
        if descriptor:
            draft_stub["playbook_descriptor"] = descriptor
        if thread_headers and not isinstance(thread_headers, dict):
            draft_stub["thread_headers"] = thread_headers

        email_payload = self._build_email_agent_payload(
            context,
            draft_payload,
            decision,
            state,
            negotiation_message,
        )

        finalization_task = self._build_email_finalization_task(
            context=context,
            identifier=identifier,
            state=state,
            thread_state=thread_state,
            draft_payload=draft_payload,
            draft_stub=draft_stub,
            email_payload=email_payload,
            decision=decision,
            negotiation_message=negotiation_message,
            supplier_message=supplier_message,
            supplier_snippets=supplier_snippets,
            supplier_name=supplier_name,
            contact_name=contact_name,
            session_reference=session_reference,
            rfq_value=rfq_value,
            round_no=round_no,
            workflow_id=workflow_id,
            supplier=supplier,
            currency=currency,
            volume_units=volume_units,
            term_days=term_days,
            valid_until=valid_until,
            market_floor=market_floor,
            normalised_inputs=normalised_inputs,
            supplier_reply_registered=supplier_reply_registered,
            state_identifier=state_identifier,
            playbook_context=playbook_context,
            recipients=recipients,
            thread_headers=thread_headers,
            counter_options=counter_options,
            savings_score=savings_score,
            decision_log=decision_log,
        )

        if bool(context.input_data.get("_batch_execution") if isinstance(context.input_data, dict) else False):
            deferred_payload = {
                "supplier": supplier,
                "rfq_id": rfq_value,
                "session_reference": session_reference,
                "unique_id": session_reference,
                "round": round_no,
                "decision": decision,
                "negotiation_message": negotiation_message,
                "deferred_email": True,
                "pending_email": finalization_task,
            }
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=deferred_payload,
                    pass_fields={
                        "pending_email": finalization_task,
                        "deferred_email": True,
                    },
                ),
            )

        email_output: Optional[AgentOutput] = None
        fallback_payload: Optional[Dict[str, Any]] = None
        if email_payload and supplier and session_reference:
            email_output = self._invoke_email_drafting_agent(context, email_payload)
            fallback_payload = dict(email_payload)
        if email_output and email_output.status == AgentStatus.SUCCESS:
            email_data = email_output.data or {}
            email_action_id = email_output.action_id or email_data.get("action_id")
            email_subject = email_data.get("subject")
            email_body = email_data.get("body")
            candidate_headers = email_data.get("thread_headers")
            if isinstance(candidate_headers, dict):
                candidate_headers = {
                    key: value
                    for key, value in candidate_headers.items()
                    if value is not None
                }
                header_message_id = self._coerce_text(
                    candidate_headers.get("Message-ID")
                    or candidate_headers.get("message_id")
                )
                if header_message_id:
                    sent_message_id = header_message_id
                draft_payload["thread_headers"] = dict(candidate_headers)
                state["last_thread_headers"] = dict(candidate_headers)
            drafts_payload = email_data.get("drafts")
            if isinstance(drafts_payload, list) and drafts_payload:
                for draft in drafts_payload:
                    if not isinstance(draft, dict):
                        continue
                    draft_copy = dict(draft)
                    if email_action_id:
                        draft_copy.setdefault("email_action_id", email_action_id)
                    if not sent_message_id:
                        candidate_id = self._coerce_text(
                            draft_copy.get("message_id")
                            or draft_copy.get("Message-ID")
                        )
                        if not candidate_id:
                            headers_payload = draft_copy.get("headers") or draft_copy.get("thread_headers")
                            if isinstance(headers_payload, dict):
                                candidate_id = self._coerce_text(
                                    headers_payload.get("Message-ID")
                                    or headers_payload.get("message_id")
                                )
                        if candidate_id:
                            sent_message_id = candidate_id
                    draft_records.append(draft_copy)
            else:
                draft_records.append(dict(draft_stub))
        else:
            draft_records.append(dict(draft_stub))
            next_agents = ["EmailDraftingAgent"]
            if fallback_payload is None:
                # Reconstruct a payload compatible with EmailDraftingAgent expectations.
                fallback_payload = self._build_email_agent_payload(
                    context,
                    draft_payload,
                    decision,
                    state,
                    negotiation_message,
                )
        subject_seed = self._coerce_text(draft_payload.get("subject"))

        subject_candidate = email_subject or subject_seed
        if subject_candidate and not state.get("base_subject"):
            base_from_email = self._normalise_base_subject(subject_candidate)
            if base_from_email:
                state["base_subject"] = base_from_email

        if email_subject:
            base_from_email = self._normalise_base_subject(email_subject)
            if base_from_email and not state.get("base_subject"):
                state["base_subject"] = base_from_email
        elif subject_seed and not state.get("base_subject"):
            fallback_base = self._normalise_base_subject(subject_seed)
            if fallback_base:
                state["base_subject"] = fallback_base

        if email_body and not state.get("initial_body"):
            state["initial_body"] = email_body
        elif not state.get("initial_body") and negotiation_message:
            state["initial_body"] = negotiation_message

        if not email_subject:
            email_subject = subject_seed
        if not email_body and negotiation_message:
            email_body = negotiation_message

        for draft_record in draft_records:
            if not isinstance(draft_record, dict):
                continue
            if not draft_record.get("html"):
                draft_record["html"] = self._build_enhanced_html_email(
                    round_number=round_no,
                    contact_name=contact_name,
                    supplier_name=supplier_name,
                    decision=decision,
                    negotiation_message=negotiation_message or "",
                    currency=currency,
                    playbook_context=playbook_context,
                    sender_name=getattr(
                        getattr(self.agent_nick, "settings", None),
                        "sender_name",
                        None,
                    ),
                )
            self._capture_email_to_history(
                workflow_id=workflow_id,
                supplier_id=supplier,
                round_number=round_no,
                draft=draft_record,
                decision=decision,
                state=state,
            )

        email_thread_history = self._get_email_thread_history(workflow_id, supplier)
        email_thread_summary = self._get_email_thread_summary(workflow_id, supplier)
        state["email_history"] = email_thread_history

        final_thread_headers = draft_payload.get("thread_headers")
        if isinstance(final_thread_headers, dict):
            if thread_state:
                subject_hint = final_thread_headers.get("Subject")
                base_subject_hint = self._normalise_base_subject(subject_hint)
                if base_subject_hint:
                    thread_state.subject_base = base_subject_hint
            if not sent_message_id:
                sent_message_id = self._coerce_text(
                    final_thread_headers.get("Message-ID")
                    or final_thread_headers.get("message_id")
                )
        elif isinstance(final_thread_headers, str) and not sent_message_id:
            sent_message_id = self._coerce_text(final_thread_headers)

        self._capture_email_to_history(
            workflow_id=identifier.workflow_id,
            supplier_id=supplier,
            state=state,
            draft_records=draft_records,
            subject=email_subject or draft_payload.get("subject"),
            body=email_body or negotiation_message,
            thread_headers=final_thread_headers,
            email_action_id=email_action_id,
            message_id=sent_message_id,
            recipients=draft_payload.get("recipients") or recipients,
            cc=draft_payload.get("cc"),
            sender=(
                draft_payload.get("from_address")
                or draft_payload.get("sender")
                or context.input_data.get("sender")
            ),
            session_reference=session_reference,
        )

        if sent_message_id and thread_state:
            thread_state.update_after_send(sent_message_id)

        state["status"] = "ACTIVE"
        state["awaiting_response"] = True
        state["last_email_sent_at"] = datetime.now(timezone.utc)
        if email_action_id:
            state["last_agent_msg_id"] = email_action_id
        self._persist_thread_state(state, thread_state)
        self._save_session_state(identifier.workflow_id, supplier, state)
        self._record_learning_snapshot(
            context,
            identifier.workflow_id or state_identifier or session_id,
            supplier,
            decision,
            state,
            True,
            supplier_reply_registered,
            rfq_id=rfq_value,
        )
        if self.response_matcher and supplier:
            try:
                self.response_matcher.register_expected_response(
                    identifier=NegotiationIdentifier(
                        workflow_id=identifier.workflow_id,
                        session_reference=session_reference,
                        supplier_id=supplier,
                        round_number=round_no,
                    ),
                    email_action_id=email_action_id,
                    expected_unique_id=session_reference,
                    timeout_seconds=int(context.input_data.get("response_timeout", 900))
                    if isinstance(context.input_data, dict)
                    else 900,
                )
            except Exception:
                logger.debug("Failed to register expected response", exc_info=True)
        cache_key: Optional[Tuple[str, str]] = None
        if state_identifier and supplier:
            cache_key = (str(state_identifier), str(supplier))
        cached_state = self._get_cached_state(cache_key)
        public_state = self._public_state(cached_state or state)
        data = {
            "supplier": supplier,
            "rfq_id": rfq_value,
            "session_reference": session_reference,
            "unique_id": session_reference,
            "round": round_no,
            "counter_proposals": counter_options,
            "decision": decision,
            "savings_score": savings_score,
            "decision_log": decision_log,
            "message": negotiation_message,
            "email_subject": None,
            "email_body": None,
            "supplier_snippets": supplier_snippets,
            "negotiation_allowed": True,
            "interaction_type": "negotiation",
            "intent": "NEGOTIATION_COUNTER",
            "draft_payload": draft_payload,
            "drafts": draft_records,
            "session_state": public_state,
            "currency": currency,
            "current_offer": price,
            "target_price": target_price,
            "supplier_message": supplier_message,
            "sent_status": False,
            "awaiting_response": True,
            "play_recommendations": play_recommendations,
            "playbook_descriptor": playbook_context.get("descriptor"),
            "playbook_examples": playbook_context.get("examples"),
            "positions": decision.get("positions"),
            "validation_issues": decision.get("validation_issues"),
            "outlier_alerts": decision.get("outlier_alerts"),
            "flags": decision.get("flags"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
            "normalised_inputs": normalised_inputs,
            "thread_state": thread_state.as_dict() if thread_state else None,
            "email_history": email_thread_history,
            "email_thread_summary": email_thread_summary,
            "current_email": draft_records[0] if draft_records else None,
            "all_emails_sent": len(email_thread_history),
        }

        history_payload, history_summary, all_sent = self._compose_email_history_payload(
            identifier.workflow_id,
            supplier,
            state,
        )
        data["email_history"] = history_payload
        data["email_thread_summary"] = history_summary
        data["all_emails_sent"] = all_sent

        if email_subject:
            data["email_subject"] = email_subject
        if email_body:
            data["email_body"] = email_body
        if email_action_id:
            data["email_action_id"] = email_action_id
        pass_fields: Dict[str, Any] = dict(data)

        supplier_watch_fields = self._build_supplier_watch_fields(
            context=context,
            workflow_id=workflow_id,
            supplier=supplier,
            drafts=draft_records,
            state=state,
            session_reference=session_reference,
            rfq_id=rfq_value,
        )
        supplier_responses: List[Dict[str, Any]] = []
        if supplier_watch_fields:
            pass_fields.update(supplier_watch_fields)
            await_response = bool(supplier_watch_fields.get("await_response"))
            awaiting_email_drafting = "EmailDraftingAgent" in next_agents
            input_payload = (
                context.input_data if isinstance(context.input_data, dict) else {}
            )
            should_wait = (
                await_response
                and not awaiting_email_drafting
                and not bool(input_payload.get("_batch_execution"))
            )
            if should_wait:
                self._log_response_wait_diagnostics(
                    workflow_id=workflow_id,
                    supplier_id=supplier,
                    drafts=draft_records,
                    watch_payload=supplier_watch_fields,
                )
                wait_results = self._await_supplier_responses(
                    context=context,
                    watch_payload=supplier_watch_fields,
                    state=state,
                )
                watch_unique_ids = [
                    self._coerce_text(entry.get("unique_id"))
                    for entry in supplier_watch_fields.get("drafts", [])
                    if isinstance(entry, dict) and self._coerce_text(entry.get("unique_id"))
                ]
                if wait_results is None:
                    logger.error(
                        "Supplier responses not received before timeout (workflow_id=%s supplier=%s unique_ids=%s)",
                        workflow_id,
                        supplier,
                        watch_unique_ids or None,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_value,
                        "workflow_id": workflow_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "round": round_no,
                        "decision": decision,
                        "message": "Supplier response not received before timeout.",
                        "unique_ids": watch_unique_ids,
                    }
                    return self._with_plan(
                        context,
                        AgentOutput(
                            status=AgentStatus.FAILED,
                            data=error_payload,
                            error="supplier response timeout",
                        ),
                    )
                wait_thread_headers: Optional[Dict[str, Any]] = None
                if isinstance(wait_results, dict):
                    candidate_headers = wait_results.get("thread_headers")
                    if isinstance(candidate_headers, dict) and candidate_headers:
                        wait_thread_headers = dict(candidate_headers)
                elif isinstance(wait_results, list):
                    for candidate in wait_results:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_headers = candidate.get("thread_headers")
                        if isinstance(candidate_headers, dict) and candidate_headers:
                            wait_thread_headers = dict(candidate_headers)
                            break
                if wait_thread_headers:
                    state["last_thread_headers"] = dict(wait_thread_headers)
                    wait_message_id = self._coerce_text(
                        wait_thread_headers.get("Message-ID")
                        or wait_thread_headers.get("message_id")
                    )
                    if wait_message_id:
                        message_id = wait_message_id
                        if thread_state:
                            thread_state.update_after_receive(wait_message_id)
                supplier_responses = [res for res in wait_results if isinstance(res, dict)]
                if not supplier_responses:
                    logger.error(
                        "No supplier responses received while waiting (workflow_id=%s supplier=%s unique_ids=%s)",
                        workflow_id,
                        supplier,
                        watch_unique_ids or None,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_value,
                        "workflow_id": workflow_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "round": round_no,
                        "decision": decision,
                        "message": "Missing supplier responses after wait.",
                        "unique_ids": watch_unique_ids,
                    }
                    return self._with_plan(
                        context,
                        AgentOutput(
                            status=AgentStatus.FAILED,
                            data=error_payload,
                            error="supplier response missing",
                        ),
                    )
                known_ids: Set[str] = set()
                current_message = self._coerce_text(context.input_data.get("message_id"))
                if current_message:
                    known_ids.add(current_message.lower())
                previous_recorded = self._coerce_text(state.get("last_supplier_msg_id"))
                if previous_recorded:
                    known_ids.add(previous_recorded.lower())

                new_responses: List[Dict[str, Any]] = []
                for response in supplier_responses:
                    message_token = self._coerce_text(
                        response.get("message_id") or response.get("id")
                    )
                    if message_token and message_token.lower() in known_ids:
                        continue
                    if message_token:
                        known_ids.add(message_token.lower())
                        if thread_state:
                            thread_state.update_after_receive(message_token)
                    new_responses.append(response)

                increment_count = len(new_responses)
                if increment_count:
                    state["supplier_reply_count"] = int(
                        state.get("supplier_reply_count", 0)
                    ) + increment_count
                    last_reply = new_responses[-1]
                    last_message_id = last_reply.get("message_id") or last_reply.get("id")
                    if last_message_id:
                        state["last_supplier_msg_id"] = last_message_id
                elif not supplier_reply_registered:
                    # Ensure at least the last observed response is recorded.
                    last_reply = supplier_responses[-1]
                    last_message_id = last_reply.get("message_id") or last_reply.get("id")
                    if last_message_id:
                        state["last_supplier_msg_id"] = last_message_id
                supplier_responses = new_responses or supplier_responses
                state["awaiting_response"] = False
                if increment_count or supplier_reply_registered:
                    try:
                        current_round_value = int(state.get("current_round", round_no))
                    except (TypeError, ValueError):
                        current_round_value = round_no
                    if current_round_value <= round_no:
                        state["current_round"] = round_no + 1
                self._persist_thread_state(state, thread_state)
                self._save_session_state(identifier.workflow_id, supplier, state)
                public_state = self._public_state(state)
                data["session_state"] = public_state
                data["awaiting_response"] = False
                pass_fields["session_state"] = public_state
                pass_fields.pop("await_response", None)
                pass_fields.pop("await_all_responses", None)
                self._record_learning_snapshot(
                    context,
                    identifier.workflow_id or state_identifier or session_id,
                    supplier,
                    decision,
                    state,
                    False,
                    bool(new_responses) or supplier_reply_registered,
                    rfq_id=rfq_value,
                )
            else:
                if await_response and "SupplierInteractionAgent" not in next_agents:
                    next_agents.append("SupplierInteractionAgent")
        if supplier_responses:
            data["supplier_responses"] = supplier_responses
            pass_fields["supplier_responses"] = supplier_responses

        if next_agents:
            merge_payload = dict(fallback_payload or {})
            merge_payload.setdefault("intent", "NEGOTIATION_COUNTER")
            merge_payload.setdefault("decision", decision)
            merge_payload["session_state"] = public_state
            if session_reference is not None:
                merge_payload.setdefault("session_reference", session_reference)
            if supplier is not None:
                merge_payload.setdefault("supplier_id", supplier)
            if supplier_name:
                merge_payload.setdefault("supplier_name", supplier_name)
            merge_payload.setdefault("round", round_no)
            merge_payload.setdefault("counter_price", decision.get("counter_price"))
            merge_payload.setdefault("target_price", target_price)
            merge_payload.setdefault("current_offer", price)
            merge_payload.setdefault("current_offer_numeric", price)
            if currency:
                merge_payload.setdefault("currency", currency)
            merge_payload.setdefault("asks", decision.get("asks", []))
            merge_payload.setdefault("lead_time_request", decision.get("lead_time_request"))
            merge_payload.setdefault("rationale", decision.get("rationale"))
            merge_payload.setdefault("negotiation_message", negotiation_message)
            if supplier_message:
                merge_payload.setdefault("supplier_message", supplier_message)
            merge_payload.setdefault("supplier_reply_count", supplier_reply_count)
            if recipients:
                merge_payload.setdefault("recipients", recipients)
            sender = context.input_data.get("sender")
            if sender and not merge_payload.get("sender"):
                merge_payload["sender"] = sender
            if play_recommendations:
                merge_payload.setdefault("play_recommendations", play_recommendations)
            descriptor = playbook_context.get("descriptor")
            if descriptor:
                merge_payload.setdefault("playbook_descriptor", descriptor)

            if counter_options:
                merge_payload.setdefault("counter_options", counter_options)

            for key, value in merge_payload.items():
                data[key] = value

            pass_fields.update(merge_payload)
        logger.info(
            "NegotiationAgent prepared counter round %s for supplier=%s session_reference=%s",
            round_no,
            supplier,
            session_reference,
        )

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=pass_fields,
                next_agents=next_agents,
            ),
        )

    # ------------------------------------------------------------------
    # NEW: Intelligence helpers
    # ------------------------------------------------------------------
    def _extract_negotiation_signals(
        self,
        *,
        supplier_message: Optional[str],
        snippets: List[str],
        market: Dict[str, Any],
        performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = " ".join([t for t in ([supplier_message] + snippets) if t])[:8000]
        signals: Dict[str, Any] = {
            "finality_hint": False,
            "capacity_tight": False,
            "moq": None,
            "payment_terms_hint": None,
            "delivery_flex": None,
            "alt_part_offered": False,
            "tone": "neutral",
            "concession_band_pct": None,
        }

        lowered = text.lower() if text else ""
        if any(p in lowered for p in ("capacity", "backlog", "constrained")):
            signals["capacity_tight"] = True
        if "moq" in lowered:
            m = re.search(r"moq[^0-9]*([0-9]{2,})", lowered)
            if m:
                try:
                    signals["moq"] = int(m.group(1))
                except Exception:
                    pass
        if any(x in lowered for x in ("net-30", "net30", "net 30", "net-45", "net 45", "early payment")):
            signals["payment_terms_hint"] = "tradeable"
        if any(x in lowered for x in ("expedite", "split shipment", "partial", "air freight")):
            signals["delivery_flex"] = "possible"
        if any(x in lowered for x in ("alternate", "alternative", "equivalent", "substitute", "brand b")):
            signals["alt_part_offered"] = True
        if any(x in lowered for x in ("cannot go lower", "final", "last price", "our best price", "rock bottom")):
            signals["finality_hint"] = True
            signals["tone"] = "firm"

        if LLM_ENABLED and text:
            try:  # pragma: no cover - optional dependency
                prompt = (
                    "Extract JSON with keys: tone (firm/flexible/neutral), finality_hint (bool), "
                    "capacity_tight (bool), moq (int or null), payment_terms_hint (tradeable/fixed/null), "
                    "delivery_flex (possible/unlikely/null), concession_band_pct (float or null). Only return JSON.\n\n"
                    f"Text:\n{text}"
                )
                client = get_lmstudio_client()
                resp = client.generate(
                    model=LLM_MODEL,
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.1},
                )
                content = (
                    resp.get("response")
                    or resp.get("message", {}).get("content", "")
                )
                if "{" in content and "}" in content:
                    content = content[content.find("{") : content.rfind("}") + 1]
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        for key in signals:
                            if key in parsed and parsed[key] is not None:
                                signals[key] = parsed[key]
            except Exception:
                logger.debug("LLM signal extraction skipped/failed", exc_info=True)

        signals["market"] = market or {}
        signals["performance"] = performance or {}
        return signals

    def _estimate_zopa(
        self,
        *,
        price: Optional[float],
        target: Optional[float],
        history: Dict[str, Any],
        benchmarks: Dict[str, Any],
        should_cost: Optional[float],
        signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        price_val = self._coerce_float(price)
        target_val = self._coerce_float(target)

        buyer_max = None
        if target_val is not None and price_val is not None:
            buyer_max = min(target_val, price_val)
        elif target_val is not None:
            buyer_max = target_val
        elif price_val is not None:
            buyer_max = price_val

        buyer_max = self._validate_buyer_max(buyer_max)

        candidates: List[float] = []
        if should_cost is not None and should_cost > 0:
            candidates.append(float(should_cost))
        bench_low = self._coerce_float(benchmarks.get("p10") or benchmarks.get("low"))
        if bench_low:
            candidates.append(bench_low)
        hist_min = self._coerce_float(history.get("min_accepted_price"))
        if hist_min:
            candidates.append(hist_min)
        supplier_floor = min(candidates) if candidates else (price * 0.85 if price else None)

        if supplier_floor:
            if signals.get("capacity_tight"):
                supplier_floor *= 1.03
            if signals.get("tone") == "firm":
                supplier_floor *= 1.01

        concession = signals.get("concession_band_pct") or 0.05
        entry = None
        if price:
            entry = round(price * (1 - max(0.03, min(0.12, concession))), 2)

        return {
            "buyer_max": float(buyer_max) if buyer_max is not None else None,
            "supplier_floor": float(supplier_floor) if supplier_floor is not None else None,
            "entry_counter": entry,
        }

    def _adaptive_strategy(
        self,
        *,
        base_decision: Dict[str, Any],
        zopa: Dict[str, Any],
        signals: Dict[str, Any],
        round_hint: Any,
        lead_weeks: Optional[float],
        target_price: Optional[float],
        price: Optional[float],
    ) -> Dict[str, Any]:
        decision = dict(base_decision)
        price_locked = bool(base_decision.get("price_plan_locked"))
        current_round = int(round_hint) if isinstance(round_hint, (int, float)) else 1

        if signals.get("finality_hint"):
            decision["strategy"] = "package-trade"
            decision["asks"] = (decision.get("asks") or []) + [
                "Consider payment-terms trade (early pay discount vs Net45/60)",
                "Bundle volume commitment for stepped pricing",
            ]
            if lead_weeks and lead_weeks > 3:
                decision["lead_time_request"] = "Split shipment or expedite slot if possible"
            return decision

        buyer_max = self._validate_buyer_max(zopa.get("buyer_max"))
        supplier_floor = self._coerce_float(zopa.get("supplier_floor"))
        entry = self._coerce_float(zopa.get("entry_counter"))
        if not price_locked and price and target_price and entry:
            if current_round == 1:
                decision["counter_price"] = min(entry, (price + target_price) / 2)
            else:
                if (
                    buyer_max is not None
                    and supplier_floor is not None
                    and supplier_floor < buyer_max
                ):
                    midpoint = (buyer_max + supplier_floor) / 2
                    decision["counter_price"] = round(
                        min(decision.get("counter_price") or entry, midpoint), 2
                    )
                else:
                    decision["counter_price"] = round(
                        min(decision.get("counter_price") or entry, (price + target_price) / 2), 2
                    )

        asks = decision.get("asks", [])
        if signals.get("payment_terms_hint") == "tradeable":
            asks.append("Offer early payment for additional 1–2% discount")
        if signals.get("delivery_flex") == "possible" and (lead_weeks or 0) > 3:
            asks.append("Split shipment or expedite window swap")
        decision["asks"] = list(dict.fromkeys(asks))
        decision["price_plan_locked"] = price_locked
        return decision

    def _optimize_multi_issue(
        self,
        *,
        price: Optional[float],
        currency: Optional[str],
        target: Optional[float],
        lead_weeks: Optional[float],
        weights: Dict[str, Any],
        policy: Dict[str, Any],
        constraints: Dict[str, Any],
        zopa: Dict[str, Any],
        signals: Dict[str, Any],
        round_no: int,
    ) -> Dict[str, Any]:
        w_price = float(weights.get("price", 0.5))
        w_delivery = float(weights.get("delivery", 0.2))
        w_risk = float(weights.get("risk", 0.2))
        w_terms = float(weights.get("terms", 0.1))
        normalizer = max(w_price + w_delivery + w_risk + w_terms, 1e-9)
        w_price, w_delivery, w_risk, w_terms = [w / normalizer for w in (w_price, w_delivery, w_risk, w_terms)]

        _ = policy, constraints  # reserved for future constraint-aware adjustments

        buyer_max = self._validate_buyer_max(zopa.get("buyer_max"))
        supplier_floor = self._coerce_float(zopa.get("supplier_floor"))
        entry = self._coerce_float(zopa.get("entry_counter"))

        price_candidates: List[float] = []
        if target and price:
            price_candidates.extend(
                [
                    round(max(supplier_floor or 0, target * 0.98), 2),
                    round(max(supplier_floor or 0, (price + target) / 2 * 0.97), 2),
                ]
            )
        if entry:
            price_candidates.append(round(entry, 2))
        if price and target:
            price_candidates.append(
                round(max(supplier_floor or 0, price * (1 - AGGRESSIVE_FIRST_COUNTER_PCT)), 2)
            )

        sanitized: List[float] = []
        for candidate in price_candidates:
            if candidate is None:
                continue
            value = candidate
            if buyer_max is not None and value > buyer_max:
                value = buyer_max
            if supplier_floor is not None and value < supplier_floor:
                value = supplier_floor
            sanitized.append(round(value, 2))
        price_candidates = sorted(set(sanitized))

        daily_rate = COST_OF_CAPITAL_APR / 365.0
        term_pkgs = [
            {"label": "Net15 with 2% disc", "apr_equiv": -0.02},
            {"label": "Net30 std", "apr_equiv": 0.0},
            {"label": "Net45 std", "apr_equiv": daily_rate * 15},
            {"label": "Net60 std", "apr_equiv": daily_rate * 30},
        ]

        if lead_weeks is not None and lead_weeks > 3:
            lead_packages = [{"label": "≤2w or split shipment", "weeks_gain": (lead_weeks - 2)}]
        else:
            lead_packages = [{"label": "as quoted", "weeks_gain": 0}]

        volume_tiers = [None, 100, 250, 500]

        option_grid: List[Dict[str, Any]] = []
        for price_option in price_candidates[:5]:
            for term_option in term_pkgs:
                for lead_option in lead_packages:
                    for tier in volume_tiers:
                        option_grid.append(
                            {
                                "price": price_option,
                                "terms": term_option,
                                "lead": lead_option,
                                "volume": tier,
                            }
                        )

        best_option = None
        best_score = -1e9
        for option in option_grid:
            counter_price = option["price"]
            price_score = 0.0
            if target and counter_price:
                price_score = (target - counter_price) / max(target, 1e-9)

            terms_score = -float(option["terms"]["apr_equiv"])

            delivery_score = 0.0
            weeks_gain = float(option["lead"].get("weeks_gain") or 0)
            delivery_score = weeks_gain * LEAD_TIME_VALUE_PCT_PER_WEEK

            risk_score = 0.0
            performance = signals.get("performance") or {}
            on_time = self._coerce_float(
                performance.get("on_time_delivery")
                or performance.get("delivery_score")
                or performance.get("otif")
            )
            if on_time is not None and on_time < 0.9:
                risk_score += 0.02
            if signals.get("capacity_tight") and "split" in option["lead"]["label"].lower():
                risk_score += 0.02

            total_score = (
                (w_price * price_score)
                + (w_terms * terms_score)
                + (w_delivery * delivery_score)
                + (w_risk * risk_score)
            )

            if total_score > best_score:
                best_score = total_score
                best_option = option

        decision_overrides: Dict[str, Any] = {}
        counter_options: List[Dict[str, Any]] = []
        if best_option:
            decision_overrides["counter_price"] = round(best_option["price"], 2)
            asks = decision_overrides.get("asks", [])
            if "≤2w" in best_option["lead"]["label"]:
                asks.append("≤ 2 weeks or split shipment (20–30% now, balance later)")
            if "2% disc" in best_option["terms"]["label"]:
                asks.append("2% discount for Net15 / early payment")
            if best_option["volume"]:
                asks.append(f"Tiers @ {best_option['volume']}/250/500 for stepped pricing")
            decision_overrides["asks"] = list(dict.fromkeys(asks))
            counter_options.append(
                {
                    "price": round(best_option["price"], 2),
                    "terms": best_option["terms"]["label"],
                    "bundle": {
                        "lead_time": best_option["lead"]["label"],
                        "volume_tier": best_option["volume"],
                    },
                }
            )

        return {"decision_overrides": decision_overrides, "counter_options": counter_options}

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------
    def _default_state(self) -> Dict[str, Any]:
        return {
            "supplier_reply_count": 0,
            "current_round": 1,
            "status": "ACTIVE",
            "awaiting_response": False,
            "last_supplier_msg_id": None,
            "last_agent_msg_id": None,
            "last_email_sent_at": None,
            "base_subject": None,
            "initial_body": None,
            "last_thread_headers": None,
            "positions": {},
            "last_counter_price": None,
            "thread_state": None,
            "workflow_id": None,
            "session_reference": None,
            "email_history": [],
        }

    def _public_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "supplier_reply_count": int(state.get("supplier_reply_count", 0)),
            "current_round": int(state.get("current_round", 1)),
            "status": state.get("status", "ACTIVE"),
            "awaiting_response": bool(state.get("awaiting_response", False)),
        }

    def _load_thread_state(
        self,
        identifier: NegotiationIdentifier,
        state: Dict[str, Any],
        *,
        subject_hint: Optional[str] = None,
    ) -> EmailThreadState:
        fallback_subject = self._normalise_base_subject(state.get("base_subject"))
        if not fallback_subject:
            fallback_subject = self._normalise_base_subject(subject_hint)
        if not fallback_subject:
            fallback_subject = DEFAULT_NEGOTIATION_SUBJECT

        thread_payload = state.get("thread_state")
        if isinstance(thread_payload, dict) and thread_payload:
            thread_state = EmailThreadState.from_dict(
                thread_payload, fallback_subject=fallback_subject
            )
        else:
            thread_state = EmailThreadState(
                thread_id=f"<{uuid.uuid4()}@procwise.co.uk>",
                subject_base=fallback_subject,
            )

        if not thread_state.subject_base:
            thread_state.subject_base = fallback_subject

        if not thread_state.references:
            base_reference = state.get("last_agent_msg_id") or state.get("last_supplier_msg_id")
            token = EmailThreadState._normalise_token(base_reference)
            if token and token not in thread_state.references:
                thread_state.references.append(token)

        state["thread_state"] = thread_state.as_dict()
        state["workflow_id"] = identifier.workflow_id
        state["session_reference"] = identifier.session_reference
        return thread_state

    @staticmethod
    def _persist_thread_state(
        state: Dict[str, Any], thread_state: Optional[EmailThreadState]
    ) -> None:
        if thread_state is None:
            return
        state["thread_state"] = thread_state.as_dict()

    def _rehydrate_email_history(
        self,
        *,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        history: Any,
    ) -> None:
        workflow_key = self._coerce_text(workflow_id)
        supplier_key = self._coerce_text(supplier_id)
        if not workflow_key or not supplier_key:
            return
        if not isinstance(history, list):
            return

        entries: List[EmailHistoryEntry] = []
        for item in history:
            if isinstance(item, EmailHistoryEntry):
                entries.append(item)
            elif isinstance(item, dict):
                try:
                    entries.append(EmailHistoryEntry.from_dict(item))
                except Exception:
                    continue

        if entries:
            self._email_thread_manager.set_thread(workflow_key, supplier_key, entries)

    def _get_cached_state(
        self, cache_key: Optional[Tuple[str, str]]
    ) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        with self._state_lock:
            cached = self._state_cache.get(cache_key)
            return dict(cached) if isinstance(cached, dict) else None

    def _ensure_state_schema(self) -> None:
        if getattr(self, "_state_schema_checked", False):
            return
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            self._state_schema_checked = True
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.negotiation_session_state (
                            workflow_id VARCHAR(255),
                            supplier_id VARCHAR(255) NOT NULL,
                            supplier_reply_count INTEGER DEFAULT 0,
                            current_round INTEGER DEFAULT 1,
                            status VARCHAR(64),
                            awaiting_response BOOLEAN DEFAULT FALSE,
                            last_supplier_msg_id TEXT,
                            last_agent_msg_id TEXT,
                            last_email_sent_at TIMESTAMPTZ,
                            base_subject TEXT,
                            initial_body TEXT,
                            thread_state JSONB,
                            email_history JSONB,
                            session_reference VARCHAR(255),
                            unique_id VARCHAR(255),
                            rfq_id TEXT,
                            updated_on TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS base_subject TEXT"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS initial_body TEXT"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS thread_state JSONB"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS email_history JSONB"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255)"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS unique_id VARCHAR(255)"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS session_reference VARCHAR(255)"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS rfq_id TEXT"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS email_history JSONB"
                    )
                    # Deprecate rfq_id requirements in favour of workflow/unique identifiers
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state DROP CONSTRAINT IF EXISTS negotiation_session_state_pk"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state DROP CONSTRAINT IF EXISTS negotiation_session_state_pkey"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ALTER COLUMN rfq_id DROP NOT NULL"
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_session_state_workflow_supplier
                            ON proc.negotiation_session_state (workflow_id, supplier_id)
                            WHERE workflow_id IS NOT NULL
                        """
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS negotiation_session_state_unique_supplier_idx
                            ON proc.negotiation_session_state (unique_id, supplier_id)
                            WHERE unique_id IS NOT NULL
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_negotiation_state_email_history
                            ON proc.negotiation_session_state USING GIN(email_history)
                        """
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.debug("failed to ensure negotiation state schema", exc_info=True)
        finally:
            setattr(self, "_proc_negotiation_session_state_column_metadata", None)
            self._state_schema_checked = True

    def _ensure_sessions_schema(self) -> None:
        if getattr(self, "_sessions_schema_checked", False):
            return
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            self._sessions_schema_checked = True
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.negotiation_sessions (
                            workflow_id VARCHAR(255),
                            unique_id VARCHAR(255),
                            supplier_id VARCHAR(255) NOT NULL,
                            round INTEGER NOT NULL DEFAULT 1,
                            counter_offer NUMERIC,
                            created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            rfq_id TEXT
                        )
                        """
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_sessions ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255)"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_sessions ADD COLUMN IF NOT EXISTS unique_id VARCHAR(255)"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_sessions ADD COLUMN IF NOT EXISTS rfq_id TEXT"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_sessions DROP CONSTRAINT IF EXISTS negotiation_sessions_pk"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_sessions DROP CONSTRAINT IF EXISTS negotiation_sessions_pkey"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_sessions ALTER COLUMN rfq_id DROP NOT NULL"
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS uq_negotiation_sessions_workflow_supplier_round
                            ON proc.negotiation_sessions (workflow_id, supplier_id, round)
                            WHERE workflow_id IS NOT NULL
                        """
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS negotiation_sessions_unique_supplier_round_idx
                            ON proc.negotiation_sessions (unique_id, supplier_id, round)
                            WHERE unique_id IS NOT NULL
                        """
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.debug("failed to ensure negotiation sessions schema", exc_info=True)
        finally:
            setattr(self, "_proc_negotiation_sessions_column_metadata", None)
            self._negotiation_sessions_identifier_column = None
            self._sessions_schema_checked = True

    def _get_identifier_column(
        self,
        table: str,
        *,
        default: str = "workflow_id",
        fallback: str = "unique_id",
        alt_fallback: str = "session_id",
    ) -> str:
        attr_name = f"_{table}_identifier_column"
        cached = getattr(self, attr_name, None)
        if isinstance(cached, str) and cached:
            return cached

        column = default
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if callable(get_conn):
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT column_name
                              FROM information_schema.columns
                             WHERE table_schema = 'proc'
                               AND table_name = %s
                               AND column_name = %s
                            """,
                            (table, default),
                        )
                        if cur.fetchone():
                            column = default
                        else:
                            cur.execute(
                                """
                                SELECT column_name
                                  FROM information_schema.columns
                                 WHERE table_schema = 'proc'
                                   AND table_name = %s
                                   AND column_name = %s
                                """,
                                (table, fallback),
                            )
                            if cur.fetchone():
                                column = fallback
                            else:
                                cur.execute(
                                    """
                                    SELECT column_name
                                      FROM information_schema.columns
                                     WHERE table_schema = 'proc'
                                       AND table_name = %s
                                       AND column_name = %s
                                    """,
                                    (table, alt_fallback),
                                )
                                if cur.fetchone():
                                    column = alt_fallback
                    conn.commit()
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "failed to detect identifier column for %s", table, exc_info=True
                )

        setattr(self, attr_name, column)
        return column

    def _get_column_metadata(
        self,
        table: str,
        *,
        schema: str = "proc",
    ) -> Dict[str, Dict[str, Any]]:
        attr_name = f"_{schema}_{table}_column_metadata"
        cached = getattr(self, attr_name, None)
        if isinstance(cached, dict) and cached:
            return cached

        metadata: Dict[str, Dict[str, Any]] = {}
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if callable(get_conn):
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT column_name, is_nullable
                              FROM information_schema.columns
                             WHERE table_schema = %s
                               AND table_name = %s
                            """,
                            (schema, table),
                        )
                        for name, is_nullable in cur.fetchall() or []:
                            nullable = (is_nullable or "").upper() != "NO"
                            metadata[name] = {"nullable": nullable}
                conn.commit()
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "failed to load column metadata for %s.%s", schema, table, exc_info=True
                )

        setattr(self, attr_name, metadata)
        return metadata

    def _get_unique_constraint(
        self,
        table: str,
        *,
        preferred: Sequence[str],
        fallbacks: Sequence[Sequence[str]],
    ) -> UniqueConstraintInfo:
        attr_name = f"_{table}_unique_constraint"
        cached = getattr(self, attr_name, None)
        if isinstance(cached, UniqueConstraintInfo) and cached.columns:
            return cached
        if isinstance(cached, (list, tuple)) and cached:
            cached_info = UniqueConstraintInfo(columns=tuple(cached))
            setattr(self, attr_name, cached_info)
            return cached_info

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        candidates: List[UniqueConstraintInfo] = []
        if callable(get_conn):
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT constraint_name
                              FROM information_schema.table_constraints
                             WHERE table_schema = 'proc'
                               AND table_name = %s
                               AND constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                            """,
                            (table,),
                        )
                        constraints = [row[0] for row in cur.fetchall() or []]
                        for constraint in constraints:
                            cur.execute(
                                """
                                SELECT column_name
                                  FROM information_schema.key_column_usage
                                 WHERE table_schema = 'proc'
                                   AND table_name = %s
                                   AND constraint_name = %s
                                 ORDER BY ordinal_position
                                """,
                                (table, constraint),
                            )
                            cols = [row[0] for row in cur.fetchall() or []]
                            if cols:
                                candidates.append(
                                    UniqueConstraintInfo(
                                        columns=tuple(cols),
                                        constraint_name=constraint,
                                    )
                                )

                        cur.execute(
                            """
                            SELECT indexname, indexdef
                              FROM pg_indexes
                             WHERE schemaname = 'proc'
                               AND tablename = %s
                            """,
                            (table,),
                        )
                        for index_name, index_def in cur.fetchall() or []:
                            if not index_def:
                                continue
                            if "CREATE UNIQUE INDEX" not in index_def.upper():
                                continue

                            columns: List[str] = []
                            predicate: Optional[str] = None

                            match = re.search(
                                r"USING\s+\w+\s*\((?P<columns>[^\)]+)\)",
                                index_def,
                                re.IGNORECASE,
                            )
                            if not match:
                                continue
                            column_segment = match.group("columns")

                            def _strip_wrapping_parentheses(value: str) -> str:
                                text = value.strip()
                                while text.startswith("(") and text.endswith(")"):
                                    depth = 0
                                    balanced = True
                                    for i, char in enumerate(text):
                                        if char == "(":
                                            depth += 1
                                        elif char == ")":
                                            depth -= 1
                                            if depth < 0:
                                                balanced = False
                                                break
                                            if depth == 0 and i < len(text) - 1:
                                                balanced = False
                                                break
                                    if not balanced:
                                        break
                                    text = text[1:-1].strip()
                                return text

                            for part in column_segment.split(","):
                                cleaned = part.strip()
                                if not cleaned:
                                    continue
                                cleaned = _strip_wrapping_parentheses(cleaned)
                                cleaned = re.sub(
                                    r"\s+(ASC|DESC|NULLS\s+(FIRST|LAST))$",
                                    "",
                                    cleaned,
                                    flags=re.IGNORECASE,
                                )
                                if cleaned.startswith('"') and cleaned.endswith('"'):
                                    cleaned = cleaned[1:-1]
                                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cleaned):
                                    columns.append(cleaned)
                                else:
                                    columns = []
                                    break

                            if not columns:
                                continue

                            predicate_match = re.search(
                                r"\)\s*WHERE\s+(?P<predicate>.+)$",
                                index_def,
                                re.IGNORECASE,
                            )
                            if predicate_match:
                                predicate = _strip_wrapping_parentheses(
                                    predicate_match.group("predicate")
                                ).rstrip(";").strip() or None

                            candidates.append(
                                UniqueConstraintInfo(
                                    columns=tuple(columns),
                                    predicate=predicate,
                                    index_name=index_name,
                                )
                            )
                conn.commit()
            except Exception:  # pragma: no cover - diagnostic only
                logger.debug(
                    "Failed to introspect unique constraints for %s", table, exc_info=True
                )

        def _match(target: Sequence[str]) -> Optional[UniqueConstraintInfo]:
            normalised = tuple(target)
            best: Optional[UniqueConstraintInfo] = None
            for candidate in candidates:
                if candidate.columns == normalised:
                    if candidate.predicate:
                        return candidate
                    if best is None:
                        best = candidate
            return best

        preferred_match = _match(preferred)
        if preferred_match:
            setattr(self, attr_name, preferred_match)
            return preferred_match

        for fallback in fallbacks:
            match = _match(fallback)
            if match:
                setattr(self, attr_name, match)
                return match

        if candidates:
            setattr(self, attr_name, candidates[0])
            return candidates[0]

        fallback_info = UniqueConstraintInfo(columns=tuple(preferred))
        setattr(self, attr_name, fallback_info)
        return fallback_info

    @staticmethod
    def _normalise_base_subject(subject: Optional[str]) -> Optional[str]:
        if subject is None:
            return None
        if not isinstance(subject, str):
            try:
                subject = str(subject)
            except Exception:
                return None
        trimmed = subject.strip()
        if not trimmed:
            return None
        trimmed = re.sub(r"(?i)^(re|fw|fwd):\s*", "", trimmed)
        cleaned = EmailDraftingAgent._strip_rfq_identifier_tokens(trimmed)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("-–: ")
        return cleaned or None

    def _format_negotiation_subject(self, state: Dict[str, Any]) -> str:
        base = self._coerce_text(state.get("base_subject"))
        if base:
            if re.match(r"(?i)^(re|fw|fwd):", base.strip()):
                return base.strip()
            return f"Re: {base}".strip()
        return DEFAULT_NEGOTIATION_SUBJECT


    def _load_session_state(
        self, session_id: Optional[str], supplier: Optional[str]
    ) -> Tuple[Dict[str, Any], bool]:
        workflow_id = self._coerce_text(session_id)
        supplier_id = self._coerce_text(supplier)
        if not workflow_id or not supplier_id:
            logger.error(
                "cannot load negotiation session state without workflow_id and supplier"
            )
            return self._default_state(), False
        self._ensure_state_schema()
        column_metadata = self._get_column_metadata("negotiation_session_state")
        if "workflow_id" not in column_metadata:
            logger.error(
                "negotiation_session_state.workflow_id column missing; unable to load state"
            )
            return self._default_state(), False
        key = (workflow_id, supplier_id)
        with self._state_lock:
            if key in self._state_cache:
                return dict(self._state_cache[key]), True

        state = self._default_state()
        exists = False
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT supplier_reply_count, current_round, status, awaiting_response,
                               last_supplier_msg_id, last_agent_msg_id, last_email_sent_at,
                               base_subject, initial_body, thread_state, workflow_id, session_reference,
                               email_history
                          FROM proc.negotiation_session_state
                         WHERE workflow_id = %s AND supplier_id = %s
                        """,
                        (workflow_id, supplier_id),
                    )
                    row = cur.fetchone()
                    if row:
                        (
                            state["supplier_reply_count"],
                            state["current_round"],
                            state["status"],
                            state["awaiting_response"],
                            state["last_supplier_msg_id"],
                            state["last_agent_msg_id"],
                            state["last_email_sent_at"],
                            base_subject,
                            initial_body,
                            thread_state_raw,
                            email_history_raw,
                            workflow_value,
                            session_reference_value,
                            email_history_raw,
                        ) = row
                        if isinstance(base_subject, (bytes, bytearray, memoryview)):
                            try:
                                base_subject = base_subject.decode("utf-8", errors="ignore")
                            except Exception:
                                base_subject = str(base_subject)
                        if isinstance(initial_body, (bytes, bytearray, memoryview)):
                            try:
                                initial_body = initial_body.decode("utf-8", errors="ignore")
                            except Exception:
                                initial_body = str(initial_body)
                        state["base_subject"] = base_subject
                        state["initial_body"] = initial_body
                        if thread_state_raw:
                            if isinstance(thread_state_raw, (bytes, bytearray, memoryview)):
                                try:
                                    thread_state_raw = thread_state_raw.tobytes().decode("utf-8")
                                except Exception:
                                    thread_state_raw = None
                            if isinstance(thread_state_raw, str):
                                try:
                                    state["thread_state"] = json.loads(thread_state_raw)
                                except Exception:
                                    state["thread_state"] = None
                            elif isinstance(thread_state_raw, dict):
                                state["thread_state"] = dict(thread_state_raw)
                        if email_history_raw:
                            parsed_history: List[Dict[str, Any]] = []
                            if isinstance(email_history_raw, (bytes, bytearray, memoryview)):
                                try:
                                    email_history_raw = email_history_raw.tobytes().decode("utf-8")
                                except Exception:
                                    email_history_raw = None
                            if isinstance(email_history_raw, str):
                                try:
                                    parsed_history = json.loads(email_history_raw)
                                except Exception:
                                    parsed_history = []
                            elif isinstance(email_history_raw, list):
                                parsed_history = [
                                    item
                                    for item in email_history_raw
                                    if isinstance(item, dict)
                                ]
                            state["email_history"] = parsed_history
                        if workflow_value:
                            state["workflow_id"] = self._coerce_text(workflow_value)
                        if session_reference_value:
                            state["session_reference"] = self._coerce_text(
                                session_reference_value
                            )
                        if email_history_raw:
                            if isinstance(email_history_raw, (bytes, bytearray, memoryview)):
                                try:
                                    email_history_raw = email_history_raw.tobytes().decode(
                                        "utf-8"
                                    )
                                except Exception:
                                    email_history_raw = None
                            if isinstance(email_history_raw, str):
                                try:
                                    state["email_history"] = json.loads(email_history_raw)
                                except Exception:
                                    state["email_history"] = []
                            elif isinstance(email_history_raw, list):
                                state["email_history"] = list(email_history_raw)
                            elif isinstance(email_history_raw, dict):
                                state["email_history"] = [email_history_raw]
                        else:
                            state.setdefault("email_history", [])
                        exists = True
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to load negotiation session state")
        state.setdefault("workflow_id", workflow_id)
        self._rehydrate_email_history(
            workflow_id=workflow_id,
            supplier_id=supplier_id,
            history=state.get("email_history"),
        )
        with self._state_lock:
            self._state_cache[key] = dict(state)
        return dict(state), exists

    def _save_session_state(
        self, session_id: Optional[str], supplier: Optional[str], state: Dict[str, Any]
    ) -> None:
        workflow_id = self._coerce_text(session_id)
        supplier_id = self._coerce_text(supplier)
        if not workflow_id or not supplier_id:
            logger.error(
                "cannot persist negotiation session state without workflow_id and supplier"
            )
            return
        key = (workflow_id, supplier_id)
        with self._state_lock:
            self._state_cache[key] = dict(state)
        state.setdefault("workflow_id", workflow_id)
        self._ensure_state_schema()
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    column_metadata = self._get_column_metadata(
                        "negotiation_session_state"
                    )
                    if "workflow_id" not in column_metadata:
                        logger.error(
                            "negotiation_session_state.workflow_id column missing; unable to persist state"
                        )
                        conn.rollback()
                        return
                    record: Dict[str, Any] = {
                        "workflow_id": workflow_id,
                        "supplier_id": supplier_id,
                        "supplier_reply_count": int(state.get("supplier_reply_count", 0)),
                        "current_round": int(state.get("current_round", 1)),
                        "status": state.get("status", "ACTIVE"),
                        "awaiting_response": bool(state.get("awaiting_response", False)),
                        "last_supplier_msg_id": state.get("last_supplier_msg_id"),
                        "last_agent_msg_id": state.get("last_agent_msg_id"),
                        "last_email_sent_at": state.get("last_email_sent_at"),
                        "base_subject": state.get("base_subject"),
                        "initial_body": state.get("initial_body"),
                        "updated_on": datetime.now(timezone.utc),
                    }

                    thread_state_payload = state.get("thread_state")
                    if thread_state_payload is not None:
                        try:
                            record["thread_state"] = json.dumps(
                                thread_state_payload, ensure_ascii=False, default=str
                            )
                        except Exception:
                            record["thread_state"] = None

                    email_history_payload = state.get("email_history")
                    if email_history_payload is not None:
                        try:
                            record["email_history"] = json.dumps(
                                email_history_payload, ensure_ascii=False, default=str
                            )
                        except Exception:
                            record["email_history"] = None

                    workflow_value = self._coerce_text(state.get("workflow_id"))
                    if workflow_value:
                        record["workflow_id"] = workflow_value
                    else:
                        record["workflow_id"] = workflow_id

                    session_reference_value = self._coerce_text(
                        state.get("session_reference")
                    )
                    if session_reference_value:
                        record.setdefault("session_reference", session_reference_value)

                    if "unique_id" in column_metadata:
                        unique_value = None
                        for key in (
                            "unique_id",
                            "uniqueId",
                            "session_reference",
                            "sessionReference",
                            "session_id",
                            "sessionId",
                        ):
                            unique_value = self._coerce_text(state.get(key))
                            if unique_value:
                                break
                        if not unique_value:
                            unique_value = session_reference_value or workflow_id

                        unique_nullable = column_metadata["unique_id"].get(
                            "nullable", True
                        )
                        if not unique_nullable:
                            record.setdefault("unique_id", unique_value)
                        elif unique_value:
                            record.setdefault("unique_id", unique_value)

                    unique_info = self._get_unique_constraint(
                        "negotiation_session_state",
                        preferred=("workflow_id", "supplier_id"),
                        fallbacks=(
                            ("workflow_id", "supplier_id"),
                            ("unique_id", "supplier_id"),
                            ("session_id", "supplier_id"),
                        ),
                    )

                    columns = list(record.keys())
                    values = [record[col] for col in columns]
                    unique_columns = unique_info.columns or (
                        "workflow_id",
                        "supplier_id",
                    )
                    conflict_columns = ", ".join(unique_columns)
                    unique_set = set(unique_columns)
                    update_targets = [col for col in columns if col not in unique_set]

                    insert_sql = (
                        f"INSERT INTO proc.negotiation_session_state ({', '.join(columns)}) "
                        f"VALUES ({', '.join(['%s'] * len(columns))})"
                    )
                    if unique_info.predicate and conflict_columns:
                        conflict_target = (
                            f"ON CONFLICT ({conflict_columns}) WHERE {unique_info.predicate}"
                        )
                    elif unique_info.constraint_name:
                        conflict_target = (
                            f"ON CONFLICT ON CONSTRAINT {unique_info.constraint_name}"
                        )
                    elif conflict_columns:
                        conflict_target = f"ON CONFLICT ({conflict_columns})"
                    else:
                        conflict_target = f"ON CONFLICT ({conflict_columns})"

                    if update_targets:
                        update_sql = ", ".join(
                            f"{col} = EXCLUDED.{col}" for col in update_targets
                        )
                        insert_sql = (
                            f"{insert_sql} {conflict_target} DO UPDATE SET {update_sql}"
                        )
                    else:
                        insert_sql = f"{insert_sql} {conflict_target} DO NOTHING"

                    cur.execute(insert_sql, tuple(values))
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist negotiation session state")

    # ------------------------------------------------------------------
    # Negotiation playbook helpers
    # ------------------------------------------------------------------

    def _resolve_playbook_context(
        self, context: AgentContext, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        playbook = self._load_playbook()
        if not playbook:
            return {"plays": [], "lever_priorities": []}

        supplier_type = self._normalise_supplier_type(
            context.input_data.get("supplier_type")
            or context.input_data.get("supplier_segment")
            or decision.get("supplier_type")
        )
        negotiation_style = self._normalise_negotiation_style(
            context.input_data.get("negotiation_style")
            or context.input_data.get("style")
            or decision.get("negotiation_style")
        )

        if not supplier_type or supplier_type not in playbook:
            return {"plays": [], "lever_priorities": []}

        supplier_entry = playbook.get(supplier_type, {})
        styles = supplier_entry.get("styles", {})
        if not negotiation_style or negotiation_style not in styles:
            return {
                "plays": [],
                "descriptor": supplier_entry.get("descriptor"),
                "examples": supplier_entry.get("examples", []),
                "lever_priorities": [],
                "style": negotiation_style,
                "supplier_type": supplier_type,
            }

        lever_priorities = self._resolve_lever_priorities(
            context,
            styles[negotiation_style],
        )

        if not lever_priorities:
            lever_priorities = list(styles[negotiation_style].keys())

        policy_guidance = self._extract_policy_guidance(context)
        supplier_performance = context.input_data.get("supplier_performance")
        if not isinstance(supplier_performance, dict):
            supplier_performance = {}
        market_context = context.input_data.get("market_context")
        if not isinstance(market_context, dict):
            market_context = {}

        plays: List[Dict[str, Any]] = []
        style_plays = styles[negotiation_style]
        for lever in lever_priorities:
            lever_plays = style_plays.get(lever)
            if not isinstance(lever_plays, list):
                continue
            for idx, play_text in enumerate(lever_plays):
                if not isinstance(play_text, str) or not play_text.strip():
                    continue
                base_score = 1.0 + (idx * 0.01)
                policy_score, policy_notes = self._score_policy_alignment(lever, policy_guidance)
                performance_score, performance_notes = self._score_supplier_performance(
                    lever, supplier_performance
                )
                market_score, market_notes = self._score_market_context(lever, market_context)
                total_score = base_score + policy_score + performance_score + market_score
                rationale = self._compose_play_rationale(
                    supplier_entry.get("descriptor"),
                    negotiation_style,
                    lever,
                    policy_notes,
                    performance_notes,
                    market_notes,
                )
                plays.append(
                    {
                        "supplier_type": supplier_type,
                        "style": negotiation_style,
                        "lever": lever,
                        "play": play_text.strip(),
                        "score": round(total_score, 4),
                        "policy_alignment": policy_notes,
                        "performance_signals": performance_notes,
                        "market_signals": market_notes,
                        "rationale": rationale,
                        "trade_offs": TRADE_OFF_HINTS.get(
                            lever, "Monitor implementation impact across stakeholders."
                        ),
                    }
                )

        plays.sort(key=lambda item: (-item["score"], item.get("lever", ""), item.get("play", "")))
        top_plays = plays[:10]

        return {
            "plays": top_plays,
            "descriptor": supplier_entry.get("descriptor"),
            "examples": supplier_entry.get("examples", []),
            "style": negotiation_style,
            "supplier_type": supplier_type,
            "lever_priorities": lever_priorities,
        }

    def _resolve_lever_priorities(self, context: AgentContext, style_plays: Dict[str, Any]) -> List[str]:
        raw_candidates = (
            context.input_data.get("lever_priorities")
            or context.input_data.get("lever_focus")
            or context.input_data.get("lever_preferences")
            or context.input_data.get("preferred_levers")
        )
        priorities: List[str] = []
        for candidate in self._ensure_list(raw_candidates):
            lever = self._normalise_lever_category(candidate)
            if lever and lever in style_plays and lever not in priorities:
                priorities.append(lever)
        if priorities:
            return priorities
        # Fall back to context-provided comma separated string
        if isinstance(raw_candidates, str):
            for chunk in raw_candidates.split(","):
                lever = self._normalise_lever_category(chunk)
                if lever and lever in style_plays and lever not in priorities:
                    priorities.append(lever)
        return priorities

    def _append_playbook_recommendations(
        self,
        summary: str,
        plays: List[Dict[str, Any]],
        playbook_context: Dict[str, Any],
    ) -> str:
        """Append structured playbook recommendations to the negotiation summary."""

        if not plays:
            return summary

        lines: List[str] = [summary.rstrip()]

        recommendation_lines: List[str] = []
        lever_hints: List[str] = []
        for play in plays[:4]:
            if not isinstance(play, dict):
                continue
            lever = str(play.get("lever", "")).strip()
            description = str(play.get("play", "")).strip()
            if description:
                if lever:
                    recommendation_lines.append(f"- {lever}: {description}")
                else:
                    recommendation_lines.append(f"- {description}")
            if lever:
                lookup_key = lever.strip().title()
                hint = TRADE_OFF_HINTS.get(lookup_key)
                if hint:
                    lever_hints.append(f"- {lookup_key}: {hint}")

        if recommendation_lines:
            lines.append("\nRecommended plays:")
            lines.extend(recommendation_lines)

        if lever_hints:
            lines.append("\nTrade-off considerations:")
            lines.extend(lever_hints[:3])

        return "\n".join(lines)

    def _compose_negotiation_message(
        self,
        *,
        context: AgentContext,
        decision: Dict[str, Any],
        positions: NegotiationPositions,
        round_no: int,
        currency: Optional[str],
        supplier: Optional[str],
        supplier_name: Optional[str],
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        zopa: Dict[str, Any],
        playbook_context: Optional[Dict[str, Any]],
        procurement_summary: Optional[Dict[str, Any]],
        contact_name: Optional[str],
    ) -> str:
        """Compose a human-like, strategically crafted negotiation message."""

        counter_price = decision.get("counter_price")
        current_offer = positions.supplier_offer
        target_price = positions.desired
        strategy = decision.get("strategy", "counter")

        # Determine negotiation tone based on round and context
        tone = self._determine_negotiation_tone(round_no, signals, strategy)

        # Build opening that establishes rapport
        opening = self._craft_opening(round_no, supplier_message, signals, tone)

        # Acknowledge supplier's position (reciprocity principle)
        acknowledgment = self._craft_acknowledgment(
            current_offer, supplier_message, signals, currency, round_no
        )

        # Present our position with reasoning (anchoring with justification)
        position_statement = self._craft_position_statement(
            counter_price=counter_price,
            current_offer=current_offer,
            target_price=target_price,
            currency=currency,
            round_no=round_no,
            signals=signals,
            zopa=zopa,
            procurement_summary=procurement_summary,
        )

        # Weave in playbook recommendations naturally
        value_proposition = self._craft_value_proposition(
            playbook_context=playbook_context,
            decision=decision,
            round_no=round_no,
            tone=tone,
        )

        # Create collaborative asks (frame as mutual benefit)
        collaborative_asks = self._craft_collaborative_asks(
            decision=decision,
            round_no=round_no,
            signals=signals,
            tone=tone,
        )

        # Closing that maintains momentum
        closing = self._craft_closing(round_no, strategy, tone)

        # Assemble the complete message
        message_parts = [
            opening,
            acknowledgment,
            position_statement,
            value_proposition,
            collaborative_asks,
            closing,
        ]

        return "\n\n".join(part for part in message_parts if part)

    def _determine_negotiation_tone(
        self, round_no: int, signals: Dict[str, Any], strategy: str
    ) -> str:
        """Determine the appropriate negotiation tone."""

        if strategy in ("accept", "decline"):
            return "decisive"

        if signals.get("finality_hint"):
            return "collaborative-firm"

        if signals.get("capacity_tight"):
            return "understanding-assertive"

        if round_no == 1:
            return "exploratory-confident"
        elif round_no == 2:
            return "focused-collaborative"
        else:
            return "closure-oriented"

    def _craft_opening(
        self,
        round_no: int,
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        tone: str,
    ) -> str:
        """Craft a rapport-building opening."""

        # Acknowledge their response if this isn't the first round
        if round_no > 1 and supplier_message:
            if signals.get("finality_hint"):
                return (
                    "Thank you for taking the time to provide such a detailed response. "
                    "I appreciate your transparency about your position, and I'd like to explore "
                    "whether there's a way we can structure this to work for both parties."
                )
            elif signals.get("capacity_tight"):
                return (
                    "I appreciate you sharing the challenges around capacity and lead times. "
                    "Understanding your operational constraints helps us find creative solutions "
                    "that work within those parameters."
                )
            else:
                return (
                    "Thank you for your continued engagement on this. I've reviewed your proposal "
                    "and believe we're making good progress toward an agreement that benefits both sides."
                )

        # First round opening
        if tone == "exploratory-confident":
            return (
                "Thank you for submitting your proposal. I've had a chance to review the details "
                "and would like to discuss how we might align this with our project requirements "
                "and budget parameters."
            )

        return "I'd like to continue our discussion on finding the right structure for this partnership."

    def _craft_acknowledgment(
        self,
        current_offer: Optional[float],
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        currency: Optional[str],
        round_no: int,
    ) -> str:
        """Acknowledge supplier's position (builds reciprocity)."""

        if round_no == 1 or not supplier_message:
            return ""

        acknowledgments: List[str] = []

        # Acknowledge specific points they raised
        if signals.get("capacity_tight"):
            acknowledgments.append(
                "I understand the capacity constraints you mentioned, and we're certainly "
                "mindful of the current market dynamics affecting lead times."
            )

        if signals.get("moq"):
            moq_value = signals.get("moq")
            acknowledgments.append(
                f"Regarding your minimum order quantity of {moq_value} units, "
                "we're confident our volumes can support that threshold."
            )

        if signals.get("payment_terms_hint"):
            acknowledgments.append(
                "Your flexibility on payment terms is noted and appreciated—"
                "that's definitely an area where we can find mutual value."
            )

        if not acknowledgments and current_offer:
            offer_text = self._format_currency(current_offer, currency)
            acknowledgments.append(
                f"Your quoted price of {offer_text} reflects the quality and service level "
                "you bring, which we certainly value."
            )

        return " ".join(acknowledgments) if acknowledgments else ""

    def _craft_position_statement(
        self,
        *,
        counter_price: Optional[float],
        current_offer: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        signals: Dict[str, Any],
        zopa: Dict[str, Any],
        procurement_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Present our position with strategic justification."""

        if counter_price is None:
            return (
                "Before we can move forward, I'd like to better understand the cost drivers "
                "behind your proposal. Could you provide a breakdown of the key components "
                "affecting the pricing structure?"
            )

        counter_text = self._format_currency(counter_price, currency)
        offer_text = self._format_currency(current_offer, currency) if current_offer else ""

        # Build justification based on available data
        justifications: List[str] = []

        # Use market intelligence if available
        supplier_floor = zopa.get("supplier_floor")

        if supplier_floor and current_offer:
            try:
                market_gap_pct = (current_offer - supplier_floor) / supplier_floor
                if market_gap_pct > 0.15:
                    justifications.append(
                        "Based on our market analysis and benchmarking across similar products, "
                        "we're seeing a meaningful opportunity to optimize the pricing structure"
                    )
            except (TypeError, ZeroDivisionError):
                pass

        # Use procurement history if available
        if procurement_summary and procurement_summary.get("metrics"):
            metrics = procurement_summary["metrics"]
            avg_value = metrics.get("average_transaction_value")
            if avg_value and counter_price:
                try:
                    if abs(counter_price - avg_value) / avg_value < 0.20:
                        avg_text = self._format_currency(avg_value, currency)
                        justifications.append(
                            f"Our historical spend on comparable items has averaged around {avg_text}, "
                            "which helps frame our expectations for this engagement"
                        )
                except (TypeError, ZeroDivisionError):
                    pass

        # Frame based on round
        if round_no == 1:
            intro = (
                f"To align with our project budget and market benchmarks, "
                f"I'd like to propose we structure this around {counter_text}. "
            )
        elif round_no == 2:
            gap_direction = "narrowing" if current_offer and counter_price and current_offer > counter_price else "refining"
            intro = (
                f"As we continue {gap_direction} the gap, I believe {counter_text} represents "
                f"a fair midpoint that reflects both the value you're delivering and our budget realities. "
            )
        else:  # Round 3+
            intro = (
                f"To help us reach closure, I'm proposing {counter_text} as our target. "
                f"This represents our best position given the total business case, "
            )

        # Add strongest justification
        if justifications:
            position = intro + justifications[0] + "."
        else:
            position = intro + "and I believe it sets us up for a successful long-term relationship."

        # Add gap context if material
        if current_offer and counter_price and target_price:
            try:
                movement_from_offer = ((current_offer - counter_price) / current_offer) * 100
                if movement_from_offer > 8:
                    position += (
                        f" This represents a meaningful movement from your {offer_text} quote, "
                        f"and demonstrates our commitment to finding common ground."
                    )
            except (TypeError, ZeroDivisionError):
                pass

        return position

    def _craft_value_proposition(
        self,
        *,
        playbook_context: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
        round_no: int,
        tone: str,
    ) -> str:
        """Weave playbook recommendations into natural value propositions."""

        if not playbook_context or not playbook_context.get("plays"):
            return self._craft_generic_value_proposition(decision, round_no)

        plays = playbook_context["plays"][:3]  # Top 3 plays
        supplier_type = playbook_context.get("supplier_type")

        value_statements: List[str] = []

        # Group plays by lever for coherent messaging
        plays_by_lever: Dict[str, List[Dict[str, Any]]] = {}
        for play in plays:
            lever = play.get("lever", "Other")
            if lever not in plays_by_lever:
                plays_by_lever[lever] = []
            plays_by_lever[lever].append(play)

        # Craft integrated value propositions
        for lever, lever_plays in list(plays_by_lever.items())[:2]:  # Focus on top 2 levers
            if lever == "Commercial":
                value_statements.append(
                    self._weave_commercial_play(lever_plays, round_no, tone)
                )
            elif lever == "Operational":
                value_statements.append(
                    self._weave_operational_play(lever_plays, round_no, tone)
                )
            elif lever == "Strategic":
                value_statements.append(
                    self._weave_strategic_play(lever_plays, round_no, tone)
                )
            elif lever == "Risk":
                value_statements.append(
                    self._weave_risk_play(lever_plays, round_no, tone)
                )
            elif lever == "Relational":
                value_statements.append(
                    self._weave_relational_play(lever_plays, round_no, tone)
                )

        if not value_statements:
            return self._craft_generic_value_proposition(decision, round_no)

        # Add context about partnership type
        intro = ""
        if supplier_type and round_no <= 2:
            if supplier_type == "Strategic":
                intro = "Given the strategic nature of this relationship, "
            elif supplier_type == "Leverage":
                intro = "To maximize the value of this partnership, "
            elif supplier_type == "Bottleneck":
                intro = "Understanding the critical nature of supply continuity, "

        return intro + " ".join(value_statements)

    def _weave_commercial_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave commercial plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Volume-based plays
        if any("volume" in p.lower() or "tier" in p.lower() for p in play_texts):
            return (
                "I'd like to explore how we can structure volume commitments to unlock "
                "better unit economics for both sides. If we can commit to tier-based volumes—"
                "say, 250 units initially with pathways to 500+—there may be room to optimize "
                "the pricing structure while giving you better demand visibility."
            )

        # Early payment plays
        if any("early payment" in p.lower() or "net-15" in p.lower() for p in play_texts):
            return (
                "One area where we can create immediate value is payment terms. "
                "If we can accelerate payment to net-15, would that open up opportunities "
                "for a 2-3% discount? This improves your cash flow while reducing our total cost."
            )

        # Bundling plays
        if any("bundle" in p.lower() or "consolidate" in p.lower() for p in play_texts):
            return (
                "We're also looking at how we might consolidate spend across multiple categories "
                "or adjacent products. If we can bundle this with other requirements coming through "
                "our pipeline, there could be meaningful volume synergies that benefit both parties."
            )

        # Generic commercial
        return (
            "I believe there's room to structure the commercial terms in a way that creates "
            "value for both organizations—whether through volume commitments, payment optimization, "
            "or longer-term price stability."
        )

    def _weave_operational_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave operational plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Delivery/lead time plays
        if any("delivery" in p.lower() or "lead time" in p.lower() for p in play_texts):
            return (
                "On the operational side, delivery timing is critical for our project schedule. "
                "If you can guarantee delivery within 2 weeks or offer split shipments "
                "(perhaps 30% upfront, balance within 4 weeks), that would significantly de-risk "
                "our production timeline and justify the investment."
            )

        # Planning/forecasting plays
        if any("forecast" in p.lower() or "planning" in p.lower() for p in play_texts):
            return (
                "We're happy to share our demand forecasts and collaborate on supply planning "
                "to give you better visibility. This integrated approach typically helps both sides "
                "reduce buffer stock and improve fulfillment rates."
            )

        # SLA/service level plays
        if any("sla" in p.lower() or "priority" in p.lower() for p in play_texts):
            return (
                "To ensure this partnership meets both our operational needs, I'd like to define "
                "clear service levels around delivery performance and fulfillment rates. "
                "Having those guardrails helps us plan effectively and holds both parties accountable."
            )

        # Generic operational
        return (
            "From an operational standpoint, we're looking for reliability and responsiveness "
            "in fulfillment. If we can align on clear delivery commitments and planning cadences, "
            "that creates a strong foundation for the partnership."
        )

    def _weave_strategic_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave strategic plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Innovation/co-development plays
        if any("innovation" in p.lower() or "co-develop" in p.lower() for p in play_texts):
            return (
                "Beyond the immediate transaction, I see potential for deeper collaboration "
                "on product development and innovation. If we can align on a shared roadmap "
                "for the next 12-24 months, there may be opportunities to co-invest in capabilities "
                "that benefit both our organizations."
            )

        # ESG/sustainability plays
        if any("esg" in p.lower() or "sustainab" in p.lower() for p in play_texts):
            return (
                "Sustainability is increasingly important to our stakeholders. "
                "If we can incorporate ESG metrics and carbon reduction targets into our partnership, "
                "that strengthens the strategic case and may unlock internal budget flexibility."
            )

        # Long-term commitment plays
        if any("long-term" in p.lower() or "multi-year" in p.lower() for p in play_texts):
            return (
                "We're thinking about this as a multi-year partnership rather than a one-off transaction. "
                "If you're open to a longer-term commitment with price stability mechanisms, "
                "we can structure this in a way that gives you revenue predictability while securing "
                "our supply chain."
            )

        # Generic strategic
        return (
            "Strategically, we're looking to build partnerships that go beyond transactional relationships. "
            "If we can align on shared objectives and longer-term value creation, "
            "that opens up different ways to structure the commercial terms."
        )

    def _weave_risk_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave risk plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Warranty plays
        if any("warranty" in p.lower() for p in play_texts):
            return (
                "Given the mission-critical nature of these components, warranty coverage is important. "
                "If you can extend warranty to 2-3 years at no additional cost, "
                "that reduces our total cost of ownership and makes the business case stronger."
            )

        # Service credit plays
        if any("service credit" in p.lower() or "sla penalt" in p.lower() for p in play_texts):
            return (
                "To manage performance risk, I'd like to incorporate service-level agreements "
                "with appropriate credits if delivery or quality targets aren't met. "
                "This ensures we have recourse mechanisms without damaging the relationship."
            )

        # Dual-sourcing plays
        if any("dual-sourc" in p.lower() or "alternative" in p.lower() for p in play_texts):
            return (
                "From a risk management perspective, we're evaluating dual-sourcing strategies "
                "to protect against supply disruption. If you can offer competitive terms and strong SLAs, "
                "you'd be well-positioned as our primary supplier with the volumes that come with that."
            )

        # Generic risk
        return (
            "We need to ensure appropriate risk mitigation through warranty coverage, "
            "performance guarantees, and clear remediation processes. "
            "Building these protections into the agreement benefits both parties."
        )

    def _weave_relational_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave relational plays into natural language."""

        return (
            "Looking beyond this specific engagement, we value suppliers who can grow with us "
            "and become trusted partners. If this initial project goes well, there are "
            "significant opportunities for expanded business across our organization. "
            "That long-term potential should factor into how we structure the initial terms."
        )

    def _craft_generic_value_proposition(
        self, decision: Dict[str, Any], round_no: int
    ) -> str:
        """Fallback value proposition when playbook unavailable."""

        asks = decision.get("asks", [])
        if not asks:
            return ""

        if round_no <= 2:
            return (
                "To make this work within our budget parameters, I'd like to explore "
                "areas where we can create mutual value—whether through volume commitments, "
                "payment terms optimization, or longer-term partnership structures."
            )
        else:
            return (
                "As we work toward closure, I believe there are still opportunities "
                "to optimize the total package through creative structuring of terms, "
                "payment schedules, and service commitments."
            )

    def _craft_collaborative_asks(
        self,
        *,
        decision: Dict[str, Any],
        round_no: int,
        signals: Dict[str, Any],
        tone: str,
    ) -> str:
        """Frame asks as collaborative opportunities."""

        asks = decision.get("asks", [])
        lead_time_request = decision.get("lead_time_request")

        if not asks and not lead_time_request:
            return ""

        # Frame based on tone
        if tone in ("decisive", "closure-oriented"):
            intro = "To finalize this agreement, there are a few specific elements I'd like to confirm:"
        else:
            intro = "To help us structure the best possible outcome, I'd like to explore a few areas:"

        formatted_asks: List[str] = []

        # Convert asks into questions/proposals rather than demands
        for ask in asks[:4]:  # Limit to top 4
            ask_text = str(ask).strip()

            # Rephrase common asks to be more collaborative
            if "volume" in ask_text.lower() or "tier" in ask_text.lower():
                formatted_asks.append(
                    "• What volume thresholds would unlock better unit economics? "
                    "We're confident we can hit meaningful tiers if the pricing justifies it."
                )
            elif "payment" in ask_text.lower() and ("early" in ask_text.lower() or "discount" in ask_text.lower()):
                formatted_asks.append(
                    "• Would accelerated payment (net-15) create value for you? "
                    "We'd be happy to explore if that opens up pricing flexibility."
                )
            elif "lead time" in ask_text.lower() or "delivery" in ask_text.lower():
                formatted_asks.append(
                    "• Can you confirm the delivery timeline and whether there's flexibility "
                    "for partial shipments if that helps manage both our schedules?"
                )
            elif "warranty" in ask_text.lower():
                formatted_asks.append(
                    "• What warranty coverage is included, and is there room to extend that "
                    "as part of the overall package?"
                )
            elif "breakdown" in ask_text.lower() or "cost" in ask_text.lower():
                formatted_asks.append(
                    "• Would you be open to sharing a high-level cost breakdown? "
                    "That transparency helps us justify the investment internally."
                )
            elif "alternative" in ask_text.lower() or "spec" in ask_text.lower():
                formatted_asks.append(
                    "• Are there alternative specifications or components that could reduce cost "
                    "while still meeting our performance requirements?"
                )
            else:
                # Generic reframe
                formatted_asks.append(f"• {ask_text}")

        # Add lead time if specified
        if lead_time_request and not any("lead time" in fa.lower() for fa in formatted_asks):
            formatted_asks.append(
                f"• Regarding timing: {lead_time_request.lower()} would be ideal for our project schedule."
            )

        if not formatted_asks:
            return ""

        return intro + "\n\n" + "\n".join(formatted_asks)

    def _craft_closing(self, round_no: int, strategy: str, tone: str) -> str:
        """Create a closing that maintains momentum."""

        if strategy == "accept":
            return (
                "If we can align on these final points, I'm ready to move forward quickly "
                "and get the paperwork in motion. Looking forward to your thoughts."
            )

        if strategy == "decline":
            return (
                "I appreciate your engagement throughout this process. Given where we've landed, "
                "I'll need to take this back to my team for further discussion. "
                "I'll circle back if our parameters change."
            )

        if round_no >= 3:
            return (
                "I'm hopeful we can find common ground here. This represents our best position "
                "given all the factors at play. I'd appreciate your thoughts on whether we can "
                "make this work, and I'm happy to jump on a call if that would be helpful "
                "to talk through any remaining sticking points."
            )

        if round_no == 2:
            return (
                "I believe we're getting close to something that works for both sides. "
                "Let me know your thoughts on this structure, and we can refine from there. "
                "Happy to discuss any concerns or questions you might have."
            )

        # Round 1 or default
        return (
            "I'd welcome your perspective on this proposal. If you have questions or "
            "want to discuss any of these points in more detail, I'm happy to set up "
            "a quick call. Looking forward to your response."
        )

    def _load_playbook(self) -> Dict[str, Any]:
        if self._playbook_cache is not None:
            return self._playbook_cache
        try:
            with PLAYBOOK_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            # Normalise lever keys to expected casing for faster lookups.
            for supplier_type, entry in data.items():
                styles = entry.get("styles")
                if not isinstance(styles, dict):
                    continue
                for style_key, style_entry in list(styles.items()):
                    if not isinstance(style_entry, dict):
                        continue
                    normalised_style_entry: Dict[str, Any] = {}
                    for lever_key, plays in style_entry.items():
                        lever_name = self._normalise_lever_category(lever_key)
                        if not lever_name:
                            continue
                        normalised_style_entry[lever_name] = plays
                    styles[style_key] = normalised_style_entry
            self._playbook_cache = data
            return data
        except FileNotFoundError:
            logger.warning("Negotiation playbook file missing at %s", PLAYBOOK_PATH)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to load negotiation playbook from %s", PLAYBOOK_PATH)
        self._playbook_cache = {}
        return {}

    def _normalise_supplier_type(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        mapping = {
            "transactional": "Transactional",
            "leverage": "Leverage",
            "strategic": "Strategic",
            "bottleneck": "Bottleneck",
        }
        for key, canonical in mapping.items():
            if key in lowered:
                return canonical
        return None

    def _normalise_negotiation_style(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        mapping = {
            "competitive": "Competitive",
            "collaborative": "Collaborative",
            "principled": "Principled",
            "accommodating": "Accommodating",
            "compromising": "Compromising",
        }
        for key, canonical in mapping.items():
            if key in lowered:
                return canonical
        return None

    def _normalise_lever_category(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        cleaned = re.sub(r"[^a-zA-Z]+", " ", text).strip().upper()
        if not cleaned:
            return None
        for category in LEVER_CATEGORIES:
            if cleaned == category or category in cleaned.split():
                return category.title()
        return None

    def _extract_policy_guidance(self, context: AgentContext) -> Dict[str, Set[str]]:
        guidance: Dict[str, Set[str]] = {
            "preferred": set(),
            "required": set(),
            "restricted": set(),
            "discouraged": set(),
        }

        def ingest(bucket: str, values: Any) -> None:
            if not values:
                return
            for item in self._ensure_list(values):
                lever = self._normalise_lever_category(item)
                if lever:
                    guidance[bucket].add(lever)

        def parse_blob(blob: Any) -> None:
            if blob is None:
                return
            if isinstance(blob, dict):
                for key, value in blob.items():
                    if isinstance(value, (dict, list)):
                        parse_blob(value)
                        continue
                    lowered = str(key).lower()
                    if lowered in {"preferred", "preferred_levers", "allowed_levers", "focus_levers", "priority_levers"}:
                        ingest("preferred", value)
                    elif lowered in {"required", "required_levers", "mandated_levers", "must_have"}:
                        ingest("required", value)
                    elif lowered in {"restricted", "restricted_levers", "blocked_levers", "prohibited_levers"}:
                        ingest("restricted", value)
                    elif lowered in {"discouraged", "discouraged_levers", "avoid_levers"}:
                        ingest("discouraged", value)
                    else:
                        lever = self._normalise_lever_category(value)
                        if lever:
                            guidance["preferred"].add(lever)
            elif isinstance(blob, list):
                for item in blob:
                    parse_blob(item)
            elif isinstance(blob, str):
                tokens = re.split(r"[,;/]", blob)
                for token in tokens:
                    lever = self._normalise_lever_category(token)
                    if lever:
                        guidance["preferred"].add(lever)

        parse_blob(context.input_data.get("policy_guidance"))
        parse_blob(context.input_data.get("policy_constraints"))
        parse_blob(context.input_data.get("policy_preferences"))
        for policy in self._ensure_list(context.input_data.get("policies")):
            parse_blob(policy)
        return guidance

    def _score_policy_alignment(
        self, lever: str, guidance: Dict[str, Set[str]]
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        score = 0.0
        if lever in guidance.get("required", set()):
            score += 0.6
            notes.append("Required by policy")
        if lever in guidance.get("preferred", set()):
            score += 0.3
            notes.append("Policy prefers this lever")
        if lever in guidance.get("discouraged", set()):
            score -= 0.3
            notes.append("Policy discourages this lever")
        if lever in guidance.get("restricted", set()):
            score -= 0.7
            notes.append("Policy restricts this lever")
        return score, notes

    def _score_supplier_performance(
        self, lever: str, performance: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        if not performance:
            return score, notes

        def _coerce_float_metric(*keys: str) -> Optional[float]:
            for key in keys:
                value = performance.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        on_time = _coerce_float_metric("on_time_delivery", "on_time", "delivery_score", "otif")
        if on_time is not None:
            if on_time < 0.9 and lever in {"Operational", "Risk"}:
                score += 0.4 if lever == "Operational" else 0.2
                notes.append("On-time delivery below 90%")
            elif on_time > 0.97 and lever == "Operational":
                score -= 0.1
                notes.append("Delivery reliability already strong")

        defect_rate = _coerce_float_metric("quality_incidents", "defect_rate", "return_rate")
        if defect_rate is not None and defect_rate > 0:
            if lever == "Risk":
                score += 0.3
            notes.append("Quality issues detected")

        esg_score = _coerce_float_metric("esg_score", "sustainability_score")
        if esg_score is not None and esg_score < 0.6 and lever == "Strategic":
            score += 0.2
            notes.append("ESG performance lagging")

        collaboration_score = _coerce_float_metric("relationship_score", "collaboration_index")
        if collaboration_score is not None and collaboration_score < 0.6 and lever == "Relational":
            score += 0.3
            notes.append("Relationship maturity low")

        innovation_score = _coerce_float_metric("innovation_score", "co_innovation_index")
        if innovation_score is not None and innovation_score < 0.5 and lever == "Strategic":
            score += 0.25
            notes.append("Innovation potential needs reinforcement")

        return score, notes

    def _score_market_context(
        self, lever: str, market: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        if not market:
            return score, notes

        supply_risk = market.get("supply_risk") or market.get("supply_risk_level")
        if isinstance(supply_risk, str) and supply_risk.strip().lower() in {"high", "elevated", "tight"}:
            if lever == "Risk":
                score += 0.3
            notes.append("Market supply risk elevated")

        demand_trend = market.get("demand_trend") or market.get("demand")
        if isinstance(demand_trend, str) and demand_trend.strip().lower() in {"rising", "high"}:
            if lever == "Commercial":
                score += 0.2
            notes.append("Demand is rising")

        inflation = market.get("inflation") or market.get("price_trend")
        if isinstance(inflation, str) and inflation.strip().lower() in {"inflationary", "increasing"}:
            if lever == "Commercial":
                score += 0.25
            notes.append("Prices trending upward")

        capacity = market.get("capacity") or market.get("capacity_constraints")
        if isinstance(capacity, str) and capacity.strip().lower() in {"limited", "constrained"}:
            if lever in {"Operational", "Strategic"}:
                score += 0.2
            notes.append("Capacity constraints present")

        esg_pressure = market.get("esg_pressure") or market.get("regulatory_focus")
        if isinstance(esg_pressure, str) and esg_pressure.strip().lower() in {"high", "tightening"}:
            if lever == "Strategic":
                score += 0.2
            notes.append("ESG expectations increasing")

        return score, notes

    def _compose_play_rationale(
        self,
        descriptor: Optional[str],
        style: Optional[str],
        lever: str,
        policy_notes: List[str],
        performance_notes: List[str],
        market_notes: List[str],
    ) -> str:
        segments: List[str] = []
        if descriptor:
            segments.append(str(descriptor))
        if style:
            segments.append(f"Supports {style.lower()} posture on the {lever.lower()} lever.")
        else:
            segments.append(f"Targets the {lever.lower()} lever.")
        if policy_notes:
            segments.append("Policy: " + "; ".join(policy_notes))
        if performance_notes:
            segments.append("Performance: " + "; ".join(performance_notes))
        if market_notes:
            segments.append("Market: " + "; ".join(market_notes))
        return " ".join(segments)

    # ------------------------------------------------------------------
    # Decision support helpers
    # ------------------------------------------------------------------
    def _detect_final_offer(
        self, supplier_message: Optional[str], supplier_snippets: List[str]
    ) -> Optional[str]:
        texts: List[str] = []
        if supplier_message:
            texts.append(supplier_message)
        texts.extend(snippet for snippet in supplier_snippets if snippet)
        for text in texts:
            lowered = text.lower()
            for pattern in FINAL_OFFER_PATTERNS:
                if pattern in lowered:
                    return f"Supplier indicated final offer via phrase '{pattern}'."
        return None

    def _should_continue(
        self,
        state: Dict[str, Any],
        supplier_reply_registered: bool,
        final_offer_reason: Optional[str],
    ) -> Tuple[bool, str, str]:
        status = state.get("status", "ACTIVE")
        if status in {"COMPLETED", "EXHAUSTED"}:
            return False, status, f"Session already {status.lower()}."
        if final_offer_reason:
            return False, "COMPLETED", final_offer_reason
        replies = int(state.get("supplier_reply_count", 0))
        if replies >= MAX_SUPPLIER_REPLIES:
            return False, "EXHAUSTED", "Supplier reply cap reached."
        if state.get("awaiting_response") and not supplier_reply_registered:
            return False, "AWAITING_SUPPLIER", "Awaiting supplier response."
        return True, "ACTIVE", ""

    def _build_summary(
        self,
        context: AgentContext,
        session_reference: Optional[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        *,
        supplier: Optional[str] = None,
        supplier_name: Optional[str] = None,
        supplier_snippets: Optional[List[str]] = None,
        supplier_message: Optional[str] = None,
        playbook_context: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
        zopa: Optional[Dict[str, Any]] = None,
        procurement_summary: Optional[Dict[str, Any]] = None,
        rag_snippets: Optional[List[str]] = None,
        contact_name: Optional[str] = None,
    ) -> str:
        """Build negotiation message using skilled negotiation techniques."""

        positions = self._build_positions_from_decision(
            decision, price, target_price, round_no
        )

        message_text: Optional[str] = None
        if self._use_enhanced_messages and hasattr(
            self, "_compose_negotiation_message"
        ):
            try:
                message_text = self._compose_negotiation_message(
                    context=context,
                    decision=decision,
                    positions=positions,
                    round_no=round_no,
                    currency=currency,
                    supplier=supplier,
                    supplier_name=supplier_name,
                    supplier_message=supplier_message,
                    signals=signals or {},
                    zopa=zopa or {},
                    playbook_context=playbook_context,
                    procurement_summary=procurement_summary,
                    contact_name=contact_name,
                )
            except Exception as exc:
                logger.warning(
                    "Enhanced message composer failed: %s, using fallback",
                    exc,
                    exc_info=True,
                )

        if message_text is None:
            message_text = self._build_summary_fallback(
                context=context,
                session_reference=session_reference,
                decision=decision,
                price=price,
                target_price=target_price,
                currency=currency,
                round_no=round_no,
                supplier=supplier,
                supplier_snippets=supplier_snippets,
                supplier_message=supplier_message,
                playbook_context=playbook_context,
                signals=signals,
                zopa=zopa,
                procurement_summary=procurement_summary,
                rag_snippets=rag_snippets,
            )
        return message_text or ""

    def _build_positions_from_decision(
        self,
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        round_no: int,
    ) -> NegotiationPositions:
        """Build NegotiationPositions object from decision data."""

        positions_dict = decision.get("positions", {})
        if isinstance(positions_dict, dict):
            history = positions_dict.get("history", [])
        else:
            history = []

        return NegotiationPositions(
            start=decision.get("start_position") or positions_dict.get("start"),
            desired=decision.get("desired_position")
            or positions_dict.get("desired")
            or target_price,
            no_deal=decision.get("no_deal_position") or positions_dict.get("no_deal"),
            supplier_offer=price,
            history=history if isinstance(history, list) else [],
        )

    def _build_summary_fallback(
        self,
        context: AgentContext,
        session_reference: Optional[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        *,
        supplier: Optional[str] = None,
        supplier_snippets: Optional[List[str]] = None,
        supplier_message: Optional[str] = None,
        playbook_context: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
        zopa: Optional[Dict[str, Any]] = None,
        procurement_summary: Optional[Dict[str, Any]] = None,
        rag_snippets: Optional[List[str]] = None,
    ) -> str:
        """Fallback basic summary builder (original implementation)."""

        workflow_reference = session_reference or context.input_data.get("workflow_id")
        if not workflow_reference:
            workflow_reference = getattr(context, "workflow_id", None)
        reference_text = str(workflow_reference).strip() if workflow_reference else "workflow"

        strategy = decision.get("strategy")
        lines: List[str] = []
        header = f"Round {round_no} plan for {reference_text}"
        if strategy:
            header += f": {strategy}"
        lines.append(header)

        if round_no >= 3:
            lines.append(
                "- Closing round: request the supplier's best and final quote and confirm readiness to award"
            )

        descriptor: Optional[str] = None
        supplier_type: Optional[str] = None
        negotiation_style: Optional[str] = None
        lever_priorities: List[str] = []
        if playbook_context:
            descriptor_candidate = playbook_context.get("descriptor")
            descriptor = str(descriptor_candidate).strip() if descriptor_candidate else None
            supplier_candidate = playbook_context.get("supplier_type")
            supplier_type = str(supplier_candidate).strip() if supplier_candidate else None
            style_candidate = playbook_context.get("style")
            negotiation_style = (
                str(style_candidate).strip() if isinstance(style_candidate, str) else None
            )
            lever_values = playbook_context.get("lever_priorities")
            if isinstance(lever_values, (list, tuple)):
                lever_priorities = [
                    str(value).strip()
                    for value in lever_values
                    if isinstance(value, str) and value.strip()
                ]

        if supplier_type or descriptor:
            descriptor_parts: List[str] = []
            if supplier_type:
                descriptor_parts.append(f"Supplier category: {supplier_type}")
            if descriptor:
                descriptor_parts.append(descriptor)
            if descriptor_parts:
                lines.append("- " + " — ".join(descriptor_parts))

        if negotiation_style:
            if lever_priorities:
                lever_text = ", ".join(lever_priorities[:3])
            else:
                lever_text = "balanced levers"
            lines.append(
                f"- Negotiation style: {negotiation_style}; focus levers: {lever_text}"
            )

        counter_price = decision.get("counter_price")
        formatted_offer: Optional[str] = None
        if price is not None:
            formatted_offer = self._format_currency(price, currency)

        if counter_price is not None:
            counter_text = self._format_currency(counter_price, currency)
            if formatted_offer:
                lines.append(
                    f"- Counter at {counter_text} against supplier offer {formatted_offer}"
                )
            else:
                lines.append(f"- Counter target: {counter_text}")
        elif formatted_offer:
            lines.append(f"- Seek clarification before accepting {formatted_offer}")

        if target_price is not None and price is not None:
            target_text = self._format_currency(target_price, currency)
            delta = price - target_price
            try:
                delta_text = self._format_currency(delta, currency)
            except Exception:
                delta_text = f"{delta:0.2f}"
            lines.append(
                f"- Target {target_text} vs offer {formatted_offer} (gap {delta_text})"
            )

        lead_time = decision.get("lead_time_request")
        if lead_time:
            lines.append(f"- Lead time ask: {lead_time}")

        asks = decision.get("asks") or []
        if asks:
            lines.append("- Key asks: " + "; ".join(str(item) for item in asks if item))

        return "\n".join(lines)

    def _compose_negotiation_message(
        self,
        *,
        context: AgentContext,
        decision: Dict[str, Any],
        positions: NegotiationPositions,
        round_no: int,
        currency: Optional[str],
        supplier: Optional[str],
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        zopa: Dict[str, Any],
        playbook_context: Optional[Dict[str, Any]],
        procurement_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Compose a human-like, strategically crafted negotiation message."""

        counter_price = decision.get("counter_price")
        current_offer = positions.supplier_offer
        target_price = positions.desired
        strategy = decision.get("strategy", "counter")

        sections: List[str] = []

        opening = self._craft_opening_simple(round_no, supplier_message, signals)
        if opening:
            sections.append(opening)

        if round_no > 1 and supplier_message:
            acknowledgment = self._craft_acknowledgment_simple(
                current_offer, signals, currency
            )
            if acknowledgment:
                sections.append(acknowledgment)

        position = self._craft_position_simple(
            counter_price, current_offer, target_price, currency, round_no
        )
        if position:
            sections.append(position)

        if playbook_context and playbook_context.get("plays"):
            value_prop = self._craft_value_proposition_simple(playbook_context, round_no)
            if value_prop:
                sections.append(value_prop)

        asks_section = self._craft_asks_simple(decision)
        if asks_section:
            sections.append(asks_section)

        closing = self._craft_closing_simple(round_no, strategy)
        if closing:
            sections.append(closing)

        return "\n\n".join(sections)

    def _craft_opening_simple(
        self, round_no: int, supplier_message: Optional[str], signals: Dict[str, Any]
    ) -> str:
        """Craft a simple opening."""
        if round_no > 1 and supplier_message:
            return (
                "Thank you for your response. I've reviewed your proposal and would like to discuss "
                "how we can align on the details."
            )
        return (
            "Thank you for your proposal. I'd like to discuss the terms to ensure we can reach an "
            "agreement that works for both parties."
        )

    def _craft_acknowledgment_simple(
        self, current_offer: Optional[float], signals: Dict[str, Any], currency: Optional[str]
    ) -> str:
        """Craft simple acknowledgment."""
        if signals.get("capacity_tight"):
            return (
                "I understand the capacity constraints you mentioned, and we're mindful of the current "
                "market dynamics."
            )
        if current_offer:
            offer_text = self._format_currency(current_offer, currency)
            return (
                f"Your quoted price of {offer_text} reflects the quality you provide, which we value."
            )
        return ""

    def _craft_position_simple(
        self,
        counter_price: Optional[float],
        current_offer: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
    ) -> str:
        """Craft simple position statement."""
        if counter_price is None:
            return "Before we proceed, could you provide more details on the cost breakdown?"

        counter_text = self._format_currency(counter_price, currency)

        if round_no == 1:
            return (
                f"To align with our project budget, I'd like to propose we structure this around {counter_text}."
            )
        if round_no == 2:
            return (
                f"As we work toward agreement, I believe {counter_text} represents a fair position that reflects both the value "
                "and our budget constraints."
            )
        return (
            f"To help us reach closure, I'm proposing {counter_text} as our target. This represents our best position given the "
            "business case."
        )

    def _craft_value_proposition_simple(
        self, playbook_context: Dict[str, Any], round_no: int
    ) -> str:
        """Craft simple value proposition from playbook."""
        plays = playbook_context.get("plays", [])[:2]

        if not plays:
            return ""

        statements: List[str] = []
        for play in plays:
            if not isinstance(play, dict):
                continue

            description = play.get("play", "")
            lowered = description.lower()
            if "volume" in lowered or "tier" in lowered:
                statements.append(
                    "We'd like to explore volume commitments that could unlock better pricing for both sides."
                )
            elif "payment" in lowered and "early" in lowered:
                statements.append(
                    "We can offer accelerated payment terms if that helps with pricing flexibility."
                )
            elif "warranty" in lowered:
                statements.append(
                    "Extended warranty coverage would strengthen the business case on our side."
                )

        return " ".join(statements[:2]) if statements else ""

    def _craft_asks_simple(self, decision: Dict[str, Any]) -> str:
        """Craft simple asks section."""
        asks = decision.get("asks", [])
        if not asks:
            return ""

        asks_list = [f"• {ask}" for ask in asks[:4] if ask]
        if not asks_list:
            return ""

        return "To help structure the best outcome:\n" + "\n".join(asks_list)

    def _craft_closing_simple(self, round_no: int, strategy: str) -> str:
        """Craft simple closing."""
        if strategy == "accept":
            return (
                "If we can align on these final points, I'm ready to move forward. Looking forward to your thoughts."
            )

        if round_no >= 3:
            return (
                "I'm hopeful we can find common ground. Please let me know your thoughts, and I'm happy to discuss further if needed."
            )

        return (
            "I'd welcome your feedback on this proposal. Happy to discuss any questions you might have."
        )

    def _get_prompt_template(self, context: AgentContext) -> str:
        template: Optional[str] = None
        prompt_ids: List[int] = []

        prompts_payload = context.input_data.get("prompts")
        if isinstance(prompts_payload, list):
            for prompt in prompts_payload:
                if not isinstance(prompt, dict):
                    continue
                candidate = prompt.get("prompt_template") or prompt.get("template")
                if candidate:
                    template = str(candidate)
                    break
                pid = prompt.get("promptId") or prompt.get("prompt_id")
                try:
                    if pid is not None:
                        prompt_ids.append(int(pid))
                except (TypeError, ValueError):
                    continue

        if template:
            return template

        prompt_engine = getattr(self, "prompt_engine", None)
        if prompt_engine is not None:
            for pid in prompt_ids:
                prompt_entry = prompt_engine.get_prompt(pid)
                if prompt_entry and prompt_entry.get("template"):
                    return str(prompt_entry["template"])

            for prompt_entry in prompt_engine.prompts_for_agent(self.__class__.__name__):
                candidate = prompt_entry.get("template")
                if candidate:
                    return str(candidate)

        return DEFAULT_NEGOTIATION_MESSAGE_TEMPLATE

    def _build_prompt_context(
        self,
        *,
        context: AgentContext,
        header: str,
        lines: List[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        supplier: Optional[str],
        supplier_snippets: List[str],
        supplier_message: Optional[str],
        playbook_context: Optional[Dict[str, Any]],
        signals: Optional[Dict[str, Any]],
        zopa: Optional[Dict[str, Any]],
        procurement_summary: Optional[Dict[str, Any]],
        rag_snippets: Optional[List[str]],
        rfq_reference: Optional[str] = None,
    ) -> Dict[str, str]:
        summary_lines = lines[1:]
        details = "\n".join(summary_lines)
        full_summary = "\n".join(lines)

        supplier_name = context.input_data.get("supplier_name")
        supplier_identifier = context.input_data.get("supplier_id") or supplier or ""

        current_offer_raw = context.input_data.get("current_offer")
        counter_price = decision.get("counter_price")

        current_offer_formatted = (
            self._format_currency(price, currency) if price is not None else ""
        )
        target_price_formatted = (
            self._format_currency(target_price, currency) if target_price is not None else ""
        )
        counter_price_formatted = (
            self._format_currency(counter_price, currency) if counter_price is not None else ""
        )

        supplier_snippet_text = "\n".join(
            f"- {snippet}" for snippet in supplier_snippets if isinstance(snippet, str) and snippet
        )

        procurement_lines = ""
        procurement_metrics = ""
        procurement_count = ""
        if procurement_summary:
            lines_payload = procurement_summary.get("summary_lines") or []
            if isinstance(lines_payload, list):
                unique_lines: List[str] = []
                for entry in lines_payload:
                    text = str(entry).strip()
                    if text and text not in unique_lines:
                        unique_lines.append(text)
                if unique_lines:
                    procurement_lines = "\n".join(f"- {text}" for text in unique_lines[:5])
            metrics_payload = procurement_summary.get("metrics")
            if metrics_payload:
                procurement_metrics = self._serialise_for_prompt(metrics_payload)
            record_count = procurement_summary.get("record_count")
            if record_count is not None:
                procurement_count = str(record_count)

        rag_text = ""
        if rag_snippets:
            rag_entries = [
                str(snippet).strip()
                for snippet in rag_snippets
                if isinstance(snippet, str) and snippet.strip()
            ]
            if rag_entries:
                rag_text = "\n".join(f"- {snippet}" for snippet in rag_entries)

        policy_payload = context.input_data.get("policies") or context.policy_context
        policy_text = self._serialise_for_prompt(policy_payload)
        performance_text = self._serialise_for_prompt(
            context.input_data.get("supplier_performance")
        )
        market_text = self._serialise_for_prompt(context.input_data.get("market_context"))

        context_sections: List[str] = []
        if procurement_lines:
            context_sections.append("\n\nContext – Procurement summary:\n" + procurement_lines)
        if procurement_metrics:
            context_sections.append("\n\nProcurement metrics:\n" + procurement_metrics)
        if rag_text:
            context_sections.append("\n\nRetrieved knowledge snippets:\n" + rag_text)
        if policy_text:
            context_sections.append("\n\nPolicy guidance:\n" + policy_text)
        if performance_text:
            context_sections.append("\n\nSupplier performance:\n" + performance_text)
        if market_text:
            context_sections.append("\n\nMarket context:\n" + market_text)

        decision_positions = self._serialise_for_prompt(decision.get("positions"))
        decision_flags = self._serialise_for_prompt(decision.get("flags"))
        decision_outliers = self._serialise_for_prompt(decision.get("outlier_alerts"))
        decision_validation = self._serialise_for_prompt(decision.get("validation_issues"))

        prompt_values: Dict[str, str] = {
            "header": header,
            "details": details,
            "summary_lines": details,
            "full_summary": full_summary,
            "context_sections": "".join(context_sections),
            "session_reference": str(rfq_reference or ""),
            "round_number": str(round_no),
            "strategy": str(decision.get("strategy") or ""),
            "supplier_id": str(supplier_identifier),
            "supplier_name": str(supplier_name or supplier_identifier or ""),
            "supplier_snippets": supplier_snippet_text,
            "supplier_message": str(supplier_message or ""),
            "current_offer": self._serialise_for_prompt(current_offer_raw),
            "current_offer_formatted": current_offer_formatted,
            "target_price": self._serialise_for_prompt(target_price),
            "target_price_formatted": target_price_formatted,
            "counter_price": self._serialise_for_prompt(counter_price),
            "counter_price_formatted": counter_price_formatted,
            "currency": str(currency or ""),
            "asks": self._serialise_for_prompt(decision.get("asks")),
            "lead_time_request": str(decision.get("lead_time_request") or ""),
            "decision_rationale": str(decision.get("rationale") or ""),
            "decision_plan_message": str(decision.get("plan_counter_message") or ""),
            "decision_positions": decision_positions,
            "decision_flags": decision_flags,
            "decision_outliers": decision_outliers,
            "decision_validation": decision_validation,
            "signals_summary": self._serialise_for_prompt(signals),
            "zopa_summary": self._serialise_for_prompt(zopa),
            "playbook_context": self._serialise_for_prompt(playbook_context),
            "policy_context": self._serialise_for_prompt(context.policy_context),
            "policy_guidance": policy_text,
            "supplier_performance": performance_text,
            "market_context": market_text,
            "task_profile": self._serialise_for_prompt(context.task_profile),
            "knowledge_base": self._serialise_for_prompt(context.knowledge_base),
            "workflow_id": str(context.workflow_id),
            "agent_id": str(context.agent_id),
            "agentic_plan": "\n".join(self.AGENTIC_PLAN_STEPS),
            "procurement_summary_lines": procurement_lines,
            "procurement_summary_metrics": procurement_metrics,
            "procurement_summary_record_count": procurement_count,
            "rag_snippets": rag_text,
            "previous_email_subject": str(
                context.input_data.get("previous_email_subject") or ""
            ),
            "thread_headers": self._serialise_for_prompt(
                context.input_data.get("thread_headers")
            ),
            "supplier_replies": self._serialise_for_prompt(
                context.input_data.get("supplier_replies")
                or context.input_data.get("supplier_responses")
            ),
            "supplier_reply_count": str(
                context.input_data.get("supplier_reply_count")
                or context.input_data.get("supplier_responses_count")
                or ""
            ),
            "positions": decision_positions,
        }

        return prompt_values

    def _apply_prompt_template(
        self, template: str, values: Dict[str, str], lines: List[str]
    ) -> str:
        safe_values: defaultdict[str, str] = defaultdict(str)
        for key, value in values.items():
            if value is None:
                continue
            safe_values[key] = value

        rendered: str = ""
        try:
            rendered = template.format_map(safe_values).strip()
        except Exception:
            logger.debug("Negotiation prompt template formatting failed", exc_info=True)

        if not rendered:
            rendered = "\n".join(lines)

        return rendered

    def _serialise_for_prompt(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            try:
                return str(value)
            except Exception:
                return ""

    def _retrieve_procurement_summary(
        self,
        *,
        supplier_id: Optional[str],
        supplier_name: Optional[str],
    ) -> Dict[str, Any]:
        engine = getattr(self.agent_nick, "query_engine", None)
        if engine is None or not hasattr(engine, "fetch_procurement_flow"):
            return {}

        try:
            df = engine.fetch_procurement_flow(
                supplier_ids=[supplier_id] if supplier_id else None,
                supplier_names=[supplier_name.lower()] if supplier_name else None,
            )
        except Exception:
            logger.debug(
                "NegotiationAgent: procurement flow retrieval failed", exc_info=True
            )
            return {}

        if df is None:
            return {}

        records: List[Dict[str, Any]] = []
        try:
            records = df.to_dict(orient="records")  # type: ignore[attr-defined]
        except Exception:
            if isinstance(df, list):
                records = [dict(row) for row in df if isinstance(row, dict)]

        if not records:
            return {}

        summary_lines: List[str] = []
        totals: List[float] = []
        currency_hint: Optional[str] = None

        for row in records[:10]:
            if not isinstance(row, dict):
                continue
            currency_hint = (
                row.get("currency")
                or row.get("default_currency")
                or row.get("invoice_currency")
                or currency_hint
            )
            amount = (
                row.get("total_amount")
                or row.get("total")
                or row.get("invoice_total_incl_tax")
                or row.get("order_total")
            )
            amount_value = self._coerce_float(amount)
            po_id = (
                row.get("po_id")
                or row.get("purchase_order_number")
                or row.get("purchase_order_id")
            )
            invoice_id = row.get("invoice_id") or row.get("invoice_number")
            order_date = (
                row.get("order_date")
                or row.get("requested_date")
                or row.get("invoice_date")
            )
            if amount_value is not None:
                totals.append(float(amount_value))
            reference = po_id or invoice_id
            if reference and amount_value is not None:
                amount_text = self._format_currency(amount_value, currency_hint)
                if order_date:
                    summary_lines.append(f"{reference} – {amount_text} on {order_date}")
                else:
                    summary_lines.append(f"{reference} – {amount_text}")

        metrics: Dict[str, Any] = {}
        if totals:
            total_spend = sum(totals)
            average_value = total_spend / len(totals)
            metrics["total_spend"] = total_spend
            metrics["average_transaction_value"] = average_value
            metrics["total_spend_formatted"] = self._format_currency(
                total_spend, currency_hint
            )
            metrics["average_transaction_value_formatted"] = self._format_currency(
                average_value, currency_hint
            )

        unique_lines: List[str] = []
        for entry in summary_lines:
            if entry and entry not in unique_lines:
                unique_lines.append(entry)

        return {
            "summary_lines": unique_lines[:5],
            "metrics": metrics,
            "record_count": len(records),
        }

    def _collect_vector_snippets(
        self,
        *,
        context: AgentContext,
        supplier_name: Optional[str],
        workflow_reference: Optional[str],
    ) -> List[str]:
        query_tokens: List[str] = []
        if supplier_name:
            query_tokens.append(str(supplier_name))
        category = (
            context.input_data.get("product_category")
            or context.input_data.get("category")
            or context.input_data.get("product_type")
        )
        if isinstance(category, str) and category.strip():
            query_tokens.append(category.strip())
        description = context.input_data.get("rfq_description") or context.input_data.get(
            "item_description"
        )
        if isinstance(description, str) and description.strip():
            query_tokens.append(description.strip())
        if workflow_reference:
            query_tokens.append(str(workflow_reference))

        query = " ".join(query_tokens[:3]).strip()
        if not query:
            return []

        results = self.vector_search(query, top_k=3)
        snippets: List[str] = []
        for result in results or []:
            payload = getattr(result, "payload", None)
            if isinstance(payload, dict):
                snippet = (
                    payload.get("text")
                    or payload.get("content")
                    or payload.get("summary")
                    or payload.get("body")
                )
                if snippet:
                    text = str(snippet).strip()
                    if text:
                        snippets.append(text)
                continue

            if isinstance(result, dict):
                payload_data = result.get("payload")
                if isinstance(payload_data, dict):
                    snippet = (
                        payload_data.get("text")
                        or payload_data.get("content")
                        or payload_data.get("summary")
                    )
                    if snippet:
                        text = str(snippet).strip()
                        if text:
                            snippets.append(text)
                            continue
                snippet = (
                    result.get("text")
                    or result.get("content")
                    or result.get("summary")
                    or result.get("body")
                )
                if snippet:
                    text = str(snippet).strip()
                    if text:
                        snippets.append(text)

        return snippets

    def _build_supplier_watch_fields(
        self,
        *,
        context: AgentContext,
        workflow_id: Optional[str],
        supplier: Optional[str],
        drafts: List[Dict[str, Any]],
        state: Dict[str, Any],
        session_reference: Optional[str] = None,
        rfq_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not workflow_id:
            return None

        supplier_id = self._coerce_text(supplier)
        if not supplier_id:
            logger.error("Cannot build watch fields without supplier_id")
            return None
        supplier = supplier_id

        candidate_drafts: List[Dict[str, Any]] = []
        observed_unique_ids: List[str] = []
        unique_id_set: Set[str] = set()

        def _record_unique_id(raw_value: Any) -> Optional[str]:
            text = self._coerce_text(raw_value)
            if text and text not in unique_id_set:
                unique_id_set.add(text)
                observed_unique_ids.append(text)
            return text

        for source in (drafts, context.input_data.get("drafts")):
            if not source:
                continue
            if isinstance(source, dict):
                source = [source]
            if not isinstance(source, list):
                continue
            for entry in source:
                if isinstance(entry, dict):
                    draft_copy = dict(entry)
                    metadata_source = draft_copy.get("metadata")
                    metadata = (
                        dict(metadata_source)
                        if isinstance(metadata_source, dict)
                        else {}
                    )
                    if rfq_id and not draft_copy.get("rfq_id"):
                        draft_copy["rfq_id"] = rfq_id
                    if rfq_id and not metadata.get("rfq_id"):
                        metadata["rfq_id"] = rfq_id
                    if metadata:
                        draft_copy["metadata"] = metadata
                    unique_value = _record_unique_id(draft_copy.get("unique_id"))
                    if not unique_value:
                        for key in (
                            "dispatch_run_id",
                            "run_id",
                            "email_action_id",
                            "action_id",
                            "draft_id",
                        ):
                            unique_value = _record_unique_id(
                                draft_copy.get(key) or metadata.get(key)
                            )
                            if unique_value:
                                break
                    if not unique_value and metadata:
                        for key in ("unique_id", "message_id", "id"):
                            unique_value = _record_unique_id(metadata.get(key))
                            if unique_value:
                                break
                    if not unique_value and session_reference:
                        unique_value = _record_unique_id(session_reference)
                    if unique_value:
                        draft_copy.setdefault("unique_id", unique_value)
                    candidate_drafts.append(draft_copy)

        candidate_drafts = self._ensure_supplier_id_in_drafts(candidate_drafts, supplier_id)

        if not candidate_drafts:
            logger.warning(
                "No valid drafts after supplier_id correction for workflow=%s, supplier=%s",
                workflow_id,
                supplier_id,
            )
            fallback_unique = session_reference or str(uuid.uuid4())
            fallback_entry = {
                "workflow_id": workflow_id,
                "supplier_id": supplier_id,
                "supplier": supplier_id,
                "unique_id": fallback_unique,
                "metadata": {
                    "supplier_id": supplier_id,
                    "supplier": supplier_id,
                    "workflow_id": workflow_id,
                },
            }
            if rfq_id:
                fallback_entry["rfq_id"] = rfq_id
                fallback_entry["metadata"]["rfq_id"] = rfq_id
            if session_reference:
                fallback_entry["session_reference"] = session_reference
            candidate_drafts = [fallback_entry]

        poll_interval = getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)

        workflow_hint = workflow_id or getattr(context, "workflow_id", None)
        if workflow_hint:
            for draft in candidate_drafts:
                if isinstance(draft, dict) and not draft.get("workflow_id"):
                    draft["workflow_id"] = workflow_hint
                if isinstance(draft, dict) and session_reference:
                    draft.setdefault("session_reference", session_reference)
                    draft.setdefault("unique_id", session_reference)
                if isinstance(draft, dict) and rfq_id and not draft.get("rfq_id"):
                    draft["rfq_id"] = rfq_id

        if candidate_drafts:
            unique_id_set = set()
            observed_unique_ids = []
            for draft in candidate_drafts:
                if not isinstance(draft, dict):
                    continue
                metadata = (
                    draft.get("metadata")
                    if isinstance(draft.get("metadata"), dict)
                    else {}
                )
                unique_value = _record_unique_id(draft.get("unique_id"))
                if not unique_value:
                    for key in (
                        "dispatch_run_id",
                        "run_id",
                        "email_action_id",
                        "action_id",
                        "draft_id",
                    ):
                        unique_value = _record_unique_id(
                            draft.get(key) or metadata.get(key)
                        )
                        if unique_value:
                            break
                if not unique_value and metadata:
                    for key in ("unique_id", "message_id", "id"):
                        unique_value = _record_unique_id(metadata.get(key))
                        if unique_value:
                            break
                if not unique_value and session_reference:
                    unique_value = _record_unique_id(session_reference)
                if unique_value:
                    draft.setdefault("unique_id", unique_value)

        expected_count = max(len(candidate_drafts), len(observed_unique_ids)) or 1

        watch_payload: Dict[str, Any] = {
            "await_response": True,
            "message": "",
            "body": "",
            "drafts": candidate_drafts,
            "supplier_id": supplier,
            "response_poll_interval": poll_interval,
            "workflow_id": workflow_hint,
            "expected_dispatch_count": expected_count,
            "expected_email_count": expected_count,
        }

        if rfq_id:
            watch_payload["rfq_id"] = rfq_id

        if session_reference:
            watch_payload.setdefault("session_reference", session_reference)
            watch_payload.setdefault("unique_id", session_reference)

        if observed_unique_ids:
            watch_payload["unique_ids"] = list(observed_unique_ids)
            watch_payload["expected_unique_ids"] = list(observed_unique_ids)

        batch_limit = getattr(self.agent_nick.settings, "email_response_batch_limit", None)
        if batch_limit:
            try:
                batch_value = int(batch_limit)
                if batch_value > 0:
                    watch_payload["response_batch_limit"] = batch_value
            except Exception:  # pragma: no cover - defensive
                logger.debug("Invalid email_response_batch_limit=%s", batch_limit)

        timeout_setting = getattr(self.agent_nick.settings, "email_response_timeout_seconds", None)
        if timeout_setting:
            try:
                timeout_value = int(timeout_setting)
                if timeout_value > 0:
                    watch_payload["response_timeout"] = timeout_value
            except Exception:  # pragma: no cover - defensive
                logger.debug("Invalid email_response_timeout_seconds=%s", timeout_setting)

        if len(watch_payload["drafts"]) > 1 or expected_count > 1:
            watch_payload["await_all_responses"] = True

        reply_count = state.get("supplier_reply_count")
        if reply_count is not None:
            watch_payload["supplier_reply_count"] = reply_count

        contact = context.input_data.get("from_address")
        if contact:
            watch_payload.setdefault("from_address", contact)

        sender = context.input_data.get("sender")
        if sender:
            watch_payload.setdefault("sender", sender)

        return watch_payload

    def _log_response_wait_diagnostics(
        self,
        *,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        drafts: List[Dict[str, Any]],
        watch_payload: Dict[str, Any],
    ) -> None:
        """Log detailed diagnostics for response waiting."""

        logger.info("=" * 80)
        logger.info("RESPONSE WAIT DIAGNOSTICS")
        logger.info("Workflow ID: %s", workflow_id)
        logger.info("Supplier ID: %s", supplier_id)
        logger.info("Number of drafts: %d", len(drafts))
        logger.info("Expected responses: %d", watch_payload.get("expected_email_count", 0))
        logger.info("Unique IDs being tracked: %s", watch_payload.get("unique_ids", []))
        logger.info("Await response flag: %s", watch_payload.get("await_response"))
        logger.info(
            "Await all responses flag: %s", watch_payload.get("await_all_responses")
        )

        for idx, draft in enumerate(drafts[:3]):
            logger.info("Draft %d:", idx)
            logger.info("  - unique_id: %s", draft.get("unique_id"))
            logger.info("  - supplier_id: %s", draft.get("supplier_id"))
            logger.info("  - workflow_id: %s", draft.get("workflow_id"))
            metadata = (
                draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
            )
            if isinstance(metadata, dict):
                logger.info(
                    "  - metadata.supplier_id: %s",
                    metadata.get("supplier_id"),
                )
        logger.info("=" * 80)

    def _await_supplier_responses(
        self,
        *,
        context: AgentContext,
        watch_payload: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[List[Optional[Dict[str, Any]]]]:
        if not watch_payload.get("await_response"):
            return []

        try:
            supplier_agent = self._get_supplier_agent()
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("Unable to initialise supplier interaction agent for wait")
            return None

        expected_dispatch = self._positive_int(
            watch_payload.get("expected_dispatch_count"), fallback=0
        )
        expected_email_count = self._positive_int(
            watch_payload.get("expected_email_count"), fallback=0
        )
        unique_expectations = [
            self._coerce_text(value)
            for value in self._ensure_list(
                watch_payload.get("expected_unique_ids")
                or watch_payload.get("unique_ids")
            )
            if self._coerce_text(value)
        ]
        expected_total = max(
            expected_dispatch,
            expected_email_count,
            len(unique_expectations),
        )

        timeout_default = getattr(
            self.agent_nick.settings, "email_response_timeout_seconds", 900
        )
        poll_default = getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)
        batch_default = getattr(self.agent_nick.settings, "email_response_batch_limit", 5)

        timeout_raw = watch_payload.get("response_timeout") or timeout_default
        poll_raw = watch_payload.get("response_poll_interval") or poll_default
        batch_raw = watch_payload.get("response_batch_limit") or batch_default

        timeout = self._positive_int(timeout_raw, fallback=timeout_default)
        poll_interval = self._positive_int(poll_raw, fallback=poll_default)
        batch_limit = self._positive_int(batch_raw, fallback=batch_default)
        if expected_total and batch_limit < expected_total:
            batch_limit = expected_total

        draft_entries: List[Dict[str, Any]] = []
        for draft in watch_payload.get("drafts", []):
            if isinstance(draft, dict):
                draft_copy = dict(draft)
                draft_unique = self._coerce_text(draft_copy.get("unique_id"))
                if not draft_unique and unique_expectations:
                    for candidate in unique_expectations:
                        if not candidate:
                            continue
                        if any(
                            self._coerce_text(entry.get("unique_id")) == candidate
                            for entry in draft_entries
                        ):
                            continue
                        draft_unique = candidate
                        break
                if draft_unique:
                    draft_copy.setdefault("unique_id", draft_unique)
                draft_entries.append(draft_copy)

        workflow_hint_raw = watch_payload.get("workflow_id") or getattr(
            context, "workflow_id", None
        )

        if not draft_entries:
            fallback_entry = {}
            if workflow_hint_raw:
                fallback_entry["workflow_id"] = workflow_hint_raw
            supplier_hint = watch_payload.get("supplier_id") or watch_payload.get("supplier")
            if supplier_hint:
                fallback_entry["supplier_id"] = supplier_hint
            if fallback_entry:
                draft_entries.append(fallback_entry)

        if not draft_entries:
            logger.warning("No draft context available while awaiting supplier response")
            return None

        workflow_hint = self._coerce_text(workflow_hint_raw)

        if draft_entries:
            logger.info(
                "Awaiting supplier responses for workflow=%s with %s drafts",
                workflow_hint or workflow_hint_raw,
                len(draft_entries),
            )
            for idx, entry in enumerate(draft_entries[:5]):
                logger.info(
                    "  Draft %s: unique=%s supplier=%s workflow=%s",
                    idx,
                    entry.get("unique_id"),
                    entry.get("supplier_id"),
                    entry.get("workflow_id"),
                )

        if draft_entries and len(draft_entries) > 1:
            draft_workflows = {
                self._coerce_text(entry.get("workflow_id"))
                for entry in draft_entries
                if self._coerce_text(entry.get("workflow_id"))
            }
            if len(draft_workflows) > 1:
                logger.error(
                    "CRITICAL: Multiple workflows in draft batch! workflows=%s",
                    sorted(draft_workflows),
                )
            elif len(draft_workflows) == 1 and workflow_hint:
                draft_workflow = draft_workflows.pop()
                if draft_workflow != workflow_hint:
                    logger.error(
                        "Workflow mismatch: context has %s but drafts have %s",
                        workflow_hint,
                        draft_workflow,
                    )

        if workflow_hint and draft_entries:
            filtered_entries: List[Dict[str, Any]] = []
            dropped_entries: List[Dict[str, Any]] = []
            for entry in draft_entries:
                entry_workflow = self._coerce_text(entry.get("workflow_id"))
                if entry_workflow and entry_workflow != workflow_hint:
                    dropped_entries.append(entry)
                    continue
                filtered_entries.append(entry)

            if dropped_entries:
                logger.warning(
                    "Discarding %s drafts due to workflow mismatch with %s",
                    len(dropped_entries),
                    workflow_hint,
                )
                for idx, entry in enumerate(dropped_entries[:5]):
                    logger.warning(
                        "  Dropped draft %s: unique=%s supplier=%s workflow=%s",
                        idx,
                        entry.get("unique_id"),
                        entry.get("supplier_id"),
                        entry.get("workflow_id"),
                    )
            if filtered_entries:
                draft_entries = filtered_entries
            elif dropped_entries:
                adopted_workflows = {
                    self._coerce_text(entry.get("workflow_id"))
                    for entry in dropped_entries
                    if self._coerce_text(entry.get("workflow_id"))
                }
                if len(adopted_workflows) == 1:
                    adopted_workflow = adopted_workflows.pop()
                    logger.warning(
                        "Adopting draft workflow_id=%s in place of context workflow_id=%s",
                        adopted_workflow,
                        workflow_hint,
                    )
                    workflow_hint = adopted_workflow
                    draft_entries = dropped_entries
                else:
                    logger.error(
                        "All provided drafts conflicted with workflow=%s; aborting await",
                        workflow_hint,
                    )
                    return None

        await_all = bool(watch_payload.get("await_all_responses") and len(draft_entries) > 1)
        tracked_unique_ids = sorted(
            {
                self._coerce_text(entry.get("unique_id"))
                for entry in draft_entries
                if self._coerce_text(entry.get("unique_id"))
            }
        )
        if unique_expectations and not tracked_unique_ids:
            tracked_unique_ids = sorted({value for value in unique_expectations if value})
        if expected_total and expected_total < len(draft_entries):
            expected_total = len(draft_entries)
        elif not expected_total:
            expected_total = len(draft_entries)

        try:
            if await_all:
                results = supplier_agent.wait_for_multiple_responses(
                    draft_entries,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    enable_negotiation=False,
                )
                valid_results = [res for res in results if isinstance(res, dict)]
                if expected_total and len(valid_results) < expected_total:
                    logger.warning(
                        "Awaited %s supplier responses but only received %s (workflow_id=%s unique_ids=%s)",
                        expected_total,
                        len(valid_results),
                        workflow_hint,
                        tracked_unique_ids or unique_expectations or None,
                    )
                    return None
                if unique_expectations:
                    observed = {
                        self._coerce_text(res.get("unique_id"))
                        for res in valid_results
                        if self._coerce_text(res.get("unique_id"))
                    }
                    missing = [
                        value
                        for value in unique_expectations
                        if value and value not in observed
                    ]
                    if missing:
                        logger.warning(
                            "Missing supplier responses for unique_ids=%s (workflow_id=%s)",
                            missing,
                            workflow_hint,
                        )
                        return None
                return results

            target = draft_entries[0]
            recipient_hint = target.get("receiver") or target.get("recipient_email")
            if not recipient_hint:
                recipients_field = target.get("recipients")
                if isinstance(recipients_field, list) and recipients_field:
                    recipient_hint = recipients_field[0]
                elif isinstance(recipients_field, str):
                    recipient_hint = recipients_field

            draft_action_id = None
            for key in ("action_id", "draft_action_id", "email_action_id"):
                candidate = target.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    draft_action_id = candidate.strip()
                    break
            workflow_hint = workflow_hint or target.get("workflow_id")
            if not workflow_hint and isinstance(target.get("metadata"), dict):
                meta_workflow = target["metadata"].get("workflow_id") or target["metadata"].get("process_workflow_id")
                if isinstance(meta_workflow, str) and meta_workflow.strip():
                    workflow_hint = meta_workflow.strip()
            dispatch_run_id = None
            for key in ("dispatch_run_id", "run_id"):
                candidate = target.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    dispatch_run_id = candidate.strip()
                    break
            if dispatch_run_id is None and isinstance(target.get("metadata"), dict):
                meta_run = target["metadata"].get("dispatch_run_id") or target["metadata"].get("run_id")
                if isinstance(meta_run, str) and meta_run.strip():
                    dispatch_run_id = meta_run.strip()

            if not workflow_hint:
                logger.warning(
                    "Cannot await supplier response without workflow_id for supplier=%s",
                    target.get("supplier_id"),
                )
                return None

            return [
                supplier_agent.wait_for_response(
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    supplier_id=target.get("supplier_id"),
                    subject_hint=target.get("subject"),
                    from_address=recipient_hint,
                    enable_negotiation=False,
                    draft_action_id=draft_action_id,
                    workflow_id=workflow_hint,
                    dispatch_run_id=dispatch_run_id,
                    unique_id=target.get("unique_id"),
                )
            ]
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed while waiting for supplier responses (workflow_id=%s, supplier=%s)",
                workflow_hint,
                watch_payload.get("supplier_id"),
            )
            return None

    def _get_supplier_agent(self) -> "SupplierInteractionAgent":
        if self._supplier_agent is None:
            with self._supplier_agent_lock:
                if self._supplier_agent is None:
                    from agents.supplier_interaction_agent import SupplierInteractionAgent

                    self._supplier_agent = SupplierInteractionAgent(self.agent_nick)
        return self._supplier_agent

    def _build_stop_message(self, status: str, reason: str, round_no: int) -> str:
        status_text = status.capitalize()
        reason_text = reason or "No further action required."
        return f"Negotiation {status_text} after round {round_no}: {reason_text}"

    def _store_session(
        self, session_id: str, supplier: str, round_no: int, counter_price: Optional[float]
    ) -> None:
        """Persist negotiation round details with proper constraint handling."""
        identifier = self._coerce_text(session_id)
        supplier_id = self._coerce_text(supplier)
        if not identifier or not supplier_id:
            logger.warning(
                "Cannot store session without identifier=%s and supplier=%s", identifier, supplier_id
            )
            return

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                            SELECT constraint_name, constraint_type
                            FROM information_schema.table_constraints
                            WHERE table_schema = 'proc'
                              AND table_name = 'negotiation_sessions'
                              AND constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                        """
                    )
                    constraints = cur.fetchall()

                    constraint_columns: Dict[str, Tuple[str, ...]] = {}
                    for constraint_name, _ in constraints:
                        cur.execute(
                            """
                                SELECT column_name
                                FROM information_schema.key_column_usage
                                WHERE table_schema = 'proc'
                                  AND table_name = 'negotiation_sessions'
                                  AND constraint_name = %s
                                ORDER BY ordinal_position
                            """,
                            (constraint_name,),
                        )
                        cols = tuple(row[0] for row in cur.fetchall())
                        if cols:
                            constraint_columns[constraint_name] = cols

                    cur.execute(
                        """
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'proc'
                              AND table_name = 'negotiation_sessions'
                              AND column_name IN ('workflow_id', 'rfq_id', 'unique_id', 'session_id')
                        """
                    )
                    available_id_columns = [row[0] for row in cur.fetchall()]

                    if not available_id_columns:
                        logger.error("No identifier column found in negotiation_sessions table")
                        return

                    id_column = next(
                        (
                            col
                            for col in [
                                "workflow_id",
                                "rfq_id",
                                "unique_id",
                                "session_id",
                            ]
                            if col in available_id_columns
                        ),
                        available_id_columns[0],
                    )

                    record: Dict[str, Any] = {
                        id_column: identifier,
                        "supplier_id": supplier_id,
                        "round": int(round_no or 1),
                        "counter_offer": counter_price,
                        "created_on": datetime.now(timezone.utc),
                    }

                    conflict_columns: Optional[Tuple[str, ...]] = None
                    for cols in constraint_columns.values():
                        if all(col in record for col in cols):
                            conflict_columns = cols
                            break

                    if not conflict_columns:
                        logger.warning(
                            "No matching unique constraint for negotiation_sessions; using INSERT with duplicate check"
                        )
                        cur.execute(
                            f"""
                                SELECT 1 FROM proc.negotiation_sessions
                                WHERE {id_column} = %s AND supplier_id = %s AND round = %s
                            """,
                            (identifier, supplier_id, int(round_no or 1)),
                        )
                        if cur.fetchone():
                            cur.execute(
                                f"""
                                    UPDATE proc.negotiation_sessions
                                    SET counter_offer = %s
                                    WHERE {id_column} = %s AND supplier_id = %s AND round = %s
                                """,
                                (counter_price, identifier, supplier_id, int(round_no or 1)),
                            )
                        else:
                            columns = list(record.keys())
                            values = [record[col] for col in columns]
                            cur.execute(
                                f"""
                                    INSERT INTO proc.negotiation_sessions ({', '.join(columns)})
                                    VALUES ({', '.join(['%s'] * len(columns))})
                                """,
                                tuple(values),
                            )
                    else:
                        columns = list(record.keys())
                        values = [record[col] for col in columns]
                        conflict_clause = ", ".join(conflict_columns)
                        update_cols = [
                            col
                            for col in columns
                            if col not in conflict_columns and col != "created_on"
                        ]

                        if update_cols:
                            update_clause = ", ".join(
                                f"{col} = EXCLUDED.{col}" for col in update_cols
                            )
                            sql = f"""
                                INSERT INTO proc.negotiation_sessions ({', '.join(columns)})
                                VALUES ({', '.join(['%s'] * len(columns))})
                                ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
                            """
                        else:
                            sql = f"""
                                INSERT INTO proc.negotiation_sessions ({', '.join(columns)})
                                VALUES ({', '.join(['%s'] * len(columns))})
                                ON CONFLICT ({conflict_clause}) DO NOTHING
                            """

                        cur.execute(sql, tuple(values))

                conn.commit()
                logger.info(
                    "Successfully stored negotiation session: %s, supplier=%s, round=%s",
                    identifier,
                    supplier_id,
                    round_no,
                )
        except Exception as e:  # pragma: no cover - best effort
            logger.exception(
                "Failed to store negotiation session (id=%s, supplier=%s, round=%s): %s",
                identifier,
                supplier_id,
                round_no,
                str(e),
            )

    def _record_learning_snapshot(
        self,
        context: AgentContext,
        session_id: Optional[str] = None,
        supplier: Optional[str] = None,
        decision: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        awaiting_response: bool = False,
        supplier_reply_registered: bool = False,
        rfq_id: Optional[str] = None,
    ) -> None:
        workflow_id = getattr(context, "workflow_id", None)
        memory = getattr(self.agent_nick, "workflow_memory", None)
        event_payload = {
            "category": "negotiation_snapshot",
            "rfq_id": rfq_id,
            "session_id": session_id,
            "supplier_id": supplier,
            "decision": dict(decision or {}),
            "state": dict(state or {}),
            "awaiting_response": awaiting_response,
            "supplier_reply_registered": supplier_reply_registered,
        }
        if memory and getattr(memory, "enabled", False) and workflow_id:
            try:
                memory.enqueue_learning_event(workflow_id, event_payload)
            except Exception:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to enqueue negotiation learning for workflow=%s",
                    workflow_id,
                    exc_info=True,
                )
            return

        repository = getattr(self.agent_nick, "learning_repository", None)
        if not repository:
            return
        try:
            repository.record_negotiation_learning(
                workflow_id=workflow_id,
                rfq_id=rfq_id,
                supplier_id=supplier,
                decision=decision or {},
                state=state or {},
                awaiting_response=awaiting_response,
                supplier_reply_registered=supplier_reply_registered,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to record negotiation learning for workflow=%s supplier=%s",
                workflow_id,
                supplier,
                exc_info=True,
            )

    def _collect_recipient_candidates(self, context: AgentContext) -> List[str]:
        seen: Set[str] = set()
        recipients: List[str] = []
        payload = context.input_data

        def _append(candidate: Any) -> None:
            email = self._coerce_text(candidate)
            if not email:
                return
            key = email.lower()
            if key in seen:
                return
            seen.add(key)
            recipients.append(email)

        for key in (
            "recipients",
            "recipient_email",
            "recipient",
            "supplier_contact_email",
            "supplier_email",
            "email",
            "contact_email",
            "contact_email_1",
            "contact_email_2",
        ):
            value = payload.get(key)
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _append(item)
            else:
                _append(value)

        contacts = payload.get("supplier_contacts")
        if isinstance(contacts, list):
            for contact in contacts:
                if not isinstance(contact, dict):
                    continue
                _append(contact.get("email"))
                _append(contact.get("contact_email"))

        drafts = payload.get("drafts")
        if isinstance(drafts, list):
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                _append(draft.get("recipient_email"))
                _append(draft.get("receiver"))
                draft_recipients = draft.get("recipients")
                if isinstance(draft_recipients, list):
                    for item in draft_recipients:
                        _append(item)

        return recipients

    def _collect_supplier_snippets(self, payload: Dict[str, Any]) -> List[str]:
        snippets: List[str] = []
        for key in (
            "supplier_snippets",
            "snippets",
            "highlights",
            "supplier_highlights",
            "response_text",
            "message",
            "raw_email",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                snippets.extend(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                snippets.append(value.strip())
        return snippets[:5]

    def _ensure_supplier_id_in_drafts(
        self,
        drafts: List[Dict[str, Any]],
        fallback_supplier_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Ensure all drafts have supplier_id populated."""

        corrected_drafts: List[Dict[str, Any]] = []

        for draft in drafts:
            if not isinstance(draft, dict):
                continue

            draft_copy = dict(draft)

            supplier_id = draft_copy.get("supplier_id") or draft_copy.get("supplier")

            metadata = draft_copy.get("metadata")
            if not supplier_id and isinstance(metadata, dict):
                supplier_id = metadata.get("supplier_id") or metadata.get("supplier")

            if not supplier_id:
                supplier_id = fallback_supplier_id

            if not supplier_id:
                logger.warning("Draft missing supplier_id even after fallback: %s", draft_copy)
                continue

            draft_copy["supplier_id"] = supplier_id
            draft_copy.setdefault("supplier", supplier_id)

            if "metadata" not in draft_copy or not isinstance(draft_copy["metadata"], dict):
                draft_copy["metadata"] = {}

            draft_metadata = cast(Dict[str, Any], draft_copy["metadata"])
            draft_metadata["supplier_id"] = supplier_id
            draft_metadata.setdefault("supplier", supplier_id)

            corrected_drafts.append(draft_copy)

        return corrected_drafts

    def _resolve_contact_name(
        self, payload: Optional[Dict[str, Any]], *, fallback: Optional[str] = None
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            payload = {}

        candidate_keys = (
            "contact_name",
            "contact_person",
            "contact",
            "supplier_contact",
            "supplier_contact_name",
            "primary_contact_name",
            "recipient_name",
            "to_name",
            "attention_to",
            "contact_full_name",
            "contact_name_1",
            "contact_name_2",
        )
        for key in candidate_keys:
            if key not in payload:
                continue
            resolved = self._normalise_contact_name(payload.get(key))
            if resolved:
                return resolved

        nested_keys = (
            "name",
            "full_name",
            "display_name",
            "contact_name",
            "first_name",
            "last_name",
        )
        for key in ("primary_contact", "contact", "supplier_contact"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                for nested_key in nested_keys:
                    resolved = self._normalise_contact_name(nested.get(nested_key))
                    if resolved:
                        return resolved

        contacts = payload.get("supplier_contacts")
        if isinstance(contacts, list):
            for contact in contacts:
                if not isinstance(contact, dict):
                    continue
                for nested_key in nested_keys:
                    resolved = self._normalise_contact_name(contact.get(nested_key))
                    if resolved:
                        return resolved

        email_keys = (
            "recipient_email",
            "receiver",
            "recipients",
            "supplier_contact_email",
            "contact_email",
            "from_address",
            "supplier_email",
            "email",
        )
        for key in email_keys:
            candidate = payload.get(key)
            if isinstance(candidate, (list, tuple, set)):
                items = candidate
            else:
                items = (candidate,)
            for item in items:
                parsed = self._extract_name_from_email(item)
                if parsed:
                    return parsed

        if fallback:
            fallback_name = self._normalise_contact_name(fallback)
            if fallback_name and not self._is_likely_identifier(fallback_name):
                return fallback_name
            parsed = self._extract_name_from_email(fallback)
            if parsed:
                return parsed

        return None

    def _normalise_contact_name(self, value: Any) -> Optional[str]:
        text = self._coerce_text(value)
        if not text:
            return None
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip(" ,.;")
        if not text:
            return None
        if "@" in text and " " not in text:
            return None
        return text

    @staticmethod
    def _is_likely_identifier(value: str) -> bool:
        token = value.strip()
        if not token:
            return False
        return bool(re.match(r"^[A-Z]{2,}[A-Z0-9._-]*$", token))

    def _extract_name_from_email(self, value: Any) -> Optional[str]:
        email = self._coerce_text(value)
        if not email or "@" not in email:
            return None
        local = email.split("@", 1)[0]
        tokens = [
            token
            for token in re.split(r"[._+\-]", local)
            if token and token.isalpha()
        ]
        if not tokens:
            return None
        return " ".join(token.capitalize() for token in tokens)

    @staticmethod
    def _has_explicit_greeting(message: Optional[str]) -> bool:
        if not message:
            return False
        snippet = message.lstrip()
        lowered = snippet.lower()
        greeting_prefixes = (
            "dear ",
            "hi ",
            "hello ",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
        )
        return any(lowered.startswith(prefix) for prefix in greeting_prefixes)

    def _build_personal_greeting(
        self, contact_name: Optional[str], supplier_name: Optional[str]
    ) -> Optional[str]:
        for candidate in (contact_name, supplier_name):
            if not candidate:
                continue
            resolved = self._normalise_contact_name(candidate)
            if resolved and not self._is_likely_identifier(resolved):
                return f"Dear {resolved},"
            parsed = self._extract_name_from_email(candidate)
            if parsed:
                return f"Dear {parsed},"
        return None

    def _build_decision_log(
        self,
        supplier: Optional[str],
        session_reference: Optional[str],
        price: Optional[float],
        target_price: Optional[float],
        decision: Dict[str, Any],
    ) -> str:
        base = (
            f"Strategy={decision.get('strategy')} counter={decision.get('counter_price')}"
            f" target={target_price} current={price} supplier={supplier} reference={session_reference}."
        )
        plays = decision.get("play_recommendations") or []
        if plays:
            top_snippets: List[str] = []
            for play in plays[:3]:
                if not isinstance(play, dict):
                    continue
                lever = play.get("lever") or play.get("category")
                description = play.get("play") or play.get("description")
                if not description:
                    continue
                if lever:
                    top_snippets.append(f"{lever}: {description}")
                else:
                    top_snippets.append(str(description))
            if top_snippets:
                base = f"{base} Plays={' | '.join(top_snippets)}."
        rationale = decision.get("rationale")
        if rationale:
            return f"{base} {rationale}"
        return base

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _normalise_negotiation_inputs(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        normalised: Dict[str, Any] = {}
        issues: List[str] = []

        price_sources = [
            payload.get("current_offer"),
            payload.get("price"),
            payload.get("supplier_offer"),
        ]
        target_sources = [payload.get("target_price"), payload.get("target")]
        walkaway_sources = [payload.get("walkaway_price"), payload.get("walkaway"), payload.get("no_deal_price")]
        previous_offer_sources = [
            payload.get("previous_offer"),
            payload.get("supplier_previous_offer"),
            payload.get("last_supplier_offer"),
            payload.get("offer_prev"),
        ]

        def _pick_first(values: List[Any], parser) -> Optional[float]:
            for candidate in values:
                parsed = parser(candidate)
                if parsed is not None:
                    return parsed
            return None

        normalised["current_offer"] = _pick_first(price_sources, self._parse_money)
        normalised["target_price"] = _pick_first(target_sources, self._parse_money)
        normalised["walkaway_price"] = _pick_first(walkaway_sources, self._parse_money)
        normalised["previous_offer"] = _pick_first(previous_offer_sources, self._parse_money)

        currency = (
            payload.get("currency")
            or payload.get("current_offer_currency")
            or payload.get("price_currency")
        )
        normalised["currency"] = self._normalise_currency(currency)

        lead_sources = [
            payload.get("lead_time_weeks"),
            payload.get("lead_time"),
            payload.get("lead_time_days"),
        ]
        lead_weeks = None
        for candidate in lead_sources:
            lead_weeks = self._parse_lead_weeks(candidate)
            if lead_weeks is not None:
                break
        normalised["lead_time_weeks"] = lead_weeks

        volume_sources = [
            payload.get("volume"),
            payload.get("volume_commitment"),
            payload.get("order_volume"),
            payload.get("order_quantity"),
            payload.get("quantity"),
            payload.get("units"),
        ]
        normalised["volume_units"] = _pick_first(volume_sources, self._parse_quantity)

        term_sources = [
            payload.get("term_days"),
            payload.get("payment_terms"),
            payload.get("payment_term"),
            payload.get("terms"),
        ]
        normalised["term_days"] = _pick_first(term_sources, self._parse_term_days)

        valid_until_sources = [
            payload.get("valid_until"),
            payload.get("offer_valid_until"),
            payload.get("validity"),
            payload.get("expiration_date"),
            payload.get("expiry_date"),
        ]
        valid_until = None
        for candidate in valid_until_sources:
            valid_until = self._parse_date(candidate)
            if valid_until is not None:
                break
        normalised["valid_until"] = valid_until

        reference_prices = self._extract_reference_prices(payload)
        normalised.update(reference_prices)

        essential_labels = {
            "current_offer": "Supplier offer",
            "target_price": "Target price",
        }
        for field, label in essential_labels.items():
            if normalised.get(field) is None:
                issues.append(f"{label} missing or invalid")

        return normalised, issues

    def _parse_money(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            cleaned = re.sub(r"[\s,]", "", text)
            match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    def _parse_quantity(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    def _parse_term_days(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidate = int(round(float(value)))
            return candidate if candidate > 0 else None
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            if not numbers:
                return None
            try:
                numeric = float(numbers[0])
            except ValueError:
                return None
            if "week" in text and numeric > 0:
                return int(round(numeric * 7))
            return int(round(numeric)) if numeric > 0 else None
        return None

    def _parse_date(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for fmt in (
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%d %b %Y",
                "%b %d, %Y",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    return dt.replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    continue
            try:
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                return None
        return None

    def _extract_reference_prices(self, payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
        candidates: List[float] = []

        def _ingest(value: Any) -> None:
            parsed = self._parse_money(value)
            if parsed is not None:
                candidates.append(parsed)

        for key in ("benchmarks", "market_context", "market_prices", "history"):
            source = payload.get(key)
            if isinstance(source, dict):
                for sub_key, sub_value in source.items():
                    lowered = str(sub_key).lower()
                    if any(
                        token in lowered
                        for token in ("price", "p10", "p25", "low", "min", "floor")
                    ):
                        _ingest(sub_value)
            elif isinstance(source, list):
                for item in source:
                    if isinstance(item, dict):
                        for sub_value in item.values():
                            _ingest(sub_value)
                    else:
                        _ingest(item)

        lowest_market = min(candidates) if candidates else None
        return {"market_floor_price": lowest_market}

    def _build_positions(
        self,
        *,
        supplier_offer: Optional[float],
        target_price: Optional[float],
        walkaway_price: Optional[float],
        previous_counter: Optional[float],
        previous_positions: Optional[Dict[str, Any]],
        round_no: int,
    ) -> NegotiationPositions:
        history: List[Dict[str, Any]] = []
        if isinstance(previous_positions, dict):
            stored_history = previous_positions.get("history")
            if isinstance(stored_history, list):
                history = [
                    entry
                    for entry in stored_history
                    if isinstance(entry, dict) and entry.get("round") is not None
                ]
        desired = self._coerce_float(target_price)
        if desired is None and isinstance(previous_positions, dict):
            desired = self._coerce_float(previous_positions.get("desired"))
        no_deal = self._coerce_float(walkaway_price)
        if no_deal is None and isinstance(previous_positions, dict):
            no_deal = self._coerce_float(previous_positions.get("no_deal"))

        candidate_starts: List[Optional[float]] = []
        candidate_starts.append(self._coerce_float(previous_counter))
        if isinstance(previous_positions, dict):
            candidate_starts.append(self._coerce_float(previous_positions.get("last_counter")))
            candidate_starts.append(self._coerce_float(previous_positions.get("start")))
        candidate_starts.append(self._coerce_float(supplier_offer))

        start_value: Optional[float] = None
        for candidate in candidate_starts:
            if candidate is not None:
                start_value = candidate
                break

        positions = NegotiationPositions(
            start=start_value,
            desired=desired,
            no_deal=no_deal,
            supplier_offer=self._coerce_float(supplier_offer),
            history=history,
        )

        if positions.supplier_offer is not None:
            positions.history.append(
                {
                    "round": round_no,
                    "type": "supplier_offer",
                    "value": positions.supplier_offer,
                }
            )

        return positions

    def _respect_positions(
        self, counter: Optional[float], positions: NegotiationPositions
    ) -> Optional[float]:
        if counter is None:
            return None
        try:
            candidate = float(counter)
        except (TypeError, ValueError):
            return None

        if positions.desired is not None:
            try:
                candidate = max(candidate, float(positions.desired))
            except (TypeError, ValueError):
                pass
        if positions.no_deal is not None:
            try:
                candidate = min(candidate, float(positions.no_deal))
            except (TypeError, ValueError):
                pass
        if positions.start is not None:
            try:
                candidate = min(candidate, float(positions.start))
            except (TypeError, ValueError):
                pass

        return round(candidate, 2)

    def _detect_outliers(
        self,
        *,
        supplier_offer: Optional[float],
        target_price: Optional[float],
        walkaway_price: Optional[float],
        market_floor: Optional[float],
        volume_units: Optional[float],
        term_days: Optional[int],
    ) -> Dict[str, Any]:
        alerts: List[str] = []
        requires_review = False
        human_override = False
        review_recommendation: Optional[str] = None
        rationale_notes: List[str] = []

        def _percentage_gap(reference: Optional[float], offer: Optional[float]) -> Optional[float]:
            try:
                if reference is None or offer is None or reference <= 0:
                    return None
                return (reference - offer) / reference
            except (TypeError, ValueError):
                return None

        market_gap = _percentage_gap(market_floor, supplier_offer)
        walkaway_gap = _percentage_gap(walkaway_price, supplier_offer)

        if market_gap is not None and market_gap >= MARKET_REVIEW_THRESHOLD:
            requires_review = True
            review_recommendation = "query_for_human_review"
            alerts.append(
                f"Supplier offer is {market_gap * 100:.1f}% below market reference {market_floor:.2f}."
            )
            rationale_notes.append("Requested price is materially below market benchmarks; seek justification.")
            if market_gap >= MARKET_ESCALATION_THRESHOLD:
                human_override = True
                rationale_notes.append(
                    "Supplier offer breaches escalation threshold relative to market floor."
                )

        if walkaway_gap is not None and walkaway_gap >= MARKET_REVIEW_THRESHOLD:
            requires_review = True
            review_recommendation = review_recommendation or "query_for_human_review"
            alerts.append(
                f"Supplier offer is {walkaway_gap * 100:.1f}% below walk-away price {walkaway_price:.2f}."
            )
            rationale_notes.append(
                "Requested price undercuts internal walk-away guardrail; confirm intent before proceeding."
            )
            if walkaway_gap >= MARKET_ESCALATION_THRESHOLD:
                human_override = True
                rationale_notes.append(
                    "The requested price is more than 20% below our walk-away price; escalation required."
                )

        if volume_units is not None and volume_units > MAX_VOLUME_LIMIT:
            requires_review = True
            review_recommendation = review_recommendation or "query_for_human_review"
            alerts.append(
                f"Requested volume {volume_units:.0f} exceeds configured limit {MAX_VOLUME_LIMIT:.0f}."
            )
            rationale_notes.append("Request supplier rationale for above-capacity volume.")
            if volume_units > MAX_VOLUME_LIMIT * 1.5:
                human_override = True
                rationale_notes.append("Volume exceeds escalation ceiling; seek human approval.")

        if term_days is not None and term_days > MAX_TERM_DAYS:
            requires_review = True
            review_recommendation = review_recommendation or "query_for_human_review"
            alerts.append(
                f"Requested payment term {term_days} days exceeds policy limit {MAX_TERM_DAYS} days."
            )
            rationale_notes.append("Payment term exceeds policy; confirm via human review.")
            if term_days > MAX_TERM_DAYS * 2:
                human_override = True
                rationale_notes.append("Payment term far exceeds tolerance; human intervention required.")

        message = None
        if human_override:
            for note in rationale_notes[::-1]:
                if "escalation required" in note.lower():
                    message = note
                    break
            message = message or "Escalation required before proceeding."
        elif requires_review and rationale_notes:
            message = rationale_notes[-1]

        return {
            "alerts": alerts,
            "requires_review": requires_review,
            "human_override": human_override,
            "recommendation": review_recommendation,
            "message": message,
        }

    def _compose_rationale(
        self,
        *,
        positions: NegotiationPositions,
        decision: Dict[str, Any],
        currency: Optional[str],
        lead_weeks: Optional[float],
        volume_units: Optional[float],
        term_days: Optional[int],
        outlier_message: Optional[str],
        validation_issues: List[str],
    ) -> str:
        counter_price = decision.get("counter_price")
        parts: List[str] = []

        def _format(amount: Optional[float]) -> Optional[str]:
            if amount is None:
                return None
            text = self._format_currency(amount, currency)
            return text or f"{amount:0.2f}"

        start_text = _format(positions.start)
        target_text = _format(positions.desired)
        counter_text = _format(counter_price)

        if start_text and target_text and counter_text:
            rationale = (
                f"Because the starting price is {start_text} and the target is {target_text}, "
                f"we counter at {counter_text}"
            )
        elif counter_text:
            rationale = f"We counter at {counter_text} based on the available pricing guardrails"
        else:
            rationale = "Pricing inputs incomplete; request clarification before committing to a counter"

        additions: List[str] = []
        if term_days:
            additions.append(f"{int(term_days)}-day term")
        if volume_units:
            additions.append(f"{int(round(volume_units))}-unit volume")
        if lead_weeks:
            additions.append(f"{lead_weeks:.1f}-week lead time request")
        if additions and counter_text:
            rationale += " with " + " and ".join(additions)

        parts.append(rationale + ".")

        if validation_issues:
            issue_text = "; ".join(sorted(set(validation_issues)))
            parts.append(f"Data validation flagged: {issue_text}.")

        if outlier_message:
            parts.append(outlier_message)

        return " ".join(part for part in parts if part)

    def _format_currency(self, value: Optional[float], currency: Optional[str]) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return ""
        code = (currency or "GBP").upper()
        symbol = (
            "£"
            if code == "GBP"
            else "$"
            if code == "USD"
            else "€"
            if code == "EUR"
            else "₹"
            if code == "INR"
            else ""
        )
        formatted = f"{amount:,.2f}"
        return f"{symbol}{formatted}" if symbol else f"{formatted} {code}"

    def _normalise_currency(self, value: Any) -> Optional[str]:
        if not value:
            return None
        if isinstance(value, str):
            trimmed = value.strip().upper()
            if len(trimmed) == 3:
                return trimmed
        return None

    def _parse_lead_weeks(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
            return number if number <= 12 else round(number / 7.0, 2)
        except ValueError:
            pass
        lowered = text.lower()
        digits = "".join(ch for ch in lowered if (ch.isdigit() or ch == "."))
        try:
            numeric = float(digits)
        except ValueError:
            return None
        if "week" in lowered:
            return numeric
        if "day" in lowered or "business" in lowered:
            return round(numeric / 7.0, 2)
        return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_buyer_max(self, value: Any) -> Optional[float]:
        parsed = self._coerce_float(value)
        if parsed is None:
            return None
        if not math.isfinite(parsed):
            return None
        if parsed <= 0:
            return None
        return parsed

    def _positive_int(self, value: Any, *, fallback: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return fallback
        return parsed if parsed > 0 else fallback

    def _coerce_text(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        return None

    def _ensure_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _ensure_email_agent(self) -> Optional[EmailDraftingAgent]:
        try:
            if self._email_agent is None:
                with self._email_agent_lock:
                    if self._email_agent is None:
                        self._email_agent = EmailDraftingAgent(self.agent_nick)
        except Exception:
            logger.debug("Unable to initialise EmailDraftingAgent", exc_info=True)
            return None
        return self._email_agent

    @staticmethod
    def _simple_html_from_text(text: str) -> str:
        lines = text.splitlines()
        html_parts: List[str] = []
        bullets: List[str] = []

        def flush() -> None:
            if bullets:
                items = "".join(f"<li>{escape(item)}</li>" for item in bullets)
                html_parts.append(f"<ul>{items}</ul>")
                bullets.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            if re.match(r"^[-*•]\s+", stripped):
                bullets.append(stripped[1:].strip())
                continue
            flush()
            html_parts.append(f"<p>{escape(stripped)}</p>")
        flush()
        return "".join(html_parts)

    @staticmethod
    def _normalise_recipient_list(value: Any) -> List[str]:
        if value is None:
            return []
        candidates: List[str] = []
        if isinstance(value, str):
            tokens = re.split(r"[;,]", value)
            candidates.extend(token.strip() for token in tokens if token.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                if isinstance(item, str):
                    tokens = re.split(r"[;,]", item)
                    candidates.extend(token.strip() for token in tokens if token.strip())
        return candidates

    @staticmethod
    def _merge_recipients_basic(to_list: List[str], cc_list: List[str]) -> List[str]:
        merged: List[str] = []
        seen: Set[str] = set()
        for addr in list(to_list) + list(cc_list):
            candidate = addr.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
        return merged

    def _build_negotiation_html_shell(
        self,
        *,
        subject: Optional[str],
        cleaned_body: str,
        email_agent: Optional[EmailDraftingAgent],
    ) -> Tuple[str, str]:
        if not cleaned_body:
            return "", ""

        safe_subject = escape((subject or "Negotiation Update").strip() or "Negotiation Update")

        html_candidate = ""
        builder_cls = NegotiationEmailHTMLShellBuilder
        if builder_cls and hasattr(builder_cls, "build"):
            try:
                builder = builder_cls()
                html_candidate = builder.build(subject=subject, body_text=cleaned_body)
            except Exception:
                logger.debug(
                    "Failed to render negotiation HTML shell with shell builder",
                    exc_info=True,
                )

        if not html_candidate:
            body_markup = self._simple_html_from_text(cleaned_body)
            if not body_markup and cleaned_body:
                body_markup = f"<p>{escape(cleaned_body)}</p>"
            html_candidate = (
                "<!DOCTYPE html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"utf-8\"/>\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>\n"
                f"  <title>{safe_subject}</title>\n"
                "</head>\n"
                "<body>\n"
                f"{body_markup}\n"
                "</body>\n"
                "</html>\n"
            )

        sanitised_html = html_candidate or ""
        if email_agent and sanitised_html:
            try:
                sanitised_html = email_agent._sanitise_generated_body(sanitised_html) or sanitised_html
            except Exception:
                logger.debug("Failed to sanitise enhanced negotiation HTML", exc_info=True)

        if not sanitised_html:
            return "", cleaned_body

        plain_text = cleaned_body
        if email_agent:
            try:
                extracted = email_agent._html_to_plain_text(sanitised_html)
                if extracted:
                    plain_text = extracted
            except Exception:
                logger.debug("Failed to derive plain text from negotiation HTML", exc_info=True)

        return sanitised_html, plain_text

    def _build_email_draft_stub(
        self,
        *,
        context: AgentContext,
        draft_payload: Dict[str, Any],
        metadata: Dict[str, Any],
        negotiation_message: Optional[str],
        supplier_id: Optional[str],
        supplier_name: Optional[str],
        contact_name: Optional[str],
        session_reference: Optional[str],
        rfq_id: Optional[str],
        recipients: Optional[Sequence[Any]],
        thread_headers: Optional[Any],
        round_number: int,
        decision: Dict[str, Any],
        currency: Optional[str],
        playbook_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        workflow_id = getattr(context, "workflow_id", None)
        subject_seed = self._coerce_text(draft_payload.get("subject"))
        subject = subject_seed or DEFAULT_NEGOTIATION_SUBJECT

        message_body = self._coerce_text(negotiation_message) or ""
        cleaned_body = EmailDraftingAgent._clean_body_text(message_body)

        enhanced_html = self._build_enhanced_html_email(
            round_number=round_number,
            contact_name=contact_name,
            supplier_name=supplier_name,
            decision=decision or {},
            negotiation_message=cleaned_body,
            currency=currency,
            playbook_context=playbook_context,
            sender_name=getattr(
                getattr(self.agent_nick, "settings", None), "sender_name", None
            ),
        )

        email_agent = self._ensure_email_agent()
        sanitised_html = enhanced_html or ""
        plain_text = cleaned_body
        if cleaned_body:
            try:
                sanitised_html, derived_plain = self._build_negotiation_html_shell(
                    subject=subject,
                    cleaned_body=cleaned_body,
                    email_agent=email_agent,
                )
                if sanitised_html:
                    plain_text = EmailDraftingAgent._clean_body_text(derived_plain or cleaned_body)
            except Exception:
                logger.debug("Failed to build enhanced negotiation HTML", exc_info=True)
                sanitised_html = ""
                plain_text = cleaned_body

        if plain_text:
            plain_text = EmailDraftingAgent._clean_body_text(plain_text)

        thread_history_entries = self._get_email_thread_history(
            workflow_id, supplier_id
        )

        transcript_plain = self._format_thread_history_plain(thread_history_entries)
        if transcript_plain:
            combined_plain = (
                f"{plain_text}\n\n{transcript_plain}" if plain_text else transcript_plain
            )
            plain_text = EmailDraftingAgent._clean_body_text(combined_plain)

        transcript_html = self._format_thread_history_html(thread_history_entries)
        if transcript_html:
            if sanitised_html:
                sanitised_html = self._inject_history_into_html(
                    sanitised_html, transcript_html
                )
            else:
                sanitised_html = transcript_html

        unique_id = self._coerce_text(session_reference) or self._coerce_text(
            draft_payload.get("unique_id")
        )
        if not unique_id:
            unique_id = str(uuid.uuid4())

        annotated_body, marker_token = attach_hidden_marker(
            plain_text or "",
            supplier_id=supplier_id,
            unique_id=unique_id,
        )

        to_candidates = self._normalise_recipient_list(recipients)
        if not to_candidates:
            to_candidates = self._normalise_recipient_list(
                draft_payload.get("recipients")
            )
        cc_candidates = self._normalise_recipient_list(draft_payload.get("cc"))
        deduped_cc = self._merge_recipients_basic([], cc_candidates)
        if to_candidates:
            lower_to = {addr.lower() for addr in to_candidates}
            deduped_cc = [addr for addr in deduped_cc if addr.lower() not in lower_to]
        if email_agent:
            try:
                merged = email_agent._merge_recipients(to_candidates, cc_candidates)
            except Exception:
                merged = self._merge_recipients_basic(to_candidates, cc_candidates)
        else:
            merged = self._merge_recipients_basic(to_candidates, cc_candidates)
        recipients_list = merged

        receiver = (
            to_candidates[0]
            if to_candidates
            else (recipients_list[0] if recipients_list else None)
        )
        sender = (
            context.input_data.get("sender")
            if isinstance(context.input_data, dict)
            else None
        )
        if not sender:
            sender = getattr(self.agent_nick.settings, "ses_default_sender", None)

        headers: Dict[str, Any] = {
            "X-ProcWise-Unique-ID": unique_id,
        }
        if workflow_id:
            headers["X-ProcWise-Workflow-ID"] = workflow_id

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("unique_id", unique_id)
        if marker_token:
            metadata_payload.setdefault("dispatch_token", marker_token)
        if workflow_id is not None:
            metadata_payload.setdefault("workflow_id", workflow_id)
        if supplier_id is not None:
            metadata_payload.setdefault("supplier_id", supplier_id)
        if contact_name:
            metadata_payload.setdefault("contact_name", contact_name)
            metadata_payload.setdefault("supplier_contact", contact_name)

        stub: Dict[str, Any] = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "contact_name": contact_name,
            "supplier_contact": contact_name,
            "subject": subject,
            "body": annotated_body,
            "text": plain_text,
            "html": sanitised_html,
            "sender": sender,
            "recipients": recipients_list,
            "receiver": receiver,
            "to": receiver,
            "cc": deduped_cc,
            "contact_level": 1 if receiver else 0,
            "sent_status": False,
            "metadata": metadata_payload,
            "headers": headers,
            "unique_id": unique_id,
            "session_reference": session_reference or unique_id,
            "rfq_id": rfq_id,
            "payload": draft_payload,
            "workflow_id": workflow_id,
        }

        if isinstance(thread_headers, dict) and thread_headers:
            stub["thread_headers"] = dict(thread_headers)

        return stub

    def _build_enhanced_html_email(
        self,
        *,
        round_number: int,
        contact_name: Optional[str],
        supplier_name: Optional[str],
        decision: Dict[str, Any],
        negotiation_message: str,
        currency: Optional[str],
        playbook_context: Optional[Dict[str, Any]],
        sender_name: Optional[str] = None,
    ) -> str:
        decision_payload = decision or {}
        positions = decision_payload.get("positions")
        if not isinstance(positions, dict):
            positions = {}

        recommendations: Optional[List[Dict[str, Any]]] = None
        if isinstance(playbook_context, dict):
            plays = playbook_context.get("plays")
            if isinstance(plays, list):
                recommendations = [rec for rec in plays if isinstance(rec, dict)]

        company_name = getattr(
            getattr(self.agent_nick, "settings", None), "company_name", "Procwise"
        )

        return self._html_builder.build_negotiation_email(
            round_number=round_number,
            contact_name=contact_name,
            supplier_name=supplier_name,
            decision=decision_payload,
            positions=positions,
            currency=currency,
            playbook_recommendations=recommendations,
            negotiation_message=negotiation_message,
            sender_name=sender_name,
            company_name=company_name or "Procwise",
        )

    def _capture_supplier_response(
        self,
        *,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        round_number: Optional[int],
        response: Optional[Dict[str, Any]],
    ) -> Optional[EmailHistoryEntry]:
        workflow_key = self._coerce_text(workflow_id)
        supplier_key = self._coerce_text(supplier_id)
        if not workflow_key or not supplier_key:
            return None
        if not isinstance(response, dict):
            return None

        metadata_payload = dict(response.get("metadata") or {})

        def _resolve_round() -> int:
            candidates = [
                round_number,
                response.get("round_number"),
                response.get("round"),
                (metadata_payload.get("round") if isinstance(metadata_payload, dict) else None),
            ]
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    parsed = int(candidate)
                except Exception:
                    continue
                if parsed > 0:
                    return parsed
            return 1

        resolved_round = _resolve_round()

        def _resolve_timestamp(*values: Any) -> datetime:
            for value in values:
                if value is None:
                    continue
                if isinstance(value, datetime):
                    if value.tzinfo is None:
                        return value.replace(tzinfo=timezone.utc)
                    return value
                if isinstance(value, (int, float)):
                    try:
                        return datetime.fromtimestamp(value, tz=timezone.utc)
                    except Exception:
                        continue
                text_value = self._coerce_text(value)
                if not text_value:
                    continue
                try:
                    return datetime.fromisoformat(text_value)
                except ValueError:
                    pass
                try:
                    return datetime.fromisoformat(text_value.replace("Z", "+00:00"))
                except Exception:
                    pass
                try:
                    parsed_dt = parsedate_to_datetime(text_value)
                except Exception:
                    parsed_dt = None
                if parsed_dt:
                    if parsed_dt.tzinfo is None:
                        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                    return parsed_dt
            return datetime.now(timezone.utc)

        sent_at = _resolve_timestamp(
            response.get("sent_at"),
            response.get("received_at"),
            response.get("received_time"),
            response.get("response_date"),
            response.get("timestamp"),
            response.get("date"),
            metadata_payload.get("sent_at") if isinstance(metadata_payload, dict) else None,
            metadata_payload.get("received_at") if isinstance(metadata_payload, dict) else None,
            metadata_payload.get("timestamp") if isinstance(metadata_payload, dict) else None,
        )

        message_id = self._coerce_text(
            response.get("message_id")
            or response.get("response_message_id")
            or response.get("Message-ID")
        )

        thread_headers_candidate = response.get("thread_headers") or response.get("headers")
        if isinstance(thread_headers_candidate, dict):
            thread_headers: Dict[str, Any] = dict(thread_headers_candidate)
        elif isinstance(thread_headers_candidate, str) and thread_headers_candidate:
            thread_headers = {"raw": thread_headers_candidate}
        else:
            thread_headers = {}

        if isinstance(thread_headers, dict):
            if not message_id:
                message_id = self._coerce_text(
                    thread_headers.get("Message-ID") or thread_headers.get("message_id")
                )
            in_reply_to = self._coerce_text(
                thread_headers.get("In-Reply-To")
                or response.get("in_reply_to")
                or response.get("original_message_id")
            )
            if in_reply_to and "In-Reply-To" not in thread_headers:
                thread_headers["In-Reply-To"] = in_reply_to
            references = response.get("references")
            if references and "References" not in thread_headers:
                thread_headers["References"] = references
        else:
            thread_headers = {}

        email_id = (
            message_id
            or self._coerce_text(response.get("email_id"))
            or str(uuid.uuid4())
        )
        if not message_id:
            message_id = email_id
        if message_id and "Message-ID" not in thread_headers:
            thread_headers["Message-ID"] = message_id

        subject = (
            self._coerce_text(response.get("subject"))
            or self._coerce_text(response.get("response_subject"))
            or self._coerce_text(thread_headers.get("Subject"))
            or "Supplier response"
        )

        body_text = (
            self._extract_message_from_response(response)
            or self._coerce_text(response.get("body_text"))
            or self._coerce_text(response.get("response_text"))
            or ""
        )

        body_html = (
            self._coerce_text(response.get("body_html"))
            or self._coerce_text(response.get("html"))
            or self._coerce_text(response.get("response_body"))
            or ""
        )
        if not body_html and body_text:
            body_html = self._simple_html_from_text(body_text)

        sender = (
            self._coerce_text(response.get("sender"))
            or self._coerce_text(response.get("from"))
            or self._coerce_text(response.get("from_addr"))
            or self._coerce_text(response.get("response_from"))
            or self._coerce_text(response.get("supplier_email"))
            or ""
        )

        recipient_candidates: List[str] = []
        for candidate in (
            response.get("recipients"),
            response.get("to"),
            response.get("to_addr"),
            response.get("buyer_email"),
            response.get("original_recipients"),
        ):
            recipient_candidates.extend(self._normalise_recipient_list(candidate))
        if isinstance(metadata_payload, dict):
            recipient_candidates.extend(
                self._normalise_recipient_list(metadata_payload.get("recipients"))
            )
            recipient_candidates.extend(
                self._normalise_recipient_list(metadata_payload.get("cc"))
            )
        buyer_address = self._coerce_text(response.get("buyer_address"))
        if buyer_address:
            recipient_candidates.extend(self._normalise_recipient_list(buyer_address))

        deduped_recipients: List[str] = []
        seen_recipients: Set[str] = set()
        for addr in recipient_candidates:
            lowered = addr.lower()
            if lowered in seen_recipients:
                continue
            seen_recipients.add(lowered)
            deduped_recipients.append(addr)

        extra_metadata_keys = (
            "workflow_id",
            "unique_id",
            "rfq_id",
            "mailbox",
            "imap_uid",
            "price",
            "lead_time",
            "match_confidence",
            "response_time",
            "supplier_email",
            "from_addr",
            "response_from",
            "original_message_id",
            "original_subject",
        )
        for key in extra_metadata_keys:
            if isinstance(metadata_payload, dict) and key in metadata_payload:
                continue
            value = response.get(key)
            if value is not None:
                metadata_payload[key] = value

        if isinstance(metadata_payload, dict):
            metadata_payload.setdefault("source", "supplier_response")
            if message_id and "message_id" not in metadata_payload:
                metadata_payload["message_id"] = message_id

        negotiation_context = {
            "entry_type": "supplier_response",
            "round_number": resolved_round,
        }

        entry = EmailHistoryEntry(
            email_id=email_id,
            round_number=resolved_round,
            supplier_id=supplier_key,
            supplier_name=self._coerce_text(response.get("supplier_name")),
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            sender=sender,
            recipients=deduped_recipients,
            sent_at=sent_at,
            message_id=message_id,
            thread_headers=thread_headers,
            metadata=dict(metadata_payload) if isinstance(metadata_payload, dict) else {},
            decision={},
            negotiation_context=negotiation_context,
        )

        thread_entries = self._email_thread_manager.get_thread(workflow_key, supplier_key)
        for idx, existing in enumerate(thread_entries):
            if existing.email_id == email_id:
                updated_entries = list(thread_entries)
                updated_entries[idx] = entry
                self._email_thread_manager.set_thread(workflow_key, supplier_key, updated_entries)
                return entry

        self._email_thread_manager.add_email(workflow_key, supplier_key, entry)
        return entry

    def _capture_email_to_history(
        self,
        *,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        round_number: Optional[int] = None,
        draft: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        draft_records: Optional[Sequence[Dict[str, Any]]] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        thread_headers: Optional[Union[Dict[str, Any], str]] = None,
        email_action_id: Optional[str] = None,
        message_id: Optional[str] = None,
        recipients: Optional[Sequence[str]] = None,
        cc: Optional[Sequence[str]] = None,
        sender: Optional[str] = None,
        session_reference: Optional[str] = None,
    ) -> Optional[EmailHistoryEntry]:
        workflow_key = self._coerce_text(workflow_id)
        supplier_key = self._coerce_text(supplier_id)
        if not workflow_key or not supplier_key:
            return None

        decision = dict(decision or {})
        state = state if isinstance(state, dict) else {}

        resolved_round = None
        if round_number is not None:
            try:
                resolved_round = int(round_number)
            except Exception:
                resolved_round = None
        if resolved_round is None:
            try:
                resolved_round = int(state.get("current_round", 1))
            except Exception:
                resolved_round = 1

        aggregated_records: List[Dict[str, Any]] = []
        if isinstance(draft, dict):
            aggregated_records.append(draft)
        if draft_records:
            for record in draft_records:
                if isinstance(record, dict):
                    aggregated_records.append(record)

        if not aggregated_records:
            return None

        subject_fallback = self._coerce_text(subject) or ""
        body_fallback = self._coerce_text(body) or ""
        sender_fallback = self._coerce_text(sender) or ""
        recipients_fallback = list(recipients or [])
        message_id_fallback = self._coerce_text(message_id)

        thread_headers_fallback: Dict[str, Any] = {}
        if isinstance(thread_headers, dict):
            thread_headers_fallback = dict(thread_headers)
        elif isinstance(thread_headers, str) and thread_headers:
            thread_headers_fallback = {"raw": thread_headers}

        cc_payload = list(cc or []) if cc else []

        history = state.get("email_history")
        if not isinstance(history, list):
            history = []

        existing_history_map: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for idx, entry in enumerate(history):
            if isinstance(entry, dict) and entry.get("email_id"):
                existing_history_map[str(entry["email_id"])] = (idx, entry)

        thread_entries = self._email_thread_manager.get_thread(
            workflow_key, supplier_key
        )
        thread_entry_map: Dict[str, EmailHistoryEntry] = {
            entry.email_id: entry for entry in thread_entries if entry.email_id
        }

        last_entry: Optional[EmailHistoryEntry] = None

        for record in aggregated_records:
            email_id = self._coerce_text(record.get("id")) or str(uuid.uuid4())
            record_subject = (
                self._coerce_text(record.get("subject")) or subject_fallback
            )
            record_text = self._coerce_text(record.get("text")) or body_fallback
            record_html = self._coerce_text(record.get("html")) or body_fallback
            record_sender = (
                self._coerce_text(record.get("sender"))
                or self._coerce_text(record.get("from"))
                or sender_fallback
            )
            record_recipients = list(record.get("recipients") or [])
            if not record_recipients and recipients_fallback:
                record_recipients = list(recipients_fallback)

            record_message_id = self._coerce_text(
                record.get("message_id")
                or record.get("Message-ID")
                or message_id_fallback
            )

            record_headers: Dict[str, Any]
            headers_candidate = record.get("thread_headers")
            if isinstance(headers_candidate, dict):
                record_headers = dict(headers_candidate)
            elif isinstance(headers_candidate, str) and headers_candidate:
                record_headers = {"raw": headers_candidate}
            else:
                record_headers = {}
            if thread_headers_fallback:
                record_headers = {**thread_headers_fallback, **record_headers}

            record_metadata = dict(record.get("metadata") or {})
            if email_action_id:
                record_metadata.setdefault("email_action_id", email_action_id)
            if session_reference:
                record_metadata.setdefault("session_reference", session_reference)
            if cc_payload:
                record_metadata.setdefault("cc", list(cc_payload))

            if not record_message_id:
                record_message_id = self._coerce_text(
                    record_headers.get("Message-ID")
                    if isinstance(record_headers, dict)
                    else None
                ) or self._coerce_text(
                    record_headers.get("message_id")
                    if isinstance(record_headers, dict)
                    else None
                )

            entry = EmailHistoryEntry(
                email_id=email_id,
                round_number=resolved_round,
                supplier_id=supplier_key,
                supplier_name=record.get("supplier_name"),
                subject=record_subject,
                body_text=record_text,
                body_html=record_html,
                sender=record_sender,
                recipients=record_recipients,
                sent_at=datetime.now(timezone.utc),
                message_id=record_message_id,
                thread_headers=record_headers,
                metadata=record_metadata,
                decision=dict(decision),
                negotiation_context={
                    "positions": decision.get("positions"),
                    "counter_price": decision.get("counter_price"),
                    "strategy": decision.get("strategy"),
                    "play_recommendations": decision.get("play_recommendations"),
                },
            )

            if email_id in thread_entry_map:
                existing_entry = thread_entry_map[email_id]
                entry.sent_at = existing_entry.sent_at
                updated_entries = list(thread_entries)
                for idx, existing in enumerate(updated_entries):
                    if existing.email_id == email_id:
                        updated_entries[idx] = entry
                        break
                self._email_thread_manager.set_thread(
                    workflow_key, supplier_key, updated_entries
                )
                thread_entries = self._email_thread_manager.get_thread(
                    workflow_key, supplier_key
                )
                thread_entry_map = {
                    existing.email_id: existing for existing in thread_entries
                }
            else:
                self._email_thread_manager.add_email(
                    workflow_key, supplier_key, entry
                )
                thread_entries = self._email_thread_manager.get_thread(
                    workflow_key, supplier_key
                )
                thread_entry_map = {
                    existing.email_id: existing for existing in thread_entries
                }

            entry_payload = entry.to_dict()
            history_index: Optional[int] = None
            if email_id in existing_history_map:
                history_index = existing_history_map[email_id][0]
                existing_history_map[email_id] = (history_index, entry_payload)
            else:
                existing_history_map[email_id] = (len(history), entry_payload)

            if history_index is not None:
                history[history_index] = entry_payload
            else:
                history.append(entry_payload)

            last_entry = entry

        state["email_history"] = history

        return last_entry

    def _get_email_thread_history(
        self, workflow_id: Optional[str], supplier_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        workflow_key = self._coerce_text(workflow_id)
        supplier_key = self._coerce_text(supplier_id)
        if not workflow_key or not supplier_key:
            return []
        thread = self._email_thread_manager.get_thread(workflow_key, supplier_key)
        return [entry.to_dict() for entry in thread]

    def _select_thread_history_entries(
        self, entries: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not entries:
            return []

        limit = THREAD_HISTORY_TRANSCRIPT_LIMIT
        if isinstance(limit, int) and limit > 0:
            return list(entries)[-limit:]

        return list(entries)

    def _format_thread_history_plain(
        self, entries: Sequence[Dict[str, Any]]
    ) -> str:
        if not entries:
            return ""

        recent_entries = self._select_thread_history_entries(entries)
        lines: List[str] = ["--- Prior Thread History ---"]

        for entry in recent_entries:
            if not isinstance(entry, dict):
                continue

            round_number = entry.get("round_number")
            round_display = (
                f"Round {round_number}"
                if isinstance(round_number, int)
                else "Previous"
            )

            sent_at = self._coerce_text(entry.get("sent_at")) or "Unknown time"
            sender = self._coerce_text(entry.get("sender")) or "Unknown sender"

            recipients_payload = entry.get("recipients")
            recipients_list: List[str] = []
            if isinstance(recipients_payload, (list, tuple, set)):
                recipients_list = [
                    self._coerce_text(candidate) or ""
                    for candidate in recipients_payload
                ]
            recipients = ", ".join(filter(None, recipients_list))

            subject = self._coerce_text(entry.get("subject")) or "(no subject)"
            raw_body = entry.get("body_text") or entry.get("body") or ""
            cleaned_body = EmailDraftingAgent._clean_body_text(str(raw_body))

            lines.extend(
                [
                    "",
                    f"{round_display} • {sent_at}",
                    f"From {sender}" + (f" → {recipients}" if recipients else ""),
                    f"Subject: {subject}",
                ]
            )
            if cleaned_body:
                lines.append(cleaned_body)

        return "\n".join(lines).strip()

    def _format_thread_history_html(
        self, entries: Sequence[Dict[str, Any]]
    ) -> str:
        if not entries:
            return ""

        recent_entries = self._select_thread_history_entries(entries)
        sections: List[str] = []

        for entry in recent_entries:
            if not isinstance(entry, dict):
                continue

            round_number = entry.get("round_number")
            round_display = (
                f"Round {round_number}"
                if isinstance(round_number, int)
                else "Previous"
            )

            sent_at = self._coerce_text(entry.get("sent_at")) or "Unknown time"
            sender = self._coerce_text(entry.get("sender")) or "Unknown sender"

            recipients_payload = entry.get("recipients")
            recipients_list: List[str] = []
            if isinstance(recipients_payload, (list, tuple, set)):
                recipients_list = [
                    self._coerce_text(candidate) or ""
                    for candidate in recipients_payload
                ]
            recipients = ", ".join(filter(None, recipients_list))

            subject = self._coerce_text(entry.get("subject")) or "(no subject)"
            raw_body = entry.get("body_text") or entry.get("body") or ""
            cleaned_body = EmailDraftingAgent._clean_body_text(str(raw_body))

            escaped_body = (
                escape(cleaned_body).replace("\n", "<br/>") if cleaned_body else ""
            )
            escaped_recipients = escape(recipients) if recipients else ""

            sections.append(
                (
                    '<div style="margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #e2e8f0;">\n'
                    f'  <div style="font-family:\'Segoe UI\',Arial,sans-serif;font-size:14px;font-weight:600;color:#0f172a;">{escape(round_display)} • {escape(sent_at)}</div>\n'
                    f'  <div style="font-family:\'Segoe UI\',Arial,sans-serif;font-size:14px;color:#475569;margin-top:4px;">From {escape(sender)}'
                    + (f" → {escaped_recipients}" if escaped_recipients else "")
                    + '</div>\n'
                    f'  <div style="font-family:\'Segoe UI\',Arial,sans-serif;font-size:14px;color:#0f172a;font-weight:600;margin-top:8px;">Subject: {escape(subject)}</div>\n'
                    + (
                        f'  <div style="font-family:\'Segoe UI\',Arial,sans-serif;font-size:14px;color:#1f2937;margin-top:6px;white-space:pre-wrap;">{escaped_body}</div>\n'
                        if escaped_body
                        else ""
                    )
                    + "</div>"
                )
            )

        if not sections:
            return ""

        history_container = (
            '<div data-procwise-thread-history="1" '
            'style="margin-top:24px;padding-top:16px;border-top:1px solid #e2e8f0;">\n'
            '  <h2 style="margin:0 0 16px 0;font-family:\'Segoe UI\',Arial,sans-serif;font-size:18px;color:#0f172a;">Prior Thread History</h2>\n'
            f"  {''.join(sections)}\n"
            "</div>"
        )
        return history_container

    def _inject_history_into_html(self, html: str, transcript_html: str) -> str:
        if not html or not transcript_html:
            return html

        insertion_marker = (
            "          <tr>\n"
            "            <td style=\"padding:20px 32px 28px 32px;background-color:#f8fafc;font-family:'Segoe UI',Arial,sans-serif;font-size:12px;line-height:1.6;color:#64748b;border-top:1px solid #e2e8f0;\">\n"
        )

        history_row = (
            "          <tr>\n"
            "            <td style=\"padding:0 32px 32px 32px;font-family:'Segoe UI',Arial,sans-serif;\">\n"
            f"              {transcript_html}\n"
            "            </td>\n"
            "          </tr>\n"
        )

        marker_index = html.find(insertion_marker)
        if marker_index == -1:
            closing_body = "</body>"
            body_index = html.lower().rfind(closing_body)
            if body_index == -1:
                return f"{html}\n{transcript_html}"
            return (
                f"{html[:body_index]}\n{transcript_html}\n{html[body_index:]}"
            )

        return html[:marker_index] + history_row + html[marker_index:]

    def _get_email_thread_summary(
        self, workflow_id: Optional[str], supplier_id: Optional[str]
    ) -> Dict[str, Any]:
        workflow_key = self._coerce_text(workflow_id)
        supplier_key = self._coerce_text(supplier_id)
        if not workflow_key or not supplier_key:
            return {
                "total_emails": 0,
                "rounds": [],
                "first_sent": None,
                "last_sent": None,
                "thread_key": None,
            }
        return self._email_thread_manager.get_thread_summary(
            workflow_key, supplier_key
        )

    def _compose_email_history_payload(
        self,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        state: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
        workflow_key = self._coerce_text(workflow_id)
        supplier_key = self._coerce_text(supplier_id)

        thread_history = self._get_email_thread_history(workflow_id, supplier_id)
        thread_summary = self._get_email_thread_summary(workflow_id, supplier_id)

        summary_payload: Dict[str, Any]
        if isinstance(thread_summary, dict):
            summary_payload = dict(thread_summary)
        else:
            summary_payload = {}

        default_thread_key: Optional[str] = None
        if workflow_key and supplier_key:
            default_thread_key = f"{workflow_key}:{supplier_key}"

        summary_payload.setdefault("total_emails", 0)
        summary_payload.setdefault("rounds", [])
        summary_payload.setdefault("first_sent", None)
        summary_payload.setdefault("last_sent", None)
        summary_payload.setdefault("thread_key", default_thread_key)

        cached_history: List[Dict[str, Any]] = []
        if isinstance(state, dict):
            history_payload = state.get("email_history")
            if isinstance(history_payload, list):
                cached_history = [
                    item for item in history_payload if isinstance(item, dict)
                ]

        def _normalise_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(entry, dict):
                return None
            payload: Dict[str, Any] = dict(entry)

            email_id = self._coerce_text(
                payload.get("email_id") or payload.get("id")
            )
            if not email_id:
                email_id = str(uuid.uuid4())
            payload["email_id"] = email_id

            try:
                payload["round_number"] = int(payload.get("round_number", 0) or 0)
            except Exception:
                payload["round_number"] = 0

            for key in ("metadata", "thread_headers", "decision", "negotiation_context"):
                value = payload.get(key)
                if isinstance(value, dict):
                    payload[key] = dict(value)
                elif value is None:
                    payload[key] = {}
                else:
                    try:
                        payload[key] = dict(value)
                    except Exception:
                        payload[key] = {}

            recipients = payload.get("recipients")
            if isinstance(recipients, list):
                payload["recipients"] = [
                    self._coerce_text(item)
                    for item in recipients
                    if self._coerce_text(item)
                ]
            elif recipients is None:
                payload["recipients"] = []
            else:
                coerced_recipient = self._coerce_text(recipients)
                payload["recipients"] = [coerced_recipient] if coerced_recipient else []

            for key in ("subject", "body_text", "body_html", "sender"):
                value = payload.get(key)
                payload[key] = self._coerce_text(value) if value else ""

            sent_at_value = payload.get("sent_at")
            if isinstance(sent_at_value, datetime):
                payload["sent_at"] = sent_at_value.isoformat()
            elif isinstance(sent_at_value, str):
                payload["sent_at"] = sent_at_value
            else:
                payload["sent_at"] = None

            if supplier_key:
                payload.setdefault("supplier_id", supplier_key)

            return payload

        merged: Dict[str, Dict[str, Any]] = {}
        for entry in thread_history:
            normalised = _normalise_entry(entry)
            if not normalised:
                continue
            merged[normalised["email_id"]] = normalised

        for entry in cached_history:
            normalised = _normalise_entry(entry)
            if not normalised:
                continue
            email_id = normalised["email_id"]
            if email_id in merged:
                existing = merged[email_id]
                for key, value in normalised.items():
                    if key in {"metadata", "thread_headers", "decision", "negotiation_context"}:
                        if not isinstance(value, dict):
                            continue
                        existing_value = existing.get(key)
                        if isinstance(existing_value, dict):
                            merged_value = dict(value)
                            merged_value.update(existing_value)
                            existing[key] = merged_value
                        else:
                            existing[key] = dict(value)
                    elif key == "recipients":
                        if not existing.get("recipients") and value:
                            existing["recipients"] = list(value)
                    elif existing.get(key) in (None, "", [], {}):
                        existing[key] = value
            else:
                merged[email_id] = normalised

        def _parse_sent_at(value: Any) -> Optional[datetime]:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str) and value:
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    try:
                        return datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except Exception:
                        return None
            return None

        merged_entries = list(merged.values())
        merged_entries.sort(
            key=lambda item: (
                _parse_sent_at(item.get("sent_at"))
                or datetime.min.replace(tzinfo=timezone.utc),
                item.get("round_number") or 0,
                item.get("email_id"),
            )
        )

        rounds = sorted(
            {
                int(entry.get("round_number") or 0)
                for entry in merged_entries
                if int(entry.get("round_number") or 0) > 0
            }
        )
        if rounds:
            summary_payload["rounds"] = rounds
        elif not isinstance(summary_payload.get("rounds"), list):
            summary_payload["rounds"] = []

        sent_at_values = [
            _parse_sent_at(entry.get("sent_at")) for entry in merged_entries if entry.get("sent_at")
        ]
        sent_at_values = [value for value in sent_at_values if value is not None]
        if sent_at_values:
            summary_payload["first_sent"] = sent_at_values[0].isoformat()
            summary_payload["last_sent"] = sent_at_values[-1].isoformat()
        else:
            summary_payload["first_sent"] = summary_payload.get("first_sent") or None
            summary_payload["last_sent"] = summary_payload.get("last_sent") or None

        total_count = len(merged_entries)
        summary_payload["total_emails"] = total_count
        if not summary_payload.get("thread_key"):
            summary_payload["thread_key"] = default_thread_key

        return merged_entries, summary_payload, total_count

    def generate_email_preview(
        self,
        *,
        round_number: int,
        decision: Dict[str, Any],
        negotiation_message: str,
        supplier_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        currency: Optional[str] = None,
        playbook_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self._html_builder.build_negotiation_email(
            round_number=round_number,
            contact_name=contact_name,
            supplier_name=supplier_name,
            decision=decision or {},
            positions=(decision or {}).get("positions") or {},
            currency=currency,
            playbook_recommendations=playbook_recommendations,
            negotiation_message=negotiation_message,
        )

    def _build_email_agent_payload(
        self,
        context: AgentContext,
        draft_payload: Dict[str, Any],
        decision: Dict[str, Any],
        state: Dict[str, Any],
        negotiation_message: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(draft_payload, dict):
            return None
        payload = dict(draft_payload)
        decision_payload = dict(decision) if isinstance(decision, dict) else {}
        workflow_id = getattr(context, "workflow_id", None)
        if workflow_id:
            payload.setdefault("workflow_id", workflow_id)
        else:
            logger.warning("NegotiationAgent building email payload without workflow_id on context")

        supplier_hint = (
            payload.get("supplier_id")
            or payload.get("supplier")
            or context.input_data.get("supplier_id")
            or context.input_data.get("supplier")
            or state.get("supplier_id")
        )
        supplier_token = self._coerce_text(supplier_hint)
        if supplier_token:
            payload["supplier_id"] = supplier_token
            payload.setdefault("supplier", supplier_token)
            decision_payload.setdefault("supplier_id", supplier_token)
            decision_payload.setdefault("supplier", supplier_token)

        logger.info(
            "Building email drafting payload for supplier",
            extra={
                "workflow_id": workflow_id,
                "supplier_id": payload.get("supplier_id") or payload.get("supplier"),
            },
        )

        thread_headers = (
            context.input_data.get("thread_headers")
            if isinstance(context.input_data, dict)
            else None
        )
        if not thread_headers:
            thread_headers = payload.get("thread_headers")
        if not thread_headers:
            cached_thread_headers = state.get("last_thread_headers")
            if isinstance(cached_thread_headers, dict):
                thread_headers = dict(cached_thread_headers)
            else:
                thread_headers = cached_thread_headers
        if thread_headers:
            payload.setdefault("thread_headers", thread_headers)
            decision_payload.setdefault("thread", thread_headers)
        payload.setdefault("decision", decision_payload)
        payload.setdefault("session_state", self._public_state(state))
        payload.setdefault("negotiation_message", negotiation_message)
        payload.setdefault("supplier_reply_count", state.get("supplier_reply_count", 0))
        supplier_name = context.input_data.get("supplier_name")
        if supplier_name:
            payload.setdefault("supplier_name", supplier_name)
        contact = context.input_data.get("from_address")
        if contact:
            payload.setdefault("recipients", [contact])
        payload.setdefault("intent", "NEGOTIATION_COUNTER")
        return payload

    def _invoke_email_drafting_agent(
        self,
        parent_context: AgentContext,
        payload: Dict[str, Any],
    ) -> Optional[AgentOutput]:
        try:
            if self._email_agent is None:
                with self._email_agent_lock:
                    if self._email_agent is None:
                        self._email_agent = EmailDraftingAgent(self.agent_nick)
            email_agent = self._email_agent
            email_context = parent_context.create_child_context(
                "EmailDraftingAgent", payload
            )
            if email_context.workflow_id != parent_context.workflow_id:
                raise RuntimeError(
                    "Email drafting child context did not inherit workflow_id"
                )
            logger.info(
                "Invoking EmailDraftingAgent",
                extra={
                    "workflow_id": email_context.workflow_id,
                    "parent_agent": parent_context.agent_id,
                },
            )
            email_output = email_agent.execute(email_context) if email_agent else None
            if email_output is not None:
                try:
                    self._ensure_structured_html_output(email_output, payload)
                except Exception:  # pragma: no cover - defensive
                    logger.debug(
                        "Failed to post-process negotiation email HTML output",
                        exc_info=True,
                    )
            return email_output
        except Exception:
            logger.exception("Failed to invoke EmailDraftingAgent for negotiation counter")
            return None

    def _ensure_structured_html_output(
        self, email_output: AgentOutput, payload: Dict[str, Any]
    ) -> None:
        """Ensure negotiation email outputs include structured HTML variants."""

        if not isinstance(email_output, AgentOutput):
            return
        if email_output.status != AgentStatus.SUCCESS:
            return

        data = email_output.data
        if not isinstance(data, dict):
            return

        email_agent = self._ensure_email_agent()

        decision_payload: Dict[str, Any] = {}
        for container in (data.get("decision"), payload.get("decision")):
            if isinstance(container, dict):
                decision_payload = dict(container)
                break

        round_candidate = None
        for container in (data, payload, decision_payload):
            if isinstance(container, dict):
                for key in ("round", "round_number", "round_no"):
                    if key in container:
                        round_candidate = container.get(key)
                        if round_candidate is not None:
                            break
                if round_candidate is not None:
                    break
        try:
            round_number = int(float(round_candidate)) if round_candidate is not None else 1
        except Exception:
            round_number = 1

        base_supplier_name = self._coerce_text(
            payload.get("supplier_name") or payload.get("supplier")
        )

        base_contact_name = self._resolve_contact_name(payload)

        currency_candidate = None
        for container in (decision_payload, payload, data):
            if isinstance(container, dict) and container.get("currency"):
                currency_candidate = container.get("currency")
                break

        playbook_context = payload.get("playbook_context")
        if not isinstance(playbook_context, dict):
            playbook_context = (
                data.get("playbook_context")
                if isinstance(data.get("playbook_context"), dict)
                else None
            )

        subject_fallback = (
            self._coerce_text(data.get("subject"))
            or self._coerce_text(payload.get("subject"))
            or self._coerce_text(decision_payload.get("subject"))
            or DEFAULT_NEGOTIATION_SUBJECT
        )

        body_fallback = (
            self._coerce_text(data.get("body"))
            or self._coerce_text(payload.get("negotiation_message"))
            or self._coerce_text(payload.get("message"))
            or ""
        )

        sender_name = getattr(getattr(self.agent_nick, "settings", None), "sender_name", None)

        def _apply_html(container: Dict[str, Any], *, fallback_subject: str, fallback_body: str) -> None:
            if not isinstance(container, dict):
                return

            existing_html = self._coerce_text(container.get("html") or container.get("body_html"))
            if existing_html:
                return

            subject_text = self._coerce_text(container.get("subject")) or fallback_subject
            body_text = (
                self._coerce_text(container.get("body"))
                or self._coerce_text(container.get("text"))
                or fallback_body
            )
            if not body_text:
                return

            cleaned_body = EmailDraftingAgent._clean_body_text(body_text)
            contact_name = self._resolve_contact_name(container, fallback=base_contact_name)

            html_candidate = ""
            try:
                html_candidate = self._build_enhanced_html_email(
                    round_number=round_number,
                    contact_name=contact_name,
                    supplier_name=base_supplier_name,
                    decision=decision_payload,
                    negotiation_message=cleaned_body,
                    currency=currency_candidate,
                    playbook_context=playbook_context,
                    sender_name=sender_name,
                )
            except Exception:
                logger.debug(
                    "Enhanced negotiation HTML build failed; using fallback shell",
                    exc_info=True,
                )
                html_candidate = ""

            plain_text = cleaned_body
            if not html_candidate:
                try:
                    html_candidate, derived_plain = self._build_negotiation_html_shell(
                        subject=subject_text,
                        cleaned_body=cleaned_body,
                        email_agent=email_agent,
                    )
                    if derived_plain:
                        plain_text = EmailDraftingAgent._clean_body_text(derived_plain)
                except Exception:
                    logger.debug(
                        "Negotiation HTML shell build failed",
                        exc_info=True,
                    )
                    html_candidate = ""
            else:
                if email_agent:
                    try:
                        derived_plain = email_agent._html_to_plain_text(html_candidate)
                        if derived_plain:
                            plain_text = EmailDraftingAgent._clean_body_text(derived_plain)
                    except Exception:
                        logger.debug(
                            "Failed to extract plain text from enhanced negotiation HTML",
                            exc_info=True,
                        )

            if not html_candidate:
                return

            container["html"] = html_candidate
            container.setdefault("body_html", html_candidate)
            if plain_text:
                container.setdefault("text", plain_text)

        _apply_html(data, fallback_subject=subject_fallback, fallback_body=body_fallback)

        drafts_payload = data.get("drafts")
        if isinstance(drafts_payload, list):
            for draft in drafts_payload:
                _apply_html(
                    draft,
                    fallback_subject=subject_fallback,
                    fallback_body=body_fallback,
                )

    def _build_email_finalization_task(
        self,
        *,
        context: AgentContext,
        identifier: NegotiationIdentifier,
        state: Dict[str, Any],
        thread_state: Optional[EmailThreadState],
        draft_payload: Dict[str, Any],
        draft_stub: Dict[str, Any],
        email_payload: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
        negotiation_message: Optional[str],
        supplier_message: Optional[str],
        supplier_snippets: Sequence[Any],
        supplier_name: Optional[str],
        contact_name: Optional[str],
        session_reference: str,
        rfq_value: Optional[str],
        round_no: int,
        workflow_id: Optional[str],
        supplier: Optional[str],
        currency: Optional[str],
        volume_units: Any,
        term_days: Any,
        valid_until: Any,
        market_floor: Any,
        normalised_inputs: Dict[str, Any],
        supplier_reply_registered: bool,
        state_identifier: Optional[str],
        playbook_context: Dict[str, Any],
        recipients: Sequence[Any],
        thread_headers: Any,
        counter_options: Sequence[Any],
        savings_score: Any,
        decision_log: Optional[str],
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "identifier": {
                "workflow_id": identifier.workflow_id,
                "supplier_id": supplier,
                "session_reference": session_reference,
                "round_number": round_no,
            },
            "state": deepcopy(state) if isinstance(state, dict) else {},
            "thread_state": thread_state.as_dict() if thread_state else None,
            "draft_payload": deepcopy(draft_payload) if isinstance(draft_payload, dict) else {},
            "draft_stub": deepcopy(draft_stub) if isinstance(draft_stub, dict) else {},
            "email_payload": deepcopy(email_payload) if isinstance(email_payload, dict) else None,
            "decision": deepcopy(decision) if isinstance(decision, dict) else {},
            "negotiation_message": negotiation_message,
            "supplier_snippets": list(supplier_snippets) if isinstance(supplier_snippets, Sequence) else [],
            "supplier_name": supplier_name,
            "contact_name": contact_name,
            "supplier_message": supplier_message,
            "rfq_value": rfq_value,
            "round_no": round_no,
            "workflow_id": workflow_id or identifier.workflow_id,
            "supplier": supplier,
            "currency": currency,
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor": market_floor,
            "normalised_inputs": deepcopy(normalised_inputs) if isinstance(normalised_inputs, dict) else {},
            "supplier_reply_registered": supplier_reply_registered,
            "state_identifier": state_identifier or identifier.workflow_id,
            "playbook_context": deepcopy(playbook_context) if isinstance(playbook_context, dict) else {},
            "recipients": list(recipients) if isinstance(recipients, Sequence) else [],
            "thread_headers": deepcopy(thread_headers) if isinstance(thread_headers, dict) else thread_headers,
            "context_input": deepcopy(context.input_data)
            if isinstance(context.input_data, dict)
            else {},
            "counter_options": list(counter_options)
            if isinstance(counter_options, Sequence)
            else [],
            "savings_score": savings_score,
            "decision_log": decision_log,
        }
        response_timeout = None
        if isinstance(context.input_data, dict):
            response_timeout = context.input_data.get("response_timeout")
        task["response_timeout"] = response_timeout
        return task

    def _finalize_email_round(
        self,
        parent_context: AgentContext,
        identifier: NegotiationIdentifier,
        task: Dict[str, Any],
    ) -> AgentOutput:
        bundle_entries = task.get("bundle") if isinstance(task, dict) else None
        if isinstance(bundle_entries, list) and bundle_entries:
            return self._finalize_email_round_bundle(
                parent_context=parent_context,
                identifier=identifier,
                task=task,
                bundle_entries=bundle_entries,
            )
        return self._finalize_single_email_round(parent_context, identifier, task)

    def _finalize_email_round_bundle(
        self,
        *,
        parent_context: AgentContext,
        identifier: NegotiationIdentifier,
        task: Dict[str, Any],
        bundle_entries: Sequence[Dict[str, Any]],
    ) -> AgentOutput:
        combined_drafts: List[Dict[str, Any]] = []
        combined_entries: List[Dict[str, Any]] = []
        combined_pass_entries: List[Dict[str, Any]] = []
        combined_suppliers: List[str] = []
        aggregated_next_agents: Set[str] = set()
        aggregated_status = AgentStatus.SUCCESS
        aggregated_errors: List[str] = []

        round_number = task.get("round_no") if isinstance(task, dict) else None
        try:
            round_number = int(round_number) if round_number is not None else identifier.round_number
        except Exception:
            round_number = identifier.round_number

        bundle_results: List[Tuple[Dict[str, Any], AgentOutput, NegotiationIdentifier]] = []

        for entry in bundle_entries:
            entry_task = entry.get("task") if isinstance(entry, dict) else entry
            if not isinstance(entry_task, dict):
                continue
            entry_identifier = entry.get("identifier") if isinstance(entry, dict) else None
            if not isinstance(entry_identifier, NegotiationIdentifier):
                identifier_payload = entry_task.get("identifier")
                identifier_info = identifier_payload if isinstance(identifier_payload, dict) else {}
                entry_identifier = NegotiationIdentifier(
                    workflow_id=identifier_info.get("workflow_id") or identifier.workflow_id,
                    supplier_id=identifier_info.get("supplier_id")
                    or identifier_info.get("supplier")
                    or identifier.supplier_id,
                    session_reference=identifier_info.get("session_reference")
                    or identifier.session_reference,
                    round_number=identifier_info.get("round_number")
                    or round_number,
                )
            result = self._finalize_single_email_round(parent_context, entry_identifier, dict(entry_task))
            bundle_results.append((entry, result, entry_identifier))

        for entry, result, entry_identifier in bundle_results:
            if not isinstance(result, AgentOutput):
                continue
            entry_data = result.data if isinstance(result.data, dict) else {}
            entry_pass_fields = result.pass_fields if isinstance(result.pass_fields, dict) else {}
            drafts_payload = entry_data.get("drafts") if isinstance(entry_data, dict) else None
            if isinstance(drafts_payload, list):
                combined_drafts.extend([draft for draft in drafts_payload if isinstance(draft, dict)])
            combined_suppliers.append(entry_identifier.supplier_id)
            aggregated_next_agents.update(result.next_agents or [])
            if result.status != AgentStatus.SUCCESS and aggregated_status == AgentStatus.SUCCESS:
                aggregated_status = result.status
            if result.error:
                aggregated_errors.append(result.error)

            combined_entries.append(
                {
                    "identifier": {
                        "workflow_id": entry_identifier.workflow_id,
                        "supplier_id": entry_identifier.supplier_id,
                        "session_reference": entry_identifier.session_reference,
                        "round_number": entry_identifier.round_number,
                    },
                    "data": deepcopy(entry_data),
                    "pass_fields": deepcopy(entry_pass_fields),
                    "status": result.status.value,
                    "next_agents": list(result.next_agents),
                    "error": result.error,
                }
            )
            combined_pass_entries.append(
                {
                    "identifier": {
                        "workflow_id": entry_identifier.workflow_id,
                        "supplier_id": entry_identifier.supplier_id,
                        "session_reference": entry_identifier.session_reference,
                        "round_number": entry_identifier.round_number,
                    },
                    "pass_fields": deepcopy(entry_pass_fields),
                }
            )

            stored_result = entry.get("result") if isinstance(entry, dict) else None
            if isinstance(stored_result, AgentOutput):
                stored_result.status = result.status
                stored_result.data = result.data
                stored_result.pass_fields = result.pass_fields
                stored_result.error = result.error
                stored_result.next_agents = list(result.next_agents)
                stored_result.action_id = result.action_id
                stored_result.agentic_plan = result.agentic_plan
                stored_result.context_snapshot = result.context_snapshot

        bundle_payload = {
            "round": round_number,
            "workflow_id": identifier.workflow_id,
            "bundled": True,
            "suppliers": combined_suppliers,
            "entries": combined_entries,
            "drafts": combined_drafts,
        }

        bundle_pass_fields = {
            "round": round_number,
            "bundled": True,
            "suppliers": combined_suppliers,
            "entries": deepcopy(combined_pass_entries),
            "drafts": deepcopy(combined_drafts),
        }

        error_text = None
        if aggregated_status != AgentStatus.SUCCESS and aggregated_errors:
            error_text = "; ".join({err for err in aggregated_errors if err})

        return AgentOutput(
            status=aggregated_status,
            data=bundle_payload,
            pass_fields=bundle_pass_fields,
            next_agents=sorted(aggregated_next_agents),
            error=error_text,
        )

    def _finalize_single_email_round(
        self,
        parent_context: AgentContext,
        identifier: NegotiationIdentifier,
        task: Dict[str, Any],
    ) -> AgentOutput:
        context_input = task.get("context_input") if isinstance(task.get("context_input"), dict) else {}

        state = deepcopy(task.get("state") or {})
        workflow_id = task.get("workflow_id") or identifier.workflow_id
        supplier = task.get("supplier") or identifier.supplier_id
        session_reference = task.get("identifier", {}).get("session_reference") or identifier.session_reference
        rfq_value = task.get("rfq_value")
        round_no = int(task.get("round_no") or identifier.round_number or 1)
        state_identifier = task.get("state_identifier") or workflow_id

        thread_state_dict = task.get("thread_state")
        thread_state: Optional[EmailThreadState] = None
        if isinstance(thread_state_dict, dict) and thread_state_dict:
            fallback_subject = self._normalise_base_subject(state.get("base_subject"))
            if not fallback_subject:
                fallback_subject = DEFAULT_NEGOTIATION_SUBJECT
            thread_state = EmailThreadState.from_dict(
                thread_state_dict, fallback_subject=fallback_subject
            )
        state["workflow_id"] = workflow_id
        state["session_reference"] = session_reference
        if supplier:
            state["supplier_id"] = supplier

        draft_payload = deepcopy(task.get("draft_payload") or {})
        draft_stub = deepcopy(task.get("draft_stub") or {})
        email_payload = deepcopy(task.get("email_payload") or {})
        decision = deepcopy(task.get("decision") or {})
        negotiation_message = task.get("negotiation_message")
        supplier_snippets = list(task.get("supplier_snippets") or [])
        supplier_message = task.get("supplier_message")
        counter_options = list(task.get("counter_options") or [])
        savings_score = task.get("savings_score")
        decision_log = task.get("decision_log")
        supplier_name = task.get("supplier_name")
        contact_name = task.get("contact_name")
        currency = task.get("currency")
        volume_units = task.get("volume_units")
        term_days = task.get("term_days")
        valid_until = task.get("valid_until")
        market_floor = task.get("market_floor")
        normalised_inputs = deepcopy(task.get("normalised_inputs") or {})
        playbook_context = deepcopy(task.get("playbook_context") or {})
        recipients = list(task.get("recipients") or [])
        thread_headers = task.get("thread_headers")
        supplier_reply_registered = bool(task.get("supplier_reply_registered"))
        response_timeout = task.get("response_timeout")

        email_output: Optional[AgentOutput] = None
        fallback_payload: Optional[Dict[str, Any]] = None
        email_action_id: Optional[str] = None
        email_subject: Optional[str] = None
        email_body: Optional[str] = None
        draft_records: List[Dict[str, Any]] = []
        sent_message_id: Optional[str] = None
        next_agents: List[str] = []

        if email_payload and supplier and session_reference:
            email_output = self._invoke_email_drafting_agent(parent_context, email_payload)
            fallback_payload = dict(email_payload)

        if email_output and email_output.status == AgentStatus.SUCCESS:
            email_data = email_output.data or {}
            email_action_id = email_output.action_id or email_data.get("action_id")
            email_subject = email_data.get("subject")
            email_body = email_data.get("body")
            candidate_headers = email_data.get("thread_headers")
            if isinstance(candidate_headers, dict):
                candidate_headers = {
                    key: value for key, value in candidate_headers.items() if value is not None
                }
                header_message_id = self._coerce_text(
                    candidate_headers.get("Message-ID")
                    or candidate_headers.get("message_id")
                )
                if header_message_id:
                    sent_message_id = header_message_id
                draft_payload["thread_headers"] = dict(candidate_headers)
                state["last_thread_headers"] = dict(candidate_headers)
            drafts_payload = email_data.get("drafts")
            if isinstance(drafts_payload, list) and drafts_payload:
                for draft in drafts_payload:
                    if not isinstance(draft, dict):
                        continue
                    draft_copy = dict(draft)
                    if email_action_id:
                        draft_copy.setdefault("email_action_id", email_action_id)
                    if not sent_message_id:
                        candidate_id = self._coerce_text(
                            draft_copy.get("message_id")
                            or draft_copy.get("Message-ID")
                        )
                        if not candidate_id:
                            headers_payload = draft_copy.get("headers") or draft_copy.get("thread_headers")
                            if isinstance(headers_payload, dict):
                                candidate_id = self._coerce_text(
                                    headers_payload.get("Message-ID")
                                    or headers_payload.get("message_id")
                                )
                        if candidate_id:
                            sent_message_id = candidate_id
                    draft_records.append(draft_copy)
            else:
                draft_records.append(dict(draft_stub))
        else:
            draft_records.append(dict(draft_stub))
            next_agents = ["EmailDraftingAgent"]
            if fallback_payload is None and email_payload:
                fallback_payload = dict(email_payload)

        subject_seed = self._coerce_text(draft_payload.get("subject"))
        if subject_seed and not state.get("base_subject"):
            base_from_email = self._normalise_base_subject(subject_seed)
            if base_from_email:
                state["base_subject"] = base_from_email

        if email_subject and not state.get("base_subject"):
            base_from_email = self._normalise_base_subject(email_subject)
            if base_from_email:
                state["base_subject"] = base_from_email
        elif subject_seed and not state.get("base_subject"):
            fallback_base = self._normalise_base_subject(subject_seed)
            if fallback_base:
                state["base_subject"] = fallback_base

        if email_body and not state.get("initial_body"):
            state["initial_body"] = email_body
        elif not state.get("initial_body") and negotiation_message:
            state["initial_body"] = negotiation_message

        if not email_subject:
            email_subject = subject_seed
        if not email_body and negotiation_message:
            email_body = negotiation_message

        for draft_record in draft_records:
            if not isinstance(draft_record, dict):
                continue
            if not draft_record.get("html"):
                draft_record["html"] = self._build_enhanced_html_email(
                    round_number=round_no,
                    contact_name=contact_name,
                    supplier_name=supplier_name,
                    decision=decision,
                    negotiation_message=negotiation_message or "",
                    currency=currency,
                    playbook_context=playbook_context,
                    sender_name=getattr(
                        getattr(self.agent_nick, "settings", None),
                        "sender_name",
                        None,
                    ),
                )
            self._capture_email_to_history(
                workflow_id=workflow_id,
                supplier_id=supplier,
                round_number=round_no,
                draft=draft_record,
                decision=decision,
                state=state,
            )

        email_thread_history = self._get_email_thread_history(workflow_id, supplier)
        email_thread_summary = self._get_email_thread_summary(workflow_id, supplier)
        state["email_history"] = email_thread_history

        final_thread_headers = draft_payload.get("thread_headers")
        if isinstance(final_thread_headers, dict):
            if thread_state:
                subject_hint = final_thread_headers.get("Subject")
                base_subject_hint = self._normalise_base_subject(subject_hint)
                if base_subject_hint:
                    thread_state.subject_base = base_subject_hint
            if not sent_message_id:
                sent_message_id = self._coerce_text(
                    final_thread_headers.get("Message-ID")
                    or final_thread_headers.get("message_id")
                )
        elif isinstance(final_thread_headers, str) and not sent_message_id:
            sent_message_id = self._coerce_text(final_thread_headers)

        if sent_message_id and thread_state:
            thread_state.update_after_send(sent_message_id)

        state["status"] = "ACTIVE"
        state["awaiting_response"] = True
        state["last_email_sent_at"] = datetime.now(timezone.utc)
        if email_action_id:
            state["last_agent_msg_id"] = email_action_id

        self._persist_thread_state(state, thread_state)
        self._save_session_state(identifier.workflow_id, supplier, state)
        self._record_learning_snapshot(
            parent_context,
            identifier.workflow_id or state_identifier,
            supplier,
            decision,
            state,
            True,
            supplier_reply_registered,
            rfq_id=rfq_value,
        )

        if self.response_matcher and supplier:
            try:
                timeout = 900
                if response_timeout is not None:
                    try:
                        timeout = int(response_timeout)
                    except Exception:
                        timeout = 900
                self.response_matcher.register_expected_response(
                    identifier=NegotiationIdentifier(
                        workflow_id=identifier.workflow_id,
                        session_reference=session_reference,
                        supplier_id=supplier,
                        round_number=round_no,
                    ),
                    email_action_id=email_action_id,
                    expected_unique_id=session_reference,
                    timeout_seconds=timeout,
                )
            except Exception:
                logger.debug("Failed to register expected response", exc_info=True)

        cache_key: Optional[Tuple[str, str]] = None
        if state_identifier and supplier:
            cache_key = (str(state_identifier), str(supplier))
        cached_state = self._get_cached_state(cache_key)
        public_state = self._public_state(cached_state or state)

        supplier_watch_fields = self._build_supplier_watch_fields(
            context=parent_context,
            workflow_id=workflow_id,
            supplier=supplier,
            drafts=draft_records,
            state=state,
            session_reference=session_reference,
            rfq_id=rfq_value,
        )

        supplier_responses: List[Dict[str, Any]] = []
        if supplier_watch_fields:
            await_response = bool(supplier_watch_fields.get("await_response"))
            awaiting_email_drafting = "EmailDraftingAgent" in next_agents
            input_payload = (
                parent_context.input_data
                if isinstance(parent_context.input_data, dict)
                else {}
            )
            should_wait = (
                await_response
                and not awaiting_email_drafting
                and not bool(input_payload.get("_batch_execution"))
            )
            if should_wait:
                self._log_response_wait_diagnostics(
                    workflow_id=workflow_id,
                    supplier_id=supplier,
                    drafts=draft_records,
                    watch_payload=supplier_watch_fields,
                )
                wait_results = self._await_supplier_responses(
                    context=parent_context,
                    watch_payload=supplier_watch_fields,
                    state=state,
                )
                watch_unique_ids = [
                    self._coerce_text(entry.get("unique_id"))
                    for entry in supplier_watch_fields.get("drafts", [])
                    if isinstance(entry, dict) and self._coerce_text(entry.get("unique_id"))
                ]
                if wait_results is None:
                    logger.error(
                        "Supplier responses not received before timeout (workflow_id=%s supplier=%s unique_ids=%s)",
                        workflow_id,
                        supplier,
                        watch_unique_ids or None,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_value,
                        "workflow_id": workflow_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "round": round_no,
                        "decision": decision,
                        "message": "Supplier response not received before timeout.",
                        "unique_ids": watch_unique_ids,
                    }
                    return self._with_plan(
                        parent_context,
                        AgentOutput(
                            status=AgentStatus.FAILED,
                            data=error_payload,
                            error="supplier response timeout",
                        ),
                    )
                wait_thread_headers: Optional[Dict[str, Any]] = None
                if isinstance(wait_results, dict):
                    candidate_headers = wait_results.get("thread_headers")
                    if isinstance(candidate_headers, dict) and candidate_headers:
                        wait_thread_headers = dict(candidate_headers)
                elif isinstance(wait_results, list):
                    for candidate in wait_results:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_headers = candidate.get("thread_headers")
                        if isinstance(candidate_headers, dict) and candidate_headers:
                            wait_thread_headers = dict(candidate_headers)
                            break
                if wait_thread_headers:
                    state["last_thread_headers"] = dict(wait_thread_headers)
                    wait_message_id = self._coerce_text(
                        wait_thread_headers.get("Message-ID")
                        or wait_thread_headers.get("message_id")
                    )
                    if wait_message_id:
                        if thread_state:
                            thread_state.update_after_receive(wait_message_id)
                supplier_responses = [res for res in wait_results if isinstance(res, dict)]
                if not supplier_responses:
                    logger.error(
                        "No supplier responses received while waiting (workflow_id=%s supplier=%s unique_ids=%s)",
                        workflow_id,
                        supplier,
                        watch_unique_ids or None,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_value,
                        "workflow_id": workflow_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "round": round_no,
                        "decision": decision,
                        "message": "Missing supplier responses after wait.",
                        "unique_ids": watch_unique_ids,
                    }
                    return self._with_plan(
                        parent_context,
                        AgentOutput(
                            status=AgentStatus.FAILED,
                            data=error_payload,
                            error="missing_supplier_responses",
                        ),
                    )

        data = {
            "supplier": supplier,
            "rfq_id": rfq_value,
            "session_reference": session_reference,
            "unique_id": session_reference,
            "round": round_no,
            "decision": decision,
            "counter_proposals": counter_options,
            "savings_score": savings_score,
            "decision_log": decision_log,
            "message": negotiation_message,
            "email_subject": email_subject,
            "email_body": email_body,
            "supplier_snippets": supplier_snippets,
            "negotiation_allowed": True,
            "interaction_type": "negotiation",
            "intent": "NEGOTIATION_COUNTER",
            "draft_payload": draft_payload,
            "drafts": draft_records,
            "session_state": public_state,
            "currency": currency,
            "current_offer": normalised_inputs.get("current_offer"),
            "target_price": normalised_inputs.get("target_price"),
            "supplier_message": supplier_message
            if supplier_message is not None
            else (
                context_input.get("supplier_message")
                if isinstance(context_input, dict)
                else None
            ),
            "sent_status": False,
            "awaiting_response": True,
            "play_recommendations": playbook_context.get("plays"),
            "playbook_descriptor": playbook_context.get("descriptor"),
            "playbook_examples": playbook_context.get("examples"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
            "normalised_inputs": normalised_inputs,
            "thread_state": thread_state.as_dict() if thread_state else None,
            "supplier_responses": supplier_responses,
            "email_history": email_thread_history,
            "email_thread_summary": email_thread_summary,
            "current_email": draft_records[0] if draft_records else None,
            "all_emails_sent": len(email_thread_history),
        }

        if email_subject:
            data["email_subject"] = email_subject
        if email_body:
            data["email_body"] = email_body
        if email_action_id:
            data["email_action_id"] = email_action_id

        pass_fields: Dict[str, Any] = dict(data)

        supplier_watch_fields = self._build_supplier_watch_fields(
            context=parent_context,
            workflow_id=workflow_id,
            supplier=supplier,
            drafts=draft_records,
            state=state,
            session_reference=session_reference,
            rfq_id=rfq_value,
        )
        if supplier_watch_fields:
            pass_fields.update(supplier_watch_fields)

        output = AgentOutput(
            status=AgentStatus.SUCCESS,
            data=data,
            pass_fields=pass_fields,
            next_agents=next_agents,
        )
        if fallback_payload:
            output.pass_fields.setdefault("fallback_email_payload", fallback_payload)
        output = self._with_plan(parent_context, output)
        self._queue_email_draft_action(
            parent_context,
            supplier_id=supplier,
            supplier_name=supplier_name,
            round_number=round_no,
            subject=email_subject or draft_payload.get("subject"),
            body=email_body or negotiation_message,
            drafts=draft_records,
            decision=decision,
            negotiation_message=negotiation_message,
            agentic_plan=output.agentic_plan,
            context_snapshot=self._build_email_context_snapshot(parent_context),
        )
        return output

    def _queue_email_draft_action(
        self,
        context: AgentContext,
        *,
        supplier_id: Optional[str],
        supplier_name: Optional[str],
        round_number: Optional[int],
        subject: Optional[Any],
        body: Optional[Any],
        drafts: Sequence[Dict[str, Any]],
        decision: Dict[str, Any],
        negotiation_message: Optional[str],
        agentic_plan: Optional[str],
        context_snapshot: Optional[Dict[str, Any]],
    ) -> None:
        if not drafts:
            return
        draft_items = [dict(draft) for draft in drafts if isinstance(draft, dict)]
        if not draft_items:
            return

        payload = self._prepare_email_action_payload(
            context=context,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            round_number=round_number,
            subject=subject,
            body=body,
            drafts=draft_items,
            decision=decision,
            negotiation_message=negotiation_message,
            agentic_plan=agentic_plan,
            context_snapshot=context_snapshot,
        )
        if not payload:
            return

        entry = {
            "supplier_id": supplier_id,
            "round": round_number,
            "process_output": payload,
            "description": (
                f"Negotiation email draft for supplier {supplier_name or supplier_id or 'unknown'}"
                + (f" round {round_number}" if round_number else "")
            ),
        }
        pending: List[Dict[str, Any]] = getattr(context, "_pending_email_actions", [])
        pending.append(entry)
        setattr(context, "_pending_email_actions", pending)

    def _prepare_email_action_payload(
        self,
        *,
        context: AgentContext,
        supplier_id: Optional[str],
        supplier_name: Optional[str],
        round_number: Optional[int],
        subject: Optional[Any],
        body: Optional[Any],
        drafts: Sequence[Dict[str, Any]],
        decision: Dict[str, Any],
        negotiation_message: Optional[str],
        agentic_plan: Optional[str],
        context_snapshot: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        primary = drafts[0]

        subject_text = self._coerce_text(subject) or self._coerce_text(primary.get("subject"))
        if not subject_text:
            subject_text = DEFAULT_NEGOTIATION_SUBJECT

        body_text = self._coerce_text(body) or primary.get("body")
        if not body_text and negotiation_message:
            body_text = self._coerce_text(negotiation_message)
        if not isinstance(body_text, str):
            body_text = ""

        sender = primary.get("sender") or getattr(
            getattr(self.agent_nick, "settings", SimpleNamespace()),
            "ses_default_sender",
            None,
        )

        recipients = primary.get("recipients")
        if not isinstance(recipients, list):
            recipients = self._normalise_recipient_list(primary.get("to") or primary.get("receiver"))

        receiver = primary.get("receiver")
        if not receiver and recipients:
            receiver = recipients[0]

        to_field = primary.get("to") or list(recipients or [])
        cc_field = primary.get("cc") or []
        if isinstance(cc_field, str):
            cc_field = self._normalise_recipient_list(cc_field)
        elif not isinstance(cc_field, list):
            cc_field = []

        metadata = dict(primary.get("metadata") or {})
        metadata.setdefault("supplier_id", supplier_id)
        if supplier_name:
            metadata.setdefault("supplier_name", supplier_name)
        if decision.get("strategy"):
            metadata.setdefault("strategy", decision.get("strategy"))
        if decision.get("intent"):
            metadata.setdefault("intent", decision.get("intent"))
        metadata.setdefault("intent", "NEGOTIATION_COUNTER")
        if round_number is not None:
            metadata.setdefault("round", round_number)
        metadata.setdefault("workflow_id", context.workflow_id)

        headers = dict(primary.get("headers") or {})
        thread_headers = primary.get("thread_headers")
        if isinstance(thread_headers, dict):
            thread_headers = {k: v for k, v in thread_headers.items() if v is not None}

        unique_id = primary.get("unique_id") or metadata.get("unique_id")
        if not unique_id and isinstance(thread_headers, dict):
            unique_id = thread_headers.get("X-ProcWise-Unique-ID") or thread_headers.get(
                "X-Procwise-Unique-Id"
            )

        message_id = primary.get("message_id") or headers.get("Message-ID")

        contact_level = primary.get("contact_level") if isinstance(primary.get("contact_level"), int) else 0
        thread_index = primary.get("thread_index")
        if not isinstance(thread_index, int):
            thread_index = 1

        text_version = primary.get("text")
        if not isinstance(text_version, str) and body_text:
            text_version = EmailDraftingAgent._clean_body_text(body_text)

        html_version = primary.get("html") or primary.get("body_html")
        if not isinstance(html_version, str) and body_text:
            html_version = self._simple_html_from_text(body_text)

        cleaned_drafts = [deepcopy(draft) for draft in drafts]

        payload: Dict[str, Any] = {
            "drafts": cleaned_drafts,
            "subject": subject_text,
            "body": body_text,
            "text": text_version or body_text,
            "html": html_version or "",
            "sender": sender,
            "recipients": list(recipients or []),
            "receiver": receiver,
            "to": list(to_field or []),
            "cc": list(cc_field or []),
            "contact_level": contact_level,
            "sent_status": bool(primary.get("sent_status", False)),
            "metadata": metadata,
            "headers": headers,
            "unique_id": unique_id,
            "thread_headers": thread_headers,
            "message_id": message_id,
            "workflow_id": context.workflow_id,
            "thread_index": thread_index,
            "intent": "NEGOTIATION_COUNTER",
            "agentic_plan": agentic_plan,
            "context_snapshot": context_snapshot,
        }

        if primary.get("id") is not None:
            payload["id"] = primary.get("id")
        if primary.get("draft_record_id") is not None:
            payload["draft_record_id"] = primary.get("draft_record_id")
        if supplier_id:
            payload.setdefault("supplier_id", supplier_id)
        if supplier_name:
            payload.setdefault("supplier_name", supplier_name)

        email_agent = self._ensure_email_agent()
        if email_agent:
            try:
                payload = email_agent._normalise_action_payload(payload)
            except Exception:
                logger.debug("Failed to normalise email draft payload", exc_info=True)

        return payload

    def _build_email_context_snapshot(self, context: AgentContext) -> Dict[str, Any]:
        snapshot = {
            "workflow_id": getattr(context, "workflow_id", None),
            "agent_id": "EmailDraftingAgent",
            "user_id": getattr(context, "user_id", None),
            "manifest": context.manifest(),
        }
        return {key: value for key, value in snapshot.items() if value}

    def _flush_email_draft_actions(
        self, context: AgentContext, result: AgentOutput
    ) -> None:
        pending: Sequence[Dict[str, Any]] = getattr(context, "_pending_email_actions", [])
        if not pending:
            return

        routing = getattr(self.agent_nick, "process_routing_service", None)
        if routing is None or not hasattr(routing, "log_action"):
            logger.debug("Process routing service missing; skipping email draft action logging")
            return

        process_id = getattr(context, "process_id", None)
        if not process_id:
            logger.debug(
                "No process_id available for negotiation email draft logging; using ad-hoc logging"
            )

        status_value = "completed" if result.status == AgentStatus.SUCCESS else "failed"

        for entry in pending:
            process_output = entry.get("process_output")
            if not process_output:
                continue
            description = entry.get("description") or "Negotiation email draft"
            try:
                routing.log_action(
                    process_id=process_id,
                    agent_type="EmailDraftingAgent",
                    action_desc=description,
                    process_output=process_output,
                    status=status_value,
                    run_id=None,
                )
            except Exception:
                logger.exception("Failed to log negotiation email draft action")

    def _log_final_negotiation_outcome(
        self, context: AgentContext, summary: Dict[str, Any]
    ) -> None:
        routing = getattr(self.agent_nick, "process_routing_service", None)
        if routing is None or not hasattr(routing, "log_action"):
            return

        try:
            serialisable = json.loads(json.dumps(summary, default=str))
        except Exception:
            logger.debug("Unable to serialise negotiation summary for logging", exc_info=True)
            serialisable = summary

        try:
            routing.log_action(
                process_id=getattr(context, "process_id", None),
                agent_type=self.__class__.__name__,
                action_desc="Negotiation summary",
                process_output=serialisable,
                status="completed",
                run_id=None,
            )
        except Exception:
            logger.exception("Failed to log final negotiation outcome")

@dataclass
class SupplierNegotiationState:
    supplier_id: str
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    received_responses: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "round_history": list(self.round_history),
            "parameters": dict(self.parameters),
            "status": self.status,
            "received_responses": list(self.received_responses),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SupplierNegotiationState":
        if not isinstance(payload, dict):
            raise TypeError("SupplierNegotiationState requires a dictionary payload")
        return cls(
            supplier_id=str(payload.get("supplier_id") or ""),
            round_history=list(payload.get("round_history") or []),
            parameters=dict(payload.get("parameters") or {}),
            status=str(payload.get("status") or "PENDING"),
            received_responses=list(payload.get("received_responses") or []),
        )


@dataclass
class NegotiationSession:
    session_id: str
    supplier_negotiations: Dict[str, SupplierNegotiationState] = field(default_factory=dict)
    current_round: int = 1
    max_rounds: int = 3
    pending_responses: List[str] = field(default_factory=list)
    received_responses: List[str] = field(default_factory=list)
    negotiation_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "supplier_negotiations": {
                supplier_id: state.to_dict()
                for supplier_id, state in self.supplier_negotiations.items()
            },
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "pending_responses": list(self.pending_responses),
            "received_responses": list(self.received_responses),
            "negotiation_parameters": dict(self.negotiation_parameters),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NegotiationSession":
        if not isinstance(payload, dict):
            raise TypeError("NegotiationSession requires a dictionary payload")
        suppliers_payload = payload.get("supplier_negotiations") or {}
        supplier_states: Dict[str, SupplierNegotiationState] = {}
        if isinstance(suppliers_payload, dict):
            for supplier_id, state_payload in suppliers_payload.items():
                try:
                    supplier_states[str(supplier_id)] = SupplierNegotiationState.from_dict(
                        state_payload
                    )
                except Exception:
                    continue
        return cls(
            session_id=str(payload.get("session_id") or ""),
            supplier_negotiations=supplier_states,
            current_round=int(payload.get("current_round") or 1),
            max_rounds=int(payload.get("max_rounds") or 3),
            pending_responses=list(payload.get("pending_responses") or []),
            received_responses=list(payload.get("received_responses") or []),
            negotiation_parameters=dict(payload.get("negotiation_parameters") or {}),
        )

    def register_supplier(self, supplier_id: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        if supplier_id not in self.supplier_negotiations:
            self.supplier_negotiations[supplier_id] = SupplierNegotiationState(
                supplier_id=supplier_id,
                parameters=dict(parameters or {}),
            )

    def update_round(self, round_number: int) -> None:
        self.current_round = max(1, round_number)
