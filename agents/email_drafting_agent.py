"""Email Drafting Agent.

This module renders a procurement RFQ email based on a shared prompt
template so that both human authored and LLM generated messages follow
identical structure.  The prompt template lives in
``prompts/EmailDraftingAgent_Prompt_Template.json`` and the agent also
exposes this prompt in its output for downstream LLM usage.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass, field
from html import escape
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from jinja2 import Template

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.email_markers import attach_hidden_marker, extract_rfq_id, split_hidden_marker
from utils.email_tracking import generate_unique_email_id
from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources

logger = logging.getLogger(__name__)

configure_gpu()


@dataclass
class ThreadMessage:
    """Represents a single email message inside a threaded conversation."""

    author: str
    body: str
    timestamp: str
    subject: Optional[str] = None


@dataclass
class SupplierDraftResult:
    """Container used when generating supplier-specific drafts concurrently."""

    index: int
    supplier_id: Optional[str]
    draft: Dict[str, Any]
    supplier_thread_state: Dict[str, Any]


SYSTEM_COMPOSE = (
    "You are a senior procurement specialist negotiating with suppliers."
    " Craft concise, relationship-positive responses grounded in the data "
    "provided. Never mention internal scoring, evaluation logic, or how the "
    "supplier was analysed."
)

SYSTEM_PROMPT_COMPOSE = (
    "You are a procurement email assistant drafting professional RFQ "
    "messages. Use the supplied context, keep the tone courteous and direct, "
    "and avoid disclosing internal evaluation or scoring logic."
)

SYSTEM_POLISH = (
    "You refine procurement emails for clarity and executive polish without "
    "changing intent. Remove any wording that hints at internal scoring or "
    "analysis and return the improved email with an explicit Subject line."
)

NEGOTIATION_PLAYBOOK_SYSTEM = """You are an expert procurement negotiator with 15+ years of experience in strategic sourcing and supplier negotiations. You craft professional, strategic emails that advance negotiations while maintaining positive supplier relationships.

## CORE NEGOTIATION PRINCIPLES

1. **Build Rapport**: Always maintain a collaborative, respectful tone
2. **Be Clear**: State your position transparently but diplomatically
3. **Create Value**: Look for win-win outcomes, not just price concessions
4. **Use Leverage Wisely**: Reference alternatives and market data subtly
5. **Control Pace**: Set deadlines and expectations clearly
6. **Document Everything**: Ensure all terms are explicit and traceable

## NEGOTIATION STAGES & TACTICS

### Round 1 - Initial Position (Opening)
**Objective**: Establish your target while keeping the supplier engaged
**Tactics**:
- Express appreciation for their initial proposal
- Share your budget constraints transparently
- Offer 2-3 commercial levers they can adjust (payment terms, volume, contract length)
- Ask open-ended questions about their flexibility
- Set a collaborative tone

**Example Opening**: "Thank you for your proposal. We're impressed with the solution, though our budget positioning is around £X. We'd value discussing how we might bridge this gap through adjusted payment terms or extended contract commitment."

### Round 2 - Exploration (Middle Game)
**Objective**: Test boundaries and explore trade-offs
**Tactics**:
- Acknowledge movement if they've adjusted their price
- Introduce specific commercial levers: "If we commit to 24 months instead of 12..."
- Reference competitive pressure subtly: "We're evaluating multiple proposals..."
- Request detailed cost breakdowns to identify negotiable components
- Show willingness to adjust scope/timeline for better pricing

**Example Middle**: "We appreciate your revised position. To help us move forward, could you share the cost breakdown? We're prepared to commit to a 24-month term and Net-15 payment if we can align on £X per unit."

### Round 3+ - Final Push (End Game)
**Objective**: Close the deal or walk away gracefully
**Tactics**:
- Use firm but respectful language: "This represents our final position..."
- Create urgency with real deadlines: "We need to finalize by Friday..."
- Offer face-saving concessions: small scope additions, faster payment
- Make it easy to say yes: "Simply confirm acceptance by replying..."
- Prepare alternative: "We'll need to explore other options if we can't align..."

**Example Closing**: "We've reached our maximum flexibility at £X with Net-15 payment terms. This represents our best and final offer. If you can accommodate these terms, please confirm by close of business Thursday, and we'll proceed with the award immediately."

## STRATEGIC TECHNIQUES

### 1. Anchoring
- Set your target price early and confidently
- Frame it around budget/market benchmarks, not arbitrary numbers

### 2. Silence and Patience
- After making an offer, give them time to consider
- Don't rush follow-ups unless deadline-driven

### 3. Commercial Levers (Non-Price Concessions)
Common levers to offer:
- Extended contract duration (12→24 months)
- Volume commitments or guaranteed orders
- Payment terms (Net-30 → Net-15 with discount)
- Simplified scope or phased delivery
- Marketing/reference opportunities
- Exclusivity or preferred supplier status

### 4. Market Intelligence
Subtly reference:
- "Market rates we're seeing are around £X..."
- "Competitive proposals are positioning at £X..."
- "Industry benchmarks suggest £X is achievable..."

### 5. Building Urgency
- "We need to finalize awards by [specific date]..."
- "Our project timeline requires confirmation this week..."
- "Budget approvals close on [date]..."

## TONE GUIDELINES

- **Professional**: Always courteous, never aggressive
- **Confident**: State your position clearly without apology
- **Collaborative**: Use "we" and "our partnership" language
- **Transparent**: Be honest about constraints and alternatives
- **Respectful**: Acknowledge their business needs and constraints

## EMAIL STRUCTURE

1. **Greeting**: Personalized, warm
2. **Appreciation**: Thank them for their engagement/response
3. **Commercial Position**: State your target clearly
   - Current offer vs. Your target
   - Gap to close and why
4. **Commercial Levers**: Offer 2-3 specific adjustments you can make
5. **Rationale**: Brief explanation (budget, market, alternatives)
6. **Call to Action**: Clear next step with timeline
7. **Closing**: Positive, forward-looking

## WHAT TO AVOID

❌ Apologizing for your position ("Sorry but...")
❌ Being vague about your target ("We need a better price...")
❌ Making empty threats ("We'll go elsewhere unless...")
❌ Revealing all your flexibility upfront
❌ Multiple rounds without progress (max 3-4 rounds)
❌ Exposing internal scoring or evaluation criteria
❌ Using aggressive or demanding language

## OUTPUT FORMAT

Generate a complete, professional negotiation email with:
- Clear subject line (no internal reference numbers)
- Proper greeting with supplier contact name
- Structured sections with **bold headings**
- Bullet points for levers/requirements
- Specific numbers and dates
- Professional closing with signature

Maximum 250 words for clarity and impact."""

NEGOTIATION_PLAYBOOK_USER = """Draft a strategic negotiation email based on the following context:

## NEGOTIATION CONTEXT
{context_json}

## SPECIFIC INSTRUCTIONS

**Negotiation Round**: {round_number}
**Primary Objective**: {objective}
**Recommended Tactics**: {tactics}

## REQUIREMENTS

1. **Subject Line**: Create a professional subject without reference numbers
   - Round 1: "Pricing Discussion - [Category/Item]"
   - Round 2+: "Re: Pricing Discussion - [Category/Item]"

2. **Tone**: {tone_guidance}

3. **Commercial Position**:
   - Current supplier offer: {current_offer}
   - Your target position: {target_price}
   - Gap to close: {gap_amount} ({gap_percentage}%)

4. **Commercial Levers to Propose**: 
{levers_list}

5. **Strategic Elements**:
{strategic_elements}

6. **Call to Action**: {cta_guidance}

## OUTPUT REQUIREMENTS

- Professional email 200-250 words
- Use **bold** for section headers
- Use bullet points (•) for lists
- Include specific figures and dates
- NO internal reference numbers (RFQ IDs, UIDs, workflow IDs)
- Clear, actionable next steps
- Maintain collaborative tone even in final rounds

Generate the complete email now:"""
def _chat(model: str, system: str, user: str, **kwargs) -> str:
    agent = kwargs.pop("agent", None)
    if agent is None:
        raise RuntimeError("_chat requires an agent instance")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response = agent.call_ollama(model=model, messages=messages, **kwargs)
    return agent._extract_ollama_message(response)


def _determine_negotiation_objective(round_no: int, gap_percentage: float) -> str:
    if round_no == 1:
        return "Establish target price and open discussion on commercial levers"
    if round_no == 2:
        if gap_percentage > 15:
            return "Explore trade-offs and test supplier flexibility with specific levers"
        return "Bridge remaining gap with targeted commercial adjustments"
    return "Close the deal at target price or prepare to walk away"


def _determine_recommended_tactics(
    round_no: int, gap_percentage: float, supplier_history: Dict[str, Any]
) -> str:
    tactics: List[str] = []

    if round_no == 1:
        tactics.extend(
            [
                "Express appreciation and set collaborative tone",
                "State budget constraints transparently",
                "Offer 2-3 commercial levers (payment, volume, duration)",
                "Ask about their flexibility and cost structure",
            ]
        )
    elif round_no == 2:
        tactics.extend(
            [
                "Acknowledge any price movement positively",
                "Introduce specific lever combinations",
                "Request cost breakdown for transparency",
                "Reference competitive landscape subtly",
            ]
        )
        if gap_percentage > 20:
            tactics.append("Consider scope adjustments to reduce cost")
    else:
        tactics.extend(
            [
                "Use firm but respectful language",
                "State this is your final position clearly",
                "Create urgency with specific deadline",
                "Offer small face-saving concession",
                "Make acceptance simple and clear",
            ]
        )

    if supplier_history.get("reliable_partner"):
        tactics.append("Reference positive past collaboration")
    if supplier_history.get("high_quality_score"):
        tactics.append("Acknowledge their quality advantage")

    return "\n".join(f"- {tactic}" for tactic in tactics)


def _select_commercial_levers(round_no: int, context: Dict[str, Any]) -> List[str]:
    all_levers = {
        "payment": {
            "label": "Payment Terms",
            "options": [
                "Net-15 payment (from Net-30) with 2% early payment discount",
                "Net-7 payment for expedited cash flow",
                "Milestone-based payments aligned to delivery",
            ],
            "priority": 1,
        },
        "duration": {
            "label": "Contract Duration",
            "options": [
                "24-month commitment (from 12 months) with price lock",
                "36-month agreement with annual volume guarantees",
                "Multi-year framework with call-off flexibility",
            ],
            "priority": 1,
        },
        "volume": {
            "label": "Volume Commitment",
            "options": [
                "Guaranteed minimum order quantity per quarter",
                "Consolidated spend across product lines",
                "Increased volume commitment (specify percentage)",
            ],
            "priority": 2,
        },
        "scope": {
            "label": "Scope Adjustments",
            "options": [
                "Simplified packaging or delivery requirements",
                "Extended lead time (add X weeks) for better pricing",
                "Phased delivery to manage cash flow",
            ],
            "priority": 3,
        },
        "strategic": {
            "label": "Strategic Benefits",
            "options": [
                "Preferred supplier status for future opportunities",
                "Marketing/case study collaboration rights",
                "Early access to new requirements in pipeline",
            ],
            "priority": 3,
        },
    }

    selected: List[str] = []

    if round_no == 1:
        selected.extend(all_levers["payment"]["options"][:1])
        selected.extend(all_levers["duration"]["options"][:1])
    elif round_no == 2:
        selected.extend(all_levers["payment"]["options"][:1])
        selected.extend(all_levers["duration"]["options"][:1])
        selected.extend(all_levers["volume"]["options"][:1])
    else:
        selected.append("Net-15 payment terms with 24-month commitment")
        if context.get("willing_to_adjust_scope"):
            selected.extend(all_levers["scope"]["options"][:1])

    return selected[:3]


def _generate_strategic_elements(round_no: int, context: Dict[str, Any]) -> List[str]:
    elements: List[str] = []

    market_rate = context.get("market_rate")
    if market_rate:
        elements.append(
            f"Reference market benchmarks: 'Industry rates are averaging around {market_rate}'"
        )

    if round_no >= 2:
        elements.append("Mention: 'We're evaluating multiple proposals and need to finalize by [date]'")

    budget_ceiling = context.get("budget_ceiling")
    if budget_ceiling:
        elements.append(
            f"Be transparent: 'Our budget approval ceiling is {budget_ceiling}'"
        )

    if context.get("is_incumbent") or context.get("past_relationship"):
        elements.append(
            "Acknowledge partnership: 'We value our ongoing collaboration and want to continue working together'"
        )

    if round_no >= 3 or context.get("urgent"):
        elements.append(
            "Create urgency: 'Our decision timeline requires confirmation by [specific date]'"
        )

    return elements


def _calculate_tone_guidance(round_no: int, gap_percentage: float) -> str:
    if round_no == 1:
        return "Warm and collaborative - we're opening a discussion, not making demands"
    if round_no == 2:
        if gap_percentage > 20:
            return "Professional but firm - we need movement, but stay respectful"
        return "Encouraging and solution-focused - we're close to agreement"
    return "Clear and decisive - this is final position, but leave door open for acceptance"


class _InMemoryCursor:
    def __init__(self) -> None:
        self.description = None

    def __enter__(self) -> "_InMemoryCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False

    def execute(self, *args, **kwargs) -> None:  # pragma: no cover - noop
        self.description = []
        return None

    def fetchone(self):  # pragma: no cover - noop
        return None

    def fetchall(self):  # pragma: no cover - noop
        return []


class _InMemoryConnection:
    def __enter__(self) -> "_InMemoryConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False

    def cursor(self) -> _InMemoryCursor:
        return _InMemoryCursor()

    def commit(self) -> None:  # pragma: no cover - noop
        return None


class _NullS3Client:
    def get_object(self, *args, **kwargs):  # pragma: no cover - noop
        raise FileNotFoundError

    def put_object(self, *args, **kwargs):  # pragma: no cover - noop
        return {}


@contextmanager
def _null_db_connection():
    yield _InMemoryConnection()


@contextmanager
def _null_s3_connection():
    yield _NullS3Client()


@dataclass
class ThreadHeaders:
    message_id: Optional[str] = None
    references: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.message_id:
            payload["message_id"] = self.message_id
        if self.references:
            payload["references"] = list(self.references)
        return payload


@dataclass
class DecisionContext:
    unique_id: Optional[str] = None
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    current_offer: Optional[float] = None
    currency: Optional[str] = None
    lead_time_weeks: Optional[float] = None
    target_price: Optional[float] = None
    round: Optional[int] = None
    strategy: Optional[str] = None
    counter_price: Optional[float] = None
    asks: List[str] = field(default_factory=list)
    lead_time_request: Optional[str] = None
    rationale: Optional[str] = None
    thread: ThreadHeaders = field(default_factory=ThreadHeaders)

    def to_public_json(self) -> Dict[str, Any]:
        payload = {
            "unique_id": self.unique_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "current_offer": self.current_offer,
            "currency": self.currency,
            "lead_time_weeks": self.lead_time_weeks,
            "target_price": self.target_price,
            "round": self.round,
            "strategy": self.strategy,
            "counter_price": self.counter_price,
            "asks": list(self.asks),
            "lead_time_request": self.lead_time_request,
            "rationale": self.rationale,
        }
        if isinstance(self.thread, ThreadHeaders):
            headers = self.thread.to_dict()
            if headers:
                payload["thread"] = headers
        return payload

DEFAULT_PROMPT_TEMPLATE = (
    "[Subject]\n\n"
    "Request for Quotation (RFQ) – Office Furniture\n\n"
    "[Greeting Block]\n\n"
    "Dear {supplier_contact_name},\n\n"
    "[Introduction and Context]\n\n"
    "We are requesting a quotation for the following items/services as part of our sourcing process.\n"
    "Please complete the table in full to ensure your proposal can be evaluated accurately.\n\n"
    "[Structured Pricing & Commercial Table]\n\n"
    "Item ID Description  UOM  Qty  Unit Price (Currency)  Extended Price  Delivery Lead Time (days)  "
    "Payment Terms (days)  Warranty / Support  Contract Ref  Comments\n\n"
    "[Terms & Conditions Confirmation]\n\n"
    "Please confirm one of the following options:\n\n"
    "[ ] I accept Your Company’s Standard Terms & Conditions\n    (https://yourcompany.com/procurement-terms)\n\n"
    "[ ] I wish to proceed under my current contract with Your Company:\n"
    "    Contract Number [__________]\n\n"
    "[Submission Instructions]\n\n"
    "Deadline for submission: {submission_deadline}\n\n"
    "Please return the completed table and confirmation via reply to this email.\n"
    "For queries, contact {category_manager_name}, {category_manager_title}, {category_manager_email}.\n\n"
    "[Closing Block]\n\n"
    "Thank you for your response. We look forward to your proposal.\n\n"
    "Kind regards,\n{your_name}\n{your_title}\n{your_company}"
)

_RFQ_TABLE_COLUMNS: List[str] = [
    "Item Description",
    "UOM",
    "Qty",
    "Target Unit Price (Currency)",
    "Extended Price",
    "Delivery Lead Time (days)",
    "Payment Terms (days)",
    "Warranty / Support",
    "Contract Ref",
    "Comments",
]
_RFQ_TABLE_STYLE = (
    "border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 13px;"
)
_RFQ_HEADER_CELL_STYLE = (
    "border: 1px solid #d0d0d0; padding: 6px; text-align: left; background-color: #f5f5f5;"
)
_RFQ_BODY_CELL_STYLE = "border: 1px solid #d0d0d0; padding: 6px; text-align: left;"
DEFAULT_RFQ_SUBJECT = "Sourcing Request – Pricing Discussion"
DEFAULT_NEGOTIATION_SUBJECT = "Re: Pricing Discussion"
DEFAULT_FOLLOW_UP_SUBJECT = "Follow Up – Procurement Enquiry"


def _current_rfq_date() -> str:
    """Return today's date formatted for legacy RFQ identifiers."""

    return datetime.utcnow().strftime("%Y%m%d")


def _build_rfq_table_html(descriptions: Iterable[str]) -> str:
    header_cells = "".join(
        f'<th style="{_RFQ_HEADER_CELL_STYLE}">{escape(col)}</th>'
        for col in _RFQ_TABLE_COLUMNS
    )

    rows = [desc for desc in descriptions if isinstance(desc, str) and desc.strip()]

    body_rows: List[str] = []
    for description in rows:
        first_cell = escape(description.strip()) if description else "&nbsp;"
        cells = [f'<td style="{_RFQ_BODY_CELL_STYLE}">{first_cell}</td>']
        for _ in range(len(_RFQ_TABLE_COLUMNS) - 1):
            cells.append(f'<td style="{_RFQ_BODY_CELL_STYLE}">&nbsp;</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    blank_cells = [
        f'<td style="{_RFQ_BODY_CELL_STYLE}">&nbsp;</td>' for _ in _RFQ_TABLE_COLUMNS
    ]
    body_rows.append(f"<tr>{''.join(blank_cells)}</tr>")

    return (
        f'<table class="rfq-table" style="{_RFQ_TABLE_STYLE}">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
    )


RFQ_TABLE_HEADER = _build_rfq_table_html([])


DEFAULT_NEGOTIATION_MODEL = "mistral"


class EmailDraftingAgent(BaseAgent):
    """Agent that drafts a plain-text RFQ email and sends it via SES."""

    AGENTIC_PLAN_STEPS = (
        "Review negotiation or sourcing context plus relevant policy guidance for tone and compliance.",
        "Compile structured data, pricing rationales, and requested actions to inform the draft email.",
        "Generate candidate drafts, validate for policy alignment, and surface routing metadata for follow-up.",
    )

    TEXT_TEMPLATE = (
        "<p>Dear {supplier_contact_name_html},</p>"
        "<p>We are writing to request a formal quotation for the requirement outlined below.</p>"
        "<p>{scope_summary_html}</p>"
        "<p>Please review the enclosed details and let us know if you require any clarification.</p>"
        "<p>Kindly complete the table and return your quotation by {deadline_html}.</p>"
        "{rfq_table_html}"
        "<p>We appreciate your timely response.</p>"
        "<p>Kind regards,<br>{your_name_html}<br>{your_title_html}<br>{your_company_html}</p>"
    )

    @staticmethod
    def _strip_all_internal_identifiers(text: Optional[str]) -> str:
        """Remove ALL internal identifiers from text."""
        if not isinstance(text, str):
            return ""

        cleaned = re.sub(
            r"(?i)\b(RFQ|rfq)[\s:-]*\d{8}[\s:-]*[A-Za-z0-9]{8}\b",
            "",
            text,
        )

        cleaned = re.sub(
            r"(?i)\bRFQ[\s:-]*[A-Za-z0-9-]{3,}\b",
            "",
            cleaned,
        )

        cleaned = re.sub(r"(?i)\bRFQ\b", "", cleaned)

        cleaned = re.sub(
            r"(?i)\b(UID|uid)[\s:-]+[A-Za-z0-9-]{6,}\b",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"(?i)\bPROC-WF-[A-F0-9]{6,}\b",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"(?i)\b(WF|workflow|wf)[\s:-]+[A-Za-z0-9-]{6,}\b",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"(?i)\b(id|ref|reference)[\s:-]+[A-Za-z0-9-]{6,}\b",
            "",
            cleaned,
        )

        cleaned = re.sub(r"\s{2,}", " ", cleaned)

        return cleaned.strip()

    @staticmethod
    def _strip_rfq_identifier_tokens(text: Optional[str]) -> str:
        """Legacy method - now calls comprehensive cleaner."""
        return EmailDraftingAgent._strip_all_internal_identifiers(text)

    @classmethod
    def _clean_subject_text(cls, subject: Optional[str], fallback: str) -> str:
        prefix = ""
        candidate = subject.strip() if isinstance(subject, str) else ""

        if candidate:
            prefix_match = re.match(r"(?i)^(re|fw|fwd):\s*", candidate)
            if prefix_match:
                prefix = prefix_match.group(0)
                candidate = candidate[prefix_match.end():]

            raw_candidate = candidate
            candidate = cls._strip_all_internal_identifiers(candidate)
            candidate = candidate.strip("-–: ")

            if candidate and raw_candidate and candidate != raw_candidate:
                if len(candidate.split()) <= 1:
                    candidate = ""

            if candidate:
                return f"{prefix}{candidate}".strip()

        fallback_clean = cls._strip_all_internal_identifiers(fallback or "").strip("-–: ")
        if prefix and fallback_clean:
            fallback_without_prefix = re.sub(r"(?i)^(re|fw|fwd):\s*", "", fallback_clean)
            combined = f"{prefix}{fallback_without_prefix}".strip()
            return combined or prefix.strip()

        result = fallback_clean or prefix.strip()
        return result if result else "Procurement Discussion"

    @staticmethod
    def _clean_body_text(body: Optional[str]) -> str:
        """Clean body text by removing ALL internal identifiers."""
        if not isinstance(body, str):
            return ""

        cleaned = EmailDraftingAgent._strip_all_internal_identifiers(body)

        cleaned = re.sub(
            r"(?i)^\s*(RFQ|UID|Ref|Reference|ID|Workflow)[\s:-]+[A-Za-z0-9-]+.*?$",
            "",
            cleaned,
            flags=re.MULTILINE,
        )

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @staticmethod
    def _detect_invocation_mode(context: AgentContext, data: Mapping[str, Any]) -> str:
        """Determine whether the agent was called by the negotiation flow or directly."""

        parent = getattr(context, "parent_agent", None)
        if isinstance(parent, str) and "negotiationagent" in parent.lower():
            return "negotiation"

        trigger = data.get("invoked_by") if isinstance(data, Mapping) else None
        if isinstance(trigger, str) and "negotiation" in trigger.lower():
            return "negotiation"

        return "independent"

    @staticmethod
    def _normalise_thread_state(raw_state: Any) -> Dict[str, Any]:
        """Coerce ``raw_state`` into a predictable structure for thread tracking."""

        if isinstance(raw_state, dict):
            state = {key: value for key, value in raw_state.items() if value is not None}
        else:
            state = {}

        suppliers = state.get("suppliers")
        if not isinstance(suppliers, dict):
            suppliers_map: Dict[str, Dict[str, Any]] = {}
        else:
            suppliers_map = {}
            for key, value in suppliers.items():
                supplier_key = str(key)
                supplier_state: Dict[str, Any]
                if isinstance(value, dict):
                    supplier_state = {
                        k: v for k, v in value.items() if v is not None
                    }
                    messages = supplier_state.get("messages")
                    if isinstance(messages, list):
                        supplier_state["messages"] = [
                            message
                            for message in messages
                            if isinstance(message, dict) and message.get("body")
                        ]
                    else:
                        supplier_state["messages"] = []
                else:
                    supplier_state = {"messages": []}
                suppliers_map[supplier_key] = supplier_state

        state["suppliers"] = suppliers_map
        return state

    @staticmethod
    def _thread_supplier_key(supplier_id: Optional[Any]) -> str:
        if supplier_id is None:
            return "default"
        try:
            text = str(supplier_id).strip()
        except Exception:
            text = "default"
        return text or "default"

    def _apply_thread_state(
        self,
        *,
        supplier_id: Optional[str],
        subject: str,
        base_body: str,
        thread_state: Dict[str, Any],
        author: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Update ``thread_state`` and return enriched body and subject."""

        state = self._normalise_thread_state(thread_state)
        supplier_key = self._thread_supplier_key(supplier_id)
        supplier_state = dict(state.get("suppliers", {}).get(supplier_key, {}))
        messages: List[Dict[str, Any]] = []

        existing_messages = supplier_state.get("messages")
        if isinstance(existing_messages, list):
            messages = [
                dict(entry)
                for entry in existing_messages
                if isinstance(entry, dict) and entry.get("body")
            ]

        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        new_message = {
            "author": author or "ProcWise",  # default identity
            "body": base_body.strip(),
            "timestamp": timestamp,
            "subject": subject,
        }

        if messages and messages[-1].get("body") == new_message["body"]:
            messages[-1].update(new_message)
        else:
            messages.append(new_message)

        supplier_state["messages"] = messages
        supplier_state["last_subject"] = subject
        supplier_state["last_updated"] = timestamp

        history_blocks: List[str] = []
        if len(messages) > 1:
            previous_messages = messages[:-1]
            previous_messages.sort(
                key=lambda entry: entry.get("timestamp") or "",
            )
            for entry in previous_messages:
                entry_body = entry.get("body") or ""
                entry_author = entry.get("author") or "Supplier"
                entry_timestamp = entry.get("timestamp") or ""
                history_header = (
                    f"On {entry_timestamp}, {entry_author} wrote:".strip()
                )
                history_segment = "\n".join(
                    part for part in [history_header, entry_body.strip()] if part
                )
                if history_segment:
                    history_blocks.append(history_segment)

        final_body = base_body.strip()
        final_subject = subject
        if history_blocks:
            history_text = "\n\n---\n".join(history_blocks)
            final_body = (
                f"{final_body}\n\n--- Previous Conversation ---\n{history_text}"
            )
            if subject and not subject.lower().startswith("re:"):
                final_subject = f"RE: {subject}"

        suppliers_map = dict(state.get("suppliers", {}))
        suppliers_map[supplier_key] = supplier_state
        state["suppliers"] = suppliers_map
        state["updated_at"] = timestamp

        return final_body, final_subject, state

    def _render_supplier_draft(
        self,
        *,
        supplier: Dict[str, Any],
        index: int,
        workflow_id: str,
        data: Dict[str, Any],
        supplier_profiles: Dict[str, Any],
        base_body_template: Any,
        subject_template_source: Optional[str],
        include_rfq_table: bool,
        additional_section_html: str,
        compliance_section_html: str,
        instruction_suffix: str,
        negotiation_section_html: str,
        negotiation_message: Optional[str],
        instruction_settings: Dict[str, Any],
        default_action_id: Optional[str],
        negotiation_message_plain: Optional[str],
        base_thread_state: Dict[str, Any],
        resolve_action_id: Callable[[Optional[Dict[str, Any]]], Optional[str]],
        sender_email: str,
    ) -> SupplierDraftResult:
        """Render a single supplier draft returning the draft and updated thread state."""

        supplier_id = supplier.get("supplier_id")
        supplier_name = supplier.get("supplier_name", supplier_id)
        unique_id = self._generate_unique_identifier(workflow_id, supplier_id)

        profile = (
            supplier_profiles.get(str(supplier_id))
            if supplier_id is not None
            else {}
        )
        if profile is None:
            profile = {}

        sender_name, sender_title = self._derive_sender_identity(
            sender_email, data.get("sender_title")
        )

        fmt_args = {
            "supplier_contact_name": supplier.get("contact_name")
            or supplier_name
            or "Supplier",
            "supplier_company": supplier_name or supplier_id or "",
            "supplier_contact_email": supplier.get("contact_email", ""),
            "deadline": data.get("deadline")
            or data.get("submission_deadline", ""),
            "category_manager_name": data.get("category_manager_name", ""),
            "category_manager_title": data.get("category_manager_title", ""),
            "category_manager_email": data.get("category_manager_email", ""),
            "your_name": sender_name,
            "your_title": sender_title,
            "your_company": data.get("your_company", "ProcWise"),
        }
        for key, value in list(fmt_args.items()):
            fmt_args[f"{key}_html"] = self._format_plain_text(value)

        body_template = base_body_template or self.TEXT_TEMPLATE
        contextual_args = self._build_template_args(
            supplier,
            profile,
            fmt_args,
            data,
            include_rfq_table=include_rfq_table,
        )
        template_args = {**data, **fmt_args, **contextual_args}
        template_args.update(
            {
                "unique_id": unique_id,
                "additional_paragraph_html": additional_section_html,
                "compliance_notice_html": compliance_section_html,
                "negotiation_message_html": negotiation_section_html,
            }
        )
        if negotiation_message:
            template_args["negotiation_message"] = negotiation_message

        interaction_type = self._determine_interaction_type(
            data, instruction_settings
        )
        template_args["interaction_type"] = interaction_type

        dynamic_meta = {
            "interaction_type": interaction_type,
            "instructions": instruction_settings,
            "tone": instruction_settings.get("tone") or data.get("tone"),
            "call_to_action": instruction_settings.get("call_to_action"),
            "context_note": instruction_settings.get("context_note"),
        }
        if negotiation_message:
            dynamic_meta["negotiation_message"] = negotiation_message

        if self._should_auto_compose(body_template, instruction_settings, data):
            rendered = self._render_dynamic_body(
                supplier,
                profile,
                template_args,
                data,
                dynamic_meta,
                include_rfq_table=include_rfq_table,
            )
        else:
            rendered = self._render_template_string(body_template, template_args)

        comment, message = self._split_existing_comment(rendered)
        message_content = message if comment else rendered
        appended_sections: List[str] = []
        if additional_section_html and additional_section_html not in message_content:
            appended_sections.append(additional_section_html)
        if compliance_section_html and compliance_section_html not in message_content:
            appended_sections.append(compliance_section_html)
        if appended_sections:
            message_content = f"{message_content}{''.join(appended_sections)}"
        elif instruction_suffix and instruction_suffix not in message_content:
            message_content = f"{message_content}{instruction_suffix}"

        content = self._sanitise_generated_body(message_content)
        if negotiation_section_html and negotiation_section_html not in content:
            content = f"{content}{negotiation_section_html}"
        body = self._clean_body_text(content)

        body, marker_token = attach_hidden_marker(
            body,
            supplier_id=supplier_id,
            unique_id=unique_id,
        )

        if subject_template_source:
            subject_args = dict(template_args)
            subject_args.setdefault("unique_id", unique_id)
            rendered_subject = self._render_template_string(
                subject_template_source, subject_args
            )
            subject = self._clean_subject_text(
                self._normalise_subject_line(rendered_subject, unique_id),
                DEFAULT_RFQ_SUBJECT,
            )
        else:
            subject = self._clean_subject_text(None, DEFAULT_RFQ_SUBJECT)

        draft_action_id = resolve_action_id(supplier) or default_action_id

        receiver = self._resolve_receiver(supplier, profile)
        recipients: List[str] = []
        if receiver:
            recipients = self._normalise_recipients([receiver])
        if recipients:
            receiver = recipients[0]
        contact_level = 1 if recipients else 0

        internal_context = {}
        if isinstance(dynamic_meta.get("internal_context"), dict):
            internal_context = {
                key: value
                for key, value in dynamic_meta["internal_context"].items()
                if value
            }
        if not internal_context:
            fallback_context = self._build_supplier_personalisation(
                supplier,
                profile,
                template_args,
                data,
                instruction_settings,
                interaction_type,
            )
            if fallback_context:
                internal_context = {
                    "supplier_context_html": fallback_context,
                    "supplier_context_text": self._html_to_plain_text(
                        fallback_context
                    ),
                }

        base_thread_state_supplier = {}
        supplier_key = self._thread_supplier_key(supplier_id)
        if isinstance(base_thread_state.get("suppliers"), dict):
            existing_state = base_thread_state["suppliers"].get(supplier_key)
            if isinstance(existing_state, dict):
                base_thread_state_supplier = dict(existing_state)

        enriched_body, enriched_subject, updated_state = self._apply_thread_state(
            supplier_id=supplier_id,
            subject=subject,
            base_body=body,
            thread_state={"suppliers": {supplier_key: base_thread_state_supplier}},
            author=data.get("author") or sender_name,
        )

        body = enriched_body
        subject = enriched_subject

        draft = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "subject": subject,
            "body": body,
            "sent_status": False,
            "sender": sender_email,
            "action_id": draft_action_id,
            "supplier_profile": profile,
            "receiver": receiver,
            "contact_level": contact_level,
            "recipients": recipients,
            "unique_id": unique_id,
            "workflow_id": workflow_id,
            "draft_id": unique_id,
        }
        if draft_action_id:
            draft["action_id"] = draft_action_id
        draft.setdefault("thread_index", 1)

        metadata: Dict[str, Any] = {
            "unique_id": unique_id,
            "interaction_type": interaction_type,
            "workflow_id": workflow_id,
        }
        if supplier_id is not None:
            metadata["supplier_id"] = supplier_id
        if supplier_name:
            metadata["supplier_name"] = supplier_name
        if internal_context:
            metadata["internal_context"] = internal_context
        if marker_token:
            metadata["dispatch_token"] = marker_token
        if negotiation_message_plain:
            metadata.setdefault("negotiation_message", negotiation_message_plain)

        draft["metadata"] = metadata

        supplier_state = updated_state.get("suppliers", {}).get(supplier_key, {})
        supplier_state = dict(supplier_state) if isinstance(supplier_state, dict) else {}

        return SupplierDraftResult(
            index=index,
            supplier_id=self._coerce_text(supplier_id),
            draft=draft,
            supplier_thread_state=supplier_state,
        )

    def _determine_parallel_workers(self, supplier_count: int) -> int:
        if supplier_count <= 1:
            return 1

        configured = 0
        try:
            raw_value = os.getenv("EMAIL_DRAFTING_WORKERS")
            if raw_value:
                configured = int(raw_value)
        except Exception:
            configured = 0

        if configured > 0:
            return max(1, min(configured, supplier_count))

        cpu_default = os.cpu_count() or 4
        return max(1, min(cpu_default, supplier_count))

    def __init__(self, agent_nick=None):
        agent_nick = self._prepare_agent_nick(agent_nick)
        super().__init__(agent_nick)
        self._draft_table_checked = False
        self._draft_table_lock = threading.Lock()
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        settings = self.agent_nick.settings
        self.compose_model = getattr(
            settings,
            "email_compose_model",
            getattr(settings, "negotiation_email_model", DEFAULT_NEGOTIATION_MODEL),
        )
        self.polish_model = getattr(settings, "email_polish_model", None)
        self.workflow_memory = getattr(self.agent_nick, "workflow_memory", None)

    # ------------------------------------------------------------------
    # Workflow memory helpers
    # ------------------------------------------------------------------

    def _memory_enabled(self) -> bool:
        return bool(self.workflow_memory and getattr(self.workflow_memory, "enabled", False))

    def _record_thread_message(
        self,
        *,
        workflow_id: Optional[str],
        unique_id: Optional[str],
        role: str,
        subject: Optional[str],
        body_html: Optional[str],
        body_text: Optional[str],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self._memory_enabled() or not workflow_id or not unique_id:
            return
        payload: Dict[str, Any] = {
            "role": role,
            "subject": subject,
            "body_html": body_html,
            "body_text": body_text,
            "metadata": dict(metadata or {}),
        }
        try:
            self.workflow_memory.record_email_message(workflow_id, unique_id, payload)
        except Exception:  # pragma: no cover - defensive telemetry
            logger.debug(
                "Failed to record email thread message for workflow=%s unique_id=%s",
                workflow_id,
                unique_id,
                exc_info=True,
            )

    def _thread_history(self, workflow_id: Optional[str], unique_id: Optional[str]) -> List[Dict[str, Any]]:
        if not self._memory_enabled() or not workflow_id or not unique_id:
            return []
        try:
            return self.workflow_memory.get_thread_messages(workflow_id, unique_id)
        except Exception:  # pragma: no cover - defensive retrieval
            logger.debug(
                "Failed to load thread history for workflow=%s unique_id=%s",
                workflow_id,
                unique_id,
                exc_info=True,
            )
            return []

    @staticmethod
    def _format_thread_header(entry: Mapping[str, Any]) -> str:
        role = str(entry.get("role") or entry.get("direction") or "Message").strip()
        subject = str(entry.get("subject") or "").strip()
        timestamp = entry.get("timestamp")
        ts_text = None
        try:
            if isinstance(timestamp, (int, float)):
                ts_text = datetime.utcfromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M UTC")
            elif isinstance(timestamp, str) and timestamp:
                ts_text = timestamp
        except Exception:
            ts_text = None
        prefix = role.capitalize() if role else "Message"
        if ts_text and subject:
            return f"On {ts_text} – {prefix} wrote (Subject: {subject})"
        if ts_text:
            return f"On {ts_text} – {prefix} wrote"
        if subject:
            return f"{prefix} wrote (Subject: {subject})"
        return f"{prefix} wrote"

    def _render_thread_history(self, workflow_id: Optional[str], unique_id: Optional[str]) -> str:
        entries = self._thread_history(workflow_id, unique_id)
        if not entries:
            return ""
        blocks: List[str] = ["<hr>", "<p><em>Previous conversation</em></p>"]
        for entry in reversed(entries):
            if not isinstance(entry, Mapping):
                continue
            header = self._format_thread_header(entry)
            body_html = entry.get("body_html") or ""
            if not body_html:
                body_text = entry.get("body_text") or ""
                body_html = self._render_html_from_text(str(body_text)) if body_text else ""
            blocks.append(f"<p><strong>{escape(header)}</strong></p>")
            if body_html:
                blocks.append(f"<blockquote>{body_html}</blockquote>")
        return "".join(blocks)

    def _inject_thread_history(
        self,
        *,
        workflow_id: Optional[str],
        unique_id: Optional[str],
        body_html: str,
    ) -> str:
        thread_html = self._render_thread_history(workflow_id, unique_id)
        if not thread_html:
            return body_html
        return f"{body_html}{thread_html}"

    @staticmethod
    def _prepare_learning_snapshot(draft: Mapping[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "unique_id": draft.get("unique_id"),
            "supplier_id": draft.get("supplier_id"),
            "supplier_name": draft.get("supplier_name"),
            "subject": draft.get("subject"),
            "contact_level": draft.get("contact_level"),
            "recipients": draft.get("recipients"),
            "sender": draft.get("sender"),
        }
        metadata = draft.get("metadata") if isinstance(draft.get("metadata"), Mapping) else {}
        if metadata:
            preserved_keys = (
                "intent",
                "round",
                "strategy",
                "dispatch_token",
                "workflow_id",
                "interaction_type",
            )
            snapshot["metadata"] = {
                key: metadata[key]
                for key in preserved_keys
                if metadata.get(key) is not None
            }
        return snapshot


    # ------------------------------------------------------------------
    # Logging overrides
    # ------------------------------------------------------------------

    def _prepare_logged_output(self, payload: Any) -> Any:
        """Normalise logged output to the contract expected by proc.action."""

        prepared = super()._prepare_logged_output(payload)
        return self._normalise_action_payload(prepared)

    def _normalise_action_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            cleaned: Dict[str, Any] = {}
            pending_unique: Optional[str] = None
            for key, value in payload.items():
                if key == "rfq_id":
                    if isinstance(value, str) and value.strip():
                        pending_unique = value.strip()
                    continue
                cleaned[key] = self._normalise_action_payload(value)

            if pending_unique and "unique_id" not in cleaned:
                cleaned["unique_id"] = pending_unique

            drafts_obj = cleaned.get("drafts")
            if isinstance(drafts_obj, list):
                normalised_drafts: List[Any] = []
                for draft_item in drafts_obj:
                    if isinstance(draft_item, dict):
                        normalised_drafts.append(self._coerce_draft_schema(draft_item))
                    else:
                        normalised_drafts.append(
                            self._normalise_action_payload(draft_item)
                        )
                cleaned["drafts"] = normalised_drafts

            return cleaned

        if isinstance(payload, list):
            return [self._normalise_action_payload(item) for item in payload]

        return payload

    def _coerce_draft_schema(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_draft = self._normalise_action_payload(draft)

        # Ensure ``unique_id`` is always populated
        unique_id = cleaned_draft.get("unique_id")
        metadata = cleaned_draft.get("metadata")
        if not unique_id and isinstance(metadata, dict):
            unique_id = metadata.get("unique_id")
        if not unique_id:
            headers = cleaned_draft.get("headers")
            if isinstance(headers, dict):
                unique_id = headers.get("X-ProcWise-Unique-ID") or headers.get(
                    "X-Procwise-Unique-Id"
                )
        if unique_id and "unique_id" not in cleaned_draft:
            cleaned_draft["unique_id"] = unique_id

        # Align top-level subject/body for log payload consumers
        if isinstance(metadata, dict):
            subject = metadata.get("subject")
            if subject and "subject" not in cleaned_draft:
                cleaned_draft["subject"] = subject
            body = metadata.get("body")
            if body and "body" not in cleaned_draft:
                cleaned_draft["body"] = body

        return cleaned_draft

    @staticmethod
    def _prepare_agent_nick(agent_nick):
        if agent_nick is None:
            settings = SimpleNamespace(
                ses_default_sender="procurement@example.com",
                script_user="AgentNick",
                negotiation_email_model=DEFAULT_NEGOTIATION_MODEL,
                email_compose_model=DEFAULT_NEGOTIATION_MODEL,
                email_polish_model=None,
            )
            process_routing = SimpleNamespace(
                log_process=lambda **kwargs: None,
                log_run_detail=lambda **kwargs: None,
                log_action=lambda **kwargs: None,
            )
            agent_nick = SimpleNamespace(
                settings=settings,
                process_routing_service=process_routing,
                prompt_engine=None,
                learning_repository=None,
                ollama_options=lambda: {},
                get_db_connection=lambda: _null_db_connection(),
                reserve_s3_connection=lambda: _null_s3_connection(),
            )
            return agent_nick

        settings = getattr(agent_nick, "settings", None)
        if settings is None:
            settings = SimpleNamespace(
                ses_default_sender="procurement@example.com",
                script_user="AgentNick",
            )
            setattr(agent_nick, "settings", settings)
        def _safe_assign(target, name, value):
            try:
                setattr(target, name, value)
                return True
            except (AttributeError, ValueError):
                return False

        if not getattr(settings, "ses_default_sender", None):
            _safe_assign(settings, "ses_default_sender", "procurement@example.com")
        if not getattr(settings, "script_user", None):
            _safe_assign(settings, "script_user", "AgentNick")

        if not getattr(settings, "negotiation_email_model", None):
            _safe_assign(settings, "negotiation_email_model", DEFAULT_NEGOTIATION_MODEL)

        if not getattr(settings, "email_compose_model", None):
            _safe_assign(
                settings,
                "email_compose_model",
                getattr(settings, "negotiation_email_model", DEFAULT_NEGOTIATION_MODEL),
            )

        if not hasattr(settings, "email_polish_model"):
            _safe_assign(settings, "email_polish_model", None)

        if not hasattr(agent_nick, "process_routing_service"):
            setattr(
                agent_nick,
                "process_routing_service",
                SimpleNamespace(
                    log_process=lambda **kwargs: None,
                    log_run_detail=lambda **kwargs: None,
                    log_action=lambda **kwargs: None,
                ),
            )
        else:
            routing = agent_nick.process_routing_service
            if not hasattr(routing, "log_process"):
                routing.log_process = lambda **kwargs: None
            if not hasattr(routing, "log_run_detail"):
                routing.log_run_detail = lambda **kwargs: None
            if not hasattr(routing, "log_action"):
                routing.log_action = lambda **kwargs: None

        if not hasattr(agent_nick, "ollama_options"):
            agent_nick.ollama_options = lambda: {}
        if not hasattr(agent_nick, "get_db_connection"):
            agent_nick.get_db_connection = lambda: _null_db_connection()
        if not hasattr(agent_nick, "reserve_s3_connection"):
            agent_nick.reserve_s3_connection = lambda: _null_s3_connection()
        if not hasattr(agent_nick, "learning_repository"):
            agent_nick.learning_repository = None

        return agent_nick

    def from_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        decision_data = dict(decision or {})
        supplier_id = decision_data.get("supplier_id")
        supplier_name = decision_data.get("supplier_name") or supplier_id
        to_list = self._normalise_recipients(
            decision_data.get("to") or decision_data.get("recipients")
        )
        cc_list = self._normalise_recipients(decision_data.get("cc"))
        recipients = self._merge_recipients(to_list, cc_list)
        receiver = to_list[0] if to_list else None
        sender = decision_data.get("sender") or self.agent_nick.settings.ses_default_sender

        existing_unique = self._normalise_tracking_value(decision_data.get("unique_id"))
        workflow_hint = decision_data.get("workflow_id") or decision_data.get("workflow_ref")
        unique_id = existing_unique or self._generate_unique_identifier(
            workflow_hint, supplier_id
        )

        asks_raw = decision_data.get("asks") or decision_data.get("counter_points") or []
        if isinstance(asks_raw, str):
            asks = [asks_raw]
        elif isinstance(asks_raw, Iterable):
            asks = [
                str(item).strip()
                for item in asks_raw
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
        else:
            asks = []

        lead_time_request = (
            decision_data.get("lead_time_request")
            or decision_data.get("lead_time")
            or decision_data.get("lead_time_days")
        )
        negotiation_message = decision_data.get("negotiation_message") or decision_data.get(
            "message"
        )

        payload = {
            "unique_id": unique_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "current_offer": decision_data.get("current_offer"),
            "counter_price": decision_data.get("counter_price"),
            "target_price": decision_data.get("target_price"),
            "currency": decision_data.get("currency"),
            "lead_time_weeks": decision_data.get("lead_time_weeks"),
            "lead_time_request": lead_time_request,
            "asks": asks,
            "strategy": decision_data.get("strategy"),
            "negotiation_message": negotiation_message,
        }
        negotiation_body = (
            negotiation_message.strip()
            if isinstance(negotiation_message, str)
            else ""
        )
        use_negotiation_content = bool(negotiation_body)

        subject_line = decision_data.get("subject") if use_negotiation_content else None
        body_text = negotiation_body if use_negotiation_content else ""
        fallback_text = negotiation_body if use_negotiation_content else ""

        if not use_negotiation_content:
            user_prompt = (
                f"Context (JSON):\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
                "Write a professional negotiation counter email with an explicit Subject line. "
                "Summarise the commercial position, list the key asks as bullet points, and close "
                "with a clear call to action."
            )

            model_name = getattr(
                self.agent_nick.settings,
                "negotiation_email_model",
                DEFAULT_NEGOTIATION_MODEL,
            )
            try:
                response_text = _chat(
                    model_name,
                    SYSTEM_COMPOSE,
                    user_prompt,
                    agent=self,
                    options={"temperature": 0.35, "top_p": 0.9},
                )
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception("Failed to compose negotiation counter email")
                response_text = ""

            subject_line, body_text = self._split_subject_and_body(response_text)
            fallback_text = self._build_negotiation_fallback(
                unique_id,
                decision_data.get("counter_price"),
                decision_data.get("target_price"),
                decision_data.get("current_offer"),
                decision_data.get("currency"),
                asks,
                lead_time_request,
                negotiation_message,
                decision_data.get("round"),
                decision_data.get("line_items") or decision_data.get("pricing_table"),
            )
            if not body_text:
                body_text = fallback_text

        html_body = self._render_html_from_text(body_text or fallback_text or "")
        sanitised_html = self._sanitise_generated_body(html_body)
        if not sanitised_html and fallback_text:
            sanitised_html = self._sanitise_generated_body(
                self._render_html_from_text(fallback_text)
            )
        plain_text = (
            self._html_to_plain_text(sanitised_html)
            if sanitised_html
            else (body_text or fallback_text)
        )
        if not sanitised_html and plain_text:
            sanitised_html = self._render_html_from_text(plain_text)

        if sanitised_html:
            sanitised_html = self._clean_body_text(sanitised_html)
        if plain_text:
            plain_text = self._clean_body_text(plain_text)

        unique_id = self._resolve_unique_id(
            workflow_id=workflow_hint,
            supplier_id=supplier_id,
            existing=unique_id,
        )

        base_body = sanitised_html or plain_text or fallback_text or ""
        threaded_body = self._inject_thread_history(
            workflow_id=workflow_hint,
            unique_id=unique_id,
            body_html=base_body,
        )
        annotated_body, marker_token = attach_hidden_marker(
            threaded_body,
            supplier_id=supplier_id,
            unique_id=unique_id,
        )
        marker_comment, visible_body = split_hidden_marker(annotated_body)
        if visible_body:
            plain_text = self._clean_body_text(visible_body)
        if subject_line:
            subject = self._clean_subject_text(subject_line.strip(), DEFAULT_NEGOTIATION_SUBJECT)
        else:
            subject = self._clean_subject_text(None, DEFAULT_NEGOTIATION_SUBJECT)

        thread_headers = decision_data.get("thread")
        resolved_thread_headers: Dict[str, Any] = {}
        if isinstance(thread_headers, dict):
            resolved_thread_headers = dict(thread_headers)
        headers: Dict[str, Any] = {
            "X-ProcWise-Unique-ID": unique_id,
            "X-ProcWise-Workflow-ID": workflow_hint,
            "X-ProcWise-Round": "0",
        }
        if supplier_id:
            headers["X-ProcWise-Supplier-ID"] = supplier_id
        message_id = None
        if resolved_thread_headers:
            message_id = resolved_thread_headers.get("Message-ID") or resolved_thread_headers.get("message_id")
            in_reply_to = (
                resolved_thread_headers.get("In-Reply-To")
                or resolved_thread_headers.get("in_reply_to")
            )
            references = resolved_thread_headers.get("References") or resolved_thread_headers.get("references")
            if message_id:
                headers["Message-ID"] = message_id
            if in_reply_to:
                headers["In-Reply-To"] = in_reply_to
            else:
                legacy_reply = (
                    resolved_thread_headers.get("message_id")
                    or resolved_thread_headers.get("Message-ID")
                )
                if legacy_reply:
                    headers.setdefault("In-Reply-To", legacy_reply)
            if isinstance(references, list) and references:
                headers["References"] = " ".join(str(ref) for ref in references if ref)
            elif isinstance(references, str) and references.strip():
                headers["References"] = references.strip()
            resolved_thread_headers = {
                key: value for key, value in resolved_thread_headers.items() if value is not None
            }
            if message_id:
                resolved_thread_headers.setdefault("Message-ID", message_id)

        headers = {k: v for k, v in headers.items() if v is not None}

        metadata = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "counter_price": decision_data.get("counter_price"),
            "target_price": decision_data.get("target_price"),
            "current_offer": decision_data.get("current_offer"),
            "currency": decision_data.get("currency"),
            "lead_time_weeks": decision_data.get("lead_time_weeks"),
            "lead_time_request": lead_time_request,
            "strategy": decision_data.get("strategy"),
            "intent": "NEGOTIATION_COUNTER",
        }
        if marker_token:
            metadata["dispatch_token"] = marker_token

        self._record_thread_message(
            workflow_id=workflow_hint,
            unique_id=unique_id,
            role="buyer",
            subject=subject,
            body_html=base_body,
            body_text=visible_body or base_body,
            metadata={
                "round": decision_data.get("round"),
                "strategy": decision_data.get("strategy"),
                "negotiation_message": negotiation_message,
            },
        )
        metadata["unique_id"] = unique_id
        if workflow_hint:
            metadata["workflow_id"] = workflow_hint
        metadata = {k: v for k, v in metadata.items() if v is not None}

        draft = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "subject": subject,
            "body": annotated_body,
            "text": plain_text,
            "html": sanitised_html,
            "sender": sender,
            "recipients": recipients,
            "receiver": receiver,
            "to": receiver,
            "cc": cc_list,
            "contact_level": 1 if receiver else 0,
            "sent_status": False,
            "metadata": metadata,
            "headers": headers,
            "unique_id": unique_id,
        }
        if resolved_thread_headers:
            draft["thread_headers"] = resolved_thread_headers
        if message_id:
            draft["message_id"] = message_id

        thread_index = decision_data.get("thread_index")
        if thread_index is not None:
            draft["thread_index"] = thread_index

        return draft

    def from_prompt(
        self, prompt: str, *, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        context = dict(context or {})
        prompt_text = str(prompt or "").strip()
        recipients_input = context.get("recipients")
        to_source = context.get("to")
        to_list = self._normalise_recipients(to_source or recipients_input)
        cc_list = self._normalise_recipients(context.get("cc"))
        if not to_source and isinstance(recipients_input, Iterable):
            derived_cc = to_list[1:]
            if derived_cc:
                cc_list = self._merge_recipients(derived_cc, cc_list)
                to_list = to_list[:1]
        recipients = self._merge_recipients(to_list, cc_list)
        receiver = to_list[0] if to_list else None
        sender = context.get("sender") or self.agent_nick.settings.ses_default_sender

        supplier_id = context.get("supplier_id")
        if not supplier_id and isinstance(context.get("supplier"), Mapping):
            supplier_payload = context.get("supplier")
            supplier_id = (
                supplier_payload.get("supplier_id")
                if isinstance(supplier_payload, Mapping)
                else None
            )
            if not supplier_id and isinstance(supplier_payload, Mapping):
                supplier_id = supplier_payload.get("id")
        supplier_id = self._coerce_text(supplier_id)
        workflow_hint = context.get("workflow_id") if isinstance(context, dict) else None
        existing_unique = self._normalise_tracking_value(context.get("unique_id"))
        unique_id = existing_unique or self._generate_unique_identifier(
            workflow_hint, supplier_id
        )

        payload = {"unique_id": unique_id, "prompt": prompt_text}
        user_prompt = (
            f"Context (JSON):\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            "Compose a professional procurement email with a clear Subject line and structured body. "
            "Keep the tone courteous and do not expose internal scoring or analysis logic."
        )

        model_name = getattr(self.agent_nick.settings, "email_compose_model", self.compose_model)
        try:
            response_text = _chat(model_name, SYSTEM_PROMPT_COMPOSE, user_prompt, agent=self)
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("Failed to compose email from prompt")
            response_text = ""

        subject_line, body_text = self._split_subject_and_body(response_text)
        if not body_text:
            body_text = prompt_text

        html_body = self._render_html_from_text(body_text)
        sanitised_html = self._sanitise_generated_body(html_body)
        plain_text = self._html_to_plain_text(sanitised_html) if sanitised_html else body_text

        subject_candidate = subject_line or context.get("subject")
        subject = self._clean_subject_text(subject_candidate, DEFAULT_FOLLOW_UP_SUBJECT)

        polish_enabled = os.getenv("EMAIL_POLISH_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if polish_enabled and self.polish_model:
            polish_payload = {
                "unique_id": unique_id,
                "subject": subject,
                "body": plain_text,
            }
            polish_prompt = (
                f"Context (JSON):\n{json.dumps(polish_payload, ensure_ascii=False, default=str)}\n\n"
                "Polish this procurement email for clarity and executive tone. Return the refined email "
                "starting with a Subject line."
            )
            try:
                polished_text = _chat(
                    self.polish_model, SYSTEM_POLISH, polish_prompt, agent=self
                )
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception("Failed to polish composed email")
                polished_text = ""
            polish_subject, polish_body = self._split_subject_and_body(polished_text)
            if polish_subject:
                subject = self._clean_subject_text(polish_subject, subject)
            if polish_body:
                html_body = self._render_html_from_text(polish_body)
                sanitised_html = self._sanitise_generated_body(html_body)
                plain_text = self._html_to_plain_text(sanitised_html) if sanitised_html else polish_body
        if not sanitised_html and plain_text:
            sanitised_html = self._render_html_from_text(plain_text)

        if sanitised_html:
            sanitised_html = self._clean_body_text(sanitised_html)
        if plain_text:
            plain_text = self._clean_body_text(plain_text)

        unique_id = self._resolve_unique_id(
            workflow_id=workflow_hint,
            supplier_id=supplier_id,
            existing=unique_id,
        )
        base_body = sanitised_html or plain_text or ""
        threaded_body = self._inject_thread_history(
            workflow_id=workflow_hint,
            unique_id=unique_id,
            body_html=base_body,
        )
        annotated_body, marker_token = attach_hidden_marker(
            threaded_body,
            supplier_id=supplier_id,
            unique_id=unique_id,
        )
        _, visible_body = split_hidden_marker(annotated_body)

        counter_price = context.get("counter_price")
        if counter_price is None:
            proposals = context.get("counter_proposals")
            if isinstance(proposals, Iterable):
                for proposal in proposals:
                    if isinstance(proposal, dict) and proposal.get("price") is not None:
                        counter_price = proposal.get("price")
                        break

        round_hint = context.get("round")
        try:
            round_number = int(round_hint) if round_hint is not None else 0
        except (TypeError, ValueError):
            round_number = 0

        metadata = {
            "intent": context.get("intent") or "PROMPT_COMPOSE",
            "unique_id": unique_id,
        }
        if counter_price is not None:
            metadata["counter_price"] = counter_price
        if marker_token:
            metadata["dispatch_token"] = marker_token
        if workflow_hint:
            metadata["workflow_id"] = workflow_hint
        if supplier_id:
            metadata["supplier_id"] = supplier_id
        metadata["round"] = round_number
        metadata["round_number"] = round_number

<<<<<<< HEAD
        headers: Dict[str, Any] = {"X-ProcWise-Unique-ID": unique_id}
=======
        self._record_thread_message(
            workflow_id=workflow_hint,
            unique_id=unique_id,
            role="buyer",
            subject=subject,
            body_html=base_body,
            body_text=visible_body or base_body,
            metadata={"intent": metadata.get("intent")},
        )

        headers: Dict[str, Any] = {"X-Procwise-Unique-Id": unique_id}
>>>>>>> f6b29da (updated changes)
        if workflow_hint:
            headers["X-ProcWise-Workflow-ID"] = workflow_hint
        if supplier_id:
            headers["X-ProcWise-Supplier-ID"] = supplier_id
        headers["X-ProcWise-Round"] = str(round_number)

        draft = {
            "subject": subject,
            "body": annotated_body,
            "text": visible_body or plain_text,
            "html": sanitised_html,
            "sender": sender,
            "recipients": recipients,
            "receiver": receiver,
            "to": receiver,
            "cc": cc_list,
            "contact_level": 1 if receiver else 0,
            "sent_status": False,
            "metadata": metadata,
            "headers": headers,
            "unique_id": unique_id,
            "workflow_id": workflow_hint,
        }
        if supplier_id:
            draft["supplier_id"] = supplier_id
        draft["round"] = round_number
        draft["round_number"] = round_number

        return draft

    def _instruction_sources_from_prompt(self, prompt: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(prompt, dict):
            return sources
        for field in ("prompt_config", "metadata", "prompts_desc", "template"):
            value = prompt.get(field)
            if value:
                sources.append(value)
        return sources

    def _extract_prompt_template(self, prompt: Dict[str, Any]) -> Optional[str]:
        if not isinstance(prompt, dict):
            return None
        payload: Any = (
            prompt.get("prompts_desc")
            or prompt.get("template")
            or prompt.get("prompt_template")
        )
        if isinstance(payload, dict):
            template = payload.get("prompt_template") or payload.get("template")
            return str(template) if template else None
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode(errors="ignore")
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                return text
            if isinstance(parsed, dict):
                template = parsed.get("prompt_template") or parsed.get("template")
                return str(template) if template else None
            return text
        return None

    def _load_prompt_template_from_db(self, prompt_id: int) -> Optional[str]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT prompts_desc FROM proc.prompt WHERE prompt_id = %s",
                        (prompt_id,),
                    )
                    row = cursor.fetchone()
            if not row:
                return None
            payload = row[0]
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode(errors="ignore")
            if isinstance(payload, str):
                payload = payload.strip()
                if not payload:
                    return None
                try:
                    parsed = json.loads(payload)
                except Exception:
                    return payload
                if isinstance(parsed, dict):
                    template = parsed.get("prompt_template") or parsed.get("template")
                    return str(template) if template else None
                return None
            if isinstance(payload, dict):
                template = payload.get("prompt_template") or payload.get("template")
                return str(template) if template else None
            return None
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load prompt %s for email drafting", prompt_id)
            return None

    def _ensure_prompt_template(self, context: AgentContext) -> None:
        if self.prompt_template and self.prompt_template != DEFAULT_PROMPT_TEMPLATE:
            return

        seen_ids: set[int] = set()
        for prompt in context.input_data.get("prompts") or []:
            template = self._extract_prompt_template(prompt)
            if template:
                self.prompt_template = template
            pid = prompt.get("promptId") if isinstance(prompt, dict) else None
            try:
                if pid is not None:
                    seen_ids.add(int(pid))
            except (TypeError, ValueError):
                continue

        if (self.prompt_template == DEFAULT_PROMPT_TEMPLATE or not self.prompt_template) and seen_ids:
            for pid in seen_ids:
                template = self._load_prompt_template_from_db(pid)
                if template:
                    self.prompt_template = template
                    break

        if not self.prompt_template:
            self.prompt_template = DEFAULT_PROMPT_TEMPLATE

    def _needs_polish(self, body: str) -> bool:
        if not body:
            return False
        word_count = len(re.findall(r"\b\w+\b", body))
        if word_count < 120:
            return True
        if not re.search(r"\b(thank|appreciat)", body, re.IGNORECASE):
            return True
        return False

    def _maybe_polish_negotiation_email(
        self, subject: Optional[str], body: str
    ) -> Tuple[Optional[str], str]:
        if not self.polish_model or not self._needs_polish(body):
            return subject, body

        polish_enabled = (
            os.getenv("EMAIL_POLISH_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        )
        if not polish_enabled:
            return subject, body

        payload = {
            "subject": subject or "Negotiation Update",
            "body": body,
        }
        polish_prompt = (
            f"Context (JSON):\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            "Please refine this procurement negotiation email for tone and completeness. "
            "Preserve any factual data, prices, deadlines, and leverage points. Return the "
            "result starting with a Subject line followed by the body."
        )

        try:
            polished = _chat(
                self.polish_model,
                SYSTEM_POLISH,
                polish_prompt,
                agent=self,
            )
        except Exception:
            logger.exception("Failed to polish negotiation email draft")
            return subject, body

        clean_subject, clean_body = self._split_subject_and_body(polished)
        if clean_subject:
            subject = self._clean_subject_text(clean_subject, subject or DEFAULT_NEGOTIATION_SUBJECT)
        if clean_body:
            body = self._clean_body_text(clean_body)
        return subject, body

    @staticmethod
    def _summarise_supplier_message(message: Optional[str]) -> Optional[str]:
        if not message:
            return None
        text = EmailDraftingAgent._clean_body_text(str(message))
        if not text:
            return None
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= 280:
            return text
        return f"{text[:277].rstrip()}…"

    @staticmethod
    def _normalise_line_items(line_items: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(line_items, list):
            return None
        normalised: List[Dict[str, Any]] = []
        for item in line_items:
            if isinstance(item, dict):
                cleaned: Dict[str, Any] = {}
                for key in ("description", "item", "sku", "quantity", "unit_price", "currency"):
                    if key in item and item[key] is not None:
                        cleaned[key] = item[key]
                if cleaned:
                    normalised.append(cleaned)
        return normalised or None

    def _draft_intelligent_negotiation_email(
        self,
        context: AgentContext,
        data: Dict[str, Any],
    ) -> str:
        round_value = data.get("round") or 1
        try:
            round_no = int(round_value)
        except Exception:
            round_no = 1

        current_offer = data.get("current_offer_numeric")
        if current_offer is None:
            current_offer = data.get("current_offer")
        target_price = data.get("target_price")
        if target_price is None:
            target_price = data.get("counter_price")
        counter_price = data.get("counter_price")
        walkaway_price = data.get("walkaway_price")
        previous_counter = data.get("previous_counter") or data.get("previous_counter_price")
        currency = data.get("currency") or data.get("currency_code") or "GBP"

        gap_amount = None
        gap_percentage = 0.0
        if current_offer is not None and target_price is not None:
            try:
                current_float = float(current_offer)
                target_float = float(target_price)
                gap_amount = current_float - target_float
                gap_percentage = (gap_amount / current_float) * 100 if current_float else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                gap_amount = None
                gap_percentage = 0.0

        supplier_name = data.get("supplier_name") or data.get("metadata", {}).get("supplier_name") or "Supplier"
        current_offer_fmt = self._format_currency_value(current_offer, currency) or "Not specified"
        target_price_fmt = self._format_currency_value(target_price, currency) or "Not specified"
        gap_amount_fmt = (
            self._format_currency_value(gap_amount, currency)
            if gap_amount is not None
            else "TBD"
        )
        counter_price_fmt = self._format_currency_value(counter_price, currency) if counter_price is not None else None
        walkaway_price_fmt = (
            self._format_currency_value(walkaway_price, currency)
            if walkaway_price is not None
            else None
        )
        previous_counter_fmt = (
            self._format_currency_value(previous_counter, currency)
            if previous_counter is not None
            else None
        )

        supplier_history = {
            "reliable_partner": data.get("reliable_partner", False),
            "high_quality_score": (data.get("quality_score", 0) or 0) > 80,
            "past_spend": data.get("total_spend"),
            "past_relationship": bool(data.get("total_spend")),
        }

        supplier_message = data.get("supplier_message")
        supplier_summary = self._summarise_supplier_message(supplier_message)
        thread_summary = data.get("email_thread_summary")
        if isinstance(thread_summary, dict):
            thread_summary = {
                key: thread_summary.get(key)
                for key in (
                    "total_emails",
                    "rounds",
                    "first_sent",
                    "last_sent",
                    "thread_key",
                )
            }

        line_items = self._normalise_line_items(data.get("line_items"))

        negotiation_context = {
            "supplier_name": supplier_name,
            "supplier_contact": data.get("contact_name", supplier_name),
            "category": data.get("category", "products/services"),
            "current_offer": current_offer_fmt,
            "target_price": target_price_fmt,
            "gap": gap_amount_fmt,
            "gap_percentage": f"{gap_percentage:.1f}%",
            "currency": currency,
            "round": round_no,
            "asks": data.get("asks", []),
            "lead_time_request": data.get("lead_time_request"),
            "supplier_message": supplier_message,
            "supplier_message_summary": supplier_summary,
            "negotiation_message": data.get("negotiation_message"),
            "strategy": data.get("strategy"),
            "counter_price": counter_price_fmt,
            "previous_counter_price": previous_counter_fmt,
            "walkaway_price": walkaway_price_fmt,
            "volume_units": data.get("volume_units"),
            "term_days": data.get("term_days"),
            "valid_until": data.get("valid_until"),
            "response_deadline": data.get("response_deadline") or data.get("deadline"),
            "market_floor_price": self._format_currency_value(
                data.get("market_floor_price"), currency
            )
            if data.get("market_floor_price") is not None
            else None,
            "final_offer_signaled": bool(data.get("final_offer_signaled")),
            "closing_round": bool(data.get("closing_round") or round_no >= 3),
            "email_thread_summary": thread_summary,
            "line_items": line_items,
            "play_recommendations": data.get("play_recommendations"),
        }

        objective = _determine_negotiation_objective(round_no, gap_percentage)
        tactics = _determine_recommended_tactics(round_no, gap_percentage, supplier_history)
        levers = _select_commercial_levers(round_no, data)
        strategic_elements = _generate_strategic_elements(round_no, data)
        tone_guidance = _calculate_tone_guidance(round_no, gap_percentage)

        if round_no == 1:
            cta_guidance = "Request their thoughts on bridging the gap by [date 5-7 days out]"
        elif round_no == 2:
            cta_guidance = "Ask for revised proposal incorporating suggested levers by [date 3-5 days out]"
        else:
            cta_guidance = "Request confirmation of acceptance by [date 2-3 days out] or we proceed with alternatives"

        levers_formatted = "\n".join(f"   • {lever}" for lever in levers)
        strategic_formatted = (
            "\n".join(f"   • {element}" for element in strategic_elements)
            if strategic_elements
            else "   • Focus on clear communication and professional tone"
        )

        user_prompt = NEGOTIATION_PLAYBOOK_USER.format(
            context_json=json.dumps(negotiation_context, indent=2),
            round_number=round_no,
            objective=objective,
            tactics=tactics,
            tone_guidance=tone_guidance,
            current_offer=current_offer_fmt,
            target_price=target_price_fmt,
            gap_amount=gap_amount_fmt,
            gap_percentage=f"{gap_percentage:.1f}",
            levers_list=levers_formatted,
            strategic_elements=strategic_formatted,
            cta_guidance=cta_guidance,
        )

        model_name = getattr(
            self.agent_nick.settings,
            "negotiation_email_model",
            DEFAULT_NEGOTIATION_MODEL,
        )

        try:
            response = self.call_ollama(
                model=model_name,
                messages=[
                    {"role": "system", "content": NEGOTIATION_PLAYBOOK_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                },
            )
            email_text = self._extract_ollama_message(response)
        except Exception:
            logger.exception("Failed to generate negotiation email via LLM")
            email_text = ""

        if not email_text or len(email_text.strip()) < 100:
            logger.warning("LLM generated insufficient content, using fallback")
            asks_value = data.get("asks")
            asks_list = asks_value if isinstance(asks_value, list) else [asks_value] if asks_value else []
            return self._build_negotiation_fallback(
                unique_id=data.get("unique_id", ""),
                counter_price=data.get("counter_price"),
                target_price=data.get("target_price"),
                current_offer=current_offer,
                currency=currency,
                asks=asks_list,
                lead_time_request=data.get("lead_time_request"),
                negotiation_message=data.get("negotiation_message"),
                round_no=round_no,
                line_items=data.get("line_items"),
            )

        return email_text

    def _handle_negotiation_counter(self, context: AgentContext, data: Dict[str, Any]) -> AgentOutput:
        supplier_id = data.get("supplier_id")
        supplier_name = data.get("supplier_name") or supplier_id or "Supplier"
        workflow_hint = data.get("workflow_id") or getattr(context, "workflow_id", None)

        decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
        combined_data: Dict[str, Any] = {}
        if decision:
            combined_data.update(decision)
        combined_data.update(data)

        email_text = self._draft_intelligent_negotiation_email(context, combined_data)
        subject_line, body_text = self._split_subject_and_body(email_text)
        body_content = self._sanitise_generated_body(body_text)
        body_clean = self._clean_body_text(body_content)
        body = body_clean

        subject_line, body = self._maybe_polish_negotiation_email(subject_line, body)

        supplier_summary = combined_data.get("supplier_message_summary")
        if not supplier_summary:
            supplier_summary = self._summarise_supplier_message(
                combined_data.get("supplier_message")
            )

        summary_block = None
        if supplier_summary:
            summary_block = f"**Recap of your last note**\n“{supplier_summary}”"

        if summary_block:
            body = f"{summary_block}\n\n{body}" if body else summary_block

        unique_id = self._resolve_unique_id(
            workflow_id=workflow_hint,
            supplier_id=supplier_id,
            existing=combined_data.get("unique_id"),
        )

        round_no = combined_data.get("round") or 1
        try:
            round_int = int(round_no)
        except Exception:
            round_int = 1

        if not subject_line or not subject_line.strip():
            category = combined_data.get("category") or combined_data.get("item") or "Requirement"
            subject_line = (
                f"Pricing Discussion - {category}" if round_int == 1 else f"Re: Pricing Discussion - {category}"
            )

        subject = self._clean_subject_text(subject_line, DEFAULT_NEGOTIATION_SUBJECT)
        if not subject:
            subject = DEFAULT_NEGOTIATION_SUBJECT

        contact_name = (
            combined_data.get("contact_name")
            or combined_data.get("supplier_contact")
            or combined_data.get("metadata", {}).get("supplier_contact")
            or supplier_name
        )
        greeting = f"Dear {contact_name},"
        if greeting not in body:
            body = f"{greeting}\n\n{body}" if body else greeting

        base_message = body

        threaded_body = self._inject_thread_history(
            workflow_id=workflow_hint,
            unique_id=unique_id,
            body_html=body,
        )
        body, marker_token = attach_hidden_marker(
            threaded_body,
            supplier_id=supplier_id,
            unique_id=unique_id,
        )

        _, visible_body = split_hidden_marker(body)
        plain_text = visible_body or self._clean_body_text(body)
        html_candidate = self._render_html_from_text(plain_text)
        sanitised_html = self._sanitise_generated_body(html_candidate)
        if not sanitised_html:
            sanitised_html = html_candidate

        sender_email = combined_data.get("sender") or self.agent_nick.settings.ses_default_sender
        to_recipients = self._normalise_recipients(
            combined_data.get("recipients") or combined_data.get("to")
        )
        cc_recipients = self._normalise_recipients(combined_data.get("cc"))
        recipients = self._merge_recipients(to_recipients, cc_recipients)
        receiver = to_recipients[0] if to_recipients else (recipients[0] if recipients else None)
        contact_level = 1 if receiver else 0

        session_reference = (
            combined_data.get("session_reference")
            or combined_data.get("unique_id")
            or unique_id
        )
        rfq_id = combined_data.get("rfq_id") or combined_data.get("rfq")
        closing_round = bool(
            combined_data.get("closing_round")
            or (isinstance(round_int, int) and round_int >= 3)
        )

        metadata = {
            "unique_id": unique_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "counter_price": combined_data.get("counter_price") or combined_data.get("target_price"),
            "target_price": combined_data.get("target_price"),
            "current_offer": combined_data.get("current_offer")
            or combined_data.get("current_offer_numeric"),
            "currency": combined_data.get("currency") or combined_data.get("currency_code"),
            "asks": combined_data.get("asks"),
            "lead_time_request": combined_data.get("lead_time_request"),
            "negotiation_message": combined_data.get("negotiation_message"),
            "supplier_message": combined_data.get("supplier_message"),
            "strategy": combined_data.get("strategy"),
            "round": round_int,
            "round_number": round_int,
            "intent": "NEGOTIATION_COUNTER",
            "dispatch_token": marker_token,
            "workflow_id": workflow_hint,
            "contact_name": contact_name,
            "supplier_contact": contact_name,
            "session_reference": session_reference,
            "rfq_id": rfq_id,
            "closing_round": closing_round,
            "play_recommendations": combined_data.get("play_recommendations"),
            "playbook_descriptor": combined_data.get("playbook_descriptor"),
            "playbook_examples": combined_data.get("playbook_examples"),
            "outlier_alerts": combined_data.get("outlier_alerts"),
            "validation_issues": combined_data.get("validation_issues"),
            "flags": combined_data.get("flags"),
            "rationale": combined_data.get("rationale"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        self._record_thread_message(
            workflow_id=workflow_hint,
            unique_id=unique_id,
            role="buyer",
            subject=subject,
            body_html=base_message,
            body_text=visible_body or base_message,
            metadata={
                "round": round_int,
                "strategy": combined_data.get("strategy"),
                "session_reference": session_reference,
            },
        )

        thread_headers_payload = combined_data.get("thread_headers") or combined_data.get("thread")
        resolved_thread_headers: Dict[str, Any] = {}
        if isinstance(thread_headers_payload, dict):
            resolved_thread_headers = dict(thread_headers_payload)

        headers: Dict[str, Any] = {"X-ProcWise-Unique-ID": unique_id}
        if workflow_hint:
            headers["X-ProcWise-Workflow-ID"] = workflow_hint
        if supplier_id:
            headers["X-ProcWise-Supplier-ID"] = supplier_id
        headers["X-ProcWise-Round"] = str(round_int if isinstance(round_int, int) else 0)
        headers["Subject"] = subject

        message_id = None
        if resolved_thread_headers:
            message_id = resolved_thread_headers.get("Message-ID") or resolved_thread_headers.get(
                "message_id"
            )
            in_reply_to = resolved_thread_headers.get("In-Reply-To") or resolved_thread_headers.get(
                "in_reply_to"
            )
            references = resolved_thread_headers.get("References") or resolved_thread_headers.get(
                "references"
            )

            if message_id:
                headers["Message-ID"] = message_id
            if in_reply_to:
                headers["In-Reply-To"] = in_reply_to
            if isinstance(references, list) and references:
                headers["References"] = " ".join(str(ref) for ref in references if ref)
            elif isinstance(references, str) and references.strip():
                headers["References"] = references.strip()

        if not message_id:
            message_id = f"<{uuid.uuid4()}@procwise.co.uk>"
            headers.setdefault("Message-ID", message_id)

        if subject:
            resolved_thread_headers.setdefault("Subject", subject)
        if message_id:
            resolved_thread_headers.setdefault("Message-ID", message_id)
        if workflow_hint:
            resolved_thread_headers.setdefault("X-ProcWise-Workflow-ID", workflow_hint)

        draft: Dict[str, Any] = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "contact_name": contact_name,
            "supplier_contact": contact_name,
            "unique_id": unique_id,
            "subject": subject,
            "body": body,
            "text": plain_text,
            "html": sanitised_html,
            "sent_status": False,
            "sender": sender_email,
            "recipients": recipients,
            "receiver": receiver,
            "to": receiver,
            "cc": cc_recipients,
            "contact_level": contact_level,
            "metadata": metadata,
            "headers": headers,
            "workflow_id": workflow_hint,
            "intent": "NEGOTIATION_COUNTER",
            "session_reference": session_reference,
            "rfq_id": rfq_id,
            "thread_index": round_int,
            "closing_round": closing_round,
        }

        if resolved_thread_headers:
            draft["thread_headers"] = resolved_thread_headers
        if message_id:
            draft["message_id"] = message_id

        action_id = combined_data.get("action_id") or context.input_data.get("action_id")
        if action_id:
            draft["action_id"] = action_id

        thread_index = combined_data.get("thread_index")
        if thread_index is not None:
            try:
                draft["thread_index"] = max(1, int(thread_index))
            except Exception:
                draft["thread_index"] = round_int or 1

        negotiation_extra = {
            "counter_price": combined_data.get("counter_price"),
            "target_price": combined_data.get("target_price"),
            "current_offer": combined_data.get("current_offer")
            or combined_data.get("current_offer_numeric"),
            "currency": combined_data.get("currency") or combined_data.get("currency_code"),
            "asks": combined_data.get("asks"),
            "lead_time_request": combined_data.get("lead_time_request"),
            "negotiation_message": combined_data.get("negotiation_message"),
            "supplier_message": combined_data.get("supplier_message"),
            "supplier_message_summary": supplier_summary,
        }
        draft.update({k: v for k, v in negotiation_extra.items() if v})

        self._store_draft(draft)
        self._record_learning_events(context, [draft], combined_data)

        output_data: Dict[str, Any] = {
            "drafts": [draft],
            "subject": subject,
            "body": body,
            "text": plain_text,
            "html": sanitised_html,
            "sender": sender_email,
            "intent": "NEGOTIATION_COUNTER",
            "session_reference": session_reference,
            "workflow_id": workflow_hint,
        }

        output_data["recipients"] = recipients
        output_data["cc"] = cc_recipients
        if action_id:
            output_data["action_id"] = action_id
        if message_id:
            output_data["message_id"] = message_id
        if resolved_thread_headers:
            output_data["thread_headers"] = resolved_thread_headers

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields={"drafts": [draft]},
            ),
        )


    def _render_negotiation_table(self, line_items: Optional[Any]) -> List[str]:
        if not line_items:
            return []
        if isinstance(line_items, str):
            try:
                line_items = json.loads(line_items)
            except Exception:
                return []
        if isinstance(line_items, Mapping):
            for key in ("rows", "items", "line_items"):
                candidate = line_items.get(key)
                if isinstance(candidate, Sequence):
                    line_items = candidate
                    break
        if not isinstance(line_items, Sequence):
            return []

        structured_rows = [item for item in line_items if isinstance(item, Mapping)]
        if not structured_rows:
            return []

        preferred_columns = [
            "item",
            "description",
            "current_price",
            "counter_price",
            "quantity",
            "total",
        ]
        available_columns: List[str] = []
        for column in preferred_columns:
            if any(column in row for row in structured_rows):
                available_columns.append(column)
        if not available_columns:
            sample = structured_rows[0]
            available_columns = [str(key) for key in sample.keys() if key]

        headers = [column.replace("_", " ").title() for column in available_columns]
        rows: List[List[str]] = []
        for row in structured_rows:
            formatted: List[str] = []
            for column in available_columns:
                value = row.get(column)
                if isinstance(value, (int, float)) and column.lower().endswith("price"):
                    formatted.append(f"{value:,.2f}")
                elif isinstance(value, (int, float)) and column.lower() in {"total", "quantity"}:
                    formatted.append(f"{value:,.2f}")
                else:
                    formatted.append(str(value) if value is not None else "-")
            rows.append(formatted)
        if not rows:
            return []

        widths = [
            max(len(headers[idx]), max(len(row[idx]) for row in rows))
            for idx in range(len(headers))
        ]
        header_line = " | ".join(
            headers[idx].ljust(widths[idx]) for idx in range(len(headers))
        )
        separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
        table_lines = [header_line, separator]
        for row in rows:
            table_lines.append(
                " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
            )
        return table_lines

    def _build_negotiation_fallback(
        self,
        unique_id: str,
        counter_price: Optional[Any],
        target_price: Optional[Any],
        current_offer: Optional[Any],
        currency: Optional[str],
        asks: List[Any],
        lead_time_request: Optional[str],
        negotiation_message: Optional[str],
        round_no: Optional[int] = None,
        line_items: Optional[Any] = None,
    ) -> str:
        counter_text = self._format_currency_value(counter_price, currency)
        target_text = self._format_currency_value(target_price, currency)
        offer_text = self._format_currency_value(current_offer, currency)

        try:
            round_value = int(round_no) if round_no is not None else None
        except Exception:
            round_value = None

        if round_value is None:
            try:
                round_value = int(self.agent_nick.settings.negotiation_round)
            except Exception:
                round_value = 1

        lines: List[str] = []
        lines.append("Dear Supplier Partner,")
        lines.append("")

        if round_value == 1:
            lines.append(
                "Thank you for your quotation. We've reviewed your proposal and would like to discuss the commercial terms to find a mutually beneficial arrangement."
            )
        elif round_value == 2:
            lines.append(
                "Thank you for your continued engagement. We appreciate your flexibility and would like to explore additional commercial adjustments to close the remaining gap."
            )
        else:
            lines.append(
                "Thank you for your responses throughout this process. We've now reached our final position and need to finalize terms this week."
            )
        lines.append("")

        lines.append("**Commercial Position:**")
        lines.append("")
        if offer_text and counter_text:
            lines.append(f"• Your current proposal: {offer_text}")
            lines.append(f"• Our budget positioning: {counter_text}")
            if target_text and target_text != counter_text:
                lines.append(f"• Our absolute ceiling: {target_text}")
        elif counter_text:
            lines.append(f"• Our target positioning: {counter_text}")
        lines.append("")

        if round_value in (1, 2):
            lines.append("**Commercial Levers:**")
            lines.append("")
            lines.append("To facilitate an agreement, we can offer:")
            lines.append("")
            if round_value == 1:
                lines.append("• 24-month contract commitment (from 12 months)")
                lines.append("• Net-15 payment terms with 2% early payment discount")
            else:
                lines.append("• 24-month commitment with price lock guarantee")
                lines.append("• Net-15 payment terms")
                lines.append("• Consolidated volume across product lines")
            lines.append("")

        asks_list = [str(item).strip() for item in asks if str(item).strip()] if asks else []
        if asks_list:
            lines.append("**Key Requirements:**")
            lines.append("")
            for ask in asks_list:
                formatted = ask if ask.endswith((".", "?")) else f"{ask}."
                lines.append(f"• {formatted}")
            lines.append("")

        if negotiation_message:
            lines.append("**Additional Context:**")
            lines.append("")
            lines.append(negotiation_message)
            lines.append("")

        if round_value >= 3:
            lines.append("**Final Position:**")
            lines.append("")
            lines.append(
                "This represents our maximum flexibility. We need confirmation by close of business this Friday to proceed with the award. If we cannot align on these terms, we'll need to explore alternative options."
            )
        elif round_value == 2:
            lines.append("**Next Steps:**")
            lines.append("")
            lines.append(
                "Could you please review the commercial levers above and confirm your revised position by end of next week? This will help us maintain our project timeline."
            )
        else:
            lines.append("**Next Steps:**")
            lines.append("")
            lines.append(
                "We'd appreciate your thoughts on how we might bridge this gap using the levers above. Please share your revised proposal by next Friday."
            )

        lines.append("")
        lines.append("We value this partnership and look forward to finding a mutually beneficial path forward.")
        lines.append("")
        lines.append("Best regards,")
        lines.append("Procurement Team")

        return "\n".join(lines)


    @staticmethod
    def _extract_ollama_message(response: Dict[str, Any]) -> str:
        if not isinstance(response, dict):
            return ""
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        content = response.get("response")
        if isinstance(content, str):
            return content.strip()
        return ""

    def _instruction_sources_from_policy(self, policy: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(policy, dict):
            return sources
        for field in ("policy_details", "details", "policy_desc", "description"):
            value = policy.get(field)
            if value:
                sources.append(value)
        return sources

    def _resolve_instruction_settings(self, context: AgentContext) -> Dict[str, Any]:
        sources: List[Any] = []
        for policy in context.input_data.get("policies") or []:
            sources.extend(self._instruction_sources_from_policy(policy))
        for prompt in context.input_data.get("prompts") or []:
            sources.extend(self._instruction_sources_from_prompt(prompt))
        instructions = parse_instruction_sources(sources)
        settings: Dict[str, Any] = {}

        def _pick(*keys: str) -> Optional[Any]:
            for key in keys:
                value = instructions.get(key)
                if value:
                    return value
            return None

        settings["body_template"] = _pick("body_template", "email_body_template", "email_body")
        settings["subject_template"] = _pick(
            "subject_template", "email_subject_template", "email_subject"
        )
        settings["include_rfq_table"] = instructions.get("include_rfq_table")
        settings["additional_paragraph"] = _pick(
            "additional_paragraph", "additional_note", "instructions"
        )
        settings["compliance_notice"] = _pick("compliance_notice", "compliance_clause")
        settings["interaction_type"] = _pick(
            "interaction_type",
            "email_type",
            "communication_type",
            "message_type",
            "intent",
            "purpose",
        )
        settings["tone"] = _pick("tone", "style", "voice", "message_tone")
        settings["call_to_action"] = _pick("call_to_action", "cta", "request")
        settings["context_note"] = _pick(
            "context_note",
            "background",
            "contextual_note",
            "sourcing_context",
        )
        settings["negotiation_target"] = _pick(
            "negotiation_target",
            "target_price",
            "counter_price",
        )
        settings["_instructions_present"] = bool(instructions)
        return settings

    @staticmethod
    def _coerce_text(value: Any) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        return None

    @staticmethod
    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "yes", "y", "1", "on"}:
                return True
            if text in {"false", "no", "n", "0", "off"}:
                return False
        return None

    def _render_template_string(self, template: str, args: Dict[str, Any]) -> str:
        if "{{" in template or "{%" in template:
            return Template(template).render(**args)
        try:
            return template.format(**args)
        except KeyError:
            return template

    def _render_instruction_paragraph(self, text: Optional[str]) -> str:
        cleaned = self._coerce_text(text)
        if not cleaned:
            return ""
        formatted = self._format_plain_text(cleaned)
        if not formatted:
            return ""
        return f"<p>{formatted}</p>"

    def _should_auto_compose(
        self,
        template: Optional[str],
        instruction_settings: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Return ``True`` when the agent should craft the body dynamically."""

        if instruction_settings.get("body_template"):
            return False

        template_text = (template or "").strip()
        if not template_text:
            return True

        normalised_template = re.sub(r"\s+", " ", template_text)
        normalised_default = re.sub(r"\s+", " ", self.TEXT_TEMPLATE.strip())

        if normalised_template == normalised_default:
            interaction_type = instruction_settings.get("interaction_type") or context.get(
                "interaction_type"
            )
            if interaction_type and interaction_type.lower() in {
                "negotiation",
                "follow_up",
                "reminder",
                "clarification",
                "thank_you",
            }:
                return True
            return False

        return False

    def _determine_interaction_type(
        self, context: Dict[str, Any], instruction_settings: Dict[str, Any]
    ) -> str:
        """Infer the interaction type from prompts, policies or input data."""

        candidates: List[str] = []
        context_keys = (
            "interaction_type",
            "email_type",
            "communication_type",
            "message_type",
            "intent",
            "purpose",
        )
        for key in context_keys:
            value = context.get(key)
            if value:
                candidates.append(str(value))

        instruction_keys = (
            "interaction_type",
            "email_type",
            "communication_type",
            "message_type",
            "intent",
            "purpose",
        )
        for key in instruction_keys:
            value = instruction_settings.get(key)
            if value:
                candidates.append(str(value))

        for candidate in candidates:
            normalised = self._normalise_interaction_type(candidate)
            if normalised:
                return normalised

        subject = context.get("subject") or ""
        if isinstance(subject, str) and "reminder" in subject.lower():
            return "reminder"

        return "rfq"

    def _normalise_interaction_type(self, candidate: str) -> Optional[str]:
        text = str(candidate or "").strip().lower()
        if not text:
            return None
        cleaned = re.sub(r"[^a-z]+", "_", text)
        cleaned = cleaned.strip("_")
        synonyms = {
            "followup": "follow_up",
            "follow_up": "follow_up",
            "follow_up_request": "follow_up",
            "reminder": "reminder",
            "rfq": "rfq",
            "request": "rfq",
            "quote_request": "rfq",
            "negotiation": "negotiation",
            "counter": "negotiation",
            "counter_offer": "negotiation",
            "clarification": "clarification",
            "information_request": "clarification",
            "update": "update",
            "status_update": "update",
            "award": "award",
            "award_notice": "award",
            "thankyou": "thank_you",
            "thank_you": "thank_you",
            "appreciation": "thank_you",
        }
        resolved = synonyms.get(cleaned)
        if resolved:
            return resolved
        if cleaned:
            return cleaned
        return None

    def _render_dynamic_body(
        self,
        supplier: Dict[str, Any],
        profile: Dict[str, Any],
        template_args: Dict[str, Any],
        context: Dict[str, Any],
        meta: Dict[str, Any],
        *,
        include_rfq_table: bool,
    ) -> str:
        interaction_type = meta.get("interaction_type") or "rfq"
        instructions = meta.get("instructions") or {}
        tone = meta.get("tone") or instructions.get("tone") or context.get("tone")
        call_to_action_override = meta.get("call_to_action") or instructions.get("call_to_action")
        context_note = meta.get("context_note") or instructions.get("context_note")

        sections: List[str] = []

        greeting_name = template_args.get("supplier_contact_name_html") or "Supplier"
        sections.append(f"<p>Dear {greeting_name},</p>")

        tone_prefix = self._interaction_tone_prefix(tone)
        opening = self._build_dynamic_opening(
            interaction_type, tone_prefix, template_args, context, instructions
        )
        if opening:
            sections.append(opening)
        else:
            sections.append(
                "<p>We are writing to request a formal quotation for the requirement outlined below.</p>"
            )

        scope_html = template_args.get("scope_summary_html")
        if scope_html and scope_html.strip():
            sections.append(f"<p>{scope_html}</p>")
        else:
            sections.append(
                "<p>Please review the detailed requirement and provide your quotation.</p>"
            )

        if context_note:
            context_paragraph = self._wrap_paragraph(str(context_note))
            if context_paragraph:
                sections.append(context_paragraph)

        highlights_html = self._build_dynamic_highlights(
            supplier, profile, context, instructions
        )
        if highlights_html:
            sections.append(highlights_html)

        supplier_context_html = self._build_supplier_personalisation(
            supplier,
            profile,
            template_args,
            context,
            instructions,
            interaction_type,
        )
        if supplier_context_html and isinstance(meta, dict):
            internal_context = meta.setdefault("internal_context", {})
            if isinstance(internal_context, dict):
                internal_context["supplier_context_html"] = supplier_context_html
                internal_context["supplier_context_text"] = self._html_to_plain_text(
                    supplier_context_html
                )

        negotiation_context = self._build_negotiation_summary(
            interaction_type,
            supplier,
            template_args,
            context,
            instructions,
        )
        if negotiation_context:
            sections.append(negotiation_context)

        cta = self._build_dynamic_call_to_action(
            interaction_type,
            template_args,
            context,
            instructions,
            call_to_action_override,
        )
        if cta:
            sections.append(cta)
        else:
            deadline = template_args.get("deadline_html") or template_args.get("deadline")
            if deadline:
                sections.append(f"<p>Please return your quotation by {deadline}.</p>")
            else:
                sections.append(
                    "<p>Please return your quotation at your earliest convenience.</p>"
                )

        if include_rfq_table:
            rfq_table = template_args.get("rfq_table_html")
            if rfq_table and rfq_table.strip():
                sections.append(rfq_table)
            else:
                sections.append(RFQ_TABLE_HEADER)

        closing = self._build_dynamic_closing(
            interaction_type,
            template_args,
            tone,
            instructions,
        )
        if closing:
            sections.append(closing)
        else:
            sections.append("<p>We appreciate your timely response.</p>")

        signature = self._build_signature_block(template_args)
        if signature:
            sections.append(signature)

        return "\n".join(section for section in sections if section)

    def _interaction_tone_prefix(self, tone: Optional[str]) -> Optional[str]:
        if not tone:
            return None
        lowered = str(tone).strip().lower()
        if "friendly" in lowered:
            return "I hope you are well."
        if "urgent" in lowered:
            return "This is an urgent request requiring your prompt attention."
        if "appreciative" in lowered or "positive" in lowered:
            return "Thank you for your continued partnership."
        if "formal" in lowered:
            return "Please note the formal notice below."
        return None

    def _build_dynamic_opening(
        self,
        interaction_type: str,
        tone_prefix: Optional[str],
        template_args: Dict[str, Any],
        context: Dict[str, Any],
        instructions: Dict[str, Any],
    ) -> str:
        phrases: List[str] = []
        if tone_prefix:
            phrases.append(tone_prefix)

        scope_hint = self._clean_html_snippet(template_args.get("scope_summary_html"))
        if interaction_type == "follow_up":
            phrases.append("I am following up on our earlier quotation request.")
        elif interaction_type == "reminder":
            phrases.append("This is a reminder regarding the outstanding quotation.")
        elif interaction_type == "negotiation":
            phrases.append("Thank you for sharing your quotation details.")
        elif interaction_type == "clarification":
            phrases.append("We require a few clarifications on the proposal below.")
        elif interaction_type == "award":
            phrases.append("We are pleased to confirm the outcome of the sourcing event.")
        elif interaction_type == "thank_you":
            phrases.append("Thank you for your continued collaboration.")
        else:
            phrases.append("We are initiating a sourcing request for your review.")

        if scope_hint and interaction_type in {"rfq", "follow_up", "reminder", "clarification"}:
            phrases.append(f"The requirement focuses on {scope_hint}.")

        return self._wrap_paragraph(" ".join(phrases))

    def _build_dynamic_highlights(
        self,
        supplier: Dict[str, Any],
        profile: Dict[str, Any],
        context: Dict[str, Any],
        instructions: Dict[str, Any],
    ) -> Optional[str]:
        highlights = self._collect_highlights(supplier, profile, context, instructions)
        if not highlights:
            return None
        items = "".join(f"<li>{escape(point)}</li>" for point in highlights[:5])
        return f"<ul>{items}</ul>"

    def _build_supplier_personalisation(
        self,
        supplier: Dict[str, Any],
        profile: Dict[str, Any],
        template_args: Dict[str, Any],
        context: Dict[str, Any],
        instructions: Dict[str, Any],
        interaction_type: str,
    ) -> Optional[str]:
        statements: List[str] = []

        currency = (
            context.get("currency")
            or instructions.get("currency")
            or template_args.get("currency")
        )

        def _append(text: Optional[str]) -> None:
            if text:
                cleaned = str(text).strip()
                if cleaned:
                    statements.append(cleaned)

        avg_price = None
        for price_key in ("avg_unit_price", "current_price", "price"):
            avg_price = self._as_float(supplier.get(price_key))
            if avg_price is not None:
                break
        if avg_price is not None:
            price_text = self._format_currency_value(avg_price, currency)
            _append(f"Recent average submitted pricing on record is {price_text}.")

        spend_value = self._as_float(supplier.get("total_spend"))
        if spend_value and spend_value > 0:
            spend_text = self._format_currency_value(spend_value, currency)
            _append(f"Year-to-date spend with your organisation totals {spend_text}.")

        po_count = self._as_float(supplier.get("po_count"))
        invoice_count = self._as_float(supplier.get("invoice_count"))
        if po_count and po_count > 0:
            volume_note = f"{int(po_count)} purchase orders"
            if invoice_count and invoice_count > 0:
                volume_note += f" across {int(invoice_count)} invoices"
            _append(f"Our records show {volume_note} in scope for this category.")

        lead_time = self._as_float(
            supplier.get("lead_time_days")
            or supplier.get("avg_lead_time_days")
            or profile.get("lead_time_days")
        )
        if lead_time and lead_time > 0:
            _append(f"Typical delivery performance has averaged {self._format_lead_time(lead_time)}.")

        relationship_summary = supplier.get("relationship_summary")
        if isinstance(relationship_summary, str) and relationship_summary.strip():
            _append(relationship_summary.strip())

        rel_statements = supplier.get("relationship_statements")
        if isinstance(rel_statements, list):
            for statement in rel_statements[:2]:
                if isinstance(statement, str) and statement.strip():
                    _append(statement.strip())

        flow_coverage = self._as_float(supplier.get("flow_coverage"))
        if flow_coverage is not None and flow_coverage > 0:
            _append(
                f"Workflow coverage against current processes is {self._format_percentage(flow_coverage)}."
            )

        relationship_cov = self._as_float(supplier.get("relationship_coverage"))
        if relationship_cov is not None and relationship_cov > 0:
            _append(
                f"Relationship coverage signals {self._format_percentage(relationship_cov)} engagement across our teams."
            )

        if interaction_type != "negotiation":
            price_score = self._as_float(supplier.get("price_score"))
            delivery_score = self._as_float(supplier.get("delivery_score"))
            risk_score = self._as_float(supplier.get("risk_score"))
            final_score = self._as_float(supplier.get("final_score"))
            score_notes: List[str] = []
            if final_score is not None:
                score_notes.append(f"composite score {final_score:.2f}")
            if price_score is not None:
                score_notes.append(f"price {price_score:.2f}")
            if delivery_score is not None:
                score_notes.append(f"delivery {delivery_score:.2f}")
            if risk_score is not None:
                score_notes.append(f"risk {risk_score:.2f}")
            if score_notes:
                _append("Performance scores on the latest evaluation: " + ", ".join(score_notes) + ".")

        if not statements:
            return None

        unique: List[str] = []
        seen: Set[str] = set()
        for sentence in statements:
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                unique.append(sentence)

        if not unique:
            return None

        if len(unique) == 1:
            return self._wrap_paragraph(unique[0])

        items = "".join(f"<li>{escape(text)}</li>" for text in unique[:5])
        return f"<p><strong>Supplier context</strong></p><ul>{items}</ul>"

    def _build_negotiation_summary(
        self,
        interaction_type: str,
        supplier: Dict[str, Any],
        template_args: Dict[str, Any],
        context: Dict[str, Any],
        instructions: Dict[str, Any],
    ) -> Optional[str]:
        if interaction_type != "negotiation":
            return None

        currency = (
            context.get("currency")
            or instructions.get("currency")
            or template_args.get("currency")
        )
        lines: List[str] = []

        current_offer = (
            context.get("current_offer")
            or supplier.get("current_offer")
            or supplier.get("avg_unit_price")
        )
        if current_offer is not None:
            lines.append(
                f"Your latest submitted position stands at {self._format_currency_value(current_offer, currency)}."
            )

        target_price = (
            instructions.get("negotiation_target")
            or context.get("target_price")
            or supplier.get("target_price")
        )
        if target_price is not None:
            lines.append(
                f"Our target for this round is {self._format_currency_value(target_price, currency)}."
            )

        counter = context.get("counter_price") or supplier.get("counter_price")
        if counter is not None and counter != target_price:
            lines.append(
                f"We are positioned to proceed at {self._format_currency_value(counter, currency)} subject to alignment on scope."
            )

        lead_time_request = (
            context.get("lead_time_request")
            or supplier.get("lead_time")
            or supplier.get("lead_time_days")
        )
        if lead_time_request:
            if isinstance(lead_time_request, (int, float)):
                lead_note = self._format_lead_time(float(lead_time_request))
            else:
                lead_note = str(lead_time_request)
            lines.append(f"Requested delivery commitment: {lead_note}.")

        if not lines:
            return None

        combined = " ".join(lines)
        return self._wrap_paragraph(combined)

    def _collect_highlights(
        self,
        supplier: Dict[str, Any],
        profile: Dict[str, Any],
        context: Dict[str, Any],
        instructions: Dict[str, Any],
    ) -> List[str]:
        highlights: List[str] = []

        for key in ("key_points", "highlights", "notes"):
            values = context.get(key)
            if isinstance(values, (list, tuple, set)):
                for value in values:
                    if isinstance(value, str) and value.strip():
                        highlights.append(value.strip())
            elif isinstance(values, str) and values.strip():
                highlights.append(values.strip())

        instruction_points = instructions.get("key_points") or instructions.get("highlights")
        if isinstance(instruction_points, (list, tuple, set)):
            for value in instruction_points:
                if isinstance(value, str) and value.strip():
                    highlights.append(value.strip())
        elif isinstance(instruction_points, str) and instruction_points.strip():
            highlights.append(instruction_points.strip())

        supplier_strengths = supplier.get("strengths") or supplier.get("differentiators")
        if isinstance(supplier_strengths, (list, tuple, set)):
            for value in supplier_strengths:
                if isinstance(value, str) and value.strip():
                    highlights.append(value.strip())
        elif isinstance(supplier_strengths, str) and supplier_strengths.strip():
            highlights.append(supplier_strengths.strip())

        profile_highlights = profile.get("capabilities") if isinstance(profile, dict) else None
        if isinstance(profile_highlights, (list, tuple, set)):
            for value in profile_highlights:
                if isinstance(value, str) and value.strip():
                    highlights.append(value.strip())

        cleaned: List[str] = []
        seen: set[str] = set()
        for value in highlights:
            token = value.strip()
            key = token.lower()
            if token and key not in seen:
                seen.add(key)
                cleaned.append(token)
        return cleaned

    def _build_dynamic_call_to_action(
        self,
        interaction_type: str,
        template_args: Dict[str, Any],
        context: Dict[str, Any],
        instructions: Dict[str, Any],
        override: Optional[str],
    ) -> str:
        """Build call to action without RFQ ID references."""
        if override:
            return self._wrap_paragraph(str(override))

        deadline = template_args.get("deadline_html") or template_args.get("deadline")
        deadline_text = self._clean_html_snippet(deadline) if deadline else None
        base: Optional[str] = None

        if interaction_type == "negotiation":
            target = instructions.get("negotiation_target")
            if target is None:
                target = context.get("target_price")
            target_text = self._format_currency_value(target, context.get("currency"))
            current_offer = self._format_currency_value(
                context.get("current_offer"), context.get("currency")
            )
            base = (
                "Could you review the pricing and confirm if a revised proposal "
                f"closer to {target_text} is achievable?"
            )
            if current_offer:
                base += (
                    f" Your current offer of {current_offer} is appreciated and forms the basis of this discussion."
                )
        elif interaction_type == "follow_up":
            base = "We would appreciate an update on your quotation at your earliest convenience."
        elif interaction_type == "reminder":
            if deadline_text:
                base = (
                    f"This is a friendly reminder that the quotation is due by {deadline_text}."
                )
            else:
                base = "This is a friendly reminder to share your quotation."
        elif interaction_type == "clarification":
            base = "Please provide clarification on the highlighted points so that we can complete the evaluation."
        elif interaction_type == "award":
            base = "Please confirm acceptance of the award and advise on next steps for mobilisation."
        elif interaction_type == "thank_you":
            base = "Please keep us informed about any support you require for the next phase."
        else:
            if deadline_text:
                base = (
                    "Please review the requirement and return your quotation using the table below "
                    f"by {deadline_text}."
                )
            else:
                base = "Please review the requirement and return your quotation using the table below."

        return self._wrap_paragraph(f"{base}")

    def _build_dynamic_closing(
        self,
        interaction_type: str,
        template_args: Dict[str, Any],
        tone: Optional[str],
        instructions: Dict[str, Any],
    ) -> str:
        closing_override = instructions.get("closing_note") or instructions.get("closing")
        if closing_override:
            return self._wrap_paragraph(str(closing_override))

        if interaction_type == "negotiation":
            text = "We appreciate your consideration and look forward to identifying a mutually beneficial outcome."
        elif interaction_type == "award":
            text = "Congratulations once again, and thank you for supporting this programme."
        elif interaction_type == "thank_you":
            text = "Thank you for your continued collaboration."
        else:
            text = "We appreciate your timely response and support." 

        if tone and "urgent" in str(tone).lower():
            text = "Your prompt attention to this matter is greatly appreciated."

        return self._wrap_paragraph(text)

    @staticmethod
    def _format_percentage(value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return "0%"
        if number < 0:
            number = 0.0
        if number <= 1:
            number *= 100
        return f"{number:.0f}%"

    @staticmethod
    def _format_lead_time(days: float) -> str:
        try:
            day_value = float(days)
        except Exception:
            return str(days)
        rounded_days = int(round(day_value))
        if rounded_days <= 1:
            return "1 day"
        weeks = day_value / 7.0
        if weeks >= 1.5:
            return f"{rounded_days} days (~{weeks:.1f} weeks)"
        return f"{rounded_days} days"

    def _build_signature_block(self, template_args: Dict[str, Any]) -> str:
        name = template_args.get("your_name_html") or template_args.get("your_name")
        title = template_args.get("your_title_html") or template_args.get("your_title")
        company = template_args.get("your_company_html") or template_args.get("your_company")
        lines = ["Kind regards,"]
        if name:
            lines.append(str(name))
        if title:
            lines.append(str(title))
        if company:
            lines.append(str(company))
        body = "<br>".join(lines)
        return f"<p>{body}</p>"

    def _wrap_paragraph(self, text: Optional[str]) -> str:
        if not text:
            return ""
        stripped = str(text).strip()
        if not stripped:
            return ""
        if "<" in stripped and ">" in stripped:
            return stripped
        return f"<p>{escape(stripped)}</p>"

    def _clean_html_snippet(self, snippet: Optional[str]) -> Optional[str]:
        if snippet is None:
            return None
        text = re.sub(r"<[^>]+>", "", str(snippet))
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned or None

    def _format_currency_value(self, value: Any, currency: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return str(value)
        code = (currency or "GBP").upper()
        symbol = "£" if code == "GBP" else "$" if code == "USD" else ""
        formatted = f"{amount:,.2f}"
        if symbol:
            return f"{symbol}{formatted}"
        return f"{formatted} {code}"

    @staticmethod
    def _normalise_rfq_identifier(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        candidate = re.sub(r"[^A-Za-z0-9_-]+", "", text).upper()
        if not candidate:
            return None
        if not candidate.startswith("RFQ"):
            return None
        return candidate

    @staticmethod
    def _split_subject_and_body(text: str) -> tuple[Optional[str], str]:
        if not isinstance(text, str):
            return None, ""
        subject: Optional[str] = None
        body_lines: List[str] = []
        for line in text.splitlines():
            if subject is None and line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()
        return subject, body

    @staticmethod
    def _merge_recipients(to_list: List[str], cc_list: List[str]) -> List[str]:
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

    def _render_html_from_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        lines = text.splitlines()
        html_parts: List[str] = []
        bullets: List[str] = []

        def render_inline(markup: str) -> str:
            if not markup:
                return ""
            result: List[str] = []
            last_index = 0
            for match in re.finditer(r"\*\*(.+?)\*\*", markup):
                start, end = match.span()
                if start > last_index:
                    result.append(escape(markup[last_index:start]))
                result.append(f"<strong>{escape(match.group(1))}</strong>")
                last_index = end
            if last_index < len(markup):
                result.append(escape(markup[last_index:]))
            return "".join(result)

        def flush() -> None:
            if bullets:
                items = "".join(f"<li>{render_inline(item)}</li>" for item in bullets)
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
            html_parts.append(f"<p>{render_inline(stripped)}</p>")
        flush()
        if not html_parts:
            return ""
        return "".join(html_parts)

    @staticmethod
    def _html_to_plain_text(html: str) -> str:
        if not html:
            return ""
        text = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        text = re.sub(
            r"<li>(.*?)</li>",
            lambda m: "- " + re.sub(r"<[^>]+>", "", m.group(1)).strip(),
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _normalise_subject_line(subject: str, rfq_id: Optional[str]) -> str:
        """Remove visible RFQ identifiers from subject lines."""

        if not isinstance(subject, str):
            return ""

        trimmed = subject.strip()
        if not trimmed:
            return ""

        if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
            trimmed = trimmed[1:-1].strip()

        cleaned = EmailDraftingAgent._strip_rfq_identifier_tokens(trimmed)
        cleaned = cleaned.strip("-–: ")
        if cleaned:
            return cleaned
        return trimmed

    @staticmethod
    def _coerce_action_id(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        try:
            candidate = str(value).strip()
        except Exception:
            return None
        return candidate or None

    def _action_belongs_to_email_agent(self, action_id: str) -> bool:
        if not action_id:
            return False
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return False
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT agent_type FROM proc.action WHERE action_id = %s",
                        (action_id,),
                    )
                    row = cursor.fetchone()
            if not row:
                return False
            agent_type = row[0]
            if isinstance(agent_type, (bytes, bytearray)):
                agent_type = agent_type.decode(errors="ignore")
            agent_label = str(agent_type).strip().lower()
            return agent_label == "emaildraftingagent".lower()
        except Exception:  # pragma: no cover - defensive lookup
            logger.debug(
                "Failed to validate ownership for action %s", action_id, exc_info=True
            )
            return False

    def run(self, context: AgentContext) -> AgentOutput:
        """Draft RFQ emails for each ranked supplier without sending."""
        logger.info("EmailDraftingAgent starting")
        self._ensure_prompt_template(context)
        data = dict(context.input_data)
        prev = data.get("previous_agent_output")
        if isinstance(prev, str):
            try:
                prev = json.loads(prev)
            except Exception:
                prev = {}
        if isinstance(prev, dict):
            data = {**prev, **data}

        invocation_mode = self._detect_invocation_mode(context, data)
        if invocation_mode == "negotiation":
            negotiation_payload: Optional[Dict[str, Any]] = None
            for key in ("negotiation_context", "negotiation"):
                candidate = data.get(key)
                if isinstance(candidate, dict):
                    negotiation_payload = candidate
                    break
            if negotiation_payload is None and isinstance(context.input_data, dict):
                candidate = context.input_data.get("negotiation_context")
                if isinstance(candidate, dict):
                    negotiation_payload = candidate
            if negotiation_payload:
                data.setdefault("negotiation", negotiation_payload)
        else:
            data.pop("negotiation", None)
            data.pop("negotiation_context", None)

        thread_state_input = data.get("email_thread_state")
        thread_state_root = self._normalise_thread_state(thread_state_input)

        decision_payload = data.get("decision")
        thread_headers_payload = data.get("thread_headers")
        if isinstance(thread_headers_payload, dict):
            if isinstance(decision_payload, dict):
                decision_payload = dict(decision_payload)
                decision_payload.setdefault("thread", thread_headers_payload)
                data["decision"] = decision_payload
            else:
                decision_payload = {"thread": thread_headers_payload}
                data["decision"] = decision_payload
        if isinstance(decision_payload, dict) and decision_payload:
            draft = self.from_decision(decision_payload)
            draft = self._apply_workflow_context(
                draft, context, source_payload=decision_payload
            )
            self._store_draft(draft)
            source_snapshot = {**decision_payload, "intent": "NEGOTIATION_COUNTER"}
            self._record_learning_events(context, [draft], source_snapshot)
            output_data: Dict[str, Any] = {
                "drafts": [draft],
                "subject": draft.get("subject"),
                "body": draft.get("body"),
                "sender": draft.get("sender"),
                "intent": draft.get("metadata", {}).get("intent", "NEGOTIATION_COUNTER"),
            }
            recipients = draft.get("recipients")
            if recipients:
                output_data["recipients"] = recipients
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=output_data,
                    pass_fields={"drafts": [draft]},
                ),
            )

        prompt_text: Optional[str] = None
        prompt_candidate = data.get("prompt")
        if isinstance(prompt_candidate, str) and prompt_candidate.strip():
            prompt_text = prompt_candidate
        else:
            decision_log = data.get("decision_log")
            if isinstance(decision_log, str) and decision_log.strip():
                prompt_text = decision_log
        if prompt_text:
            draft = self.from_prompt(prompt_text, context=data)
            draft = self._apply_workflow_context(draft, context, source_payload=data)
            self._store_draft(draft)
            self._record_learning_events(context, [draft], data)
            output_data = {
                "drafts": [draft],
                "subject": draft.get("subject"),
                "body": draft.get("body"),
                "sender": draft.get("sender"),
                "intent": draft.get("metadata", {}).get("intent"),
            }
            recipients = draft.get("recipients")
            if recipients:
                output_data["recipients"] = recipients
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=output_data,
                    pass_fields={"drafts": [draft]},
                ),
            )

        intent_value = data.get("intent")
        if isinstance(intent_value, str) and intent_value.upper() == "NEGOTIATION_COUNTER":
            return self._handle_negotiation_counter(context, data)

        ranking_input = data.get("ranking")
        if isinstance(ranking_input, list):
            ranking = [entry for entry in ranking_input if isinstance(entry, dict)]
        elif isinstance(ranking_input, (tuple, set)):
            ranking = [
                entry for entry in ranking_input if isinstance(entry, dict)
            ]
        else:
            ranking = []

        if not ranking:
            supplier_info = data.get("supplier_information")
            if isinstance(supplier_info, list):
                ranking = [
                    entry for entry in supplier_info if isinstance(entry, dict)
                ]
        findings = data.get("findings", [])
        supplier_profiles = (
            data.get("supplier_profiles") if isinstance(data.get("supplier_profiles"), dict) else {}
        )

        validated_actions: Dict[str, bool] = {}

        def _is_email_action(action_id: str) -> bool:
            if action_id in validated_actions:
                return validated_actions[action_id]
            belongs = self._action_belongs_to_email_agent(action_id)
            validated_actions[action_id] = belongs
            return belongs

        def _resolve_action_id(source: Optional[Dict[str, Any]]) -> Optional[str]:
            if not isinstance(source, dict):
                return None
            for key in ("email_action_id", "draft_action_id", "action_id"):
                candidate = self._coerce_action_id(source.get(key))
                if not candidate:
                    continue
                if key == "action_id":
                    if _is_email_action(candidate):
                        return candidate
                    continue
                return candidate
            return None

        default_action_id = _resolve_action_id(data)
        drafts = []

        manual_recipients = self._normalise_recipients(data.get("recipients"))
        manual_sender = data.get("sender") or self.agent_nick.settings.ses_default_sender
        manual_body_input = data.get("body") if isinstance(data.get("body"), str) else None
        manual_subject_input = data.get("subject") if isinstance(data.get("subject"), str) else None
        manual_has_body = bool(manual_body_input and manual_body_input.strip())
        manual_subject_rendered: Optional[str] = None
        manual_body_rendered: Optional[str] = None
        manual_unique_id: Optional[str] = None
        manual_sent_flag: bool = data.get("sent_status", True)

        instruction_settings = self._resolve_instruction_settings(context)
        negotiation_context = data.get("negotiation")
        negotiation_message = None
        if isinstance(negotiation_context, dict):
            negotiation_message = self._coerce_text(
                negotiation_context.get("negotiation_message")
                or negotiation_context.get("message")
            )
            if not negotiation_message:
                transcript = negotiation_context.get("transcript")
                if isinstance(transcript, list) and transcript:
                    negotiation_message = self._coerce_text(transcript[-1])
        if negotiation_message is None:
            negotiation_message = self._coerce_text(
                data.get("negotiation_message") or data.get("message")
            )
        negotiation_section_html = ""
        if negotiation_message:
            negotiation_section_html = self._render_instruction_paragraph(
                negotiation_message
            )
            instruction_settings.setdefault("interaction_type", "negotiation")

        body_template_override = self._coerce_text(
            instruction_settings.get("body_template")
        )
        subject_template_override = self._coerce_text(
            instruction_settings.get("subject_template")
        )
        include_rfq_setting = instruction_settings.get("include_rfq_table")
        include_rfq_table = self._coerce_bool(include_rfq_setting)
        if include_rfq_table is None:
            include_rfq_table = True
        additional_paragraph = self._coerce_text(
            instruction_settings.get("additional_paragraph")
        )
        compliance_notice = self._coerce_text(
            instruction_settings.get("compliance_notice")
        )
        instructions_present = instruction_settings.get("_instructions_present", False)

        additional_section_html = self._render_instruction_paragraph(additional_paragraph)
        compliance_section_html = ""
        if compliance_notice:
            compliance_text = self._format_plain_text(compliance_notice)
            if compliance_text:
                compliance_section_html = (
                    f"<p><strong>Compliance:</strong> {compliance_text}</p>"
                )
        instruction_suffix = "".join(
            section
            for section in (additional_section_html, compliance_section_html)
            if section
        )

        manual_template_candidate = (
            manual_body_input if isinstance(manual_body_input, str) else None
        )
        has_template_candidate = bool(
            manual_template_candidate and manual_template_candidate.strip()
        )
        explicit_body_template = self._coerce_text(data.get("body_template"))
        base_body_template = (
            body_template_override
            or (
                manual_template_candidate
                if has_template_candidate and not manual_recipients
                else None
            )
            or explicit_body_template
            or self.TEXT_TEMPLATE
        )
        subject_template_source = (
            subject_template_override
            or self._coerce_text(data.get("subject_template"))
        )

        workflow_id = getattr(context, "workflow_id", None)
        if not workflow_id:
            workflow_id = f"WF-{uuid.uuid4().hex[:16]}"

        ranking_enumerated = list(enumerate(ranking))
        supplier_results: List[SupplierDraftResult] = []
        draft_supplier_map: Dict[str, Optional[str]] = {}
        if ranking_enumerated:
            max_workers = self._determine_parallel_workers(len(ranking_enumerated))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._render_supplier_draft,
                        supplier=supplier,
                        index=index,
                        workflow_id=workflow_id,
                        data=data,
                        supplier_profiles=supplier_profiles,
                        base_body_template=base_body_template,
                        subject_template_source=subject_template_source,
                        include_rfq_table=include_rfq_table,
                        additional_section_html=additional_section_html,
                        compliance_section_html=compliance_section_html,
                        instruction_suffix=instruction_suffix,
                        negotiation_section_html=negotiation_section_html,
                        negotiation_message=negotiation_message,
                        instruction_settings=instruction_settings,
                        default_action_id=default_action_id,
                        negotiation_message_plain=negotiation_message,
                        base_thread_state=thread_state_root,
                        resolve_action_id=_resolve_action_id,
                        sender_email=self.agent_nick.settings.ses_default_sender,
                    )
                    for index, supplier in ranking_enumerated
                ]

                for future in as_completed(futures):
                    try:
                        supplier_results.append(future.result())
                    except Exception:
                        logger.exception("EmailDraftingAgent failed to build supplier draft")

            supplier_results.sort(key=lambda result: result.index)

            aggregated_suppliers_state = dict(thread_state_root.get("suppliers", {}))

            for result in supplier_results:
                draft = self._apply_workflow_context(
                    result.draft, context, source_payload=data
                )
<<<<<<< HEAD
                drafts.append(draft)
                draft_supplier_map[draft.get("draft_id") or draft.get("unique_id") or ""] = (
                    result.supplier_id
=======
            else:
                rendered = self._render_template_string(body_template, template_args)

            comment, message = self._split_existing_comment(rendered)
            message_content = message if comment else rendered
            appended_sections: List[str] = []
            if additional_section_html and additional_section_html not in message_content:
                appended_sections.append(additional_section_html)
            if compliance_section_html and compliance_section_html not in message_content:
                appended_sections.append(compliance_section_html)
            if appended_sections:
                message_content = f"{message_content}{''.join(appended_sections)}"
            elif instruction_suffix and instruction_suffix not in message_content:
                message_content = f"{message_content}{instruction_suffix}"

            content = self._sanitise_generated_body(message_content)
            if negotiation_section_html and negotiation_section_html not in content:
                content = f"{content}{negotiation_section_html}"
            body_clean = self._clean_body_text(content)
            threaded_body = self._inject_thread_history(
                workflow_id=workflow_id,
                unique_id=unique_id,
                body_html=body_clean,
            )
            body, marker_token = attach_hidden_marker(
                threaded_body,
                supplier_id=supplier_id,
                unique_id=unique_id,
            )
            _, visible_body = split_hidden_marker(body)
            if subject_template_source:
                subject_args = dict(template_args)
                subject_args.setdefault("unique_id", unique_id)
                rendered_subject = self._render_template_string(
                    subject_template_source, subject_args
>>>>>>> f6b29da (updated changes)
                )
                supplier_key = self._thread_supplier_key(result.supplier_id)
                aggregated_suppliers_state[supplier_key] = result.supplier_thread_state
                self._store_draft(draft)
                logger.debug(
                    "EmailDraftingAgent created draft %s for supplier %s",
                    draft.get("unique_id"),
                    result.supplier_id,
                )

<<<<<<< HEAD
            thread_state_root["suppliers"] = aggregated_suppliers_state
            thread_state_root["updated_at"] = datetime.now(timezone.utc).isoformat()
=======
            draft_action_id = _resolve_action_id(supplier) or default_action_id

            receiver = self._resolve_receiver(supplier, profile)
            recipients: List[str] = []
            if receiver:
                recipients = self._normalise_recipients([receiver])
            if recipients:
                receiver = recipients[0]
            contact_level = 1 if recipients else 0

            internal_context = {}
            if isinstance(dynamic_meta.get("internal_context"), dict):
                internal_context = {
                    key: value
                    for key, value in dynamic_meta["internal_context"].items()
                    if value
                }
            if not internal_context:
                fallback_context = self._build_supplier_personalisation(
                    supplier,
                    profile,
                    template_args,
                    data,
                    instruction_settings,
                    interaction_type,
                )
                if fallback_context:
                    internal_context = {
                        "supplier_context_html": fallback_context,
                        "supplier_context_text": self._html_to_plain_text(
                            fallback_context
                        ),
                    }

            draft = {
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "subject": subject,
                "body": body,
                "sent_status": False,
                "sender": self.agent_nick.settings.ses_default_sender,
                "action_id": draft_action_id,
                "supplier_profile": profile,
                "receiver": receiver,
                "contact_level": contact_level,
                "recipients": recipients,
                "unique_id": unique_id,
                "workflow_id": workflow_id,
            }
            if draft_action_id:
                draft["action_id"] = draft_action_id
            draft.setdefault("thread_index", 1)
            metadata: Dict[str, Any] = {
                "unique_id": unique_id,
                "interaction_type": interaction_type,
                "workflow_id": workflow_id,
            }
            if supplier_id is not None:
                metadata["supplier_id"] = supplier_id
            if supplier_name:
                metadata["supplier_name"] = supplier_name
            if internal_context:
                metadata["internal_context"] = internal_context
            if marker_token:
                metadata["dispatch_token"] = marker_token
            draft["metadata"] = metadata
            self._record_thread_message(
                workflow_id=workflow_id,
                unique_id=unique_id,
                role="buyer",
                subject=subject,
                body_html=body_clean,
                body_text=visible_body or body_clean,
                metadata={
                    "interaction_type": interaction_type,
                    "supplier_id": supplier_id,
                    "supplier_name": supplier_name,
                },
            )
            draft = self._apply_workflow_context(draft, context, source_payload=data)
            drafts.append(draft)
            self._store_draft(draft)
            logger.debug(
                "EmailDraftingAgent created draft %s for supplier %s", unique_id, supplier_id
            )
>>>>>>> f6b29da (updated changes)

        if manual_recipients and manual_has_body:
            manual_comment, manual_message = self._split_existing_comment(manual_body_input)
            manual_body_content = (
                self._sanitise_generated_body(manual_message)
                if manual_comment
                else self._sanitise_generated_body(manual_body_input)
            )
            extracted_manual_id = self._extract_rfq_id(manual_comment)
            manual_unique_id = extracted_manual_id or self._generate_unique_identifier(
                workflow_id, None
            )
            manual_body_clean = self._clean_body_text(manual_body_content)
            manual_threaded_body = self._inject_thread_history(
                workflow_id=workflow_id,
                unique_id=manual_unique_id,
                body_html=manual_body_clean,
            )
            manual_body_rendered, manual_marker_token = attach_hidden_marker(
                manual_threaded_body,
                supplier_id=None,
                unique_id=manual_unique_id,
            )
            _, manual_visible_body = split_hidden_marker(manual_body_rendered)
            manual_subject_rendered = self._clean_subject_text(
                manual_subject_input,
                DEFAULT_RFQ_SUBJECT,
            )

            manual_metadata: Dict[str, Any] = {
                "unique_id": manual_unique_id,
                "workflow_id": workflow_id,
            }
            if manual_marker_token:
                manual_metadata["dispatch_token"] = manual_marker_token

            self._record_thread_message(
                workflow_id=workflow_id,
                unique_id=manual_unique_id,
                role="buyer",
                subject=manual_subject_rendered,
                body_html=manual_body_clean,
                body_text=manual_visible_body or manual_body_clean,
                metadata={"manual": True},
            )

            manual_draft = {
                "supplier_id": None,
                "supplier_name": ", ".join(manual_recipients),
                "subject": manual_subject_rendered,
                "body": manual_body_rendered,
                "sent_status": bool(manual_sent_flag),
                "sender": manual_sender,
                "action_id": default_action_id,
                "supplier_profile": {},
                "recipients": manual_recipients,
                "receiver": manual_recipients[0] if manual_recipients else None,
                "contact_level": 1 if manual_recipients else 0,
                "metadata": manual_metadata,
                "unique_id": manual_unique_id,
                "workflow_id": workflow_id,
            }
            if default_action_id:
                manual_draft["action_id"] = default_action_id
            manual_draft.setdefault("thread_index", 1)
            manual_draft = self._apply_workflow_context(
                manual_draft, context, source_payload=data
            )
            drafts.append(manual_draft)
            self._store_draft(manual_draft)
            if manual_unique_id:
                draft_supplier_map.setdefault(manual_unique_id, None)

        logger.info("EmailDraftingAgent generated %d drafts", len(drafts))
        output_data: Dict[str, Any] = {"drafts": drafts, "prompt": self.prompt_template}
        if thread_state_root:
            output_data["email_thread_state"] = thread_state_root
        if draft_supplier_map:
            output_data["draft_supplier_map"] = {
                key: value for key, value in draft_supplier_map.items() if key
            }
        if manual_subject_rendered is not None:
            output_data["subject"] = manual_subject_rendered
        if manual_body_rendered is not None:
            output_data["body"] = manual_body_rendered
        if manual_recipients:
            output_data["recipients"] = manual_recipients
        if manual_sender:
            output_data["sender"] = manual_sender
        if data.get("attachments") is not None:
            output_data["attachments"] = data.get("attachments")
        if drafts and "body" not in output_data:
            output_data["body"] = drafts[0].get("body")
        if drafts and "subject" not in output_data:
            output_data["subject"] = drafts[0].get("subject")
        if drafts and "sender" not in output_data:
            output_data["sender"] = drafts[0].get("sender")
        if default_action_id:
            output_data["action_id"] = default_action_id

        pass_fields: Dict[str, Any] = {"drafts": drafts}
        if default_action_id:
            pass_fields["action_id"] = default_action_id
        if manual_subject_rendered is not None:
            pass_fields["subject"] = manual_subject_rendered
        if manual_body_rendered is not None:
            pass_fields["body"] = manual_body_rendered
        if manual_recipients:
            pass_fields["recipients"] = manual_recipients
        if draft_supplier_map:
            pass_fields["draft_supplier_map"] = {
                key: value for key, value in draft_supplier_map.items() if key
            }
        if thread_state_root:
            pass_fields["email_thread_state"] = thread_state_root

        self._record_learning_events(context, drafts, data)

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields=pass_fields,
            ),
        )


    def _derive_sender_identity(
        self, sender_email: str, override_title: Optional[str] = None
    ) -> tuple[str, str]:
        local_part = sender_email.split('@', 1)[0]
        tokens = [tok for tok in re.split(r"[._]", local_part) if tok]
        name = " ".join(token.capitalize() for token in tokens) if tokens else "ProcWise Sourcing"
        title = override_title or "Procurement Lead"
        return name, title

    def _build_template_args(
        self,
        supplier: Dict,
        profile: Dict,
        base_args: Dict,
        context: Dict,
        *,
        include_rfq_table: bool = True,
    ) -> Dict:
        html_augmented = {
            f"{key}_html": self._format_plain_text(value)
            for key, value in base_args.items()
            if f"{key}_html" not in base_args
        }
        base_args.update(html_augmented)

        rfq_table = self._render_rfq_table(profile) if include_rfq_table else ""
        scope_summary = self._compose_scope_summary(supplier, profile, context)

        deadline_value = base_args.get("deadline")
        if not deadline_value:
            fallback = context.get("deadline") or context.get("submission_deadline")
            deadline_value = fallback if fallback else "TBC"
            base_args["deadline"] = deadline_value
        deadline_clean = str(deadline_value).strip() or "TBC"
        base_args["deadline_html"] = self._format_plain_text(deadline_clean)

        return {
            "scope_summary_html": self._format_plain_text(scope_summary),
            "rfq_table_html": rfq_table,
            "deadline": base_args.get("deadline"),
            "deadline_html": base_args.get("deadline_html"),
        }

    def _compose_scope_summary(
        self, supplier: Dict, profile: Dict, context: Dict
    ) -> str:
        sentences: List[str] = []
        relationship = self._relationship_sentence(supplier)
        scope_sentences = self._scope_sentences(supplier, profile, context)

        for candidate in [relationship, *scope_sentences]:
            formatted = self._ensure_sentence(candidate)
            if formatted:
                sentences.append(formatted)
            if len(sentences) >= 2:
                break

        if not sentences:
            sentences.append("The detailed requirement is TBC.")
        return " ".join(sentences)

    def _relationship_sentence(self, supplier: Dict) -> str:
        spend = self._as_float(supplier.get("total_spend")) or 0.0
        po_count = self._as_float(supplier.get("po_count")) or 0.0
        if po_count > 0:
            return (
                "We appreciate the recent collaboration and look forward to your updated quotation"
            )
        if spend > 0:
            return (
                "We value our ongoing work together and would welcome your refreshed proposal"
            )
        return "We would welcome your response for this sourcing need"

    def _scope_sentences(
        self, supplier: Dict, profile: Dict, context: Dict
    ) -> List[str]:
        sentences: List[str] = []
        items = profile.get("items") or []
        categories = profile.get("categories") or {}
        descriptions = [
            str(description).strip()
            for description in items
            if isinstance(description, str) and str(description).strip()
        ]
        category_focus = self._first_category_value(categories)
        if descriptions and category_focus:
            scope = ", ".join(descriptions[:3])
            sentences.append(
                f"The request covers {scope} within {category_focus}"
            )
        elif descriptions:
            scope = ", ".join(descriptions[:3])
            sentences.append(f"The request covers {scope}")
        elif category_focus:
            sentences.append(f"The sourcing focus is in {category_focus}")
        else:
            fallback_scope = context.get("requirement_summary")
            if isinstance(fallback_scope, str) and fallback_scope.strip():
                sentences.append(fallback_scope.strip())

        justification_clean = self._clean_justification(
            supplier.get("justification")
        )
        if justification_clean:
            sentences.append(justification_clean)

        if not sentences:
            sentences.append("The detailed requirement is TBC")

        return sentences

    @staticmethod
    def _clean_justification(text: Optional[str]) -> Optional[str]:
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        banned = (
            "rank",
            "ranking",
            "analysis",
            "score",
            "scoring",
            "evaluation",
            "assess",
            "assessment",
            "scorecard",
            "calculation",
            "internal analysis",
        )
        if any(keyword in lowered for keyword in banned):
            return None
        return cleaned

    @staticmethod
    def _ensure_sentence(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        sentence = str(text).strip()
        if not sentence:
            return None
        if sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    @staticmethod
    def _first_category_value(categories: Dict) -> Optional[str]:
        if not isinstance(categories, dict):
            return None
        for key in sorted(categories.keys()):
            values = categories.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return None

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    def _render_rfq_table(self, profile: Dict) -> str:
        items = profile.get("items") or []
        if not items:
            return RFQ_TABLE_HEADER
        descriptions = [
            str(description).strip()
            for description in items
            if isinstance(description, str) and str(description).strip()
        ]
        return _build_rfq_table_html(descriptions)

    def _format_plain_text(self, value: Optional[str]) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return escape(text).replace("\n", "<br>")

    def _split_existing_comment(self, body: str) -> tuple[Optional[str], str]:
        return split_hidden_marker(body)

    def _extract_rfq_id(self, comment: Optional[str]) -> Optional[str]:
        return extract_rfq_id(comment)

    def _sanitise_generated_body(self, body: str) -> str:
        if not body:
            return ""
        text = body.strip()
        text = re.sub(r"^```(?:html|markdown)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text).strip()
        greeting = re.search(r"(<p[^>]*>|Dear\s+|Hello\s+|Hi\s+)", text, flags=re.IGNORECASE)
        if greeting and greeting.start() > 0:
            text = text[greeting.start():].lstrip()
        if "<" not in text:
            paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
            if not paragraphs:
                return ""
            text = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)

        suggestion_pattern = re.compile(
            r"<p[^>]*>[^<]*(suggest|recommend|consider)[^<]*</p>", re.IGNORECASE
        )
        text = suggestion_pattern.sub("", text)
        restricted_pattern = re.compile(
            r"<p[^>]*>[^<]*(rank|ranking|analysis|score|scoring|scorecard|evaluation|assess|assessment|calculation|internal\\s+analysis)[^<]*</p>",
            re.IGNORECASE,
        )
        text = restricted_pattern.sub("", text)
        return text

    def _normalise_recipients(self, recipients: Any) -> List[str]:
        if recipients is None:
            return []
        values: List[str] = []
        if isinstance(recipients, str):
            values = [recipients]
        elif isinstance(recipients, Iterable):
            values = [
                str(value)
                for value in recipients
                if isinstance(value, str) and str(value).strip()
            ]
        else:
            return []

        normalised: List[str] = []
        seen: set[str] = set()
        for value in values:
            candidate = value.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalised.append(candidate)
        return normalised

    @staticmethod
    def _normalise_tracking_value(value: Optional[Any]) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    def _generate_unique_identifier(
        self, workflow_id: Optional[str], supplier_id: Optional[str]
    ) -> str:
        """Generate unique identifier for emails."""

        workflow_hint = self._normalise_tracking_value(workflow_id)
        if not workflow_hint:
            workflow_hint = f"WF-{uuid.uuid4().hex}"

        supplier_hint = self._normalise_tracking_value(supplier_id)
        return generate_unique_email_id(workflow_hint, supplier_hint)

    def _resolve_unique_id(
        self,
        *,
        workflow_id: Optional[Any],
        supplier_id: Optional[Any],
        existing: Optional[Any] = None,
    ) -> str:
        existing_value = self._normalise_tracking_value(existing)
        if existing_value:
            return existing_value

        workflow_hint = self._normalise_tracking_value(workflow_id)
        return self._generate_unique_identifier(workflow_hint, supplier_id)

    def execute(self, context: AgentContext) -> AgentOutput:
        result = super().execute(context)
        try:
            self._synchronise_draft_records(result)
        except Exception:  # pragma: no cover - defensive sync
            logger.exception("failed to synchronise draft action metadata")
        return result

    @staticmethod
    def _normalise_workflow_context_payload(
        candidate: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(candidate, Mapping):
            return None

        normalised: Dict[str, Any] = {}
        for key, value in candidate.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalised[str(key)] = value
            else:
                text = str(value).strip()
                if text:
                    normalised[str(key)] = text
        return normalised or None

    def _apply_workflow_context(
        self,
        draft: Dict[str, Any],
        context: AgentContext,
        source_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ensure drafts inherit workflow metadata and invocation context."""

        if not isinstance(draft, dict):
            return draft

        metadata_source = draft.get("metadata")
        metadata: Dict[str, Any]
        if isinstance(metadata_source, Mapping):
            metadata = dict(metadata_source)
        else:
            metadata = {}

        workflow_id = getattr(context, "workflow_id", None) or metadata.get("workflow_id")
        if workflow_id:
            draft.setdefault("workflow_id", workflow_id)
            metadata.setdefault("workflow_id", workflow_id)

        parent_agent = getattr(context, "parent_agent", None)
        if parent_agent:
            metadata.setdefault("parent_agent", parent_agent)

        workflow_context_candidate: Optional[Mapping[str, Any]] = None
        if isinstance(source_payload, Mapping):
            workflow_context_candidate = (
                source_payload.get("workflow_dispatch_context")
                or source_payload.get("workflow_context")
            )
            if workflow_context_candidate is None and isinstance(
                source_payload.get("metadata"), Mapping
            ):
                inner_meta = source_payload.get("metadata")
                if isinstance(inner_meta, Mapping):
                    workflow_context_candidate = (
                        workflow_context_candidate
                        or inner_meta.get("workflow_dispatch_context")
                        or inner_meta.get("workflow_context")
                    )

        if workflow_context_candidate is None and isinstance(context.input_data, Mapping):
            workflow_context_candidate = (
                context.input_data.get("workflow_dispatch_context")
                or context.input_data.get("workflow_context")
            )

        if workflow_context_candidate is None and metadata.get("workflow_context"):
            existing_context = metadata.get("workflow_context")
            if isinstance(existing_context, Mapping):
                workflow_context_candidate = existing_context

        workflow_context = self._normalise_workflow_context_payload(
            workflow_context_candidate
        )
        if workflow_context:
            metadata.setdefault("workflow_context", workflow_context)
            draft.setdefault("workflow_context", workflow_context)

        workflow_email_flag: Optional[Any] = None
        if isinstance(source_payload, Mapping):
            if "workflow_email" in source_payload:
                workflow_email_flag = source_payload.get("workflow_email")
            elif "is_workflow_email" in source_payload:
                workflow_email_flag = source_payload.get("is_workflow_email")
        if workflow_email_flag is None and isinstance(context.input_data, Mapping):
            input_meta = context.input_data
            if "workflow_email" in input_meta:
                workflow_email_flag = input_meta.get("workflow_email")
            elif "is_workflow_email" in input_meta:
                workflow_email_flag = input_meta.get("is_workflow_email")
        if workflow_email_flag is None and "workflow_email" in metadata:
            workflow_email_flag = metadata.get("workflow_email")
        if (
            workflow_email_flag is None
            and workflow_context
            and isinstance(parent_agent, str)
            and parent_agent.lower() == "negotiationagent"
        ):
            workflow_email_flag = True
        if workflow_email_flag is not None:
            metadata["workflow_email"] = bool(workflow_email_flag)
            draft.setdefault("workflow_email", bool(workflow_email_flag))

        draft["metadata"] = metadata

        return draft

    def _store_draft(self, draft: dict) -> None:
        """Persist email draft to ``proc.draft_rfq_emails``."""
        if not isinstance(draft, dict):
            return

        metadata_source = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
        supplier_candidates = (
            draft.get("supplier_id"),
            metadata_source.get("supplier_id"),
            draft.get("supplier"),
            metadata_source.get("supplier"),
        )
        supplier_id: Optional[str] = None
        for candidate in supplier_candidates:
            text = self._coerce_text(candidate)
            if text:
                supplier_id = text
                break

        if not supplier_id:
            raise ValueError("EmailDraftingAgent requires supplier_id before storing draft")

        metadata = dict(metadata_source)
        metadata["supplier_id"] = supplier_id
        draft["supplier_id"] = supplier_id
        draft["metadata"] = metadata

        try:
            with self.agent_nick.get_db_connection() as conn:
                self._ensure_table_exists(conn)

                workflow_id = draft.get("workflow_id")
                metadata = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}

                if not workflow_id:
                    workflow_id = metadata.get("workflow_id")
                if not workflow_id:
                    workflow_id = f"WF-{uuid.uuid4().hex[:16]}"

                draft["workflow_id"] = workflow_id
                metadata = dict(metadata)
                metadata["workflow_id"] = workflow_id
                draft["metadata"] = metadata
                metadata.setdefault("workflow_id", workflow_id)

                logger.info(
                    "Storing draft with workflow_id=%s unique_id=%s supplier=%s",
                    workflow_id,
                    draft.get("unique_id"),
                    draft.get("supplier_id"),
                )

                unique_id = self._resolve_unique_id(
                    workflow_id=workflow_id,
                    supplier_id=draft.get("supplier_id"),
                    existing=draft.get("unique_id") or metadata.get("unique_id"),
                )
                draft["unique_id"] = unique_id
                metadata.setdefault("unique_id", unique_id)

                run_id = draft.get("run_id")
                if not run_id:
                    run_id = metadata.get("run_id")
                if run_id:
                    draft["run_id"] = run_id
                    metadata.setdefault("run_id", run_id)

                body_text = draft.get("body") or ""
                body_text = self._clean_body_text(body_text)

                marker_token = metadata.get("dispatch_token")
                body_text, marker_token = attach_hidden_marker(
                    body_text,
                    supplier_id=draft.get("supplier_id"),
                    token=marker_token,
                    run_id=run_id,
                    unique_id=unique_id,
                )
                draft["body"] = body_text
                if marker_token:
                    metadata["dispatch_token"] = marker_token

                subject_input = draft.get("subject", "")
                subject_clean = self._clean_subject_text(subject_input, DEFAULT_RFQ_SUBJECT)
                draft["subject"] = subject_clean

                mailbox_hint = draft.get("mailbox")
                if not mailbox_hint:
                    mailbox_hint = metadata.get("mailbox")
                if mailbox_hint:
                    draft["mailbox"] = mailbox_hint
                    metadata.setdefault("mailbox", mailbox_hint)

                headers = (
                    draft.get("headers")
                    if isinstance(draft.get("headers"), dict)
                    else {}
                )
                round_number_raw = (
                    draft.get("round")
                    or metadata.get("round")
                    or draft.get("thread_index")
                )
                try:
                    round_number = int(round_number_raw) if round_number_raw is not None else 1
                except Exception:
                    round_number = 1
                if round_number < 1:
                    round_number = 1
                metadata["round"] = round_number
                draft["round"] = round_number

                required_headers = {
                    "X-ProcWise-Workflow-ID": workflow_id,
                    "X-ProcWise-Unique-ID": unique_id,
                    "X-ProcWise-Supplier-ID": supplier_id,
                    "X-ProcWise-Round": str(round_number),
                }
                for key, value in required_headers.items():
                    if value in (None, ""):
                        raise ValueError(
                            f"EmailDraftingAgent requires non-empty header {key}"
                        )
                    headers[key] = str(value)

                if mailbox_hint:
                    headers.setdefault("X-ProcWise-Mailbox", mailbox_hint)

                logger.info(
                    "EmailDraftingAgent: embedded headers workflow=%s, supplier=%s, unique_id=%s, round=%s",
                    workflow_id,
                    supplier_id,
                    unique_id,
                    round_number,
                )

                draft["headers"] = {k: v for k, v in headers.items() if v is not None}

                if not workflow_id or not unique_id or not supplier_id:
                    raise ValueError(
                        "EmailDraftingAgent requires workflow_id, unique_id, and supplier_id for header embedding"
                    )

                logger.info(
                    "EmailDraftingAgent: embedded headers workflow=%s, supplier=%s, unique_id=%s, round=%s",
                    workflow_id,
                    supplier_id,
                    unique_id,
                    round_number,
                )

                thread_index = draft.get("thread_index")
                if not isinstance(thread_index, int) or thread_index < 1:
                    thread_index = self._next_thread_index(conn, unique_id)
                    draft["thread_index"] = thread_index

                recipients = draft.get("recipients")
                if isinstance(recipients, list) and recipients:
                    receiver = str(recipients[0]).strip()
                else:
                    receiver = draft.get("receiver")
                    receiver = str(receiver).strip() if isinstance(receiver, str) else None

                contact_level = draft.get("contact_level")
                try:
                    contact_level_int = int(contact_level)
                except Exception:
                    contact_level_int = 1 if receiver else 0
                contact_level_int = max(0, contact_level_int)
                draft["contact_level"] = contact_level_int

                persisted_draft = dict(draft)
                persisted_draft.pop("draft_record_id", None)
                persisted_draft.pop("id", None)
                payload = json.dumps(persisted_draft, default=str)

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.draft_rfq_emails
                        (rfq_id, unique_id, supplier_id, supplier_name, subject, body, created_on, sent,
                         recipient_email, contact_level, thread_index, sender, payload,
                         workflow_id, run_id, mailbox)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (workflow_id, unique_id) DO UPDATE SET
                            rfq_id = EXCLUDED.rfq_id,
                            unique_id = EXCLUDED.unique_id,
                            supplier_id = EXCLUDED.supplier_id,
                            supplier_name = EXCLUDED.supplier_name,
                            subject = EXCLUDED.subject,
                            body = EXCLUDED.body,
                            sent = EXCLUDED.sent,
                            recipient_email = EXCLUDED.recipient_email,
                            contact_level = EXCLUDED.contact_level,
                            thread_index = EXCLUDED.thread_index,
                            sender = EXCLUDED.sender,
                            payload = EXCLUDED.payload,
                            workflow_id = EXCLUDED.workflow_id,
                            run_id = EXCLUDED.run_id,
                            mailbox = EXCLUDED.mailbox
                        RETURNING id
                        """,
                        (
                            unique_id,
                            unique_id,
                            draft.get("supplier_id"),
                            draft.get("supplier_name"),
                            draft["subject"],
                            draft["body"],
                            bool(draft.get("sent_status")),
                            receiver,
                            contact_level_int,
                            thread_index,
                            draft.get("sender"),
                            payload,
                            workflow_id,
                            run_id,
                            mailbox_hint,
                        ),
                    )
                    row = cur.fetchone()
                    record_id = row[0] if row else None
                conn.commit()
                if record_id is not None:
                    draft["id"] = record_id
                    draft["draft_record_id"] = record_id
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store RFQ draft")

    def _record_learning_events(
        self,
        context: AgentContext,
        drafts: Iterable[Dict[str, Any]],
        source_data: Dict[str, Any],
    ) -> None:
        workflow_id = getattr(context, "workflow_id", None)
        context_snapshot = {
            "intent": source_data.get("intent"),
            "document_origin": source_data.get("document_origin")
            or source_data.get("document_type"),
            "target_price": source_data.get("target_price"),
            "current_offer": source_data.get("current_offer"),
        }
        if self._memory_enabled():
            for draft in drafts:
                if not isinstance(draft, Mapping):
                    continue
                snapshot = self._prepare_learning_snapshot(draft)
                if not snapshot:
                    continue
                try:
                    self.workflow_memory.enqueue_learning_event(
                        workflow_id,
                        {
                            "category": "email_draft",
                            "draft": snapshot,
                            "context": context_snapshot,
                        },
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.debug(
                        "Failed to queue email learning event for workflow=%s",
                        workflow_id,
                        exc_info=True,
                    )
            return

        repository = getattr(self, "learning_repository", None)
        if repository is None:
            return
        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            try:
                repository.record_email_learning(
                    workflow_id=workflow_id,
                    draft=draft,
                    context=context_snapshot,
                )
            except Exception:
                logger.debug(
                    "Learning capture failed for draft %s", draft.get("unique_id"), exc_info=True
                )

    def _synchronise_draft_records(self, result: AgentOutput) -> None:
        drafts = (result.data or {}).get("drafts") if isinstance(result, AgentOutput) else None
        if not drafts:
            return

        try:
            with self.agent_nick.get_db_connection() as conn:
                for draft in drafts:
                    if not isinstance(draft, dict):
                        continue
                    record_id = draft.get("draft_record_id") or draft.get("id")
                    if not record_id:
                        continue
                    payload_doc = dict(draft)
                    payload_doc.pop("draft_record_id", None)
                    payload_doc.pop("id", None)
                    payload = json.dumps(payload_doc, default=str)
                    sent_flag = bool(draft.get("sent_status"))
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE proc.draft_rfq_emails
                            SET payload = %s,
                                sent = %s,
                                updated_on = NOW()
                            WHERE id = %s
                            """,
                            (payload, sent_flag, record_id),
                        )
                conn.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to update stored draft metadata")

    def _ensure_table_exists(self, conn) -> None:
        """Create the backing draft table on demand."""
        if self._draft_table_checked:
            return

        with self._draft_table_lock:
            if self._draft_table_checked:
                return

            with conn.cursor() as cur:
                cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS proc.draft_rfq_emails (
                        id BIGSERIAL PRIMARY KEY,
                        rfq_id TEXT NOT NULL,
                        supplier_id TEXT,
                        supplier_name TEXT,

                        subject TEXT NOT NULL,
                        body TEXT NOT NULL,
                        created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        sent BOOLEAN NOT NULL DEFAULT FALSE,
                        sent_on TIMESTAMPTZ,
                        recipient_email TEXT,
                        contact_level INTEGER NOT NULL DEFAULT 0,
                        thread_index INTEGER NOT NULL DEFAULT 1,
                        sender TEXT,
                        payload JSONB,
                        updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS supplier_name TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS sent_on TIMESTAMPTZ"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS recipient_email TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS contact_level INTEGER NOT NULL DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS thread_index INTEGER NOT NULL DEFAULT 1"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS sender TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS payload JSONB"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS workflow_id TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS run_id TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS unique_id TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS mailbox TEXT"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS dispatched_at TIMESTAMPTZ"
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS dispatch_run_id TEXT"
                )
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_draft_rfq_emails_wf_uid"
                    " ON proc.draft_rfq_emails (workflow_id, unique_id)"
                )


            self._draft_table_checked = True

    def _next_thread_index(self, conn, unique_id: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(thread_index), 0) FROM proc.draft_rfq_emails WHERE unique_id = %s",
                (unique_id,),
            )
            row = cur.fetchone()
        try:
            current = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            current = 0
        return current + 1

    def _resolve_receiver(self, supplier: Dict[str, Any], profile: Dict[str, Any]) -> Optional[str]:
        """Determine the best receiver email for a supplier."""

        candidates: List[str] = []

        def _append_candidate(value: Optional[str]) -> None:
            if not value:
                return
            candidate = str(value).strip()
            if candidate and candidate.lower() not in {c.lower() for c in candidates}:
                candidates.append(candidate)

        _append_candidate(supplier.get("contact_email"))
        _append_candidate(supplier.get("contact_email_1"))
        _append_candidate(supplier.get("contact_email_2"))

        contacts = profile.get("contacts") if isinstance(profile, dict) else None
        if isinstance(contacts, Sequence):
            for contact in contacts:
                if isinstance(contact, dict):
                    _append_candidate(contact.get("email"))
                    _append_candidate(contact.get("contact_email"))

        return candidates[0] if candidates else None
