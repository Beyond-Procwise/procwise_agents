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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from jinja2 import Template

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.email_markers import attach_hidden_marker, extract_rfq_id, split_hidden_marker
from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources

logger = logging.getLogger(__name__)

configure_gpu()


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


def _current_rfq_date() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def _generate_rfq_id() -> str:
    return f"RFQ-{_current_rfq_date()}-{uuid.uuid4().hex[:8].upper()}"


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
    rfq_id: Optional[str] = None
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
            "rfq_id": self.rfq_id,
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
_VISIBLE_RFQ_ID_PATTERN = re.compile(r"(?i)RFQ[\w\s:-]*\d[\w-]*")

DEFAULT_RFQ_SUBJECT = "Request for Quotation – Procurement Opportunity"
DEFAULT_NEGOTIATION_SUBJECT = "Re: Procurement Negotiation Update"
DEFAULT_FOLLOW_UP_SUBJECT = "Procurement Follow Up"


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


NEGOTIATION_COUNTER_SYSTEM_PROMPT = (
    "You are an elite procurement negotiator crafting decisive, relationship-aware "
    "emails. Blend firmness with partnership, ground every request in the data "
    "provided, and never invent figures. Keep replies under 160 words, use clear "
    "paragraphs or short bullet points, and always reference the RFQ ID. Do not "
    "discuss internal scorecards, evaluation formulas, or any analysis logic used to "
    "assess suppliers."
)

NEGOTIATION_COUNTER_USER_PROMPT = (
    "Context (JSON):\n{payload}\n\n"
    "Draft a high-impact counter email that: (1) anchors the counter price or asks, "
    "(2) justifies the request with business rationale, (3) offers alternate value "
    "creation paths (tiered pricing, lead time flex, added terms), and (4) requests "
    "confirmation within 2–3 business days. Maintain professional warmth and avoid "
    "hyperbole."
)

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
        "<p><strong>Note:</strong> Please do not change the subject line when replying.</p>"
        "{rfq_table_html}"
        "<p>We appreciate your timely response.</p>"
        "<p>Kind regards,<br>{your_name_html}<br>{your_title_html}<br>{your_company_html}</p>"
    )

    @staticmethod
    def _strip_rfq_identifier_tokens(text: Optional[str]) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = _VISIBLE_RFQ_ID_PATTERN.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def _clean_subject_text(cls, subject: Optional[str], fallback: str) -> str:
        prefix = ""
        candidate = subject.strip() if isinstance(subject, str) else ""
        if candidate:
            prefix_match = re.match(r"(?i)^(re|fw|fwd):\s*", candidate)
            if prefix_match:
                prefix = prefix_match.group(0)
                candidate = candidate[prefix_match.end():]
            candidate = cls._strip_rfq_identifier_tokens(candidate)
            candidate = candidate.strip("-–: ")
            if candidate:
                return f"{prefix}{candidate}".strip()

        fallback_clean = cls._strip_rfq_identifier_tokens(fallback or "").strip("-–: ")
        if prefix and fallback_clean:
            fallback_without_prefix = re.sub(r"(?i)^(re|fw|fwd):\s*", "", fallback_clean)
            combined = f"{prefix}{fallback_without_prefix}".strip()
            return combined or prefix.strip()
        result = fallback_clean or prefix.strip()
        return result if result else "Procurement Update"

    @staticmethod
    def _clean_body_text(body: Optional[str]) -> str:
        if not isinstance(body, str):
            return ""
        cleaned = _VISIBLE_RFQ_ID_PATTERN.sub("", body)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

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

        supplied_rfq = self._normalise_rfq_identifier(decision_data.get("rfq_id"))
        payload_rfq = _generate_rfq_id()
        rfq_id_output = supplied_rfq or payload_rfq

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
            "rfq_id": payload_rfq,
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
        user_prompt = (
            f"Context (JSON):\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            "Write a professional negotiation counter email with an explicit Subject line. "
            "Reference the RFQ ID provided, summarise the commercial position, list the key asks "
            "as bullet points, and close with a clear call to action."
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
            rfq_id_output,
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

        html_body = self._render_html_from_text(body_text)
        sanitised_html = self._sanitise_generated_body(html_body)
        if not sanitised_html:
            sanitised_html = self._sanitise_generated_body(
                self._render_html_from_text(fallback_text)
            )
        plain_text = self._html_to_plain_text(sanitised_html) if sanitised_html else fallback_text
        if not sanitised_html and plain_text:
            sanitised_html = self._render_html_from_text(plain_text)

        if sanitised_html:
            sanitised_html = self._clean_body_text(sanitised_html)
        if plain_text:
            plain_text = self._clean_body_text(plain_text)

        annotated_body, marker_token = attach_hidden_marker(
            plain_text or "",
            rfq_id=rfq_id_output,
            supplier_id=supplier_id,
        )

        if subject_line:
            subject = self._clean_subject_text(subject_line.strip(), DEFAULT_NEGOTIATION_SUBJECT)
        else:
            subject = self._clean_subject_text(None, DEFAULT_NEGOTIATION_SUBJECT)

        thread_headers = decision_data.get("thread")
        headers: Dict[str, Any] = {"X-Procwise-RFQ-ID": rfq_id_output}
        if isinstance(thread_headers, dict):
            message_id = thread_headers.get("message_id")
            if message_id:
                headers["In-Reply-To"] = message_id
            references = thread_headers.get("references")
            if isinstance(references, list) and references:
                headers["References"] = " ".join(str(ref) for ref in references if ref)

        metadata = {
            "rfq_id": rfq_id_output,
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
        metadata = {k: v for k, v in metadata.items() if v is not None}

        draft = {
            "rfq_id": rfq_id_output,
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
        }

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

        supplied_rfq = self._normalise_rfq_identifier(context.get("rfq_id"))
        rfq_id_value = supplied_rfq or _generate_rfq_id()

        payload = {"rfq_id": rfq_id_value, "prompt": prompt_text}
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
                "rfq_id": rfq_id_value,
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

        supplier_id = context.get("supplier_id")
        annotated_body, marker_token = attach_hidden_marker(
            plain_text or "",
            rfq_id=rfq_id_value,
            supplier_id=supplier_id,
        )

        counter_price = context.get("counter_price")
        if counter_price is None:
            proposals = context.get("counter_proposals")
            if isinstance(proposals, Iterable):
                for proposal in proposals:
                    if isinstance(proposal, dict) and proposal.get("price") is not None:
                        counter_price = proposal.get("price")
                        break

        metadata = {
            "intent": context.get("intent") or "PROMPT_COMPOSE",
            "rfq_id": rfq_id_value,
        }
        if counter_price is not None:
            metadata["counter_price"] = counter_price
        if marker_token:
            metadata["dispatch_token"] = marker_token

        draft = {
            "rfq_id": rfq_id_value,
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
            "headers": {"X-Procwise-RFQ-ID": rfq_id_value},
        }

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

    def _handle_negotiation_counter(self, context: AgentContext, data: Dict[str, Any]) -> AgentOutput:
        supplier_id = data.get("supplier_id")
        supplier_name = data.get("supplier_name") or supplier_id or "Supplier"
        rfq_id_raw = data.get("rfq_id")
        rfq_id = str(rfq_id_raw).strip().upper() if rfq_id_raw else "RFQ"
        decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
        counter_price = data.get("counter_price")
        if counter_price is None:
            counter_price = decision.get("counter_price")
        target_price = data.get("target_price")
        if target_price is None:
            target_price = decision.get("target_price")
        current_offer = data.get("current_offer_numeric")
        if current_offer is None:
            current_offer = data.get("current_offer")
        currency = data.get("currency") or decision.get("currency")
        asks = data.get("asks") if isinstance(data.get("asks"), list) else decision.get("asks", [])
        if not isinstance(asks, list):
            asks = [asks] if asks else []
        lead_time_request = data.get("lead_time_request") or decision.get("lead_time_request")
        rationale = data.get("rationale") or decision.get("rationale")
        negotiation_message = (
            data.get("negotiation_message")
            or decision.get("negotiation_message")
            or data.get("message")
        )
        supplier_message = data.get("supplier_message")
        round_value = data.get("round") or decision.get("round") or 1
        try:
            round_no = int(round_value)
        except Exception:
            round_no = 1
        session_state = data.get("session_state") if isinstance(data.get("session_state"), dict) else {}
        reply_count_value = data.get("supplier_reply_count")
        if reply_count_value is None and session_state:
            reply_count_value = session_state.get("supplier_reply_count")
        try:
            supplier_reply_count = int(reply_count_value) if reply_count_value is not None else 0
        except Exception:
            supplier_reply_count = 0
        strategy = decision.get("strategy")

        payload = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "round": round_no,
            "current_offer": current_offer,
            "target_price": target_price,
            "counter_price": counter_price,
            "currency": currency,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "rationale": rationale,
            "negotiation_message": negotiation_message,
            "supplier_message": supplier_message,
            "supplier_reply_count": supplier_reply_count,
            "strategy": strategy,
        }

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        model_name = getattr(self.agent_nick.settings, "negotiation_email_model", DEFAULT_NEGOTIATION_MODEL)
        messages = [
            {"role": "system", "content": NEGOTIATION_COUNTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": NEGOTIATION_COUNTER_USER_PROMPT.format(payload=payload_json),
            },
        ]

        email_text = ""
        try:
            response = self.call_ollama(
                model=model_name,
                messages=messages,
                options={"temperature": 0.35, "top_p": 0.9, "num_ctx": 4096},
            )
            email_text = self._extract_ollama_message(response)
        except Exception:  # pragma: no cover - defensive composition
            logger.exception(
                "Failed to compose negotiation counter email via model %s", model_name
            )

        if not email_text:
            email_text = self._build_negotiation_fallback(
                rfq_id,
                counter_price,
                target_price,
                current_offer,
                currency,
                asks,
                lead_time_request,
                negotiation_message,
                round_no,
                decision.get("line_items") or data.get("line_items"),
            )

        body_content = self._sanitise_generated_body(email_text)
        body = self._clean_body_text(body_content)
        body, marker_token = attach_hidden_marker(
            body,
            rfq_id=rfq_id,
            supplier_id=supplier_id,
        )

        subject_base = data.get("subject") or DEFAULT_NEGOTIATION_SUBJECT
        subject = self._clean_subject_text(subject_base, DEFAULT_NEGOTIATION_SUBJECT)

        sender_email = data.get("sender") or self.agent_nick.settings.ses_default_sender
        recipients = self._normalise_recipients(data.get("recipients") or data.get("to"))
        receiver = recipients[0] if recipients else None
        contact_level = 1 if recipients else 0

        metadata = {
            "counter_price": counter_price,
            "target_price": target_price,
            "current_offer": current_offer,
            "round": round_no,
            "supplier_reply_count": supplier_reply_count,
            "strategy": strategy,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "rationale": rationale,
            "intent": "NEGOTIATION_COUNTER",
        }

        if marker_token:
            metadata["dispatch_token"] = marker_token

        draft = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "rfq_id": rfq_id,
            "subject": subject,
            "body": body,
            "sent_status": False,
            "sender": sender_email,
            "recipients": recipients,
            "receiver": receiver,
            "contact_level": contact_level,
            "metadata": metadata,
        }

        negotiation_extra = {
            "negotiation_message": negotiation_message,
            "supplier_message": supplier_message,
        }
        draft.update({k: v for k, v in negotiation_extra.items() if v})

        action_id = data.get("action_id") or context.input_data.get("action_id")
        if action_id:
            draft["action_id"] = action_id
        thread_index = data.get("thread_index")
        try:
            thread_index_int = int(thread_index) if thread_index is not None else 1
        except Exception:
            thread_index_int = 1
        draft.setdefault("thread_index", max(1, thread_index_int))

        self._store_draft(draft)
        self._record_learning_events(context, [draft], data)

        output_data: Dict[str, Any] = {
            "drafts": [draft],
            "subject": subject,
            "body": body,
            "sender": sender_email,
            "intent": "NEGOTIATION_COUNTER",
        }
        if recipients:
            output_data["recipients"] = recipients
        if action_id:
            output_data["action_id"] = action_id

        pass_fields: Dict[str, Any] = {"drafts": [draft]}
        if action_id:
            pass_fields["action_id"] = action_id

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields=pass_fields,
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
        rfq_id: str,
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
        lines = ["Dear Procwise,"]

        if round_no is None:
            try:
                round_no = int(self.agent_nick.settings.negotiation_round)
            except Exception:
                round_no = 1
        if round_no <= 1:
            intro = (
                "Thank you for the RFQ update and for the competitive pricing you have shared "
                "with us previously."
            )
            lines.extend(["", intro])
            if counter_text:
                clause = f"If we can finalise this week, I am authorised to move to {counter_text}"
                if offer_text:
                    clause += f" against your current {offer_text}"
                if target_text and target_text != counter_text:
                    clause += f" (target {target_text})"
                clause += "."
                lines.append(clause)
        elif round_no == 2:
            lines.extend(
                [
                    "",
                    "Our last position reflected the scope as we understood it."
                    " To explore any further movement we would need to adjust a deal lever.",
                ]
            )
            lever_options = [
                "a longer commitment term",
                "a higher consolidated volume",
                "accelerated payment terms",
            ]
            lever = lever_options[0]
            if asks:
                lever = str(asks[0]).strip() or lever
            lines.append(f"If we can secure {lever}, I can revisit the pricing immediately.")
        else:
            lines.extend(
                [
                    "",
                    "We have now exhausted our internal flexibility and the figures below represent the final position we can hold.",
                ]
            )

        if counter_text and round_no != 3:
            lines.append(f"Our counter position stands at {counter_text}.")
        elif target_text and not counter_text:
            lines.append(f"Our pricing objective remains {target_text}.")
        if lead_time_request:
            lines.append(f"Lead time request: {lead_time_request}.")

        asks_list = [str(item).strip() for item in asks if str(item).strip()]
        if asks_list:
            lines.append("")
            lines.append("Key considerations:")
            lines.extend(f"- {item}" for item in asks_list)

        table_lines = self._render_negotiation_table(line_items)
        if table_lines:
            lines.extend(["", *table_lines])

        if negotiation_message:
            lines.extend(["", negotiation_message])

        lines.append("")
        lines.append(
            "Could you confirm we can proceed this week so that we can lock in the pricing and availability?"
        )
        lines.append("")
        lines.append("Thank you,")
        lines.append("Procwise Sourcing Team")
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
            return True

        if instruction_settings.get("interaction_type") or context.get("interaction_type"):
            return True

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

        scope_html = template_args.get("scope_summary_html")
        if scope_html:
            sections.append(f"<p>{scope_html}</p>")

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

        if include_rfq_table and template_args.get("rfq_table_html"):
            sections.append(template_args["rfq_table_html"])

        closing = self._build_dynamic_closing(
            interaction_type,
            template_args,
            tone,
            instructions,
        )
        if closing:
            sections.append(closing)

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
                base += f" Your current offer of {current_offer} is appreciated and forms the basis of this discussion."
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

        # note = " Kindly retain the RFQ ID in the email subject when replying."
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

        ranking = data.get("ranking", [])
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
        manual_rfq_id: Optional[str] = None
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

        for supplier in ranking:
            supplier_id = supplier.get("supplier_id")
            supplier_name = supplier.get("supplier_name", supplier_id)
            rfq_id = self._generate_rfq_id()

            profile = supplier_profiles.get(str(supplier_id)) if supplier_id is not None else {}
            if profile is None:
                profile = {}

            sender_email = self.agent_nick.settings.ses_default_sender
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
                    "rfq_id": rfq_id,
                    "additional_paragraph_html": additional_section_html,
                    "compliance_notice_html": compliance_section_html,
                    "negotiation_message_html": negotiation_section_html,
                }
            )
            if negotiation_message:
                template_args["negotiation_message"] = negotiation_message
            interaction_type = self._determine_interaction_type(data, instruction_settings)
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

            rendered = None
            if self._should_auto_compose(
                body_template, instruction_settings, data
            ):
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
                rfq_id=rfq_id,
                supplier_id=supplier_id,
            )
            if subject_template_source:
                subject_args = dict(template_args)
                subject_args.setdefault("rfq_id", rfq_id)
                rendered_subject = self._render_template_string(
                    subject_template_source, subject_args
                )
                subject = self._clean_subject_text(
                    self._normalise_subject_line(rendered_subject, rfq_id),
                    DEFAULT_RFQ_SUBJECT,
                )
            else:
                subject = self._clean_subject_text(None, DEFAULT_RFQ_SUBJECT)

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

            draft = {
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "rfq_id": rfq_id,
                "subject": subject,
                "body": body,
                "sent_status": False,
                "sender": self.agent_nick.settings.ses_default_sender,
                "action_id": draft_action_id,
                "supplier_profile": profile,
                "receiver": receiver,
                "contact_level": contact_level,
                "recipients": recipients,
            }
            if draft_action_id:
                draft["action_id"] = draft_action_id
            draft.setdefault("thread_index", 1)
            metadata: Dict[str, Any] = {
                "rfq_id": rfq_id,
                "interaction_type": interaction_type,
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
            drafts.append(draft)
            self._store_draft(draft)
            logger.debug("EmailDraftingAgent created draft %s for supplier %s", rfq_id, supplier_id)

        if manual_recipients and manual_has_body:
            manual_comment, manual_message = self._split_existing_comment(manual_body_input)
            manual_body_content = (
                self._sanitise_generated_body(manual_message)
                if manual_comment
                else self._sanitise_generated_body(manual_body_input)
            )
            manual_rfq_id = self._extract_rfq_id(manual_comment) or self._generate_rfq_id()
            manual_body_rendered = self._clean_body_text(manual_body_content)
            manual_body_rendered, manual_marker_token = attach_hidden_marker(
                manual_body_rendered,
                rfq_id=manual_rfq_id,
                supplier_id=None,
            )
            manual_subject_rendered = self._clean_subject_text(
                manual_subject_input,
                DEFAULT_RFQ_SUBJECT,
            )

            manual_metadata: Dict[str, Any] = {"rfq_id": manual_rfq_id}
            if manual_marker_token:
                manual_metadata["dispatch_token"] = manual_marker_token

            manual_draft = {
                "supplier_id": None,
                "supplier_name": ", ".join(manual_recipients),
                "rfq_id": manual_rfq_id,
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
            }
            if default_action_id:
                manual_draft["action_id"] = default_action_id
            manual_draft.setdefault("thread_index", 1)
            drafts.append(manual_draft)
            self._store_draft(manual_draft)

        logger.info("EmailDraftingAgent generated %d drafts", len(drafts))
        output_data: Dict[str, Any] = {"drafts": drafts, "prompt": self.prompt_template}
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

        self._record_learning_events(context, drafts, data)

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields=pass_fields,
            ),
        )


    def _generate_rfq_id(self) -> str:
        return _generate_rfq_id()

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

    def execute(self, context: AgentContext) -> AgentOutput:
        result = super().execute(context)
        try:
            self._synchronise_draft_records(result)
        except Exception:  # pragma: no cover - defensive sync
            logger.exception("failed to synchronise draft action metadata")
        return result

    def _store_draft(self, draft: dict) -> None:
        """Persist email draft to ``proc.draft_rfq_emails``."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                self._ensure_table_exists(conn)

                workflow_id = draft.get("workflow_id")
                if not workflow_id and isinstance(draft.get("metadata"), dict):
                    workflow_id = draft["metadata"].get("workflow_id")
                if not workflow_id:
                    workflow_id = uuid.uuid4().hex
                draft["workflow_id"] = workflow_id

                unique_id = draft.get("unique_id")
                if not unique_id:
                    unique_id = uuid.uuid4().hex
                draft["unique_id"] = unique_id

                run_id = draft.get("run_id")
                if not run_id and isinstance(draft.get("metadata"), dict):
                    run_id = draft["metadata"].get("run_id")
                if run_id:
                    draft["run_id"] = run_id

                mailbox_hint = draft.get("mailbox")
                if not mailbox_hint and isinstance(draft.get("metadata"), dict):
                    mailbox_hint = draft["metadata"].get("mailbox")
                if mailbox_hint:
                    draft["mailbox"] = mailbox_hint

                thread_index = draft.get("thread_index")
                if not isinstance(thread_index, int) or thread_index < 1:
                    thread_index = self._next_thread_index(conn, draft["rfq_id"])
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
                        (rfq_id, supplier_id, supplier_name, subject, body, created_on, sent,
                         recipient_email, contact_level, thread_index, sender, payload,
                         workflow_id, run_id, unique_id, mailbox)
                        VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            draft["rfq_id"],
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
                            unique_id,
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
        repository = getattr(self, "learning_repository", None)
        if repository is None:
            return
        workflow_id = getattr(context, "workflow_id", None)
        context_snapshot = {
            "intent": source_data.get("intent"),
            "document_origin": source_data.get("document_origin")
            or source_data.get("document_type"),
            "target_price": source_data.get("target_price"),
            "current_offer": source_data.get("current_offer"),
        }
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
                    "Learning capture failed for draft %s", draft.get("rfq_id"), exc_info=True
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


            self._draft_table_checked = True

    def _next_thread_index(self, conn, rfq_id: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(thread_index), 0) FROM proc.draft_rfq_emails WHERE rfq_id = %s",
                (rfq_id,),
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