"""Email Drafting Agent.

This module renders a procurement RFQ email based on a shared prompt
template so that both human authored and LLM generated messages follow
identical structure.  The prompt template lives in
``prompts/EmailDraftingAgent_Prompt_Template.json`` and the agent also
exposes this prompt in its output for downstream LLM usage.
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Sequence

from jinja2 import Template

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources

logger = logging.getLogger(__name__)

configure_gpu()

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
_RFQ_ID_PATTERN = re.compile(r"RFQ-ID:\s*([A-Za-z0-9_-]+)", flags=re.IGNORECASE)


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


class EmailDraftingAgent(BaseAgent):
    """Agent that drafts a plain-text RFQ email and sends it via SES."""

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

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self._draft_table_checked = False
        self._draft_table_lock = threading.Lock()
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE

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

        note = " Kindly retain the RFQ ID in the email subject when replying."
        return self._wrap_paragraph(f"{base}{note}")

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
    def _normalise_subject_line(subject: str, rfq_id: Optional[str]) -> str:
        """Remove duplicated RFQ tokens that often arise from templates."""

        if not isinstance(subject, str):
            return ""

        trimmed = subject.strip()
        if not trimmed:
            return ""

        if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
            trimmed = trimmed[1:-1].strip()

        if rfq_id:
            identifier = str(rfq_id).strip()
            if identifier:
                pattern = re.compile(
                    rf"(?i)\bRFQ[\s:-]+{re.escape(identifier)}"
                )
                trimmed = pattern.sub(identifier, trimmed, count=1)

        return trimmed

    def run(self, context: AgentContext) -> AgentOutput:
        """Draft RFQ emails for each ranked supplier without sending."""
        logger.info("EmailDraftingAgent starting with input %s", context.input_data)
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

        ranking = data.get("ranking", [])
        findings = data.get("findings", [])
        supplier_profiles = (
            data.get("supplier_profiles") if isinstance(data.get("supplier_profiles"), dict) else {}
        )
        default_action_id = data.get("action_id")
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
                }
            )
            interaction_type = self._determine_interaction_type(data, instruction_settings)
            template_args["interaction_type"] = interaction_type
            dynamic_meta = {
                "interaction_type": interaction_type,
                "instructions": instruction_settings,
                "tone": instruction_settings.get("tone") or data.get("tone"),
                "call_to_action": instruction_settings.get("call_to_action"),
                "context_note": instruction_settings.get("context_note"),
            }

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
            comment = comment if comment else f"<!-- RFQ-ID: {rfq_id} -->"
            body = comment if not content else f"{comment}\n{content}"
            if subject_template_source:
                subject_args = dict(template_args)
                subject_args.setdefault("rfq_id", rfq_id)
                subject = self._render_template_string(
                    subject_template_source, subject_args
                )
                subject = self._normalise_subject_line(subject, rfq_id)
                subject = subject or f"{rfq_id} – Request for Quotation"
            else:
                subject = f"{rfq_id} – Request for Quotation"

            draft_action_id = supplier.get("action_id") or default_action_id

            receiver = self._resolve_receiver(supplier, profile)
            recipients: List[str] = []
            if receiver and not instructions_present:
                recipients = self._normalise_recipients([receiver])
            if recipients:
                receiver = recipients[0]
            contact_level = 1 if recipients else 0

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
            manual_comment = manual_comment or f"<!-- RFQ-ID: {manual_rfq_id} -->"
            manual_body_rendered = (
                manual_comment
                if not manual_body_content
                else f"{manual_comment}\n{manual_body_content}"
            )
            manual_subject_rendered = manual_subject_input or (
                f"{manual_rfq_id} – Request for Quotation"
            )

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

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=output_data,
            pass_fields=pass_fields,
        )


    def _generate_rfq_id(self) -> str:
        return f"RFQ-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

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
        supplier_name = (
            supplier.get("supplier_name")
            or supplier.get("supplier_id")
            or "your team"
        )
        spend = self._as_float(supplier.get("total_spend")) or 0.0
        po_count = self._as_float(supplier.get("po_count")) or 0.0
        if po_count > 0:
            return (
                f"We appreciate the recent collaboration with {supplier_name} and "
                "look forward to your updated quotation"
            )
        if spend > 0:
            return (
                f"We value our ongoing work with {supplier_name} and would welcome your refreshed proposal"
            )
        return f"We would welcome {supplier_name}'s response for this sourcing need"

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
        if not isinstance(body, str):
            return None, ""
        match = re.match(r"\s*(<!--.*?-->)", body, flags=re.DOTALL)
        if match and "RFQ-ID" in match.group(1):
            comment = match.group(1).strip()
            remainder = body[match.end():].lstrip("\n")
            return comment, remainder
        return None, body

    def _extract_rfq_id(self, comment: Optional[str]) -> Optional[str]:
        if not comment:
            return None
        match = _RFQ_ID_PATTERN.search(comment)
        if match:
            return match.group(1).strip()
        return None

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
            r"<p[^>]*>[^<]*(rank|ranking|analysis|score|scoring|evaluation|assess|assessment)[^<]*</p>",
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

    def _store_draft(self, draft: dict) -> None:
        """Persist email draft to ``proc.draft_rfq_emails``."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                self._ensure_table_exists(conn)
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

                payload = json.dumps(draft, default=str)

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.draft_rfq_emails
                        (rfq_id, supplier_id, supplier_name, subject, body, created_on, sent,
                         recipient_email, contact_level, thread_index, sender, payload)
                        VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)

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

                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store RFQ draft")

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

