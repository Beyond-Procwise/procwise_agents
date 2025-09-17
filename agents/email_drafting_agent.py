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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from jinja2 import Template

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

configure_gpu()

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "EmailDraftingAgent_Prompt_Template.json"
)

with PROMPT_PATH.open("r", encoding="utf-8") as fp:
    PROMPT_TEMPLATE = json.load(fp)["prompt_template"]

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
    if not rows:
        rows = [""]

    body_rows: List[str] = []
    for description in rows:
        first_cell = escape(description.strip()) if description else "&nbsp;"
        cells = [f'<td style="{_RFQ_BODY_CELL_STYLE}">{first_cell}</td>']
        for _ in range(len(_RFQ_TABLE_COLUMNS) - 1):
            cells.append(f'<td style="{_RFQ_BODY_CELL_STYLE}">&nbsp;</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

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
        "<p>{relationship_opening_html}</p>"
        "<p>{opportunity_summary_html}</p>"
        "<p>Please review the requirement below and share your best quotation."
        " Kindly complete the table with your commercial response.</p>"
        "{rfq_table_html}"
        "<p>Evaluation focus:</p>"
        "{evaluation_focus_html}"
        "<p>Deadline for submission: {deadline_html}</p>"
        "<p>Primary contact: {contact_line_html}</p>"
        "<p><strong>Note:</strong> Please do not change the subject line when replying.</p>"
        "<p>Kind regards,<br>{your_name_html}<br>{your_title_html}<br>{your_company_html}</p>"
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self._draft_table_checked = False
        self._draft_table_lock = threading.Lock()

    def run(self, context: AgentContext) -> AgentOutput:
        """Draft RFQ emails for each ranked supplier without sending."""
        logger.info("EmailDraftingAgent starting with input %s", context.input_data)
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
            body_template = data.get("body") or self.TEXT_TEMPLATE
            contextual_args = self._build_template_args(
                supplier, profile, fmt_args, data
            )
            template_args = {**data, **fmt_args, **contextual_args}
            if "{{" in body_template or "{%" in body_template:
                rendered = Template(body_template).render(**template_args)
            else:
                try:
                    rendered = body_template.format(**template_args)
                except KeyError:
                    rendered = body_template

            comment, message = self._split_existing_comment(rendered)
            content = self._sanitise_generated_body(message if comment else rendered)
            comment = comment if comment else f"<!-- RFQ-ID: {rfq_id} -->"
            body = comment if not content else f"{comment}\n{content}"
            subject = f"{rfq_id} – Request for Quotation"

            draft_action_id = supplier.get("action_id") or default_action_id

            receiver = self._resolve_receiver(supplier, profile)
            recipients = self._normalise_recipients([receiver]) if receiver else []
            receiver = recipients[0] if recipients else receiver
            contact_level = 1 if receiver else 0

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
        output_data: Dict[str, Any] = {"drafts": drafts, "prompt": PROMPT_TEMPLATE}
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
    ) -> Dict:
        html_augmented = {
            f"{key}_html": self._format_plain_text(value)
            for key, value in base_args.items()
            if f"{key}_html" not in base_args
        }
        base_args.update(html_augmented)
        relationship = self._relationship_opening(supplier)
        summary = self._opportunity_summary(supplier, profile)
        evaluation_focus = self._build_evaluation_focus(supplier, profile)
        rfq_table = self._render_rfq_table(profile)
        if not base_args.get("deadline"):
            fallback = context.get("deadline", "TBC")
            base_args["deadline"] = fallback
            base_args["deadline_html"] = self._format_plain_text(fallback)
        else:
            base_args.setdefault(
                "deadline_html", self._format_plain_text(base_args.get("deadline"))
            )

        args = {
            "relationship_opening": relationship,
            "relationship_opening_html": self._format_plain_text(relationship),
            "opportunity_summary": summary,
            "opportunity_summary_html": self._format_plain_text(summary),
            "evaluation_focus": evaluation_focus,
            "evaluation_focus_html": self._format_focus_html(evaluation_focus),
            "rfq_table": rfq_table,
            "rfq_table_html": rfq_table,
            "contact_line_html": self._compose_contact_line(base_args),
        }
        args.setdefault("deadline", base_args.get("deadline"))
        args.setdefault("deadline_html", base_args.get("deadline_html", ""))
        return args

    def _relationship_opening(self, supplier: Dict) -> str:
        supplier_name = supplier.get("supplier_name") or "your organisation"
        spend = supplier.get("total_spend")
        po_count = supplier.get("po_count")
        try:
            spend_value = float(spend) if spend is not None else 0.0
        except Exception:
            spend_value = 0.0
        try:
            po_count_value = float(po_count) if po_count is not None else 0.0
        except Exception:
            po_count_value = 0.0
        if po_count_value > 0:
            return (
                f"Thank you for the continued collaboration with {supplier_name}. "
                "We are refreshing our sourcing position and would value an updated quotation from your team."
            )
        if spend_value > 0:
            return (
                f"Our records show meaningful recent engagement with {supplier_name}. "
                "We would appreciate a refreshed proposal aligned to the updated scope below."
            )
        return (
            "We are initiating a focused sourcing exercise for the scope outlined below and "
            f"would welcome {supplier_name}'s proposal."
        )

    def _opportunity_summary(self, supplier: Dict, profile: Dict) -> str:
        justification = supplier.get("justification")
        items = profile.get("items") or []
        categories = profile.get("categories") or {}
        category_focus = next(
            (
                values[0]
                for key, values in sorted(categories.items())
                if isinstance(values, list) and values
            ),
            None,
        )
        if items and category_focus:
            scope = ", ".join(items[:3])
            return (
                f"Our opportunity analysis highlights forthcoming demand in {category_focus}. "
                f"Please provide your best commercial and service position for {scope}."
            )
        if items:
            scope = ", ".join(items[:3])
            return (
                f"We are consolidating pricing for the following items: {scope}. "
                "Kindly confirm your latest offer and any value-added services."
            )
        if justification:
            return justification
        return (
            "We are conducting a market test and would appreciate a detailed quote "
            "covering pricing, lead times and support arrangements."
        )

    def _build_evaluation_focus(self, supplier: Dict, profile: Dict) -> str:
        focus_lines: List[str] = []
        weights = supplier.get("weights") or {}
        if isinstance(weights, dict) and weights:
            parts = []
            for key, value in weights.items():
                try:
                    parts.append(f"{key.title()}: {float(value):.0%}")
                except Exception:
                    parts.append(f"{key.title()}: {value}")
            focus_lines.append("Commercial weighting – " + ", ".join(parts))
        categories = profile.get("categories") or {}
        collected = []
        for level in sorted(categories.keys()):
            collected.extend(categories[level])
        unique_categories = [c for c in sorted({c for c in collected if c}) if c]
        if unique_categories:
            focus_lines.append(
                "Category scope – " + ", ".join(unique_categories[:3])
            )
        justification = supplier.get("justification")
        if justification and justification not in focus_lines:
            focus_lines.append(f"Opportunity driver – {justification}")
        if not focus_lines:
            focus_lines.append(
                "We will evaluate on total cost of ownership, responsiveness and compliance with policy obligations."
            )
        return "\n".join(f"- {line}" for line in focus_lines)

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

    def _format_focus_html(self, focus: str) -> str:
        if not focus:
            return "<p></p>"
        items = []
        for line in focus.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.startswith("- "):
                cleaned = cleaned[2:].strip()
            items.append(cleaned)
        if not items:
            return f"<p>{escape(focus)}</p>"
        return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"

    def _compose_contact_line(self, base_args: Dict[str, Any]) -> str:
        name = base_args.get("category_manager_name_html") or ""
        email = base_args.get("category_manager_email_html") or ""
        if name and email:
            return f"{name} ({email})"
        return name or email or ""

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

