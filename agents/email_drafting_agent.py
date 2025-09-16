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
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Template

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_service import EmailService
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

RFQ_TABLE_HEADER = (
    "Item Description | UOM | Qty | Target Unit Price (Currency) | Extended Price | Delivery Lead Time (days) | Payment Terms (days) | Warranty / Support | Contract Ref | Comments\n"
    "---------------- | --- | --- | --------------------------- | -------------- | ------------------------- | -------------------- | ----------------- | ------------ | --------"
)



class EmailDraftingAgent(BaseAgent):
    """Agent that drafts a plain-text RFQ email and sends it via SES."""

    TEXT_TEMPLATE = (
        "Dear {supplier_contact_name},\n\n"
        "{relationship_opening}\n\n"
        "{opportunity_summary}\n\n"
        "To support a fair evaluation please complete the table below with your latest commercial and service offer.\n\n"
        "{rfq_table}\n\n"
        "Evaluation focus:\n"
        "{evaluation_focus}\n\n"
        "Deadline for submission: {deadline}\n"
        "Primary contact: {category_manager_name} ({category_manager_email})\n\n"
        "Kind regards,\n"
        "{your_name}\n"
        "{your_title}\n"
        "{your_company}\n"
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.email_service = EmailService(agent_nick)
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

        for supplier in ranking:
            supplier_id = supplier.get("supplier_id")
            supplier_name = supplier.get("supplier_name", supplier_id)
            rfq_id = f"RFQ-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

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
            body_template = data.get("body") or self.TEXT_TEMPLATE
            contextual_args = self._build_template_args(
                supplier, profile, fmt_args, data
            )
            template_args = {**data, **fmt_args, **contextual_args}
            if "{{" in body_template or "{%" in body_template:
                body = Template(body_template).render(**template_args)
            else:
                try:
                    body = body_template.format(**template_args)
                except KeyError:
                    body = body_template
            body = f"<!-- RFQ-ID: {rfq_id} -->\n" + body
            subject = f"RFQ {rfq_id} – Request for Quotation"

            draft_action_id = supplier.get("action_id") or default_action_id

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
            }
            if draft_action_id:
                draft["action_id"] = draft_action_id
            drafts.append(draft)
            self._store_draft(draft)
            logger.debug("EmailDraftingAgent created draft %s for supplier %s", rfq_id, supplier_id)

        logger.info("EmailDraftingAgent generated %d drafts", len(drafts))
        output_data = {"drafts": drafts}
        if default_action_id:
            output_data["action_id"] = default_action_id

        pass_fields = {"drafts": drafts}
        if default_action_id:
            pass_fields["action_id"] = default_action_id

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=output_data,
            pass_fields=pass_fields,
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
    ) -> Dict:
        relationship = self._relationship_opening(supplier)
        summary = self._opportunity_summary(supplier, profile)
        evaluation_focus = self._build_evaluation_focus(supplier, profile)
        rfq_table = self._render_rfq_table(profile)
        args = {
            "relationship_opening": relationship,
            "opportunity_summary": summary,
            "evaluation_focus": evaluation_focus,
            "rfq_table": rfq_table,
        }
        if not base_args.get("deadline"):
            args.setdefault("deadline", context.get("deadline", "TBC"))
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
            "We are engaging leading suppliers for an upcoming procurement exercise and "
            f"would like {supplier_name} to participate."
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
        lines = [RFQ_TABLE_HEADER]
        for description in items:
            lines.append(
                f"{description} |  |  |  |  |  |  |  |  | "
            )
        return "\n".join(lines)

    def _store_draft(self, draft: dict) -> None:
        """Persist email draft to ``proc.draft_rfq_emails``."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                self._ensure_table_exists(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.draft_rfq_emails
                        (rfq_id, supplier_id, supplier_name, subject, body, created_on, sent)
                        VALUES (%s, %s, %s, %s, %s, NOW(), FALSE)
                        """,
                        (
                            draft["rfq_id"],
                            draft["supplier_id"],
                            draft.get("supplier_name"),
                            draft["subject"],
                            draft["body"],
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
                        sent BOOLEAN NOT NULL DEFAULT FALSE
                    )
                    """
                )
                cur.execute(
                    "ALTER TABLE proc.draft_rfq_emails ADD COLUMN IF NOT EXISTS supplier_name TEXT"
                )


            self._draft_table_checked = True

