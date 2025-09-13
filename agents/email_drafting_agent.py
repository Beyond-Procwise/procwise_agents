"""Email Drafting Agent.

This module renders a procurement RFQ email based on a shared prompt
template so that both human authored and LLM generated messages follow
identical structure.  The prompt template lives in
``prompts/EmailDraftingAgent_Prompt_Template.json`` and the agent also
exposes this prompt in its output for downstream LLM usage.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

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


class EmailDraftingAgent(BaseAgent):
    """Agent that drafts a plain-text RFQ email and sends it via SES."""

    TEXT_TEMPLATE = (
        "Dear {supplier_contact_name},\n\n"
        "We are requesting a quotation for the following items/services as part of our sourcing process.\n"
        "Please complete the table in full to ensure your proposal can be evaluated accurately.\n\n"
        "Item ID | Description | UOM | Qty | Unit Price (Currency) | Extended Price | Delivery Lead Time (days) | Payment Terms (days) | Warranty / Support | Contract Ref | Comments\n"
        "------- | ----------- | --- | --- | --------------------- | -------------- | ------------------------- | ------------------- | ----------------- | ------------ | --------\n"
        "\n"
        "Please confirm one of the following options:\n"
        "[ ] I accept Your Company’s Standard Terms & Conditions (https://yourcompany.com/procurement-terms)\n"
        "[ ] I wish to proceed under my current contract with Your Company: Contract Number [__________]\n"
        "Deadline for submission: {deadline}\n"
        "Please return the completed table and confirmation via reply to this email. For queries, contact {category_manager_name}, {category_manager_title}, {category_manager_email}.\n"
        "Thank you for your response. We look forward to your proposal.\n\n"
        "Kind regards,\n"
        "{your_name}\n"
        "{your_title}\n"
        "{your_company}\n"
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.email_service = EmailService(agent_nick)

    def run(self, context: AgentContext) -> AgentOutput:
        """Draft RFQ emails for each ranked supplier without sending."""
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
        drafts = []

        for supplier in ranking:
            supplier_id = supplier.get("supplier_id")
            supplier_name = supplier.get("supplier_name", supplier_id)
            rfq_id = f"RFQ-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

            fmt_args = {
                "supplier_contact_name": supplier_name or "Supplier",
                "submission_deadline": data.get("submission_deadline", ""),
                "category_manager_name": data.get("category_manager_name", ""),
                "category_manager_title": data.get("category_manager_title", ""),
                "category_manager_email": data.get("category_manager_email", ""),
                "your_name": data.get("your_name", ""),
                "your_title": data.get("your_title", ""),
                "your_company": data.get("your_company", ""),
            }
            body_template = data.get("body") or self.TEXT_TEMPLATE
            template_args = {**data, **fmt_args}
            body = Template(body_template).render(**template_args)
            body = f"<!-- RFQ-ID: {rfq_id} -->\n" + body
            subject = f"RFQ {rfq_id} – Request for Quotation"

            draft = {
                "supplier_id": supplier_id,
                "rfq_id": rfq_id,
                "subject": subject,
                "body": body,
            }
            drafts.append(draft)
            self._store_draft(draft)

        return AgentOutput(status=AgentStatus.SUCCESS, data={"drafts": drafts})

    def _store_draft(self, draft: dict) -> None:
        """Persist email draft to ``proc.draft_rfq_emails``."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.draft_rfq_emails
                        (rfq_id, supplier_id, subject, body, created_on, sent)
                        VALUES (%s, %s, %s, %s, NOW(), FALSE)
                        """,
                        (
                            draft["rfq_id"],
                            draft["supplier_id"],
                            draft["subject"],
                            draft["body"],
                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store RFQ draft")

