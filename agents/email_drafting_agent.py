"""Email Drafting Agent.

This module renders a procurement RFQ email based on a shared prompt
template so that both human authored and LLM generated messages follow
identical structure.  The prompt template lives in
``prompts/EmailDraftingAgent_Prompt_Template.json`` and the agent also
exposes this prompt in its output for downstream LLM usage.
"""

import json
import logging
from pathlib import Path

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
    """Agent that drafts an HTML RFQ email and sends it via SES."""

    HTML_TEMPLATE = """<html><body>
<p>Dear {supplier_contact_name},</p>
<p>We are requesting a quotation for the following items/services as part of our sourcing process.
Please complete the table in full to ensure your proposal can be evaluated accurately.</p>
<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">
<tr><th>Item ID</th><th>Description</th><th>UOM</th><th>Qty</th><th>Unit Price (Currency)</th><th>Extended Price</th><th>Delivery Lead Time (days)</th><th>Payment Terms (days)</th><th>Warranty / Support</th><th>Contract Ref</th><th>Comments</th></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</table>
<p>Please confirm one of the following options:</p>
<p>[ ] I accept Your Company’s Standard Terms &amp; Conditions (<a href=\"https://yourcompany.com/procurement-terms\">https://yourcompany.com/procurement-terms</a>)</p>
<p>[ ] I wish to proceed under my current contract with Your Company: Contract Number [__________]</p>
<p>Deadline for submission: {deadline}</p>
<p>Please return the completed table and confirmation via reply to this email. For queries, contact {category_manager_name}, {category_manager_title}, {category_manager_email}.</p>
<p>Thank you for your response. We look forward to your proposal.</p>
<p>Kind regards,<br/>{your_name}<br/>{your_title}<br/>{your_company}</p>
</body></html>"""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.email_service = EmailService(agent_nick)

    def run(self, context: AgentContext) -> AgentOutput:
        data = context.input_data
        subject = data.get(
            "subject", "Request for Quotation (RFQ) – Office Furniture"
        )
        recipient = data.get("recipient")
        sender = data.get("sender", self.agent_nick.settings.ses_default_sender)
        attachments = data.get("attachments")

        if not recipient:
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error="recipient not provided",
            )

        fmt_args = {
            "supplier_contact_name": data.get("supplier_contact_name", "Supplier"),
            "submission_deadline": data.get("submission_deadline", ""),
            "category_manager_name": data.get("category_manager_name", ""),
            "category_manager_title": data.get("category_manager_title", ""),
            "category_manager_email": data.get("category_manager_email", ""),
            "your_name": data.get("your_name", ""),
            "your_title": data.get("your_title", ""),
            "your_company": data.get("your_company", ""),
        }

        html_body = self.HTML_TEMPLATE.format(
            supplier_contact_name=fmt_args["supplier_contact_name"],
            deadline=fmt_args["submission_deadline"],
            category_manager_name=fmt_args["category_manager_name"],
            category_manager_title=fmt_args["category_manager_title"],
            category_manager_email=fmt_args["category_manager_email"],
            your_name=fmt_args["your_name"],
            your_title=fmt_args["your_title"],
            your_company=fmt_args["your_company"],
        )

        prompt = PROMPT_TEMPLATE.format(**fmt_args)

        success = self.email_service.send_email(
            subject, html_body, recipient, sender, attachments
        )
        status = AgentStatus.SUCCESS if success else AgentStatus.FAILED
        return AgentOutput(
            status=status,
            data={
                "subject": subject,
                "body": html_body,
                "prompt": prompt,
                "recipient": recipient,
                "sender": sender,
                "attachments": attachments,
            },
        )

