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
        data = dict(context.input_data)

        # Merge any structured output from previous agents so that extracted
        # variables can populate the email template automatically.
        prev = data.get("previous_agent_output")
        if isinstance(prev, dict):
            data = {**prev, **data}
        elif isinstance(prev, str):
            try:
                parsed = json.loads(prev)
                if isinstance(parsed, dict):
                    data = {**parsed, **data}
            except Exception:  # pragma: no cover - best effort
                logger.debug("previous_agent_output not JSON parsable")

        subject = data.get("subject")
        recipients = data.get("recipients") or data.get("recipient")
        if isinstance(recipients, str):
            recipients = [recipients]
        sender = data.get("sender", self.agent_nick.settings.ses_default_sender)
        attachments = data.get("attachments")

        draft_only = data.get("draft_only", False)
        if not recipients:
            recipients = None

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

        # Generate an LLM response based on upstream context if provided.  The
        # result is injected into the email body via the ``response`` variable.
        context_text = (
            data.get("context")
            or data.get("summary")
            or ""
        )

        # Generate a subject line if one was not supplied.
        DEFAULT_SUBJECT = "Request for Quotation (RFQ) – Office Furniture"
        if not subject:
            subject = DEFAULT_SUBJECT
            if context_text:
                try:
                    resp = self.call_ollama(
                        messages=[
                            {
                                "role": "system",
                                "content": "You write concise email subject lines for procurement RFQs.",
                            },
                            {
                                "role": "user",
                                "content": f"Generate a subject line for the following request:\n{context_text}",
                            },
                        ]
                    )
                    subject = resp.get("response", "").strip() or DEFAULT_SUBJECT
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed to generate subject via LLM")
        llm_response = ""
        if context_text:
            try:
                resp = self.call_ollama(
                    messages=[
                        {"role": "system", "content": "You are a helpful procurement assistant."},
                        {"role": "user", "content": context_text},
                    ]
                )
                llm_response = resp.get("response", "").strip()
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to generate email body via LLM")

        # Allow a custom body template to be supplied via input data. When no
        # template is provided we fall back to the default TEXT_TEMPLATE.
        body_template = data.get("body") or self.TEXT_TEMPLATE
        template_args = {
            **data,
            **fmt_args,
            "deadline": fmt_args["submission_deadline"],
            "response": llm_response,
        }
        text_body = Template(body_template).render(**template_args)
        if llm_response and "{{ response }}" not in body_template:
            text_body = f"{llm_response}\n\n" + text_body

        prompt = PROMPT_TEMPLATE.format(**fmt_args)

        sent = False
        if not draft_only and recipients:
            try:
                sent = self.email_service.send_email(
                    subject, text_body, recipients, sender, attachments
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to send email")
        if recipients is None:
            message = "recipient not provided"
        else:
            message = "email sent" if sent else "email drafted"

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "subject": subject,
                "body": text_body,
                "prompt": prompt,
                "recipients": recipients,
                "sender": sender,
                "attachments": attachments,
                "sent": sent,
                "message": message,
                "response": llm_response,
            },
        )

