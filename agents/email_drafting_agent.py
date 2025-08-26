import logging

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_service import EmailService
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

configure_gpu()


class EmailDraftingAgent(BaseAgent):
    """Agent that drafts an HTML RFQ email and sends it via SES."""

    TEMPLATE = """<html><body>
<p>Dear {supplier_contact_name},</p>
<p>We are requesting a quotation for the following items/services as part of our sourcing process.
Please complete the table in full to ensure your proposal can be evaluated accurately.</p>
<table border=\"1\" cellpadding=\"5\" cellspacing=\"0\">
<tr><th>Item ID</th><th>Description</th><th>UOM</th><th>Qty</th><th>Unit Price (Currency)</th><th>Extended Price</th><th>Delivery Lead Time (days)</th><th>Payment Terms (days)</th><th>Warranty / Support</th><th>Contract Ref</th><th>Comments</th></tr>
<tr><td>1</td><td>Office Desk (120cm)</td><td>Each</td><td>50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>2</td><td>Ergonomic Chair</td><td>Each</td><td>50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>3</td><td>Filing Cabinet (2dr)</td><td>Each</td><td>20</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</table>
<p>Please confirm one of the following options:</p>
<p>[ ] I accept Your Company’s Standard Terms &amp; Conditions
(<a href=\"https://yourcompany.com/procurement-terms\">https://yourcompany.com/procurement-terms</a>)</p>
<p>[ ] I wish to proceed under my current contract with Your Company:
Contract Number [__________]</p>
<p>Deadline for submission: {deadline}</p>
<p>Please return the completed table and confirmation via reply to this email.
For queries, contact {category_manager_name}, {category_manager_title}, {category_manager_email}.</p>
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

        html_body = self.TEMPLATE.format(
            supplier_contact_name=data.get("supplier_contact_name", "Supplier"),
            deadline=data.get("submission_deadline", ""),
            category_manager_name=data.get("category_manager_name", ""),
            category_manager_title=data.get("category_manager_title", ""),
            category_manager_email=data.get("category_manager_email", ""),
            your_name=data.get("your_name", ""),
            your_title=data.get("your_title", ""),
            your_company=data.get("your_company", ""),
        )

        success = self.email_service.send_email(
            subject, html_body, recipient, sender, attachments
        )
        status = AgentStatus.SUCCESS if success else AgentStatus.FAILED
        return AgentOutput(
            status=status,
            data={
                "subject": subject,
                "body": html_body,
                "recipient": recipient,
                "sender": sender,
                "attachments": attachments,
            },
        )

