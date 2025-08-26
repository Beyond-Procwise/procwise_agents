import logging
import smtplib
from typing import List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from utils.gpu import configure_gpu

configure_gpu()


class EmailService:
    """Service responsible for sending emails via Amazon SES."""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.logger = logging.getLogger(__name__)

    def send_email(
        self,
        subject: str,
        body: str,
        recipient: str,
        sender: str,
        attachments: Optional[List[Tuple[bytes, str]]] = None,
    ) -> bool:
        """Send an email using SMTP credentials configured in settings.

        Attachments should be provided as a list of ``(content, filename)`` tuples.
        Returns True on success, False otherwise.
        """

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.attach(MIMEText(body, "html"))

        if attachments:
            for content, filename in attachments:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(content)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename=\"{filename}\"",
                )
                msg.attach(part)

        try:
            with smtplib.SMTP(
                self.settings.ses_smtp_endpoint, self.settings.ses_smtp_port
            ) as server:
                server.starttls()
                server.login(
                    self.settings.ses_smtp_user, self.settings.ses_smtp_password
                )
                server.sendmail(sender, [recipient], msg.as_string())
            return True
        except Exception as exc:  # pragma: no cover - network/runtime
            self.logger.error("Email send failed: %s", exc)
            return False
