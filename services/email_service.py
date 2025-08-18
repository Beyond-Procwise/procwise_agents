import logging
import smtplib
from email.mime.text import MIMEText


class EmailService:
    """Service responsible for sending emails via Amazon SES."""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.logger = logging.getLogger(__name__)

    def send_email(self, subject: str, body: str, recipient: str, sender: str) -> bool:
        """Send an email using SMTP credentials configured in settings.

        Returns True on success, False otherwise.
        """
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        try:
            with smtplib.SMTP(self.settings.ses_smtp_endpoint, self.settings.ses_smtp_port) as server:
                server.starttls()
                server.login(self.settings.ses_smtp_user, self.settings.ses_smtp_password)
                server.sendmail(sender, [recipient], msg.as_string())
            return True
        except Exception as exc:  # pragma: no cover - network/runtime
            self.logger.error("Email send failed: %s", exc)
            return False
