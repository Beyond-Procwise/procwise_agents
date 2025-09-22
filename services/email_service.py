import json
import logging
import smtplib
from typing import Iterable, List, Optional, Tuple, Union
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import boto3
from botocore.exceptions import ClientError

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
        recipients: Union[str, Iterable[str]],
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
        if isinstance(recipients, str):
            recipient_list = [recipients]
        else:
            recipient_list = list(recipients)
        msg["To"] = ", ".join(recipient_list)
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
            smtp_username, smtp_password = self._fetch_smtp_credentials()
        except Exception as exc:
            self.logger.error("Unable to retrieve SMTP credentials: %s", exc)
            return False

        try:
            with smtplib.SMTP(
                self.settings.ses_smtp_endpoint, self.settings.ses_smtp_port
            ) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.sendmail(sender, recipient_list, msg.as_string())
            return True
        except Exception as exc:  # pragma: no cover - network/runtime
            self.logger.error("Email send failed: %s", exc)
            return False

    def _fetch_smtp_credentials(self) -> Tuple[str, str]:
        """Retrieve freshly rotated SMTP credentials from AWS Secrets Manager."""

        secret_name = getattr(self.settings, "ses_smtp_secret_name", None)
        if not secret_name:
            raise ValueError("SES SMTP secret name is not configured")

        region = getattr(self.settings, "ses_region", None) or "eu-west-1"

        client = self._secrets_manager_client(region)
        try:
            secret_value = client.get_secret_value(SecretId=secret_name)
        except ClientError as exc:
            raise RuntimeError(
                f"Failed to retrieve SES SMTP secret '{secret_name}'"
            ) from exc
        secret_string = secret_value.get("SecretString")
        if not secret_string:
            raise ValueError("Secret does not contain a SecretString payload")

        try:
            payload = json.loads(secret_string)
        except json.JSONDecodeError as exc:
            raise ValueError("Secret payload is not valid JSON") from exc

        username = payload.get("SMTP_USERNAME")
        password = payload.get("SMTP_PASSWORD")
        if not username or not password:
            raise ValueError("Secret payload missing SMTP credentials")

        return username, password

    def _secrets_manager_client(self, region: str):
        """Create a Secrets Manager client, assuming a role when configured."""

        role_arn = getattr(self.settings, "ses_secret_role_arn", None)
        if not role_arn:
            return boto3.client("secretsmanager", region_name=region)

        self.logger.debug(
            "Assuming role %s to access SES SMTP credentials", role_arn
        )
        sts_kwargs = {"region_name": region} if region else {}
        sts_client = boto3.client("sts", **sts_kwargs)
        try:
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="ProcWiseEmailSecrets",
            )
        except ClientError as exc:
            raise RuntimeError(
                f"Unable to assume role {role_arn} for SES secret access"
            ) from exc

        credentials = response.get("Credentials")
        if not credentials:
            raise ValueError("STS did not return temporary credentials")

        access_key = credentials.get("AccessKeyId")
        secret_key = credentials.get("SecretAccessKey")
        session_token = credentials.get("SessionToken")
        if not all([access_key, secret_key, session_token]):
            raise ValueError("Incomplete credentials received from STS")

        return boto3.client(
            "secretsmanager",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
