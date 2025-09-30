import json
import logging
import smtplib
import ssl
import time
from dataclasses import dataclass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from typing import Dict, Iterable, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError

from utils.gpu import configure_gpu

from .email_credentials_manager import (
    CredentialsRotationError,
    SESSMTPAccessManager,
)


@dataclass
class EmailSendResult:
    """Outcome of an SES SMTP send attempt."""

    success: bool
    message_id: Optional[str] = None


configure_gpu()


class EmailService:
    """Service responsible for sending emails via Amazon SES."""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.logger = logging.getLogger(__name__)
        self.credential_manager = SESSMTPAccessManager(agent_nick)
        self.smtp_propagation_attempts = self._coerce_int_setting(
            getattr(self.settings, "ses_smtp_propagation_attempts", 6),
            default=6,
            minimum=1,
        )
        self.smtp_propagation_wait_seconds = self._coerce_int_setting(
            getattr(self.settings, "ses_smtp_propagation_wait_seconds", 30),
            default=30,
            minimum=0,
        )

    def send_email(
        self,
        subject: str,
        body: str,
        recipients: Union[str, Iterable[str]],
        sender: str,
        attachments: Optional[List[Tuple[bytes, str]]] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        message_id: Optional[str] = None,
    ) -> EmailSendResult:
        """Send an email using SMTP credentials configured in settings.

        Attachments should be provided as a list of ``(content, filename)`` tuples.
        ``headers`` allows callers to inject additional RFC-2822 headers such as
        ``X-Procwise-RFQ-ID`` while ``message_id`` can be provided to guarantee a
        stable value across retries.
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

        generated_message_id = message_id or make_msgid(domain=self._infer_domain(sender))
        msg["Message-ID"] = generated_message_id
        msg["Date"] = formatdate(localtime=True)

        if headers:
            for key, value in headers.items():
                if key and value is not None:
                    msg[str(key)] = str(value)

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
            return EmailSendResult(False, generated_message_id)

        message_payload = msg.as_string()

        try:
            self._deliver_via_smtp(
                message_payload, sender, recipient_list, smtp_username, smtp_password
            )

            return EmailSendResult(True, generated_message_id)
        except smtplib.SMTPAuthenticationError as auth_exc:
            self.logger.warning(
                "SMTP authentication failed with current credentials; attempting fallback",
                exc_info=auth_exc,
            )
            try:
                fallback_username, fallback_password = self._fetch_smtp_credentials(
                    version_stage="AWSPREVIOUS"
                )
            except Exception as fallback_exc:
                self.logger.error(
                    "Unable to retrieve fallback SMTP credentials: %s", fallback_exc
                )
                success = self._attempt_rotation_and_resend(
                    message_payload,
                    sender,
                    recipient_list,
                )
                return EmailSendResult(success, generated_message_id)

            try:
                self._deliver_via_smtp(
                    message_payload,
                    sender,
                    recipient_list,
                    fallback_username,
                    fallback_password,
                )
                return EmailSendResult(True, generated_message_id)
            except smtplib.SMTPAuthenticationError as fallback_auth_exc:
                self.logger.error(
                    "Fallback SES SMTP credentials failed authentication: %s",
                    fallback_auth_exc,
                )
                success = self._attempt_rotation_and_resend(
                    message_payload,
                    sender,
                    recipient_list,
                )
                return EmailSendResult(success, generated_message_id)
            except Exception as fallback_exc:  # pragma: no cover - defensive
                self.logger.error("Email send failed: %s", fallback_exc)
                return EmailSendResult(False, generated_message_id)
        except Exception as exc:  # pragma: no cover - network/runtime
            self.logger.error("Email send failed: %s", exc)
            return EmailSendResult(False, generated_message_id)

    def _attempt_rotation_and_resend(
        self,
        message_payload: str,
        sender: str,
        recipient_list: List[str],
    ) -> bool:
        """Rotate SMTP credentials and retry delivery once."""

        self.logger.info(
            "Attempting to rotate SES SMTP credentials after repeated authentication failures"
        )

        try:
            rotated = self.credential_manager.rotate_smtp_credentials()
        except CredentialsRotationError as exc:
            self.logger.error("SES SMTP credential rotation failed: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                "Unexpected error while rotating SES SMTP credentials: %s", exc
            )
            return False

        try:
            confirmed_username, confirmed_password = self._fetch_smtp_credentials()
        except Exception as exc:  # pragma: no cover - network/runtime
            self.logger.warning(
                "Unable to confirm rotated SES SMTP credentials from Secrets Manager: %s",
                exc,
            )
            confirmed_username = rotated.smtp_username
            confirmed_password = rotated.smtp_password

        username = confirmed_username or rotated.smtp_username
        password = confirmed_password or rotated.smtp_password

        if not username or not password:
            self.logger.error("Rotated SES SMTP credentials are incomplete")
            return False

        return self._deliver_with_propagation_retry(
            message_payload,
            sender,
            recipient_list,
            username,
            password,
        )

    def _deliver_with_propagation_retry(
        self,
        message_payload: str,
        sender: str,
        recipient_list: List[str],
        smtp_username: str,
        smtp_password: str,
    ) -> bool:
        """Attempt delivery with rotated credentials, retrying for propagation."""

        attempts = max(1, int(self.smtp_propagation_attempts))
        wait_seconds = max(0, int(self.smtp_propagation_wait_seconds))

        for attempt in range(1, attempts + 1):
            try:
                self._deliver_via_smtp(
                    message_payload,
                    sender,
                    recipient_list,
                    smtp_username,
                    smtp_password,
                )
                return True
            except smtplib.SMTPAuthenticationError as auth_exc:
                if attempt >= attempts:
                    self.logger.error(
                        "Email send failed even after rotating SES SMTP credentials: %s",
                        auth_exc,
                    )
                    return False

                self.logger.warning(
                    "SMTP authentication failed using rotated SES credentials (attempt %s/%s); retrying after %ss",
                    attempt,
                    attempts,
                    wait_seconds,
                )
                if wait_seconds:
                    time.sleep(wait_seconds)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error(
                    "Email send failed after rotating SES SMTP credentials: %s",
                    exc,
                )
                return False

        return False

    def _fetch_smtp_credentials(self, *, version_stage: str = "AWSCURRENT") -> Tuple[str, str]:
        """Retrieve SMTP credentials from AWS Secrets Manager.

        Parameters
        ----------
        version_stage:
            The Secrets Manager version stage to request. Defaults to
            ``"AWSCURRENT"`` but can be overridden (e.g. ``"AWSPREVIOUS"``)
            for retry scenarios during credential rotation.
        """


        secret_name = getattr(self.settings, "ses_smtp_secret_name", None)
        if not secret_name:
            raise ValueError("SES SMTP secret name is not configured")

        region = getattr(self.settings, "ses_region", None) or "eu-west-1"

        client = self._secrets_manager_client(region)
        try:
            get_kwargs = {"SecretId": secret_name}
            if version_stage:
                get_kwargs["VersionStage"] = version_stage
            secret_value = client.get_secret_value(**get_kwargs)

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

        username = (
            payload.get("SMTP_USERNAME")
            or payload.get("smtp_username")
            or payload.get("username")
        )
        password = (
            payload.get("SMTP_PASSWORD")
            or payload.get("smtp_password")
            or payload.get("password")
        )
        if not username or not password:
            raise ValueError(
                f"Secret payload missing SMTP credentials for stage {version_stage}"
            )

        username_str = str(username).strip()
        password_str = str(password).strip()
        if not username_str or not password_str:
            raise ValueError(
                f"Secret payload missing SMTP credentials for stage {version_stage}"
            )

        return username_str, password_str

    def _deliver_via_smtp(
        self,
        message_payload: str,
        sender: str,
        recipient_list: List[str],
        smtp_username: str,
        smtp_password: str,
    ) -> None:
        """Send the prepared email payload using the provided credentials."""

        with smtplib.SMTP(
            self.settings.ses_smtp_endpoint, self.settings.ses_smtp_port
        ) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender, recipient_list, message_payload)


    @staticmethod
    def _coerce_int_setting(value, *, default: int, minimum: int) -> int:
        """Coerce integer-like configuration values while enforcing bounds."""

        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default

        return coerced if coerced >= minimum else minimum

    @staticmethod
    def _infer_domain(sender: str) -> Optional[str]:
        """Best-effort extraction of the sender's domain for Message-ID generation."""

        if not sender:
            return None
        if "@" not in sender:
            return None
        domain = sender.split("@", 1)[-1].strip()
        return domain or None

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
