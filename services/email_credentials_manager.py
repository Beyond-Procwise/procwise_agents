"""Utilities for rotating SES SMTP credentials and updating Secrets Manager."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from utils.gpu import configure_gpu

configure_gpu()


class CredentialsRotationError(RuntimeError):
    """Raised when SES SMTP credential rotation fails."""


@dataclass
class RotatedCredentials:
    """Value object holding newly rotated SMTP credentials."""

    smtp_username: str
    smtp_password: str
    access_key_id: str


class SESSMTPAccessManager:
    """Manage SES SMTP access keys stored in AWS Secrets Manager."""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.logger = logging.getLogger(__name__)

    def rotate_smtp_credentials(self) -> RotatedCredentials:
        """Rotate the SES SMTP IAM user's credentials and persist them."""

        secret_name = getattr(self.settings, "ses_smtp_secret_name", None)
        if not secret_name:
            raise CredentialsRotationError("SES SMTP secret name is not configured")

        region = getattr(self.settings, "ses_region", None) or "eu-west-1"

        iam_client = boto3.client("iam")
        secrets_client = self._secrets_manager_client(region)

        iam_user = getattr(self.settings, "ses_smtp_iam_user", None)
        if not iam_user:
            iam_user = self._infer_iam_user(secrets_client, secret_name, iam_client)
            if not iam_user:
                raise CredentialsRotationError("SES SMTP IAM user is not configured")

        try:
            response = iam_client.list_access_keys(UserName=iam_user)
            keys = response.get("AccessKeyMetadata", [])
        except ClientError as exc:  # pragma: no cover - boto3 error handling
            raise CredentialsRotationError(
                "Failed to list SES SMTP IAM access keys"
            ) from exc

        if len(keys) >= 2:
            oldest_key = sorted(keys, key=lambda item: item.get("CreateDate"))[0]
            oldest_key_id = oldest_key.get("AccessKeyId")
            if oldest_key_id:
                try:
                    iam_client.delete_access_key(
                        UserName=iam_user,
                        AccessKeyId=oldest_key_id,
                    )
                    self.logger.info(
                        "Deleted oldest SES SMTP access key %s for user %s",
                        oldest_key_id,
                        iam_user,
                    )
                except ClientError as exc:  # pragma: no cover - boto3 error handling
                    raise CredentialsRotationError(
                        f"Failed to delete SES SMTP access key {oldest_key_id}"
                    ) from exc

        try:
            new_key = iam_client.create_access_key(UserName=iam_user)["AccessKey"]
        except ClientError as exc:  # pragma: no cover - boto3 error handling
            raise CredentialsRotationError(
                "Failed to create a new SES SMTP access key"
            ) from exc

        access_key_id = new_key.get("AccessKeyId")
        secret_access_key = new_key.get("SecretAccessKey")
        if not access_key_id or not secret_access_key:
            raise CredentialsRotationError(
                "IAM did not return a complete set of credentials"
            )

        smtp_password = self._create_smtp_password(secret_access_key, region)

        secret_value = json.dumps(
            {
                "SMTP_USERNAME": access_key_id,
                "SMTP_PASSWORD": smtp_password,
                "IAM_USER": iam_user,
            }
        )

        try:
            secrets_client.put_secret_value(
                SecretId=secret_name,
                SecretString=secret_value,
            )
        except ClientError as exc:  # pragma: no cover - boto3 error handling
            raise CredentialsRotationError(
                f"Failed to update secret {secret_name} with rotated credentials"
            ) from exc

        self.logger.info(
            "Rotated SES SMTP credentials for user %s and updated secret %s",
            iam_user,
            secret_name,
        )

        return RotatedCredentials(
            smtp_username=access_key_id,
            smtp_password=smtp_password,
            access_key_id=access_key_id,
        )

    def _create_smtp_password(self, secret_access_key: str, region: str) -> str:
        """Convert an IAM secret access key into an SES SMTP password."""

        def _sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        signing_key = ("AWS4" + secret_access_key).encode("utf-8")
        date_key = _sign(signing_key, "11111111")
        region_key = _sign(date_key, region)
        service_key = _sign(region_key, "ses")
        terminal_key = _sign(service_key, "aws4_request")
        signature = _sign(terminal_key, "SendRawEmail")

        return base64.b64encode(b"\x04" + signature).decode()

    def _infer_iam_user(self, secrets_client, secret_name: str, iam_client) -> str | None:
        """Infer the SES SMTP IAM user from the stored credentials when absent."""

        try:
            secret_value = secrets_client.get_secret_value(
                SecretId=secret_name,
                VersionStage="AWSCURRENT",
            )
        except ClientError as exc:  # pragma: no cover - boto3 error handling
            self.logger.warning(
                "Unable to retrieve SES SMTP secret %s to infer IAM user: %s",
                secret_name,
                exc,
            )
            return None

        secret_string = secret_value.get("SecretString")
        if not secret_string:
            self.logger.warning(
                "SES SMTP secret %s does not contain a SecretString payload", secret_name
            )
            return None

        try:
            payload = json.loads(secret_string)
        except json.JSONDecodeError:
            self.logger.warning(
                "SES SMTP secret %s payload is not valid JSON; cannot infer IAM user",
                secret_name,
            )
            return None

        metadata_keys = (
            "IAM_USER",
            "iam_user",
            "IAMUser",
            "user",
            "USER",
        )

        for key in metadata_keys:
            candidate = payload.get(key)
            if candidate:
                candidate_str = str(candidate).strip()
                if candidate_str:
                    self.logger.info(
                        "Inferred SES SMTP IAM user %s from secret %s field %s",
                        candidate_str,
                        secret_name,
                        key,
                    )
                    return candidate_str

        username = (
            payload.get("SMTP_USERNAME")
            or payload.get("smtp_username")
            or payload.get("username")
        )
        if not username:
            self.logger.warning(
                "SES SMTP secret %s does not include an SMTP username; cannot infer IAM user",
                secret_name,
            )
            return None

        access_key = str(username).strip()
        if not access_key:
            self.logger.warning(
                "SES SMTP secret %s includes an empty SMTP username; cannot infer IAM user",
                secret_name,
            )
            return None

        try:
            response = iam_client.get_access_key_last_used(AccessKeyId=access_key)
        except ClientError as exc:  # pragma: no cover - boto3 error handling
            error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
            error_code = error.get("Code")
            if error_code == "AccessDenied":
                self.logger.warning(
                    "Access denied when attempting to infer IAM user for access key %s via GetAccessKeyLastUsed; configure SES_SMTP_IAM_USER or store IAM_USER in the secret",
                    access_key,
                )
            else:
                self.logger.warning(
                    "Unable to look up IAM user for access key %s via GetAccessKeyLastUsed: %s",
                    access_key,
                    error_code or exc,
                )
            return None

        inferred_user = response.get("UserName")
        if not inferred_user:
            self.logger.warning(
                "IAM did not return a user name when looking up access key %s", access_key
            )
            return None

        self.logger.info(
            "Inferred SES SMTP IAM user %s from secret %s via GetAccessKeyLastUsed",
            inferred_user,
            secret_name,
        )
        return inferred_user

    def _secrets_manager_client(self, region: str):
        """Create a Secrets Manager client, assuming a role when configured."""

        role_arn = getattr(self.settings, "ses_secret_role_arn", None)
        if not role_arn:
            return boto3.client("secretsmanager", region_name=region)

        self.logger.debug(
            "Assuming role %s to update SES SMTP credentials", role_arn
        )

        sts_kwargs = {"region_name": region} if region else {}
        sts_client = boto3.client("sts", **sts_kwargs)

        try:
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="ProcWiseEmailSecrets",
            )
        except ClientError as exc:  # pragma: no cover - boto3 error handling
            raise CredentialsRotationError(
                f"Unable to assume role {role_arn} for SES secret access"
            ) from exc

        credentials = response.get("Credentials")
        if not credentials:
            raise CredentialsRotationError("STS did not return temporary credentials")

        access_key = credentials.get("AccessKeyId")
        secret_key = credentials.get("SecretAccessKey")
        session_token = credentials.get("SessionToken")
        if not all([access_key, secret_key, session_token]):
            raise CredentialsRotationError("Incomplete credentials received from STS")

        return boto3.client(
            "secretsmanager",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
