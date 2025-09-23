import base64
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.email_credentials_manager import (
    CredentialsRotationError,
    SESSMTPAccessManager,
)


class DummyAgentNick:
    def __init__(self, **overrides):
        defaults = {
            "ses_smtp_iam_user": "ses-smtp-user.20250922-152128",
            "ses_smtp_secret_name": "ses/smtp/credentials",
            "ses_region": "eu-west-1",
            "ses_secret_role_arn": None,
        }
        defaults.update(overrides)
        self.settings = SimpleNamespace(**defaults)


def test_rotate_smtp_credentials_deletes_oldest_and_updates_secret():
    nick = DummyAgentNick()
    manager = SESSMTPAccessManager(nick)

    iam_client = MagicMock()
    iam_client.list_access_keys.return_value = {
        "AccessKeyMetadata": [
            {"AccessKeyId": "AKIAOLD", "CreateDate": datetime(2024, 1, 1)},
            {"AccessKeyId": "AKIAKEEP", "CreateDate": datetime(2024, 2, 1)},
        ]
    }
    iam_client.create_access_key.return_value = {
        "AccessKey": {"AccessKeyId": "AKIANEW", "SecretAccessKey": "supersecret"}
    }

    secrets_client = MagicMock()

    def client_side_effect(service_name, **kwargs):
        if service_name == "iam":
            return iam_client
        if service_name == "secretsmanager":
            assert kwargs.get("region_name") == "eu-west-1"
            return secrets_client
        raise AssertionError(f"Unexpected service requested: {service_name}")

    with patch("services.email_credentials_manager.boto3.client", side_effect=client_side_effect):
        rotated = manager.rotate_smtp_credentials()

    assert rotated.smtp_username == "AKIANEW"
    assert rotated.smtp_password
    iam_client.delete_access_key.assert_called_once_with(
        UserName="ses-smtp-user.20250922-152128",
        AccessKeyId="AKIAOLD",
    )
    secrets_client.put_secret_value.assert_called_once()
    _, kwargs = secrets_client.put_secret_value.call_args
    assert kwargs["SecretId"] == "ses/smtp/credentials"
    payload = json.loads(kwargs["SecretString"])
    assert payload["SMTP_USERNAME"] == "AKIANEW"
    assert payload["SMTP_PASSWORD"] == rotated.smtp_password


def test_rotate_smtp_credentials_uses_sts_when_role_configured():
    nick = DummyAgentNick(ses_secret_role_arn="arn:aws:iam::123456789012:role/test")
    manager = SESSMTPAccessManager(nick)

    iam_client = MagicMock()
    iam_client.list_access_keys.return_value = {"AccessKeyMetadata": []}
    iam_client.create_access_key.return_value = {
        "AccessKey": {"AccessKeyId": "AKIANEW", "SecretAccessKey": "supersecret"}
    }

    sts_client = MagicMock()
    sts_client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIAEXAMPLE",
            "SecretAccessKey": "secretkey",
            "SessionToken": "session",
        }
    }

    secrets_client = MagicMock()

    def client_side_effect(service_name, **kwargs):
        if service_name == "iam":
            return iam_client
        if service_name == "sts":
            assert kwargs.get("region_name") == "eu-west-1"
            return sts_client
        if service_name == "secretsmanager":
            assert kwargs.get("region_name") == "eu-west-1"
            assert kwargs.get("aws_access_key_id") == "AKIAEXAMPLE"
            assert kwargs.get("aws_secret_access_key") == "secretkey"
            assert kwargs.get("aws_session_token") == "session"
            return secrets_client
        raise AssertionError(f"Unexpected service requested: {service_name}")

    with patch("services.email_credentials_manager.boto3.client", side_effect=client_side_effect):
        manager.rotate_smtp_credentials()

    sts_client.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::123456789012:role/test",
        RoleSessionName="ProcWiseEmailSecrets",
    )
    secrets_client.put_secret_value.assert_called_once()


def test_rotate_smtp_credentials_infers_iam_user_when_not_configured():
    nick = DummyAgentNick(ses_smtp_iam_user=None)
    manager = SESSMTPAccessManager(nick)

    iam_client = MagicMock()
    iam_client.get_access_key_last_used.return_value = {
        "UserName": "ses-smtp-user.20250922-152128"
    }
    iam_client.list_access_keys.return_value = {
        "AccessKeyMetadata": [
            {"AccessKeyId": "AKIAOLD", "CreateDate": datetime(2024, 1, 1)}
        ]
    }
    iam_client.create_access_key.return_value = {
        "AccessKey": {"AccessKeyId": "AKIANEW", "SecretAccessKey": "supersecret"}
    }

    secrets_client = MagicMock()
    secrets_client.get_secret_value.return_value = {
        "SecretString": json.dumps({"SMTP_USERNAME": "AKIAOLD"})
    }

    def client_side_effect(service_name, **kwargs):
        if service_name == "iam":
            return iam_client
        if service_name == "secretsmanager":
            return secrets_client
        raise AssertionError(f"Unexpected service requested: {service_name}")

    with patch("services.email_credentials_manager.boto3.client", side_effect=client_side_effect):
        rotated = manager.rotate_smtp_credentials()

    iam_client.get_access_key_last_used.assert_called_once_with(AccessKeyId="AKIAOLD")
    iam_client.list_access_keys.assert_called_once_with(
        UserName="ses-smtp-user.20250922-152128"
    )
    assert rotated.smtp_username == "AKIANEW"


def test_rotate_smtp_credentials_errors_when_inference_fails():
    nick = DummyAgentNick(ses_smtp_iam_user=None)
    manager = SESSMTPAccessManager(nick)

    iam_client = MagicMock()
    secrets_client = MagicMock()
    secrets_client.get_secret_value.return_value = {"SecretString": "{}"}

    def client_side_effect(service_name, **kwargs):
        if service_name == "iam":
            return iam_client
        if service_name == "secretsmanager":
            return secrets_client
        raise AssertionError(f"Unexpected service requested: {service_name}")

    with patch("services.email_credentials_manager.boto3.client", side_effect=client_side_effect):
        with pytest.raises(CredentialsRotationError):
            manager.rotate_smtp_credentials()


def test_create_smtp_password_matches_sigv4_reference():
    nick = DummyAgentNick()
    manager = SESSMTPAccessManager(nick)

    password = manager._create_smtp_password(
        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "us-east-1"
    )

    assert password == "BOntiZFm/r+5s3psZ/RpsjB+aSGsj2J0rXdiLuO0cQL7"
    decoded = base64.b64decode(password.encode("utf-8"))
    assert decoded[0] == 0x04
