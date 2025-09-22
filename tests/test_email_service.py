import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.email_service import EmailService


class DummyAgentNick:
    def __init__(self, **overrides):
        defaults = {
            "ses_smtp_secret_name": "ses/smtp/credentials",
            "ses_region": "eu-west-1",
            "ses_smtp_endpoint": "email-smtp.eu-west-1.amazonaws.com",
            "ses_smtp_port": 587,
            "ses_secret_role_arn": None,
        }
        defaults.update(overrides)
        self.settings = SimpleNamespace(**defaults)


def _secret_payload(username: str = "user", password: str = "pass") -> str:
    return json.dumps({"SMTP_USERNAME": username, "SMTP_PASSWORD": password})


def test_fetch_smtp_credentials_without_role_uses_direct_client():
    nick = DummyAgentNick()
    service = EmailService(nick)

    secrets_client = MagicMock()
    secrets_client.get_secret_value.return_value = {"SecretString": _secret_payload()}

    with patch("services.email_service.boto3.client", return_value=secrets_client) as client_mock:
        username, password = service._fetch_smtp_credentials()

    assert username == "user"
    assert password == "pass"
    client_mock.assert_called_once_with("secretsmanager", region_name="eu-west-1")
    secrets_client.get_secret_value.assert_called_once_with(
        SecretId="ses/smtp/credentials"
    )


def test_fetch_smtp_credentials_with_role_assumes_and_uses_temporary_creds():
    nick = DummyAgentNick(ses_secret_role_arn="arn:aws:iam::123456789012:role/test")
    service = EmailService(nick)

    secrets_client = MagicMock()
    secrets_client.get_secret_value.return_value = {"SecretString": _secret_payload("rotated", "secret")}

    sts_client = MagicMock()
    sts_client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "AKIAEXAMPLE",
            "SecretAccessKey": "secretkey",
            "SessionToken": "sessiontoken",
        }
    }

    def client_side_effect(service_name, **kwargs):
        if service_name == "sts":
            assert kwargs.get("region_name") == "eu-west-1"
            return sts_client
        if service_name == "secretsmanager":
            assert kwargs.get("region_name") == "eu-west-1"
            assert kwargs.get("aws_access_key_id") == "AKIAEXAMPLE"
            assert kwargs.get("aws_secret_access_key") == "secretkey"
            assert kwargs.get("aws_session_token") == "sessiontoken"
            return secrets_client
        raise AssertionError(f"Unexpected service requested: {service_name}")

    with patch("services.email_service.boto3.client", side_effect=client_side_effect):
        username, password = service._fetch_smtp_credentials()

    assert username == "rotated"
    assert password == "secret"
    sts_client.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::123456789012:role/test",
        RoleSessionName="ProcWiseEmailSecrets",
    )
    secrets_client.get_secret_value.assert_called_once_with(
        SecretId="ses/smtp/credentials"
    )
