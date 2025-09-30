import json
import smtplib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call


from services.email_credentials_manager import (
    CredentialsRotationError,
    RotatedCredentials,
)
from services.email_service import EmailService


class DummyAgentNick:
    def __init__(self, **overrides):
        defaults = {
            "ses_smtp_secret_name": "ses/smtp/credentials",
            "ses_region": "eu-west-1",
            "ses_smtp_endpoint": "email-smtp.eu-west-1.amazonaws.com",
            "ses_smtp_port": 587,
            "ses_secret_role_arn": None,
            "ses_smtp_propagation_attempts": 6,
            "ses_smtp_propagation_wait_seconds": 30,
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
        SecretId="ses/smtp/credentials", VersionStage="AWSCURRENT"
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
        SecretId="ses/smtp/credentials", VersionStage="AWSCURRENT"
    )


def test_send_email_retries_with_previous_secret_on_auth_failure():
    nick = DummyAgentNick()
    service = EmailService(nick)

    with patch.object(
        service,
        "_fetch_smtp_credentials",
        side_effect=[("user", "pass"), ("legacy", "secret")],
    ) as fetch_mock, patch.object(
        service.credential_manager, "rotate_smtp_credentials"
    ) as rotate_mock, patch("services.email_service.smtplib.SMTP") as smtp_mock:
        first_smtp = MagicMock()
        first_smtp.__enter__.return_value = first_smtp
        first_smtp.__exit__.return_value = False
        first_smtp.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"Invalid"
        )

        second_smtp = MagicMock()
        second_smtp.__enter__.return_value = second_smtp
        second_smtp.__exit__.return_value = False

        smtp_mock.side_effect = [first_smtp, second_smtp]

        result = service.send_email(
            subject="Test",
            body="<p>Hello</p>",
            recipients=["to@example.com"],
            sender="from@example.com",
        )

    assert result.success is True
    assert result.message_id
    assert fetch_mock.call_args_list[0] == call()
    assert fetch_mock.call_args_list[1] == call(version_stage="AWSPREVIOUS")
    rotate_mock.assert_not_called()
    first_smtp.sendmail.assert_not_called()
    second_smtp.sendmail.assert_called_once()


def test_send_email_returns_false_when_retry_also_fails_authentication():
    nick = DummyAgentNick()
    service = EmailService(nick)

    auth_error = smtplib.SMTPAuthenticationError(535, b"Invalid")

    with patch.object(
        service,
        "_fetch_smtp_credentials",
        side_effect=[("user", "pass"), ("legacy", "secret")],
    ) as fetch_mock, patch.object(
        service.credential_manager,
        "rotate_smtp_credentials",
        side_effect=CredentialsRotationError("rotation failed"),
    ) as rotate_mock, patch("services.email_service.smtplib.SMTP") as smtp_mock:
        first_smtp = MagicMock()
        first_smtp.__enter__.return_value = first_smtp
        first_smtp.__exit__.return_value = False
        first_smtp.login.side_effect = auth_error

        second_smtp = MagicMock()
        second_smtp.__enter__.return_value = second_smtp
        second_smtp.__exit__.return_value = False
        second_smtp.login.side_effect = auth_error

        smtp_mock.side_effect = [first_smtp, second_smtp]

        result = service.send_email(
            subject="Test",
            body="<p>Hello</p>",
            recipients=["to@example.com"],
            sender="from@example.com",
        )

    assert result.success is False
    assert result.message_id
    assert fetch_mock.call_args_list[0] == call()
    assert fetch_mock.call_args_list[1] == call(version_stage="AWSPREVIOUS")
    rotate_mock.assert_called_once()


def test_send_email_rotates_credentials_after_auth_failures():
    nick = DummyAgentNick()
    service = EmailService(nick)

    rotated = RotatedCredentials(
        smtp_username="AKIANEW",
        smtp_password="smtp-secret",
        access_key_id="AKIANEW",
    )

    with patch.object(
        service,
        "_fetch_smtp_credentials",
        side_effect=[("user", "pass"), ("legacy", "secret")],
    ) as fetch_mock, patch.object(
        service.credential_manager,
        "rotate_smtp_credentials",
        return_value=rotated,
    ) as rotate_mock, patch("services.email_service.smtplib.SMTP") as smtp_mock:
        first_smtp = MagicMock()
        first_smtp.__enter__.return_value = first_smtp
        first_smtp.__exit__.return_value = False
        first_smtp.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"Invalid"
        )

        second_smtp = MagicMock()
        second_smtp.__enter__.return_value = second_smtp
        second_smtp.__exit__.return_value = False
        second_smtp.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"Invalid"
        )

        third_smtp = MagicMock()
        third_smtp.__enter__.return_value = third_smtp
        third_smtp.__exit__.return_value = False

        smtp_mock.side_effect = [first_smtp, second_smtp, third_smtp]

        result = service.send_email(
            subject="Test",
            body="<p>Hello</p>",
            recipients=["to@example.com"],
            sender="from@example.com",
        )

    assert result.success is True
    assert result.message_id
    assert fetch_mock.call_args_list[0] == call()
    assert fetch_mock.call_args_list[1] == call(version_stage="AWSPREVIOUS")
    rotate_mock.assert_called_once_with()
    third_smtp.login.assert_called_once_with("AKIANEW", "smtp-secret")
    third_smtp.sendmail.assert_called_once()


def test_rotated_credentials_retry_until_propagation_succeeds():
    nick = DummyAgentNick(
        ses_smtp_propagation_attempts=2,
        ses_smtp_propagation_wait_seconds=1,
    )
    service = EmailService(nick)

    rotated = RotatedCredentials(
        smtp_username="AKIANEW",
        smtp_password="smtp-secret",
        access_key_id="AKIANEW",
    )

    auth_error = smtplib.SMTPAuthenticationError(535, b"Invalid")

    with patch.object(
        service,
        "_fetch_smtp_credentials",
        side_effect=[
            ("user", "pass"),
            ("legacy", "secret"),
            ("AKIANEW", "smtp-secret"),
        ],
    ) as fetch_mock, patch.object(
        service.credential_manager,
        "rotate_smtp_credentials",
        return_value=rotated,
    ) as rotate_mock, patch(
        "services.email_service.smtplib.SMTP"
    ) as smtp_mock, patch(
        "services.email_service.time.sleep"
    ) as sleep_mock:
        first_smtp = MagicMock()
        first_smtp.__enter__.return_value = first_smtp
        first_smtp.__exit__.return_value = False
        first_smtp.login.side_effect = auth_error

        second_smtp = MagicMock()
        second_smtp.__enter__.return_value = second_smtp
        second_smtp.__exit__.return_value = False
        second_smtp.login.side_effect = auth_error

        third_smtp = MagicMock()
        third_smtp.__enter__.return_value = third_smtp
        third_smtp.__exit__.return_value = False
        third_smtp.login.side_effect = auth_error

        fourth_smtp = MagicMock()
        fourth_smtp.__enter__.return_value = fourth_smtp
        fourth_smtp.__exit__.return_value = False

        smtp_mock.side_effect = [first_smtp, second_smtp, third_smtp, fourth_smtp]

        result = service.send_email(
            subject="Test",
            body="<p>Hello</p>",
            recipients=["to@example.com"],
            sender="from@example.com",
        )

    assert result.success is True
    assert result.message_id
    assert fetch_mock.call_args_list[0] == call()
    assert fetch_mock.call_args_list[1] == call(version_stage="AWSPREVIOUS")
    assert fetch_mock.call_args_list[2] == call()
    rotate_mock.assert_called_once_with()
    sleep_mock.assert_called_once_with(1)
    fourth_smtp.sendmail.assert_called_once()


def test_rotated_credentials_retry_exhausts_attempts():
    nick = DummyAgentNick(
        ses_smtp_propagation_attempts=2,
        ses_smtp_propagation_wait_seconds=2,
    )
    service = EmailService(nick)

    rotated = RotatedCredentials(
        smtp_username="AKIANEW",
        smtp_password="smtp-secret",
        access_key_id="AKIANEW",
    )

    auth_error = smtplib.SMTPAuthenticationError(535, b"Invalid")

    with patch.object(
        service,
        "_fetch_smtp_credentials",
        side_effect=[
            ("user", "pass"),
            ("legacy", "secret"),
            ("AKIANEW", "smtp-secret"),
        ],
    ) as fetch_mock, patch.object(
        service.credential_manager,
        "rotate_smtp_credentials",
        return_value=rotated,
    ) as rotate_mock, patch(
        "services.email_service.smtplib.SMTP"
    ) as smtp_mock, patch(
        "services.email_service.time.sleep"
    ) as sleep_mock:
        first_smtp = MagicMock()
        first_smtp.__enter__.return_value = first_smtp
        first_smtp.__exit__.return_value = False
        first_smtp.login.side_effect = auth_error

        second_smtp = MagicMock()
        second_smtp.__enter__.return_value = second_smtp
        second_smtp.__exit__.return_value = False
        second_smtp.login.side_effect = auth_error

        third_smtp = MagicMock()
        third_smtp.__enter__.return_value = third_smtp
        third_smtp.__exit__.return_value = False
        third_smtp.login.side_effect = auth_error

        fourth_smtp = MagicMock()
        fourth_smtp.__enter__.return_value = fourth_smtp
        fourth_smtp.__exit__.return_value = False
        fourth_smtp.login.side_effect = auth_error

        smtp_mock.side_effect = [first_smtp, second_smtp, third_smtp, fourth_smtp]

        result = service.send_email(
            subject="Test",
            body="<p>Hello</p>",
            recipients=["to@example.com"],
            sender="from@example.com",
        )

    assert result.success is False
    assert result.message_id
    assert fetch_mock.call_args_list[0] == call()
    assert fetch_mock.call_args_list[1] == call(version_stage="AWSPREVIOUS")
    assert fetch_mock.call_args_list[2] == call()
    rotate_mock.assert_called_once_with()
    sleep_mock.assert_called_once_with(2)

