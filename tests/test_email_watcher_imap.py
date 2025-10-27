import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from repositories.workflow_email_tracking_repo import WorkflowDispatchRow


_DEFAULT_DISPATCHED_AT = object()


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    # Ensure tests have a clean credential state
    for key in ["IMAP_HOST", "IMAP_USERNAME", "IMAP_USER", "IMAP_PASSWORD", "IMAP_MAILBOX"]:
        monkeypatch.delenv(key, raising=False)
    yield


def _sample_row(
    message_id: str | None = "msg-123",
    *,
    dispatched_at: datetime | object = _DEFAULT_DISPATCHED_AT,
) -> WorkflowDispatchRow:
    if dispatched_at is _DEFAULT_DISPATCHED_AT:
        dispatch_value = datetime.now(timezone.utc)
    else:
        dispatch_value = dispatched_at  # type: ignore[assignment]
    return WorkflowDispatchRow(
        workflow_id="wf-test",
        unique_id="uid-123",
        supplier_id="sup-1",
        supplier_email="supplier@example.com",
        message_id=message_id,
        subject="RFQ",
        dispatched_at=dispatch_value,
        responded_at=None,
        response_message_id=None,
        matched=False,
        thread_headers={"message_id": [message_id] if message_id else []},
    )


def test_watcher_skips_without_credentials(monkeypatch):
    from services import email_watcher_service

    monkeypatch.setattr(email_watcher_service, "app_settings", None)
    monkeypatch.setattr(email_watcher_service, "_load_dispatch_rows", lambda workflow_id: [_sample_row()])

    result = email_watcher_service.run_email_watcher_for_workflow(workflow_id="wf-test", run_id=None)

    assert result["status"] == "skipped"
    assert result["reason"] == "IMAP credentials not configured"


def test_watcher_waits_until_dispatch_complete(monkeypatch):
    from services import email_watcher_service

    monkeypatch.setattr(
        email_watcher_service,
        "app_settings",
        SimpleNamespace(
            imap_host=None,
            imap_username=None,
            imap_user=None,
            imap_login=None,
            imap_password=None,
            imap_mailbox=None,
        ),
    )

    monkeypatch.setattr(
        email_watcher_service,
        "_load_dispatch_rows",
        lambda workflow_id: [
            _sample_row(message_id="msg-123", dispatched_at=None)
        ],
    )

    instantiated = {"count": 0}

    class _FakeWatcher:
        def __init__(self, *args, **kwargs):
            instantiated["count"] += 1

        def wait_and_collect_responses(self, workflow_id):  # pragma: no cover - should not be called
            return {}

    monkeypatch.setattr(email_watcher_service, "EmailWatcherV2", _FakeWatcher)

    os.environ["IMAP_HOST"] = "imap.test"
    os.environ["IMAP_USERNAME"] = "user@test"
    os.environ["IMAP_PASSWORD"] = "secret"

    result = email_watcher_service.run_email_watcher_for_workflow(workflow_id="wf-test", run_id=None)

    assert result["status"] == "waiting_for_dispatch"
    assert result["pending_unique_ids"] == ["uid-123"]
    assert result["missing_required_fields"].get("uid-123") == ["dispatched_at"]
    assert instantiated["count"] == 0


def test_watcher_waits_for_message_ids(monkeypatch):
    from services import email_watcher_service

    env_values = {
        "IMAP_HOST": "imap.test",
        "IMAP_USERNAME": "user@test",
        "IMAP_PASSWORD": "secret",
    }

    monkeypatch.setattr(email_watcher_service, "app_settings", None)
    monkeypatch.setattr(email_watcher_service, "_env", lambda key, default=None: env_values.get(key, default))
    monkeypatch.setattr(email_watcher_service, "_setting", lambda *args: None)
    monkeypatch.setattr(
        email_watcher_service,
        "_load_dispatch_rows",
        lambda workflow_id: [_sample_row(message_id=None)],
    )

    instantiated = {"count": 0}

    class _FakeWatcher:
        def __init__(self, *args, **kwargs):
            instantiated["count"] += 1

        def wait_and_collect_responses(self, workflow_id):  # pragma: no cover - should not be called
            return {}

    monkeypatch.setattr(email_watcher_service, "EmailWatcherV2", _FakeWatcher)

    result = email_watcher_service.run_email_watcher_for_workflow(workflow_id="wf-test", run_id=None)

    assert result["status"] == "waiting_for_dispatch"
    assert result["pending_unique_ids"] == ["uid-123"]
    assert result["missing_required_fields"].get("uid-123") == ["message_id"]
    assert instantiated["count"] == 0


def test_run_email_watcher_uses_agent_registry(monkeypatch):
    from services import email_watcher_service

    env_values = {
        "IMAP_HOST": "imap.test",
        "IMAP_USERNAME": "user@test",
        "IMAP_PASSWORD": "secret",
    }

    monkeypatch.setattr(email_watcher_service, "app_settings", None)
    monkeypatch.setattr(email_watcher_service, "_env", lambda key, default=None: env_values.get(key, default))
    monkeypatch.setattr(email_watcher_service, "_setting", lambda *args: None)
    monkeypatch.setattr(
        email_watcher_service,
        "_load_dispatch_rows",
        lambda workflow_id: [_sample_row()],
    )
    monkeypatch.setattr(
        email_watcher_service.supplier_response_repo,
        "fetch_pending",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        email_watcher_service.supplier_response_repo,
        "delete_responses",
        lambda **kwargs: None,
    )

    captured = {}

    class _FakeWatcher:
        def __init__(self, *, supplier_agent=None, negotiation_agent=None, **kwargs):
            captured["supplier_agent"] = supplier_agent
            captured["negotiation_agent"] = negotiation_agent

        def wait_and_collect_responses(self, workflow_id):
            return {
                "complete": False,
                "dispatched_count": 1,
                "responded_count": 0,
                "matched_responses": {},
            }

    monkeypatch.setattr(email_watcher_service, "EmailWatcherV2", _FakeWatcher)

    supplier_agent = object()
    negotiation_agent = object()

    result = email_watcher_service.run_email_watcher_for_workflow(
        workflow_id="wf-agent",
        run_id=None,
        agent_registry={
            "supplier_interaction": supplier_agent,
            "negotiation": negotiation_agent,
        },
    )

    assert captured.get("supplier_agent") is supplier_agent
    assert captured.get("negotiation_agent") is negotiation_agent
    assert result["status"] == "not_ready"


def test_watcher_accepts_login_without_username(monkeypatch):
    from services import email_watcher_service

    env_values = {
        "IMAP_HOST": "imap.test",
        "IMAP_LOGIN": "login@test",
        "IMAP_PASSWORD": "secret",
    }

    monkeypatch.setattr(email_watcher_service, "app_settings", None)
    monkeypatch.setattr(email_watcher_service, "_env", lambda key, default=None: env_values.get(key, default))
    monkeypatch.setattr(email_watcher_service, "_setting", lambda *args: None)
    monkeypatch.setattr(
        email_watcher_service,
        "_load_dispatch_rows",
        lambda workflow_id: [_sample_row()],
    )
    monkeypatch.setattr(
        email_watcher_service.supplier_response_repo,
        "fetch_pending",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        email_watcher_service.supplier_response_repo,
        "delete_responses",
        lambda **kwargs: None,
    )

    captured = {}

    class _FakeWatcher:
        def __init__(self, *, imap_username=None, imap_login=None, **kwargs):
            captured["imap_username"] = imap_username
            captured["imap_login"] = imap_login

        def wait_and_collect_responses(self, workflow_id):
            return {
                "complete": False,
                "dispatched_count": 1,
                "responded_count": 0,
                "matched_responses": {},
            }

    monkeypatch.setattr(email_watcher_service, "EmailWatcherV2", _FakeWatcher)

    result = email_watcher_service.run_email_watcher_for_workflow(
        workflow_id="wf-login",
        run_id=None,
    )

    assert result["status"] == "not_ready"
    assert captured.get("imap_login") == "login@test"
    assert captured.get("imap_username") == "login@test"
