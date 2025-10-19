import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from repositories.workflow_email_tracking_repo import WorkflowDispatchRow


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    # Ensure tests have a clean credential state
    for key in ["IMAP_HOST", "IMAP_USERNAME", "IMAP_USER", "IMAP_PASSWORD", "IMAP_MAILBOX"]:
        monkeypatch.delenv(key, raising=False)
    yield


def _sample_row(message_id: str | None = "msg-123") -> WorkflowDispatchRow:
    return WorkflowDispatchRow(
        workflow_id="wf-test",
        unique_id="uid-123",
        supplier_id="sup-1",
        supplier_email="supplier@example.com",
        message_id=message_id,
        subject="RFQ",
        dispatched_at=datetime.now(timezone.utc),
        responded_at=None,
        response_message_id=None,
        matched=False,
        thread_headers={"message_id": [message_id] if message_id else []},
    )


def test_watcher_skips_without_credentials(monkeypatch):
    from services import email_watcher_imap

    monkeypatch.setattr(email_watcher_imap, "app_settings", None)
    monkeypatch.setattr(email_watcher_imap, "_load_dispatch_rows", lambda workflow_id: [_sample_row()])

    result = email_watcher_imap.run_email_watcher_for_workflow(workflow_id="wf-test", run_id=None)

    assert result["status"] == "skipped"
    assert result["reason"] == "IMAP credentials not configured"


def test_watcher_waits_until_dispatch_complete(monkeypatch):
    from services import email_watcher_imap

    monkeypatch.setattr(
        email_watcher_imap,
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
        email_watcher_imap,
        "_load_dispatch_rows",
        lambda workflow_id: [_sample_row(message_id=None)],
    )

    instantiated = {"count": 0}

    class _FakeWatcher:
        def __init__(self, *args, **kwargs):
            instantiated["count"] += 1

        def wait_and_collect_responses(self, workflow_id):  # pragma: no cover - should not be called
            return {}

    monkeypatch.setattr(email_watcher_imap, "EmailWatcherV2", _FakeWatcher)

    os.environ["IMAP_HOST"] = "imap.test"
    os.environ["IMAP_USERNAME"] = "user@test"
    os.environ["IMAP_PASSWORD"] = "secret"

    result = email_watcher_imap.run_email_watcher_for_workflow(workflow_id="wf-test", run_id=None)

    assert result["status"] == "waiting_for_dispatch"
    assert "pending_unique_ids" in result
    assert instantiated["count"] == 0


def test_watcher_uses_agent_registry_supplier_factory(monkeypatch):
    from services import email_watcher_imap

    monkeypatch.setattr(email_watcher_imap, "app_settings", None)
    monkeypatch.setattr(
        email_watcher_imap, "_load_dispatch_rows", lambda workflow_id: [_sample_row()]
    )

    captured = {}

    class _FakeWatcher:
        def __init__(self, *args, **kwargs):
            captured["factory"] = kwargs.get("supplier_agent_factory")

        def wait_and_collect_responses(self, workflow_id):
            return {
                "complete": True,
                "workflow_id": workflow_id,
                "dispatched_count": 1,
                "responded_count": 1,
                "matched_responses": {},
            }

    monkeypatch.setattr(email_watcher_imap, "EmailWatcherV2", _FakeWatcher)
    monkeypatch.setattr(
        email_watcher_imap.supplier_response_repo,
        "fetch_pending",
        lambda workflow_id: [],
    )

    os.environ["IMAP_HOST"] = "imap.test"
    os.environ["IMAP_USERNAME"] = "user@test"
    os.environ["IMAP_PASSWORD"] = "secret"

    agent_factory = lambda workflow_id: f"agent-{workflow_id}"

    result = email_watcher_imap.run_email_watcher_for_workflow(
        workflow_id="wf-test",
        run_id=None,
        agent_registry={"supplier_interaction": agent_factory},
    )

    resolved_factory = captured.get("factory")
    assert resolved_factory is not None
    assert resolved_factory("wf-test") == agent_factory("wf-test")
    assert result["status"] == "processed"
