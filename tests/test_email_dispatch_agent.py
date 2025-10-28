import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.email_dispatch_agent import EmailDispatchAgent
from agents.base_agent import AgentContext, AgentStatus
from services.event_bus import get_event_bus


class DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):  # pragma: no cover - agent test does not query DB
        raise AssertionError("cursor should not be used in dispatch agent test")


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace()

    def get_db_connection(self):
        return DummyConnection()


@pytest.fixture(autouse=True)
def reset_event_bus(monkeypatch):
    bus = get_event_bus()
    listeners = list(bus._listeners.keys())  # type: ignore[attr-defined]
    for key in listeners:
        bus._listeners.pop(key, None)  # type: ignore[attr-defined]
    yield
    listeners = list(bus._listeners.keys())  # type: ignore[attr-defined]
    for key in listeners:
        bus._listeners.pop(key, None)  # type: ignore[attr-defined]


def test_email_dispatch_agent_sends_all_drafts(monkeypatch):
    nick = DummyNick()
    agent = EmailDispatchAgent(nick)

    sent_calls: List[Dict[str, Any]] = []

    def fake_send(identifier: str, **kwargs):
        sent_calls.append({"identifier": identifier, **kwargs})
        unique_id = identifier
        message_id = f"<{unique_id}-msg>"
        return {
            "sent": True,
            "message_id": message_id,
            "draft": {
                "unique_id": unique_id,
                "message_id": message_id,
                "thread_headers": {"Message-ID": message_id},
                "sent_status": True,
            },
        }

    monkeypatch.setattr(agent.dispatch_service, "send_draft", fake_send)

    coordinator_calls: List[Dict[str, Any]] = []

    def fake_register(workflow_id: str, unique_ids: List[str], expected_count: int):
        coordinator_calls.append(
            {
                "workflow_id": workflow_id,
                "unique_ids": list(unique_ids),
                "expected_count": expected_count,
            }
        )

    monkeypatch.setattr(
        agent, "_response_coordinator", SimpleNamespace(register_expected_responses=fake_register)
    )

    bus = get_event_bus()
    captured: List[Dict[str, Any]] = []
    bus.subscribe("round_dispatch_complete", lambda payload: captured.append(dict(payload)), once=True)

    drafts = [
        {
            "unique_id": f"PROC-WF-UID-{idx}",
            "supplier_id": f"SUP-{idx}",
            "recipients": [f"supplier{idx}@example.com"],
            "sender": "buyer@example.com",
            "subject": f"RFQ Update {idx}",
            "body": "<p>Hello</p>",
        }
        for idx in range(1, 4)
    ]

    context = AgentContext(
        workflow_id="wf-123",
        agent_id="email_dispatch",
        user_id="user",
        input_data={"drafts": drafts, "round": 2},
    )

    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["expected_dispatches"] == 3
    records = result.data["dispatch_records"]
    assert len(records) == 3
    assert all(record["status"] == "sent" for record in records)
    assert {record["unique_id"] for record in records} == {
        "PROC-WF-UID-1",
        "PROC-WF-UID-2",
        "PROC-WF-UID-3",
    }
    assert len(captured) == 1
    event_payload = captured[0]
    assert event_payload["workflow_id"] == "wf-123"
    assert event_payload["expected_count"] == 3
    assert set(event_payload["unique_ids"]) == {
        "PROC-WF-UID-1",
        "PROC-WF-UID-2",
        "PROC-WF-UID-3",
    }
    assert coordinator_calls == [
        {
            "workflow_id": "wf-123",
            "unique_ids": ["PROC-WF-UID-1", "PROC-WF-UID-2", "PROC-WF-UID-3"],
            "expected_count": 3,
        }
    ]

