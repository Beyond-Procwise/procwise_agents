import os
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers.agents import router as agents_router
from api.routers.workflows import router as workflows_router


class DummyPRS:
    def __init__(self):
        self.logged = []
        self.updated_details = None

    def log_process(self, **kwargs):
        return 1

    def log_action(self, **kwargs):
        self.logged.append(kwargs)
        return kwargs.get("action_id", "a1")

    def log_run_detail(self, **kwargs):
        return kwargs.get("run_id", "r1")

    def validate_workflow_id(self, *_args, **_kwargs):
        return True

    def update_process_status(self, *args, **kwargs):
        pass

    def update_process_details(self, process_id, process_details, **kwargs):
        self.updated_details = process_details


class DummyOrchestrator:
    def __init__(self):
        self.agent_nick = SimpleNamespace(process_routing_service=DummyPRS())

    def execute_workflow(self, workflow_name, input_data):
        if workflow_name == "email_drafting":
            output = {
                **input_data,
                "action_id": input_data.get("action_id", "a1"),
                "body": input_data.get("body", "<p>generated</p>"),
                "sent": False,
                "drafts": [
                    {
                        "rfq_id": "RFQ-123",
                        "action_id": input_data.get("action_id", "a1"),
                        "sent_status": False,
                    }
                ],
            }
            return {
                "status": "completed",
                "workflow_id": "wf",
                "result": {"email_drafting": output},
            }
        return {
            "status": "completed",
            "workflow_id": "wf",
            "result": {"echo": input_data},
        }



def test_agent_execute_endpoint():
    app = FastAPI()
    app.include_router(agents_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    resp = client.post(
        "/agents/execute",
        json={"agent_type": "test_agent", "payload": {"foo": "bar"}},
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["echo"]["foo"] == "bar"
    prs = orchestrator.agent_nick.process_routing_service
    assert len(prs.logged) == 2
    assert prs.logged[0]["status"] == "started"
    assert prs.logged[1]["status"] == "completed"


def test_workflow_types_endpoint():
    app = FastAPI()
    app.include_router(workflows_router)
    client = TestClient(app)

    resp = client.get("/workflows/types")
    assert resp.status_code == 200
    types = [item["agentType"] for item in resp.json()]
    assert "OpportunityMinerAgent" in types
    assert "DiscrepancyDetectionAgent" not in types


def test_email_workflow_returns_action_id(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    calls = {}

    class StubDispatch:
        def __init__(self, agent_nick):
            calls["agent_nick"] = agent_nick

        def send_draft(
            self,
            rfq_id,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            calls["args"] = (rfq_id, recipients, sender, subject_override, body_override)
            return {
                "rfq_id": rfq_id,
                "sent": True,
                "recipients": recipients or ["r1", "r2"],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "s",
                "body": body_override or "<p>generated</p>",
                "thread_index": 1,
                "draft": {
                    "rfq_id": rfq_id,
                    "sent_status": True,
                    "dispatch_metadata": {"rfq_id": rfq_id},
                },
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post(
        "/workflows/email",
        data={
            "rfq_id": "RFQ-123",
            "subject": "s",
            "recipients": "r1,r2",
            "action_id": "a1",
            "body": "<p>generated</p>",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action_id"] == "a1"
    assert data["status"] == "completed"
    assert data["result"]["sent"] is True
    assert data["result"]["recipients"] == ["r1", "r2"]
    assert data["result"]["draft"]["sent_status"] is True
    assert calls["args"][0] == "RFQ-123"

    prs = orchestrator.agent_nick.process_routing_service
    assert len(prs.logged) == 2
    assert prs.logged[0]["status"] == "started"
    assert prs.logged[1]["status"] == "completed"
    assert prs.logged[0]["action_desc"]["rfq_id"] == "RFQ-123"
    assert prs.updated_details["output"]["sent"] is True
    assert prs.updated_details["status"] == "completed"


def test_email_workflow_marks_failed_dispatch(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    class StubDispatchFail:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def send_draft(
            self,
            rfq_id,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            return {
                "rfq_id": rfq_id,
                "sent": False,
                "recipients": recipients or [],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
                "body": body_override or "<p>body</p>",
                "thread_index": 1,
                "draft": {"rfq_id": rfq_id, "sent_status": False},
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatchFail)

    resp = client.post(
        "/workflows/email",
        data={
            "rfq_id": "RFQ-456",
            "subject": "supplier action",
            "recipients": "buyer@example.com",
            "action_id": "workflow-action",
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "failed"
    assert data["result"]["sent"] is False

    prs = orchestrator.agent_nick.process_routing_service
    assert prs.updated_details["status"] == "failed"


