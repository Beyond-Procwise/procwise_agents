import os
import os
import sys
from typing import Any, Dict, List
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

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            calls["args"] = (
                identifier,
                recipients,
                sender,
                subject_override,
                body_override,
            )
            unique_id = f"PROC-WF-{identifier}"
            return {
                "unique_id": unique_id,
                "sent": True,
                "recipients": recipients or ["r1", "r2"],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "s",
                "body": body_override or "<p>generated</p>",
                "thread_index": 1,
                "draft": {
                    "rfq_id": identifier,
                    "unique_id": unique_id,
                    "sent_status": True,
                    "dispatch_metadata": {"unique_id": unique_id},
                },
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post(
        "/workflows/email/batch",
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


def test_email_workflow_accepts_list_recipients(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    captured: Dict[str, Any] = {}

    class StubDispatch:
        def __init__(self, agent_nick):
            captured["agent_nick"] = agent_nick

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            captured["call"] = {
                "identifier": identifier,
                "recipients": recipients,
            }
            return {
                "unique_id": identifier,
                "sent": True,
                "recipients": recipients,
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    payload = {
        "unique_id": "PROC-WF-XYZ",
        "recipients": ["quotes@example.com", "buyer@example.com"],
    }

    resp = client.post("/workflows/email/batch", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["recipients"] == payload["recipients"]
    assert captured["call"]["recipients"] == payload["recipients"]


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

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            return {
                "unique_id": f"PROC-WF-{identifier}",
                "sent": False,
                "recipients": recipients or [],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
                "body": body_override or "<p>body</p>",
                "thread_index": 1,
                "draft": {
                    "rfq_id": identifier,
                    "unique_id": f"PROC-WF-{identifier}",
                    "sent_status": False,
                },
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatchFail)

    resp = client.post(
        "/workflows/email/batch",
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


def test_email_dispatch_without_workflow_is_rejected(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    class StubDispatch:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def resolve_workflow_id(self, identifier):
            return None

        def send_draft(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("send_draft should not be invoked when workflow is missing")

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post("/workflows/email/batch", json={"unique_id": "PROC-WF-123"})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "WorkflowUnavailable"
    assert detail["identifier"] == "PROC-WF-123"


def test_email_dispatch_detects_workflow_mismatch(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    class StubDispatch:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def resolve_workflow_id(self, identifier):
            return "wf-stored"

        def send_draft(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("send_draft should not be invoked when workflows mismatch")

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post(
        "/workflows/email/batch",
        json={"unique_id": "PROC-WF-456", "workflow_id": "wf-request"},
    )

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "WorkflowMismatch"
    assert detail["request_workflow_id"] == "wf-request"
    assert detail["stored_workflow_id"] == "wf-stored"


def test_email_batch_dispatch_multiple_drafts(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    calls: List[Dict[str, Any]] = []

    class StubDispatch:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            calls.append(
                {
                    "identifier": identifier,
                    "recipients": recipients,
                    "sender": sender,
                    "subject": subject_override,
                }
            )
            return {
                "unique_id": identifier,
                "sent": True,
                "recipients": recipients or [],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
                "body": body_override or "<p>body</p>",
                "message_id": f"mid-{identifier}",
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    payload = {
        "drafts": [
            {
                "unique_id": "PROC-WF-1",
                "recipients": ["buyer1@example.com"],
                "subject": "Subject 1",
                "body": "<p>Body 1</p>",
            },
            {
                "unique_id": "PROC-WF-2",
                "recipients": ["buyer2@example.com"],
                "subject": "Subject 2",
                "body": "<p>Body 2</p>",
            },
        ]
    }

    resp = client.post("/workflows/email/batch", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["sent"] == 2
    assert data["failed"] == 0
    assert data["success"] is True
    assert len(data["results"]) == 2
    assert {entry["unique_id"] for entry in data["results"]} == {"PROC-WF-1", "PROC-WF-2"}
    assert len(calls) == 2
    assert {call["identifier"] for call in calls} == {"PROC-WF-1", "PROC-WF-2"}



def test_email_batch_accepts_draft_records(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    captured: Dict[str, Any] = {}

    class StubDispatch:
        def __init__(self, agent_nick):
            captured.setdefault("agent_nick", agent_nick)

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
        ):
            captured.setdefault("calls", []).append(
                {
                    "identifier": identifier,
                    "recipients": recipients,
                    "subject": subject_override,
                    "body": body_override,
                }
            )
            return {
                "unique_id": identifier,
                "sent": True,
                "recipients": recipients or ["buyer@example.com"],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    payload = {
        "draft_records": [
            {
                "unique_id": "PROC-WF-REC1",
                "recipients": ["buyer@example.com"],
                "subject": "Subject from record",
                "body": "<p>Body</p>",
            }
        ]
    }

    resp = client.post("/workflows/email/batch", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["result"]["unique_id"] == "PROC-WF-REC1"
    assert body["result"]["sent"] is True
    assert captured["calls"][0]["identifier"] == "PROC-WF-REC1"
