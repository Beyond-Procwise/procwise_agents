import importlib
import os
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List
from types import ModuleType, SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

services_pkg = sys.modules.get("services")
if services_pkg is None:
    services_pkg = importlib.import_module("services")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = ModuleType("sentence_transformers")

    class _StubSentenceTransformer:  # pragma: no cover - test shim
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            return []

    sentence_transformers_stub.SentenceTransformer = _StubSentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub

if "ollama" not in sys.modules:
    ollama_stub = ModuleType("ollama")

    def _noop_response(*_args, **_kwargs):  # pragma: no cover - test shim
        return {}

    def _noop_list(*_args, **_kwargs):  # pragma: no cover - test shim
        return {"models": []}

    ollama_stub.generate = _noop_response
    ollama_stub.chat = _noop_response
    ollama_stub.list = _noop_list

    ollama_types_stub = ModuleType("ollama._types")

    class _StubResponseError(Exception):
        pass

    ollama_stub.ResponseError = _StubResponseError
    ollama_types_stub.ResponseError = _StubResponseError

    sys.modules["ollama"] = ollama_stub
    sys.modules["ollama._types"] = ollama_types_stub

if "pydantic_settings" not in sys.modules:
    pydantic_settings_stub = ModuleType("pydantic_settings")

    class _StubBaseSettings:  # pragma: no cover - test shim
        def __init__(self, *args, **kwargs):
            pass

    pydantic_settings_stub.BaseSettings = _StubBaseSettings
    sys.modules["pydantic_settings"] = pydantic_settings_stub

if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    torch_cuda_stub = ModuleType("torch.cuda")

    class _StubTensor:  # pragma: no cover - test shim
        pass

    def _no_grad():  # pragma: no cover - test shim
        return nullcontext()

    def _is_available():  # pragma: no cover - test shim
        return False

    def _empty_cache():  # pragma: no cover - test shim
        return None

    torch_stub.Tensor = _StubTensor
    torch_stub.no_grad = _no_grad
    torch_stub.cuda = torch_cuda_stub
    torch_cuda_stub.is_available = _is_available
    torch_cuda_stub.empty_cache = _empty_cache

    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = torch_cuda_stub

if "orchestration" not in sys.modules:
    orchestration_stub = ModuleType("orchestration")
    orchestrator_module_stub = ModuleType("orchestration.orchestrator")

    class _StubOrchestrator:  # pragma: no cover - test shim
        def __init__(self, *args, **kwargs):
            self.agent_nick = SimpleNamespace()

    orchestrator_module_stub.Orchestrator = _StubOrchestrator
    orchestration_stub.orchestrator = orchestrator_module_stub

    sys.modules["orchestration"] = orchestration_stub
    sys.modules["orchestration.orchestrator"] = orchestrator_module_stub

if "services.model_selector" not in sys.modules:
    model_selector_stub = ModuleType("services.model_selector")

    class _StubRAGPipeline:  # pragma: no cover - test shim
        pass

    model_selector_stub.RAGPipeline = _StubRAGPipeline
    setattr(services_pkg, "model_selector", model_selector_stub)
    sys.modules["services.model_selector"] = model_selector_stub

if "services.email_dispatch_service" not in sys.modules:
    email_dispatch_stub = ModuleType("services.email_dispatch_service")

    class _StubDraftNotFoundError(ValueError):
        pass

    class _StubEmailDispatchService:  # pragma: no cover - test shim
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def dispatch_from_context(self, *_args, **_kwargs):
            raise NotImplementedError

        def send_draft(self, *_args, **_kwargs):
            raise NotImplementedError

        def resolve_workflow_id(self, *_args, **_kwargs):
            return None

    email_dispatch_stub.DraftNotFoundError = _StubDraftNotFoundError
    email_dispatch_stub.EmailDispatchService = _StubEmailDispatchService
    setattr(services_pkg, "email_dispatch_service", email_dispatch_stub)
    sys.modules["services.email_dispatch_service"] = email_dispatch_stub

import api.routers.workflows as workflows_module
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


def test_context_email_dispatch_recomputes_idempotency(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)

    agent_stub = SimpleNamespace()
    orchestrator = SimpleNamespace(agent_nick=agent_stub)
    app.state.agent_nick = agent_stub
    app.state.orchestrator = orchestrator

    client = TestClient(app)

    original_helper = workflows_module._ensure_request_idempotency_key
    helper_calls = []

    def tracking_helper(request, *, workflow_id, subject, body, identifier=None, force=False):
        helper_calls.append(
            {
                "workflow_id": workflow_id,
                "subject": subject,
                "body": body,
                "identifier": identifier,
                "force": force,
            }
        )
        return original_helper(
            request,
            workflow_id=workflow_id,
            subject=subject,
            body=body,
            identifier=identifier,
            force=force,
        )

    monkeypatch.setattr(
        workflows_module,
        "_ensure_request_idempotency_key",
        tracking_helper,
    )

    class StubDispatchService:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def dispatch_from_context(self, workflow_id, *, overrides=None):
            resolved_workflow = workflow_id or "WF-001"
            return {
                "unique_id": "PROC-WF-001",
                "subject": "Resolved subject",
                "body": "<p>Resolved body</p>",
                "workflow_id": resolved_workflow,
                "draft": {
                    "unique_id": "PROC-WF-001",
                    "workflow_id": resolved_workflow,
                },
            }

    monkeypatch.setattr(
        workflows_module,
        "EmailDispatchService",
        StubDispatchService,
    )

    response = client.post(
        "/workflows/email",
        headers={"X-Workflow-Id": "WF-001"},
    )

    assert response.status_code == 200
    assert len(helper_calls) == 2
    assert helper_calls[0]["identifier"] is None
    assert helper_calls[1]["identifier"] == "PROC-WF-001"
    assert helper_calls[1]["force"] is True
    assert helper_calls[1]["workflow_id"] == "WF-001"


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


def test_email_endpoint_accepts_single_draft_payload(monkeypatch):
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
                "subject": subject_override,
                "body": body_override,
            }
            return {
                "unique_id": identifier,
                "sent": True,
                "recipients": recipients or ["buyer@example.com"],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    payload = {
        "drafts": [
            {
                "unique_id": "PROC-WF-SINGLE",
                "recipients": ["buyer@example.com"],
                "subject": "Subject from draft",
                "body": "<p>Body</p>",
            }
        ]
    }

    resp = client.post("/workflows/email", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["unique_id"] == "PROC-WF-SINGLE"
    assert data["result"]["sent"] is True
    assert captured["call"]["identifier"] == "PROC-WF-SINGLE"
    assert captured["call"]["recipients"] == ["buyer@example.com"]


def test_email_endpoint_batches_multiple_drafts(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    calls: List[Dict[str, Any]] = []

    class StubDispatch:
        def __init__(self, agent_nick):
            calls.append({"agent_nick": agent_nick})

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
                    "subject": subject_override,
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
        "drafts": [
            {
                "unique_id": "PROC-WF-ONE",
                "recipients": ["buyer1@example.com"],
                "subject": "Subject 1",
                "body": "<p>Body 1</p>",
            },
            {
                "unique_id": "PROC-WF-TWO",
                "recipients": ["buyer2@example.com"],
                "subject": "Subject 2",
                "body": "<p>Body 2</p>",
            },
        ]
    }

    resp = client.post("/workflows/email", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["sent"] == 2
    assert body["failed"] == 0
    assert body["success"] is True
    identifiers = {call["identifier"] for call in calls if "identifier" in call}
    assert identifiers == {"PROC-WF-ONE", "PROC-WF-TWO"}
