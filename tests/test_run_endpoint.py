import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers.run import router as run_router


class DummyPRS:
    def __init__(self):
        self.status_updates = []
        self.details_updates = []

    def log_process(self, **kwargs):
        return 1

    def log_action(self, **kwargs):
        return kwargs.get("action_id", "a1")

    def get_process_details(self, process_id):
        return {
            "status": "saved",
            "agent_type": "1",
            "agent_property": {"llm": "mistral", "prompts": [1], "policies": [2]},
        }

    def update_process_status(self, process_id, status, **kwargs):
        self.status_updates.append((process_id, status))

    def update_process_details(self, process_id, details, **kwargs):
        self.details_updates.append(details)


class DummyOrchestrator:
    def __init__(self):
        self.agent_nick = SimpleNamespace(process_routing_service=DummyPRS())

    def execute_workflow(self, workflow_name, input_data, user_id=None):
        return {
            "status": "completed",
            "workflow_id": "wf",
            "result": {
                "echo": input_data,
                "workflow": workflow_name,
                "user": user_id,
            },
        }

    def execute_agent_flow(self, flow):
        # Simple echo implementation for testing
        self.received_flow = flow
        return {"status": "validated", "plan": [flow]}


def test_run_endpoint_executes_workflow():
    app = FastAPI()
    app.include_router(run_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    client = TestClient(app)

    resp = client.post(
        "/run", json={"workflow": "test", "payload": {"foo": "bar"}, "user_id": "u1"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["echo"]["foo"] == "bar"
    assert data["result"]["workflow"] == "test"
    assert data["result"]["user"] == "u1"


def test_run_endpoint_validates_agent_flow():
    app = FastAPI()
    app.include_router(run_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    client = TestClient(app)

    flow = {
        "status": "saved",
        "agent_type": "1",
        "agent_property": {"llm": "mistral", "prompts": [1], "policies": [2]},
        "onSuccess": {
            "status": "saved",
            "agent_type": "2",
            "agent_property": {"llm": "phi", "prompts": [3], "policies": [4]},
        },
    }

    resp = client.post("/run", json={"agent_flow": flow})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "validated"
    assert data["plan"][0]["agent_type"] == "1"


def test_run_endpoint_process_id_executes_flow():
    app = FastAPI()
    app.include_router(run_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    client = TestClient(app)

    resp = client.post("/run", json={"process_id": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "validated"

    prs = orchestrator.agent_nick.process_routing_service
    assert prs.status_updates == [(5, 1), (5, 2)]
    assert prs.details_updates[0]["status"] == "validated"
    assert orchestrator.received_flow["agent_type"] == "1"
