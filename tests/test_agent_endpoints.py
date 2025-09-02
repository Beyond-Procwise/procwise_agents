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

    def log_process(self, **kwargs):
        return 1

    def log_action(self, **kwargs):
        self.logged.append(kwargs)
        return kwargs.get("action_id", "a1")

    def log_run_detail(self, **kwargs):
        return kwargs.get("run_id", "r1")

    def update_process_status(self, *args, **kwargs):
        pass


class DummyOrchestrator:
    def __init__(self):
        self.agent_nick = SimpleNamespace(process_routing_service=DummyPRS())

    def execute_workflow(self, workflow_name, input_data):
        return {"status": "completed", "workflow_id": "wf", "result": {"echo": input_data}}


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

