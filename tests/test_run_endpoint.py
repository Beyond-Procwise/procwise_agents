import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers.run import router as run_router
import copy


class DummyPRS:
    def __init__(self, status="saved"):
        self.status_updates = []
        self.details_updates = []
        self.status = status

    def get_process_details(self, process_id):
        return {
            "status": self.status,
            "agent_type": "1",
            "agent_property": {"llm": "mistral", "prompts": [1], "policies": [2]},
        }

    def update_process_status(self, process_id, status, **kwargs):
        self.status_updates.append((process_id, status))

    def update_process_details(self, process_id, details, **kwargs):
        # Store a copy so that subsequent mutations do not affect earlier
        # snapshots used in assertions.
        self.details_updates.append(copy.deepcopy(details))

    def log_run_detail(self, **kwargs):
        return kwargs.get("run_id", "r1")


class DummyOrchestrator:
    def __init__(self, prs=None):
        self.agent_nick = SimpleNamespace(
            process_routing_service=prs or DummyPRS()
        )

    def execute_agent_flow(self, flow):
        self.received_flow = flow
        flow["status"] = "completed"
        return flow


def create_client(prs=None):
    return create_client_with_orchestrator(DummyOrchestrator(prs))


def create_client_with_orchestrator(orchestrator):
    app = FastAPI()
    app.include_router(run_router)
    app.state.orchestrator = orchestrator
    client = TestClient(app)
    return client, orchestrator


def test_run_endpoint_process_id_executes_flow():
    client, orchestrator = create_client()
    resp = client.post("/run", json={"process_id": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"

    prs = orchestrator.agent_nick.process_routing_service
    # Wait for background thread to update status and details
    import time

    for _ in range(50):
        if prs.status_updates == [(5, 1)] and len(prs.details_updates) >= 2:
            break
        time.sleep(0.01)

    assert prs.status_updates == [(5, 1)]
    # First update marks the run as started
    assert prs.details_updates[0]["process_status"] == 0
    # Final update contains completion information
    assert prs.details_updates[1]["process_status"] == 1
    assert prs.details_updates[1]["status"] == "completed"
    assert orchestrator.received_flow["agent_type"] == "1"


def test_run_endpoint_updates_nested_statuses_independently():
    class NestedPRS(DummyPRS):
        def get_process_details(self, process_id):
            return {
                "status": "saved",
                "agent_type": "1",
                "agent_property": {"llm": "mistral", "prompts": [1], "policies": [2]},
                "onSuccess": {
                    "status": "saved",
                    "agent_type": "1",
                    "agent_property": {"llm": "mistral", "prompts": [1], "policies": [2]},
                },
                "onFailure": {
                    "status": "saved",
                    "agent_type": "1",
                    "agent_property": {"llm": "mistral", "prompts": [1], "policies": [2]},
                },
            }

    class NestedOrchestrator(DummyOrchestrator):
        def execute_agent_flow(self, flow):
            self.received_flow = flow
            flow["status"] = "completed"
            flow["onSuccess"]["status"] = "completed"
            return flow

    prs = NestedPRS()
    orchestrator = NestedOrchestrator(prs)
    client, orchestrator = create_client_with_orchestrator(orchestrator)

    resp = client.post("/run", json={"process_id": 9})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"

    import time

    for _ in range(50):
        if len(prs.details_updates) >= 2:
            break
        time.sleep(0.01)

    saved = prs.details_updates[1]
    assert saved["status"] == "completed"
    assert saved["onSuccess"]["status"] == "completed"
    assert saved["onFailure"]["status"] == "saved"
    assert saved["process_status"] == 1
