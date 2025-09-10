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
        self.details = {
            "status": status,
            "agents": [{"agent": "A1", "status": "saved", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "agent_ref_id": "1"}],
        }

    def get_process_details(self, process_id, raw=False):
        return copy.deepcopy(self.details)

    def update_process_status(self, process_id, status, **kwargs):
        self.status_updates.append((process_id, status))

    def update_process_details(self, process_id, details, **kwargs):
        self.details = copy.deepcopy(details)
        self.details_updates.append(copy.deepcopy(details))

    def update_agent_status(self, process_id, agent_name, status, **kwargs):
        for agent in self.details["agents"]:
            if agent["agent"] == agent_name:
                agent["status"] = status
        statuses = [a["status"] for a in self.details["agents"]]
        if any(s == "failed" for s in statuses):
            self.details["status"] = "failed"
        elif all(s == "completed" for s in statuses):
            self.details["status"] = "completed"
        else:
            self.details["status"] = "saved"
        self.update_process_details(process_id, self.details)

    def log_run_detail(self, **kwargs):
        return kwargs.get("run_id", "r1")

    def convert_agents_to_flow(self, details):
        from services.process_routing_service import ProcessRoutingService

        return ProcessRoutingService.convert_agents_to_flow(details)

    def _load_agent_links(self):
        return {}, {}, {}

    def _enrich_node(self, node, agent_defs, prompt_map, policy_map):
        return None


class DummyOrchestrator:
    def __init__(self, prs=None):
        self.agent_nick = SimpleNamespace(
            process_routing_service=prs or DummyPRS()
        )

    def execute_agent_flow(self, flow, payload=None, process_id=None, prs=None):
        self.received_flow = flow
        self.received_payload = payload
        if prs and process_id is not None:
            prs.update_agent_status(process_id, "A1", "completed")
        return {"status": 100}


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
    resp = client.post("/run", json={"process_id": 5, "payload": {"foo": "bar"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"

    prs = orchestrator.agent_nick.process_routing_service
    import time

    for _ in range(50):
        if prs.status_updates == [(5, 1)] and len(prs.details_updates) >= 2:
            break
        time.sleep(0.01)

    assert prs.status_updates == [(5, 1)]
    assert "agents" in prs.details_updates[0]
    assert prs.details_updates[0]["status"] == "saved"
    assert prs.details_updates[-1]["status"] == "completed"

    assert orchestrator.received_payload == {"foo": "bar"}


def test_run_endpoint_triggers_all_agents_and_updates_status():
    class MultiPRS(DummyPRS):
        def __init__(self):
            super().__init__()
            self.details = {
                "status": "saved",
                "agents": [
                    {
                        "agent": "A1",
                        "status": "saved",
                        "dependencies": {
                            "onSuccess": ["A2"],
                            "onFailure": [],
                            "onCompletion": [],
                        },
                        "agent_ref_id": "1",
                    },
                    {
                        "agent": "A2",
                        "status": "saved",
                        "dependencies": {
                            "onSuccess": [],
                            "onFailure": [],
                            "onCompletion": [],
                        },
                        "agent_ref_id": "2",
                    },
                ],
            }
            self.agent_updates = []

        def update_agent_status(self, process_id, agent_name, status, **kwargs):
            self.agent_updates.append((process_id, agent_name, status))
            super().update_agent_status(process_id, agent_name, status, **kwargs)

    class MultiOrchestrator(DummyOrchestrator):
        def execute_agent_flow(self, flow, payload=None, process_id=None, prs=None):
            self.received_flow = flow
            self.received_payload = payload
            if prs and process_id is not None:
                prs.update_agent_status(process_id, "A1", "completed")
                prs.update_agent_status(process_id, "A2", "completed")
            return {"status": 100}

    prs = MultiPRS()
    orchestrator = MultiOrchestrator(prs)
    client, orchestrator = create_client_with_orchestrator(orchestrator)

    resp = client.post("/run", json={"process_id": 42})
    assert resp.status_code == 200

    import time

    for _ in range(50):
        if prs.agent_updates == [
            (42, "A1", "completed"),
            (42, "A2", "completed"),
        ] and prs.status_updates == [(42, 1)]:
            break
        time.sleep(0.01)

    assert prs.agent_updates == [
        (42, "A1", "completed"),
        (42, "A2", "completed"),
    ]
    assert prs.details["status"] == "completed"
    assert [a["status"] for a in prs.details["agents"]] == ["completed", "completed"]


def test_run_endpoint_updates_nested_statuses_independently():
    class NestedPRS(DummyPRS):
        def get_process_details(self, process_id, raw=False):
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
        def execute_agent_flow(self, flow, payload=None, process_id=None, prs=None):
            self.received_flow = flow
            self.received_payload = payload
            if prs and process_id is not None:
                prs.update_process_details(process_id, flow)
                flow["status"] = "completed"
                flow["onSuccess"]["status"] = "completed"
                prs.update_process_details(process_id, flow)
            return {"status": 100, **flow}

    prs = NestedPRS()
    orchestrator = NestedOrchestrator(prs)
    client, orchestrator = create_client_with_orchestrator(orchestrator)

    resp = client.post("/run", json={"process_id": 9})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"

    import time

    for _ in range(50):
        if len(prs.details_updates) >= 3:
            break
        time.sleep(0.01)

    saved = prs.details_updates[-1]
    assert saved["status"] == "completed"
    assert saved["onSuccess"]["status"] == "completed"
    assert saved["onFailure"]["status"] == "saved"
