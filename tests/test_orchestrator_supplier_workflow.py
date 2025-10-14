import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentOutput, AgentStatus
from orchestration.orchestrator import Orchestrator


class StubEmailAgent:
    def __init__(self):
        self.calls = []

    def execute(self, context):
        self.calls.append(context.input_data)
        drafts = [
            {
                "supplier_id": "S1",
                "workflow_id": context.workflow_id,
                "unique_id": "uid-1",
            },
            {
                "supplier_id": "S2",
                "workflow_id": context.workflow_id,
                "unique_id": "uid-2",
            },
        ]
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={"drafts": drafts},
            pass_fields={"drafts": drafts},
        )


class StubSupplierAgent:
    def __init__(self):
        self.calls = []

    def execute(self, context):
        self.calls.append(context.input_data)
        return AgentOutput(status=AgentStatus.SUCCESS, data={"processed": True})


class StubCoordinator:
    def __init__(self):
        self.calls = []

    def ensure_ready(
        self,
        *,
        workflow_id,
        unique_ids,
        dispatch_timeout,
        dispatch_poll_interval,
        response_timeout,
        response_poll_interval,
    ):
        self.calls.append(
            {
                "workflow_id": workflow_id,
                "unique_ids": list(unique_ids),
                "dispatch_timeout": dispatch_timeout,
                "dispatch_poll_interval": dispatch_poll_interval,
                "response_timeout": response_timeout,
                "response_poll_interval": response_poll_interval,
            }
        )
        return {
            "activation": {"activated": True, "first_unique_id": "uid-1"},
            "dispatch": {"complete": True, "completed_dispatches": 2},
            "responses": {"complete": False, "completed_responses": 1},
        }


class StubSettings:
    script_user = "tester"
    max_workers = 1
    email_response_poll_seconds = 1
    email_response_timeout_seconds = 5


class StubNick:
    def __init__(self, email_agent, supplier_agent):
        self.settings = StubSettings()
        self.agents = {
            "email_drafting": email_agent,
            "supplier_interaction": supplier_agent,
        }
        self.policy_engine = SimpleNamespace(
            supplier_policies=[],
            validate_workflow=lambda *_args, **_kwargs: {"allowed": True},
        )
        self.query_engine = SimpleNamespace(fetch_supplier_data=lambda *_: {})
        self.routing_engine = SimpleNamespace(routing_model={"global_settings": {"max_chain_depth": 3}})
        self.backend_scheduler = None

    def get_db_connection(self):
        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, *args, **kwargs):
                return None

            def fetchall(self):
                return []

            def close(self):
                return None

        class _Conn:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def cursor(self_inner):
                return _Cursor()

        return _Conn()


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("orchestration.orchestrator.configure_gpu", lambda: "cpu")
    monkeypatch.setattr(
        "orchestration.orchestrator.BackendScheduler.ensure",
        classmethod(lambda cls, *_args, **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        "orchestration.orchestrator.get_event_bus",
        lambda: SimpleNamespace(publish=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(patch_dependencies, "coordinator", None, raising=False)
    monkeypatch.setattr(
        "services.supplier_response_workflow.SupplierResponseWorkflow",
        lambda: patch_dependencies.coordinator,
    )
    monkeypatch.setattr(
        "orchestration.orchestrator.SupplierResponseWorkflow",
        lambda: patch_dependencies.coordinator,
    )
    yield


def test_supplier_workflow_coordinates_agents(monkeypatch):
    email_agent = StubEmailAgent()
    supplier_agent = StubSupplierAgent()
    coordinator = StubCoordinator()

    monkeypatch.setattr(
        patch_dependencies, "coordinator", coordinator, raising=False
    )

    nick = StubNick(email_agent, supplier_agent)
    orchestrator = Orchestrator(nick)

    response = orchestrator.execute_workflow(
        "supplier_interaction",
        {
            "draft_payload": {"subject": "RFQ"},
            "dispatch_poll_interval": 2,
            "dispatch_timeout_seconds": 15,
            "response_timeout_seconds": 20,
            "response_poll_interval": 3,
        },
    )

    result = response.get("result", {})

    assert email_agent.calls
    assert supplier_agent.calls
    supplier_input = supplier_agent.calls[0]
    assert supplier_input.get("await_response") is True
    assert supplier_input.get("await_all_responses") is True
    assert len(supplier_input.get("drafts", [])) == 2
    assert sorted(set(coordinator.calls[0]["unique_ids"])) == ["uid-1", "uid-2"]
    assert result["activation_monitor"]["activated"] is True
    assert result["dispatch_monitor"]["complete"] is True
    assert result["response_monitor"]["complete"] is False
    assert result["supplier_interaction"]["processed"] is True
