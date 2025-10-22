import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone

from orchestration.orchestrator import Orchestrator
from agents.base_agent import AgentOutput, AgentStatus


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
        dispatch_context=None,
    ):
        self.calls.append(
            {
                "workflow_id": workflow_id,
                "unique_ids": list(unique_ids),
                "dispatch_timeout": dispatch_timeout,
                "dispatch_poll_interval": dispatch_poll_interval,
                "response_timeout": response_timeout,
                "response_poll_interval": response_poll_interval,
                "dispatch_context": dispatch_context,
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
    assert coordinator.calls[0]["dispatch_context"] is None
    assert result["activation_monitor"]["activated"] is True
    assert result["dispatch_monitor"]["complete"] is True
    assert result["response_monitor"]["complete"] is False
    assert result["supplier_interaction"]["processed"] is True


def test_supplier_workflow_passes_dispatch_freshness(monkeypatch):
    email_agent = StubEmailAgent()
    supplier_agent = StubSupplierAgent()
    coordinator = StubCoordinator()

    monkeypatch.setattr(
        patch_dependencies, "coordinator", coordinator, raising=False
    )

    nick = StubNick(email_agent, supplier_agent)
    orchestrator = Orchestrator(nick)

    now = datetime.now(timezone.utc)
    payload = {
        "draft_payload": {
            "subject": "RFQ",
            "dispatch_freshness": {
                "uid-1": {
                    "min_dispatched_at": now.isoformat(),
                    "thread_index": 3,
                }
            },
            "draft_created_at": now.isoformat(),
            "round": 3,
        }
    }

    orchestrator.execute_workflow("supplier_interaction", payload)

    context = coordinator.calls[0]["dispatch_context"]
    assert context is not None
    uid1_context = context.get("uid-1")
    assert uid1_context is not None
    assert uid1_context.get("thread_index") == 3
    min_dt = uid1_context.get("min_dispatched_at")
    assert isinstance(min_dt, datetime)
    if "uid-2" in context:
        assert isinstance(context["uid-2"].get("min_dispatched_at"), datetime)


def test_supplier_workflow_realigns_draft_workflow_ids(monkeypatch):
    class DivergentEmailAgent:
        def __init__(self):
            self.calls = []

        def execute(self, context):
            self.calls.append(context.input_data)
            drafts = [
                {
                    "supplier_id": "S1",
                    "workflow_id": "legacy-1",
                    "unique_id": "uid-1",
                    "metadata": {"workflow_id": "legacy-1"},
                },
                {
                    "supplier_id": "S2",
                    "unique_id": "uid-2",
                    "metadata": {
                        "workflow_id": "legacy-2",
                        "context": {"workflow_id": "legacy-2"},
                    },
                },
            ]
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"drafts": drafts},
                pass_fields={"drafts": drafts},
            )

    email_agent = DivergentEmailAgent()
    supplier_agent = StubSupplierAgent()
    coordinator = StubCoordinator()

    monkeypatch.setattr(
        patch_dependencies, "coordinator", coordinator, raising=False
    )

    nick = StubNick(email_agent, supplier_agent)
    orchestrator = Orchestrator(nick)

    response = orchestrator.execute_workflow(
        "supplier_interaction",
        {"draft_payload": {"subject": "RFQ"}},
    )

    result = response.get("result", {})
    supplier_input = supplier_agent.calls[0]
    draft_workflow_ids = {
        draft.get("workflow_id") for draft in supplier_input.get("drafts", [])
    }
    assert len(draft_workflow_ids) == 1
    canonical_workflow_id = next(iter(draft_workflow_ids))
    assert canonical_workflow_id not in {"legacy-1", "legacy-2"}

    for draft in supplier_input.get("drafts", []):
        metadata = draft.get("metadata") or {}
        assert draft.get("workflow_id") == canonical_workflow_id
        assert metadata.get("workflow_id") == canonical_workflow_id
        context_meta = metadata.get("context") or {}
        assert context_meta.get("workflow_id") == canonical_workflow_id

    coordinator_call = coordinator.calls[0]
    assert coordinator_call["workflow_id"] == canonical_workflow_id
    assert sorted(coordinator_call["unique_ids"]) == ["uid-1", "uid-2"]
    assert coordinator_call["dispatch_context"] is None
    assert result["supplier_interaction"]["processed"] is True


def test_filter_drafts_for_workflow_respects_workflow_id():
    drafts = [
        {"workflow_id": "wf-keep", "unique_id": "uid-1"},
        {"metadata": {"workflow_id": "wf-keep"}, "unique_id": "uid-2"},
        {"workflow_id": "wf-drop", "unique_id": "uid-3"},
        {
            "unique_id": "uid-4",
            "metadata": {"context": {"workflow_id": "wf-drop"}},
        },
    ]

    filtered = Orchestrator._filter_drafts_for_workflow(drafts, "wf-keep")

    assert len(filtered) == 4
    for draft in filtered:
        workflow_value = draft.get("workflow_id")
        if not workflow_value and isinstance(draft.get("metadata"), dict):
            workflow_value = draft["metadata"].get("workflow_id")
        assert workflow_value == "wf-keep"


def test_filter_drafts_realigns_conflicting_workflow_ids():
    drafts = [
        {"workflow_id": "wf-a", "unique_id": "a1"},
        {
            "unique_id": "a2",
            "metadata": {"workflow_id": "wf-b", "context": {"workflow_id": "wf-b"}},
        },
    ]

    filtered = Orchestrator._filter_drafts_for_workflow(drafts, "wf-parent")

    assert len(filtered) == 2
    for draft in filtered:
        assert draft.get("workflow_id") == "wf-parent"
        metadata = draft.get("metadata") or {}
        assert metadata.get("workflow_id") == "wf-parent"
        context_meta = metadata.get("context") or {}
        assert context_meta.get("workflow_id") == "wf-parent"


def test_select_workflow_identifier_prefers_unique_draft_id():
    drafts = [
        {"workflow_id": "wf-dispatch", "unique_id": "uid-1"},
        {"unique_id": "uid-2", "metadata": {"workflow_id": "wf-dispatch"}},
    ]

    result = Orchestrator._select_workflow_identifier(drafts, "generated-workflow")

    assert result == "wf-dispatch"


def test_supplier_workflow_uses_dispatch_workflow_id_when_unique(monkeypatch):
    class SingleWorkflowEmailAgent:
        def __init__(self):
            self.calls = []

        def execute(self, context):
            self.calls.append(context.input_data)
            drafts = [
                {
                    "supplier_id": "S1",
                    "workflow_id": "wf-dispatch",
                    "unique_id": "uid-unique",
                }
            ]
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"drafts": drafts},
                pass_fields={"drafts": drafts},
            )

    email_agent = SingleWorkflowEmailAgent()
    supplier_agent = StubSupplierAgent()
    coordinator = StubCoordinator()

    monkeypatch.setattr(
        patch_dependencies, "coordinator", coordinator, raising=False
    )

    nick = StubNick(email_agent, supplier_agent)
    orchestrator = Orchestrator(nick)

    response = orchestrator.execute_workflow(
        "supplier_interaction", {"draft_payload": {"subject": "RFQ"}}
    )

    result = response.get("result", {})
    supplier_input = supplier_agent.calls[0]

    assert supplier_input.get("workflow_id") == "wf-dispatch"
    assert coordinator.calls[0]["workflow_id"] == "wf-dispatch"
    assert result["activation_monitor"]["activated"] is True
