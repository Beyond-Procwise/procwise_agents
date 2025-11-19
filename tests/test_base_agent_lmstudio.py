import os
import sys
from typing import Callable, List, Optional
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent


class StubLMStudioClient:
    def __init__(self, models=None):
        self._models = models or []
        self.chat_calls = []
        self.generate_calls = []

    def list_models(self):
        return list(self._models)

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return {"message": {"content": "ok"}, "response": "ok"}

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return {"response": "ok", "model": kwargs.get("model")}


class DummyProcessRoutingService:
    def __init__(self):
        self._counter = 0

    def log_process(self, **_kwargs):
        self._counter += 1
        return self._counter

    def log_run_detail(self, **_kwargs):
        return f"run-{self._counter}"

    def log_action(self, **_kwargs):
        return f"action-{self._counter}"

    def validate_workflow_id(self, *_args, **_kwargs):
        return True


class DummyAgentNick:
    def __init__(self, connection_factory: Optional[Callable[[], object]] = None):
        self.settings = SimpleNamespace(
            script_user="tester",
            extraction_model="gpt-oss",
        )
        self.prompt_engine = SimpleNamespace()
        self.learning_repository = None
        self.process_routing_service = DummyProcessRoutingService()
        self._connection_factory = connection_factory

    def lmstudio_options(self):
        return {}

    def get_db_connection(self):
        if self._connection_factory is None:
            raise AttributeError("No connection factory configured")
        return self._connection_factory()


def make_base_agent():
    return base_agent.BaseAgent(DummyAgentNick())


def test_call_lmstudio_prefers_fallback_when_list_empty(monkeypatch):
    stub_client = StubLMStudioClient(models=[])
    monkeypatch.setattr(base_agent, "get_lmstudio_client", lambda: stub_client)

    agent = make_base_agent()
    result = agent.call_lmstudio(prompt="hi", model="totally-missing")

    assert (
        stub_client.generate_calls[0]["model"]
        == base_agent._LMSTUDIO_FALLBACK_MODELS[0]
    )
    assert result["model"] == base_agent._LMSTUDIO_FALLBACK_MODELS[0]
    assert getattr(agent.agent_nick, "_available_lmstudio_models") == list(
        base_agent._LMSTUDIO_FALLBACK_MODELS
    )


def test_get_available_models_returns_fallback_when_empty(monkeypatch):
    stub_client = StubLMStudioClient(models=[])
    monkeypatch.setattr(base_agent, "get_lmstudio_client", lambda: stub_client)

    agent = make_base_agent()
    models = agent._get_available_lmstudio_models(force_refresh=True)

    assert models == list(base_agent._LMSTUDIO_FALLBACK_MODELS)


def test_prepare_logged_output_strips_knowledge_without_touching_original():
    agent = make_base_agent()
    payload = {
        "manifest": {
            "task": {"name": "rfq"},
            "knowledge": {"tables": ["proc.supplier"]},
        },
        "knowledge": {"tables": ["proc.action"]},
        "drafts": [
            {
                "content": "hello",
                "knowledge": {"debug": True},
            }
        ],
    }

    logged_output = agent._prepare_logged_output(payload)

    # Original payload is preserved for runtime consumers
    assert "knowledge" in payload
    assert "knowledge" in payload["manifest"]
    assert "knowledge" in payload["drafts"][0]

    # Logged payload removes heavy knowledge blobs at every level
    assert "knowledge" not in logged_output
    assert "knowledge" not in logged_output["manifest"]
    assert "knowledge" not in logged_output["drafts"][0]


def test_context_snapshot_strips_knowledge_but_preserves_context_base():
    class SnapshotAgent(base_agent.BaseAgent):
        def run(self, ctx):
            # Ensure knowledge remains available to the agent at runtime
            assert ctx.knowledge_base == {"tables": ["proc.invoice_agent"]}
            return base_agent.AgentOutput(
                status=base_agent.AgentStatus.SUCCESS,
                data={},
            )

    agent = SnapshotAgent(DummyAgentNick())

    context = base_agent.AgentContext(
        workflow_id="wf-knowledge",
        agent_id="agent-knowledge",
        user_id="user-knowledge",
        input_data={"prompt": "hello"},
    )
    context.apply_manifest(
        {
            "task": {"name": "rfq"},
            "knowledge": {"tables": ["proc.invoice_agent"]},
        }
    )

    result = agent.execute(context)

    snapshot = result.data.get("context_snapshot")
    assert snapshot is not None
    assert "knowledge" not in snapshot
    assert "knowledge" not in snapshot.get("manifest", {})

    # Underlying manifest supplied to the context remains unchanged
    assert context.manifest()["knowledge"] == {"tables": ["proc.invoice_agent"]}


def test_execute_persists_agentic_plan(monkeypatch):
    executed_statements: List[str] = []
    executed_params: List[Optional[tuple]] = []
    commit_calls: List[bool] = []

    class RecordingCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            executed_statements.append(statement.strip())
            executed_params.append(params)

    class RecordingConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return RecordingCursor()

        def commit(self):
            commit_calls.append(True)

    def connection_factory():
        return RecordingConnection()

    class PlanningAgent(base_agent.BaseAgent):
        AGENTIC_PLAN_STEPS = ("Gather inputs", "Compute outputs")

        def run(self, context):
            return base_agent.AgentOutput(
                status=base_agent.AgentStatus.SUCCESS,
                data={},
            )

    agent = PlanningAgent(DummyAgentNick(connection_factory=connection_factory))

    context_one = base_agent.AgentContext(
        workflow_id="wf-1",
        agent_id="agent-1",
        user_id="user-1",
        input_data={},
    )
    context_two = base_agent.AgentContext(
        workflow_id="wf-2",
        agent_id="agent-1",
        user_id="user-1",
        input_data={},
    )

    agent.execute(context_one)
    agent.execute(context_two)

    create_statements = [
        stmt for stmt in executed_statements if "CREATE TABLE IF NOT EXISTS proc.agent_plan" in stmt
    ]
    insert_statements = [
        stmt for stmt in executed_statements if "INSERT INTO proc.agent_plan" in stmt
    ]

    assert len(create_statements) == 1
    assert len(insert_statements) == 2
    assert len(commit_calls) == 2

    insert_params = [params for params in executed_params if isinstance(params, tuple)]
    assert len(insert_params) == 2

    expected_plan = "1. Gather inputs\n2. Compute outputs"
    first_params = insert_params[0]
    second_params = insert_params[1]

    assert first_params[0] == "wf-1"
    assert first_params[1] == "agent-1"
    assert first_params[2] == "PlanningAgent"
    assert first_params[3] == "action-1"
    assert first_params[4] == expected_plan

    assert second_params[0] == "wf-2"
    assert second_params[3] == "action-2"
    assert second_params[4] == expected_plan
