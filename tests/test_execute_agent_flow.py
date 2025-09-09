import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from agents.base_agent import AgentOutput, AgentStatus


class DummyAgent:
    def __init__(self):
        self.ran = False

    def execute(self, context):  # pragma: no cover - simple stub
        self.ran = True
        return AgentOutput(status=AgentStatus.SUCCESS, data={})


def test_execute_agent_flow_runs_and_updates_status():
    agent = DummyAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"data_extraction": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "data_extraction": "DataExtractionAgent"
    }
    orchestrator._load_prompts = lambda: {1: {}}
    orchestrator._load_policies = lambda: {1: {}}
    flow = {
        "status": "saved",
        "agent_type": "data_extraction",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
    }
    result = orchestrator.execute_agent_flow(flow)
    assert result["status"] == "completed"
    assert flow["status"] == "completed"
    assert agent.ran is True


def test_execute_agent_flow_preserves_root_status_on_failure_chain():
    """If the first agent fails, the overall flow should remain failed even
    if a downstream agent succeeds."""

    class ConfigurableAgent:
        def __init__(self, status):
            self.status = status

        def execute(self, context):  # pragma: no cover - simple stub
            return AgentOutput(status=self.status, data={})

    fail_agent = ConfigurableAgent(AgentStatus.FAILED)
    success_agent = ConfigurableAgent(AgentStatus.SUCCESS)

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"first": fail_agent, "second": success_agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)

    # Stub out lookups so our dummy agents are used
    orchestrator._load_agent_definitions = lambda: {
        "first": "FirstAgent",
        "second": "SecondAgent",
    }
    orchestrator._load_prompts = lambda: {1: {}}
    orchestrator._load_policies = lambda: {1: {}}

    flow = {
        "status": "saved",
        "agent_type": "first",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
        "onFailure": {
            "status": "saved",
            "agent_type": "second",
            "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
        },
    }

    orchestrator.execute_agent_flow(flow)

    assert flow["status"] == "failed"
    assert flow["onFailure"]["status"] == "completed"


def test_execute_agent_flow_ignores_agent_id_suffix():
    agent = DummyAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"supplier_ranking": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "supplier_ranking": "SupplierRankingAgent"
    }
    orchestrator._load_prompts = lambda: {1: {}}
    orchestrator._load_policies = lambda: {1: {}}

    flow = {
        "status": "saved",
        "agent_type": "supplier_ranking_000055_1757337997564",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
    }

    orchestrator.execute_agent_flow(flow)

    assert flow["status"] == "completed"
    assert agent.ran is True


def test_execute_agent_flow_handles_prefixed_agent_names():
    agent = DummyAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"quote_evaluation": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "quote_evaluation": "QuoteEvaluationAgent"
    }
    orchestrator._load_prompts = lambda: {1: {}}
    orchestrator._load_policies = lambda: {1: {}}

    flow = {
        "status": "saved",
        "agent_type": "admin_quote_agent_000067_1757404210002",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
    }

    orchestrator.execute_agent_flow(flow)

    assert flow["status"] == "completed"
    assert agent.ran is True
