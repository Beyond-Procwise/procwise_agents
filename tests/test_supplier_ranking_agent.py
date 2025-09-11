import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.supplier_ranking_agent import SupplierRankingAgent
from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration.orchestrator import Orchestrator


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.settings = SimpleNamespace(extraction_model="llama3", script_user="tester")


def test_top_n_parsed_from_query(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame({
        "supplier_name": [f"S{i}" for i in range(6)],
        "price": [60, 50, 40, 30, 20, 10],
    })

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="u1",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price"]}},
            "query": "Rank top 5 suppliers by price",
        },
    )

    output = agent.run(context)
    assert len(output.data["ranking"]) == 5


def test_execute_ranking_flow_extracts_criteria_from_query():
    """Orchestrator should derive ranking criteria from the free-text query."""

    class DummyPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0, "delivery": 1.0}}},
                }
            ]
            self.last_input = None

        def validate_workflow(self, workflow_name, user_id, input_data):
            self.last_input = input_data
            return {"allowed": True, "reason": ""}

    class DummyQueryEngine:
        def fetch_supplier_data(self, intent):  # pragma: no cover - simple stub
            return pd.DataFrame([{"supplier_name": "S1", "price": 10, "delivery": 8}])

    class DummyRankingAgent:
        def execute(self, context):  # pragma: no cover - simple stub
            assert context.input_data["intent"]["parameters"]["criteria"] == ["price"]
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"supplier_ranking": DummyRankingAgent()},
        policy_engine=DummyPolicyEngine(),
        query_engine=DummyQueryEngine(),
        routing_engine=SimpleNamespace(routing_model={}),
    )

    orchestrator = Orchestrator(nick)
    result = orchestrator.execute_ranking_flow("Rank suppliers by price")

    assert result["status"] == "completed"
    assert nick.policy_engine.last_input["criteria"] == ["price"]
