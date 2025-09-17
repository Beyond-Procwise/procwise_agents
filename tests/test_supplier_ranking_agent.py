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
        self.settings = SimpleNamespace(
            extraction_model="llama3", script_user="tester"
        )
        # Minimal query engine stub for agent initialisation
        self.query_engine = SimpleNamespace(
            fetch_supplier_data=lambda *_: []
        )


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


def test_supplier_ranking_trains_when_possible(monkeypatch):
    class Nick(DummyNick):
        def __init__(self):
            super().__init__()
            self.trained = False

            def train():
                self.trained = True

            self.query_engine = SimpleNamespace(
                fetch_supplier_data=lambda *_: pd.DataFrame(
                    {"supplier_name": ["S1"], "price": [1]}
                ),
                train_procurement_context=train,
            )

    nick = Nick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="u1",
        input_data={
            "intent": {"parameters": {"criteria": ["price"]}},
            "query": "Rank suppliers by price",
        },
    )

    agent.run(context)

    assert nick.trained is True


def test_supplier_ranking_injects_missing_candidates(monkeypatch):
    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0}}},
                },
                {"policyName": "CategoricalScoringPolicy", "details": {"rules": {}}},
                {
                    "policyName": "NormalizationDirectionPolicy",
                    "details": {"rules": {"price": "lower_is_better"}},
                },
            ]

    class StubQueryEngine:
        def fetch_purchase_order_data(self):
            return pd.DataFrame()

        def fetch_invoice_data(self):
            return pd.DataFrame()

        def fetch_procurement_flow(self, embed: bool = False):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="llama3", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(agent, "_read_table", lambda table: pd.DataFrame())

    supplier_df = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Alpha"],
            "avg_unit_price": [10.0],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": supplier_df,
            "supplier_candidates": ["S1", "S2"],
            "supplier_directory": [
                {"supplier_id": "S1", "supplier_name": "Alpha"},
                {"supplier_id": "S2", "supplier_name": "Beta"},
            ],
            "intent": {"parameters": {"criteria": ["price"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    ranking_ids = {entry["supplier_id"] for entry in output.data["ranking"]}
    assert ranking_ids == {"S1", "S2"}
    assert any(entry["supplier_name"] == "Beta" for entry in output.data["ranking"])


def test_supplier_ranking_limits_to_opportunity_directory(monkeypatch):
    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0}}},
                },
                {"policyName": "CategoricalScoringPolicy", "details": {"rules": {}}},
                {
                    "policyName": "NormalizationDirectionPolicy",
                    "details": {"rules": {"price": "lower_is_better"}},
                },
            ]

    class StubQueryEngine:
        def fetch_purchase_order_data(self):
            return pd.DataFrame()

        def fetch_invoice_data(self):
            return pd.DataFrame()

        def fetch_procurement_flow(self, embed: bool = False):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="llama3", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(agent, "_read_table", lambda table: pd.DataFrame())

    supplier_df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["Alpha", "Beta", "Gamma"],
            "avg_unit_price": [10.0, 8.0, 12.0],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": supplier_df,
            "supplier_directory": [
                {"supplier_id": "S1", "supplier_name": "Alpha"},
                {"supplier_id": "S2", "supplier_name": "Beta"},
            ],
            "intent": {"parameters": {"criteria": ["price"], "top_n": 3}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    ranking_ids = {entry["supplier_id"] for entry in output.data["ranking"]}
    assert ranking_ids == {"S1", "S2"}
    assert all(entry["supplier_id"] in {"S1", "S2"} for entry in output.data["ranking"])
