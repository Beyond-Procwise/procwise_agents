import json
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


def _supplier_policy_rows():
    weight_details = {
        "rules": {
            "default_weights": {
                "price": 0.4,
                "delivery": 0.3,
                "risk": 0.2,
                "payment_terms": 0.1,
            }
        }
    }
    categorical_details = {
        "rules": {
            "payment_terms": {
                "Net 30": 10,
                "Net 45": 8,
                "Net 60": 6,
                "default": 5,
            }
        }
    }
    normalization_details = {
        "rules": {
            "price": "lower_is_better",
            "delivery": "higher_is_better",
            "risk": "lower_is_better",
            "payment_terms": "higher_is_better",
        }
    }
    return [
        {
            "policy_id": 201,
            "policy_name": "WeightAllocationPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Default supplier ranking weights",
            "policy_details": json.dumps(weight_details),
        },
        {
            "policy_id": 202,
            "policy_name": "CategoricalScoringPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Categorical scoring rules",
            "policy_details": json.dumps(categorical_details),
        },
        {
            "policy_id": 203,
            "policy_name": "NormalizationDirectionPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Normalization direction rules",
            "policy_details": json.dumps(normalization_details),
        },
    ]


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine(policy_rows=_supplier_policy_rows())
        self.settings = SimpleNamespace(
            extraction_model="gpt-oss", script_user="tester"
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
    assert output.data["rank_count"] == 5
    assert output.data["ranking"][0]["rank_position"] == 1
    assert all(entry["rank_count"] == 5 for entry in output.data["ranking"])


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
        def fetch_purchase_order_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_invoice_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_procurement_flow(
            self,
            embed: bool = False,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(agent, "_read_table", lambda *args, **kwargs: pd.DataFrame())

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


def test_supplier_ranking_normalises_weights_to_available_metrics(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(
        agent,
        "_build_supplier_profiles",
        lambda _tables, ids: {str(s): {} for s in ids},
    )
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "price": [50.0, 40.0],
            "risk": [5.0, 3.0],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price", "delivery", "risk"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    weights = output.data["ranking"][0]["weights"]
    assert "price" in weights and "risk" in weights
    assert weights["price"] > 0
    assert weights["risk"] > 0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert output.data["ranking"][0]["final_score"] > 0


def test_supplier_ranking_includes_flow_coverage_from_snapshot(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(
        agent,
        "_build_supplier_profiles",
        lambda _tables, ids: {str(s): {} for s in ids},
    )
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Alpha"],
            "price": [50.0],
        }
    )

    snapshot = {
        "supplier_flows": [
            {
                "supplier_id": "S1",
                "supplier_name": "Alpha",
                "coverage_ratio": 0.6,
                "purchase_orders": {"count": 2},
            }
        ]
    }

    context = AgentContext(
        workflow_id="wf2",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "data_flow_snapshot": snapshot,
            "intent": {"parameters": {"criteria": ["price"], "top_n": 1}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    ranking_entry = output.data["ranking"][0]
    assert ranking_entry["flow_coverage"] == pytest.approx(0.6)
    assert ranking_entry["final_score"] > 0


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
        def fetch_purchase_order_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_invoice_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_procurement_flow(
            self,
            embed: bool = False,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(agent, "_read_table", lambda *args, **kwargs: pd.DataFrame())

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


def test_ranking_uses_context_policy_weights(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "delivery": [9, 6],
            "price": [12.0, 8.0],
            "risk": [4.0, 2.0],
        }
    )

    policies = [
        {
            "policyName": "WeightAllocationPolicy",
            "details": {
                "rules": {"default_weights": {"delivery": 1.0, "price": 0.0, "risk": 0.0}}
            },
        },
        {
            "policyName": "NormalizationDirectionPolicy",
            "details": {"rules": {"delivery": "higher_is_better"}},
        },
    ]

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="tester",
        input_data={
            "supplier_data": df,
            "policies": policies,
            "intent": {"parameters": {"criteria": ["delivery"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["ranking"][0]["supplier_id"] == "S1"
    assert output.data["ranking"][0]["weights"]["delivery"] == 1.0


def test_supplier_ranking_instruction_overrides(monkeypatch):
    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = []

    class StubQueryEngine:
        def fetch_supplier_data(self, *_args, **_kwargs):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(agent, "_build_supplier_profiles", lambda _tables, ids: {sid: {} for sid in ids})
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["Alpha", "Beta", "Gamma"],
            "price": [50, 40, 60],
            "delivery": [5, 10, 7],
        }
    )

    prompts = [
        {
            "promptId": 1,
            "prompts_desc": "{\"criteria\": [\"price\"], \"top_n\": 2}",
        }
    ]

    policies = [
        {
            "policyId": 10,
            "policyName": "WeightAllocationPolicy",
            "details": {"rules": {"default_weights": {"price": 1.0}}},
        },
        {
            "policyId": 11,
            "policyName": "NormalizationDirectionPolicy",
            "details": {"rules": {"price": "lower_is_better"}},
        },
    ]

    context = AgentContext(
        workflow_id="wf-instruction",
        agent_id="supplier_ranking",
        user_id="tester",
        input_data={
            "supplier_data": df,
            "prompts": prompts,
            "policies": policies,
            "intent": {},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    ranking = output.data["ranking"]
    assert len(ranking) == 2
    assert ranking[0]["supplier_id"] == "S2"
    weights = ranking[0]["weights"]
    assert weights["price"] == pytest.approx(1.0)
    assert context.input_data["intent"].get("parameters", {}).get("criteria") == ["price"]

