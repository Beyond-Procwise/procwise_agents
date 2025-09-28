import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from services.process_routing_service import ProcessRoutingService


class EchoAgent:
    def execute(self, context):
        return AgentOutput(status=AgentStatus.SUCCESS, data={"result": context.input_data.get("number")})


def test_json_flow_executes_steps_with_context_mapping():
    agent = EchoAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"echo": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)

    flow = {
        "entrypoint": "step1",
        "steps": {
            "step1": {
                "agent": "echo",
                "input": {"number": "{{ payload.value }}"},
                "outputs": {"calc": "$.result"},
                "next": "step2",
            },
            "step2": {
                "agent": "echo",
                "condition": "{{ ctx.calc == 1 }}",
                "input": {"number": "{{ ctx.calc + 1 }}"},
                "outputs": {"final": "$.result"},
            },
        },
    }

    result = orchestrator.execute_agent_flow(flow, {"value": 1})
    assert result["status"] == 100
    assert result["ctx"]["calc"] == 1
    assert result["ctx"]["final"] == 2


def test_json_flow_inherits_payload_when_input_missing():
    class EchoAgent:
        def __init__(self):
            self.received = None

        def execute(self, context):
            self.received = context.input_data
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    agent = EchoAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"echo": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)

    flow = {
        "entrypoint": "step1",
        "steps": {"step1": {"agent": "echo"}},
    }

    payload = {"foo": "bar"}
    orchestrator.execute_agent_flow(flow, payload)
    assert agent.received.get("foo") == "bar"
    assert agent.received.get("llm") == ProcessRoutingService.DEFAULT_LLM_MODEL
    assert set(agent.received.keys()) == {"foo", "llm"}


def test_json_flow_injects_proc_agent_prompts_and_policies():
    captured: Dict[str, Any] = {}

    class CaptureAgent:
        def execute(self, context):
            captured["input"] = dict(context.input_data)
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    class DummyPRS:
        def __init__(self):
            self._agent_defaults_cache = {
                "opportunity_miner": {
                    "llm": "gpt-oss",
                    "prompts": [50],
                    "policies": [9],
                }
            }

        def _load_agent_links(self):  # pragma: no cover - cache primed
            return {}, {}, {}

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"opportunity_miner": CaptureAgent()},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        process_routing_service=DummyPRS(),
    )

    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "opportunity_miner": "OpportunityMinerAgent"
    }
    orchestrator._load_prompts = lambda: {
        50: {"promptId": 50, "template": "from-db"}
    }
    orchestrator._load_policies = lambda: {
        9: {"policyId": 9, "policyName": "VolumeDiscount", "policy_desc": "{}"}
    }

    flow = {
        "entrypoint": "start",
        "steps": {
            "start": {
                "agent": "opportunity_miner",
            }
        },
    }

    orchestrator.execute_agent_flow(flow)

    input_data = captured.get("input", {})
    prompts = input_data.get("prompts") or []
    policies = input_data.get("policies") or []
    assert input_data.get("llm") == "gpt-oss"
    assert any(p.get("promptId") == 50 for p in prompts)
    assert any(p.get("policyId") == 9 for p in policies)


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
    orchestrator._load_policies = lambda: {1: {"policyId": 1, "policyName": "ExamplePolicy"}}
    flow = {
        "status": "saved",
        "agent_type": "data_extraction",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
    }
    result = orchestrator.execute_agent_flow(flow)
    assert result["status"] == 100
    assert flow["status"] == 100
    assert agent.ran is True


def test_json_flow_handles_decorated_agent_names():
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
    flow = {
        "entrypoint": "step1",
        "steps": {
            "step1": {"agent": "user_quote_evaluation_agent_1"}
        },
    }
    orchestrator.execute_agent_flow(flow)
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

    assert flow["status"] == 0
    assert flow["onFailure"]["status"] == "completed"


def test_execute_agent_flow_returns_failure_if_any_step_fails():
    class ConfigurableAgent:
        def __init__(self, status):
            self.status = status

        def execute(self, context):  # pragma: no cover - simple stub
            return AgentOutput(status=self.status, data={})

    first = ConfigurableAgent(AgentStatus.SUCCESS)
    second = ConfigurableAgent(AgentStatus.FAILED)

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"first": first, "second": second},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
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
        "onSuccess": {
            "status": "saved",
            "agent_type": "second",
            "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
        },
    }

    result = orchestrator.execute_agent_flow(flow)
    assert result["status"] == 0


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

    assert flow["status"] == 100
    assert agent.ran is True


def test_ranking_workflow_runs_full_supplier_flow():
    executions = []

    class StubOpportunityAgent:
        def execute(self, context):
            executions.append(("opportunity_miner", context.input_data.get("workflow")))
            findings = [
                {"category_id": "Raw Materials", "financial_impact_gbp": 1500.0},
                {"category_id": "Services", "financial_impact_gbp": 500.0},
            ]
            payload = {
                "supplier_candidates": ["S1", "S2"],
                "findings": findings,
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=payload,
                pass_fields=payload,
            )

    class StubRankingAgent:
        def __init__(self):
            self.candidates = None
            self.directory = None

        def execute(self, context):
            supplier_candidates = context.input_data.get("supplier_candidates")
            directory = context.input_data.get("supplier_directory")
            executions.append(("supplier_ranking", supplier_candidates))
            self.candidates = supplier_candidates
            self.directory = directory
            ranking = [
                {"supplier_id": sid, "supplier_name": f"Supplier {sid}"}
                for sid in supplier_candidates
            ]
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"ranking": ranking},
                pass_fields={"ranking": ranking},
            )

    class StubQuoteAgent:
        def __init__(self):
            self.ranking_seen = None
            self.category_seen = None


        def execute(self, context):
            ranking = context.input_data.get("ranking")
            executions.append(("quote_evaluation", ranking))
            self.ranking_seen = ranking
            self.category_seen = context.input_data.get("product_category")

            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"quotes": ["Q1", "Q2"]},
                pass_fields={"quotes": ["Q1", "Q2"]},
            )

    class AllowAllPolicy:
        def validate_workflow(self, *args, **kwargs):
            return {"allowed": True}

    class StubQueryEngine:
        def fetch_supplier_data(self, *_):
            return [{"supplier_id": "S1"}, {"supplier_id": "S2"}]

    ranking_agent = StubRankingAgent()
    quote_agent = StubQuoteAgent()

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={
            "opportunity_miner": StubOpportunityAgent(),
            "supplier_ranking": ranking_agent,
            "quote_evaluation": quote_agent,
        },
        policy_engine=AllowAllPolicy(),
        query_engine=StubQueryEngine(),
        routing_engine=SimpleNamespace(routing_model={}),
    )

    orchestrator = Orchestrator(nick)
    payload = {
        "workflow": "contract_expiry_check",
        "conditions": {"negotiation_window_days": 90},
    }

    result = orchestrator.execute_workflow("supplier_ranking", payload)

    assert result["status"] == "completed"
    assert [step[0] for step in executions] == [
        "opportunity_miner",
        "supplier_ranking",
        "quote_evaluation",
    ]
    assert ranking_agent.candidates == ["S1", "S2"]
    assert ranking_agent.directory is None
    assert quote_agent.ranking_seen[0]["supplier_id"] == "S1"
    assert quote_agent.category_seen == "Raw Materials"

    assert result["result"]["opportunities"]["supplier_candidates"] == ["S1", "S2"]
    assert result["result"]["opportunities"]["product_category"] == "Raw Materials"
    assert result["result"]["ranking"]["ranking"][0]["supplier_id"] == "S1"
    assert result["result"]["downstream_results"]["quote_evaluation"]["quotes"] == [
        "Q1",
        "Q2",
    ]


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

    assert flow["status"] == 100
    assert agent.ran is True


def test_inject_agent_instructions_uses_agent_defaults():
    class DummyPRS:
        def __init__(self):
            self._agent_defaults_cache = {
                "opportunity_miner": {
                    "llm": "gpt-oss",
                    "prompts": [50],
                    "policies": [9],
                    "conditions": {"negotiation_window_days": 45},
                }
            }

        def _load_agent_links(self):  # pragma: no cover - not triggered
            return {}, {}, {}

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1, parallel_processing=False),
        agents={"opportunity_miner": DummyAgent()},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        process_routing_service=DummyPRS(),
    )

    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "opportunity_miner": "OpportunityMinerAgent"
    }
    orchestrator._load_prompts = lambda: {
        50: {"promptId": 50, "template": "workflow: dynamic"}
    }
    orchestrator._load_policies = lambda: {
        9: {"policyId": 9, "policyName": "VolumeDiscount", "policy_desc": "{}"}
    }

    payload = {"conditions": {"existing": True}}
    orchestrator._inject_agent_instructions("opportunity_miner", payload)

    assert payload["llm"] == "gpt-oss"
    assert payload["conditions"]["negotiation_window_days"] == 45
    assert payload["conditions"]["existing"] is True
    assert any(p.get("promptId") == 50 for p in payload["prompts"])
    assert any(p.get("policyId") == 9 for p in payload["policies"])


def test_supplier_ranking_flow_applies_agent_prompts_and_policies():
    seen: Dict[str, Any] = {}

    class CaptureOpportunityAgent:
        def execute(self, context):
            seen["opportunity_prompts"] = context.input_data.get("prompts")
            seen["opportunity_policies"] = context.input_data.get("policies")
            pass_fields = {
                "supplier_candidates": ["S1"],
                "findings": [],
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=pass_fields,
                pass_fields=pass_fields,
            )

    class CaptureRankingAgent:
        def execute(self, context):
            seen["ranking_prompts"] = context.input_data.get("prompts")
            seen["ranking_policies"] = context.input_data.get("policies")
            ranking_payload = {"ranking": [{"supplier_id": "S1"}]}
            pass_fields = {
                "ranking": ranking_payload,
                "supplier_candidates": context.input_data.get("supplier_candidates", []),
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"ranking": ranking_payload},
                pass_fields=pass_fields,
            )

    class CaptureEmailAgent:
        def execute(self, context):
            seen["email_prompts"] = context.input_data.get("prompts")
            seen["email_policies"] = context.input_data.get("policies")
            return AgentOutput(status=AgentStatus.SUCCESS, data={"drafts": []})

    class DummyPRS:
        def __init__(self):
            self._agent_defaults_cache = {
                "opportunity_miner": {"prompts": [50], "policies": [9]},
                "supplier_ranking": {"prompts": [200], "policies": [25]},
            }

        def _load_agent_links(self):  # pragma: no cover - cache already primed
            return {}, {}, {}

    class AllowAllPolicy:
        def validate_workflow(self, *_, **__):  # pragma: no cover - simple stub
            return {"allowed": True}

    class StubQueryEngine:
        def fetch_supplier_data(self, *_):  # pragma: no cover - simple stub
            return [{"supplier_id": "S1"}]

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1, parallel_processing=False),
        agents={
            "opportunity_miner": CaptureOpportunityAgent(),
            "supplier_ranking": CaptureRankingAgent(),
            "email_drafting": CaptureEmailAgent(),
        },
        policy_engine=AllowAllPolicy(),
        query_engine=StubQueryEngine(),
        routing_engine=SimpleNamespace(routing_model={}),
        process_routing_service=DummyPRS(),
    )

    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "opportunity_miner": "OpportunityMinerAgent",
        "supplier_ranking": "SupplierRankingAgent",
        "email_drafting": "EmailDraftingAgent",
    }
    orchestrator._load_prompts = lambda: {
        50: {"promptId": 50, "template": "workflow: contract_expiry_check"},
        200: {"promptId": 200, "template": "criteria: risk"},
    }
    orchestrator._load_policies = lambda: {
        9: {"policyId": 9, "policyName": "VolumeDiscount", "policy_desc": "{}"},
        25: {
            "policyId": 25,
            "policyName": "SupplierRanking",
            "policy_desc": "{\"rules\": {}}",
        },
    }

    result = orchestrator.execute_workflow("supplier_ranking", {"query": "top 1"})

    assert result["status"] == "completed"
    opportunity_prompts = seen.get("opportunity_prompts") or []
    ranking_prompts = seen.get("ranking_prompts") or []
    assert any(p.get("promptId") == 50 for p in opportunity_prompts)
    assert any(p.get("promptId") == 200 for p in ranking_prompts)
    opportunity_policies = seen.get("opportunity_policies") or []
    ranking_policies = seen.get("ranking_policies") or []
    assert any(p.get("policyId") == 9 for p in opportunity_policies)
    assert any(p.get("policyId") == 25 for p in ranking_policies)


def test_execute_agent_flow_passes_fields_to_children():
    class ParentAgent:
        def execute(self, context):
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={},
                pass_fields={"shared": "value"},
            )

    class ChildAgent:
        def __init__(self):
            self.seen = None

        def execute(self, context):  # pragma: no cover - simple stub
            self.seen = context.input_data.get("shared")
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    parent = ParentAgent()
    child = ChildAgent()

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"parent": parent, "child": child},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )

    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "parent": "ParentAgent",
        "child": "ChildAgent",
    }
    orchestrator._load_prompts = lambda: {}
    orchestrator._load_policies = lambda: {}

    flow = {
        "status": "saved",
        "agent_type": "parent",
        "agent_property": {},
        "onSuccess": {
            "status": "saved",
            "agent_type": "child",
            "agent_property": {},
        },
    }

    orchestrator.execute_agent_flow(flow)

    assert flow["status"] == 100
    assert child.seen == "value"


def test_execute_agent_flow_accepts_class_name():
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
        "agent_type": "QuoteEvaluationAgent",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
    }

    orchestrator.execute_agent_flow(flow)

    assert flow["status"] == 100
    assert agent.ran is True


def test_execute_legacy_flow_injects_workflow_metadata():
    captured = {}

    class CaptureAgent:
        def execute(self, context):
            captured["input"] = context.input_data
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    agent = CaptureAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"opportunity_miner": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "opportunity_miner": "OpportunityMinerAgent"
    }
    orchestrator._load_prompts = lambda: {}
    orchestrator._load_policies = lambda: {}

    flow = {
        "status": "saved",
        "agent_type": "opportunity_miner",
        "agent_property": {"workflow": "price_variance_check"},
    }

    orchestrator.execute_agent_flow(flow)

    assert captured["input"]["workflow"] == "price_variance_check"


def test_execute_legacy_flow_leaves_workflow_unset_for_opportunity_miner():
    captured = {}

    class CaptureAgent:
        def execute(self, context):
            captured["input"] = context.input_data
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"opportunity_miner": CaptureAgent()},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "opportunity_miner": "OpportunityMinerAgent"
    }
    orchestrator._load_prompts = lambda: {}
    orchestrator._load_policies = lambda: {}

    flow = {
        "status": "saved",
        "agent_type": "opportunity_miner",
        "agent_property": {},
    }

    orchestrator.execute_agent_flow(flow, {"ranking": []})

    assert "workflow" not in captured["input"]
    conditions = captured["input"].get("conditions")
    assert not conditions or "negotiation_window_days" not in conditions

def test_convert_agents_to_flow_promotes_root_workflow():
    details = {
        "status": "saved",
        "workflow": "price_variance_check",
        "agents": [
            {
                "agent": "OpportunityMinerAgent",
                "status": "saved",
                "agent_type": "opportunity_miner",
                "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []},
                "agent_property": {"llm": None, "workflow": None},
            }
        ],
    }

    flow = ProcessRoutingService.convert_agents_to_flow(details)

    assert flow["agent_property"]["workflow"] == "price_variance_check"
    assert flow["workflow"] == "price_variance_check"


def test_execute_workflow_promotes_falsy_workflow_value():
    captured = {}

    class CaptureAgent:
        def execute(self, context):
            captured["input"] = context.input_data
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"supplier_candidates": []},
            )

    policy_engine = SimpleNamespace(
        validate_workflow=lambda *args, **kwargs: {"allowed": True}
    )
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"opportunity_miner": CaptureAgent()},
        policy_engine=policy_engine,
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)

    result = orchestrator.execute_workflow("opportunity_mining", {"workflow": None})

    assert result["status"] == "completed"
    assert captured["input"]["workflow"] == "opportunity_mining"
    conditions = captured["input"].get("conditions")
    assert not conditions or "negotiation_window_days" not in conditions


def test_create_child_context_preserves_parent_workflow():
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)

    parent = AgentContext(
        workflow_id="wf-id",
        agent_id="root",
        user_id="tester",
        input_data={"workflow": "price_variance_check"},
    )

    child = orchestrator._create_child_context(
        parent,
        "child_agent",
        {"workflow": None, "extra": "value"},
    )

    assert child.input_data["workflow"] == "price_variance_check"
    assert child.input_data["extra"] == "value"


def test_json_flow_uses_flow_level_workflow_hint():
    captured = {}

    class CaptureAgent:
        def execute(self, context):
            captured["input"] = context.input_data
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"opportunity_miner": CaptureAgent()},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    orchestrator._load_agent_definitions = lambda: {
        "opportunity_miner": "OpportunityMinerAgent"
    }
    orchestrator._load_prompts = lambda: {}
    orchestrator._load_policies = lambda: {}

    flow = {
        "workflow": "price_variance_check",
        "entrypoint": "start",
        "steps": {
            "start": {
                "agent": "opportunity_miner",
            }
        },
    }

    orchestrator.execute_agent_flow(flow, {})

    assert captured["input"]["workflow"] == "price_variance_check"

