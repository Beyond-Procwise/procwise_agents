import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.negotiation_agent import NegotiationAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            qdrant_collection_name="dummy",
            extraction_model="llama3",
            script_user="tester",
            ses_default_sender="noreply@example.com",
        )
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: None,
            log_action=lambda **_: None,
        )
        self.ollama_options = lambda: {}
        self.qdrant_client = SimpleNamespace()
        self.embedding_model = SimpleNamespace(encode=lambda x: [0.0])
        def get_db_connection():
            class DummyConn:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def cursor(self):
                    class DummyCursor:
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                        def execute(self, *args, **kwargs): pass
                        def fetchone(self): return None
                    return DummyCursor()
            return DummyConn()
        self.get_db_connection = get_db_connection



def test_negotiation_agent_handles_missing_fields():
    nick = DummyNick()
    agent = NegotiationAgent(nick)
    context = AgentContext(
        workflow_id="wf1",
        agent_id="negotiation",
        user_id="u1",
        input_data={},
    )
    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["counter_proposals"] == []


def test_negotiation_agent_builds_contextual_prompt(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    captured = {}

    supplier_context = {
        "profile": {
            "supplier_name": "Acme Parts",
            "default_currency": "GBP",
            "risk_score": "Low",
            "delivery_lead_time_days": "5",
            "is_preferred_supplier": True,
        },
        "spend": {"total_amount": 120000.0, "po_count": 12},
        "contracts": [
            {
                "contract_id": "CO0001",
                "contract_end_date": None,
                "total_contract_value": 250000.0,
            }
        ],
    }

    monkeypatch.setattr(agent, "_load_supplier_context", lambda *args, **kwargs: supplier_context)

    def fake_call(**kwargs):
        captured["prompt"] = kwargs.get("prompt")
        return {"response": "Revised counter proposal."}

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    context = AgentContext(
        workflow_id="wf2",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S1",
            "current_offer": 1500.0,
            "target_price": 1200.0,
            "rfq_id": "RFQ-123",
            "round": 1,
            "item_id": "ITEM-1",
            "benchmark_price": 1150.0,
        },
    )

    output = agent.run(context)
    prompt = captured.get("prompt", "")

    assert "Acme Parts" in prompt
    assert "Historic purchase orders total" in prompt
    assert "risk rating" in prompt.lower()
    assert output.data["message"] == "Revised counter proposal."


def test_negotiation_agent_generates_fallback_message(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    supplier_context = {
        "profile": {"supplier_name": "Omega", "default_currency": "USD"},
        "spend": {"total_amount": 54000.0},
        "contracts": [],
    }

    monkeypatch.setattr(agent, "_load_supplier_context", lambda *args, **kwargs: supplier_context)
    monkeypatch.setattr(agent, "call_ollama", lambda **_: {"response": ""})

    context = AgentContext(
        workflow_id="wf3",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S2",
            "current_offer": 1800.0,
            "target_price": 1500.0,
            "rfq_id": "RFQ-789",
            "round": 2,
            "item_id": "COMP-5",
        },
    )

    output = agent.run(context)
    message = output.data["message"]

    assert "pricing closer" in message
    assert "Historic collaboration" in message
    assert output.data["decision_log"].startswith("Targeting 1500.0 against current offer")
