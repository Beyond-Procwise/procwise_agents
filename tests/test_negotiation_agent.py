import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

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
            extraction_model="gpt-oss",
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
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def cursor(self):
                    class DummyCursor:
                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            pass

                        def execute(self, *args, **kwargs):
                            pass

                        def fetchone(self):
                            return None

                        def fetchall(self):
                            return []

                    return DummyCursor()

                def commit(self):
                    pass

            return DummyConn()

        self.get_db_connection = get_db_connection


def test_negotiation_agent_handles_missing_fields(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    monkeypatch.setattr(
        agent,
        "call_ollama",
        lambda **_: {"message": {"content": "Please share the quote details."}},
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="negotiation",
        user_id="u1",
        input_data={},
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["counter_proposals"] == []
    assert output.data["decision"]["strategy"] == "clarify"
    assert "quote details" in output.data["message"].lower()


def test_negotiation_agent_composes_counter(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    captured: Dict[str, Any] = {}

    def fake_call(model=None, messages=None, **kwargs):
        captured.setdefault("calls", []).append({"model": model, "messages": messages})
        return {"message": {"content": "Email body"}}

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    context = AgentContext(
        workflow_id="wf2",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S1",
            "current_offer": 1300.0,
            "target_price": 1200.0,
            "rfq_id": "RFQ-123",
            "round": 1,
            "currency": "USD",
            "lead_time_weeks": 4,
            "supplier_snippets": ["We can ship in four weeks."],
        },
    )

    output = agent.run(context)

    assert output.data["decision"]["strategy"] == "midpoint"
    assert output.data["decision"]["counter_price"] == 1250.0
    assert output.data["message"] == "Email body"

    first_call = captured["calls"][0]
    assert first_call["model"] == "llama3.2"
    user_prompt = first_call["messages"][1]["content"]
    assert "RFQ-123" in user_prompt
    assert "decision" in user_prompt
    assert "We can ship in four weeks." in user_prompt


def test_negotiation_agent_uses_fallback_email(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    monkeypatch.setattr(
        agent,
        "call_ollama",
        lambda **_: {"message": {"content": ""}},
    )

    context = AgentContext(
        workflow_id="wf3",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S2",
            "current_offer": 2000.0,
            "target_price": 1500.0,
            "rfq_id": "RFQ-789",
            "currency": "GBP",
        },
    )

    output = agent.run(context)
    message = output.data["message"]

    assert "align on pricing" in message.lower()
    assert "rfq RFQ-789".lower() in message.lower()
