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
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            qdrant_collection_name="dummy",
            extraction_model="gpt-oss",
            script_user="tester",
            ses_default_sender="noreply@example.com",
        )
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: 1,
            log_run_detail=lambda **_: "run-1",
            log_action=lambda **_: "action-1",
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
    assert output.data["negotiation_allowed"] is True
    assert output.data["intent"] == "NEGOTIATION_COUNTER"
    assert output.data["message"].lower().startswith("round 1 plan")


def test_negotiation_agent_composes_counter(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    stub_email_output = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={
            "drafts": [
                {
                    "supplier_id": "S1",
                    "rfq_id": "RFQ-123",
                    "subject": "Re: RFQ-123 – Updated commercial terms",
                    "body": "Email body",
                    "sent_status": False,
                    "metadata": {
                        "counter_price": 1250.0,
                        "target_price": 1200.0,
                        "current_offer": 1300.0,
                        "round": 1,
                    },
                }
            ],
            "subject": "Re: RFQ-123 – Updated commercial terms",
            "body": "Email body",
        },
    )
    stub_email_output.action_id = "email-action-1"

    monkeypatch.setattr(
        agent,
        "_invoke_email_drafting_agent",
        lambda ctx, payload: stub_email_output,
    )
    monkeypatch.setattr(
        agent,
        "_await_supplier_responses",
        lambda **_: [{"message_id": "reply-1", "supplier_id": "S1", "rfq_id": "RFQ-123"}],
    )

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
            "supplier_email": ["quotes@supplier.test"],
        },
    )

    output = agent.run(context)

    assert output.data["decision"]["strategy"] == "midpoint"
    assert output.data["decision"]["counter_price"] == 1250.0
    assert output.data["negotiation_allowed"] is True
    assert output.data["intent"] == "NEGOTIATION_COUNTER"
    assert "$" in output.data["message"] or "1250" in output.data["message"]
    draft_payload = output.data["draft_payload"]
    assert draft_payload["counter_price"] == 1250.0
    assert draft_payload["negotiation_message"].startswith("Round 1 plan")
    assert draft_payload["recipients"] == ["quotes@supplier.test"]
    assert output.data["session_state"]["current_round"] == 2
    assert output.data["session_state"]["supplier_reply_count"] == 1
    drafts = output.data["drafts"]
    assert isinstance(drafts, list) and len(drafts) == 1
    stub = drafts[0]
    assert stub["rfq_id"] == "RFQ-123"
    assert stub["sent_status"] is False
    assert stub.get("email_action_id") == "email-action-1"
    assert output.data.get("email_subject") == "Re: RFQ-123 – Updated commercial terms"
    assert output.data.get("email_body") == "Email body"
    assert output.data["awaiting_response"] is False
    assert output.next_agents == []
    assert "await_response" not in output.pass_fields
    assert output.pass_fields["supplier_responses"][0]["message_id"] == "reply-1"


def test_negotiation_agent_detects_final_offer(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    context = AgentContext(
        workflow_id="wf3",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S2",
            "current_offer": 1750.0,
            "target_price": 1500.0,
            "rfq_id": "RFQ-789",
            "supplier_message": "This is our final offer with no further reductions.",
            "message_id": "msg-1",
        },
    )

    output = agent.run(context)

    assert output.data["negotiation_allowed"] is False
    assert output.next_agents == []
    assert "final offer" in output.data["message"].lower()
    assert output.data["session_state"]["status"].upper() == "COMPLETED"
    assert output.data["awaiting_response"] is False
    assert output.data["drafts"] == []


def test_negotiation_agent_caps_supplier_replies(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    state = {
        "supplier_reply_count": 3,
        "current_round": 4,
        "status": "ACTIVE",
        "awaiting_response": False,
    }
    saved: Dict[str, Any] = {}

    monkeypatch.setattr(
        agent,
        "_load_session_state",
        lambda rfq, supplier: (dict(state), True),
    )

    def fake_save(rfq, supplier, new_state):
        saved.update(new_state)

    monkeypatch.setattr(agent, "_save_session_state", fake_save)

    context = AgentContext(
        workflow_id="wf4",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S3",
            "current_offer": 2200.0,
            "target_price": 1800.0,
            "rfq_id": "RFQ-456",
            "message_id": "msg-final",
        },
    )

    output = agent.run(context)

    assert output.data["negotiation_allowed"] is False
    assert output.data["session_state"]["status"].upper() == "EXHAUSTED"
    assert saved.get("status") == "EXHAUSTED"


def test_negotiation_agent_counts_replies_only_after_counter(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    state = {
        "supplier_reply_count": 1,
        "current_round": 3,
        "status": "ACTIVE",
        "awaiting_response": True,
        "last_supplier_msg_id": "msg-prev",
    }

    saved: Dict[str, Any] = {}

    monkeypatch.setattr(
        agent,
        "_load_session_state",
        lambda rfq, supplier: (dict(state), True),
    )

    def fake_save(rfq, supplier, new_state):
        saved.update(new_state)

    monkeypatch.setattr(agent, "_save_session_state", fake_save)

    stub_email_output = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={
            "drafts": [
                {
                    "supplier_id": "S4",
                    "rfq_id": "RFQ-567",
                    "subject": "Re: RFQ-567 – Updated commercial terms",
                    "body": "Email body",
                    "sent_status": False,
                }
            ],
            "subject": "Re: RFQ-567 – Updated commercial terms",
            "body": "Email body",
        },
    )
    stub_email_output.action_id = "email-action-2"

    monkeypatch.setattr(
        agent,
        "_invoke_email_drafting_agent",
        lambda ctx, payload: stub_email_output,
    )
    monkeypatch.setattr(
        agent,
        "_await_supplier_responses",
        lambda **_: [
            {"message_id": "msg-new", "supplier_id": "S4", "rfq_id": "RFQ-567"}
        ],
    )

    context = AgentContext(
        workflow_id="wf5",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S4",
            "current_offer": 2100.0,
            "target_price": 1800.0,
            "rfq_id": "RFQ-567",
            "message_id": "msg-new",
            "round": 3,
        },
    )

    output = agent.run(context)

    assert output.data["session_state"]["supplier_reply_count"] == 2
    assert saved.get("supplier_reply_count") == 2
    assert saved.get("awaiting_response") is False
    assert output.data.get("email_action_id") == "email-action-2"
    assert output.next_agents == []
    assert "await_response" not in output.pass_fields


def test_negotiation_agent_waits_for_supplier_timeout(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    stub_email_output = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={
            "drafts": [
                {
                    "supplier_id": "S8",
                    "rfq_id": "RFQ-321",
                    "subject": "Re: RFQ-321",
                    "body": "Email body",
                }
            ],
            "subject": "Re: RFQ-321",
            "body": "Email body",
        },
    )

    monkeypatch.setattr(
        agent,
        "_invoke_email_drafting_agent",
        lambda ctx, payload: stub_email_output,
    )
    monkeypatch.setattr(agent, "_await_supplier_responses", lambda **_: None)

    context = AgentContext(
        workflow_id="wf-timeout",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S8",
            "current_offer": 1500.0,
            "target_price": 1300.0,
            "rfq_id": "RFQ-321",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.FAILED
    assert output.error == "supplier response timeout"
    assert output.data["rfq_id"] == "RFQ-321"


def test_negotiation_agent_stays_active_while_waiting(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    state = {
        "supplier_reply_count": 1,
        "current_round": 2,
        "status": "ACTIVE",
        "awaiting_response": True,
        "last_supplier_msg_id": "msg-1",
    }

    monkeypatch.setattr(
        agent,
        "_load_session_state",
        lambda rfq, supplier: (dict(state), True),
    )

    saved: Dict[str, Any] = {}

    def fake_save(rfq, supplier, new_state):
        saved.update(new_state)

    monkeypatch.setattr(agent, "_save_session_state", fake_save)

    context = AgentContext(
        workflow_id="wf6",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S5",
            "current_offer": 2100.0,
            "target_price": 1800.0,
            "rfq_id": "RFQ-890",
            "message_id": "msg-1",
            "round": 2,
        },
    )

    output = agent.run(context)

    assert output.data["negotiation_allowed"] is False
    assert output.data["session_state"]["status"] == "AWAITING_SUPPLIER"
    assert output.data["session_state"]["awaiting_response"] is True
    assert output.data["drafts"] == []
    assert saved.get("awaiting_response") is True
    assert saved.get("status") == "AWAITING_SUPPLIER"
    assert output.next_agents == ["SupplierInteractionAgent"]
    assert output.pass_fields["drafts"][0]["rfq_id"] == "RFQ-890"


def test_negotiation_agent_falls_back_to_email_agent_queue(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    state = {"supplier_reply_count": 0, "current_round": 1, "status": "ACTIVE"}

    monkeypatch.setattr(
        agent,
        "_load_session_state",
        lambda rfq, supplier: (dict(state), False),
    )

    recorded_state: Dict[str, Any] = {}

    def fake_save(rfq, supplier, new_state):
        recorded_state.update(new_state)

    monkeypatch.setattr(agent, "_save_session_state", fake_save)

    monkeypatch.setattr(
        agent,
        "_invoke_email_drafting_agent",
        lambda ctx, payload: None,
    )

    context = AgentContext(
        workflow_id="wf7",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S6",
            "current_offer": 1000.0,
            "target_price": 900.0,
            "rfq_id": "RFQ-234",
            "round": 1,

        },
    )

    output = agent.run(context)

    assert output.next_agents == ["EmailDraftingAgent", "SupplierInteractionAgent"]
    assert output.data.get("email_action_id") is None
    assert recorded_state.get("awaiting_response") is True
