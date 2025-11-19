import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.negotiation_agent import NegotiationAgent
from repositories import supplier_response_repo
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


class DummyNick:
    def __init__(self, *, enable_learning: bool = False):
        self.settings = SimpleNamespace(
            qdrant_collection_name="dummy",
            extraction_model="gpt-oss",
            script_user="tester",
            ses_default_sender="noreply@example.com",
            enable_learning=enable_learning,
            hitl_enabled=True,
        )
        self.action_logs: List[Dict[str, Any]] = []

        def _log_action(**kwargs):
            self.action_logs.append(dict(kwargs))
            return kwargs.get("action_id") or f"action-{len(self.action_logs)}"

        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: 1,
            log_run_detail=lambda **_: "run-1",
            log_action=_log_action,
            validate_workflow_id=lambda *args, **kwargs: True,
        )
        self.lmstudio_options = lambda: {}
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
    email_actions = [entry for entry in nick.action_logs if entry.get("agent_type") == "EmailDraftingAgent"]
    assert not email_actions


def test_negotiation_agent_composes_counter(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    thread_headers_stub = {
        "Message-ID": "<agent-1@procwise.test>",
        "In-Reply-To": "<supplier-previous@test>",
        "References": ["<supplier-previous@test>"]
    }
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
                    "thread_headers": thread_headers_stub,
                    "message_id": thread_headers_stub["Message-ID"],
                }
            ],
            "subject": "Re: RFQ-123 – Updated commercial terms",
            "body": "Email body",
            "thread_headers": thread_headers_stub,
            "message_id": thread_headers_stub["Message-ID"],
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
            "supplier_type": "Leverage",
            "negotiation_style": "Collaborative",
            "lever_priorities": ["Operational", "Commercial"],
            "policies": [
                {
                    "preferred_levers": ["Operational"],
                    "restricted_levers": ["Commercial"],
                }
            ],
            "supplier_performance": {"on_time_delivery": 0.82},
            "market_context": {"supply_risk": "high"},
        },
    )

    output = agent.run(context)

    assert output.data["decision"]["strategy"] == "counter"
    assert output.data["decision"]["counter_price"] == pytest.approx(1250.0, rel=1e-4)
    assert output.data["negotiation_allowed"] is True
    assert output.data["intent"] == "NEGOTIATION_COUNTER"
    assert "$" in output.data["message"] or "1250" in output.data["message"]
    draft_payload = output.data["draft_payload"]
    assert draft_payload["counter_price"] == pytest.approx(1250.0, rel=1e-4)
    assert draft_payload["negotiation_message"].startswith("Round 1 plan")
    assert draft_payload["recipients"] == ["quotes@supplier.test"]
    assert "Recommended plays" in draft_payload["negotiation_message"]
    plays = draft_payload["play_recommendations"]
    assert isinstance(plays, list) and plays
    assert plays[0]["lever"] == "Operational"
    assert "Trade-off" in draft_payload["negotiation_message"]
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
    assert output.data["play_recommendations"] == plays
    assert output.data["awaiting_response"] is False
    assert output.next_agents == []
    assert "await_response" not in output.pass_fields
    assert output.pass_fields["supplier_responses"][0]["message_id"] == "reply-1"
    assert output.pass_fields["play_recommendations"] == plays
    assert output.data["draft_payload"]["thread_headers"]["Message-ID"].startswith("<agent-1")
    thread_state = output.data.get("thread_state")
    assert isinstance(thread_state, dict) and thread_headers_stub["Message-ID"] in thread_state.get("references", [])
    email_actions = [entry for entry in nick.action_logs if entry.get("agent_type") == "EmailDraftingAgent"]
    assert len(email_actions) == 1
    payload = email_actions[0]["process_output"]
    assert payload["subject"].startswith("Re:")
    assert payload["drafts"][0]["supplier_id"] == "S1"
    assert payload["intent"] == "NEGOTIATION_COUNTER"


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


def test_negotiation_playbook_policy_ranking(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    stub_email_output = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={
            "drafts": [],
            "subject": "Re: RFQ-222 – Negotiation update",
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
        lambda **_: [{"message_id": "reply-risk", "supplier_id": "S3", "rfq_id": "RFQ-222"}],
    )

    context = AgentContext(
        workflow_id="wf4",
        agent_id="negotiation",
        user_id="tester",
        input_data={
            "supplier": "S3",
            "current_offer": 150.0,
            "target_price": 120.0,
            "rfq_id": "RFQ-222",
            "currency": "GBP",
            "supplier_type": "Leverage",
            "negotiation_style": "Competitive",
            "lever_priorities": ["Commercial", "Risk", "Operational"],
            "policies": [
                {
                    "preferred_levers": ["Risk"],
                    "restricted_levers": ["Commercial"],
                }
            ],
            "supplier_performance": {
                "on_time_delivery": 0.82,
                "quality_incidents": 3,
            },
            "market_context": {
                "supply_risk": "high",
                "inflation": "increasing",
            },
        },
    )

    output = agent.run(context)

    plays = output.data["play_recommendations"]
    assert plays, "Expected play recommendations to be returned"
    assert plays[0]["lever"] == "Risk"
    assert plays[0]["score"] >= plays[-1]["score"]
    policy_notes = plays[0]["policy_alignment"]
    assert "Policy:" in plays[0]["rationale"]
    assert any("policy" in note.lower() for note in policy_notes)
    # Ensure restricted lever appears but with lower score
    commercial_entries = [p for p in plays if p["lever"] == "Commercial"]
    if commercial_entries:
        for entry in commercial_entries:
            assert entry["score"] < plays[0]["score"]
    assert output.data["decision"]["play_recommendations"] == plays
    assert "Recommended plays" in output.data["message"]
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


def test_learning_snapshot_noop_when_learning_disabled():
    class Recorder:
        def __init__(self):
            self.calls = 0

        def record_negotiation_learning(self, **_):
            self.calls += 1

    nick = DummyNick(enable_learning=False)
    repo = Recorder()
    nick.learning_repository = repo

    agent = NegotiationAgent(nick)
    context = AgentContext(
        workflow_id="wf-disabled",
        agent_id="negotiation",
        user_id="tester",
        input_data={},
    )

    agent._record_learning_snapshot(
        context,
        rfq_id="RFQ-001",
        supplier="SUP-1",
        decision={"strategy": "counter"},
        state={"current_round": 1},
        awaiting_response=False,
        supplier_reply_registered=False,
    )

    assert repo.calls == 0


def test_learning_snapshot_noop_even_when_learning_enabled():
    class Recorder:
        def __init__(self):
            self.calls = 0

        def record_negotiation_learning(self, **_):
            self.calls += 1

    nick = DummyNick(enable_learning=True)
    repo = Recorder()
    nick.learning_repository = repo

    agent = NegotiationAgent(nick)
    context = AgentContext(
        workflow_id="wf-enabled",
        agent_id="negotiation",
        user_id="tester",
        input_data={},
    )

    agent._record_learning_snapshot(
        context,
        rfq_id="RFQ-002",
        supplier="SUP-2",
        decision={"strategy": "counter", "counter_price": 100.0},
        state={"current_round": 2, "supplier_reply_count": 1},
        awaiting_response=True,
        supplier_reply_registered=True,
    )

    assert repo.calls == 0


def test_learning_snapshot_ignores_training_endpoint():
    class EndpointRecorder:
        def __init__(self):
            self.calls = 0

        def queue_negotiation_learning(self, **_):
            self.calls += 1

    class RepoSpy:
        def __init__(self):
            self.calls = 0

        def record_negotiation_learning(self, **_):
            self.calls += 1

    nick = DummyNick(enable_learning=True)
    endpoint = EndpointRecorder()
    repo = RepoSpy()
    nick.model_training_endpoint = endpoint
    nick.learning_repository = repo

    agent = NegotiationAgent(nick)
    context = AgentContext(
        workflow_id="wf-queue",
        agent_id="negotiation",
        user_id="tester",
        input_data={},
    )

    agent._record_learning_snapshot(
        context,
        rfq_id="RFQ-100",
        supplier="SUP-9",
        decision={"strategy": "counter", "round": 2},
        state={"supplier_reply_count": 3},
        awaiting_response=False,
        supplier_reply_registered=True,
    )

    assert endpoint.calls == 0
    assert repo.calls == 0


def test_negotiation_agent_runs_batch_in_parallel(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    processed: List[str] = []

    def fake_single(self, context):
        supplier_id = context.input_data.get("supplier_id")
        processed.append(supplier_id)
        assert context.input_data.get("target_price") == 800.0
        assert "negotiation_batch" not in context.input_data
        data = {
            "supplier": supplier_id,
            "rfq_id": context.input_data.get("rfq_id"),
            "drafts": [
                {
                    "supplier_id": supplier_id,
                    "rfq_id": context.input_data.get("rfq_id"),
                    "intent": "NEGOTIATION_COUNTER",
                    "body": f"Counter for {supplier_id}",
                }
            ],
        }
        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=dict(data),
            ),
        )

    monkeypatch.setattr(NegotiationAgent, "_run_single_negotiation", fake_single)

    context = AgentContext(
        workflow_id="wf-batch",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={
            "negotiation_batch": [
                {"supplier_id": "S1", "rfq_id": "RFQ-1", "current_offer": 1000.0},
                {"supplier_id": "S2", "rfq_id": "RFQ-2", "current_offer": 900.0},
            ],
            "shared_context": {"target_price": 800.0},
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["negotiation_batch"] is True
    assert len(output.data["results"]) == 2
    assert sorted(processed) == ["S1", "S2"]
    assert len(output.data["drafts"]) == 2
    assert output.data["results_by_supplier"]["S1"]["output"]["rfq_id"] == "RFQ-1"
    assert output.data["successful_suppliers"] == ["S1", "S2"]
    assert output.data["failed_suppliers"] == []
    assert output.pass_fields["negotiation_batch"] is True
    assert len(output.pass_fields["batch_results"]) == 2
    summaries = output.data.get("round_summaries")
    assert summaries and summaries[0]["suppliers"] == ["S1", "S2"]
    assert summaries[0]["round"] is None
    assert output.pass_fields["round_summaries"][0]["suppliers"] == ["S1", "S2"]
    email_actions = [entry for entry in nick.action_logs if entry.get("agent_type") == "EmailDraftingAgent"]
    assert len(email_actions) == 2
    first_payload = email_actions[0]["process_output"]
    assert first_payload["intent"] == "NEGOTIATION_COUNTER"
    assert first_payload["drafts"][0]["supplier_id"] in {"S1", "S2"}


def test_negotiation_agent_batch_records_failures(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    def fake_single(self, context):
        supplier_id = context.input_data.get("supplier_id")
        if supplier_id == "S2":
            raise RuntimeError("missing pricing")
        data = {
            "supplier": supplier_id,
            "rfq_id": context.input_data.get("rfq_id"),
            "drafts": [
                {
                    "supplier_id": supplier_id,
                    "rfq_id": context.input_data.get("rfq_id"),
                    "intent": "NEGOTIATION_COUNTER",
                }
            ],
        }
        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=dict(data),
            ),
        )

    monkeypatch.setattr(NegotiationAgent, "_run_single_negotiation", fake_single)

    context = AgentContext(
        workflow_id="wf-batch-failure",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={
            "negotiation_batch": [
                {"supplier_id": "S1", "rfq_id": "RFQ-1"},
                {"supplier_id": "S2", "rfq_id": "RFQ-2"},
            ],
            "shared_context": {"target_price": 750.0},
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["negotiation_batch"] is True
    assert output.data["batch_size"] == 2
    assert len(output.data["results"]) == 2
    failures = [record for record in output.data["results"] if record["status"] == AgentStatus.FAILED.value]
    assert len(failures) == 1
    assert failures[0]["supplier_id"] == "S2"
    assert output.data["failed_suppliers"]
    assert output.data["successful_suppliers"] == ["S1"]
    assert len(output.data["drafts"]) == 1
    summaries = output.data.get("round_summaries")
    assert summaries and set(summaries[0]["suppliers"]) == {"S1", "S2"}
    email_actions = [entry for entry in nick.action_logs if entry.get("agent_type") == "EmailDraftingAgent"]
    assert len(email_actions) == 1
    assert email_actions[0]["process_output"]["drafts"][0]["supplier_id"] == "S1"


def test_negotiation_agent_batches_wait_for_prior_round(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    call_order: List[int] = []

    def fake_execute_batch_entry(self, context, payload):
        round_no = int(payload.get("round") or payload.get("round_number") or 1)
        call_order.append(round_no)
        supplier_id = payload.get("supplier_id")
        session_ref = payload.get("session_reference") or f"{supplier_id}-session"
        response_payload = {
            "message_id": f"{supplier_id}-r{round_no}",
            "round": round_no,
        }
        data = {
            "supplier": supplier_id,
            "rfq_id": payload.get("rfq_id"),
            "session_reference": session_ref,
            "unique_id": session_ref,
            "round": round_no,
            "decision": {"strategy": "counter"},
            "drafts": [
                {
                    "supplier_id": supplier_id,
                    "rfq_id": payload.get("rfq_id"),
                    "round": round_no,
                }
            ],
            "draft_payload": {
                "supplier_id": supplier_id,
                "round": round_no,
            },
            "supplier_responses": [response_payload],
        }
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=data,
            pass_fields=dict(data),
            next_agents=[],
        )

    monkeypatch.setattr(NegotiationAgent, "_execute_batch_entry", fake_execute_batch_entry)

    context = AgentContext(
        workflow_id="wf-seq",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={
            "negotiation_batch": [
                {
                    "supplier_id": "S1",
                    "rfq_id": "RFQ-1",
                    "round": 1,
                    "session_reference": "S1-session",
                },
                {
                    "supplier_id": "S2",
                    "rfq_id": "RFQ-2",
                    "round": 1,
                    "session_reference": "S2-session",
                },
                {
                    "supplier_id": "S1",
                    "rfq_id": "RFQ-1",
                    "round": 2,
                    "session_reference": "S1-session",
                },
                {
                    "supplier_id": "S2",
                    "rfq_id": "RFQ-2",
                    "round": 2,
                    "session_reference": "S2-session",
                },
            ],
            "shared_context": {"target_price": 800.0},
        },
    )

    output = agent.run(context)

    assert call_order[:2] == [1, 1]
    assert call_order[2:] == [2, 2]

    assert output.status == AgentStatus.SUCCESS
    summaries = output.data.get("round_summaries") or []
    assert len(summaries) == 2
    assert summaries[0]["round"] == 1
    assert set(summaries[0]["suppliers"]) == {"S1", "S2"}
    supplier_map = output.data.get("results_by_supplier") or {}
    supplier_record = supplier_map["S1"]
    responses = supplier_record["output"].get("supplier_responses")
    assert {resp["message_id"] for resp in responses} == {"S1-r1", "S1-r2"}
    draft_payload = supplier_record["output"].get("draft_payload")
    assert draft_payload["supplier_responses"] == responses


def test_negotiation_agent_adopts_workflow_from_drafts_when_mismatched(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    class StubSupplierAgent:
        def __init__(self):
            self.calls: List[Any] = []

        def wait_for_multiple_responses(self, entries, **kwargs):
            self.calls.append((entries, kwargs))
            return [
                {
                    "unique_id": entry.get("unique_id"),
                    "workflow_id": entry.get("workflow_id"),
                }
                for entry in entries
            ]

    stub_supplier_agent = StubSupplierAgent()
    monkeypatch.setattr(agent, "_get_supplier_agent", lambda: stub_supplier_agent)

    context = AgentContext(
        workflow_id="context-workflow",
        agent_id="negotiation",
        user_id="tester",
        input_data={},
    )

    watch_payload = {
        "await_response": True,
        "await_all_responses": True,
        "expected_dispatch_count": 3,
        "drafts": [
            {
                "unique_id": "DRAFT-1",
                "supplier_id": "S-1",
                "workflow_id": "draft-workflow",
            },
            {
                "unique_id": "DRAFT-2",
                "supplier_id": "S-2",
                "workflow_id": "draft-workflow",
            },
            {
                "unique_id": "DRAFT-3",
                "supplier_id": "S-3",
                "workflow_id": "draft-workflow",
            },
        ],
    }

    results = agent._await_supplier_responses(
        context=context,
        watch_payload=watch_payload,
        state={},
    )

    assert isinstance(results, list)
    assert len(results) == 3
    assert stub_supplier_agent.calls, "Supplier agent should have been invoked"
    entries, kwargs = stub_supplier_agent.calls[0]
    assert len(entries) == 3
    assert all(entry.get("workflow_id") == "draft-workflow" for entry in entries)
    assert kwargs["enable_negotiation"] is False


def test_multi_round_waits_for_hitl_before_second_round(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    def fake_resolve(self, context, payload):
        supplier = payload.get("supplier_id")
        round_no = int(payload.get("round") or 1)
        data = {
            "supplier": supplier,
            "decision": {
                "strategy": "counter",
                "counter_price": 900.0 - round_no,
                "currency": "USD",
            },
            "drafts": [
                {
                    "supplier_id": supplier,
                    "unique_id": f"{supplier}-r{round_no}",
                    "metadata": {"round": round_no},
                }
            ],
        }
        return AgentOutput(status=AgentStatus.SUCCESS, data=data)

    monkeypatch.setattr(
        NegotiationAgent,
        "_resolve_batch_entry_output",
        fake_resolve,
    )

    context = AgentContext(
        workflow_id="wf-hitl",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={},
    )

    batch_entries = [
        {"supplier_id": "S1", "current_offer": 1000.0},
        {"supplier_id": "S2", "current_offer": 1100.0},
    ]

    shared_context = {"workflow_id": "wf-hitl"}

    output = agent._run_multi_round_negotiation(
        context,
        batch_entries,
        shared_context,
        max_rounds=2,
    )

    assert output.status == AgentStatus.SUCCESS
    assert output.data["hitl_pending_rounds"] == [1]
    assert output.data["hitl_reviews"][0]["status"] == "pending"
    assert output.data["ready_for_quote_evaluation"] is False
    assert "QuoteEvaluationAgent" not in (output.next_agents or [])


def test_multi_round_routes_to_quote_evaluation_after_hitl(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    def fake_resolve(self, context, payload):
        supplier = payload.get("supplier_id")
        round_no = int(payload.get("round") or 1)
        counter_price = 900.0 - (round_no * 10)
        data = {
            "supplier": supplier,
            "decision": {
                "strategy": "counter",
                "counter_price": counter_price,
                "currency": "USD",
            },
            "drafts": [
                {
                    "supplier_id": supplier,
                    "unique_id": f"{supplier}-r{round_no}",
                    "metadata": {"round": round_no},
                }
            ],
        }
        return AgentOutput(status=AgentStatus.SUCCESS, data=data)

    def fake_wait(self, context, round_result, round_num, negotiation_state):
        responses: Dict[str, List[Dict[str, Any]]] = {}
        for record in round_result.get("results", []):
            supplier = record.get("supplier_id")
            decision = record.get("decision") or {}
            responses[supplier] = [
                {
                    "supplier_id": supplier,
                    "price": decision.get("counter_price", 880.0),
                }
            ]
        return responses, True

    monkeypatch.setattr(
        NegotiationAgent,
        "_resolve_batch_entry_output",
        fake_resolve,
    )
    monkeypatch.setattr(NegotiationAgent, "_wait_for_round_responses", fake_wait)

    context = AgentContext(
        workflow_id="wf-approved",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={},
    )

    batch_entries = [
        {"supplier_id": "S1", "current_offer": 1000.0},
        {"supplier_id": "S2", "current_offer": 1100.0},
    ]

    shared_context = {
        "workflow_id": "wf-approved",
        "hitl_decisions": {"1": "approved", "2": "approved"},
    }

    output = agent._run_multi_round_negotiation(
        context,
        batch_entries,
        shared_context,
        max_rounds=2,
    )

    assert output.status == AgentStatus.SUCCESS
    assert output.data["hitl_pending_rounds"] == []
    assert output.data["ready_for_quote_evaluation"] is True
    assert output.next_agents == ["QuoteEvaluationAgent"]
    quotes = output.data["final_quotes"]
    assert quotes
    assert {quote["supplier_id"] for quote in quotes} == {"S1", "S2"}


def test_wait_for_round_responses_uses_repository(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    class StubSupplierAgent:
        def wait_for_multiple_responses(self, *_, **__):
            return []

    monkeypatch.setattr(agent, "_get_supplier_agent", lambda: StubSupplierAgent())

    captured: Dict[str, Any] = {}

    def fake_fetch_for_unique_ids(
        *, workflow_id: str, unique_ids: Sequence[str], supplier_ids: Sequence[str], include_processed: bool
    ) -> List[Dict[str, Any]]:
        captured["workflow_id"] = workflow_id
        captured["unique_ids"] = tuple(unique_ids)
        captured["supplier_ids"] = tuple(supplier_ids)
        return [
            {"supplier_id": "S1", "unique_id": "S1-thread", "message_id": "m1"},
            {"supplier_id": "S2", "unique_id": "S2-thread", "message_id": "m2"},
        ]

    monkeypatch.setattr(
        supplier_response_repo,
        "fetch_for_unique_ids",
        fake_fetch_for_unique_ids,
    )

    context = AgentContext(
        workflow_id="wf-repo",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={},
    )

    round_result = {
        "workflow_id": "wf-repo",
        "drafts": [
            {"supplier_id": "S1", "unique_id": "S1-thread"},
            {"supplier_id": "S2", "unique_id": "S2-thread"},
        ],
    }

    responses, all_received = agent._wait_for_round_responses(
        context=context,
        round_result=round_result,
        round_num=1,
        negotiation_state={"workflow_id": "wf-repo"},
    )

    assert all_received is True
    assert set(responses) == {"S1", "S2"}
    assert captured["workflow_id"] == "wf-repo"
    assert set(captured["unique_ids"]) == {"S1-thread", "S2-thread"}


def test_multi_round_reuses_unique_id_and_max_round(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    def fake_resolve(self, context, payload):
        supplier = payload.get("supplier_id")
        round_no = int(payload.get("round") or 1)
        unique_id = f"{supplier}-thread"
        data = {
            "supplier": supplier,
            "decision": {"strategy": "counter", "counter_price": 1000 - round_no},
            "drafts": [
                {
                    "supplier_id": supplier,
                    "unique_id": unique_id,
                    "metadata": {"round": round_no},
                }
            ],
            "draft_payload": {"unique_id": unique_id, "subject": "Re: Updated terms"},
        }
        return AgentOutput(status=AgentStatus.SUCCESS, data=data)

    def fake_wait(self, context, round_result, round_num, negotiation_state):
        responses: Dict[str, List[Dict[str, Any]]] = {}
        for draft in round_result.get("drafts", []):
            supplier_id = draft.get("supplier_id")
            unique_id = draft.get("unique_id")
            responses.setdefault(supplier_id, []).append(
                {
                    "supplier_id": supplier_id,
                    "unique_id": unique_id,
                    "message_id": f"{unique_id}-reply-{round_num}",
                }
            )
        return responses, True

    monkeypatch.setattr(NegotiationAgent, "_resolve_batch_entry_output", fake_resolve)
    monkeypatch.setattr(NegotiationAgent, "_wait_for_round_responses", fake_wait)

    context = AgentContext(
        workflow_id="wf-unique",
        agent_id="NegotiationAgent",
        user_id="tester",
        input_data={},
    )

    batch_entries = [{"supplier_id": "S1"}, {"supplier_id": "S2"}]
    shared_context = {
        "workflow_id": "wf-unique",
        "hitl_decisions": {"1": "approved", "2": "approved"},
    }

    output = agent._run_multi_round_negotiation(
        context,
        batch_entries,
        shared_context,
        max_rounds=2,
    )

    assert output.data["total_rounds_executed"] == 2
    drafts = output.data.get("all_drafts") or []
    supplier_one_drafts = [draft for draft in drafts if draft.get("supplier_id") == "S1"]
    assert len(supplier_one_drafts) == 2
    assert {draft.get("unique_id") for draft in supplier_one_drafts} == {"S1-thread"}
    assert output.data.get("ready_for_quote_evaluation") is True
    assert "QuoteEvaluationAgent" in (output.next_agents or [])
