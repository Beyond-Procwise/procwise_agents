import json
import os
import sys
from types import SimpleNamespace
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents import email_drafting_agent as module
from agents.base_agent import AgentContext, AgentStatus
from agents.email_drafting_agent import DecisionContext, EmailDraftingAgent, ThreadHeaders


@pytest.fixture(autouse=True)
def restore_env(monkeypatch):
    monkeypatch.delenv("EMAIL_POLISH_ENABLED", raising=False)
    yield


def test_from_decision_formats_payload(monkeypatch):
    calls = []

    def fake_chat(model, system, user, **kwargs):
        calls.append({"model": model, "system": system, "payload": json.loads(user.split("Context (JSON):\n", 1)[1].split("\n\nWrite", 1)[0])})
        return (
            "Subject: Re: RFQ RFQ-001\n"
            "Hello Acme team,\n"
            "RFQ RFQ-001 relates to your current offer of 47.5 GBP.\n"
            "- Please confirm if 44.8 GBP is workable.\n"
            "- Advise if you can deliver within 3 weeks.\n"
            "Regards,\n"
        )

    monkeypatch.setattr(module, "_chat", fake_chat)
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20250930")
    monkeypatch.setattr(module, "_generate_rfq_id", lambda: "RFQ-20250930-RFQ00001")

    agent = EmailDraftingAgent()
    decision = {
        "rfq_id": "RFQ-001",
        "supplier_id": "S-1",
        "supplier_name": "Acme",
        "to": "sales@acme.test",
        "cc": ["buyer@test"],
        "current_offer": 47.50,
        "currency": "GBP",
        "lead_time_weeks": 4,
        "counter_price": 44.80,
        "asks": ["Confirm revised pricing", "Update on lead time"],
        "thread": {"message_id": "<msg1>", "references": ["<msg0>"]},
    }

    result = agent.from_decision(decision)

    assert result["subject"] == "Re: RFQ RFQ-001"
    assert "Please confirm if 44.8 GBP is workable." in result["text"]
    assert "<ul>" in result["html"] and "<li>Please confirm if 44.8 GBP is workable.</li>" in result["html"]
    assert result["headers"]["In-Reply-To"] == "<msg1>"
    assert result["headers"]["References"] == "<msg0>"
    assert result["metadata"]["supplier_id"] == "S-1"
    assert calls[0]["payload"]["rfq_id"] == "RFQ-20250930-RFQ00001"


def test_from_decision_subject_fallback(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20250930")
    monkeypatch.setattr(module, "_generate_rfq_id", lambda: "RFQ-20250930-RFQ00001")
    agent = EmailDraftingAgent()
    result = agent.from_decision({"rfq_id": "ABC"})
    assert result["subject"] == "RFQ-20250930-RFQ00001 – Counter Offer & Next Steps"
    assert result["text"] == "Body without explicit subject"


def test_subject_fallback_does_not_duplicate_rfq_prefix(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    agent = EmailDraftingAgent()
    rfq_id = "RFQ-20250930-84d44c85"
    result = agent.from_decision({"rfq_id": rfq_id})
    assert result["subject"] == "RFQ-20250930-84D44C85 – Counter Offer & Next Steps"


def test_from_decision_generates_unique_rfq_id(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20250930")
    monkeypatch.setattr(module, "_generate_rfq_id", lambda: "RFQ-20250930-UN1QUEID")

    agent = EmailDraftingAgent()
    decision = {"rfq_id": None, "supplier_id": "S-42"}

    result = agent.from_decision(decision)

    assert result["rfq_id"] == "RFQ-20250930-UN1QUEID"
    assert result["headers"]["X-Procwise-RFQ-ID"] == "RFQ-20250930-UN1QUEID"
    assert result["subject"] == "RFQ-20250930-UN1QUEID – Counter Offer & Next Steps"


def test_from_decision_normalises_lowercase_rfq_id(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    agent = EmailDraftingAgent()

    result = agent.from_decision({"rfq_id": "rfq-20250930-abc12345"})

    assert result["rfq_id"] == "RFQ-20250930-ABC12345"


def test_prompt_mode_with_polish(monkeypatch):
    responses = iter([
        "Subject: RFQ XYZ follow-up\nInitial draft",  # compose
        "Subject: RFQ XYZ follow-up\nPolished draft referencing RFQ XYZ",  # polish
    ])

    def fake_chat(*_, **__):
        return next(responses)

    monkeypatch.setattr(module, "_chat", fake_chat)
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20250930")
    monkeypatch.setattr(module, "_generate_rfq_id", lambda: "RFQ-20250930-96F5YFY9")

    agent = EmailDraftingAgent()
    agent.polish_model = "gemma3"

    result = agent.from_prompt("Please follow up on RFQ XYZ")

    assert result["subject"] == "RFQ XYZ follow-up"
    assert result["text"] == "Polished draft referencing RFQ XYZ"
    assert result["rfq_id"] == "RFQ-20250930-96F5YFY9"


def test_decision_context_to_public_json():
    ctx = DecisionContext(
        rfq_id="R1",
        supplier_id="S",
        supplier_name="Name",
        current_offer=12.3,
        currency="GBP",
        lead_time_weeks=2.0,
        target_price=11.5,
        round=2,
        strategy="midpoint",
        counter_price=11.8,
        asks=["item"],
        lead_time_request="3 weeks",
        rationale="alignment",
        thread=ThreadHeaders(message_id="m"),
    )
    data = ctx.to_public_json()
    assert data["rfq_id"] == "R1"
    assert data["asks"] == ["item"]


def _make_context(payload: dict) -> AgentContext:
    return AgentContext(
        workflow_id="wf-1",
        agent_id="email_drafting",
        user_id="tester",
        input_data=payload,
    )


def test_execute_uses_decision_payload(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Subject: Update\nBody line")

    agent = EmailDraftingAgent()
    context = _make_context(
        {
            "decision": {
                "rfq_id": "RFQ-900",
                "supplier_id": "S-900",
                "to": "buyer@example.com",
                "counter_price": 42.5,
            }
        }
    )

    result = agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["drafts"]
    draft = result.data["drafts"][0]
    assert draft["subject"] == "Update"
    assert draft["metadata"]["counter_price"] == 42.5


def test_execute_falls_back_to_prompt(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Subject: Follow-up\nBody")

    agent = EmailDraftingAgent()
    context = _make_context({"prompt": "Please follow up", "rfq_id": "RFQ-XYZ"})

    result = agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["drafts"][0]["subject"] == "Follow-up"


def test_execute_prompt_payload_skips_decision_fallback(monkeypatch):
    def fake_chat(model, system, user, **kwargs):
        assert system != module.SYSTEM_COMPOSE, "prompt payload should not use decision pathway"
        return "Subject: Follow-up\nBody"

    monkeypatch.setattr(module, "_chat", fake_chat)

    agent = EmailDraftingAgent()
    context = _make_context({"prompt": "Prompt only flow", "rfq_id": "RFQ-XYZ"})

    result = agent.execute(context)

    assert result.status == AgentStatus.SUCCESS
    assert result.data["drafts"][0]["subject"] == "Follow-up"


def test_execute_infers_recipients_and_counter_price(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Subject: Hello\nBody")

    agent = EmailDraftingAgent()
    payload = {
        "rfq_id": "RFQ-321",
        "recipients": ["primary@example.com", "cc1@example.com"],
        "counter_proposals": [{"price": 11.75}],
        "decision_log": "Request revision",
    }
    result = agent.execute(_make_context(payload))

    draft = result.data["drafts"][0]
    assert draft["to"] == "primary@example.com"
    assert draft["cc"] == ["cc1@example.com"]
    assert draft["metadata"]["counter_price"] == 11.75


def test_execute_logs_process_action_when_service_available(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Subject: Hello\nBody")

    class Recorder:
        def __init__(self):
            self.logged = {}

        def log_process(self, **kwargs):
            self.logged["log_process"] = kwargs
            return 101

        def log_run_detail(self, **kwargs):
            self.logged["log_run_detail"] = kwargs
            return "run-101"

        def log_action(self, **kwargs):
            self.logged["log_action"] = kwargs
            return "action-101"

    recorder = Recorder()
    agent_nick = SimpleNamespace(
        process_routing_service=recorder,
        settings=SimpleNamespace(script_user="script-user"),
    )
    agent = EmailDraftingAgent(agent_nick=agent_nick)
    context = _make_context({"prompt": "Follow up", "rfq_id": "RFQ-XYZ"})

    result = agent.execute(context)

    assert result.action_id == "action-101"
    assert result.data["action_id"] == "action-101"
    assert result.data["drafts"][0]["action_id"] == "action-101"
    assert recorder.logged["log_process"]["user_id"] == "tester"
    assert recorder.logged["log_action"]["status"] == "completed"
    assert recorder.logged["log_action"]["process_output"]["drafts"]
