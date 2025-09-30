import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import email_drafting_agent as module
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
    assert calls[0]["payload"]["rfq_id"] == "RFQ-001"


def test_from_decision_subject_fallback(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    agent = EmailDraftingAgent()
    result = agent.from_decision({"rfq_id": "ABC"})
    assert result["subject"] == "RFQ ABC – Counter Offer & Next Steps"
    assert result["text"] == "Body without explicit subject"


def test_prompt_mode_with_polish(monkeypatch):
    responses = iter([
        "Subject: RFQ XYZ follow-up\nInitial draft",  # compose
        "Subject: RFQ XYZ follow-up\nPolished draft referencing RFQ XYZ",  # polish
    ])

    def fake_chat(*_, **__):
        return next(responses)

    monkeypatch.setattr(module, "_chat", fake_chat)

    agent = EmailDraftingAgent()
    agent.polish_model = "gemma3"

    result = agent.from_prompt("Please follow up on RFQ XYZ")

    assert result["subject"] == "RFQ XYZ follow-up"
    assert result["text"] == "Polished draft referencing RFQ XYZ"
    assert result["rfq_id"].lower().startswith("rfq xyz".replace(" ", ""))


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
