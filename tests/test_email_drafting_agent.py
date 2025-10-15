import json
import os
import sys
from types import SimpleNamespace
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents import email_drafting_agent as module
from agents.base_agent import AgentContext, AgentStatus
from agents.email_drafting_agent import DecisionContext, EmailDraftingAgent, ThreadHeaders
from utils.email_markers import extract_rfq_id, split_hidden_marker


@pytest.fixture(autouse=True)
def restore_env(monkeypatch):
    monkeypatch.delenv("EMAIL_POLISH_ENABLED", raising=False)
    yield


@pytest.fixture
def fixed_unique_id(monkeypatch):
    token = "UID-1234567890ABCD"
    monkeypatch.setattr(module, "generate_unique_email_id", lambda *_, **__: token)
    return token


def _visible_body(body: str) -> str:
    _comment, remainder = split_hidden_marker(body)
    return remainder or ""


def test_from_decision_formats_payload(monkeypatch, fixed_unique_id):
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

    assert result["subject"] == module.DEFAULT_NEGOTIATION_SUBJECT
    comment, remainder = split_hidden_marker(result["body"])
    assert comment and extract_rfq_id(comment) == result["unique_id"]
    assert "RFQ_ID" not in comment
    assert remainder.strip() == result["text"].strip()
    assert result["metadata"].get("dispatch_token")
    assert "Please confirm if 44.8 GBP is workable." in result["text"]
    assert "<ul>" in result["html"] and "<li>Please confirm if 44.8 GBP is workable.</li>" in result["html"]
    assert result["headers"]["In-Reply-To"] == "<msg1>"
    assert result["headers"]["References"] == "<msg0>"
    assert result["metadata"]["supplier_id"] == "S-1"
    assert result["receiver"] == "sales@acme.test"
    assert result["recipients"] == ["sales@acme.test", "buyer@test"]
    assert result["contact_level"] == 1
    assert result["sender"]
    assert result["sent_status"] is False
    assert calls[0]["payload"]["rfq_id"] == "RFQ-20250930-RFQ00001"
    assert result["headers"]["X-Procwise-Unique-Id"] == fixed_unique_id
    assert result["metadata"]["unique_id"] == fixed_unique_id
    assert result["unique_id"] == fixed_unique_id


def test_from_decision_subject_fallback(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20250930")
    monkeypatch.setattr(module, "_generate_rfq_id", lambda: "RFQ-20250930-RFQ00001")
    agent = EmailDraftingAgent()
    result = agent.from_decision({"rfq_id": "ABC"})
    assert result["subject"] == module.DEFAULT_NEGOTIATION_SUBJECT
    assert result["text"] == "Body without explicit subject"


def test_subject_fallback_does_not_duplicate_rfq_prefix(monkeypatch):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    agent = EmailDraftingAgent()
    rfq_id = "RFQ-20250930-84d44c85"
    result = agent.from_decision({"rfq_id": rfq_id})
    assert result["subject"] == module.DEFAULT_NEGOTIATION_SUBJECT


def test_from_decision_generates_unique_rfq_id(monkeypatch, fixed_unique_id):
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Body without explicit subject")
    monkeypatch.setattr(module, "_current_rfq_date", lambda: "20250930")
    monkeypatch.setattr(module, "_generate_rfq_id", lambda: "RFQ-20250930-UN1QUEID")

    agent = EmailDraftingAgent()
    decision = {"rfq_id": None, "supplier_id": "S-42"}

    result = agent.from_decision(decision)

    assert result["rfq_id"] == "RFQ-20250930-UN1QUEID"
    assert result["headers"]["X-Procwise-RFQ-ID"] == "RFQ-20250930-UN1QUEID"
    assert result["headers"]["X-Procwise-Unique-Id"] == fixed_unique_id
    assert result["metadata"]["unique_id"] == fixed_unique_id
    assert result["unique_id"] == fixed_unique_id
    assert result["subject"] == module.DEFAULT_NEGOTIATION_SUBJECT


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


def test_run_sets_recipient_when_instructions_present():
    agent = EmailDraftingAgent()
    context = _make_context(
        {
            "ranking": [
                {
                    "supplier_id": "SUP-1",
                    "supplier_name": "Acme",
                    "contact_email": "quotes@acme.test",
                }
            ],
            "supplier_profiles": {"SUP-1": {}},
            "policies": [{"policy_details": "Tone: formal"}],
        }
    )

    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    draft = result.data["drafts"][0]
    assert draft["recipients"] == ["quotes@acme.test"]
    assert draft["receiver"] == "quotes@acme.test"


def test_supplier_context_is_not_injected_into_email_body():
    agent = EmailDraftingAgent()
    context = _make_context(
        {
            "ranking": [
                {
                    "supplier_id": "SUP-CTX",
                    "supplier_name": "Context Corp",
                    "contact_name": "Alex",
                    "contact_email": "rfq@context.test",
                    "avg_unit_price": 42.78,
                    "currency": "GBP",
                    "po_count": 2,
                    "invoice_count": 2,
                    "relationship_summary": (
                        "Year-to-date spend with your organisation totals £28,487.89."
                    ),
                }
            ],
            "supplier_profiles": {"SUP-CTX": {}},
        }
    )

    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    draft = result.data["drafts"][0]
    visible = _visible_body(draft["body"])
    assert "Supplier context" not in visible
    assert "£28,487.89" not in visible

    metadata = draft.get("metadata") or {}
    internal = metadata.get("internal_context") or {}
    html_context = internal.get("supplier_context_html")
    text_context = internal.get("supplier_context_text")

    assert html_context and "Supplier context" in html_context
    assert "£28,487.89" in html_context
    assert text_context and "£28,487.89" in text_context


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
    monkeypatch.setattr(module, "_chat", lambda *_, **__: "Subject: Hello\nBody", raising=False)

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


def test_email_drafting_personalises_supplier_content(monkeypatch):
    monkeypatch.setattr(
        module,
        "_chat",
        lambda *_, **__: "Subject: Hello\nBody",
        raising=False,
    )

    def fake_prompt_engine(agent_nick, prompt_rows=None):
        return SimpleNamespace(get_prompt=lambda *_, **__: None)

    monkeypatch.setattr(module, "PromptEngine", fake_prompt_engine, raising=False)
    monkeypatch.setattr("agents.base_agent.PromptEngine", fake_prompt_engine)
    monkeypatch.setattr("agents.base_agent.configure_gpu", lambda *_, **__: "cpu")

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            pass

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return DummyCursor()

        def commit(self):
            pass

    class DummyRouting:
        def log_process(self, **kwargs):
            return None

        def log_run_detail(self, **kwargs):
            return None

        def log_action(self, **kwargs):
            return None

    agent_nick = SimpleNamespace(
        settings=SimpleNamespace(
            script_user="tester",
            ses_default_sender="buyer@example.com",
            ses_inbound_bucket=None,
            ses_inbound_prefix=None,
            ses_inbound_s3_uri=None,
            s3_bucket_name=None,
            email_response_poll_seconds=1,
            email_inbound_initial_wait_seconds=0,
        ),
        process_routing_service=DummyRouting(),
        agents={},
        prompt_engine=fake_prompt_engine(None),
    )

    agent_nick.get_db_connection = lambda: DummyConn()

    agent = EmailDraftingAgent(agent_nick=agent_nick)
    agent._store_draft = lambda draft: None

    ranking = [
        {
            "supplier_id": "SUP-1",
            "supplier_name": "Vendor A",
            "avg_unit_price": 1250.0,
            "total_spend": 250000.0,
            "po_count": 12,
            "invoice_count": 18,
            "lead_time_days": 14,
            "relationship_summary": "Long-term frame agreement",
        },
        {
            "supplier_id": "SUP-2",
            "supplier_name": "Vendor B",
            "avg_unit_price": 1310.0,
            "total_spend": 80000.0,
            "po_count": 3,
            "relationship_statements": ["New vendor onboarding in progress"],
            "flow_coverage": 0.6,
        },
    ]

    profiles = {
        "SUP-1": {"items": ["High-spec laptop"], "categories": {"IT": ["Hardware"]}},
        "SUP-2": {"items": ["Monitor"], "categories": {"IT": ["Displays"]}},
    }

    context = _make_context(
        {
            "ranking": ranking,
            "supplier_profiles": profiles,
            "currency": "GBP",
            "tone": "friendly",
        }
    )

    result = agent.run(context)
    assert result.status == AgentStatus.SUCCESS
    drafts = result.data["drafts"]
    assert len(drafts) == 2

    first = drafts[0]
    second = drafts[1]

    assert {draft.get("workflow_id") for draft in drafts} == {"wf-1"}
    assert all(
        draft.get("metadata", {}).get("workflow_id") == "wf-1" for draft in drafts
    )

    assert "£250,000.00" not in _visible_body(first["body"])
    assert "Long-term frame agreement" not in _visible_body(first["body"])
    assert "£80,000.00" not in _visible_body(second["body"])
    assert "New vendor onboarding" not in _visible_body(second["body"])

    first_internal = (first.get("metadata") or {}).get("internal_context") or {}
    second_internal = (second.get("metadata") or {}).get("internal_context") or {}

    assert "£250,000.00" in first_internal.get("supplier_context_html", "")
    assert "Long-term frame agreement" in first_internal.get("supplier_context_text", "")

    assert "£80,000.00" in second_internal.get("supplier_context_html", "")
    assert "New vendor onboarding" in second_internal.get("supplier_context_text", "")

    assert first_internal != second_internal


def test_manual_draft_inherits_workflow_id(monkeypatch):
    monkeypatch.setattr(
        module,
        "_chat",
        lambda *_, **__: "Subject: Hello\nBody",
        raising=False,
    )

    def fake_prompt_engine(agent_nick, prompt_rows=None):
        return SimpleNamespace(get_prompt=lambda *_, **__: None)

    monkeypatch.setattr(module, "PromptEngine", fake_prompt_engine, raising=False)
    monkeypatch.setattr("agents.base_agent.PromptEngine", fake_prompt_engine)
    monkeypatch.setattr("agents.base_agent.configure_gpu", lambda *_, **__: "cpu")

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            pass

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return DummyCursor()

        def commit(self):
            pass

    class DummyRouting:
        def log_process(self, **kwargs):
            return None

        def log_run_detail(self, **kwargs):
            return None

        def log_action(self, **kwargs):
            return None

    agent_nick = SimpleNamespace(
        settings=SimpleNamespace(
            script_user="tester",
            ses_default_sender="buyer@example.com",
            ses_inbound_bucket=None,
            ses_inbound_prefix=None,
            ses_inbound_s3_uri=None,
            s3_bucket_name=None,
            email_response_poll_seconds=1,
            email_inbound_initial_wait_seconds=0,
        ),
        process_routing_service=DummyRouting(),
        agents={},
        prompt_engine=fake_prompt_engine(None),
    )

    agent_nick.get_db_connection = lambda: DummyConn()

    agent = EmailDraftingAgent(agent_nick=agent_nick)
    agent._store_draft = lambda draft: None

    context = _make_context(
        {
            "recipients": ["contact@example.com"],
            "body": "Hello supplier",
            "subject": "Follow up",
        }
    )

    result = agent.run(context)
    drafts = result.data["drafts"]
    assert drafts
    manual_draft = drafts[-1]
    assert manual_draft.get("workflow_id") == "wf-1"
    assert manual_draft.get("metadata", {}).get("workflow_id") == "wf-1"
