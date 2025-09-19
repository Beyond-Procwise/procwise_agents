import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.email_drafting_agent import EmailDraftingAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            ses_default_sender="sender@example.com",
            qdrant_collection_name="dummy",
            extraction_model="llama3",
            script_user="tester",
        )
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: None,
            log_action=lambda **_: None,
        )


def test_email_drafting_agent(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)

    captured = {}
    monkeypatch.setattr(agent, "_store_draft", lambda draft: captured.setdefault("drafts", []).append(draft))

    ranking = [
        {
            "supplier_id": "S1",
            "supplier_name": "Acme",
            "justification": "Ranked 1 following spend analysis.",
        }
    ]
    context = AgentContext(
        workflow_id="wf1",
        agent_id="email_drafting",
        user_id="u1",
        input_data={
            "ranking": ranking,
            "submission_deadline": "01/01/2025",
            "category_manager_name": "Cat",
            "category_manager_title": "Mgr",
            "category_manager_email": "cat@example.com",
            "your_name": "Buyer",
            "your_title": "Procurement",
            "your_company": "Your Company",
        },
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["drafts"]
    draft = output.data["drafts"][0]
    assert draft["supplier_id"] == "S1"
    assert draft["supplier_name"] == "Acme"
    assert draft["rfq_id"].startswith("RFQ-")
    assert f"<!-- RFQ-ID: {draft['rfq_id']} -->" in draft["body"]
    assert "<p>Dear Acme,</p>" in draft["body"]
    assert "formal quotation for the requirement outlined below" in draft["body"]
    assert "<p>Kindly complete the table and return your quotation by 01/01/2025.</p>" in draft["body"]
    assert "<p><strong>Note:</strong> Please do not change the subject line when replying.</p>" in draft["body"]
    assert "<p>We appreciate your timely response.</p>" in draft["body"]
    assert "<table" in draft["body"]
    assert draft["subject"].count("RFQ") == 1
    body_lower = draft["body"].lower()
    assert "rank" not in body_lower
    assert "analysis" not in body_lower
    assert draft["sent_status"] is False
    assert draft["sender"] == "sender@example.com"
    assert draft["contact_level"] == 0
    assert draft["recipients"] == []
    assert draft.get("receiver") is None
    assert draft["thread_index"] == 1
    assert "action_id" in draft
    assert draft["action_id"] is None
    assert captured["drafts"][0] == draft


def test_email_drafting_applies_instruction_settings(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)

    stored = []
    monkeypatch.setattr(agent, "_store_draft", lambda draft: stored.append(draft))

    ranking = [
        {
            "supplier_id": "S1",
            "supplier_name": "Alpha Co",
            "contact_email": "alpha@example.com",
        }
    ]

    prompts = [
        {
            "promptId": 1,
            "prompts_desc": (
                "body_template: \"<p>Dear {supplier_contact_name_html}, we kindly request a quotation.</p>\"\n"
                "include_rfq_table: false\n"
                "additional_paragraph: \"Please confirm receipt of this request.\""
            ),
        }
    ]

    policies = [
        {
            "policyId": 2,
            "policy_desc": (
                "subject_template: \"RFQ {rfq_id} for {supplier_company}\"\n"
                "compliance_notice: \"All quotations are confidential.\""
            ),
        }
    ]

    context = AgentContext(
        workflow_id="wf-instruction",
        agent_id="email_drafting",
        user_id="tester",
        input_data={
            "ranking": ranking,
            "prompts": prompts,
            "policies": policies,
            "submission_deadline": "2025-01-15",
        },
    )

    output = agent.run(context)
    draft = output.data["drafts"][0]

    assert draft["subject"].startswith("RFQ")
    assert "Alpha Co" in draft["subject"]
    body = draft["body"]
    assert "we kindly request a quotation" in body
    assert "Please confirm receipt of this request." in body
    assert "All quotations are confidential" in body
    assert "Compliance" in body
    assert "<table" not in body
    assert draft["recipients"] == []


def test_email_drafting_uses_template_from_previous_agent(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)
    monkeypatch.setattr(agent, "_store_draft", lambda draft: None)

    ranking = [{"supplier_id": "S1", "supplier_name": "Bob"}]
    context = AgentContext(
        workflow_id="wf3",
        agent_id="email_drafting",
        user_id="u1",
        input_data={
            "ranking": ranking,
            "body": "Hello {{ supplier_contact_name }}",
        },
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert "<p>Hello Bob</p>" in output.data["drafts"][0]["body"]


def test_email_drafting_handles_missing_ranking(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)
    monkeypatch.setattr(agent, "_store_draft", lambda draft: None)

    context = AgentContext(
        workflow_id="wf2",
        agent_id="email_drafting",
        user_id="u1",
        input_data={"body": "hi"},
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["drafts"] == []


def test_email_drafting_includes_action_ids(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)
    monkeypatch.setattr(agent, "_store_draft", lambda draft: None)

    ranking = [
        {"supplier_id": "S1", "supplier_name": "Acme", "action_id": "rank-1"},
        {"supplier_id": "S2", "supplier_name": "Beta"},
    ]
    context = AgentContext(
        workflow_id="wf4",
        agent_id="email_drafting",
        user_id="u1",
        input_data={
            "ranking": ranking,
            "action_id": "email-action",
        },
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    drafts = output.data["drafts"]
    assert output.data["action_id"] == "email-action"
    assert drafts[0]["action_id"] == "rank-1"
    assert drafts[0]["supplier_name"] == "Acme"
    assert drafts[1]["action_id"] == "email-action"
    assert drafts[1]["supplier_name"] == "Beta"

    assert output.pass_fields["action_id"] == "email-action"


def test_email_drafting_creates_manual_draft_without_sending(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)

    stored = []
    monkeypatch.setattr(agent, "_store_draft", lambda draft: stored.append(draft))

    context = AgentContext(
        workflow_id="wf5",
        agent_id="email_drafting",
        user_id="u1",
        input_data={
            "subject": "Manual Subject",
            "recipients": ["user@example.com", "team@example.com", "user@example.com"],
            "body": "<!-- RFQ-ID: RFQ-MANUAL --><p>Hello supplier</p>",
            "action_id": "manual-action",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert "sent" not in output.data

    drafts = output.data["drafts"]
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft["rfq_id"] == "RFQ-MANUAL"
    assert draft["sent_status"] is True
    assert draft["action_id"] == "manual-action"
    assert draft["recipients"] == ["user@example.com", "team@example.com"]
    assert draft["receiver"] == "user@example.com"
    assert draft["contact_level"] == 1
    assert draft["thread_index"] == 1
    assert stored[0] == draft
    assert output.data["recipients"] == ["user@example.com", "team@example.com"]
    assert output.data["action_id"] == "manual-action"
    assert output.pass_fields["body"].startswith("<!-- RFQ-ID: RFQ-MANUAL -->")

