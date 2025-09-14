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

    ranking = [{"supplier_id": "S1", "supplier_name": "Acme"}]
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
    assert draft["rfq_id"].startswith("RFQ-")
    assert f"<!-- RFQ-ID: {draft['rfq_id']} -->" in draft["body"]
    assert "Dear Acme," in draft["body"]
    assert "Deadline for submission: 01/01/2025" in draft["body"]
    assert captured["drafts"][0] == draft


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
    assert "Hello Bob" in output.data["drafts"][0]["body"]


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

