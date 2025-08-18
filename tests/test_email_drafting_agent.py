import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.email_drafting_agent import EmailDraftingAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(ses_default_sender='sender@example.com')


def test_email_drafting_agent(monkeypatch):
    nick = DummyNick()
    agent = EmailDraftingAgent(nick)

    # Mock LLM response
    monkeypatch.setattr(agent, 'call_ollama', lambda prompt=None, **kwargs: {'response': 'draft'})

    sent = {}

    def fake_send(subject, body, recipient, sender):
        sent.update({'subject': subject, 'body': body, 'recipient': recipient, 'sender': sender})
        return True

    monkeypatch.setattr(agent.email_service, 'send_email', fake_send)

    context = AgentContext(
        workflow_id='wf1',
        agent_id='email_drafting',
        user_id='u1',
        input_data={'prompt': 'hi', 'subject': 'Sub', 'recipient': 'to@example.com'},
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert sent['body'] == 'draft'
    assert sent['subject'] == 'Sub'
