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

    sent = {}

    def fake_send(subject, body, recipient, sender, attachments=None):
        sent.update({
            'subject': subject,
            'body': body,
            'recipient': recipient,
            'sender': sender,
            'attachments': attachments,
        })
        return True

    monkeypatch.setattr(agent.email_service, 'send_email', fake_send)

    context = AgentContext(
        workflow_id='wf1',
        agent_id='email_drafting',
        user_id='u1',
        input_data={
            'recipient': 'to@example.com',
            'supplier_contact_name': 'John',
            'submission_deadline': '01/01/2025',
            'category_manager_name': 'Cat',
            'category_manager_title': 'Mgr',
            'category_manager_email': 'cat@example.com',
            'your_name': 'Buyer',
            'your_title': 'Procurement',
            'your_company': 'Your Company',
            'attachments': [(b'data', 'file.txt')],
        },
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert '<html>' in sent['body']
    assert sent['subject'] == 'Request for Quotation (RFQ) – Office Furniture'
    assert sent['attachments'] == [(b'data', 'file.txt')]
    assert 'Request for Quotation (RFQ) – Office Furniture' in output.data['prompt']
    assert 'Deadline for submission: 01/01/2025' in output.data['prompt']
