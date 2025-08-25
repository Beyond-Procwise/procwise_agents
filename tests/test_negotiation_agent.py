import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.negotiation_agent import NegotiationAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, *args, **kwargs):
        self._row = ("counter",)

    def fetchone(self):
        return self._row


class DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def cursor(self):
        return DummyCursor()


class DummyQdrant:
    def search(self, **kwargs):
        return [SimpleNamespace(payload={"document_type": "policy"})]


class DummyEmbedding:
    def encode(self, _):
        return [0.0]


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(qdrant_collection_name="test")
        self.qdrant_client = DummyQdrant()
        self.embedding_model = DummyEmbedding()

    def get_db_connection(self):
        return DummyConn()


def test_negotiation_agent(monkeypatch):
    nick = DummyNick()
    agent = NegotiationAgent(nick)

    monkeypatch.setattr(
        agent,
        'call_ollama',
        lambda prompt=None, **kwargs: {'response': 'counter offer'}
    )

    context = AgentContext(
        workflow_id='wf1',
        agent_id='negotiation',
        user_id='u1',
        input_data={'supplier': 'Acme', 'current_offer': 1000, 'target_price': 900}
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert 'counter offer' in output.data['message']
    assert output.data['strategy'] == 'counter'
    assert output.next_agents == ['EmailDraftingAgent']
