import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.approvals_agent import ApprovalsAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, *args, **kwargs):
        self._row = (1000,)

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
        return [SimpleNamespace(payload={"document_type": "invoice"})]


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


def test_approvals_agent():
    nick = DummyNick()
    agent = ApprovalsAgent(nick)

    context = AgentContext(
        workflow_id='wf1',
        agent_id='approvals',
        user_id='u1',
        input_data={'amount': 500, 'supplier_id': 's1'}
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data['decision'] == 'approve'
