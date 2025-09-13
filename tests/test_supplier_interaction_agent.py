import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supplier_interaction_agent import SupplierInteractionAgent
from agents.base_agent import AgentContext, AgentStatus


class DummyCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, *args, **kwargs):
        self._row = ("gold",)

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
        return [SimpleNamespace(payload={"document_type": "email"})]


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


def test_supplier_interaction_agent():
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    context = AgentContext(
        workflow_id='wf1',
        agent_id='supplier_interaction',
        user_id='u1',
        input_data={
            'subject': 'Re: RFQ-20240101-abcd1234',
            'message': 'Our price is 1500 with lead time 10 days',
            'supplier_id': 's1',
            'target_price': '1000',
        }
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.next_agents == ['NegotiationAgent']
    assert output.data['price'] == 1500.0
    assert output.data['lead_time'] == '10'
