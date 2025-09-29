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


def test_supplier_interaction_wait_for_response():
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    calls = {"count": 0}

    def poll_once(limit=None, match_filters=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                {
                    "rfq_id": "RFQ-20240101-abcd1234",
                    "supplier_id": "S1",
                    "negotiation_output": {"message": "counter"},
                }
            ]
        return []

    watcher = SimpleNamespace(poll_once=poll_once)

    result = agent.wait_for_response(watcher=watcher, timeout=1, poll_interval=0)

    assert result is not None
    assert result["rfq_id"] == "RFQ-20240101-abcd1234"
    assert calls["count"] == 1


def test_supplier_interaction_wait_for_response_respects_attempt_limit(monkeypatch):
    nick = DummyNick()
    nick.settings.email_response_max_attempts = 3
    agent = SupplierInteractionAgent(nick)

    calls = {"count": 0}

    def poll_once(limit=None, match_filters=None):
        calls["count"] += 1
        return []

    watcher = SimpleNamespace(poll_once=poll_once)

    monkeypatch.setattr("agents.supplier_interaction_agent.time.sleep", lambda *_args, **_kwargs: None)

    result = agent.wait_for_response(
        watcher=watcher,
        timeout=30,
        poll_interval=None,
        rfq_id="RFQ-20240101-missing",
        max_attempts=3,
    )

    assert result is None
    assert calls["count"] == 3


def test_supplier_interaction_waits_using_drafts():
    nick = DummyNick()
    agent = SupplierInteractionAgent(nick)

    drafts = [
        {
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "S1",
            "receiver": "supplier@example.com",
        }
    ]

    def fake_wait_for_response(**kwargs):
        assert kwargs["rfq_id"] == "RFQ-20240101-abcd1234"
        assert kwargs["supplier_id"] == "S1"
        return {
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "S1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "supplier_status": "success",
            "target_price": 1000.0,
            "supplier_output": {
                "price": 900.0,
                "lead_time": "5",
                "response_text": "Quoted price 900 in 5 days",
                "related_documents": [{"doc": 1}],
            },
        }

    agent.wait_for_response = fake_wait_for_response  # type: ignore[assignment]

    context = AgentContext(
        workflow_id="wf2",
        agent_id="supplier_interaction",
        user_id="u1",
        input_data={
            "subject": "RFQ-20240101-abcd1234 – Request for Quotation",
            "drafts": drafts,
            "supplier_id": "S1",
            "target_price": "1000",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["rfq_id"] == "RFQ-20240101-abcd1234"
    assert output.data["price"] == 900.0
    assert output.data["target_price"] == 1000.0
    assert output.next_agents == ["QuoteEvaluationAgent"]
