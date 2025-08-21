import os
import sys
import json
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.quote_evaluation_agent import QuoteEvaluationAgent
from agents.base_agent import AgentContext, AgentStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers.workflows import router


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            extraction_model="llama3",
            script_user="tester",
            qdrant_collection_name="dummy",
        )
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: None,
            log_action=lambda **_: None,
        )
        self.ollama_options = lambda: {}
        self.qdrant_client = SimpleNamespace()


def _mock_quotes(*args, **kwargs):
    return [
        {
            "quote_id": "Q1",
            "supplier_name": "Supplier A",
            "total_amount": 100,
            "payment_terms": "Net 30",
            "delivery_terms": "3 days",
            "discount_percentage": 5,
            "baseline_total_amount": 120,
            "baseline_payment_terms": "Net 30",
        },
        {
            "quote_id": "Q2",
            "supplier_name": "Supplier B",
            "total_amount": 120,
            "payment_terms": "Net 45",
            "delivery_terms": "5 days",
            "discount_percentage": 0,
            "baseline_total_amount": 110,
            "baseline_payment_terms": "Net 30",
        },
    ]


def _mock_ollama(*args, **kwargs):
    return {"response": json.dumps({"negotiation_points": ["discount"]})}


def _mock_comparison(quotes):
    return {
        "price_range": {"min": 100, "max": 120, "mean": 110, "std": 10},
        "suppliers": ["Supplier A", "Supplier B"],
        "best_payment_terms": {},
        "discount_range": {"min": 0, "max": 5},
    }


def test_quote_evaluation_agent_run(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)
    monkeypatch.setattr(agent, "_fetch_quotes", _mock_quotes)
    monkeypatch.setattr(agent, "call_ollama", _mock_ollama)
    monkeypatch.setattr(agent, "_generate_comparison", _mock_comparison)
    context = AgentContext(
        workflow_id="wf1",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={},
    )
    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["best_quote"]["quote_id"] == "Q1"
    assert output.data["evaluation_scores"]["Q1"]["opportunity_value"] == 20
    assert output.data["evaluation_scores"]["Q1"]["supplier_score"] == 100


class DummyOrchestrator:
    def __init__(self, agent):
        self.agent = agent

    def execute_workflow(self, workflow_name, input_data):
        assert workflow_name == "quote_evaluation"
        context = AgentContext(
            workflow_id="wf2",
            agent_id="quote_evaluation",
            user_id="u1",
            input_data=input_data,
        )
        result = self.agent.run(context)
        return {"status": "completed", "workflow_id": "wf2", "result": result.data}


def test_quote_evaluation_endpoint(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)
    monkeypatch.setattr(agent, "_fetch_quotes", _mock_quotes)
    monkeypatch.setattr(agent, "call_ollama", _mock_ollama)
    monkeypatch.setattr(agent, "_generate_comparison", _mock_comparison)
    app = FastAPI()
    app.include_router(router)
    app.state.orchestrator = DummyOrchestrator(agent)
    client = TestClient(app)
    resp = client.post("/workflows/quotes/evaluate", json={})
    assert resp.status_code == 200
    assert resp.json()["result"]["best_quote"]["quote_id"] == "Q1"
    assert resp.json()["result"]["evaluation_scores"]["Q1"]["opportunity_value"] == 20


def test_fetch_quotes_from_qdrant():
    nick = DummyNick()

    class DummyPoint:
        def __init__(self):
            self.id = "p1"
            self.payload = {
                "quote_id": "Q1",
                "supplier_name": "Supplier A",
                "total_amount": 100,
                "payment_terms": "Net 30",
                "delivery_terms": "3 days",
                "discount_percentage": 5,
                "baseline_total_amount": 120,
                "baseline_payment_terms": "Net 30",
                "document_type": "quote",
            }

    class DummyClient:
        def __init__(self):
            self.last_filter = None

        def scroll(self, scroll_filter=None, **_):
            self.last_filter = scroll_filter
            return [DummyPoint()], None

    nick.qdrant_client = DummyClient()
    agent = QuoteEvaluationAgent(nick)
    quotes = agent._fetch_quotes(["Supplier A"])
    assert quotes[0]["quote_id"] == "Q1"
    # Ensure document type filter is applied
    must_filters = nick.qdrant_client.last_filter.must
    assert any(f.key == "document_type" for f in must_filters)
