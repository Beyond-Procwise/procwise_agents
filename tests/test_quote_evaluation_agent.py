import os
import sys
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

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
            "supplier_id": "S1",
            "total_spend": 1000,
            "tenure": "12 months",
            "total_cost": 1000,
            "unit_price": 10,
            "volume": 100,
            "quote_file_s3_path": "s3://bucket/q1.pdf",
        },
        {
            "quote_id": "Q2",
            "supplier_name": "Supplier B",
            "supplier_id": "S2",
            "total_spend": 1200,
            "tenure": "12 months",
            "total_cost": 1200,
            "unit_price": 12,
            "volume": 100,
            "quote_file_s3_path": "s3://bucket/q2.pdf",
        },
    ]


class QuoteCursorStub:
    def __init__(self, connection):
        self.connection = connection
        self.description = []
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.connection.queries.append((sql, params))
        if "FROM proc.quote_agent" in sql:
            self._rows = self.connection.quotes
            self.description = [(col,) for col in self.connection.quote_columns]
        elif "FROM proc.quote_line_items_agent" in sql:
            self._rows = self.connection.lines
            self.description = [(col,) for col in self.connection.line_columns]
        else:
            self._rows = []
            self.description = []

    def fetchall(self):
        return list(self._rows)


class QuoteConnectionStub:
    def __init__(self, quotes, lines, quote_cols, line_cols):
        self.quotes = quotes
        self.lines = lines
        self.quote_columns = quote_cols
        self.line_columns = line_cols
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return QuoteCursorStub(self)


def build_quote_connection(quotes, lines, quote_columns, line_columns):
    return QuoteConnectionStub(quotes, lines, quote_columns, line_columns)


def test_quote_evaluation_agent_run(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)
    monkeypatch.setattr(agent, "_fetch_quotes_from_database", lambda *_, **__: [])
    monkeypatch.setattr(agent, "_fetch_quotes", _mock_quotes)
    context = AgentContext(
        workflow_id="wf1",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={},
    )
    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    quotes = output.data["quotes"]
    weighting = quotes[0]
    assert weighting["name"] == "weighting"
    assert weighting["total_spend"] == pytest.approx(output.data["weights"])
    supplier_a = next(q for q in quotes if q["name"] == "Supplier A")
    assert supplier_a["total_spend"] == 1000
    assert output.data["weights"] == pytest.approx(1.0)


def test_quote_evaluation_handles_no_quotes(monkeypatch):
    """Agent should succeed gracefully when no quotes are found."""
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)
    monkeypatch.setattr(agent, "_fetch_quotes_from_database", lambda *_, **__: [])
    monkeypatch.setattr(agent, "_fetch_quotes", lambda *_, **__: [])
    context = AgentContext(
        workflow_id="wf_empty",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={},
    )
    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    weighting = output.data["quotes"][0]
    assert weighting["name"] == "weighting"
    assert weighting["total_spend"] == pytest.approx(output.data["weights"])
    assert len(output.data["quotes"]) == 1
    assert output.data["weights"] == pytest.approx(1.0)


def test_quote_evaluation_handles_no_quotes_message(monkeypatch):
    """Agent should succeed gracefully when no quotes are found."""
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)
    monkeypatch.setattr(agent, "_fetch_quotes_from_database", lambda *_, **__: [])
    monkeypatch.setattr(agent, "_fetch_quotes", lambda *_, **__: [])
    context = AgentContext(
        workflow_id="wf_empty",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={},
    )
    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["quotes"][0]["name"] == "weighting"
    assert output.data.get("message") == "No quotes found"


def test_quote_evaluation_limits_to_ranked_suppliers(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)

    captured = {}
    captured_db = {}

    def mock_db_fetch(supplier_names, supplier_ids, product_category=None):
        captured_db["supplier_names"] = supplier_names
        captured_db["supplier_ids"] = supplier_ids
        captured_db["product_category"] = product_category
        return []

    def mock_fetch(supplier_names, supplier_ids=None, product_category=None):
        captured["supplier_names"] = supplier_names
        captured["supplier_ids"] = supplier_ids

        return [
            {
                "quote_id": "Q1",
                "supplier_name": "Supplier B",
                "supplier_id": "S2",
                "total_cost": 220,
                "unit_price": 11,
            },
            {
                "quote_id": "Q2",
                "supplier_name": "Supplier A",
                "supplier_id": "S1",
                "total_cost": 180,
                "unit_price": 9,
            },
            {
                "quote_id": "Q3",
                "supplier_name": "Supplier D",
                "supplier_id": "S4",
                "total_cost": 260,
                "unit_price": 13,
            },
            {
                "quote_id": "Q4",
                "supplier_name": "Supplier C",
                "supplier_id": "S3",
                "total_cost": 200,
                "unit_price": 10,
            },
        ]

    monkeypatch.setattr(agent, "_fetch_quotes_from_database", mock_db_fetch)
    monkeypatch.setattr(agent, "_fetch_quotes", mock_fetch)

    ranking = [
        {"supplier_name": "Supplier C", "final_score": 9.5},
        {"supplier_name": "Supplier A", "final_score": 9.1},
        {"supplier_name": "Supplier B", "final_score": 8.9},
        {"supplier_name": "Supplier D", "final_score": 8.2},
    ]

    context = AgentContext(
        workflow_id="wf_rank",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={"ranking": ranking},
    )

    output = agent.run(context)

    assert captured_db["supplier_names"] == [
        "Supplier C",
        "Supplier A",
        "Supplier B",
    ]
    assert captured_db["supplier_ids"] == []
    assert captured["supplier_names"] == [
        "Supplier C",
        "Supplier A",
        "Supplier B",
    ]
    assert captured.get("supplier_ids") == []

    names = [q["name"] for q in output.data["quotes"] if q["name"] != "weighting"]
    assert names == ["Supplier C", "Supplier A", "Supplier B"]
    assert len(names) == 3


def test_quote_evaluation_uses_supplier_ids(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)

    captured = {}
    captured_db = {}

    def mock_db_fetch(supplier_names, supplier_ids, product_category=None):
        captured_db["supplier_names"] = supplier_names
        captured_db["supplier_ids"] = supplier_ids
        captured_db["product_category"] = product_category
        return []

    def mock_fetch(supplier_names, supplier_ids=None, product_category=None):
        captured["supplier_names"] = supplier_names
        captured["supplier_ids"] = supplier_ids
        return [
            {
                "quote_id": "Q1",
                "supplier_name": "Beta Manufacturing",
                "supplier_id": "S100",
                "total_cost": 180,
                "unit_price": 9,
            },
            {
                "quote_id": "Q2",
                "supplier_name": "Alpha Supplies",
                "supplier_id": "S200",
                "total_cost": 220,
                "unit_price": 11,
            },
            {
                "quote_id": "Q3",
                "supplier_name": "Gamma Goods",
                "supplier_id": "S300",
                "total_cost": 260,
                "unit_price": 13,
            },
        ]

    monkeypatch.setattr(agent, "_fetch_quotes_from_database", mock_db_fetch)
    monkeypatch.setattr(agent, "_fetch_quotes", mock_fetch)

    ranking = [
        {"supplier_id": "S100", "final_score": 9.8},
        {"supplier_id": "S200", "final_score": 9.0},
        {"supplier_id": "S300", "final_score": 8.4},
    ]

    context = AgentContext(
        workflow_id="wf_rank_ids",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={"ranking": ranking},
    )

    output = agent.run(context)

    assert captured_db["supplier_names"] == []
    assert captured_db["supplier_ids"] == ["S100", "S200", "S300"]
    assert captured["supplier_names"] == []
    assert captured["supplier_ids"] == ["S100", "S200", "S300"]

    names = [q["name"] for q in output.data["quotes"] if q["name"] != "weighting"]
    assert names == ["Beta Manufacturing", "Alpha Supplies", "Gamma Goods"]


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
    monkeypatch.setattr(agent, "_fetch_quotes_from_database", lambda *_, **__: [])
    monkeypatch.setattr(agent, "_fetch_quotes", _mock_quotes)
    app = FastAPI()
    app.include_router(router)
    app.state.orchestrator = DummyOrchestrator(agent)
    client = TestClient(app)
    resp = client.post("/workflows/quotes/evaluate", json={})
    assert resp.status_code == 200
    quotes = resp.json()["result"]["quotes"]
    supplier_b = next(q for q in quotes if q["name"] == "Supplier B")
    assert supplier_b["unit_price"] == 12


def test_fetch_quotes_from_qdrant():
    nick = DummyNick()

    class DummyPoint:
        def __init__(self):
            self.id = "p1"
            self.payload = {
                "quote_id": "Q1",
                "supplier_name": "Supplier A",
                "total_spend": 1000,
                "tenure": "12 months",
                "total_cost": 1000,
                "unit_price": 10,
                "volume": 100,
                "quote_file_s3_path": "s3://bucket/q1.pdf",
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


def test_fetch_quotes_handles_missing_supplier_index(monkeypatch):
    nick = DummyNick()

    class DummyPoint:
        def __init__(self):
            self.id = "p1"
            self.payload = {
                "quote_id": "Q1",
                "supplier_name": "Supplier A",
                "total_spend": 1000,
                "tenure": "12 months",
                "total_cost": 1000,
                "unit_price": 10,
                "volume": 100,
                "quote_file_s3_path": "s3://bucket/q1.pdf",
                "document_type": "quote",
            }

    class FailingClient:
        def __init__(self):
            self.attempts = 0

        def scroll(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise Exception(
                    "Bad request: Index required but not found for \"supplier_name\""
                )
            return [DummyPoint()], None

    nick.qdrant_client = FailingClient()
    agent = QuoteEvaluationAgent(nick)
    quotes = agent._fetch_quotes(["Supplier A"], product_category="")
    assert quotes[0]["quote_id"] == "Q1"
    assert nick.qdrant_client.attempts == 2
    assert quotes[0]["quote_file_s3_path"] == "s3://bucket/q1.pdf"


def test_fetch_quotes_from_database_enriches_results():
    nick = DummyNick()

    quote_rows = [
        (
            "Q1",
            "S1",
            "Supplier One",
            "B1",
            date(2024, 5, 1),
            date(2024, 6, 1),
            "USD",
            Decimal("120.00"),
            Decimal("5.00"),
            Decimal("6.00"),
            Decimal("126.00"),
            "PO1",
            "US",
            "NA",
            "NO",
            "AUTO",
            "Widget contract",
            datetime(2024, 5, 1, 10, 0, 0),
            "tester",
            "tester",
            datetime(2024, 5, 2, 10, 0, 0),
        )
    ]
    line_rows = [
        (
            "Q1",
            "QL1",
            1,
            "I1",
            "Widget Basic",
            10,
            "EA",
            Decimal("12.00"),
            Decimal("120.00"),
            Decimal("5.00"),
            Decimal("6.00"),
            Decimal("126.00"),
            "USD",
        )
    ]

    quote_columns = [
        "quote_id",
        "supplier_id",
        "supplier_name",
        "buyer_id",
        "quote_date",
        "validity_date",
        "currency",
        "total_amount",
        "tax_percent",
        "tax_amount",
        "total_amount_incl_tax",
        "po_id",
        "country",
        "region",
        "ai_flag_required",
        "trigger_type",
        "trigger_context_description",
        "created_date",
        "created_by",
        "last_modified_by",
        "last_modified_date",
    ]
    line_columns = [
        "quote_id",
        "quote_line_id",
        "line_number",
        "item_id",
        "item_description",
        "quantity",
        "unit_of_measure",
        "unit_price",
        "line_total",
        "tax_percent",
        "tax_amount",
        "total_amount",
        "currency",
    ]

    connection = build_quote_connection(
        quote_rows, line_rows, quote_columns, line_columns
    )
    nick.get_db_connection = lambda: connection

    agent = QuoteEvaluationAgent(nick)
    quotes = agent._fetch_quotes_from_database(["Supplier One"], ["S1"])

    assert len(quotes) == 1
    record = quotes[0]
    assert record["supplier_id"] == "S1"
    assert record["line_items_count"] == 1
    assert record["line_items"][0]["item_description"] == "Widget Basic"
    assert record["avg_unit_price"] == pytest.approx(12.0)
    assert record["total_line_amount"] == pytest.approx(120.0)
    assert record["total_amount"] == Decimal("120.00")
    assert record["category_match"] is False
    assert connection.queries[0][0].strip().startswith("SELECT")


def test_fetch_quotes_from_database_retains_quotes_when_category_misses():
    nick = DummyNick()

    quote_rows = [
        (
            "Q1",
            "S1",
            "Supplier One",
            "B1",
            date(2024, 5, 1),
            date(2024, 6, 1),
            "USD",
            Decimal("120.00"),
            Decimal("5.00"),
            Decimal("6.00"),
            Decimal("126.00"),
            "PO1",
            "US",
            "NA",
            "NO",
            "AUTO",
            "Widget contract",
            datetime(2024, 5, 1, 10, 0, 0),
            "tester",
            "tester",
            datetime(2024, 5, 2, 10, 0, 0),
        )
    ]
    line_rows = [
        (
            "Q1",
            "QL1",
            1,
            "I1",
            "Widget Basic",
            10,
            "EA",
            Decimal("12.00"),
            Decimal("120.00"),
            Decimal("5.00"),
            Decimal("6.00"),
            Decimal("126.00"),
            "USD",
        )
    ]
    quote_columns = [
        "quote_id",
        "supplier_id",
        "supplier_name",
        "buyer_id",
        "quote_date",
        "validity_date",
        "currency",
        "total_amount",
        "tax_percent",
        "tax_amount",
        "total_amount_incl_tax",
        "po_id",
        "country",
        "region",
        "ai_flag_required",
        "trigger_type",
        "trigger_context_description",
        "created_date",
        "created_by",
        "last_modified_by",
        "last_modified_date",
    ]
    line_columns = [
        "quote_id",
        "quote_line_id",
        "line_number",
        "item_id",
        "item_description",
        "quantity",
        "unit_of_measure",
        "unit_price",
        "line_total",
        "tax_percent",
        "tax_amount",
        "total_amount",
        "currency",
    ]

    connection = build_quote_connection(
        quote_rows, line_rows, quote_columns, line_columns
    )
    nick.get_db_connection = lambda: connection

    agent = QuoteEvaluationAgent(nick)
    quotes = agent._fetch_quotes_from_database(
        ["Supplier One"], ["S1"], product_category="Raw Materials"
    )

    assert len(quotes) == 1
    assert quotes[0]["category_match"] is False


def test_quote_evaluation_uses_supplier_candidates(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)

    captured_db = {}
    captured_vector = {}

    def mock_db_fetch(supplier_names, supplier_ids, product_category=None):
        captured_db["names"] = supplier_names
        captured_db["ids"] = supplier_ids
        return []

    def mock_fetch(supplier_names, supplier_ids=None, product_category=None):
        captured_vector["names"] = supplier_names
        captured_vector["ids"] = supplier_ids
        return []

    monkeypatch.setattr(agent, "_fetch_quotes_from_database", mock_db_fetch)
    monkeypatch.setattr(agent, "_fetch_quotes", mock_fetch)

    context = AgentContext(
        workflow_id="wf_candidates",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={"supplier_candidates": ["S10", "S20"]},
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert captured_db["ids"] == ["S10", "S20"]
    assert captured_vector["ids"] == ["S10", "S20"]
    assert output.data.get("message") == "No quotes found"


def test_quote_evaluation_supplier_fallback(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)

    db_calls = []

    def mock_db_fetch(names, ids, product_category=None):
        db_calls.append((tuple(names), tuple(ids), product_category))
        if product_category is None and list(ids) == ["S1"]:
            return [
                {
                    "quote_id": "QF1",
                    "supplier_id": "S1",
                    "supplier_name": "Supplier One",
                    "total_amount": 100,
                    "line_items": [],
                    "line_items_count": 0,
                }
            ]
        return []

    monkeypatch.setattr(agent, "_fetch_quotes_from_database", mock_db_fetch)
    monkeypatch.setattr(agent, "_fetch_quotes", lambda *_, **__: [])

    context = AgentContext(
        workflow_id="wf_supplier_fallback",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={
            "product_category": "Raw",
            "ranking": [{"supplier_name": "Supplier One", "supplier_id": "S1"}],
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["retrieval_strategy"] == "supplier_fallback"
    supplier_entries = [q for q in output.data["quotes"] if q["name"] == "Supplier One"]
    assert supplier_entries
    assert any(call[2] is None for call in db_calls)


def test_quote_evaluation_category_fallback_retained(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)

    db_calls = []

    def mock_db_fetch(names, ids, product_category=None):
        db_calls.append((tuple(names), tuple(ids), product_category))
        if not names and not ids and product_category == "Raw Materials":
            return [
                {
                    "quote_id": "QCAT",
                    "supplier_id": "S99",
                    "supplier_name": "Category Supplier",
                    "total_amount": 250,
                    "line_items": [],
                    "line_items_count": 0,
                    "category_match": True,
                }
            ]
        return []

    monkeypatch.setattr(agent, "_fetch_quotes_from_database", mock_db_fetch)
    monkeypatch.setattr(agent, "_fetch_quotes", lambda *_, **__: [])

    context = AgentContext(
        workflow_id="wf_category_fallback",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={
            "product_category": "Raw Materials",
            "ranking": [{"supplier_name": "Supplier One", "supplier_id": "S1"}],
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["retrieval_strategy"] == "category_fallback"
    supplier_names = {q["name"] for q in output.data["quotes"]}
    assert "Category Supplier" in supplier_names
    assert any(call[0] == tuple() and call[2] == "Raw Materials" for call in db_calls)


def test_process_handles_empty_product_type(monkeypatch):
    nick = DummyNick()
    agent = QuoteEvaluationAgent(nick)

    def capture_fetch(supplier_names, supplier_ids=None, product_category=None):
        capture_fetch.captured = product_category
        capture_fetch.captured_ids = supplier_ids
        return _mock_quotes()

    monkeypatch.setattr(agent, "_fetch_quotes_from_database", lambda *_, **__: [])
    monkeypatch.setattr(agent, "_fetch_quotes", capture_fetch)
    context = AgentContext(
        workflow_id="wf3",
        agent_id="quote_evaluation",
        user_id="u1",
        input_data={"product_type": ""},
    )
    agent.run(context)
    assert capture_fetch.captured is None
