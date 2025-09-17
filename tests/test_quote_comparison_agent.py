import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentContext, AgentStatus
from agents.quote_comparison_agent import QuoteComparisonAgent

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(script_user="tester")
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: None,
            log_action=lambda **_: None,
        )
        self.ollama_options = lambda: {}
        self.pandas_connection = None

    def get_db_connection(self):  # pragma: no cover - defensive
        raise AssertionError("Database access should not be required in this test")


def _build_context(quotes_payload, extra_input=None):
    base_input = {
        "quotes": quotes_payload,
        "weights": 1.0,
        "supplier_ids": ["S1", "S2"],
    }
    if extra_input:
        base_input.update(extra_input)
    return AgentContext(
        workflow_id="wf-1",
        agent_id="quote_comparison",
        user_id="tester",
        input_data=base_input,
    )


def test_quote_comparison_prefers_passed_quotes(monkeypatch):
    nick = DummyNick()
    agent = QuoteComparisonAgent(nick)

    def fail_read(*_args, **_kwargs):  # pragma: no cover - should not be invoked
        raise AssertionError("QuoteComparisonAgent should not read from the database")

    monkeypatch.setattr(agent, "_read_table", fail_read)

    quotes_payload = [
        {
            "name": "weighting",
            "total_spend": 1.0,
            "total_cost": 0,
            "unit_price": 0,
            "quote_file_s3_path": None,
            "tenure": None,
            "volume": None,
        },
        {
            "name": "Supplier A",
            "supplier_id": "S1",
            "total_spend": 100,
            "total_cost": 90,
            "unit_price": 10,
            "volume": 10,
            "tenure": "Net 30",
            "quote_file_s3_path": "s3://bucket/s1.pdf",
        },
        {
            "name": "Supplier B",
            "supplier_id": "S2",
            "total_spend": 200,
            "total_cost": 180,
            "unit_price": 9,
            "volume": 20,
            "tenure": "Net 45",
            "quote_file_s3_path": "s3://bucket/s2.pdf",
        },
    ]

    context = _build_context(quotes_payload)
    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    comparison = result.data["comparison"]
    assert len(comparison) == 3
    assert comparison[0]["name"] == "weighting"
    suppliers = {row["supplier_id"] for row in comparison if row["name"] != "weighting"}
    assert suppliers == {"S1", "S2"}
    assert comparison[1]["quote_file_s3_path"] == "s3://bucket/s1.pdf"
    assert comparison[1]["currency"] == "GBP"
    assert comparison[1]["weighting_score"] > comparison[2]["weighting_score"]
    recommended = result.data.get("recommended_quote")
    assert recommended is not None
    assert recommended["supplier_id"] == "S1"
    assert recommended["ticker"] == "RECOMMENDED"
    assert recommended["weighting_score"] == comparison[1]["weighting_score"]


def test_quote_comparison_filters_by_supplier_tokens(monkeypatch):
    nick = DummyNick()
    agent = QuoteComparisonAgent(nick)

    # Avoid database fallbacks for the test scenario
    monkeypatch.setattr(agent, "_read_table", lambda *_args, **_kwargs: pd.DataFrame())

    quotes_payload = [
        {
            "name": "weighting",
            "total_spend": 1.0,
            "total_cost": 0,
            "unit_price": 0,
            "quote_file_s3_path": None,
            "tenure": None,
            "volume": None,
        },
        {
            "name": "Supplier A",
            "supplier_id": "S1",
            "total_spend": 100,
            "total_cost": 90,
            "unit_price": 10,
            "volume": 10,
        },
        {
            "name": "Supplier B",
            "supplier_id": None,
            "total_spend": 200,
            "total_cost": 180,
            "unit_price": 9,
            "volume": 20,
        },
    ]

    context = _build_context(
        quotes_payload,
        extra_input={"supplier_ids": [], "supplier_names": ["Supplier B"]},
    )

    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    comparison = result.data["comparison"]
    assert len(comparison) == 2
    assert comparison[0]["name"] == "weighting"
    assert comparison[1]["name"] == "Supplier B"
    assert comparison[1]["supplier_id"] is None
    recommended = result.data.get("recommended_quote")
    assert recommended is not None
    assert recommended["name"] == "Supplier B"
