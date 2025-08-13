import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import discrepancy_detection_agent as dd_module
from agents.base_agent import AgentContext, AgentStatus
from psycopg2.errors import UndefinedColumn


class FakeCursor:
    def __init__(self, row, raise_undefined=False):
        self.row = row
        self.raise_undefined = raise_undefined
        self.calls = 0

    def execute(self, query, params):
        self.calls += 1
        if self.raise_undefined and self.calls == 1:
            raise UndefinedColumn()

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def build_agent(cursor):
    nick = SimpleNamespace(get_db_connection=lambda: FakeConn(cursor), settings=SimpleNamespace())
    return dd_module.DiscrepancyDetectionAgent(nick)


def make_context(doc):
    return AgentContext(workflow_id="w", agent_id="a", user_id="u", input_data={"extracted_docs": [doc]})


def test_fallback_to_vendor_column(monkeypatch):
    cursor = FakeCursor(("Acme", "2025-01-01", 100.0), raise_undefined=True)
    agent = build_agent(cursor)
    out = agent.run(make_context({"doc_type": "Invoice", "id": "1"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["mismatches"] == []


def test_detects_missing_fields(monkeypatch):
    cursor = FakeCursor((None, None, 0))
    agent = build_agent(cursor)
    out = agent.run(make_context({"doc_type": "Invoice", "id": "1"}))
    assert out.data["mismatches"][0]["checks"]["vendor_name"] == "missing"
    assert out.data["mismatches"][0]["checks"]["invoice_date"] == "missing"
    assert out.data["mismatches"][0]["checks"]["total_amount"] == "invalid"
