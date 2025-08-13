import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import discrepancy_detection_agent as dd_module
from agents.base_agent import AgentContext, AgentStatus
from psycopg2.errors import UndefinedColumn


class FakeCursor:
    def __init__(self, row, undefined_cols=None, conn=None):
        self.row = row
        self.undefined_cols = set(undefined_cols or [])
        self.conn = conn

    def execute(self, query, params=None):
        for col in self.undefined_cols:
            if col in query:
                raise UndefinedColumn()

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class FakeConn:
    def __init__(self, row, undefined_cols=None):
        self.rollback_called = False
        self.rollback_calls = 0
        self._cursor = FakeCursor(row, undefined_cols, self)

    def cursor(self):
        return self._cursor

    def rollback(self):
        self.rollback_called = True
        self.rollback_calls += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def build_agent(conn):
    nick = SimpleNamespace(get_db_connection=lambda: conn, settings=SimpleNamespace())
    return dd_module.DiscrepancyDetectionAgent(nick)


def make_context(doc):
    return AgentContext(workflow_id="w", agent_id="a", user_id="u", input_data={"extracted_docs": [doc]})


def test_fallback_to_vendor_column(monkeypatch):
    conn = FakeConn(("Acme", "2025-01-01", 100.0), undefined_cols={"vendor_name"})
    agent = build_agent(conn)
    out = agent.run(make_context({"doc_type": "Invoice", "id": "1"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["mismatches"] == []
    assert conn.rollback_calls == 1


def test_detects_missing_fields(monkeypatch):
    conn = FakeConn((None, None, 0))
    agent = build_agent(conn)
    out = agent.run(make_context({"doc_type": "Invoice", "id": "1"}))
    assert out.data["mismatches"][0]["checks"]["vendor_name"] == "missing"
    assert out.data["mismatches"][0]["checks"]["invoice_date"] == "missing"
    assert out.data["mismatches"][0]["checks"]["total_amount"] == "invalid"


def test_handles_missing_vendor_columns(monkeypatch):
    conn = FakeConn(("2025-01-01", 100.0), undefined_cols={"vendor_name", "vendor"})
    agent = build_agent(conn)
    out = agent.run(make_context({"doc_type": "Invoice", "id": "1"}))
    checks = out.data["mismatches"][0]["checks"]
    assert checks["vendor_name"] == "missing"
    assert "db_error" not in checks
    assert conn.rollback_calls == 2
