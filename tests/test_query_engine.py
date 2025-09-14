import os
import sys
import types
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.query_engine import QueryEngine


class DummyCursor:
    def __init__(self, cols):
        self._cols = cols

    def execute(self, sql, params):
        pass

    def fetchall(self):
        return [(c,) for c in self._cols]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyConn:
    def __init__(self, cols):
        self._cols = cols

    def cursor(self):
        return DummyCursor(self._cols)


def test_quantity_expression_defaults_to_one_when_missing():
    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    conn = DummyConn(["supplier_id"])  # no quantity column
    assert engine._quantity_expression(conn, "schema", "table") == "1"


def test_quantity_expression_detects_quantity_column():
    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    conn = DummyConn(["quantity", "other"])
    assert engine._quantity_expression(conn, "schema", "table") == "COALESCE(quantity, 1)"


def test_fetch_supplier_data_uses_line_items(monkeypatch):
    calls = []
    engine = QueryEngine(agent_nick=types.SimpleNamespace(
        get_db_connection=lambda: DummyContext()
    ))

    def fake_price(conn, schema, table):
        calls.append(("price", schema, table))
        return "0"

    def fake_qty(conn, schema, table):
        calls.append(("qty", schema, table))
        return "1"

    monkeypatch.setattr(engine, "_price_expression", fake_price)
    monkeypatch.setattr(engine, "_quantity_expression", fake_qty)

    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: pd.DataFrame({"supplier_id": []}))

    engine.fetch_supplier_data()

    assert ("price", "proc", "po_line_items_agent") in calls
    assert ("qty", "proc", "po_line_items_agent") in calls
    assert ("price", "proc", "invoice_line_items_agent") in calls
    assert ("qty", "proc", "invoice_line_items_agent") in calls


class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        pass
