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
    assert engine._quantity_expression(conn, "schema", "table", "li") == "1"


def test_quantity_expression_detects_quantity_column():
    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    conn = DummyConn(["quantity", "other"])
    assert (
        engine._quantity_expression(conn, "schema", "table", "li")
        == "COALESCE(li.quantity, 1)"
    )


def test_fetch_supplier_data_uses_line_items(monkeypatch):
    calls = []
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )

    def fake_price(conn, schema, table, alias):
        calls.append(("price", schema, table, alias))
        return "0"

    def fake_qty(conn, schema, table, alias):
        calls.append(("qty", schema, table, alias))
        return "1"

    def fake_cols(conn, schema, table):
        if (schema, table) == ("proc", "supplier"):
            calls.append(("cols", schema, table))
        return []

    monkeypatch.setattr(engine, "_price_expression", fake_price)
    monkeypatch.setattr(engine, "_quantity_expression", fake_qty)
    monkeypatch.setattr(engine, "_get_columns", fake_cols)
    monkeypatch.setattr(
        pd, "read_sql", lambda sql, conn: pd.DataFrame({"supplier_id": []})
    )

    engine.fetch_supplier_data()

    assert ("price", "proc", "po_line_items_agent", "li") in calls
    assert ("qty", "proc", "po_line_items_agent", "li") in calls
    assert ("price", "proc", "invoice_line_items_agent", "ili") in calls
    assert ("qty", "proc", "invoice_line_items_agent", "ili") in calls
    assert ("cols", "proc", "supplier") in calls


def test_fetch_supplier_data_uses_delivery_lead_time(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(
        engine,
        "_get_columns",
        lambda conn, schema, table: [
            "delivery_lead_time_days",
            "trading_name",
            "legal_structure",
        ],
    )
    captured = {}

    def fake_read_sql(sql, conn):
        captured["sql"] = sql
        return pd.DataFrame({"supplier_id": []})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    engine.fetch_supplier_data()

    assert "delivery_lead_time_days" in captured["sql"]
    assert "on_time_pct" in captured["sql"]
    # ensure string values are guarded by a numeric regex and cast
    assert "~ '^-?\\d+(\\.\\d+)?$'" in captured["sql"]
    assert "::numeric" in captured["sql"]

    # ensure additional supplier fields are projected explicitly
    assert "s.trading_name" in captured["sql"]
    assert "s.legal_structure" in captured["sql"]


def test_fetch_supplier_data_defaults_on_time_pct(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(engine, "_get_columns", lambda conn, schema, table: [])
    captured = {}

    def fake_read_sql(sql, conn):
        captured["sql"] = sql
        return pd.DataFrame({"supplier_id": []})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    engine.fetch_supplier_data()

    assert "0.0 AS on_time_pct" in captured["sql"]


def test_fetch_procurement_flow_builds_expected_query(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    captured = {}

    def fake_read_sql(sql, conn):
        captured["sql"] = sql
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    engine.fetch_procurement_flow()

    sql = captured["sql"]
    for table in [
        "proc.contracts",
        "proc.supplier",
        "proc.purchase_order_agent",
        "proc.po_line_items_agent",
        "proc.invoice_agent",
        "proc.invoice_line_items_agent",
        "proc.cat_product_mapping",
    ]:
        assert table in sql


def test_fetch_procurement_flow_embeds_summary(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    sample_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "product": ["Widget"],
            "category_level_1": ["Goods"],
            "category_level_2": ["Hardware"],
            "category_level_3": ["Components"],
            "category_level_4": ["Widgets"],
            "category_level_5": ["Widget Type"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: sample_df)

    called = {}

    def fake_embed(df):
        called["df"] = df

    monkeypatch.setattr(engine, "_embed_procurement_summary", fake_embed)

    engine.fetch_procurement_flow(embed=True)

    assert called["df"].equals(sample_df)


def test_train_procurement_context_embeds_schema(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )

    # stub column discovery
    monkeypatch.setattr(
        engine,
        "_get_columns",
        lambda conn, schema, table: [f"{table}_c1", f"{table}_c2"],
    )

    captured = {}

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def upsert_texts(self, texts, metadata=None):
            captured["texts"] = texts
            captured["metadata"] = metadata

    def fake_flow(self, embed=False):
        captured["embed"] = embed
        return pd.DataFrame()

    monkeypatch.setattr(QueryEngine, "fetch_procurement_flow", fake_flow)

    import services.rag_service as rag_module

    monkeypatch.setattr(rag_module, "RAGService", DummyRAG)

    engine.train_procurement_context()

    assert "contracts_c1" in captured["texts"][0]
    assert captured["metadata"]["record_id"] == "procurement_schema"
    assert captured["embed"] is True


class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        pass
