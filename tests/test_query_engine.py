import os
import sys
import types
from typing import Any

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.query_engine import PROCUREMENT_CATEGORY_FIELDS, QueryEngine


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


def test_fetch_supplier_data_filters_candidates(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(engine, "_get_columns", lambda *a, **k: [])

    sample = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "po_spend": [10.0, 20.0],
            "invoice_spend": [5.0, 6.0],
            "total_spend": [15.0, 26.0],
            "invoice_count": [1, 2],
            "on_time_pct": [1.0, 0.5],
        }
    )

    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: sample.copy())

    result = engine.fetch_supplier_data({"supplier_candidates": ["S2"]})

    assert list(result["supplier_id"]) == ["S2"]
    assert list(result["supplier_name"]) == ["Beta"]


def test_fetch_supplier_data_uses_directory_fallback(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(engine, "_get_columns", lambda *a, **k: [])

    sample = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Alpha"],
            "po_spend": [10.0],
            "invoice_spend": [5.0],
            "total_spend": [15.0],
            "invoice_count": [1],
            "on_time_pct": [1.0],
        }
    )

    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: sample.copy())

    payload = {
        "supplier_candidates": ["S3"],
        "supplier_directory": [
            {"supplier_id": "S3", "supplier_name": "Gamma Corp"}
        ],
    }

    result = engine.fetch_supplier_data(payload)

    assert list(result["supplier_id"]) == ["S3"]
    assert result.loc[0, "supplier_name"] == "Gamma Corp"
    assert set(result.columns) == set(sample.columns)


def test_fetch_procurement_flow_builds_expected_query(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    queries = []

    base_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    category_df = pd.DataFrame(
        {
            "product": ["Widget"],
            "category_level_1": ["Goods"],
            "category_level_2": ["Hardware"],
            "category_level_3": ["Components"],
            "category_level_4": ["Widgets"],
            "category_level_5": ["Widget Type"],
        }
    )

    def fake_read_sql(sql, conn):
        queries.append(sql)
        if "proc.cat_product_mapping" in sql:
            return category_df
        return base_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df = engine.fetch_procurement_flow()

    assert queries, "No SQL queries captured"
    assert any("proc.cat_product_mapping" in q for q in queries)

    main_query = next(q for q in queries if "proc.cat_product_mapping" not in q)
    for table in [
        "proc.contracts",
        "proc.supplier",
        "proc.purchase_order_agent",
        "proc.po_line_items_agent",
        "proc.invoice_agent",
        "proc.invoice_line_items_agent",
    ]:
        assert table in main_query

    assert not df.empty
    assert df.loc[0, "product"] == "Widget"


def test_fetch_procurement_flow_embeds_summary(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    base_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    category_df = pd.DataFrame(
        {
            "product": ["Widget"],
            "category_level_1": ["Goods"],
            "category_level_2": ["Hardware"],
            "category_level_3": ["Components"],
            "category_level_4": ["Widgets"],
            "category_level_5": ["Widget Type"],
        }
    )

    def fake_read_sql(sql, conn):
        if "proc.cat_product_mapping" in sql:
            return category_df
        return base_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    called = {}

    def fake_embed(df):
        called["df"] = df

    monkeypatch.setattr(engine, "_embed_procurement_summary", fake_embed)

    engine.fetch_procurement_flow(embed=True)

    expected_df = base_df.copy()
    for field in PROCUREMENT_CATEGORY_FIELDS:
        expected_df[field] = category_df.loc[0, field]

    pd.testing.assert_frame_equal(
        called["df"].reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_like=True,
    )


def test_fetch_procurement_flow_handles_missing_category_mapping(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )

    base_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    empty_category_df = pd.DataFrame(columns=PROCUREMENT_CATEGORY_FIELDS)

    def fake_read_sql(sql, conn):
        if "proc.cat_product_mapping" in sql:
            return empty_category_df
        return base_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df = engine.fetch_procurement_flow()

    for field in PROCUREMENT_CATEGORY_FIELDS:
        assert df[field].isna().all()


def test_train_procurement_context_embeds_schema(monkeypatch):
    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    agent = types.SimpleNamespace(
        get_db_connection=lambda: DummyConnection(),
        settings=types.SimpleNamespace(qdrant_collection_name="ProcWise_document_embeddings"),
    )
    engine = QueryEngine(agent_nick=agent)

    # stub column discovery across all procurement tables
    monkeypatch.setattr(
        engine,
        "_get_columns",
        lambda conn, schema, table: [f"{table}_c1", f"{table}_c2"],
    )

    sample_frames = {
        "proc.contracts": pd.DataFrame({"contract_id": ["CO1"], "supplier_id": ["SI1"]}),
        "proc.supplier": pd.DataFrame({"supplier_id": ["SI1"], "supplier_name": ["Acme"]}),
        "proc.purchase_order_agent": pd.DataFrame({"po_id": ["PO1"], "supplier_id": ["SI1"]}),
        "proc.po_line_items_agent": pd.DataFrame({"po_id": ["PO1"], "item_description": ["Widget"]}),
        "proc.invoice_agent": pd.DataFrame({"invoice_id": ["IN1"], "po_id": ["PO1"]}),
        "proc.invoice_line_items_agent": pd.DataFrame({"invoice_id": ["IN1"], "po_id": ["PO1"]}),
        "proc.cat_product_mapping": pd.DataFrame({"product": ["Widget"], "category_level_2": ["Hardware"]}),
        "proc.quote_agent": pd.DataFrame({"quote_id": ["Q1"], "po_id": ["PO1"]}),
        "proc.quote_line_items_agent": pd.DataFrame({"quote_id": ["Q1"], "line_total": [100.0]}),
    }

    def fake_read_sql(sql, conn):
        for canonical, df in sample_frames.items():
            if canonical in sql:
                return df.copy()
        return pd.DataFrame()

    captured: dict[str, Any] = {}

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def upsert_texts(self, texts, metadata=None):
            captured["texts"] = texts
            captured["metadata"] = metadata

    class DummyManager:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection_name")

        def build_data_flow_map(self, tables, table_name_map=None):
            captured["tables"] = tables
            captured["table_name_map"] = table_name_map
            return ([{"status": "linked", "relationship_type": "references"}], {"paths": [], "supplier_flows": []})

        def persist_knowledge_graph(self, relations, graph):
            captured["persist"] = (relations, graph)

    def fake_flow(self, embed=False):
        captured["embed"] = embed
        return pd.DataFrame()

    monkeypatch.setattr(QueryEngine, "fetch_procurement_flow", fake_flow)

    import services.rag_service as rag_module
    import services.data_flow_manager as df_module
    import engines.query_engine as qe_module
    import services.procurement_knowledge_service as pk_module

    class DummyKnowledgeService:
        def __init__(self, *args, **kwargs):
            pass

        def load_briefs(self):
            captured["knowledge_loaded"] = True
            return [types.SimpleNamespace(identifier="brief-1", title="t", summary="s")]

        def embed_briefs(self, briefs):
            captured["knowledge_embedded"] = [b.identifier for b in briefs]

    monkeypatch.setattr(rag_module, "RAGService", DummyRAG)
    monkeypatch.setattr(df_module, "DataFlowManager", DummyManager)
    monkeypatch.setattr(qe_module, "read_sql_compat", fake_read_sql)
    monkeypatch.setattr(pk_module, "ProcurementKnowledgeService", DummyKnowledgeService)

    engine.train_procurement_context()

    assert captured["metadata"]["record_id"] == "procurement_schema"
    assert captured["metadata"]["document_type"] == "procurement_schema"
    assert any("contracts_c1" in text for text in captured["texts"])
    assert isinstance(captured["tables"], dict) and "contracts" in captured["tables"]
    assert captured["table_name_map"]["contracts"] == "proc.contracts"
    assert captured["persist"][0][0]["status"] == "linked"
    assert captured["embed"] is True
    assert captured["knowledge_loaded"] is True
    assert captured["knowledge_embedded"] == ["brief-1"]


class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        pass
