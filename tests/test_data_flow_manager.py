from types import SimpleNamespace

import pandas as pd

from services.data_flow_manager import DataFlowManager


class DummyEmbedder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, text, normalize_embeddings=True):
        length = float(len(text))
        return [length, length / 2.0, 1.0]


class DummyQdrant:
    def __init__(self):
        self.collections = {}
        self.upserts = []

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise ValueError("collection missing")
        return SimpleNamespace(payload_schema={})

    def create_collection(self, collection_name: str, vectors_config):
        self.collections[collection_name] = {
            "vectors_config": vectors_config,
            "indexes": set(),
        }

    def create_payload_index(self, collection_name: str, field_name: str, field_schema, wait: bool):
        if collection_name in self.collections:
            self.collections[collection_name]["indexes"].add(field_name)

    def upsert(self, collection_name: str, points, wait: bool):
        self.upserts.append({"collection": collection_name, "points": points, "wait": wait})


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace()
        self.qdrant_client = DummyQdrant()
        self.embedding_model = DummyEmbedder()


def _tables_fixture():
    contracts = pd.DataFrame(
        [
            {"contract_id": "CO1", "supplier_id": "SI1"},
            {"contract_id": "CO2", "supplier_id": "SI2"},
        ]
    )
    supplier_master = pd.DataFrame(
        [
            {"supplier_id": "SI1", "supplier_name": "Supplier One"},
            {"supplier_id": "SI2", "supplier_name": "Supplier Two"},
        ]
    )
    purchase_orders = pd.DataFrame(
        [
            {"po_id": "PO1", "contract_id": "CO1", "supplier_id": "SI1"},
            {"po_id": "PO2", "contract_id": "CO2", "supplier_id": "SI2"},
        ]
    )
    purchase_order_lines = pd.DataFrame(
        [
            {"po_id": "PO1", "item_description": "Widget Service"},
            {"po_id": "PO2", "item_description": "Widget Service"},
        ]
    )
    invoices = pd.DataFrame(
        [
            {"invoice_id": "INV1", "po_id": "PO1"},
        ]
    )
    invoice_lines = pd.DataFrame(
        [
            {"invoice_id": "INV1", "po_id": "PO1"},
        ]
    )
    product_mapping = pd.DataFrame(
        [
            {"product": "Widget Service", "category_level_2": "Services"},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"quote_id": "Q1", "po_id": "PO1"},
        ]
    )
    quote_lines = pd.DataFrame(
        [
            {"quote_id": "Q1", "line_total": 100.0},
        ]
    )

    return {
        "contracts": contracts,
        "supplier_master": supplier_master,
        "purchase_orders": purchase_orders,
        "purchase_order_lines": purchase_order_lines,
        "invoices": invoices,
        "invoice_lines": invoice_lines,
        "product_mapping": product_mapping,
        "quotes": quotes,
        "quote_lines": quote_lines,
    }


def test_data_flow_manager_builds_graph_and_persists():
    nick = DummyNick()
    manager = DataFlowManager(nick)
    tables = _tables_fixture()

    table_name_map = {
        "contracts": "proc.contracts",
        "supplier_master": "proc.supplier",
        "purchase_orders": "proc.purchase_order_agent",
        "purchase_order_lines": "proc.po_line_items_agent",
        "invoices": "proc.invoice_agent",
        "invoice_lines": "proc.invoice_line_items_agent",
        "product_mapping": "proc.cat_product_mapping",
        "quotes": "proc.quote_agent",
        "quote_lines": "proc.quote_line_items_agent",
    }

    relations, graph = manager.build_data_flow_map(tables, table_name_map=table_name_map)

    assert relations
    assert any(
        rel["status"] == "linked"
        for rel in relations
        if rel["source_table"] == "proc.contracts" and rel["target_table"] == "proc.supplier"
    )
    assert graph["nodes"]["proc.contracts"]["row_count"] == 2
    assert graph["paths"]
    for path in graph["paths"]:
        assert path["canonical"]

    flows = graph.get("supplier_flows")
    assert isinstance(flows, list) and flows
    first_flow = flows[0]
    assert first_flow.get("supplier_id")
    assert "coverage_ratio" in first_flow

    manager.persist_knowledge_graph(relations, graph)

    assert nick.qdrant_client.collections
    assert nick.qdrant_client.upserts
    assert all(entry["collection"] == "Knowledge Graph" for entry in nick.qdrant_client.upserts)
    payload_types = {
        point.payload.get("document_type")
        for entry in nick.qdrant_client.upserts
        for point in entry["points"]
    }
    assert "supplier_flow" in payload_types

    supplier_points = [
        point
        for entry in nick.qdrant_client.upserts
        for point in entry["points"]
        if point.payload.get("document_type") == "supplier_flow"
    ]
    assert supplier_points
    for point in supplier_points:
        summary = point.payload.get("summary")
        assert isinstance(summary, str) and summary.strip()
        mapping_summary = point.payload.get("mapping_summary")
        assert isinstance(mapping_summary, list) and mapping_summary
        assert any("proc." in statement for statement in mapping_summary)
