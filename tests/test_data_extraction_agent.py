import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.data_extraction_agent import DataExtractionAgent


def test_vectorize_document_normalizes_labels(monkeypatch):
    """doc_type and product_type may be returned as lists; ensure they're handled."""

    captured = {}

    import numpy as np

    def fake_encode(chunks, **kwargs):
        # return a numpy vector per chunk to mimic real encoder output
        return [np.zeros(3) for _ in chunks]

    def fake_upsert(collection_name, points, wait):
        captured["points"] = points

    nick = SimpleNamespace(
        embedding_model=SimpleNamespace(encode=fake_encode),
        qdrant_client=SimpleNamespace(upsert=fake_upsert),
        _initialize_qdrant_collection=lambda: None,
        settings=SimpleNamespace(qdrant_collection_name="test", extraction_model="m"),
    )

    agent = DataExtractionAgent(nick)
    monkeypatch.setattr(agent, "_chunk_text", lambda text: [text])

    agent._vectorize_document("hello world", "1", ["Invoice"], ["Hardware"], "doc.pdf")

    payload = captured["points"][0].payload
    assert payload["document_type"] == "invoice"
    assert payload["product_type"] == "hardware"
    assert payload["record_id"] == "1"
    assert payload["content"] == "hello world"


def test_non_structured_docs_are_vectorized(monkeypatch):
    """Documents other than POs or invoices should only be vectorized."""

    captured = {}

    # Fake dependencies for the agent
    import numpy as np
    from io import BytesIO
    nick = SimpleNamespace(
        s3_client=SimpleNamespace(
            get_object=lambda Bucket, Key: {"Body": BytesIO(b"fake")}
        ),
        embedding_model=SimpleNamespace(encode=lambda chunks, **kwargs: [np.zeros(3) for _ in chunks]),
        qdrant_client=SimpleNamespace(upsert=lambda **kwargs: captured.setdefault("vectorized", True)),
        _initialize_qdrant_collection=lambda: None,
        settings=SimpleNamespace(
            s3_bucket_name="b",
            s3_prefixes=[],
            qdrant_collection_name="c",
            extraction_model="m",
        ),
    )

    agent = DataExtractionAgent(nick)

    monkeypatch.setattr(agent, "_extract_text", lambda b, k: "contract text")
    monkeypatch.setattr(agent, "_classify_doc_type", lambda t: "Contract")
    monkeypatch.setattr(agent, "_persist_to_postgres", lambda *args, **kwargs: captured.setdefault("persist", True))

    def fake_vectorize(text, pk, dt, pt, key):
        captured.setdefault("doc_types", []).append(dt)

    monkeypatch.setattr(agent, "_vectorize_document", fake_vectorize)

    res = agent._process_single_document("doc.pdf")

    assert captured.get("doc_types", [])[0] == "raw"
    assert "persist" not in captured  # should not attempt DB insert
    assert res["id"] == "doc.pdf"
    assert res["doc_type"] == "Contract"


def test_vectorize_structured_data_creates_points(monkeypatch):
    captured = {}

    import numpy as np

    def fake_encode(chunks, **kwargs):
        return [np.zeros(3) for _ in chunks]

    def fake_upsert(collection_name, points, wait):
        captured["points"] = points

    nick = SimpleNamespace(
        embedding_model=SimpleNamespace(encode=fake_encode),
        qdrant_client=SimpleNamespace(upsert=fake_upsert),
        _initialize_qdrant_collection=lambda: None,
        settings=SimpleNamespace(qdrant_collection_name="test", extraction_model="m"),
    )

    agent = DataExtractionAgent(nick)
    header = {"invoice_id": "1", "vendor_name": "acme"}
    line_items = [{"item_id": "A1", "description": "Widget"}]

    agent._vectorize_structured_data(header, line_items, "Invoice", "1")
    types = {p.payload["data_type"] for p in captured["points"]}
    assert types == {"header", "line_item"}
    # ensure points are associated with the same record id
    for p in captured["points"]:
        assert p.payload["record_id"] == "1"


def test_contextual_field_normalisation():
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    header = {"invoice_total": "100", "to": "Acme"}
    normalised_header = agent._normalize_header_fields(header, "Invoice")
    assert normalised_header["invoice_amount"] == "100"
    assert normalised_header["supplier_id"] == "Acme"

    items = [{"description": "Widget", "qty": "2", "price": "5"}]
    normalised_items = agent._normalize_line_item_fields(items, "Invoice")
    assert normalised_items[0]["item_description"] == "Widget"
    assert normalised_items[0]["quantity"] == "2"
    assert normalised_items[0]["unit_price"] == "5"


def test_po_line_items_unit_mapping(monkeypatch):
    executed = []

    class DummyCursor:
        def execute(self, sql, params=None):
            executed.append((sql, params))

        def fetchall(self):
            return [
                ("po_line_id",),
                ("po_id",),
                ("line_number",),
                ("unit_of_measue",),
                ("quantity",),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    nick = SimpleNamespace(
        get_db_connection=lambda: DummyConn(),
        settings=SimpleNamespace(extraction_model="m"),
    )
    agent = DataExtractionAgent(nick)

    item = {"unit_of_measure": "pcs", "quantity": "1"}
    agent._persist_line_items_to_postgres(
        "PO1", [item], "Purchase_Order", {}, None
    )

    insert_sql, params = executed[-1]
    assert "proc.po_line_items_agent" in insert_sql
    assert "unit_of_measue" in insert_sql
    assert "pcs" in params


def test_extract_header_with_ner(monkeypatch):
    """Ensure NER-based extraction supplements header parsing."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    sample_entities = [
        {"entity_group": "ORG", "word": "ACME Corp"},
        {"entity_group": "DATE", "word": "2024-01-01"},
        {"entity_group": "MONEY", "word": "$100"},
    ]

    monkeypatch.setattr(
        "agents.data_extraction_agent.extract_entities", lambda text: sample_entities
    )

    header = agent._extract_header_with_ner("Invoice from ACME Corp dated 2024-01-01 total $100")

    assert header["vendor_name"] == "ACME Corp"
    assert header["invoice_date"] == "2024-01-01"
    assert header["invoice_total_incl_tax"] == 100.0
