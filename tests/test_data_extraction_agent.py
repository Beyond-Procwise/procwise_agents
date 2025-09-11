import os
import sys
import json
from types import SimpleNamespace

from agents.base_agent import AgentContext, AgentOutput, AgentStatus

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
    monkeypatch.setattr(agent, "_classify_product_type", lambda t: "it hardware")
    monkeypatch.setattr(agent, "_extract_unique_id", lambda t, dt: "C123")
    monkeypatch.setattr(agent, "_persist_to_postgres", lambda *args, **kwargs: captured.setdefault("persist", True))

    def fake_vectorize(text, pk, dt, pt, key):
        captured["doc_type"] = dt
        captured["product_type"] = pt
        captured["pk"] = pk

    monkeypatch.setattr(agent, "_vectorize_document", fake_vectorize)

    res = agent._process_single_document("doc.pdf")

    assert captured["doc_type"] == "Contract"
    assert captured["product_type"] == "it hardware"
    assert captured["pk"] == "C123"
    assert "persist" not in captured  # should not attempt DB insert
    assert res["id"] == "C123"
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

    agent._vectorize_structured_data(header, line_items, "Invoice", "1", "Hardware")
    types = {p.payload["data_type"] for p in captured["points"]}
    assert types == {"header", "line_item"}
    # ensure points are associated with the same record id
    for p in captured["points"]:
        assert p.payload["record_id"] == "1"
        assert p.payload["product_type"] == "hardware"


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


def test_extract_unique_id():
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)
    text = "Invoice Number: INV-42"
    assert agent._extract_unique_id(text, "Invoice") == "INV-42"


def test_classify_product_type():
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)
    text = "Procurement of laptops and printers"
    assert agent._classify_product_type(text) == "it hardware"


def test_run_summarises_discrepancies(monkeypatch):
    docs = [
        {"id": "1", "doc_type": "Invoice"},
        {"id": "2", "doc_type": "Invoice"},
    ]

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    monkeypatch.setattr(
        agent, "_process_documents", lambda p, k: {"status": "completed", "details": docs}
    )

    def fake_run_disc(self, docs, ctx):
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "mismatches": [
                    {"doc_type": "Invoice", "id": "2", "checks": {"vendor_name": "missing"}}
                ]
            },
        )

    monkeypatch.setattr(DataExtractionAgent, "_run_discrepancy_detection", fake_run_disc)

    ctx = AgentContext(
        workflow_id="w1",
        agent_id="data_extraction",
        user_id="u1",
        input_data={},
    )
    output = agent.run(ctx)
    summary = output.data["summary"]
    assert summary["documents_provided"] == 2
    assert summary["documents_valid"] == 1
    assert summary["documents_with_discrepancies"] == 1


def test_run_propagates_discrepancy_fail(monkeypatch):
    docs = [{"id": "1", "doc_type": "Invoice"}]
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    monkeypatch.setattr(
        agent, "_process_documents", lambda p, k: {"status": "completed", "details": docs}
    )

    def fake_run_disc(self, docs, ctx):
        return AgentOutput(status=AgentStatus.FAILED, data={}, error="db down")

    monkeypatch.setattr(DataExtractionAgent, "_run_discrepancy_detection", fake_run_disc)

    ctx = AgentContext(
        workflow_id="w1",
        agent_id="data_extraction",
        user_id="u1",
        input_data={},
    )
    output = agent.run(ctx)
    assert output.status is AgentStatus.FAILED
    assert output.error == "db down"


def test_fill_missing_fields_with_llm(monkeypatch):
    """LLM call fills only missing fields after heuristic parsing."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    # Heuristic extractors return nothing
    monkeypatch.setattr(
        "agents.data_extraction_agent.convert_document_to_json",
        lambda text, dt: {"header_data": {}, "line_items": []},
    )
    monkeypatch.setattr(agent, "_parse_header", lambda text: {})
    monkeypatch.setattr(
        agent, "_extract_line_items_from_pdf_tables", lambda b, dt: []
    )

    llm_payload = {
        "response": json.dumps(
            {
                "header_data": {"invoice_id": "INV1", "vendor_name": "ACME"},
                "line_items": [
                    {
                        "item_id": "A1",
                        "item_description": "Widget",
                        "quantity": "1",
                    }
                ],
            }
        )
    }
    monkeypatch.setattr(agent, "call_ollama", lambda **kwargs: llm_payload)

    header, items = agent._extract_structured_data(
        "Invoice INV1 from ACME for 1 Widget", b"", "Invoice"
    )
    assert header["invoice_id"] == "INV1"
    assert header["vendor_name"] == "ACME"
    assert items and items[0]["item_id"] == "A1"


def test_classify_doc_type_keyword_scoring(monkeypatch):
    """Keyword scoring should classify common documents without LLM calls."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    def fail_call(**kwargs):
        raise AssertionError("LLM should not be used for obvious keywords")

    monkeypatch.setattr(agent, "call_ollama", fail_call)

    assert (
        agent._classify_doc_type("Invoice number INV-1 for services rendered")
        == "Invoice"
    )
    assert (
        agent._classify_doc_type("PO number PO-2 to purchase goods")
        == "Purchase_Order"
    )


def test_classify_doc_type_llm_fallback(monkeypatch):
    """When no keywords are present an LLM is queried for classification."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    monkeypatch.setattr(
        agent, "call_ollama", lambda **kwargs: {"response": "Contract"}
    )

    # Text deliberately avoids any predefined keywords so the LLM is used.
    assert agent._classify_doc_type("Memorandum of understanding") == "Contract"


def test_classification_prompt_includes_context(monkeypatch):
    """LLM classification prompt should include procurement context."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    captured = {}

    def fake_call(prompt, model):
        captured["prompt"] = prompt
        return {"response": "Invoice"}

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    agent._classify_doc_type("irrelevant text")
    assert "buyer sends a purchase order" in captured["prompt"].lower()


def test_fill_missing_fields_prompt_includes_context(monkeypatch):
    """LLM field completion should receive doc-type specific context."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    captured = {}

    def fake_call(prompt, model, format=None):
        captured["prompt"] = prompt
        return {"response": json.dumps({"header_data": {}, "line_items": []})}

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    agent._fill_missing_fields_with_llm("text", "Invoice", {}, [])
    assert "vendor sends an invoice" in captured["prompt"].lower()
