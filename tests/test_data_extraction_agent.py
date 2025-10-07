import os
import sys
import json
from types import SimpleNamespace

import pandas as pd
from pytest import approx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.data_extraction_agent import (
    DataExtractionAgent,
    DocumentTextBundle,
    StructuredExtractionResult,
)


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


def test_validate_and_cast_truncates_varchar_fields():
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    long_terms = "Payment due 45 days post receipt of compliant invoice"
    header, _ = agent._validate_and_cast(
        {"payment_terms": long_terms},
        [],
        "Purchase_Order",
    )

    assert len(header["payment_terms"]) == 30
    assert header["payment_terms"] == long_terms[:30]


def test_contract_docs_are_persisted_and_vectorized(monkeypatch):
    """Contracts should be persisted and vectorized even without line items."""

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

    monkeypatch.setattr(
        agent,
        "_extract_text",
        lambda b, k, force_ocr=False: DocumentTextBundle(
            full_text="contract text",
            page_results=[],
            raw_text="contract text",
            ocr_text="",
        ),
    )
    monkeypatch.setattr(agent, "_classify_doc_type", lambda t: "Contract")
    monkeypatch.setattr(agent, "_classify_product_type", lambda t: "it hardware")
    monkeypatch.setattr(agent, "_extract_unique_id", lambda t, dt: "C123")
    monkeypatch.setattr(agent, "_persist_to_postgres", lambda *args, **kwargs: captured.setdefault("persist", True))
    monkeypatch.setattr(agent, "_trigger_document_extraction", lambda *args, **kwargs: (None, None))

    def fake_vectorize(text, pk, dt, pt, key):
        captured["doc_type"] = dt
        captured["product_type"] = pt
        captured["pk"] = pk

    monkeypatch.setattr(agent, "_vectorize_document", fake_vectorize)

    res = agent._process_single_document("doc.pdf")

    assert captured["doc_type"] == "Contract"
    assert captured["product_type"] == "it hardware"
    assert captured["pk"] == "C123"
    assert "persist" in captured  # should attempt DB insert
    assert res["id"] == "C123"
    assert res["doc_type"] == "Contract"


def test_raw_payload_merges_into_persistence(monkeypatch):
    from io import BytesIO
    import numpy as np

    captured = {}
    flags = {}

    nick = SimpleNamespace(
        s3_client=SimpleNamespace(
            get_object=lambda Bucket, Key: {"Body": BytesIO(b"fake")}
        ),
        embedding_model=SimpleNamespace(
            encode=lambda chunks, **kwargs: [np.zeros(3) for _ in chunks]
        ),
        qdrant_client=SimpleNamespace(upsert=lambda **kwargs: None),
        _initialize_qdrant_collection=lambda: None,
        settings=SimpleNamespace(
            s3_bucket_name="bucket",
            s3_prefixes=[],
            qdrant_collection_name="collection",
            extraction_model="model",
        ),
    )

    agent = DataExtractionAgent(nick)

    monkeypatch.setattr(
        agent,
        "_extract_text",
        lambda b, k, force_ocr=False: DocumentTextBundle(
            full_text="invoice text",
            page_results=[],
            raw_text="invoice raw",
            ocr_text="",
        ),
    )
    monkeypatch.setattr(agent, "_classify_doc_type", lambda t: "Invoice")
    monkeypatch.setattr(agent, "_classify_product_type", lambda t: "hardware")
    monkeypatch.setattr(agent, "_extract_unique_id", lambda t, dt: "")
    monkeypatch.setattr(agent, "_vectorize_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_vectorize_structured_data", lambda *args, **kwargs: None)

    structured = StructuredExtractionResult(
        header={"invoice_id": "INV-1", "payment_terms": "Net 30"},
        line_items=[{"item_description": "Laptop", "quantity": "2"}],
        header_df=pd.DataFrame([{"invoice_id": "INV-1"}]),
        line_df=pd.DataFrame([{"item_description": "Laptop"}]),
        report={"table_method": "none"},
    )
    monkeypatch.setattr(
        agent, "_extract_structured_data", lambda text, fb, dt: structured
    )

    stub_result = SimpleNamespace(
        document_type="Invoice",
        document_id="INV-1",
        header={"invoice_id": "INV-1"},
        line_items=[{"item_description": "Laptop", "quantity": "2"}],
        tables=[],
        raw_text="invoice text from extractor",
        metadata={"ingestion_mode": "digital"},
        schema_reference={"document_type": "Invoice"},
    )

    def fake_run(object_key, file_bytes, document_type_hint=None, metadata=None):
        flags["extract_called"] = True
        return stub_result

    monkeypatch.setattr(agent, "_run_document_extraction", fake_run)

    stub_raw_payload = {
        "header": {"invoice_id": "INV-1", "supplier_name": "ACME Components"},
        "line_items": [
            {
                "item_description": "Laptop",
                "quantity": "2",
                "unit_price": "700",
                "line_amount": "1400",
            }
        ],
        "tables": [],
        "raw_text": "invoice text from extractor",
        "metadata": {"ingestion_mode": "digital"},
        "schema_reference": {"document_type": "Invoice"},
    }

    def fake_fetch(doc_id, doc_type):
        flags["raw_fetch"] = (doc_id, doc_type)
        return stub_raw_payload

    monkeypatch.setattr(agent, "_fetch_raw_document_payload", fake_fetch)

    def fake_persist(header, line_items, doc_type, pk_value):
        captured["header"] = header
        captured["line_items"] = line_items
        captured["doc_type"] = doc_type
        captured["pk"] = pk_value

    monkeypatch.setattr(agent, "_persist_to_postgres", fake_persist)

    res = agent._process_single_document("invoice.pdf")

    assert flags.get("extract_called") is True
    assert flags.get("raw_fetch") == ("INV-1", "Invoice")
    assert captured["header"]["supplier_name"] == "ACME Components"
    assert captured["doc_type"] == "Invoice"
    assert captured["pk"] == "INV-1"
    assert captured["line_items"][0]["line_amount"] == "1400"
    assert res["data"]["raw_extraction"]["header"]["supplier_name"] == "ACME Components"
    assert res["data"]["header_data"]["invoice_id"] == "INV-1"


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


def test_invoice_row_numeric_repair():
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    row = {"item_description": "Widget", "quantity": "2"}
    repaired = agent._repair_invoice_line_values(row, "Widget Service 2 15.00 30.00", "Invoice")
    assert repaired["line_amount"] == approx(30.0)
    assert repaired["unit_price"] == approx(15.0)
    assert repaired["total_amount_incl_tax"] == approx(30.0)

    row_with_tax = {"item_description": "Widget", "quantity": "2"}
    repaired_tax = agent._repair_invoice_line_values(
        row_with_tax,
        "Widget Service 2 15.00 30.00 6.00 36.00",
        "Invoice",
    )
    assert repaired_tax["tax_amount"] == approx(6.0)
    assert repaired_tax["total_amount_incl_tax"] == approx(36.0)

    row_implicit_tax = {"item_description": "Widget", "quantity": "2"}
    repaired_implicit_tax = agent._repair_invoice_line_values(
        row_implicit_tax,
        "Widget Service 2 15.00 30.00 36.00",
        "Invoice",
    )
    assert repaired_implicit_tax["tax_amount"] == approx(6.0)
    assert repaired_implicit_tax["unit_price"] == approx(15.0)


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

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
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


def test_llm_structured_pass_populates_header(monkeypatch):
    """Initial LLM pass should provide structured values and context."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    # Heuristic extractors return nothing to ensure LLM data is used
    monkeypatch.setattr(agent, "_extract_header_regex", lambda text, dt: {})
    monkeypatch.setattr(agent, "_extract_header_with_ner", lambda text: {})
    monkeypatch.setattr(agent, "_parse_header", lambda text, file_bytes=None: {})
    monkeypatch.setattr(
        agent, "_extract_line_items_from_pdf_tables", lambda b, dt: ([], None, [])
    )
    monkeypatch.setattr(agent, "_extract_line_items_regex", lambda text, dt: [])

    llm_payload = {
        "response": json.dumps(
            {
                "header_data": {
                    "invoice_id": "INV1",
                    "vendor_name": "ACME",
                },
                "line_items": [
                    {
                        "item_id": "A1",
                        "item_description": "Widget",
                        "quantity": "1",
                    }
                ],
                "field_contexts": {
                    "invoice_id": "Invoice number INV1",
                    "vendor_name": "Supplier ACME Pty",
                },
            }
        )
    }

    captured_prompt = {}

    def fake_call(prompt, model, format=None):
        captured_prompt["prompt"] = prompt
        return llm_payload

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    result = agent._extract_structured_data(
        "Invoice INV1 from ACME for 1 Widget", b"", "Invoice"
    )

    header = result.header
    items = result.line_items
    assert header["invoice_id"] == "INV1"
    assert header["vendor_name"] == "ACME"
    assert "invoice_id" in header.get("_field_context", {})
    assert header.get("_field_confidence", {}).get("invoice_id", 0) >= 0.7
    assert items and items[0]["item_id"] == "A1"
    assert result.report["table_method"] == "none"
    assert "header_data" in captured_prompt["prompt"].lower()


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


def test_llm_structured_prompt_includes_context(monkeypatch):
    """Structured extraction prompt should include procurement guidance."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    captured = {}

    def fake_call(prompt, model, format=None):
        captured["prompt"] = prompt
        return {
            "response": json.dumps(
                {"header_data": {}, "line_items": [], "field_contexts": {}}
            )
        }

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    agent._llm_structured_extraction("text", "Invoice")
    prompt_lower = captured["prompt"].lower()
    assert "vendor sends an invoice" in prompt_lower
    assert "field_contexts" in prompt_lower


def test_persist_to_postgres_sanitizes_values(monkeypatch):
    """Insertion into target tables should strip stray symbols from values."""

    executed = {"header": None, "line": None}

    class DummyCursor:
        def __init__(self):
            self.sql = ""
            self.params = None

        def execute(self, sql, params=None):
            self.sql = sql.lower().strip()
            self.params = params
            if self.sql.startswith("insert into proc.invoice_agent"):
                executed["header"] = params
            elif self.sql.startswith("insert into proc.invoice_line_items_agent"):
                executed["line"] = params

        def fetchall(self):
            if "information_schema.columns" in self.sql:
                table = self.params[1] if self.params and len(self.params) > 1 else ""
                if table == "invoice_agent":
                    return [("invoice_id", "text"), ("supplier_id", "text")]
                if table == "invoice_line_items_agent":
                    return [
                        ("invoice_line_id",),
                        ("invoice_id",),
                        ("line_no",),
                        ("item_description",),
                        ("quantity",),
                    ]
            if "table_constraints" in self.sql:
                return []
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class InvoiceConn:
        def cursor(self):
            return DummyCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    nick = SimpleNamespace(
        get_db_connection=lambda: InvoiceConn(),
        settings=SimpleNamespace(extraction_model="m"),
    )

    agent = DataExtractionAgent(nick)

    header = {"invoice_id": "INV{123}", "supplier_id": "ACME|Co"}
    line_items = [{"item_description": "Widget?", "quantity": "2"}]

    agent._persist_to_postgres(header, line_items, "Invoice", "INV{123}")

    assert executed["header"] == ["INV123", "ACMECo"]
    assert executed["line"] == ["INV123", 1, "INV123-1", "Widget", 2]

