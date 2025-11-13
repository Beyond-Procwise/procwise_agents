import os
import sys
import json
import concurrent.futures
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pandas as pd
from pytest import approx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.data_extraction_agent import (
    DataExtractionAgent,
    DocumentTextBundle,
    PageExtractionResult,
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
        captured["collection"] = collection_name
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

    assert captured["collection"] == "test"
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
    monkeypatch.setattr(
        agent, "_classify_doc_type", lambda text, **kwargs: "Contract"
    )
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
    monkeypatch.setattr(
        agent, "_classify_doc_type", lambda text, **kwargs: "Invoice"
    )
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
        agent,
        "_extract_structured_data",
        lambda text, fb, dt, source_hint=None: structured,
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


def test_traceability_includes_validation_review_flag(monkeypatch):
    from io import BytesIO
    import numpy as np

    trace_calls: Dict[str, Any] = {}

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
            document_extraction_model="parser-v1",
        ),
    )

    agent = DataExtractionAgent(nick)

    bundle = DocumentTextBundle(
        full_text="invoice body",
        page_results=[
            PageExtractionResult(
                page_number=1,
                route="digital",
                digital_text="invoice body",
                ocr_text="",
                char_count=20,
            )
        ],
        raw_text="invoice body",
        ocr_text="",
        routing_log=[{"page": 1, "route": "digital"}],
    )
    monkeypatch.setattr(agent, "_extract_text", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        agent, "_classify_doc_type", lambda text, **kwargs: "Invoice"
    )
    monkeypatch.setattr(agent, "_classify_product_type", lambda t: "hardware")
    monkeypatch.setattr(agent, "_extract_unique_id", lambda t, dt: "INV-2")
    monkeypatch.setattr(agent, "_persist_to_postgres", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_vectorize_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_vectorize_structured_data", lambda *args, **kwargs: None)

    structured = StructuredExtractionResult(
        header={
            "invoice_id": "INV-2",
            "_validation": {"ok": False, "confidence": 0.5, "notes": ["missing buyer"]},
        },
        line_items=[{"item_description": "Service", "quantity": "1"}],
        header_df=pd.DataFrame([{"invoice_id": "INV-2"}]),
        line_df=pd.DataFrame([{"item_description": "Service"}]),
        report={
            "validation": {
                "is_valid": False,
                "confidence_score": 0.5,
                "errors": ["Missing buyer"],
                "warnings": ["No buyer field"],
            }
        },
    )
    monkeypatch.setattr(
        agent,
        "_extract_structured_data",
        lambda text, fb, dt, source_hint=None: structured,
    )

    stub_payload = {
        "header": {"invoice_id": "INV-2"},
        "line_items": [{"item_description": "Service", "quantity": "1"}],
        "tables": [],
        "raw_text": "invoice text",
        "metadata": {
            "ingestion_mode": "digital",
            "parser_version": "parser-v2",
            "page_count": 1,
        },
        "schema_reference": {},
    }

    monkeypatch.setattr(
        agent,
        "_trigger_document_extraction",
        lambda *args, **kwargs: (None, stub_payload),
    )
    monkeypatch.setattr(
        agent,
        "_record_etl_errors",
        lambda **kwargs: trace_calls.setdefault("payload", kwargs),
    )

    events: List[Dict[str, Any]] = []

    def capture_event(**kwargs):
        events.append(kwargs)

    monkeypatch.setattr(agent, "_log_workflow_event", capture_event)

    ctx = SimpleNamespace(workflow_id="wf-logs", agent_id="data_extraction")

    result = agent._process_single_document("docs/invoice.pdf", context=ctx)

    validation = result["data"]["validation"]
    traceability = result["data"]["traceability"]
    accuracy = result["data"].get("accuracy_report")

    assert validation["requires_review"] is True
    assert result["needs_review"] is True
    assert traceability["s3_key"] == "docs/invoice.pdf"
    assert traceability["parser_version"] == "parser-v2"
    assert traceability["confidence_score"] == approx(0.5)
    assert trace_calls["payload"]["validation"]["requires_review"] is True
    assert trace_calls["payload"]["trace_metadata"]["s3_key"] == "docs/invoice.pdf"
    assert isinstance(accuracy, list)
    event_names = [entry.get("event") for entry in events]
    assert "document_start" in event_names
    assert "document_complete" in event_names
    complete = next(entry for entry in events if entry.get("event") == "document_complete")
    assert complete.get("workflow_id") == "wf-logs"
    assert complete.get("record_id") == "INV-2"


def test_run_document_extraction_handles_extractor_initialisation_failure(monkeypatch):
    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)

    def boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(agent, "_get_document_extractor", boom)

    result = agent._run_document_extraction(
        "invoice.pdf",
        b"binary",
        document_type_hint=None,
        metadata={},
    )

    assert result is None


def test_process_documents_paginates_and_passes_context(monkeypatch):
    """S3 listing should walk pagination and filter supported keys."""

    class FakeClient:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []

        def list_objects_v2(self, Bucket, Prefix, **kwargs):
            self.calls.append({"Bucket": Bucket, "Prefix": Prefix, "token": kwargs.get("ContinuationToken")})
            if kwargs.get("ContinuationToken"):
                return {
                    "IsTruncated": False,
                    "Contents": [
                        {"Key": "docs/b.pdf"},
                        {"Key": "docs/skip.txt"},
                    ],
                }
            return {
                "IsTruncated": True,
                "NextContinuationToken": "token-1",
                "Contents": [{"Key": "docs/a.pdf"}],
            }

    fake_client = FakeClient()

    @contextmanager
    def fake_borrow():
        yield fake_client

    nick = SimpleNamespace(
        settings=SimpleNamespace(
            s3_bucket_name="bucket",
            s3_prefixes=["docs/"],
            data_extraction_max_workers=2,
            qdrant_collection_name="collection",
            extraction_model="model",
            document_extraction_model="parser",
            force_ocr_vendors=[],
        ),
        s3_pool_size=2,
    )

    agent = DataExtractionAgent(nick)

    monkeypatch.setattr(agent, "_borrow_s3_client", fake_borrow)

    processed: List[str] = []
    contexts: List[Any] = []

    def fake_single(key, *, context=None):
        processed.append(key)
        contexts.append(context)
        return {"object_key": key, "status": "success"}

    monkeypatch.setattr(agent, "_process_single_document", fake_single)

    class ImmediateExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            class ImmediateFuture:
                def result(self_inner):
                    return fn(*args, **kwargs)

            return ImmediateFuture()

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(concurrent.futures, "as_completed", lambda futures: futures)

    ctx = SimpleNamespace(workflow_id="wf-1", agent_id="data_extraction")

    result = agent._process_documents(context=ctx)

    assert processed == ["docs/a.pdf", "docs/b.pdf"]
    assert all(item is ctx for item in contexts)
    assert fake_client.calls[0]["token"] is None
    assert fake_client.calls[1]["token"] == "token-1"
    assert result["status"] == "completed"
    assert len(result["details"]) == 2


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
    class DummyCursor:
        def __init__(self):
            self.calls: List[Tuple[str, Any]] = []
            self.last_sql = ""
            self.params: Any = None

        def execute(self, sql, params=None):
            text = str(sql)
            self.calls.append((text, params))
            self.last_sql = text
            self.params = params

        def fetchall(self):
            if "information_schema.columns" in self.last_sql:
                return [
                    ("po_line_id", "text"),
                    ("po_id", "text"),
                    ("line_number", "integer"),
                    ("unit_of_measue", "text"),
                    ("quantity", "integer"),
                ]
            if "table_constraints" in self.last_sql:
                return []
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class DummyConn:
        def __init__(self):
            self.cursors: List[DummyCursor] = []

        def cursor(self):
            cur = DummyCursor()
            self.cursors.append(cur)
            return cur

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

    conn = DummyConn()
    nick = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(extraction_model="m"),
    )
    agent = DataExtractionAgent(nick)

    item = {"unit_of_measure": "pcs", "quantity": "1"}
    agent._persist_line_items_to_postgres(
        "PO1", [item], "Purchase_Order", {}, None
    )

    all_calls = [call for cursor in conn.cursors for call in cursor.calls]

    stage_insert = next(
        (
            params
            for sql, params in all_calls
            if sql.startswith('INSERT INTO "proc_stage"."po_line_items_agent_staging"')
        ),
        None,
    )
    final_insert = next(
        (
            sql
            for sql, params in all_calls
            if sql.startswith('INSERT INTO "proc"."po_line_items_agent"')
        ),
        "",
    )

    assert stage_insert is not None
    assert any(str(value) == "pcs" for value in stage_insert)
    assert "unit_of_measue" in final_insert


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
        agent,
        "_process_documents",
        lambda p, k, **kwargs: {"status": "completed", "details": docs},
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
        agent,
        "_process_documents",
        lambda p, k, **kwargs: {"status": "completed", "details": docs},
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

    captured_prompts = []

    def fake_call(prompt, model, format=None):
        captured_prompts.append(prompt)
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
    assert any("header_data" in prompt.lower() for prompt in captured_prompts)


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


def test_classify_doc_type_prefers_llamaparse_metadata():
    """Metadata produced by LlamaParse should take precedence over heuristics."""

    nick = SimpleNamespace(settings=SimpleNamespace(extraction_model="m"))
    agent = DataExtractionAgent(nick)
    bundle = DocumentTextBundle(full_text="", page_results=[], raw_text="", ocr_text="")
    bundle.llamaparse_metadata = {"document_type": "Tax Invoice"}

    assert agent._classify_doc_type("ambiguous", text_bundle=bundle) == "Invoice"


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


def test_low_confidence_guard_blanks_critical_fields(monkeypatch):
    """Critical header values are cleared when the document confidence is low."""

    settings = SimpleNamespace(
        extraction_model="m",
        structured_low_confidence_threshold=0.6,
        field_low_confidence_threshold=0.55,
    )
    agent = DataExtractionAgent(SimpleNamespace(settings=settings))

    base_header = {
        "invoice_id": "INV-9999",
        "vendor_name": "Fuzzy Corp",
        "invoice_total_incl_tax": "250.00",
        "_field_confidence": {
            "invoice_id": 0.3,
            "vendor_name": 0.2,
            "invoice_total_incl_tax": 0.4,
        },
    }

    monkeypatch.setattr(
        agent,
        "_parse_header_improved",
        lambda text, file_bytes=None, source_hint=None: dict(base_header),
    )
    monkeypatch.setattr(
        agent,
        "_extract_line_items_from_layout",
        lambda file_bytes, text, doc_type: [],
    )
    monkeypatch.setattr(
        agent,
        "_extract_line_items_improved",
        lambda text, doc_type: [],
    )
    monkeypatch.setattr(
        agent,
        "_normalize_header_fields",
        lambda header, doc_type: header,
    )
    monkeypatch.setattr(
        agent,
        "_normalize_line_item_fields",
        lambda items, doc_type: items,
    )
    monkeypatch.setattr(
        agent,
        "_reconcile_header_from_lines",
        lambda header, items, doc_type: header,
    )
    monkeypatch.setattr(agent, "_infer_currency", lambda text, header: None)
    monkeypatch.setattr(agent, "_sanitize_party_names", lambda header: header)
    monkeypatch.setattr(
        agent,
        "_recover_missing_critical_fields",
        lambda *args, **kwargs: {},
    )

    def fake_validate(header, line_items, doc_type, text):
        return {
            "is_valid": False,
            "confidence_score": 0.4,
            "errors": [],
            "warnings": [],
            "field_scores": {},
        }

    monkeypatch.setattr(agent, "_validate_extraction_quality", fake_validate)

    captured_frames: Dict[str, Any] = {}

    def fake_build(data, doc_type, segment):
        captured_frames[segment] = (
            dict(data)
            if isinstance(data, dict)
            else [dict(item) for item in data]
        )
        return pd.DataFrame(), [], []

    monkeypatch.setattr(agent, "_build_dataframe_from_records", fake_build)
    monkeypatch.setattr(
        agent,
        "_dataframe_to_header",
        lambda df: captured_frames.get("header", {}),
    )
    monkeypatch.setattr(
        agent,
        "_dataframe_to_records",
        lambda df: captured_frames.get("line_items", []),
    )
    monkeypatch.setattr(
        agent,
        "_validate_and_cast",
        lambda header, items, doc_type: (header, items),
    )

    result = agent._extract_structured_data("sample text", b"", "Invoice")

    validation_warnings = result.report["validation"]["warnings"]
    assert result.header["invoice_id"] is None
    assert result.header["vendor_name"] is None
    assert result.header["invoice_total_incl_tax"] is None
    assert result.header.get("_field_confidence", {}).get("invoice_id") == 0.0
    assert any("invoice_id" in warning for warning in validation_warnings)


def test_recover_missing_critical_fields_uses_document_context(monkeypatch):
    settings = SimpleNamespace(
        extraction_model="m",
        field_recovery_confidence_threshold=0.7,
    )
    agent = DataExtractionAgent(SimpleNamespace(settings=settings))

    document_text = (
        "Invoice Number: INV-321\n"
        "Vendor: Beyond Supplies Ltd.\n"
        "Invoice Total: USD 2500.00"
    )

    header = {
        "invoice_id": None,
        "vendor_name": "",
        "invoice_total_incl_tax": None,
    }

    captured_prompt: Dict[str, Any] = {}

    def fake_call(prompt, model, format):
        captured_prompt["prompt"] = prompt
        payload = {
            "invoice_id": {
                "value": "INV-321",
                "confidence": 0.91,
                "context": "Invoice Number: INV-321",
            },
            "vendor_name": {
                "value": "Beyond Supplies Ltd.",
                "confidence": 0.88,
                "context": "Vendor: Beyond Supplies Ltd.",
            },
            "invoice_total_incl_tax": {
                "value": "2500.00",
                "confidence": 0.86,
                "context": "Invoice Total: USD 2500.00",
            },
        }
        return {"response": json.dumps(payload)}

    monkeypatch.setattr(agent, "call_ollama", fake_call)

    validation_report = {"warnings": []}
    recovered = agent._recover_missing_critical_fields(
        document_text, "Invoice", header, validation_report
    )

    assert set(recovered) == {
        "invoice_id",
        "vendor_name",
        "invoice_total_incl_tax",
    }
    assert recovered["invoice_id"]["value"] == "INV-321"
    assert recovered["vendor_name"]["context"].startswith("Vendor: Beyond")
    assert "Invoice Number" in captured_prompt["prompt"]


def test_structured_extraction_injects_recovered_fields(monkeypatch):
    settings = SimpleNamespace(
        extraction_model="m",
        structured_low_confidence_threshold=0.8,
        field_low_confidence_threshold=0.7,
    )
    agent = DataExtractionAgent(SimpleNamespace(settings=settings))

    base_header = {
        "invoice_id": None,
        "vendor_name": None,
        "invoice_total_incl_tax": None,
        "_field_confidence": {},
        "_field_context": {},
    }

    monkeypatch.setattr(
        agent,
        "_parse_header_improved",
        lambda text, file_bytes=None, source_hint=None: dict(base_header),
    )
    monkeypatch.setattr(
        agent,
        "_extract_line_items_from_layout",
        lambda file_bytes, text, doc_type: [],
    )
    monkeypatch.setattr(
        agent,
        "_extract_line_items_improved",
        lambda text, doc_type: [],
    )
    monkeypatch.setattr(
        agent,
        "_normalize_header_fields",
        lambda header, doc_type: header,
    )
    monkeypatch.setattr(
        agent,
        "_normalize_line_item_fields",
        lambda items, doc_type: items,
    )
    monkeypatch.setattr(
        agent,
        "_reconcile_header_from_lines",
        lambda header, items, doc_type: header,
    )
    monkeypatch.setattr(agent, "_infer_currency", lambda text, header: None)
    monkeypatch.setattr(agent, "_sanitize_party_names", lambda header: header)

    monkeypatch.setattr(
        agent,
        "_validate_extraction_quality",
        lambda header, items, doc_type, text: {
            "is_valid": True,
            "confidence_score": 0.9,
            "errors": [],
            "warnings": [],
            "field_scores": {},
        },
    )

    monkeypatch.setattr(
        agent,
        "_recover_missing_critical_fields",
        lambda text, doc_type, header, report: {
            "invoice_id": {
                "value": "INV-2024-001",
                "confidence": 0.93,
                "context": "Invoice Number: INV-2024-001",
            },
            "vendor_name": {
                "value": "Beyond Components",
                "confidence": 0.91,
                "context": "Vendor: Beyond Components",
            },
        },
    )

    captured: Dict[str, Any] = {}

    def fake_build(data, doc_type, segment):
        captured[segment] = data
        if segment == "header":
            return pd.DataFrame([data]), [], []
        return pd.DataFrame(), [], []

    monkeypatch.setattr(agent, "_build_dataframe_from_records", fake_build)
    monkeypatch.setattr(agent, "_dataframe_to_header", lambda df: df.iloc[0].to_dict())
    monkeypatch.setattr(agent, "_dataframe_to_records", lambda df: [])
    monkeypatch.setattr(agent, "_validate_and_cast", lambda header, items, doc_type: (header, items))

    result = agent._extract_structured_data(
        "Invoice Number: INV-2024-001\nVendor: Beyond Components", b"", "Invoice"
    )

    assert result.header["invoice_id"] == "INV-2024-001"
    assert result.header.get("_field_confidence", {}).get("vendor_name") == approx(0.91)
    assert "field_contexts" in result.report
    warnings = " ".join(result.report["validation"].get("warnings", []))
    assert "Recovered critical fields via document-wide context" in warnings


def test_persist_to_postgres_sanitizes_values(monkeypatch):
    """Insertion into target tables should strip stray symbols from values."""

    class DummyCursor:
        def __init__(self):
            self.calls: List[Tuple[str, Any]] = []
            self.last_sql = ""
            self.params: Any = None

        def execute(self, sql, params=None):
            text = str(sql)
            self.calls.append((text, params))
            self.last_sql = text
            self.params = params

        def fetchall(self):
            if "information_schema.columns" in self.last_sql:
                table = self.params[1] if self.params and len(self.params) > 1 else ""
                if table == "invoice_agent":
                    return [("invoice_id", "text"), ("supplier_id", "text")]
                if table == "invoice_line_items_agent":
                    return [
                        ("invoice_line_id", "text"),
                        ("invoice_id", "text"),
                        ("line_no", "integer"),
                        ("item_description", "text"),
                        ("quantity", "integer"),
                    ]
            if "table_constraints" in self.last_sql:
                return []
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class InvoiceConn:
        def __init__(self):
            self.cursors: List[DummyCursor] = []

        def cursor(self):
            cur = DummyCursor()
            self.cursors.append(cur)
            return cur

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

    conn = InvoiceConn()
    nick = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(extraction_model="m"),
    )

    agent = DataExtractionAgent(nick)

    header = {"invoice_id": "INV{123}", "supplier_id": "ACME|Co"}
    line_items = [{"item_description": "Widget?", "quantity": "2"}]

    agent._persist_to_postgres(header, line_items, "Invoice", "INV{123}")

    all_calls = [call for cursor in conn.cursors for call in cursor.calls]

    header_stage = next(
        (
            params
            for sql, params in all_calls
            if sql.startswith('INSERT INTO "proc_stage"."invoice_agent_staging"')
        ),
        None,
    )
    line_stage = next(
        (
            params
            for sql, params in all_calls
            if sql.startswith('INSERT INTO "proc_stage"."invoice_line_items_agent_staging"')
        ),
        None,
    )
    header_insert = next(
        (
            (sql, params)
            for sql, params in all_calls
            if sql.startswith('INSERT INTO "proc"."invoice_agent"')
        ),
        None,
    )
    line_insert = next(
        (
            (sql, params)
            for sql, params in all_calls
            if sql.startswith('INSERT INTO "proc"."invoice_line_items_agent"')
        ),
        None,
    )

    assert header_stage is not None
    assert any(str(value) == "INV123" for value in header_stage)
    assert any(str(value) == "ACMECo" for value in header_stage)
    assert line_stage is not None
    assert any(str(value) == "INV123-1" for value in line_stage)
    assert any(str(value) == "INV123" for value in line_stage)
    assert any(str(value).startswith("Widget") for value in line_stage)
    assert any(str(value) in {"2", "2.0"} for value in line_stage)
    assert header_insert is not None
    assert "SELECT \"invoice_id\"" in header_insert[0]
    assert len(header_insert[1]) == 1  # ingestion id placeholder
    assert line_insert is not None
    assert "SELECT \"invoice_id\", \"line_no\"" in line_insert[0]
    assert len(line_insert[1]) == 1


def test_record_etl_errors_inserts_rows(monkeypatch):
    executed: List[Any] = []

    class DummyCursor:
        def __init__(self):
            self.last_sql = ""

        def execute(self, sql, params=None):
            self.last_sql = sql
            executed.append((sql, params))

        def fetchall(self):
            if "information_schema.columns" in self.last_sql:
                return [
                    ("record_id",),
                    ("document_type",),
                    ("source_object_key",),
                    ("error_category",),
                    ("error_detail",),
                    ("confidence_score",),
                    ("source_table",),
                ]
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="m"),
        get_db_connection=lambda: DummyConn(),
    )
    agent = DataExtractionAgent(nick)

    agent._record_etl_errors(
        doc_type="Invoice",
        record_id="INV-9",
        object_key="docs/inv.pdf",
        validation={
            "is_valid": False,
            "confidence_score": 0.6,
            "errors": ["missing"],
            "warnings": [],
        },
        table_name="proc.invoice_agent",
        trace_metadata={"s3_key": "docs/inv.pdf"},
    )

    insert_sql, params = executed[-1]
    assert "INSERT INTO proc.etl_errors" in insert_sql
    assert "ON CONFLICT" in insert_sql
    assert params[0] == "INV-9"
    detail_payload = json.loads(params[4])
    assert detail_payload["errors"] == ["missing"]
    assert params[3] == "validation_error"


def test_record_etl_errors_skips_when_confident():
    def boom():
        raise AssertionError("should not connect")

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="m"),
        get_db_connection=boom,
    )
    agent = DataExtractionAgent(nick)

    agent._record_etl_errors(
        doc_type="Invoice",
        record_id="INV-10",
        object_key="docs/inv.pdf",
        validation={
            "is_valid": True,
            "confidence_score": 0.95,
            "errors": [],
            "warnings": [],
        },
        table_name="proc.invoice_agent",
        trace_metadata={},
    )
