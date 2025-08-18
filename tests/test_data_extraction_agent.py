import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    monkeypatch.setattr(agent, "_vectorize_document", lambda text, pk, dt, pt, key: captured.setdefault("doc_type", dt))

    res = agent._process_single_document("doc.pdf")

    assert captured.get("doc_type") == "Contract"
    assert "persist" not in captured  # should not attempt DB insert
    assert res["id"] == "doc.pdf"
    assert res["doc_type"] == "Contract"
