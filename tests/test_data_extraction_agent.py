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
    monkeypatch.setattr(agent, "_generate_document_summary", lambda *a, **k: "summary")
    monkeypatch.setattr(agent, "_chunk_text", lambda text: [text])

    agent._vectorize_document({}, "hello world", "1", ["Invoice"], ["Hardware"], "doc.pdf")

    payload = captured["points"][0].payload
    assert payload["document_type"] == "invoice"
    assert payload["product_type"] == "hardware"
