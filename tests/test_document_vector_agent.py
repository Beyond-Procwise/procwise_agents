from types import SimpleNamespace
from io import BytesIO

from agents.document_vector_agent import DocumentVectorAgent


def test_document_vector_agent_vectorizes(monkeypatch):
    captured = {}
    import numpy as np

    nick = SimpleNamespace(
        s3_client=SimpleNamespace(get_object=lambda Bucket, Key: {"Body": BytesIO(b"fake")}),
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

    agent = DocumentVectorAgent(nick)
    monkeypatch.setattr(agent, "_extract_text", lambda b, k: "contract text")
    monkeypatch.setattr(agent, "_classify_doc_type", lambda t: "Contract")

    res = agent._vectorize_single_document("doc.pdf")

    assert captured.get("vectorized")
    assert res["id"] == "doc.pdf"
    assert res["doc_type"] == "Contract"
