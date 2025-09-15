import os
import sys
import uuid
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.rag_service import RAGService
from services.model_selector import RAGPipeline


class DummyEmbed:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, list):
            return [[0.1] for _ in texts]
        return [0.1]


class DummyQdrant:
    def __init__(self):
        self.upserts = []
        self.search_calls = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return []


class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.5 for _ in pairs]


def test_upsert_and_search():
    nick = SimpleNamespace(device="cpu", settings=SimpleNamespace(qdrant_collection_name="c", reranker_model="x"),
                           qdrant_client=DummyQdrant(), embedding_model=DummyEmbed())
    rag = RAGService(nick, cross_encoder_cls=DummyCrossEncoder)
    rag.upsert_texts(["hello world"], {"record_id": "test"})
    assert nick.qdrant_client.upserts  # ensure upsert called
    point_ids = [p.id for p in nick.qdrant_client.upserts[0]["points"]]
    for pid in point_ids:
        uuid.UUID(str(pid))
    hits = rag.search("hello")
    assert hits[0].payload["record_id"] == "test"
    # ensure local indexes were used without calling qdrant search
    assert not nick.qdrant_client.search_calls


def test_upsert_gpu_fallback(monkeypatch, caplog):
    monkeypatch.delattr("services.rag_service.faiss.StandardGpuResources", raising=False)
    nick = SimpleNamespace(
        device="cuda",
        settings=SimpleNamespace(qdrant_collection_name="c", reranker_model="x"),
        qdrant_client=DummyQdrant(),
        embedding_model=DummyEmbed(),
    )
    rag = RAGService(nick, cross_encoder_cls=DummyCrossEncoder)
    with caplog.at_level("WARNING"):
        rag.upsert_texts(["gpu fallback"], {"record_id": "gpu"})
    assert "falling back to CPU index" in caplog.text
    hits = rag.search("gpu")
    assert hits[0].payload["record_id"] == "gpu"


def test_pipeline_answer_returns_documents(monkeypatch):
    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None):
            return [SimpleNamespace(id="1", payload={"record_id": "R1", "summary": "s"})]

        def upsert_texts(self, texts, metadata=None):
            pass

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    nick = SimpleNamespace(
        device="cpu",
        s3_client=SimpleNamespace(get_object=lambda **_: {"Body": SimpleNamespace(read=lambda: b"[]")},
                                  put_object=lambda **_: None),
        settings=SimpleNamespace(qdrant_collection_name="c", s3_bucket_name="b", reranker_model="x"),
        embedding_model=DummyEmbed(),
        qdrant_client=SimpleNamespace(
            search=lambda **_: [SimpleNamespace(id="1", payload={"record_id": "R1", "summary": "s"}, score=1.0)]
        ),
        ollama_options=lambda: {},
    )
    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder)
    result = pipeline.answer_question("q", "user")
    assert result["retrieved_documents"][0]["record_id"] == "R1"
