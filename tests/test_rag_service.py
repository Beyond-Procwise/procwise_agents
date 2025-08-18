import os
import sys
from types import SimpleNamespace

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.rag_service import RAGService
from services.model_selector import RAGPipeline


class DummyEmbed:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, list):
            return [SimpleNamespace(tolist=lambda: [0.1])] * len(texts)
        return SimpleNamespace(tolist=lambda: [0.1])


class DummyQdrant:
    def __init__(self):
        self.upserts = []
        self.search_calls = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        # Return empty list first to trigger exact search fallback
        if len(self.search_calls) == 1:
            return []
        hit = SimpleNamespace(id="1", payload={"record_id": "R1", "content": "doc"}, score=1.0)
        return [hit]


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
    hits = rag.search("query")
    assert hits[0].payload["record_id"] == "R1"
    # ensure fallback search executed
    assert nick.qdrant_client.search_calls[0]["search_params"].exact is False
    assert nick.qdrant_client.search_calls[1]["search_params"].exact is True


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
