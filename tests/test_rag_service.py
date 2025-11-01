import os
import sys
import uuid
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentOutput, AgentStatus

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
    assert nick.qdrant_client.upserts[0]["collection_name"] == "c"
    point_ids = [p.id for p in nick.qdrant_client.upserts[0]["points"]]
    for pid in point_ids:
        uuid.UUID(str(pid))
    hits = rag.search("hello")
    assert hits[0].payload["record_id"] == "test"
    # ensure qdrant was queried across configured collections even with local hits
    collections = {call["collection_name"] for call in nick.qdrant_client.search_calls}
    assert {"c", "uploaded_documents", "static_policy"}.issubset(collections)


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
    collections = {call["collection_name"] for call in nick.qdrant_client.search_calls}
    assert {"c", "uploaded_documents", "static_policy"}.issubset(collections)


def test_upsert_payloads_respects_source_collection():
    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(qdrant_collection_name="procwise_document_embeddings", reranker_model="x"),
        qdrant_client=DummyQdrant(),
        embedding_model=DummyEmbed(),
    )
    rag = RAGService(nick, cross_encoder_cls=DummyCrossEncoder)

    rag.upsert_payloads(
        [
            {
                "record_id": "doc-1",
                "content": "Uploaded content",
                "source_collection": "uploaded_documents",
            }
        ]
    )

    assert nick.qdrant_client.upserts
    assert nick.qdrant_client.upserts[0]["collection_name"] == "uploaded_documents"


def test_pipeline_answer_returns_documents(monkeypatch):
    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None):
            return [SimpleNamespace(id="1", payload={"record_id": "R1", "summary": "s"})]

        def upsert_texts(self, texts, metadata=None):
            pass

        primary_collection = "c"
        uploaded_collection = "uploaded_documents"
        static_policy_collection = "static_policy"
        learning_collection = "learning"

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    class DummyStaticAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, query, user_id, session_id=None, **kwargs):
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"answer": "", "related_prompts": []},
                confidence=0.1,
            )

    monkeypatch.setattr("services.model_selector.RAGAgent", DummyStaticAgent)

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
    assert "[doc 1]" in result["answer"]
    assert len(result["follow_ups"]) == 3


def test_pipeline_returns_fallback_when_no_retrieval(monkeypatch):
    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None):
            return []

        def upsert_texts(self, texts, metadata=None):
            pass

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    class DummyStaticAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, query, user_id, session_id=None, **kwargs):
            return AgentOutput(status=AgentStatus.SUCCESS, data={}, confidence=0.1)

    monkeypatch.setattr("services.model_selector.RAGAgent", DummyStaticAgent)

    history_store = {"payloads": []}

    def _get_object(**kwargs):
        return {"Body": SimpleNamespace(read=lambda: b"[]")}

    def _put_object(**kwargs):
        history_store["payloads"].append(kwargs)

    nick = SimpleNamespace(
        device="cpu",
        s3_client=SimpleNamespace(get_object=_get_object, put_object=_put_object),
        settings=SimpleNamespace(
            qdrant_collection_name="c",
            s3_bucket_name="b",
            reranker_model="x",
            static_qa_confidence_threshold=0.5,
        ),
        embedding_model=DummyEmbed(),
        qdrant_client=SimpleNamespace(),
        ollama_options=lambda: {},
    )

    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder)
    result = pipeline.answer_question("What is our current total savings year to date?", "user-123")

    assert result["answer"] == "I do not have that information as per my knowledge."
    assert result["retrieved_documents"] == []
    assert len(result["follow_ups"]) == 3
    assert history_store["payloads"], "Fallback answers should still be recorded in history"
