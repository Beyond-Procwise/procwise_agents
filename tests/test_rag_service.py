import os
import sys
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentOutput, AgentStatus

from services.rag_service import RAGService
from services.model_selector import RAGPipeline
from services.semantic_chunker import SemanticChunk


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


def test_chunk_text_fallback_splits_long_documents(monkeypatch):
    settings = SimpleNamespace(
        qdrant_collection_name="c",
        reranker_model="x",
        rag_chunk_chars=600,
        rag_chunk_overlap=150,
    )
    nick = SimpleNamespace(
        device="cpu",
        settings=settings,
        qdrant_client=DummyQdrant(),
        embedding_model=DummyEmbed(),
    )
    rag = RAGService(nick, cross_encoder_cls=DummyCrossEncoder)

    def _raise_layout_error(*args, **kwargs):
        raise ValueError("no layout available")

    monkeypatch.setattr(rag._layout_parser, "from_text", _raise_layout_error)

    long_text = "Paragraph one explains procurement policies." + " " + (
        "This sentence elaborates on approval workflows. " * 80
    )

    chunks = rag._chunk_text(long_text)
    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, SemanticChunk)
        assert chunk.metadata.get("chunk_strategy") == "fallback"
        assert chunk.metadata.get("fallback_index") == idx
        assert len(chunk.content) <= rag._chunk_char_limit()


def test_pipeline_answer_returns_documents(monkeypatch):
    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None, **kwargs):
            return [SimpleNamespace(id="1", payload={"record_id": "R1", "summary": "s"})]

        def upsert_texts(self, texts, metadata=None):
            pass

        primary_collection = "c"
        uploaded_collection = "uploaded_documents"
        static_policy_collection = "static_policy"
        learning_collection = "learning"

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    def _fake_generate_response(self, prompt, model):
        return {
            "answer": "Supplier summary includes s.",
            "follow_ups": [
                "Would you like deeper supplier insights?",
                "Any other procurement roadblocks I can remove?",
            ],
        }

    monkeypatch.setattr(
        RAGPipeline,
        "_generate_response",
        _fake_generate_response,
    )

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
    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder, use_nltk=False)
    result = pipeline.answer_question("q", "user")
    assert "record_id" not in result["retrieved_documents"][0]
    assert result["retrieved_documents"][0]["summary"] == "s"
    assert "[doc" not in result["answer"]
    assert "[redacted" not in result["answer"].lower()
    assert "s" in result["answer"]
    assert "\n-" not in result["answer"]
    assert len(result["follow_ups"]) == 3


def test_pipeline_returns_fallback_when_no_retrieval(monkeypatch):
    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None, **kwargs):
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

    monkeypatch.setattr(
        RAGPipeline,
        "_generate_response",
        lambda self, prompt, model: {"answer": "", "follow_ups": []},
    )

    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder, use_nltk=False)
    result = pipeline.answer_question("What is our current total savings year to date?", "user-123")

    assert (
        result["answer"]
        == "I'm sorry, but I couldn't find that information in the available knowledge base."
    )
    assert result["retrieved_documents"] == []


def test_pipeline_uploaded_context_mode(monkeypatch):
    captured: Dict[str, Any] = {}

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None, **kwargs):
            captured.update({"filters": filters, "collections": kwargs.get("collections")})
            return [
                SimpleNamespace(
                    id="chunk-1",
                    payload={
                        "collection_name": "uploaded_documents",
                        "summary": "Uploaded insight",
                        "document_id": "doc-123",
                    },
                    combined_score=1.0,
                    rerank_score=1.0,
                )
            ]

        def upsert_texts(self, texts, metadata=None):
            pass

        primary_collection = "c"
        uploaded_collection = "uploaded_documents"
        static_policy_collection = "static_policy"
        learning_collection = "learning"

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    class GuardStaticAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise AssertionError("Static agent should not run when uploaded context is active")

    monkeypatch.setattr("services.model_selector.RAGAgent", GuardStaticAgent)

    monkeypatch.setattr(
        RAGPipeline,
        "_generate_response",
        lambda self, prompt, model: {"answer": "Uploaded insight", "follow_ups": []},
    )

    nick = SimpleNamespace(
        device="cpu",
        s3_client=SimpleNamespace(
            get_object=lambda **_: {"Body": SimpleNamespace(read=lambda: b"[]")},
            put_object=lambda **_: None,
        ),
        settings=SimpleNamespace(qdrant_collection_name="c", s3_bucket_name="b", reranker_model="x"),
        embedding_model=DummyEmbed(),
        qdrant_client=SimpleNamespace(),
        ollama_options=lambda: {},
    )

    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder, use_nltk=False)
    pipeline.activate_uploaded_context(
        ["doc-123"],
        metadata={"filenames": ["file.pdf"]},
        session_id="user-456",
    )

    result = pipeline.answer_question("What does the document say?", "user-456")

    assert captured["collections"] == ("uploaded_documents",)
    filter_obj = captured["filters"]
    assert filter_obj is not None
    assert any(
        getattr(condition, "key", "") == "document_id"
        for condition in getattr(filter_obj, "must", [])
    )
    assert result["retrieved_documents"]
    assert all(
        doc.get("collection_name") == "uploaded_documents"
        for doc in result["retrieved_documents"]
    )
    assert len(result["follow_ups"]) == 3


def test_pipeline_uploaded_context_scoped_to_session(monkeypatch):
    collections_history: List[Optional[tuple]] = []

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None, **kwargs):
            collections = kwargs.get("collections")
            collections_history.append(collections)
            active_collection = collections[0] if collections else "primary_collection"
            document_id = "doc-uploaded" if collections else "doc-primary"
            return [
                SimpleNamespace(
                    id=f"chunk-{document_id}",
                    payload={
                        "collection_name": active_collection,
                        "summary": "Relevant insight",
                        "document_id": document_id,
                    },
                    combined_score=1.0,
                    rerank_score=1.0,
                )
            ]

        def upsert_texts(self, texts, metadata=None):
            pass

        primary_collection = "primary_collection"
        uploaded_collection = "uploaded_documents"
        static_policy_collection = "static_policy"
        learning_collection = "learning"

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    class PassiveStaticAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return AgentOutput(status=AgentStatus.FAILURE, data={})

    monkeypatch.setattr("services.model_selector.RAGAgent", PassiveStaticAgent)

    monkeypatch.setattr(
        RAGPipeline,
        "_generate_response",
        lambda self, prompt, model: {"answer": "Insight", "follow_ups": []},
    )

    nick = SimpleNamespace(
        device="cpu",
        s3_client=SimpleNamespace(
            get_object=lambda **_: {"Body": SimpleNamespace(read=lambda: b"[]")},
            put_object=lambda **_: None,
        ),
        settings=SimpleNamespace(qdrant_collection_name="c", s3_bucket_name="b", reranker_model="x"),
        embedding_model=DummyEmbed(),
        qdrant_client=SimpleNamespace(),
        ollama_options=lambda: {},
    )

    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder, use_nltk=False)
    pipeline.activate_uploaded_context(
        ["doc-uploaded"],
        metadata={"filenames": ["contract.pdf"]},
        session_id="session-123",
    )

    first_result = pipeline.answer_question(
        "Summarise the uploaded contract", "user-123", session_id="session-123"
    )
    second_result = pipeline.answer_question(
        "Provide other insights", "user-456", session_id="session-999"
    )

    assert collections_history[0] == ("uploaded_documents",)
    assert collections_history[1] != ("uploaded_documents",)
    assert first_result["retrieved_documents"][0]["collection_name"] == "uploaded_documents"
    assert second_result["retrieved_documents"][0]["collection_name"] != "uploaded_documents"

def test_pipeline_prefers_explicit_session_id(monkeypatch):
    captured: Dict[str, Any] = {}

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, **kwargs):
            captured["query"] = query
            captured["kwargs"] = kwargs
            return []

        def upsert_texts(self, texts, metadata=None):
            pass

    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)

    class DummyStaticAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return AgentOutput(status=AgentStatus.SUCCESS, data={}, confidence=0.0)

    monkeypatch.setattr("services.model_selector.RAGAgent", DummyStaticAgent)

    history_store: Dict[str, Any] = {"payloads": []}

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

    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder, use_nltk=False)
    pipeline.register_session_upload("session-xyz", ["doc-1"])

    pipeline.answer_question("q", "user-123", session_id="session-xyz")

    assert captured["kwargs"]["session_id"] == "session-xyz"
    assert history_store["payloads"], "History should still be saved even without retrieval"
