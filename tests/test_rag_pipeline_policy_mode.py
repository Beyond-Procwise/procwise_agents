import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentOutput, AgentStatus
from services.model_selector import RAGPipeline


class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.5 for _ in pairs]


class DummyEmbed:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, list):
            return [[0.1] for _ in texts]
        return [0.1]


def _history_payload(entries):
    body = json.dumps(entries).encode("utf-8")

    class _Body:
        def read(self):
            return body

    return {"Body": _Body()}


def test_supplier_history_does_not_trigger_policy_mode(monkeypatch):
    class DummyStaticAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, query, user_id, session_id=None, **kwargs):
            return AgentOutput(status=AgentStatus.FAILURE, data={}, confidence=0.0)

    class DummyRAG:
        def __init__(self, agent_nick, *_, **__):
            self.agent_nick = agent_nick
            self.primary_collection = "procwise_document_embeddings"
            self.uploaded_collection = "uploaded_documents"
            self.static_policy_collection = "static_policy"
            self.search_calls = []

        def search(self, query, **kwargs):
            self.search_calls.append(kwargs)
            payload = {
                "collection_name": self.primary_collection,
                "summary": "Supplier master data is available in the knowledge base.",
            }
            return [SimpleNamespace(payload=payload, score=0.9)]

        def upsert_texts(self, texts, metadata=None):
            return None

    monkeypatch.setattr("services.model_selector.RAGAgent", DummyStaticAgent)
    monkeypatch.setattr("services.model_selector.RAGService", DummyRAG)
    monkeypatch.setattr(
        RAGPipeline,
        "_generate_response",
        lambda self, prompt, model: {"answer": "LLM answer", "follow_up_questions": []},
    )

    history_entries = [
        {
            "query": "Show me the supplier list for this quarter",
            "answer": "The supplier master data is stored in the vendor registry.",
        },
        {
            "query": "Any supplier onboarding updates?",
            "answer": "The vendor management team is reviewing new partners.",
        },
    ]

    s3_client = SimpleNamespace(
        get_object=lambda **_: _history_payload(history_entries),
        put_object=lambda **_: None,
    )

    settings = SimpleNamespace(
        qdrant_collection_name="procwise_document_embeddings",
        reranker_model="reranker",
        s3_bucket_name="bucket",
    )

    nick = SimpleNamespace(
        device="cpu",
        s3_client=s3_client,
        settings=settings,
        embedding_model=DummyEmbed(),
        qdrant_client=SimpleNamespace(),
        ollama_options=lambda: {},
    )

    pipeline = RAGPipeline(nick, cross_encoder_cls=DummyCrossEncoder, use_nltk=False)

    result = pipeline.answer_question("Who manages our supplier relationships?", "user-123")

    assert pipeline.rag.search_calls, "Search should be invoked"
    kwargs = pipeline.rag.search_calls[0]
    assert kwargs.get("policy_mode") is False
    collections = kwargs.get("collections")
    assert not collections or pipeline.rag.primary_collection in collections

    assert result["retrieved_documents"], "Documents from the primary collection should be returned"
    first_doc = result["retrieved_documents"][0]
    assert first_doc["collection_name"] == pipeline.rag.primary_collection

