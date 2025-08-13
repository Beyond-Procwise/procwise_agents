import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import rag_agent as rag_module


class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.5 for _ in pairs]


def test_expand_query_parses_llm_response(monkeypatch):
    monkeypatch.setattr(rag_module, "CrossEncoder", DummyCrossEncoder)
    nick = SimpleNamespace(device="cpu", settings=SimpleNamespace(reranker_model="x", extraction_model="m"))
    agent = rag_module.RAGAgent(nick)

    def fake_ollama(prompt, model=None, format=None):
        return {"response": json.dumps({"expansions": ["alt1", "alt2"]})}

    monkeypatch.setattr(agent, "call_ollama", fake_ollama)
    expansions = agent._expand_query("test query")
    assert expansions == ["alt1", "alt2"]


def test_run_returns_search_hits(monkeypatch):
    monkeypatch.setattr(rag_module, "CrossEncoder", DummyCrossEncoder)

    class DummyEmbed:
        def encode(self, text):
            return SimpleNamespace(tolist=lambda: [0.0])

    class DummyQdrant:
        def search(self, **kwargs):
            hit = SimpleNamespace(id="1", payload={"record_id": "INV-2025-055", "content": "doc"}, score=1.0)
            return [hit]

    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(
            reranker_model="x",
            extraction_model="m",
            qdrant_collection_name="c",
            s3_bucket_name="b",
        ),
        embedding_model=DummyEmbed(),
        qdrant_client=DummyQdrant(),
    )
    agent = rag_module.RAGAgent(nick)
    monkeypatch.setattr(agent, "_load_chat_history", lambda user_id: [])
    monkeypatch.setattr(agent, "_save_chat_history", lambda user_id, hist: None)
    monkeypatch.setattr(agent, "_expand_query", lambda q: [])
    monkeypatch.setattr(agent, "_generate_followups", lambda q, c: [])
    monkeypatch.setattr(agent, "_rerank", lambda q, hits, k: hits)
    monkeypatch.setattr(agent, "call_ollama", lambda p, model=None: {"response": "ans"})

    result = agent.run("details for INV-2025-055", "user")
    assert result["retrieved_documents"][0]["record_id"] == "INV-2025-055"
