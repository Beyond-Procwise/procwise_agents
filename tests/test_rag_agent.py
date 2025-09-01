from types import SimpleNamespace

from agents import rag_agent as rag_module


class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.0 for _ in pairs]


def test_rag_agent_applies_filters(monkeypatch):
    monkeypatch.setattr(rag_module, "CrossEncoder", DummyCrossEncoder)

    captured = {}

    class DummyRAGService:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None):
            captured["filters"] = filters
            return []

    monkeypatch.setattr(rag_module, "RAGService", DummyRAGService)

    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(reranker_model="dummy"),
        qdrant_client=SimpleNamespace(),
        embedding_model=SimpleNamespace(),
    )
    agent = rag_module.RAGAgent(nick)
    agent._expand_query = lambda q: []
    agent._load_chat_history = lambda u: []
    agent._save_chat_history = lambda u, h: None
    agent.call_ollama = lambda prompt, model: {"response": "ans"}
    agent._generate_followups = lambda q, c: []

    agent.run("q", "u", doc_type="Invoice")

    flt = captured.get("filters")
    assert flt is not None
    assert flt.must[0].key == "document_type"
    assert flt.must[0].match.value == "invoice"
