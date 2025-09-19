from types import SimpleNamespace

from agents import rag_agent as rag_module


class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.0 for _ in pairs]


class DummyIntelligentRetrieval:
    def __init__(self, *args, **kwargs):
        pass
    
    def adaptive_search(self, query, top_k=5, filters=None):
        return []
    
    def detect_query_intent(self, query):
        from services.intelligent_retrieval import QueryIntent
        return QueryIntent.FACTUAL


def test_rag_agent_applies_filters(monkeypatch):
    monkeypatch.setattr(rag_module, "CrossEncoder", DummyCrossEncoder)
    monkeypatch.setattr(rag_module, "IntelligentRetrievalService", DummyIntelligentRetrieval)

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

    # Mock intelligent retrieval to test fallback behavior
    def mock_adaptive_search(query, top_k=5, filters=None):
        captured["filters"] = filters
        # Simulate failure to test fallback
        raise Exception("Mock failure")
    
    agent.intelligent_retrieval.adaptive_search = mock_adaptive_search

    agent.run("q", "u", doc_type="Invoice")

    flt = captured.get("filters")
    assert flt is not None
    assert flt.must[0].key == "document_type"
    assert flt.must[0].match.value == "invoice"


def test_rag_agent_uses_intelligent_retrieval(monkeypatch):
    """Test that RAG agent uses intelligent retrieval when available."""
    monkeypatch.setattr(rag_module, "CrossEncoder", DummyCrossEncoder)
    
    search_calls = []
    
    class DummyIntelligentRetrieval:
        def __init__(self, *args, **kwargs):
            pass
        
        def adaptive_search(self, query, top_k=5, filters=None):
            search_calls.append({"query": query, "top_k": top_k, "filters": filters})
            return [SimpleNamespace(
                id="test_doc",
                payload={"content": "test content", "record_id": "test_id"},
                score=0.9
            )]
        
        def detect_query_intent(self, query):
            from services.intelligent_retrieval import QueryIntent
            return QueryIntent.FACTUAL

    monkeypatch.setattr(rag_module, "IntelligentRetrievalService", DummyIntelligentRetrieval)
    monkeypatch.setattr(rag_module, "RAGService", lambda *args: SimpleNamespace())

    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(reranker_model="dummy", extraction_model="test"),
        qdrant_client=SimpleNamespace(),
        embedding_model=SimpleNamespace(),
    )
    
    agent = rag_module.RAGAgent(nick)
    agent._load_chat_history = lambda u, s=None: []
    agent._save_chat_history = lambda u, h, s=None: None
    agent.call_ollama = lambda prompt, model: {"response": "test answer"}
    agent._generate_followups = lambda q, c: ["follow up 1", "follow up 2"]
    # Mock the rerank method to return the hits
    agent._rerank = lambda query, hits, top_k: hits[:top_k]
    
    result = agent.run("test query", "test_user")
    
    # Should have used intelligent retrieval
    assert len(search_calls) == 1
    assert search_calls[0]["query"] == "test query"
    
    # Should return proper response
    assert "answer" in result
    assert result["answer"] == "test answer"
    assert len(result["follow_up_questions"]) == 2
