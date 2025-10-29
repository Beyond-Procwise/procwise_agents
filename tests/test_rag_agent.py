from types import SimpleNamespace
from typing import Dict

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
    agent._load_chat_history = lambda u, s=None: []
    agent._save_chat_history = lambda u, h, s=None: None
    agent.call_ollama = lambda *args, **kwargs: {"response": "ans"}
    agent._generate_followups = lambda q, c: []

    agent.run("q", "u", doc_type="Invoice")

    flt = captured.get("filters")
    assert flt is not None
    assert flt.must[0].key == "document_type"
    assert flt.must[0].match.value == "invoice"


def test_rag_agent_generates_agentic_plan(monkeypatch):
    monkeypatch.setattr(rag_module, "CrossEncoder", DummyCrossEncoder)

    class DummyRAGService:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query, top_k=5, filters=None):
            return [
                SimpleNamespace(
                    id="doc-1",
                    payload={
                        "record_id": "DOC1",
                        "document_type": "invoice",
                        "content": "Invoice total 1,200 GBP with extended warranty",
                        "supplier_name": "Supplier One",
                    },
                    score=0.7,
                ),
                SimpleNamespace(
                    id="kg-1",
                    payload={
                        "document_type": "supplier_flow",
                        "supplier_id": "S1",
                        "supplier_name": "Supplier One",
                        "coverage_ratio": 0.75,
                        "purchase_orders": {"count": 3},
                    },
                    score=0.9,
                ),
            ]

    monkeypatch.setattr(rag_module, "RAGService", DummyRAGService)

    captured_plan: Dict[str, str] = {}

    def fake_plan(self, query, outline, knowledge_summary):
        captured_plan["query"] = query
        captured_plan["outline"] = outline
        captured_plan["knowledge_summary"] = knowledge_summary
        return "1. Investigate supplier coverage\n2. Confirm invoice specifics"

    monkeypatch.setattr(
        rag_module.RAGAgent, "_generate_agentic_plan", fake_plan, raising=False
    )

    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(reranker_model="dummy", extraction_model="test-model"),
        qdrant_client=SimpleNamespace(),
        embedding_model=SimpleNamespace(),
    )

    agent = rag_module.RAGAgent(nick)
    agent._expand_query = lambda q: []
    agent._load_chat_history = lambda u, s=None: []
    agent._save_chat_history = lambda u, h, s=None: None
    agent.call_ollama = (
        lambda *args, **kwargs: {"response": "Final answer"}
    )
    agent._generate_followups = lambda q, c: ["Next step"]

    result = agent.run("How is Supplier One performing?", "user-1")

    assert isinstance(result, rag_module.AgentOutput)
    answer = result.data["answer"]
    assert answer.startswith("Q: How is Supplier One performing?")
    assert "A: Final answer" in answer
    assert "Suggested Follow-Up Prompts:" in answer
    assert answer.count("- ") >= 3

    followups = result.data["follow_up_questions"]
    assert len(followups) == 3
    assert followups[0] == "Next step"

    retrieved = result.data["retrieved_documents"]
    knowledge_entry = next(
        item
        for item in retrieved
        if item.get("document_type") == "knowledge_summary"
    )
    assert "Supplier One" in knowledge_entry["summary"]
    assert "coverage" in knowledge_entry["summary"].lower()
    assert knowledge_entry.get("topic") == "knowledge"
    assert captured_plan["knowledge_summary"]
    assert "Supplier One" in captured_plan["knowledge_summary"]
    assert "Invoice" in captured_plan["outline"]
    assert result.agentic_plan == "1. Investigate supplier coverage\n2. Confirm invoice specifics"
