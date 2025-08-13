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
