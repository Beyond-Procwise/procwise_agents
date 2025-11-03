import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import rag_agent as rag_module
from agents.base_agent import AgentStatus


class DummyEmbedder:
    def encode(self, texts):
        import hashlib

        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        size = 64
        for text in texts:
            vec = np.zeros(size, dtype=float)
            for token in text.lower().split():
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = digest[0] % size
                vec[index] += 1.0
            vectors.append(vec)
        return vectors if len(vectors) > 1 else vectors[0]


class DummyRAGService:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.uploaded_collection = "uploaded_documents"
        self.primary_collection = "procwise_document_embeddings"
        self.static_policy_collection = "static_policy"

    def search(self, query, **kwargs):
        return list(self._hits)


@pytest.fixture
def agent(monkeypatch):
    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(rag_model="phi4:latest"),
        embedding_model=DummyEmbedder(),
    )
    monkeypatch.setattr(rag_module, "logger", rag_module.logging.getLogger("rag_agent_test"))
    return rag_module.RAGAgent(nick, rag_service=DummyRAGService())


def _mock_llm(monkeypatch, agent, answer_text, followups=None):
    payload = {
        "message": {
            "content": "<answer>{}</answer><followups>{}</followups>".format(
                answer_text,
                "\n".join(f"- {item}" for item in (followups or [])),
            )
        }
    }

    monkeypatch.setattr(agent, "call_ollama", lambda *_, **__: payload)


def test_rag_agent_static_answer(agent):
    result = agent.run("What is our current total savings year to date?", "user-1")
    assert result.status == AgentStatus.SUCCESS
    answer = result.data["answer"]
    assert answer.startswith("Got it. You're asking about What is our current total savings year to date")
    assert "£2.8 million" in answer
    assert result.data["follow_up_questions"]
    retrieved = result.data["retrieved_documents"]
    assert retrieved and retrieved[0]["collection"] == "static_faq"
    assert result.context_snapshot["static_match"] is True


def test_rag_agent_generates_dynamic_answer(monkeypatch):
    hit = SimpleNamespace(
        payload={
            "title": "Corporate Card Policy",
            "text_summary": "Use the corporate card strictly for business expenses and log receipts within 15 days.",
            "document_type": "policy",
            "collection_name": "static_policy",
        },
        combined_score=7.5,
    )
    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(rag_model="phi4:latest"),
        embedding_model=DummyEmbedder(),
    )
    agent = rag_module.RAGAgent(nick, rag_service=DummyRAGService([hit]))
    _mock_llm(
        monkeypatch,
        agent,
        "Stick to business spend and submit receipts within 15 days.",
        ["Need me to pull the card escalation path?"],
    )

    result = agent.run("What is our credit card policy?", "user-77", session_id="sess-1")
    assert result.status == AgentStatus.SUCCESS
    answer = result.data["answer"]
    assert answer.startswith("Got it. You're asking about What is our credit card policy")
    assert "business spend" in answer
    assert result.data["follow_up_questions"] == ["Need me to pull the card escalation path?"]
    assert result.data["retrieved_documents"][0]["collection"] == "static_policy"
    assert result.context_snapshot["policy_mode"] is True


def test_rag_agent_handles_continuation(monkeypatch):
    hit = SimpleNamespace(
        payload={
            "title": "Corporate Card Policy",
            "text_summary": "Manager approval is required for purchases above £500 and travel must be booked via the TMC.",
            "document_type": "policy",
            "collection_name": "static_policy",
        },
        combined_score=6.8,
    )
    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(rag_model="phi4:latest"),
        embedding_model=DummyEmbedder(),
    )
    agent = rag_module.RAGAgent(nick, rag_service=DummyRAGService([hit]))
    _mock_llm(monkeypatch, agent, "Approvals kick in at £500 and travel needs pre-booking.")
    agent.run("What is our credit card policy?", "user-88", session_id="sess-9")
    _mock_llm(monkeypatch, agent, "Approval thresholds sit at £500 and exceptions require finance sign-off.")
    follow_up = agent.run("Could you elaborate on the card limits?", "user-88", session_id="sess-9")

    assert follow_up.status == AgentStatus.SUCCESS
    ack_line = follow_up.data["answer"].splitlines()[0]
    assert "and you'd like more detail on" in ack_line
    assert "card limits" in ack_line.lower()
    assert follow_up.context_snapshot["retrieved_count"] >= 1
