from types import SimpleNamespace
import pathlib
import sys

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


@pytest.fixture
def agent(monkeypatch):
    nick = SimpleNamespace(
        device="cpu",
        settings=SimpleNamespace(disable_rag_atoms_llm=True),
        embedding_model=DummyEmbedder(),
    )
    # Avoid GPU initialisation noise in tests
    monkeypatch.setattr(rag_module, "logger", rag_module.logging.getLogger("rag_agent_test"))
    return rag_module.RAGAgent(nick)


def test_rag_agent_returns_static_answer(agent):
    result = agent.run("What is our current total savings year to date?", "user-1")
    assert result.status == AgentStatus.SUCCESS
    answer = result.data["answer"]
    assert answer.startswith("<section>")
    assert "<h2>Summary</h2>" in answer
    assert "<h3>Scope &amp; Applicability</h3>" in answer
    assert "<h3>Key Rules</h3>" in answer
    assert "<h3>Prohibited / Exclusions</h3>" in answer
    assert "<h3>Effective Dates &amp; Ownership</h3>" in answer
    assert "<h3>Next Steps</h3>" in answer
    assert result.data["structure_type"] == "financial_analysis"
    assert result.data["structured"] is True
    assert any("£" in point for point in result.data["main_points"])
    assert "£2.8 million" in result.data["original_answer"]
    assert result.data["topic"] == "Current Total Savings Year-to-Date"
    assert len(result.data["related_prompts"]) == 3


def test_rag_agent_maintains_topic_context(agent):
    first = agent.run(
        "Who are my approved suppliers I can use for events?",
        "user-1",
        session_id="session-A",
    )
    follow_up = agent.run(
        "How do I request event supplier engagement?",
        "user-1",
        session_id="session-A",
    )
    assert follow_up.data["topic"] == first.data["topic"]
    assert "<h2>Summary</h2>" in follow_up.data["answer"]
    assert "<h3>Scope &amp; Applicability</h3>" in follow_up.data["answer"]


def test_rag_agent_switches_topic_on_new_question(agent):
    agent.run("What is our current total savings year to date?", "user-2", session_id="session-B")
    next_answer = agent.run(
        "What business expenses can I not claim?",
        "user-2",
        session_id="session-B",
    )
    assert next_answer.data["topic"] == "Business Expenses That Cannot Be Claimed"
    assert "<h2>Summary</h2>" in next_answer.data["answer"]
    assert "expenses" in next_answer.data["answer"].lower()
