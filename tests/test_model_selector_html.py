import os
import re
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from services.model_selector import RAGPipeline


def _wrap(body: str) -> str:
    return (
        "<section class=\"llm-answer\">"
        "<article class=\"llm-answer__content\">"
        f"{body}"
        "</article>"
        "</section>"
    )


def _make_pipeline() -> RAGPipeline:
    return RAGPipeline.__new__(RAGPipeline)


def test_normalise_answer_html_preserves_angle_brackets():
    pipeline = _make_pipeline()

    result = pipeline._normalise_answer_html("Cost A < Cost B")

    assert result == _wrap(
        '<section class="llm-answer__segment"><p>Cost A &lt; Cost B</p></section>'
    )


def test_normalise_answer_html_strips_basic_tags():
    pipeline = _make_pipeline()

    result = pipeline._normalise_answer_html("<strong>Winner:</strong> Supplier < 2")

    assert result == _wrap(
        '<section class="llm-answer__segment"><p>Winner: Supplier &lt; 2</p></section>'
    )
    assert not re.search(r"<strong>.*</strong>", result)


def test_generate_response_uses_stream_when_enabled(monkeypatch):
    pipeline = _make_pipeline()
    pipeline.agent_nick = SimpleNamespace(ollama_options=lambda: {})

    payload = "{\"answer\": \"Hello\", \"follow_ups\": [\"Next\"]}"
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            def _iterator():
                yield {"message": {"content": payload}}

            return _iterator()
        return {"message": {"content": payload}}

    original = settings.stream_llm_responses
    settings.stream_llm_responses = True
    try:
        monkeypatch.setattr("services.model_selector.ollama.chat", fake_chat)
        result = pipeline._generate_response("prompt", "model")
    finally:
        settings.stream_llm_responses = original

    assert result == {"answer": "Hello", "follow_ups": ["Next"]}
    assert len(calls) == 1
    assert calls[0]["stream"] is True
