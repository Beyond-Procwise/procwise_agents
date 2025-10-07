import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent


class DummyProcessRoutingService:
    def log_process(self, **_kwargs):
        return None

    def log_run_detail(self, **_kwargs):
        return None

    def log_action(self, **_kwargs):
        return "action"


class DummyAgentNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            script_user="tester",
            extraction_model="gpt-oss",
            ollama_quantized_model=None,
        )
        self.prompt_engine = SimpleNamespace()
        self.learning_repository = None
        self.process_routing_service = DummyProcessRoutingService()

    def ollama_options(self):
        return {}


def make_base_agent():
    return base_agent.BaseAgent(DummyAgentNick())


def test_call_ollama_prefers_fallback_when_list_empty(monkeypatch):
    monkeypatch.setattr(base_agent.ollama, "list", lambda: {"models": []})

    captured = {}

    def fake_generate(model, **kwargs):
        captured["model"] = model
        return {"response": "ok", "model": model, "kwargs": kwargs}

    monkeypatch.setattr(base_agent.ollama, "generate", fake_generate)

    agent = make_base_agent()

    result = agent.call_ollama(prompt="hi", model="totally-missing")

    assert result["model"] == base_agent._OLLAMA_FALLBACK_MODELS[0]
    assert captured["model"] == base_agent._OLLAMA_FALLBACK_MODELS[0]
    assert getattr(agent.agent_nick, "_available_ollama_models") == list(
        base_agent._OLLAMA_FALLBACK_MODELS
    )


def test_get_available_models_returns_fallback_when_empty(monkeypatch):
    agent = make_base_agent()

    monkeypatch.setattr(base_agent.ollama, "list", lambda: {"models": []})

    models = agent._get_available_ollama_models(force_refresh=True)

    assert models == list(base_agent._OLLAMA_FALLBACK_MODELS)
