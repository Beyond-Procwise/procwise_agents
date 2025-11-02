import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services import model_training_service as mts
from services.model_training_service import ModelTrainingService


class DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        class DummyCursor:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                pass

            def execute(self_inner, *args, **kwargs):
                pass

            def fetchone(self_inner):
                return None

            def fetchall(self_inner):
                return []

        return DummyCursor()

    def commit(self):
        pass


class RepoRecorder:
    def __init__(self):
        self.calls = []

    def record_negotiation_learning(self, **payload):
        self.calls.append(payload)
        return f"point-{len(self.calls)}"

    def get_recent_learnings(self, **_):
        return []


class StubService(ModelTrainingService):
    def dispatch_due_jobs(self, *, force: bool = True, limit=None):
        return []

    def _get_relationship_scheduler(self):
        return None


@pytest.fixture(autouse=True)
def patch_event_bus(monkeypatch):
    bus = SimpleNamespace(subscribe=lambda *a, **k: None, unsubscribe=lambda *a, **k: None)
    monkeypatch.setattr(mts, "get_event_bus", lambda: bus)
    yield


@pytest.fixture(autouse=True)
def patch_rag_trainer_factory(monkeypatch):
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.calls: List[Dict[str, Any]] = []

        def train(self, payload: Dict[str, Any]):
            self.calls.append(dict(payload))
            return {"layer0": {}}

    trainer = DummyTrainer()
    monkeypatch.setattr(mts, "LayeredRAGTrainer", lambda *a, **k: trainer)
    yield trainer


@pytest.fixture(autouse=True)
def patch_phi4_fine_tuner(monkeypatch):
    created: List[Any] = []

    class DummyFineTuner:
        def __init__(self, *args, **kwargs):
            self.calls: List[bool] = []
            created.append(self)

        def dispatch(self, *, force: bool = False):
            self.calls.append(force)
            return {
                "status": "completed",
                "sft_dataset": {"path": "datasets/phi4.jsonl", "sample_count": 0},
                "preference_dataset": {
                    "path": "datasets/phi4_pref.jsonl",
                    "sample_count": 0,
                },
                "artifacts": {
                    "report": "artifacts/report.json",
                    "adapters": "artifacts/adapters",
                    "quantized_model": "artifacts/model.gguf",
                },
            }

    monkeypatch.setattr(mts, "Phi4HumanizationFineTuner", DummyFineTuner)
    yield created


def test_dispatch_flushes_pending_negotiation_learnings():
    repo = RepoRecorder()
    nick = SimpleNamespace(
        settings=SimpleNamespace(enable_learning=True),
        policy_engine=SimpleNamespace(),
        learning_repository=repo,
        get_db_connection=lambda: DummyConn(),
    )

    service = StubService(nick)
    service.queue_negotiation_learning(
        workflow_id="wf-1",
        rfq_id="RFQ-10",
        supplier_id="SUP-10",
        decision={"round": 1, "strategy": "counter"},
        state={"supplier_reply_count": 1},
        awaiting_response=True,
        supplier_reply_registered=False,
    )

    assert len(service._pending_negotiation_learnings) == 1

    result = service.dispatch_training_and_refresh(force=True)

    assert repo.calls and repo.calls[0]["rfq_id"] == "RFQ-10"
    assert result["negotiation_learnings"][0]["status"] == "recorded"
    assert result["phi4_fine_tuning"]["status"] == "completed"
    assert not service._pending_negotiation_learnings


def test_dispatch_requeues_when_repository_missing():
    nick = SimpleNamespace(
        settings=SimpleNamespace(enable_learning=True),
        policy_engine=SimpleNamespace(),
        learning_repository=None,
        get_db_connection=lambda: DummyConn(),
    )

    service = StubService(nick)
    service.queue_negotiation_learning(
        workflow_id="wf-2",
        rfq_id="RFQ-20",
        supplier_id="SUP-20",
        decision={"round": 2},
        state={},
        awaiting_response=False,
        supplier_reply_registered=True,
    )

    result = service.dispatch_training_and_refresh(force=True)

    assert result["negotiation_learnings"][0]["status"] == "skipped"
    assert result["phi4_fine_tuning"]["status"] == "completed"
    assert len(service._pending_negotiation_learnings) == 1


def test_run_training_job_triggers_rag_pipeline(monkeypatch):
    nick = SimpleNamespace(
        settings=SimpleNamespace(enable_learning=False),
        policy_engine=SimpleNamespace(),
        learning_repository=None,
        get_db_connection=lambda: DummyConn(),
    )

    service = StubService(nick)

    class Recorder:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []

        def train(self, payload: Dict[str, Any]):
            self.calls.append(dict(payload))
            return {"layer0": {"primary": True}}

    recorder = Recorder()
    service._rag_trainer = recorder  # type: ignore[assignment]

    job = {"agent_slug": "rag_agent", "payload": {"documents": []}}
    service._run_training_job(job)

    assert recorder.calls, "Expected rag trainer to be invoked"
    assert job["payload"]["training_result"]["layer0"]["primary"] is True
