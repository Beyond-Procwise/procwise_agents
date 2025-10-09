import os
import sys
from types import SimpleNamespace

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
    assert len(service._pending_negotiation_learnings) == 1
