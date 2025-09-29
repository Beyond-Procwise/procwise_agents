import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.backend_scheduler as backend_scheduler


class DummyTrainingEndpoint:
    def __init__(self):
        self.dispatched = []

    def dispatch(self, *, force=True, limit=None):
        self.dispatched.append({"force": force, "limit": limit})
        return {"training_jobs": [], "relationship_jobs": []}

    def configure_capture(self, enable):  # pragma: no cover - not used in these tests
        self.capture_state = enable

    def get_service(self):  # pragma: no cover - helper to satisfy interface
        return Mock()


def _prepare_scheduler(monkeypatch, nick, endpoint=None):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_register_default_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )
    if endpoint is not None:
        monkeypatch.setattr(
            backend_scheduler.BackendScheduler,
            "_resolve_training_endpoint",
            lambda self: endpoint,
        )
    return backend_scheduler.BackendScheduler(nick, training_endpoint=endpoint)


def test_submit_once_executes_and_removes_job(monkeypatch):
    scheduler = _prepare_scheduler(monkeypatch, SimpleNamespace())

    executed = []

    scheduler.submit_once("once", lambda: executed.append("ran"))

    job = scheduler._jobs["once"]
    scheduler._execute_job(job)

    assert executed == ["ran"]
    assert "once" not in scheduler._jobs

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_run_model_training_dispatches_via_endpoint(monkeypatch):
    endpoint = DummyTrainingEndpoint()
    scheduler = _prepare_scheduler(
        monkeypatch,
        SimpleNamespace(),
        endpoint=endpoint,
    )

    scheduler._run_model_training()

    assert endpoint.dispatched == [{"force": False, "limit": None}]

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_ensure_updates_training_endpoint_reference(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_register_default_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )

    first_endpoint = DummyTrainingEndpoint()
    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(), training_endpoint=first_endpoint
    )
    assert scheduler._training_endpoint is first_endpoint

    second_endpoint = DummyTrainingEndpoint()
    backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(), training_endpoint=second_endpoint
    )
    assert scheduler._training_endpoint is second_endpoint

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_resolve_training_endpoint_creates_instance(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_register_default_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )

    scheduler = backend_scheduler.BackendScheduler(SimpleNamespace())
    endpoint = scheduler._resolve_training_endpoint()

    from services.model_training_endpoint import ModelTrainingEndpoint

    assert isinstance(endpoint, ModelTrainingEndpoint)

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None

