import os
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.backend_scheduler as backend_scheduler


class DummyEmailWatcher:
    def __init__(self, *_, **__):
        self.started = False
        self.notifications: list[str] = []

    def start(self):
        self.started = True

    def stop(self, timeout: float = 5.0):  # pragma: no cover - simple stub
        self.started = False

    def notify_workflow(self, workflow_id: str):
        self.notifications.append(workflow_id)


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
    monkeypatch.setattr(backend_scheduler, "EmailWatcherService", DummyEmailWatcher)
    if endpoint is not None:
        monkeypatch.setattr(
            backend_scheduler.BackendScheduler,
            "_resolve_training_endpoint",
            lambda self: endpoint,
        )
    return backend_scheduler.BackendScheduler(nick, training_endpoint=endpoint)


def _spawn_scheduler(monkeypatch, settings):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
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
    monkeypatch.setattr(backend_scheduler, "EmailWatcherService", DummyEmailWatcher)
    nick = SimpleNamespace(settings=settings)
    return backend_scheduler.BackendScheduler(nick)


def test_submit_once_executes_and_removes_job(monkeypatch):
    scheduler = _prepare_scheduler(monkeypatch, SimpleNamespace())

    executed = []

    scheduler.submit_once("once", lambda: executed.append("ran"))

    job = scheduler._jobs["once"]
    scheduler._execute_job(job)

    assert executed == ["ran"]
    assert "once" not in scheduler._jobs
    assert isinstance(scheduler._email_watcher_service, DummyEmailWatcher)
    assert scheduler._email_watcher_service.started is True

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


def test_notify_email_dispatch_wakes_watcher(monkeypatch):
    scheduler = _prepare_scheduler(monkeypatch, SimpleNamespace())

    assert isinstance(scheduler._email_watcher_service, DummyEmailWatcher)
    assert scheduler._email_watcher_service.started is True

    scheduler.notify_email_dispatch("wf-demo")

    assert scheduler._email_watcher_service.notifications == ["wf-demo"]

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


def test_training_scheduler_skips_job_when_setting_missing(monkeypatch):
    scheduler = _spawn_scheduler(monkeypatch, SimpleNamespace())

    job_name = backend_scheduler.BackendScheduler.TRAINING_JOB_NAME
    assert job_name not in scheduler._jobs

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_training_scheduler_registers_job_when_enabled(monkeypatch):
    scheduler = _spawn_scheduler(
        monkeypatch, SimpleNamespace(enable_training_scheduler=True)
    )

    job_name = backend_scheduler.BackendScheduler.TRAINING_JOB_NAME
    assert job_name in scheduler._jobs
    assert scheduler._jobs[job_name].interval == timedelta(hours=6)

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_training_scheduler_reconfigures_on_ensure(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
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
    monkeypatch.setattr(backend_scheduler, "EmailWatcherService", DummyEmailWatcher)

    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(settings=SimpleNamespace())
    )
    job_name = backend_scheduler.BackendScheduler.TRAINING_JOB_NAME
    assert job_name not in scheduler._jobs

    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(settings=SimpleNamespace(enable_training_scheduler=True))
    )
    assert job_name in scheduler._jobs

    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(settings=SimpleNamespace(enable_training_scheduler=False))
    )
    assert job_name not in scheduler._jobs

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None

