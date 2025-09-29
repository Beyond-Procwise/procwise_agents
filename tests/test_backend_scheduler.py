import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.backend_scheduler as backend_scheduler


def test_submit_once_executes_and_removes_job(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: "gpu")
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
        "_init_training_service",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )

    scheduler = backend_scheduler.BackendScheduler(types.SimpleNamespace())

    executed = []

    scheduler.submit_once("once", lambda: executed.append("ran"))

    job = scheduler._jobs["once"]
    scheduler._execute_job(job)

    assert executed == ["ran"]
    assert "once" not in scheduler._jobs

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def _prepare_scheduler(monkeypatch, nick):
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
    return backend_scheduler.BackendScheduler(nick)


def test_init_training_service_disables_capture_when_learning_off(monkeypatch):
    events = {"disabled": 0, "enabled": 0, "auto": []}

    import services.model_training_service as training_module

    class DummyService:
        def __init__(self, *_args, auto_subscribe=False, **_kwargs):
            events["auto"].append(auto_subscribe)
            if auto_subscribe:
                self.enable_workflow_capture()

        def disable_workflow_capture(self):
            events["disabled"] += 1

        def enable_workflow_capture(self):
            events["enabled"] += 1

        def dispatch_due_jobs(self):  # pragma: no cover - helper stub
            return []

    monkeypatch.setattr(training_module, "ModelTrainingService", DummyService)

    nick = types.SimpleNamespace(
        settings=types.SimpleNamespace(enable_learning=False)
    )

    scheduler = _prepare_scheduler(monkeypatch, nick)

    assert isinstance(getattr(nick, "model_training_service", None), DummyService)
    assert events["auto"] == [False]
    assert events["disabled"] == 1
    assert events["enabled"] == 0

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_init_training_service_enables_capture_when_learning_on(monkeypatch):
    events = {"disabled": 0, "enabled": 0, "auto": []}

    import services.model_training_service as training_module

    class DummyService:
        def __init__(self, *_args, auto_subscribe=False, **_kwargs):
            events["auto"].append(auto_subscribe)
            if auto_subscribe:
                self.enable_workflow_capture()

        def disable_workflow_capture(self):
            events["disabled"] += 1

        def enable_workflow_capture(self):
            events["enabled"] += 1

        def dispatch_due_jobs(self):  # pragma: no cover - helper stub
            return []

    monkeypatch.setattr(training_module, "ModelTrainingService", DummyService)

    nick = types.SimpleNamespace(
        settings=types.SimpleNamespace(enable_learning=True)
    )

    scheduler = _prepare_scheduler(monkeypatch, nick)

    assert isinstance(getattr(nick, "model_training_service", None), DummyService)
    assert events["auto"] == [True]
    assert events["enabled"] >= 1
    assert events["disabled"] == 0

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None
