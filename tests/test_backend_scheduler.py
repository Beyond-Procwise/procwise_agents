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
