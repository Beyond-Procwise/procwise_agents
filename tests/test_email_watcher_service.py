import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.email_watcher_service as email_watcher_service
from repositories import workflow_lifecycle_repo


def test_email_watcher_service_notify_triggers_runner(monkeypatch):
    calls = []

    def fake_load_active_workflow_ids():
        return []

    def fake_runner(**kwargs):
        calls.append(kwargs.get("workflow_id"))
        return {"status": "skipped"}

    monkeypatch.setattr(
        email_watcher_service.workflow_email_tracking_repo,
        "load_active_workflow_ids",
        fake_load_active_workflow_ids,
    )

    watcher = email_watcher_service.EmailWatcherService(
        poll_interval_seconds=60,
        post_dispatch_interval_seconds=60,
        dispatch_wait_seconds=0,
        watcher_runner=fake_runner,
    )

    watcher.start()
    try:
        watcher.notify_workflow("wf-priority")
        for _ in range(40):
            if calls:
                break
            time.sleep(0.05)
        assert "wf-priority" in calls
    finally:
        watcher.stop()


def test_email_watcher_service_should_skip_completed_negotiation():
    workflow_id = "wf-skip-completed"
    workflow_lifecycle_repo.init_schema()
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_negotiation_status(workflow_id, "completed")
    workflow_lifecycle_repo.record_watcher_event(
        workflow_id,
        "watcher_stopped",
        expected_responses=2,
        received_responses=1,
        metadata={
            "stop_reason": "negotiation_completed",
            "negotiation_completed": True,
            "status": "incomplete",
        },
    )

    watcher = email_watcher_service.EmailWatcherService(
        poll_interval_seconds=1,
        post_dispatch_interval_seconds=1,
        dispatch_wait_seconds=0,
    )

    assert watcher._should_skip_workflow(workflow_id) is True


def test_email_watcher_service_does_not_skip_active_negotiation():
    workflow_id = "wf-skip-active"
    workflow_lifecycle_repo.init_schema()
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_negotiation_status(workflow_id, "running")
    workflow_lifecycle_repo.record_watcher_event(
        workflow_id,
        "watcher_stopped",
        metadata={"stop_reason": "negotiation_completed"},
    )

    watcher = email_watcher_service.EmailWatcherService(
        poll_interval_seconds=1,
        post_dispatch_interval_seconds=1,
        dispatch_wait_seconds=0,
    )

    assert watcher._should_skip_workflow(workflow_id) is False


def test_email_watcher_service_loop_skips_completed_workflow(monkeypatch):
    calls = []

    def fake_load_active_workflow_ids():
        return ["wf-loop-skip"]

    def fake_runner(**kwargs):
        calls.append(kwargs.get("workflow_id"))
        return {"status": "skipped"}

    monkeypatch.setattr(
        email_watcher_service.workflow_email_tracking_repo,
        "load_active_workflow_ids",
        fake_load_active_workflow_ids,
    )
    monkeypatch.setattr(
        email_watcher_service.EmailWatcherService,
        "_should_skip_workflow",
        lambda self, workflow_id: True,
    )

    def fast_wait(self, seconds):
        self._stop_event.set()

    monkeypatch.setattr(
        email_watcher_service.EmailWatcherService,
        "_wait_for_next_cycle",
        fast_wait,
        raising=False,
    )

    watcher = email_watcher_service.EmailWatcherService(
        poll_interval_seconds=1,
        post_dispatch_interval_seconds=1,
        dispatch_wait_seconds=0,
        watcher_runner=fake_runner,
    )

    watcher.start()
    try:
        time.sleep(0.05)
        assert calls == []
    finally:
        watcher.stop()
