import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.email_watcher_service as email_watcher_service


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
