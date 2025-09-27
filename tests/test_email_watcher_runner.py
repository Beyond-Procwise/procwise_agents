from types import SimpleNamespace

from services.email_watcher_runner import (
    EmailWatcherStartupResult,
    preview_recent_emails,
    preview_then_watch,
)


class StubWatcher:
    def __init__(self, *, preview=None, processed=0):
        self.preview = preview or []
        self.processed = processed
        self.mailbox_address = "stub@example.com"
        self.peek_calls = []
        self.watch_calls = []

    def peek_recent_messages(self, limit):
        self.peek_calls.append(limit)
        return list(self.preview)

    def watch(self, *, interval=None, limit=None, stop_after=None):
        self.watch_calls.append((interval, limit, stop_after))
        return self.processed


def test_preview_recent_emails_trims_and_orders(caplog):
    watcher = StubWatcher(
        preview=[
            {"id": "oldest", "received_at": "Mon, 01 Jan 2024 12:00:00 +0000"},
            {"id": "newest", "received_at": "Wed, 03 Jan 2024 08:00:00 +0000"},
            {"id": "middle", "received_at": "Tue, 02 Jan 2024 09:30:00 +0000"},
            {"id": "stale"},
        ]
    )


    with caplog.at_level("INFO"):
        result = preview_recent_emails(watcher, limit=3)

    assert watcher.peek_calls == [3]
    assert [message["id"] for message in result] == ["newest", "middle", "oldest"]
    assert all(entry["id"] != "stale" for entry in result)
    assert any("Previewing" in record.message for record in caplog.records)


def test_preview_then_watch_runs_foreground_watch():
    watcher = StubWatcher(preview=[{"id": "a"}], processed=5)

    result = preview_then_watch(SimpleNamespace(), watcher=watcher, stop_after=1)

    assert isinstance(result, EmailWatcherStartupResult)
    assert result.preview == watcher.preview
    assert result.processed == 5
    assert watcher.watch_calls == [(None, None, 1)]
    assert result.background_thread is None


def test_preview_then_watch_background_starts_thread():
    watcher = StubWatcher(preview=[{"id": "b"}], processed=2)

    result = preview_then_watch(SimpleNamespace(), watcher=watcher, run_in_background=True, stop_after=1)

    assert result.processed is None
    assert result.background_thread is not None
    result.background_thread.join(timeout=1)
    assert watcher.watch_calls == [(None, None, 1)]
