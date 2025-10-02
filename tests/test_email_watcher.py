
import io
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from services.email_watcher import (
    InMemoryEmailWatcherState,
    SESEmailWatcher,
    S3ObjectWatcher,
)


class StubSupplierInteractionAgent:
    RFQ_PATTERN = re.compile(r"RFQ-\d{8}-[a-f0-9]{8}", re.IGNORECASE)

    def __init__(self):
        self.contexts: List[AgentContext] = []

    def execute(self, context: AgentContext) -> AgentOutput:
        self.contexts.append(context)
        text = context.input_data.get("message", "")
        price_match = re.search(r"(\d+(?:\.\d+)?)", text)
        price = float(price_match.group(1)) if price_match else None
        data = {"price": price, "rfq_id": context.input_data.get("rfq_id")}
        next_agents: List[str] = []
        target = context.input_data.get("target_price")
        if target is not None and price is not None and price > target:
            next_agents = ["NegotiationAgent"]
        return AgentOutput(status=AgentStatus.SUCCESS, data=data, next_agents=next_agents)


class StubNegotiationAgent:
    def __init__(self):
        self.contexts: List[AgentContext] = []

    def execute(self, context: AgentContext) -> AgentOutput:
        self.contexts.append(context)
        return AgentOutput(status=AgentStatus.SUCCESS, data={"counter": context.input_data.get("target_price")})


class DummyNick:
    class _DummyCursor:
        def __init__(self, owner):
            self._owner = owner

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, *args, **kwargs):
            self._owner.ddl_statements.append(statement.strip())

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    class _DummyConn:
        def __init__(self, owner):
            self._owner = owner

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return DummyNick._DummyCursor(self._owner)

        def commit(self):
            pass

    def __init__(self):
        self.settings = SimpleNamespace(
            script_user="AgentNick",
            ses_default_sender="nicholasgeelen@procwise.co.uk",
            ses_smtp_endpoint="email-smtp.eu-west-1.amazonaws.com",
            ses_smtp_secret_name="ses/smtp/credentials",
            ses_region="eu-west-1",
            ses_inbound_bucket=None,
            ses_inbound_prefix=None,
            ses_inbound_s3_uri=None,
            s3_bucket_name=None,
            email_response_poll_seconds=1,
            email_inbound_initial_wait_seconds=0,
        )
        self.agents: Dict[str, object] = {}
        self.ddl_statements: List[str] = []
        self.s3_client = None

    def get_db_connection(self):
        return DummyNick._DummyConn(self)


def _make_watcher(nick, *, loader=None, state_store=None):
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    return SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        message_loader=loader,
        state_store=state_store or InMemoryEmailWatcherState(),
    )


def test_email_watcher_bootstraps_negotiation_tables():
    nick = DummyNick()
    _make_watcher(nick, loader=lambda limit=None: [])
    ddl = "".join(nick.ddl_statements)
    assert "CREATE TABLE IF NOT EXISTS proc.rfq_targets" in ddl
    assert "CREATE TABLE IF NOT EXISTS proc.negotiation_sessions" in ddl
    assert "CREATE TABLE IF NOT EXISTS proc.negotiation_session_state" in ddl
    assert "CREATE TABLE IF NOT EXISTS proc.processed_emails" in ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS processed_emails_bucket_key_etag_uidx" in ddl
    assert "CREATE INDEX IF NOT EXISTS negotiation_session_state_status_idx" in ddl



def test_poll_once_triggers_supplier_agent_on_match():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1500",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
            "target_price": 1000,
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)
    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(results) == 1
    result = results[0]
    assert result["rfq_id"].lower() == "rfq-20240101-abcd1234"
    assert watcher.supplier_agent.contexts
    assert "msg-1" in state


def test_poll_once_continues_until_all_filters_match():
    nick = DummyNick()
    messages = [
        {
            "id": "msg-1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Initial response 1100",
            "from": "supplier-a@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "SUP-1",
        },
        {
            "id": "msg-2",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Preferred supplier 900",
            "from": "supplier-b@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "SUP-2",
        },
    ]

    def loader(limit=None):
        return list(messages)

    watcher = _make_watcher(nick, loader=loader)

    results = watcher.poll_once(
        match_filters={"rfq_id": "RFQ-20240101-abcd1234", "supplier_id": "SUP-2"}
    )

    # RFQ match is authoritative, so the first candidate ends the poll
    assert [result["supplier_id"] for result in results] == ["SUP-1"]
    assert len(watcher.supplier_agent.contexts) == 1


def test_poll_once_matches_on_rfq_tail():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-tail",
            "subject": "Re: RFQ-20240101-00001234",
            "body": "Quoted price 1200",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-00001234",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)
    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20249999-00001234"})

    assert len(results) == 1
    assert results[0]["rfq_id"] == "RFQ-20240101-00001234"
    assert "msg-tail" in state


def test_normalise_rfq_value_uses_last_eight_digits():
    assert SESEmailWatcher._normalise_rfq_value("RFQ-20240101-ABCD1234") == "abcd1234"
    assert SESEmailWatcher._normalise_rfq_value("RFQ-20240101-00001234") == "00001234"


def test_poll_once_retries_until_target_found(monkeypatch):
    nick = DummyNick()
    batches = [
        [],
        [
            {
                "id": "msg-2",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Price 950",
                "from": "supplier@example.com",
                "rfq_id": "RFQ-20240101-abcd1234",
            }
        ],
    ]
    calls = {"count": 0}

    def loader(limit=None):
        calls["count"] += 1
        return batches.pop(0) if batches else []

    watcher = _make_watcher(nick, loader=loader)
    watcher.poll_interval_seconds = 0

    result = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(result) == 1
    assert result[0]["rfq_id"].lower() == "rfq-20240101-abcd1234"
    assert calls["count"] == 2


def test_poll_once_supports_like_filters():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-like",
            "subject": "Re: RFQ-20240101-abcd1234 Follow up",
            "body": "Quoted 1200",
            "from": "Muthu Subramanian <muthu.subramanian@dhsit.co.uk>",
            "rfq_id": "RFQ-20240101-ABCD1234",
            "supplier_id": "SUP-12345",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    results = watcher.poll_once(
        match_filters={
            "rfq_id_like": "rfq-20240101-abcd%",
            "from_address_like": "%@dhsit.co.uk>",
            "subject_like": "%follow up",
            "supplier_id_like": "sup-123",
        }
    )

    assert len(results) == 1
    assert results[0]["supplier_id"] == "SUP-12345"
    assert "msg-like" in state


def test_poll_once_matches_display_name_sender_with_plain_filter():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-display",
            "subject": "Re: RFQ-20240101-abcd1234 clarification",
            "body": "Quoted 1200",
            "from": "Muthu Subramanian <muthu.subramanian@dhsit.co.uk>",
            "rfq_id": "RFQ-20240101-ABCD1234",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    results = watcher.poll_once(
        match_filters={
            "rfq_id": "RFQ-20240101-abcd1234",
            "from_address": "muthu.subramanian@dhsit.co.uk",
        }
    )

    assert len(results) == 1
    assert results[0]["from_address"].startswith("Muthu Subramanian")
    assert "msg-display" in state


def test_poll_once_stops_future_polls_after_match():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()
    calls = {"count": 0}

    message = {
        "id": "msg-1",
        "subject": "Re: RFQ-20240101-abcd1234",
        "body": "Quoted price 1500",
        "from": "supplier@example.com",
        "rfq_id": "RFQ-20240101-ABCD1234",
    }

    def loader(limit=None):
        calls["count"] += 1
        return [message]

    watcher = _make_watcher(nick, loader=loader, state_store=state)
    watcher.poll_interval_seconds = 0

    first = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-ABCD1234"})
    assert first and first[0]["rfq_id"].lower() == "rfq-20240101-abcd1234"
    assert calls["count"] >= 1

    second = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-ABCD1234"})
    assert second == []
    assert calls["count"] > 1


def test_poll_once_without_filters_marks_match(caplog):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    message = {
        "id": "match-1",
        "subject": "Re: RFQ-20240101-abcd1234",
        "body": "Quoted price 1500",
        "from": "supplier@example.com",
        "rfq_id": "RFQ-20240101-abcd1234",
    }

    watcher = _make_watcher(nick, loader=lambda limit=None: [message], state_store=state)

    with caplog.at_level(logging.INFO):
        results = watcher.poll_once()

    assert len(results) == 1
    assert any(
        "matched=True" in record.getMessage()
        and "Completed scan" in record.getMessage()
        for record in caplog.records
    )


def test_poll_once_short_circuits_with_index(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick)

    hit = {
        "rfq_id": "RFQ-20240101-ABC12345",
        "key": "emails/RFQ-20240101-ABC12345/ingest/msg-1",
        "processed_at": "2024-01-01T00:00:00Z",
    }

    monkeypatch.setattr(watcher, "_lookup_rfq_hit_pg", lambda rfq: dict(hit))

    called = {"load": False}

    def fail_load(*args, **kwargs):
        called["load"] = True
        raise AssertionError("Should not load messages when index hit is present")

    monkeypatch.setattr(watcher, "_load_messages", fail_load)

    results = watcher.poll_once(match_filters={"rfq_id": hit["rfq_id"]})

    assert len(results) == 1
    assert results[0]["canonical_s3_key"] == hit["key"]
    assert results[0]["matched_via"] == "index"
    assert called["load"] is False


def test_watermark_persistence_across_runs():
    storage: Dict[tuple, tuple] = {}

    class WatermarkCursor:
        def __init__(self, connection):
            self.connection = connection
            self._result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            normalized = " ".join(statement.strip().lower().split())
            self.connection.owner.ddl_statements.append(statement.strip())
            if "select ts, key from proc.email_watcher_watermarks" in normalized:
                key = (params[0], params[1])
                self._result = self.connection.storage.get(key)
            elif "insert into proc.email_watcher_watermarks" in normalized:
                mailbox, prefix, ts_value, key_value = params
                self.connection.storage[(mailbox, prefix)] = (ts_value, key_value)
                self._result = None
            else:
                self._result = None

        def fetchone(self):
            return self._result

    class WatermarkConnection:
        def __init__(self, owner, storage):
            self.owner = owner
            self.storage = storage

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return WatermarkCursor(self)

        def commit(self):
            pass

        def rollback(self):
            pass

    class WatermarkNick(DummyNick):
        def __init__(self, backing):
            super().__init__()
            self._storage = backing

        def get_db_connection(self):
            return WatermarkConnection(self, self._storage)

    message_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    message = {
        "id": "wm-1",
        "subject": "Re: RFQ-20240101-ABCD1234",
        "body": "Quoted price 1500",
        "from": "supplier@example.com",
        "rfq_id": "RFQ-20240101-ABCD1234",
        "_last_modified": message_ts,
    }

    nick = WatermarkNick(storage)
    watcher = _make_watcher(nick, loader=lambda limit=None: [message])
    watcher.poll_once()

    key = (watcher.mailbox_address, watcher._prefixes[0])
    assert key in storage
    stored_ts, stored_key = storage[key]
    assert stored_ts == message_ts
    assert stored_key == "wm-1"

    nick_reload = WatermarkNick(storage)
    watcher_reload = _make_watcher(nick_reload, loader=lambda limit=None: [])
    assert watcher_reload._last_watermark_ts == message_ts
    assert watcher_reload._last_watermark_key == "wm-1"


def test_poll_once_repeats_scan_when_index_hit_reused(monkeypatch):
    nick = DummyNick()
    calls = {"count": 0}

    def loader(limit=None):
        calls["count"] += 1
        return []

    watcher = _make_watcher(nick, loader=loader)

    hit_payload = {
        "rfq_id": "RFQ-20240101-ABCD1234",
        "key": "emails/RFQ-20240101-ABCD1234/ingest/msg-1",
        "processed_at": "2024-01-01T12:00:00Z",
    }

    monkeypatch.setattr(watcher, "_lookup_rfq_hit_pg", lambda rfq: dict(hit_payload))

    first = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-ABCD1234"})
    assert first and first[0]["canonical_s3_key"] == hit_payload["key"]
    assert calls["count"] == 0

    second = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-ABCD1234"})
    assert second == []
    assert calls["count"] > 0


def test_poll_once_stops_after_match_without_extra_attempts():
    nick = DummyNick()
    calls = {"count": 0}

    def loader(limit=None):
        calls["count"] += 1
        return [
            {
                "id": "msg-1",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Price 875",
                "from": "supplier@example.com",
                "rfq_id": "RFQ-20240101-abcd1234",
            }
        ]

    watcher = _make_watcher(nick, loader=loader)
    watcher.poll_interval_seconds = 0
    watcher._match_poll_attempts = 5

    result = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(result) == 1
    assert calls["count"] == 1


def test_poll_once_returns_empty_after_attempts(monkeypatch):
    nick = DummyNick()
    calls = {"count": 0}

    def loader(limit=None):
        calls["count"] += 1
        return []

    watcher = _make_watcher(nick, loader=loader)

    watcher.poll_interval_seconds = 0
    watcher._match_poll_attempts = 2

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-missing"})

    assert results == []
    assert calls["count"] == 2


def test_poll_once_respects_dispatch_wait(monkeypatch):
    import services.email_watcher as email_watcher_module

    nick = DummyNick()
    nick.settings.email_inbound_initial_wait_seconds = 5

    fake_clock = {"now": 0.0}
    sleep_calls: List[float] = []

    def fake_time() -> float:
        return fake_clock["now"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(
        email_watcher_module,
        "time",
        SimpleNamespace(time=fake_time, sleep=fake_sleep),
    )

    watcher = _make_watcher(nick, loader=lambda limit=None: [])
    watcher.record_dispatch_timestamp()

    watcher.poll_interval_seconds = 1

    watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})
    assert sleep_calls[0] == pytest.approx(5.0)
    assert sleep_calls[1:] == [pytest.approx(1.0), pytest.approx(1.0)]

    sleep_calls.clear()
    watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})
    assert sleep_calls == [pytest.approx(1.0), pytest.approx(1.0)]


def test_s3_poll_prioritises_newest_objects(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick)
    watcher.bucket = "procwisemvp"
    watcher._prefixes = ["emails/"]
    watcher.poll_interval_seconds = 0

    now = datetime.now(timezone.utc)
    s3_objects = [
        {"Key": "emails/newest.eml", "LastModified": now},
        {"Key": "emails/matching.eml", "LastModified": now - timedelta(seconds=30)},
        {"Key": "emails/older.eml", "LastModified": now - timedelta(minutes=5)},
    ]

    raw_non_match = b"Subject: General update\nFrom: supplier@example.com\n\nNo RFQ reference here."
    raw_match = (
        b"Subject: Re: RFQ-20240101-abcd1234\n"
        b"From: supplier@example.com\n"
        b"Message-ID: <matching@example.com>\n\n"
        b"Quoted price 950"
    )
    raw_older = b"Subject: Re: RFQ-20231231-deadbeef\nFrom: supplier@example.com\n\nSome other offer"

    object_store = {
        "emails/newest.eml": raw_non_match,
        "emails/matching.eml": raw_match,
        "emails/older.eml": raw_older,
    }

    class DummyBody:
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)

        def read(self) -> bytes:
            return self._buffer.getvalue()

    expected_ingest_prefix = watcher._ensure_trailing_slash(
        "emails/RFQ-20240101-ABCD1234/ingest"
    )

    class DummyPaginator:
        def paginate(self, *, Bucket, Prefix):
            assert Bucket == watcher.bucket
            if Prefix == expected_ingest_prefix:
                return [{"Contents": []}]
            assert Prefix == watcher._prefixes[0]
            return [{"Contents": list(s3_objects)}]

    requested_keys: List[str] = []

    class DummyClient:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return DummyPaginator()

        def get_object(self, *, Bucket, Key):
            assert Bucket == watcher.bucket
            requested_keys.append(Key)
            return {"Body": DummyBody(object_store[Key])}

    monkeypatch.setattr(watcher, "_get_s3_client", lambda: DummyClient())

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(results) == 1
    assert results[0]["message_id"] == "emails/matching.eml"
    assert requested_keys == ["emails/newest.eml", "emails/matching.eml"]
    assert watcher._last_watermark_key == "emails/newest.eml"
    assert watcher._last_watermark_ts is not None


def test_scan_recent_objects_includes_backlog_without_watermark():
    nick = DummyNick()
    watcher = _make_watcher(nick, loader=lambda limit=None: [])

    prefix = watcher._prefixes[0]
    old_time = datetime.now(timezone.utc) - timedelta(minutes=120)
    contents = [
        {"Key": f"{prefix}msg-old", "LastModified": old_time, "ETag": '"etag-old"'},
    ]

    class DummyPaginator:
        def __init__(self, payload):
            self._payload = payload

        def paginate(self, **kwargs):
            yield {"Contents": list(self._payload)}

    class FakeClient:
        def __init__(self, payload):
            self._payload = payload

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return DummyPaginator(self._payload)

    fake_client = FakeClient(contents)
    watcher._s3_prefix_watchers[prefix] = S3ObjectWatcher(limit=8)

    refs_without_window = watcher._scan_recent_s3_objects(
        fake_client,
        [prefix],
        watermark_ts=None,
        watermark_key="",
        enforce_window=False,
    )

    assert refs_without_window == [
        (prefix, f"{prefix}msg-old", old_time, '"etag-old"')
    ]

    refs_with_window = watcher._scan_recent_s3_objects(
        fake_client,
        [prefix],
        watermark_ts=old_time,
        watermark_key="",
        enforce_window=True,
    )

    assert refs_with_window == []
