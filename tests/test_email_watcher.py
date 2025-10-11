
import io
import json
import logging
import re
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from services.email_sqs_loader import sqs_email_loader
from services.email_watcher import (
    InMemoryEmailWatcherState,
    SESEmailWatcher,
    S3ObjectWatcher,
)


class StubSupplierInteractionAgent:
    RFQ_PATTERN = re.compile(r"RFQ-\d{8}-[A-Za-z0-9]{8}", re.IGNORECASE)

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
        self._lock = threading.Lock()

    def execute(self, context: AgentContext) -> AgentOutput:
        with self._lock:
            self.contexts.append(context)
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={"counter": context.input_data.get("target_price"), "round": context.input_data.get("round")},
        )


class StubQuoteComparisonAgent:
    def __init__(self):
        self.contexts: List[AgentContext] = []

    def execute(self, context: AgentContext) -> AgentOutput:
        self.contexts.append(context)
        return AgentOutput(status=AgentStatus.SUCCESS, data={})


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
            email_inbound_post_dispatch_delay_seconds=0,
        )
        self.agents: Dict[str, object] = {}
        self.ddl_statements: List[str] = []
        self.s3_client = None

    def get_db_connection(self):
        return DummyNick._DummyConn(self)


def _make_watcher(nick, *, loader=None, state_store=None):
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    quote_agent = StubQuoteComparisonAgent()
    return SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        message_loader=loader,
        state_store=state_store or InMemoryEmailWatcherState(),
        quote_comparison_agent=quote_agent,
    )


@pytest.fixture
def similarity_dispatch_row():
    now = datetime.now(timezone.utc)
    return {
        "rfq_id": "RFQ-20240215-SIM12345",
        "supplier_id": "supplier-9001",
        "supplier_name": "Industrial Bolts Ltd",
        "subject": "RFQ request: Immediate quote for flange bolts",
        "body": "Please provide your immediate quote for flange bolts and lead times.",
        "recipient_email": "sales@acme-industrial.com",
        "sender": "buyer@procwise.test",
        "sent_on": now - timedelta(minutes=30),
        "created_on": now - timedelta(hours=1),
        "updated_on": now - timedelta(minutes=5),
    }


def test_email_watcher_bootstraps_negotiation_tables():
    nick = DummyNick()
    _make_watcher(nick, loader=lambda limit=None: [])
    ddl = "".join(nick.ddl_statements)
    assert "CREATE TABLE IF NOT EXISTS proc.rfq_targets" in ddl
    assert "CREATE TABLE IF NOT EXISTS proc.negotiation_sessions" in ddl
    assert "CREATE TABLE IF NOT EXISTS proc.negotiation_session_state" in ddl
    assert "CREATE TABLE IF NOT EXISTS proc.processed_emails" in ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS processed_emails_bucket_key_etag_uidx" in ddl
    assert "ADD COLUMN IF NOT EXISTS mailbox TEXT" in ddl
    assert "ADD COLUMN IF NOT EXISTS message_id TEXT" in ddl
    assert "CREATE INDEX IF NOT EXISTS processed_emails_message_id_idx" in ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS processed_emails_mailbox_message_id_uidx" in ddl
    assert "CREATE INDEX IF NOT EXISTS negotiation_session_state_status_idx" in ddl


def test_extract_dispatch_entries_handles_email_dispatch_payload():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    payload = {
        "status": "completed",
        "result": {
            "rfq_id": "RFQ-20240601-ABC12345",
            "draft": {
                "id": 501,
                "supplier_id": "SUP-001",
                "recipients": ["one@example.com"],
            },
            "dispatches": [
                {
                    "draft": {
                        "draft_record_id": 777,
                        "supplier_id": "SUP-002",
                        "recipients": ["two@example.com"],
                    }
                },
                {
                    "draft": {
                        "draft_record_id": 778,
                        "supplier_id": "SUP-003",
                        "recipients": ["three@example.com"],
                    }
                },
            ],
        },
    }

    entries = watcher._extract_dispatch_entries(payload)

    assert {entry.get("supplier_id") for entry in entries} == {
        "SUP-001",
        "SUP-002",
        "SUP-003",
    }
    assert {entry.get("id") for entry in entries} == {501, None}
    assert {entry.get("draft_record_id") for entry in entries} == {None, 777, 778}


def test_build_dispatch_expectation_from_dispatch_entries():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    entries = [
        {"draft_record_id": 101, "supplier_id": "SUP-100"},
        {"id": 102, "supplier_id": "SUP-200"},
        {"supplier_id": "SUP-300", "rfq_id": "RFQ-XYZ", "recipients": ["a@ex.com"]},
    ]

    expectation = watcher._build_dispatch_expectation("action-999", "workflow-555", entries)

    assert expectation is not None
    assert expectation.action_id == "action-999"
    assert expectation.workflow_id == "workflow-555"
    assert expectation.draft_count == 3
    assert expectation.draft_ids == (101, 102)
    assert expectation.supplier_count == 3



def test_extract_dispatch_entries_ignores_unsent_or_failed_dispatches():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    payload = {
        "result": {
            "dispatches": [
                {
                    "draft": {
                        "draft_record_id": 501,
                        "supplier_id": "SUP-SENT",
                        "recipients": ["sent@example.com"],
                        "sent_status": True,
                    }
                },
                {
                    "draft": {
                        "draft_record_id": 502,
                        "supplier_id": "SUP-FAILED",
                        "recipients": ["failed@example.com"],
                        "sent_status": False,
                    }
                },
                {
                    "draft": {
                        "draft_record_id": 503,
                        "supplier_id": "SUP-PENDING",
                        "recipients": ["pending@example.com"],
                        "metadata": {"status": "pending"},
                    }
                },
                {
                    "draft": {
                        "draft_record_id": 504,
                        "supplier_id": "SUP-ERROR",
                        "recipients": ["error@example.com"],
                        "metadata": {"status": "failed"},
                    }
                },
                {
                    "draft": {
                        "draft_record_id": 505,
                        "supplier_id": "SUP-SUCCESS",
                        "recipients": ["ok@example.com"],
                        "metadata": {"status": "success"},
                    }
                },
            ]
        }
    }

    entries = watcher._extract_dispatch_entries(payload)

    suppliers = {entry.get("supplier_id") for entry in entries}
    assert suppliers == {"SUP-SENT", "SUP-SUCCESS"}

    expectation = watcher._build_dispatch_expectation(
        "action-ok",
        "workflow-ok",
        entries,
    )

    assert expectation is not None
    assert expectation.draft_count == 2
    assert set(expectation.draft_ids) == {501, 505}


def test_extract_dispatch_entries_ignores_draft_metadata_records():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    payload = {
        "result": {
            "draft": {
                "id": 900,
                "supplier_id": "SUP-META",
                "recipients": ["meta@example.com"],
                "metadata": {
                    "supplier_id": "SUP-META",
                    "rfq_id": "RFQ-META",
                },
            }
        }
    }

    entries = watcher._extract_dispatch_entries(payload)

    assert entries == [
        {
            "id": 900,
            "supplier_id": "SUP-META",
            "recipients": ["meta@example.com"],
            "metadata": {
                "supplier_id": "SUP-META",
                "rfq_id": "RFQ-META",
            },
        }
    ]


def test_email_watcher_watch_respects_limit(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick)

    batches = [
        [{"message_id": "msg-1"}],
        [{"message_id": "msg-2"}],
        [{"message_id": "msg-3"}],
    ]

    limits_seen: List[Optional[int]] = []

    def fake_poll(limit=None):
        limits_seen.append(limit)
        return batches.pop(0) if batches else []

    monkeypatch.setattr(watcher, "poll_once", fake_poll)

    processed = watcher.watch(limit=3, timeout_seconds=5)

    assert processed == 3
    assert limits_seen[:3] == [3, 2, 1]


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
            "supplier_id": "S1",
            "dispatch_run_id": "run-001",
            "target_price": 1000,
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)
    results = watcher.poll_once(
        match_filters={"supplier_id": "S1", "dispatch_run_id": "run-001"}
    )

    assert len(results) == 1
    result = results[0]
    assert result["rfq_id"].lower() == "rfq-20240101-abcd1234"
    assert watcher.supplier_agent.contexts
    assert "msg-1" in state


def test_poll_once_requires_supplier_match_when_filters_present():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-2",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1400",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "SUP-MISMATCH",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)
    results = watcher.poll_once(
        match_filters={"supplier_id": "SUP-EXPECTED", "dispatch_run_id": "run-202"}
    )

    assert results == []


def test_poll_once_matches_with_recent_supplier_when_run_diff(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-3",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1350",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "S1",
            "dispatch_run_id": "run-303",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)
    draft = watcher._DraftSnapshot(
        id=99,
        rfq_id="RFQ-20240101-ABCD1234",
        subject="RFQ message",
        body="Thanks",
        dispatch_token="run-expected",
        run_id="run-expected",
        recipients=("supplier@example.com",),
        supplier_id="S1",
    )

    monkeypatch.setattr(
        watcher,
        "_load_recent_drafts_for_fallback",
        lambda supplier_filter, limit=10, within_minutes=10: [draft],
    )

    results = watcher.poll_once(
        match_filters={"supplier_id": "S1", "dispatch_run_id": "run-404"}
    )

    assert len(results) == 1
    processed = results[0]
    assert processed.get("matched_via") == "fallback"
    assert processed.get("match_score") == 0.5
    assert processed.get("supplier_id") == "S1"
    assert processed.get("dispatch_run_id") == "run-303"


def test_poll_once_uses_draft_fallback_when_run_missing(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-fallback",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1400",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-ABCD1234",
            "supplier_id": "SUP-EXPECTED",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    draft = watcher._DraftSnapshot(
        id=301,
        rfq_id="RFQ-20240101-ABCD1234",
        subject="Re: RFQ-20240101-ABCD1234",
        body="Thank you",
        dispatch_token="run-202",
        run_id="run-202",
        recipients=("supplier@example.com",),
        supplier_id="SUP-EXPECTED",
    )

    monkeypatch.setattr(
        watcher,
        "_load_recent_drafts_for_fallback",
        lambda supplier_filter, limit=10, within_minutes=10: [draft],
    )

    results = watcher.poll_once(
        match_filters={"supplier_id": "SUP-EXPECTED", "dispatch_run_id": "run-202"}
    )

    assert len(results) == 1
    processed = results[0]
    assert processed["dispatch_run_id"] == "run-202"
    assert processed.get("matched_via") == "fallback"
    assert processed.get("match_score") == 0.5


def test_load_recent_drafts_filters_by_supplier_and_time():
    nick = DummyNick()
    watcher = _make_watcher(nick, loader=lambda limit=None: [])

    captured: Dict[str, object] = {}

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params):
            captured["statement"] = statement.strip()
            captured["params"] = params

        def fetchall(self):
            return [
                (
                    501,
                    "RFQ-20240101-ABCD1234",
                    "SUPPLIER-EXPECTED",
                    "Subject",
                    "Body",
                    None,
                )
            ]

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return _Cursor()

    watcher.agent_nick.get_db_connection = lambda: _Conn()

    snapshots = watcher._load_recent_drafts_for_fallback(
        "supplier-expected", limit=5, within_minutes=10
    )

    assert snapshots and snapshots[0].supplier_id == "SUPPLIER-EXPECTED"
    assert "CURRENT_TIMESTAMP - (%s * INTERVAL '1 minute')" in captured.get("statement", "")
    assert captured.get("params") == (10, "supplier-expected", 5)


def test_register_processed_response_waits_for_complete_run(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick, loader=lambda limit=None: [])

    expectation_calls: List[Optional[str]] = []

    def fake_expectations(workflow_id, metadata, *, group_key=None):
        expectation_calls.append(group_key)
        return 2

    monkeypatch.setattr(
        watcher,
        "_ensure_workflow_expectations",
        fake_expectations,
        raising=False,
    )

    tracking_key = watcher._normalise_group_key("run-500", "wf-500")

    metadata = {"dispatch_run_id": "run-500", "status": "processed", "payload": {}}
    processed_one = {"message_id": "msg-run-1", "dispatch_run_id": "run-500"}
    processed_two = {"message_id": "msg-run-2", "dispatch_run_id": "run-500"}

    result_one = watcher._register_processed_response("wf-500", metadata, processed_one, None)
    assert result_one == (False, None)
    assert watcher._workflow_processed_counts.get(tracking_key) == 1

    result_two = watcher._register_processed_response("wf-500", metadata, processed_two, None)
    assert result_two == (False, None)
    assert tracking_key not in watcher._workflow_processed_counts
    assert expectation_calls and expectation_calls[0] == tracking_key
    assert not watcher.supplier_agent.contexts


def test_email_watcher_resolves_missing_rfq_via_thread_map(monkeypatch):
    nick = DummyNick()

    message = {
        "id": "msg-thread-1",
        "subject": "Supplier response",
        "body": "Here is our quote 1200",
        "from": "supplier@example.com",
        "in_reply_to": "<dispatch-001@example.com>",
        "rfq_id": None,
    }

    lookup_calls: List[Tuple[str, List[str]]] = []
    ensure_calls: List[str] = []

    def fake_lookup(conn, table_name, identifiers, logger=None):
        lookup_calls.append((table_name, list(identifiers)))
        return "RFQ-20240101-THREAD42"

    def fake_ensure(conn, table_name, logger=None):
        ensure_calls.append(table_name)

    monkeypatch.setattr("services.email_watcher.lookup_rfq_from_threads", fake_lookup)
    monkeypatch.setattr("services.email_watcher.ensure_thread_table", fake_ensure)

    watcher = _make_watcher(nick, loader=lambda limit=None: [dict(message)])
    results = watcher.poll_once()

    assert len(results) == 1
    processed = results[0]
    assert processed["rfq_id"] == "RFQ-20240101-THREAD42"
    assert lookup_calls
    assert lookup_calls[0][0] == "proc.email_thread_map"
    assert lookup_calls[0][1] == ["dispatch-001@example.com"]
    assert ensure_calls == ["proc.email_thread_map"]


def test_email_watcher_resolves_missing_rfq_from_dispatch_history(monkeypatch):
    nick = DummyNick()

    dispatch_row = {
        "rfq_id": "RFQ-20240201-XYZ12345",
        "supplier_id": "supplier-42",
        "supplier_name": "Acme Supplies",
        "sent_on": datetime.now(timezone.utc) - timedelta(minutes=5),
        "created_on": datetime.now(timezone.utc) - timedelta(minutes=10),
        "updated_on": datetime.now(timezone.utc) - timedelta(minutes=2),
    }

    class DispatchCursor:
        def __init__(self, owner):
            self._owner = owner
            self._result: List[Tuple] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            normalized = " ".join(statement.split()).lower()
            if normalized.startswith(
                "select id, rfq_id, supplier_id, subject, body, payload from proc.draft_rfq_emails"
            ):
                self._result = [
                    (
                        101,
                        dispatch_row["rfq_id"],
                        dispatch_row["supplier_id"],
                        "Re: Pricing",
                        "Here is our latest quote",
                        json.dumps({}),
                    )
                ]
            elif normalized.startswith(
                "select rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on, payload from proc.draft_rfq_emails"
            ):
                self._result = [
                    (
                        dispatch_row["rfq_id"],
                        dispatch_row["supplier_id"],
                        dispatch_row["supplier_name"],
                        dispatch_row["sent_on"],
                        True,
                        dispatch_row["created_on"],
                        dispatch_row["updated_on"],
                        json.dumps({}),
                    )
                ]
            elif normalized.startswith(
                "select supplier_id, supplier_name, rfq_id, sent_on, sent, created_on, updated_on, payload from proc.draft_rfq_emails"
            ):
                self._result = [
                    (
                        dispatch_row["supplier_id"],
                        dispatch_row["supplier_name"],
                        dispatch_row["rfq_id"],
                        dispatch_row["sent_on"],
                        True,
                        dispatch_row["created_on"],
                        dispatch_row["updated_on"],
                        json.dumps({}),
                    )
                ]
            elif normalized.startswith(
                "select rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on from proc.draft_rfq_emails"
            ):
                self._result = [
                    (
                        dispatch_row["rfq_id"],
                        dispatch_row["supplier_id"],
                        dispatch_row["supplier_name"],
                        dispatch_row["sent_on"],
                        True,
                        dispatch_row["created_on"],
                        dispatch_row["updated_on"],
                    )
                ]
            elif normalized.startswith("select target_price"):
                self._result = []
            else:
                self._owner.ddl_statements.append(statement.strip())
                self._result = []

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result)

    class DispatchConn:
        def __init__(self, owner):
            self._owner = owner

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return DispatchCursor(self._owner)

        def commit(self):
            pass

    nick.get_db_connection = lambda: DispatchConn(nick)

    captured_mark = {}

    def fake_mark_response(connection, **kwargs):
        captured_mark.update(kwargs)
        return True

    monkeypatch.setattr(
        "services.email_watcher.mark_dispatch_response",
        fake_mark_response,
    )

    message = {
        "id": "msg-dispatch-1",
        "subject": "Re: Pricing",
        "body": "Here is our latest quote",
        "from": "Acme Rep <buyer@acme.test>",
        "rfq_id": None,
    }

    watcher = _make_watcher(nick, loader=lambda limit=None: [dict(message)])

    results = watcher.poll_once()

    assert len(results) == 1
    processed = results[0]
    assert processed["rfq_id"] == dispatch_row["rfq_id"]
    assert processed["supplier_id"] == dispatch_row["supplier_id"]
    assert processed["matched_via"] == "fallback"
    assert processed["match_score"] == 0.5
    assert watcher.supplier_agent.contexts
    context = watcher.supplier_agent.contexts[0]
    assert context.input_data["rfq_id"] == dispatch_row["rfq_id"]
    assert context.input_data["supplier_id"] == dispatch_row["supplier_id"]
    assert captured_mark["rfq_id"] == dispatch_row["rfq_id"]


def test_email_watcher_resolves_rfq_via_dispatch_similarity(
    monkeypatch, similarity_dispatch_row
):
    nick = DummyNick()

    row = similarity_dispatch_row

    class SimilarityCursor:
        def __init__(self, owner):
            self._owner = owner
            self._result: List[Tuple] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            normalized = " ".join(statement.split()).lower()
            if normalized.startswith(
                "select rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on from proc.draft_rfq_emails"
            ):
                self._result = []
            elif normalized.startswith(
                "select rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on, payload from proc.draft_rfq_emails"
            ):
                payload = json.dumps(
                    {
                        "subject": row["subject"],
                        "body": row["body"],
                        "recipients": [row["recipient_email"]],
                    }
                )
                self._result = [
                    (
                        row["rfq_id"],
                        row["supplier_id"],
                        row["supplier_name"],
                        row["sent_on"],
                        True,
                        row["created_on"],
                        row["updated_on"],
                        payload,
                    )
                ]
            elif normalized.startswith(
                "select rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on, subject, body, recipient_email, sender from proc.draft_rfq_emails"
            ):
                self._result = [
                    (
                        row["rfq_id"],
                        row["supplier_id"],
                        row["supplier_name"],
                        row["sent_on"],
                        True,
                        row["created_on"],
                        row["updated_on"],
                        row["subject"],
                        row["body"],
                        row["recipient_email"],
                        row["sender"],
                    )
                ]
            elif normalized.startswith(
                "select supplier_id, supplier_name, rfq_id, sent_on, sent, created_on, updated_on from proc.draft_rfq_emails"
            ):
                self._result = [
                    (
                        row["supplier_id"],
                        row["supplier_name"],
                        row["rfq_id"],
                        row["sent_on"],
                        True,
                        row["created_on"],
                        row["updated_on"],
                    )
                ]
            elif normalized.startswith("select target_price"):
                self._result = []
            else:
                self._owner.ddl_statements.append(statement.strip())
                self._result = []

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result)

    class SimilarityConn:
        def __init__(self, owner):
            self._owner = owner

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return SimilarityCursor(self._owner)

        def commit(self):
            pass

    nick.get_db_connection = lambda: SimilarityConn(nick)

    captured_mark = {}

    def fake_mark_response(connection, **kwargs):
        captured_mark.update(kwargs)
        return True

    monkeypatch.setattr(
        "services.email_watcher.mark_dispatch_response",
        fake_mark_response,
    )

    message = {
        "id": "msg-similarity-1",
        "subject": "Re: Immediate quote for flange bolts",
        "body": "Hello, our quote for flange bolts is 1250.",
        "from": "Quotes <sales@acme-industrial.com>",
        "rfq_id": None,
    }

    watcher = _make_watcher(nick, loader=lambda limit=None: [dict(message)])

    results = watcher.poll_once()

    assert len(results) == 1
    processed = results[0]
    assert processed["rfq_id"] == row["rfq_id"]
    assert processed["supplier_id"] == row["supplier_id"]
    assert processed["matched_via"] == "fallback"
    assert processed["match_score"] == 0.5
    assert watcher.supplier_agent.contexts
    context = watcher.supplier_agent.contexts[0]
    assert context.input_data["rfq_id"] == row["rfq_id"]
    assert context.input_data["supplier_id"] == row["supplier_id"]
    assert captured_mark["rfq_id"] == row["rfq_id"]

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

    assert len(results) == 1
    matched = results[0]
    assert matched["supplier_id"] == "SUP-2"
    assert matched["rfq_id"].upper() == "RFQ-20240101-ABCD1234"
    assert watcher.supplier_agent.contexts
    assert watcher.supplier_agent.contexts[0].input_data["supplier_id"] == "SUP-2"


def test_poll_once_expands_loader_limit_for_supplier_filters(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick)
    watcher._last_watermark_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    initial_watermark = watcher._last_watermark_ts

    captured_limits: List[Optional[int]] = []
    captured_since: List[Optional[datetime]] = []

    messages = [
        {
            "id": "msg-1",
            "supplier_id": "SUP-1",
            "_last_modified": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
        {
            "id": "msg-2",
            "supplier_id": "SUP-2",
            "_last_modified": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
    ]

    def fake_process(
        self,
        message,
        *,
        match_filters,
        target_rfq_normalised,
        results,
        effective_limit,
    ):
        matched = message.get("supplier_id") == match_filters.get("supplier_id")
        if matched:
            results.append({"message_id": message.get("id"), "supplier_id": message.get("supplier_id")})
        return matched, False, False, True

    def fake_load_messages(
        limit,
        *,
        mark_seen,
        prefixes,
        since,
        on_message,
    ):
        captured_limits.append(limit)
        captured_since.append(since)
        for payload in messages:
            if on_message is not None:
                on_message(payload, payload.get("_last_modified"))
        return []

    monkeypatch.setattr(watcher, "_process_candidate_message", fake_process.__get__(watcher, type(watcher)))
    monkeypatch.setattr(watcher, "_load_messages", fake_load_messages)

    results = watcher.poll_once(match_filters={"supplier_id": "SUP-2"})

    assert [payload["supplier_id"] for payload in results] == ["SUP-2"]
    assert captured_limits == [None]
    assert captured_since == [initial_watermark]


def test_poll_once_processes_all_matches_even_when_limit_reached():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1000",
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "SUP-1",
            "from": "supplier@example.com",
        },
        {
            "id": "msg-2",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Updated quote 950",
            "rfq_id": "RFQ-20240101-abcd1234",
            "supplier_id": "SUP-1",
            "from": "supplier@example.com",
        },
    ]

    def loader(limit=None):
        return list(messages)

    watcher = _make_watcher(nick, loader=loader, state_store=state)

    results = watcher.poll_once(
        limit=1,
        match_filters={"rfq_id": "RFQ-20240101-abcd1234", "supplier_id": "SUP-1"},
    )

    assert len(results) == 1
    context_ids = {context.input_data["message_id"] for context in watcher.supplier_agent.contexts}
    assert context_ids == {"msg-1", "msg-2"}

    seen_entries = dict(state.items())
    assert set(seen_entries) == {"msg-1", "msg-2"}
    assert all(entry.get("status") == "processed" for entry in seen_entries.values())


def test_poll_once_defers_watermark_until_batch_complete(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick)

    messages = [
        {
            "id": "msg-1",
            "supplier_id": "SUP-1",
            "_last_modified": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
        {
            "id": "msg-2",
            "supplier_id": "SUP-2",
            "_last_modified": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
    ]

    def fake_process(
        self,
        message,
        *,
        match_filters,
        target_rfq_normalised,
        results,
        effective_limit,
    ):
        matched = message.get("supplier_id") == match_filters.get("supplier_id")
        if matched:
            results.append({"message_id": message.get("id"), "supplier_id": message.get("supplier_id")})
        return matched, False, False, True

    calls: List[Tuple[datetime, str]] = []
    original_update = watcher._update_watermark

    def capture_update(last_modified, key):
        calls.append((last_modified, key))
        original_update(last_modified, key)

    def fake_loader(limit=None):
        return list(messages)

    monkeypatch.setattr(watcher, "_process_candidate_message", fake_process.__get__(watcher, type(watcher)))
    monkeypatch.setattr(watcher, "_custom_loader", fake_loader)
    monkeypatch.setattr(watcher, "_update_watermark", capture_update)

    results = watcher.poll_once(match_filters={"supplier_id": "SUP-2"})

    assert [payload["supplier_id"] for payload in results] == ["SUP-2"]
    assert calls
    assert len(calls) == 1
    last_modified, key = calls[0]
    assert isinstance(last_modified, datetime)
    assert key == "msg-2"


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
    assert results[0]["matched_via"] != "rfq_hint"


def test_match_dispatched_message_prefers_run_id():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    drafts = [
        watcher._DraftSnapshot(
            id=201,
            rfq_id="RFQ-1",
            subject="Subject",
            body="Body A",
            dispatch_token="token-a",
            run_id="run-a",
        ),
        watcher._DraftSnapshot(
            id=202,
            rfq_id="RFQ-2",
            subject="Subject",
            body="Body B",
            dispatch_token="token-b",
            run_id="run-b",
        ),
    ]

    message = {
        "subject": "Subject",
        "body": "<!-- PROCWISE:RFQ_ID=RFQ-2;TOKEN=token-b;RUN_ID=run-b -->\nBody B",
    }

    match = watcher._match_dispatched_message(message, drafts)

    assert match is drafts[1]
    assert match.matched_via == "dispatch_token"


def test_compose_imap_search_criteria_includes_since():
    ts = datetime(2024, 2, 15, 9, 30, tzinfo=timezone.utc)
    criteria = SESEmailWatcher._compose_imap_search_criteria("ALL", ts)
    assert criteria == "(UNSEEN) SINCE 15-Feb-2024"

    custom = SESEmailWatcher._compose_imap_search_criteria('UNSEEN FROM "foo"', None)
    assert custom == 'UNSEEN FROM "foo"'


def test_email_watcher_falls_back_to_imap(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = None
    watcher._last_watermark_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    class StubIMAP:
        last_search: Optional[str] = None

        def __init__(self, host, port=None):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.logout()
            return False

        def login(self, user, password):
            self.user = user
            self.password = password

        def select(self, mailbox):
            self.mailbox = mailbox
            return "OK", [b""]

        def search(self, charset, criteria):
            StubIMAP.last_search = criteria
            return "OK", [b"1"]

        def fetch(self, msg_id, command):
            message = EmailMessage()
            message["Subject"] = "Re: RFQ-20240101-abcd1234"
            message["From"] = "supplier@example.com"
            message["Message-ID"] = "<imap-msg-1>"
            message["Date"] = "Mon, 01 Jan 2024 10:00:00 +0000"
            message.set_content("Quoted price 1200")
            return "OK", [(b"1", message.as_bytes())]

        def store(self, msg_id, flags, values):
            self.stored = (msg_id, flags, values)

        def logout(self):
            return "BYE", []

    monkeypatch.setattr("services.email_watcher.imaplib.IMAP4_SSL", StubIMAP)
    monkeypatch.setattr(SESEmailWatcher, "_load_from_s3", lambda self, *args, **kwargs: [])

    results = watcher.poll_once(limit=1)

    assert results
    payload = results[0]
    assert payload["message_id"] in {"<imap-msg-1>", "imap/1"}
    assert payload["subject"] == "Re: RFQ-20240101-abcd1234"
    assert StubIMAP.last_search is not None
    assert "UNSEEN" in StubIMAP.last_search
    assert "SINCE 01-Jan-2024" in StubIMAP.last_search


def test_imap_primary_prefers_imap_over_s3(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = "procwise-bucket"

    counters = {"s3": 0, "imap": 0}

    def _stub_s3(self, limit, *, prefixes=None, on_message=None):
        counters["s3"] += 1
        return []

    def _stub_imap(self, limit, *, mark_seen, since=None, on_message=None):
        counters["imap"] += 1
        return [
            {
                "id": f"imap-msg-{counters['imap']}",
                "message_id": f"imap-msg-{counters['imap']}",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Quoted price 1250",
                "rfq_id": "RFQ-20240101-abcd1234",
                "from": "supplier@example.com",
                "from_address": "supplier@example.com",
            }
        ]

    monkeypatch.setattr(SESEmailWatcher, "_load_from_s3", _stub_s3, raising=False)
    monkeypatch.setattr(SESEmailWatcher, "_load_from_imap", _stub_imap, raising=False)

    batch = watcher.poll_once(limit=1)

    assert counters["imap"] == 1
    assert counters["s3"] == 0
    assert batch
    assert batch[0]["message_id"].startswith("imap-msg-")
    assert watcher._last_candidate_source == "imap"


def test_imap_falls_back_to_s3_after_three_empty_polls(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = "procwise-bucket"
    watcher._imap_fallback_attempts = 3

    counters = {"s3": 0, "imap": 0}

    def _stub_s3(self, limit, *, prefixes=None, on_message=None):
        counters["s3"] += 1
        return [
            {
                "id": "s3-msg-1",
                "message_id": "s3-msg-1",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Quoted price 1250",
                "rfq_id": "RFQ-20240101-abcd1234",
                "from": "supplier@example.com",
                "from_address": "supplier@example.com",
            }
        ]

    def _stub_imap(self, limit, *, mark_seen, since=None, on_message=None):
        counters["imap"] += 1
        return []

    monkeypatch.setattr(SESEmailWatcher, "_load_from_s3", _stub_s3, raising=False)
    monkeypatch.setattr(SESEmailWatcher, "_load_from_imap", _stub_imap, raising=False)

    batch_one = watcher.poll_once(limit=1)
    assert batch_one == []
    assert counters["imap"] == 1
    assert counters["s3"] == 0
    assert watcher._last_candidate_source == "imap"

    batch_two = watcher.poll_once(limit=1)
    assert batch_two == []
    assert counters["imap"] == 2
    assert counters["s3"] == 0
    assert watcher._last_candidate_source == "imap"

    batch_three = watcher.poll_once(limit=1)
    assert counters["imap"] == 3
    assert counters["s3"] == 1
    assert batch_three
    assert batch_three[0]["message_id"].startswith("s3-msg")
    assert watcher._last_candidate_source == "s3"


def test_imap_loader_records_processed_email(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = None

    class StubIMAP:
        def __init__(self, host, port=None):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.logout()
            return False

        def login(self, user, password):
            self.user = user
            self.password = password

        def select(self, mailbox):
            self.mailbox = mailbox
            return "OK", [b""]

        def search(self, charset, criteria):
            return "OK", [b"1"]

        def fetch(self, msg_id, command):
            message = EmailMessage()
            message["Subject"] = "Re: RFQ-20240101-abcd1234"
            message["From"] = "supplier@example.com"
            message["Message-ID"] = "<imap-msg-42>"
            message["Date"] = "Mon, 01 Jan 2024 10:00:00 +0000"
            message.set_content("Quoted price 1150 for RFQ-20240101-ABCD1234")
            return "OK", [(b"1", message.as_bytes())]

        def store(self, msg_id, flags, values):
            self.stored = (msg_id, flags, values)

        def logout(self):
            return "BYE", []

    monkeypatch.setattr("services.email_watcher.imaplib.IMAP4_SSL", StubIMAP)
    monkeypatch.setattr(SESEmailWatcher, "_load_from_s3", lambda self, *args, **kwargs: [])

    recorded: List[tuple] = []

    def _capture_registry(
        self,
        bucket,
        key,
        etag,
        rfq_id,
        *,
        message_id=None,
        mailbox=None,
    ):
        recorded.append((bucket, key, etag, rfq_id, message_id, mailbox))

    monkeypatch.setattr(SESEmailWatcher, "_record_processed_in_registry", _capture_registry, raising=False)

    results = watcher.poll_once(limit=1)

    assert results
    assert recorded
    bucket, key, _, rfq_id, message_id, mailbox = recorded[0]
    assert bucket.startswith("imap::")
    assert key.startswith("imap/")
    assert message_id == "<imap-msg-42>"
    assert mailbox == watcher.mailbox_address


def test_imap_never_invokes_s3_even_when_configured(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = "procwise-bucket"
    watcher._imap_fallback_attempts = 0

    counters = {"s3": 0, "imap": 0}

    def _stub_s3(self, limit, *, prefixes=None, on_message=None):
        counters["s3"] += 1
        return [
            {
                "id": "s3-msg-disabled",
                "message_id": "s3-msg-disabled",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Quoted price 1250",
                "rfq_id": "RFQ-20240101-abcd1234",
                "from": "supplier@example.com",
                "from_address": "supplier@example.com",
            }
        ]

    def _stub_imap(self, limit, *, mark_seen, since=None, on_message=None):
        counters["imap"] += 1
        return []

    monkeypatch.setattr(SESEmailWatcher, "_load_from_s3", _stub_s3, raising=False)
    monkeypatch.setattr(SESEmailWatcher, "_load_from_imap", _stub_imap, raising=False)

    for _ in range(3):
        batch = watcher.poll_once(limit=1)
        assert batch == []
        assert watcher._last_candidate_source == "imap"

    assert counters["imap"] == 3
    assert counters["s3"] == 0


def test_email_watcher_maps_multiple_rfq_ids():
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-multi",
            "subject": "RFQ-20240101-abcd1234 & RFQ-20240101-efgh5678",
            "body": (
                "Pricing updates: RFQ-20240101-ABCD1234 is 900 GBP and RFQ-20240101-EFGH5678 is 850 GBP."
            ),
            "from": "supplier@example.com",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    results = watcher.poll_once()
    assert len(results) == 1
    payload = results[0]

    assert payload["workflow_id"]
    assert payload["related_rfq_ids"]
    assert any(r.lower() == "rfq-20240101-efgh5678" for r in payload["related_rfq_ids"])

    workflow_key = watcher._normalise_filter_value(payload["workflow_id"])
    canonical_primary = watcher._canonical_rfq(payload["rfq_id"])
    canonical_related = {
        watcher._canonical_rfq(rfq) for rfq in payload["related_rfq_ids"] if rfq
    }

    assert canonical_primary in watcher._workflow_rfq_index.get(workflow_key, set())
    assert canonical_related.issubset(watcher._workflow_rfq_index.get(workflow_key, set()))

    assert watcher._matches_filters(payload, {"workflow_id": payload["workflow_id"]})

    secondary_payload = dict(payload)
    secondary_payload["rfq_id"] = payload["related_rfq_ids"][0]
    assert watcher._matches_filters(secondary_payload, {"workflow_id": payload["workflow_id"]})


def test_email_watcher_matches_mixed_case_workflow_filters(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-workflow",
            "subject": "RFQ-20240101-abcd1234",
            "body": "Offer for RFQ-20240101-ABCD1234 is 975",
            "from": "supplier@example.com",
        }
    ]

    monkeypatch.setattr("services.email_watcher.uuid.uuid4", lambda: "WF-Alpha-01")

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    results = watcher.poll_once(match_filters={"workflow_id": "wf-alpha-01"})

    assert len(results) == 1
    payload = results[0]
    assert payload["workflow_id"] == "WF-Alpha-01"

    workflow_key = watcher._normalise_filter_value(payload["workflow_id"])
    canonical_rfq = watcher._canonical_rfq(payload["rfq_id"])
    assert canonical_rfq in watcher._workflow_rfq_index.get(workflow_key, set())


def test_email_watcher_collects_all_workflow_responses(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-workflow-1",
            "subject": "Re: RFQ-20240101-aaaa1111",
            "body": "Offer 900",
            "from": "supplier1@example.com",
            "rfq_id": "RFQ-20240101-AAAA1111",
            "supplier_id": "SUP-001",
        },
        {
            "id": "msg-workflow-2",
            "subject": "Re: RFQ-20240101-bbbb2222",
            "body": "Offer 880",
            "from": "supplier2@example.com",
            "rfq_id": "RFQ-20240101-BBBB2222",
            "supplier_id": "SUP-002",
        },
        {
            "id": "msg-workflow-3",
            "subject": "Re: RFQ-20240101-cccc3333",
            "body": "Offer 910",
            "from": "supplier3@example.com",
            "rfq_id": "RFQ-20240101-CCCC3333",
            "supplier_id": "SUP-003",
        },
    ]

    watcher = _make_watcher(
        nick, loader=lambda limit=None: list(messages), state_store=state
    )

    metadata_map = {
        msg["rfq_id"]: {
            "workflow_id": "WF-Group-01",
            "target_price": 1000,
        }
        for msg in messages
    }

    monkeypatch.setattr(
        watcher,
        "_load_metadata",
        lambda rfq: dict(metadata_map.get(rfq, {})),
    )

    results = watcher.poll_once(match_filters={"workflow_id": "WF-Group-01"})

    assert len(results) == 3
    suppliers = {result["supplier_id"] for result in results}
    assert suppliers == {"SUP-001", "SUP-002", "SUP-003"}
    for message in messages:
        assert message["id"] in state


def test_email_watcher_filter_matches_legacy_payload_without_workflow():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    payload = {
        "rfq_id": "RFQ-20240101-ABCD1234",
        "subject": "Re: RFQ-20240101-ABCD1234",
        "from_address": "supplier@example.com",
        "related_rfq_ids": ["RFQ-20240101-EFGH5678"],
    }

    filters = {"workflow_id": "WF-Legacy-01"}
    watcher._apply_filter_defaults(payload, filters)

    assert payload["workflow_id"] == "WF-Legacy-01"
    assert watcher._matches_filters(payload, filters)


def test_email_watcher_ignores_workflow_when_run_filter_present(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-run-check",
            "subject": "Re: RFQ-20240101-ABCD1234",
            "body": "Quoted price 1230",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-ABCD1234",
            "supplier_id": "SUP-1",
            "_dispatch_record": {
                "rfq_id": "RFQ-20240101-ABCD1234",
                "supplier_id": "SUP-1",
                "run_id": "run-fallback",
            },
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    def fake_metadata(_rfq_id):
        return {"supplier_id": "SUP-1", "workflow_id": "WF-FALLBACK"}

    monkeypatch.setattr(watcher, "_load_metadata", fake_metadata, raising=False)

    filters = {
        "supplier_id": "SUP-1",
        "dispatch_run_id": "run-fallback",
        "workflow_id": "WF-IGNORED",
    }

    results = watcher.poll_once(match_filters=filters)

    assert len(results) == 1
    payload = results[0]
    assert payload["dispatch_run_id"] == "run-fallback"
    assert payload["workflow_id"] == "WF-FALLBACK"


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
    watcher.bucket = None

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


def test_email_watcher_resolves_tail_without_prefix(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-tail-fallback",
            "subject": "Quote update 20240101-ABCD1234",
            "body": "Thanks for the request 20240101-ABCD1234.",
            "from": "supplier@example.com",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: list(messages), state_store=state)

    monkeypatch.setattr(watcher, "_load_metadata", lambda rfq: {})
    monkeypatch.setattr(watcher, "_ensure_s3_mapping", lambda *args, **kwargs: None)

    def fake_tail_lookup(tail: str):
        if tail == "abcd1234":
            return [{"rfq_id": "RFQ-20240101-ABCD1234", "supplier_id": "SUP-TAIL"}]
        return []

    monkeypatch.setattr(watcher, "_get_rfq_candidates_for_tail", fake_tail_lookup)

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-ABCD1234"})

    assert len(results) == 1
    payload = results[0]
    assert payload["rfq_id"] == "RFQ-20240101-ABCD1234"
    assert payload["supplier_id"] == "SUP-TAIL"
    assert payload["matched_via"]
    assert "msg-tail-fallback" in state


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
    watcher.bucket = None

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
    watcher.bucket = None
    watcher.record_dispatch_timestamp()

    watcher.poll_interval_seconds = 1

    watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})
    assert sleep_calls[0] == pytest.approx(5.0)
    assert sleep_calls[1:] == [pytest.approx(1.0), pytest.approx(1.0)]

    sleep_calls.clear()
    watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})
    assert sleep_calls == [pytest.approx(1.0), pytest.approx(1.0)]


def test_poll_once_waits_for_sent_dispatch_count(monkeypatch):
    import services.email_watcher as email_watcher_module

    class DispatchCursor(DummyNick._DummyCursor):
        def __init__(self, owner):
            super().__init__(owner)
            self._result: List[Tuple] = []

        def execute(self, statement, params=None):
            normalized = " ".join(statement.split()).lower()
            if normalized.startswith("select action_id, process_output from proc.action where action_id ="):
                self._result = [(self._owner.action_id, self._owner.process_output)]
            elif normalized.startswith(
                "select count(*) from proc.draft_rfq_emails where sent = true and id = any"
            ):
                index = min(self._owner.sent_count_index, len(self._owner.sent_count_sequence) - 1)
                count = self._owner.sent_count_sequence[index]
                self._owner.sent_count_index = min(
                    self._owner.sent_count_index + 1, len(self._owner.sent_count_sequence) - 1
                )
                self._owner.sent_count_calls += 1
                self._result = [(count,)]
            elif normalized.startswith(
                "select count(*) from proc.draft_rfq_emails where sent = true and payload::jsonb ->> 'action_id'"
            ):
                index = min(self._owner.sent_count_index, len(self._owner.sent_count_sequence) - 1)
                count = self._owner.sent_count_sequence[index]
                self._owner.sent_count_index = min(
                    self._owner.sent_count_index + 1, len(self._owner.sent_count_sequence) - 1
                )
                self._owner.sent_count_calls += 1
                self._result = [(count,)]
            else:
                super().execute(statement, params)

        def fetchone(self):
            return self._result[0] if self._result else None

        def fetchall(self):
            return list(self._result)

    class DispatchConnection(DummyNick._DummyConn):
        def cursor(self):
            return DispatchCursor(self._owner)

    class DispatchNick(DummyNick):
        def __init__(self):
            super().__init__()
            self.settings.email_inbound_initial_wait_seconds = 5
            self.action_id = "action-test-001"
            self.process_output = json.dumps(
                {
                    "drafts": [
                        {"draft_record_id": 101, "supplier_id": "S1"},
                        {"draft_record_id": 102, "supplier_id": "S2"},
                    ]
                }
            )
            self.sent_count_sequence = [1, 2]
            self.sent_count_index = 0
            self.sent_count_calls = 0

        def get_db_connection(self):
            return DispatchConnection(self)

    fake_clock = {"now": 0.0, "sleeps": []}

    def fake_time() -> float:
        return fake_clock["now"]

    def fake_sleep(seconds: float) -> None:
        fake_clock["sleeps"].append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(
        email_watcher_module,
        "time",
        SimpleNamespace(time=fake_time, sleep=fake_sleep),
    )

    nick = DispatchNick()
    watcher = _make_watcher(nick, loader=lambda limit=None: [])
    watcher.bucket = None

    watcher.record_dispatch_timestamp()

    results = watcher.poll_once(match_filters={"action_id": nick.action_id})
    assert results == []
    assert fake_clock["sleeps"][0] == pytest.approx(5.0)
    assert nick.sent_count_calls >= 2

    fake_clock["sleeps"].clear()
    previous_calls = nick.sent_count_calls
    watcher.poll_once(match_filters={"action_id": nick.action_id})
    assert fake_clock["sleeps"] == []
    assert nick.sent_count_calls == previous_calls


def test_wait_for_dispatch_completion_adds_post_dispatch_delay(monkeypatch):
    nick = DummyNick()
    nick.settings.email_inbound_post_dispatch_delay_seconds = 3

    watcher = _make_watcher(nick, loader=lambda limit=None: [])
    expectation = watcher._DispatchExpectation(
        action_id="action-123",
        workflow_id=None,
        draft_ids=(1,),
        draft_count=1,
        supplier_count=1,
    )

    monkeypatch.setattr(watcher, "_count_sent_drafts", lambda exp: exp.draft_count)

    sleep_calls: List[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("services.email_watcher.time.sleep", fake_sleep)

    assert watcher._wait_for_dispatch_completion(expectation) is True
    assert sleep_calls and sleep_calls[-1] == pytest.approx(3.0)


def test_poll_once_skips_s3_polling_even_when_configured(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"
    watcher = _make_watcher(nick)
    watcher.bucket = "procwisemvp"
    watcher._prefixes = ["emails/"]
    watcher.poll_interval_seconds = 0

    def _fake_imap(self, limit, *, mark_seen, since=None, on_message=None):
        return []

    def _unexpected_s3_call():  # pragma: no cover - defensive guard
        raise AssertionError("S3 client should not be requested when IMAP is available")

    monkeypatch.setattr(SESEmailWatcher, "_load_from_imap", _fake_imap, raising=False)
    monkeypatch.setattr(watcher, "_get_s3_client", _unexpected_s3_call)

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert results == []
    assert watcher._last_candidate_source == "imap"


def test_poll_once_logs_error_when_imap_fails_and_uses_s3(monkeypatch, caplog):
    nick = DummyNick()
    watcher = _make_watcher(nick)
    watcher.bucket = "procwisemvp"
    watcher._prefixes = ["emails/"]
    watcher._imap_fallback_attempts = 3
    fallback_calls: List[int] = []

    class FailingIMAP:
        def __init__(self, host, port=None, *args, **kwargs):
            raise OSError("unable to connect")

    def _stub_s3(
        self,
        limit=None,
        *,
        prefixes=None,
        parser=None,
        newest_first=True,
        on_message=None,
    ):
        fallback_calls.append(1)
        return []

    monkeypatch.setattr("services.email_watcher.imaplib.IMAP4_SSL", FailingIMAP)
    monkeypatch.setattr(SESEmailWatcher, "_load_from_s3", _stub_s3, raising=False)

    caplog.set_level(logging.ERROR)

    results: List[Dict[str, object]] = []
    attempts = 3
    for _ in range(attempts):
        results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert results == []
    expected_calls = watcher._imap_fallback_attempts * attempts
    assert len(fallback_calls) == expected_calls
    assert fallback_calls == [1] * expected_calls
    assert watcher._last_candidate_source == "s3"
    assert "IMAP fallback polling failed" in caplog.text


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


def test_loader_metadata_is_hydrated_from_s3(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    bucket_name = nick.settings.ses_inbound_bucket or "procwisemvp"
    key_name = "emails/metadata-only.eml"

    raw_email = (
        b"Subject: RFQ-20240101-ABCD1234\n"
        b"From: supplier@example.com\n\n"
        b"Quoted price 1200 for RFQ-20240101-ABCD1234"
    )

    def loader(limit=None):
        return [{"id": key_name, "s3_key": key_name, "_bucket": bucket_name}]

    watcher = _make_watcher(nick, loader=loader, state_store=state)

    watcher._get_s3_client = lambda: object()
    watcher._download_object = lambda client, key, bucket=None: (raw_email, len(raw_email))

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-ABCD1234"})

    assert results
    payload = results[0]
    assert payload["rfq_id"].lower() == "rfq-20240101-abcd1234"
    assert state.items()


def test_watermark_disables_recent_window(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick, loader=lambda limit=None: [])

    watcher._last_watermark_ts = datetime.now(timezone.utc) - timedelta(hours=1)
    watcher._last_watermark_key = "emails/previous.eml"

    captured = {}

    def fake_scan(client, prefixes, *, watermark_ts, watermark_key, enforce_window):
        captured["enforce_window"] = enforce_window
        return []

    watcher._scan_recent_s3_objects = fake_scan  # type: ignore[assignment]
    watcher._get_s3_client = lambda: None

    watcher._load_from_s3(limit=None)

    assert captured.get("enforce_window") is False




def test_negotiation_executes_after_expected_responses(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-queued-1",
            "subject": "Re: RFQ-20240101-AAAA1111",
            "body": "Price 1500",
            "from": "supplier1@example.com",
            "rfq_id": "RFQ-20240101-AAAA1111",
            "supplier_id": "SUP-1",
        },
        {
            "id": "msg-queued-2",
            "subject": "Re: RFQ-20240101-BBBB2222",
            "body": "Price 900",
            "from": "supplier2@example.com",
            "rfq_id": "RFQ-20240101-BBBB2222",
            "supplier_id": "SUP-2",
        },
    ]

    batches = [messages[:1], messages[1:], []]

    def loader(limit=None):
        return batches.pop(0) if batches else []

    watcher = _make_watcher(nick, loader=loader, state_store=state)

    def fake_metadata(rfq_id):
        supplier = "SUP-1" if rfq_id.endswith("1111") else "SUP-2"
        return {
            "supplier_id": supplier,
            "target_price": 1000,
            "round": 1,
            "workflow_id": "WF-MULTI-01",
            "expected_supplier_count": 2,
        }

    monkeypatch.setattr(watcher, "_load_metadata", fake_metadata, raising=False)

    first_results = watcher.poll_once(limit=1)
    assert first_results and len(first_results) == 1
    first_payload = first_results[0]
    assert first_payload["negotiation_triggered"] is False
    assert watcher.negotiation_agent.contexts == []

    second_results = watcher.poll_once(limit=1)
    assert second_results and len(second_results) == 1
    second_payload = second_results[0]

    assert len(watcher.negotiation_agent.contexts) == 1
    negotiation_context = watcher.negotiation_agent.contexts[0]
    assert negotiation_context.input_data["current_offer"] == 1500
    assert negotiation_context.input_data["rfq_id"] == "RFQ-20240101-AAAA1111"

    cached_first = watcher._processed_cache.get("msg-queued-1")
    assert cached_first
    assert cached_first["negotiation_triggered"] is True
    assert cached_first["negotiation_status"] == AgentStatus.SUCCESS.value
    negotiation_output = cached_first["negotiation_output"]
    assert isinstance(negotiation_output, dict)
    assert negotiation_output.get("counter") == 1000

    assert second_payload["negotiation_triggered"] is False


def test_negotiation_counters_reset_between_rounds(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    messages = [
        {
            "id": "msg-round1-s1",
            "subject": "Re: RFQ-20240102-R1A",
            "body": "Price 1600",
            "from": "supplier1@example.com",
            "rfq_id": "RFQ-20240102-R1A",
            "supplier_id": "SUP-1",
        },
        {
            "id": "msg-round1-s2",
            "subject": "Re: RFQ-20240102-R1B",
            "body": "Price 900",
            "from": "supplier2@example.com",
            "rfq_id": "RFQ-20240102-R1B",
            "supplier_id": "SUP-2",
        },
        {
            "id": "msg-round2-s1",
            "subject": "Re: RFQ-20240102-R2A",
            "body": "Price 1700",
            "from": "supplier1@example.com",
            "rfq_id": "RFQ-20240102-R2A",
            "supplier_id": "SUP-1",
        },
        {
            "id": "msg-round2-s2",
            "subject": "Re: RFQ-20240102-R2B",
            "body": "Price 950",
            "from": "supplier2@example.com",
            "rfq_id": "RFQ-20240102-R2B",
            "supplier_id": "SUP-2",
        },
    ]

    batches = [[messages[0]], [messages[1]], [messages[2]], [messages[3]], []]

    def loader(limit=None):
        return batches.pop(0) if batches else []

    watcher = _make_watcher(nick, loader=loader, state_store=state)

    metadata_map = {
        "RFQ-20240102-R1A": {
            "supplier_id": "SUP-1",
            "target_price": 1000,
            "round": 1,
            "workflow_id": "WF-MULTI-RESET",
            "expected_supplier_count": 2,
        },
        "RFQ-20240102-R1B": {
            "supplier_id": "SUP-2",
            "target_price": 1000,
            "round": 1,
            "workflow_id": "WF-MULTI-RESET",
            "expected_supplier_count": 2,
        },
        "RFQ-20240102-R2A": {
            "supplier_id": "SUP-1",
            "target_price": 1000,
            "round": 2,
            "workflow_id": "WF-MULTI-RESET",
            "expected_supplier_count": 2,
        },
        "RFQ-20240102-R2B": {
            "supplier_id": "SUP-2",
            "target_price": 1000,
            "round": 2,
            "workflow_id": "WF-MULTI-RESET",
            "expected_supplier_count": 2,
        },
    }

    def fake_metadata(rfq_id):
        return dict(metadata_map[rfq_id])

    monkeypatch.setattr(watcher, "_load_metadata", fake_metadata, raising=False)

    first_round_first = watcher.poll_once(limit=1)
    assert first_round_first and len(first_round_first) == 1
    assert watcher.negotiation_agent.contexts == []

    first_round_second = watcher.poll_once(limit=1)
    assert first_round_second and len(first_round_second) == 1
    assert len(watcher.negotiation_agent.contexts) == 1

    workflow_key = watcher._normalise_group_key(None, "WF-MULTI-RESET")
    assert workflow_key not in watcher._workflow_processed_counts
    assert workflow_key not in watcher._workflow_expected_counts

    second_round_first = watcher.poll_once(limit=1)
    assert second_round_first and len(second_round_first) == 1
    assert len(watcher.negotiation_agent.contexts) == 1
    assert watcher._workflow_processed_counts.get(workflow_key) == 1
    assert watcher._workflow_expected_counts.get(workflow_key) == 2

    second_round_second = watcher.poll_once(limit=1)
    assert second_round_second and len(second_round_second) == 1
    assert len(watcher.negotiation_agent.contexts) == 2
    assert workflow_key not in watcher._workflow_processed_counts
    assert workflow_key not in watcher._workflow_expected_counts



def test_workflow_expectation_reduction_flushes_queued_jobs():
    nick = DummyNick()
    watcher = _make_watcher(nick)
    workflow_id = "WF-EXPECT-REDUCE"
    workflow_key = watcher._normalise_group_key(None, workflow_id)

    def make_metadata(expected: int) -> Dict[str, object]:
        return {
            "workflow_id": workflow_id,
            "expected_supplier_count": expected,
            "round": 1,
        }

    def make_processed(idx: int) -> Dict[str, object]:
        return {
            "message_id": f"msg-{idx}",
            "negotiation_triggered": False,
            "rfq_id": f"RFQ-{idx}",
            "supplier_id": f"SUP-{idx}",
            "subject": f"Quote update {idx}",
            "message_body": f"Supplier {idx} price {1000 + idx}",
            "workflow_id": workflow_id,
            "round": 1,
            "supplier_output": {"price": 1000 + idx, "lead_time": 4 + idx},
        }

    def make_job(idx: int) -> Dict[str, object]:
        context = AgentContext(
            workflow_id=workflow_id,
            agent_id="NegotiationAgent",
            user_id="tester",
            input_data={
                "rfq_id": f"RFQ-{idx}",
                "target_price": 1000 + idx,
                "round": 1,
            },
        )
        return {
            "context": context,
            "rfq_id": f"RFQ-{idx}",
            "round": 1,
            "supplier_id": f"SUP-{idx}",
        }

    for idx in range(3):
        triggered, output = watcher._register_processed_response(
            workflow_id,
            make_metadata(5),
            make_processed(idx),
            make_job(idx),
        )
        assert triggered is False
        assert output is None

    assert len(watcher._workflow_negotiation_jobs[workflow_key]) == 3
    assert watcher._workflow_processed_counts.get(workflow_key) == 3
    assert watcher._workflow_expected_counts.get(workflow_key) == 5

    final_processed = make_processed(3)
    triggered, output = watcher._register_processed_response(
        workflow_id,
        make_metadata(2),
        final_processed,
        make_job(3),
    )

    assert triggered is True
    assert output is not None
    assert final_processed["negotiation_triggered"] is True
    assert workflow_key not in watcher._workflow_negotiation_jobs
    assert workflow_key not in watcher._workflow_processed_counts
    assert workflow_key not in watcher._workflow_expected_counts
    assert len(watcher.negotiation_agent.contexts) == 4
    convo = watcher.negotiation_agent.contexts[-1].input_data.get("conversation_history")
    assert isinstance(convo, list) and len(convo) == 4
    quote_contexts = watcher.quote_comparison_agent.contexts
    assert quote_contexts and quote_contexts[-1].input_data.get("quotes")


def test_acknowledge_recent_dispatch_marks_all_messages(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()
    watcher = _make_watcher(nick, loader=lambda limit=None: [], state_store=state)

    expectation = watcher._DispatchExpectation(
        action_id="act-123",
        workflow_id="wf-123",
        draft_ids=(101, 102, 103),
        draft_count=3,
        supplier_count=3,
    )

    ts = datetime.now(timezone.utc)
    messages = [
        {
            "id": f"emails/msg-{idx}.eml",
            "subject": f"Subject {idx}",
            "body": f"<!-- PROCWISE:RFQ_ID=RFQ-{idx};TOKEN=token-{idx};RUN_ID=run-{idx} -->\nBody {idx}",
            "_last_modified": ts,
            "_prefix": watcher._prefixes[0],
        }
        for idx in range(1, 4)
    ]

    drafts = [
        watcher._DraftSnapshot(
            id=100 + idx,
            rfq_id=f"RFQ-{idx}",
            subject=f"Subject {idx}",
            body=f"Body {idx}",
            dispatch_token=f"token-{idx}",
            run_id=f"run-{idx}",
        )
        for idx in range(1, 4)
    ]

    monkeypatch.setattr(
        watcher,
        "_load_from_s3",
        lambda limit, prefixes=None, newest_first=True: list(messages)[:limit],
    )
    monkeypatch.setattr(
        watcher,
        "_fetch_recent_dispatched_drafts",
        lambda exp, limit: list(drafts)[:limit],
    )

    watcher._s3_prefix_watchers = {watcher._prefixes[0]: S3ObjectWatcher(limit=10)}

    watcher._acknowledge_recent_dispatch(expectation, completed=True)

    for idx, message in enumerate(messages, start=1):
        key = message["id"]
        assert key in state
        metadata = state.get(key)
        assert metadata["status"] == "dispatch_copy"
        assert metadata["dispatch_completed"] is True
        assert metadata["run_id"] == f"run-{idx}"


def test_acknowledge_recent_dispatch_prefers_imap(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.procwise.test"
    nick.settings.imap_user = "agent@procwise.test"
    nick.settings.imap_password = "secret"
    state = InMemoryEmailWatcherState()
    watcher = _make_watcher(nick, state_store=state)

    expectation = watcher._DispatchExpectation(
        action_id="act-imap",
        workflow_id="wf-imap",
        draft_ids=(301, 302),
        draft_count=2,
        supplier_count=2,
    )

    ts = datetime.now(timezone.utc)
    imap_messages = [
        {
            "id": f"imap/msg-{idx}",
            "subject": f"Subject {idx}",
            "body": f"<!-- PROCWISE:RFQ_ID=RFQ-{idx};TOKEN=token-{idx};RUN_ID=run-{idx} -->\nBody {idx}",
            "_last_modified": ts,
        }
        for idx in range(1, 3)
    ]

    drafts = [
        watcher._DraftSnapshot(
            id=300 + idx,
            rfq_id=f"RFQ-{idx}",
            subject=f"Subject {idx}",
            body=f"Body {idx}",
            dispatch_token=f"token-{idx}",
            run_id=f"run-{idx}",
        )
        for idx in range(1, 3)
    ]

    imap_calls: List[Tuple[Optional[int], bool]] = []
    s3_called = False

    def fake_imap(limit, mark_seen=False, since=None):
        imap_calls.append((limit, mark_seen))
        return list(imap_messages)

    def fake_s3(*args, **kwargs):
        nonlocal s3_called
        s3_called = True
        return []

    monkeypatch.setattr(watcher, "_load_from_imap", fake_imap)
    monkeypatch.setattr(watcher, "_load_from_s3", fake_s3)
    monkeypatch.setattr(
        watcher,
        "_fetch_recent_dispatched_drafts",
        lambda exp, limit: list(drafts)[:limit],
    )
    watcher._s3_prefix_watchers = {
        prefix: S3ObjectWatcher(limit=10) for prefix in watcher._prefixes
    }

    watcher._acknowledge_recent_dispatch(expectation, completed=False)

    assert imap_calls == [(2, False)]
    assert s3_called is False
    for message in imap_messages:
        metadata = state.get(message["id"])
        assert metadata["status"] == "dispatch_copy"
        assert metadata["dispatch_completed"] is False


def test_acknowledge_recent_dispatch_falls_back_to_s3(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.procwise.test"
    nick.settings.imap_user = "agent@procwise.test"
    nick.settings.imap_password = "secret"
    state = InMemoryEmailWatcherState()
    watcher = _make_watcher(nick, state_store=state)

    expectation = watcher._DispatchExpectation(
        action_id="act-hybrid",
        workflow_id="wf-hybrid",
        draft_ids=(401, 402, 403),
        draft_count=3,
        supplier_count=3,
    )

    ts = datetime.now(timezone.utc)
    imap_messages = [
        {
            "id": "imap/msg-1",
            "subject": "Subject 1",
            "body": "<!-- PROCWISE:RFQ_ID=RFQ-1;TOKEN=token-1;RUN_ID=run-1 -->\nBody 1",
            "_last_modified": ts,
        }
    ]
    s3_messages = [
        {
            "id": f"emails/msg-{idx}",
            "subject": f"Subject {idx}",
            "body": f"<!-- PROCWISE:RFQ_ID=RFQ-{idx};TOKEN=token-{idx};RUN_ID=run-{idx} -->\nBody {idx}",
            "_last_modified": ts,
            "_prefix": watcher._prefixes[0],
        }
        for idx in range(2, 4)
    ]

    drafts = [
        watcher._DraftSnapshot(
            id=400 + idx,
            rfq_id=f"RFQ-{idx}",
            subject=f"Subject {idx}",
            body=f"Body {idx}",
            dispatch_token=f"token-{idx}",
            run_id=f"run-{idx}",
        )
        for idx in range(1, 4)
    ]

    s3_calls: List[Tuple[Optional[int], Tuple[str, ...]]] = []

    def fake_imap(limit, mark_seen=False, since=None):
        return list(imap_messages)

    def fake_s3(limit, prefixes=None, newest_first=True):
        s3_calls.append((limit, tuple(prefixes or ())))
        return list(s3_messages)

    monkeypatch.setattr(watcher, "_load_from_imap", fake_imap)
    monkeypatch.setattr(watcher, "_load_from_s3", fake_s3)
    monkeypatch.setattr(
        watcher,
        "_fetch_recent_dispatched_drafts",
        lambda exp, limit: list(drafts)[:limit],
    )
    watcher._s3_prefix_watchers = {
        prefix: S3ObjectWatcher(limit=10) for prefix in watcher._prefixes
    }

    watcher._acknowledge_recent_dispatch(expectation, completed=True)

    assert s3_calls == [(3, tuple(watcher._prefixes))]
    for message in [*imap_messages, *s3_messages]:
        metadata = state.get(message["id"])
        assert metadata["status"] == "dispatch_copy"

def test_sqs_email_loader_extracts_records():
    class StubSQSClient:
        def __init__(self):
            self.deleted: List[str] = []
            self._messages = [
                {
                    "Body": json.dumps(
                        {
                            "Records": [
                                {
                                    "s3": {
                                        "bucket": {"name": "procwisemvp"},
                                        "object": {"key": "emails/test%20file.eml"},
                                    },
                                    "eventTime": "2025-10-04T06:05:58Z",
                                }
                            ]
                        }
                    ),
                    "ReceiptHandle": "r1",
                }
            ]

        def receive_message(self, **kwargs):
            return {"Messages": list(self._messages)}

        def delete_message(self, **kwargs):
            self.deleted.append(kwargs.get("ReceiptHandle"))
            self._messages.clear()

    stub = StubSQSClient()
    loader = sqs_email_loader(
        queue_url="https://sqs.example.com/queue",
        max_batch=5,
        wait_seconds=0,
        visibility_timeout=30,
        sqs_client=stub,
    )

    batch = loader()

    assert batch
    record = batch[0]
    assert record["s3_key"] == "emails/test file.eml"
    assert record["_bucket"] == "procwisemvp"
    assert isinstance(record.get("_last_modified"), datetime)
    assert stub.deleted == ["r1"]


def test_should_not_stop_after_supplier_or_rfq_filters():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    assert watcher._should_stop_after_match({"supplier_id": "supplier-1"}) is False
    assert (
        watcher._should_stop_after_match({"rfq_id": "RFQ-20240215-SIM12345"})
        is False
    )
    assert (
        watcher._should_stop_after_match(
            {
                "supplier_id": "supplier-1",
                "rfq_id": "RFQ-20240215-SIM12345",
                "workflow_id": "workflow-1",
            }
        )
        is False
    )


def test_should_stop_after_unrelated_filters():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    assert watcher._should_stop_after_match({"status": "open"}) is True


def test_filters_should_expand_limit_for_supplier_and_workflow():
    nick = DummyNick()
    watcher = _make_watcher(nick)

    assert watcher._filters_should_expand_limit({"supplier_id": "SUP-1"}) is True
    assert watcher._filters_should_expand_limit({"rfq_id": "RFQ-2024"}) is True
    assert (
        watcher._filters_should_expand_limit({"workflow_id": "wf-1", "action_id": "act-1"})
        is True
    )
    assert watcher._filters_should_expand_limit({"status": "pending"}) is False
