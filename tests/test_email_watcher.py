
import io
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from email.message import EmailMessage
from typing import Dict, List, Tuple

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
    assert processed["matched_via"] == "dispatch"
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
    assert processed["matched_via"] == "dispatch_similarity"
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


def test_email_watcher_falls_back_to_imap(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = None

    class StubIMAP:
        def __init__(self, host):
            self.host = host

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


def test_imap_fallback_triggers_after_three_empty_s3_batches(monkeypatch):
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

    def _stub_imap(self, limit, *, mark_seen, on_message=None):
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

    assert watcher.poll_once(limit=1) == []
    assert watcher.poll_once(limit=1) == []
    third_batch = watcher.poll_once(limit=1)

    assert counters["s3"] >= 3
    assert counters["imap"] == 1
    assert third_batch
    assert third_batch[0]["message_id"].startswith("imap-msg-")


def test_imap_loader_records_processed_email(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "inbound@example.com"
    nick.settings.imap_password = "secret"
    nick.settings.imap_mailbox = "INBOX"

    watcher = _make_watcher(nick, loader=None)
    watcher.bucket = None

    class StubIMAP:
        def __init__(self, host):
            self.host = host

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

    def _capture_registry(self, bucket, key, etag, rfq_id):
        recorded.append((bucket, key, etag, rfq_id))

    monkeypatch.setattr(SESEmailWatcher, "_record_processed_in_registry", _capture_registry, raising=False)

    results = watcher.poll_once(limit=1)

    assert results
    assert recorded
    bucket, key, _, rfq_id = recorded[0]
    assert bucket.startswith("imap::")
    assert key.startswith("imap/")
    assert rfq_id


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
    assert results[0]["message_id"] in {"emails/matching.eml", "<matching@example.com>"}
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
    assert cached_first["negotiation_output"] == {"counter": 1000}

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

    workflow_key = watcher._normalise_workflow_key("WF-MULTI-RESET")
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
