import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from services.email_watcher import InMemoryEmailWatcherState, SESEmailWatcher


class StubSupplierInteractionAgent:
    RFQ_PATTERN = re.compile(r"RFQ-\d{8}-[a-f0-9]{8}", re.IGNORECASE)

    def __init__(self):
        self.contexts = []

    def execute(self, context: AgentContext) -> AgentOutput:
        self.contexts.append(context)
        text = context.input_data.get("message", "")
        price_match = re.search(r"(\d+(?:\.\d+)?)", text)
        price = float(price_match.group(1)) if price_match else None
        data = {"price": price, "rfq_id": context.input_data.get("rfq_id")}
        next_agents = []
        target = context.input_data.get("target_price")
        if target is not None and price is not None and price > target:
            next_agents = ["NegotiationAgent"]
        return AgentOutput(status=AgentStatus.SUCCESS, data=data, next_agents=next_agents)


class StubNegotiationAgent:
    def __init__(self):
        self.contexts = []

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
        )
        self.agents = {}
        self.ddl_statements = []
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
    watcher = _make_watcher(nick, loader=lambda limit=None: [])
    assert "CREATE TABLE IF NOT EXISTS proc.rfq_targets" in "\n".join(nick.ddl_statements)
    assert "CREATE TABLE IF NOT EXISTS proc.negotiation_sessions" in "\n".join(nick.ddl_statements)


def test_email_watcher_triggers_negotiation_when_price_high():
    nick = DummyNick()
    messages = [
        {
            "id": "msg-1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1500 USD",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
            "target_price": 1000,
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: messages)

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(results) == 1
    result = results[0]
    assert result["negotiation_triggered"] is True
    assert result["rfq_id"].lower() == "rfq-20240101-abcd1234"


def test_poll_once_filters_by_rfq_id():
    nick = DummyNick()
    messages = [
        {
            "id": "msg-1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Price 1000",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
        },
        {
            "id": "msg-2",
            "subject": "Re: RFQ-20240101-deadbeef",
            "body": "Price 800",
            "from": "other@example.com",
            "rfq_id": "RFQ-20240101-deadbeef",
        },
    ]
    state = InMemoryEmailWatcherState()
    watcher = _make_watcher(nick, loader=lambda limit=None: messages, state_store=state)

    filtered = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(filtered) == 1
    assert filtered[0]["rfq_id"].lower() == "rfq-20240101-abcd1234"
    assert "msg-1" in state


def test_poll_once_uses_s3_loader_when_no_custom_loader(monkeypatch):
    nick = DummyNick()
    watcher = _make_watcher(nick)

    captured_limits = []

    def fake_load(limit=None, *, prefixes=None, parser=None, newest_first=False):
        captured_limits.append(limit)
        return [
            {
                "id": "emails/msg-1.eml",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Price 900",
                "from": "supplier@example.com",
                "rfq_id": "RFQ-20240101-abcd1234",
            }
        ]

    monkeypatch.setattr(watcher, "_load_from_s3", fake_load)

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})

    assert len(results) == 1
    assert captured_limits == [None]


def test_watch_retries_until_message_found(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    batches = [
        [],
        [
            {
                "id": "emails/msg-1.eml",
                "subject": "Re: RFQ-20240101-abcd1234",
                "body": "Price 750",
                "from": "supplier@example.com",
                "rfq_id": "RFQ-20240101-abcd1234",
            }
        ],
    ]

    def loader(limit=None):
        if batches:
            return batches.pop(0)
        return []

    watcher = _make_watcher(nick, loader=loader, state_store=state)

    processed = watcher.watch(interval=1, limit=None, timeout_seconds=5)

    assert processed == 1
    assert any(item.get("rfq_id") == "RFQ-20240101-abcd1234" for item in state._seen.values())


def test_cached_match_is_returned_without_reprocessing(monkeypatch):
    nick = DummyNick()
    state = InMemoryEmailWatcherState()

    first_batch = [
        {
            "id": "emails/msg-1.eml",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Price 700",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
        }
    ]

    batches = [first_batch, []]

    def loader(limit=None):
        return batches.pop(0) if batches else []

    watcher = _make_watcher(nick, loader=loader, state_store=state)

    first_results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})
    assert len(first_results) == 1

    # Subsequent poll should use the cached payload even though the loader returns nothing.
    second_results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-abcd1234"})
    assert len(second_results) == 1
    assert second_results[0]["rfq_id"].lower() == "rfq-20240101-abcd1234"


def test_poll_once_logs_and_returns_empty_when_no_match(caplog):
    nick = DummyNick()
    messages = [
        {
            "id": "emails/msg-1.eml",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Price 1000",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
        }
    ]

    watcher = _make_watcher(nick, loader=lambda limit=None: messages)

    caplog.set_level("INFO", logger="services.email_watcher")
    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-UNKNOWN"})

    assert results == []
    assert any("without matching filters" in record.message for record in caplog.records)
