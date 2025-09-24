import re
from types import SimpleNamespace

from agents.base_agent import AgentOutput, AgentStatus, AgentContext
from services.email_watcher import (
    InMemoryEmailWatcherState,
    SESEmailWatcher,
)


class StubSupplierInteractionAgent:
    RFQ_PATTERN = re.compile(r"RFQ-\d{8}-[a-f0-9]{8}")

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
    def __init__(self):
        self.settings = SimpleNamespace(
            script_user="AgentNick",
            ses_default_sender="nicholasgeelen@procwise.co.uk",
            ses_smtp_endpoint="email-smtp.eu-west-1.amazonaws.com",
            ses_smtp_secret_name="ses/smtp/credentials",
            ses_region="eu-west-1",
            ses_inbound_prefix="ses/inbound/",
            s3_bucket_name="bucket",
            email_response_poll_seconds=60,
        )
        self.agents = {}

    def get_db_connection(self):  # pragma: no cover - safety
        raise AssertionError("DB access not expected in tests")


def test_email_watcher_triggers_negotiation_when_price_high():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    messages = [
        {
            "id": "msg-1",
            "subject": "Re: RFQ-20240101-abcd1234",
            "body": "Quoted price 1500 USD",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-abcd1234",
        }
    ]

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-1", "target_price": 1000},
        message_loader=lambda limit=None: messages,
        state_store=InMemoryEmailWatcherState(),
    )

    results = watcher.poll_once()

    assert len(results) == 1
    assert negotiation_agent.contexts
    assert results[0]["negotiation_triggered"] is True
    assert results[0]["rfq_id"] == "RFQ-20240101-abcd1234"
    assert results[0]["supplier_id"] == "SUP-1"
    assert watcher.state_store.get("msg-1")["status"] == "processed"

    # Subsequent polls should skip already processed emails
    assert watcher.poll_once() == []


def test_email_watcher_skips_negotiation_when_price_within_target():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    messages = [
        {
            "id": "msg-2",
            "subject": "Re: RFQ-20240101-deadbeef",
            "body": "Quoted price 800 USD",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-deadbeef",
        }
    ]

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-2", "target_price": 1000},
        message_loader=lambda limit=None: messages,
        state_store=InMemoryEmailWatcherState(),
    )

    results = watcher.poll_once()

    assert len(results) == 1
    assert results[0]["negotiation_triggered"] is False
    assert negotiation_agent.contexts == []
    assert watcher.state_store.get("msg-2")["status"] == "processed"


def test_email_watcher_records_skipped_messages_without_rfq():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    messages = [
        {
            "id": "msg-3",
            "subject": "General update",
            "body": "No RFQ reference in this email",
            "from": "supplier@example.com",
        }
    ]

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {},
        message_loader=lambda limit=None: messages,
        state_store=InMemoryEmailWatcherState(),
    )

    results = watcher.poll_once()

    assert results == []
    assert watcher.state_store.get("msg-3")["status"] == "skipped"
