import gzip
import re
from types import SimpleNamespace

from agents.base_agent import AgentOutput, AgentStatus, AgentContext
from services.email_watcher import (
    InMemoryEmailWatcherState,
    SESEmailWatcher,
)


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
            imap_host=None,
            imap_user=None,
            imap_password=None,
            imap_port=993,
            imap_folder="INBOX",
            imap_search_criteria="UNSEEN",
            imap_use_ssl=True,
            imap_mark_seen=True,
            imap_batch_size=25,
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


def test_email_watcher_matches_uppercase_rfq_identifier():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    messages = [
        {
            "id": "msg-4",
            "subject": "Re: RFQ-20240101-ABCD1234",
            "body": "Quoted price 1200 USD",
            "from": "supplier@example.com",
        }
    ]

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-4", "target_price": 1500},
        message_loader=lambda limit=None: messages,
        state_store=InMemoryEmailWatcherState(),
    )

    results = watcher.poll_once()

    assert len(results) == 1
    assert results[0]["rfq_id"] == "RFQ-20240101-ABCD1234"
    assert results[0]["supplier_id"] == "SUP-4"
    # Supplier agent should have been invoked once with the parsed RFQ ID
    assert supplier_agent.contexts[0].input_data["rfq_id"] == "RFQ-20240101-ABCD1234"


def test_email_watcher_extracts_rfq_from_html_comment():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    html_body = """
        <html>
            <body>
                <!-- RFQ-ID: RFQ-20240202-a1b2c3d4 -->
                <p>Supplier quotation attached.</p>
            </body>
        </html>
    """
    messages = [
        {
            "id": "msg-5",
            "subject": "Re: Request for Quotation",
            "body": html_body,
            "from": "supplier@example.com",
        }
    ]

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-5", "target_price": 1500},
        message_loader=lambda limit=None: messages,
        state_store=InMemoryEmailWatcherState(),
    )

    results = watcher.poll_once()

    assert len(results) == 1
    assert results[0]["rfq_id"] == "RFQ-20240202-a1b2c3d4"
    assert supplier_agent.contexts[0].input_data["rfq_id"] == "RFQ-20240202-a1b2c3d4"


def test_email_watcher_fallbacks_to_imap_when_s3_empty(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "supplierconnect@procwise.co.uk"
    nick.settings.imap_password = "secret"

    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    imap_message = {
        "id": "imap:1",
        "subject": "Re: RFQ-20240303-feedface",
        "body": "Quoted price 1800 USD",
        "from": "supplier@example.com",
        "rfq_id": "RFQ-20240303-feedface",
    }

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-IMAP", "target_price": 1500},
        state_store=InMemoryEmailWatcherState(),
    )

    monkeypatch.setattr(watcher, "_load_from_s3", lambda limit=None: [])

    call_args = {}

    def fake_imap(limit=None, mark_seen=False):
        call_args["limit"] = limit
        call_args["mark_seen"] = mark_seen
        return [imap_message]

    monkeypatch.setattr(watcher, "_load_from_imap", fake_imap)

    results = watcher.poll_once(limit=5)

    assert call_args["mark_seen"] is True
    assert len(results) == 1
    assert results[0]["rfq_id"] == "RFQ-20240303-feedface"
    assert watcher.state_store.get("imap:1")["status"] == "processed"


def test_email_watcher_peek_recent_messages_uses_non_destructive_imap(monkeypatch):
    nick = DummyNick()
    nick.settings.imap_host = "imap.example.com"
    nick.settings.imap_user = "supplierconnect@procwise.co.uk"
    nick.settings.imap_password = "secret"

    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {},
        state_store=InMemoryEmailWatcherState(),
    )

    monkeypatch.setattr(watcher, "_load_from_s3", lambda limit=None: [])

    def fake_imap(limit=None, mark_seen=False):
        assert mark_seen is False
        return [
            {
                "id": "imap:preview",
                "subject": "Re: RFQ-20240404-cafebabe",
                "body": "Quoted price 950 USD with delivery in 2 weeks",
                "from": "supplier@example.com",
                "rfq_id": "RFQ-20240404-cafebabe",
                "received_at": "Thu, 04 Apr 2024 10:00:00 +0000",
            }
        ]

    monkeypatch.setattr(watcher, "_load_from_imap", fake_imap)

    preview = watcher.peek_recent_messages(limit=2)

    assert preview == [
        {
            "id": "imap:preview",
            "subject": "Re: RFQ-20240404-cafebabe",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240404-cafebabe",
            "received_at": "Thu, 04 Apr 2024 10:00:00 +0000",
            "snippet": "Quoted price 950 USD with delivery in 2 weeks",
        }
    ]
    assert watcher.state_store._seen == {}


def test_email_watcher_reuses_agent_s3_client():
    nick = DummyNick()
    nick.s3_client = object()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {},
        state_store=InMemoryEmailWatcherState(),
    )

    assert watcher._get_s3_client() is nick.s3_client


def test_email_watcher_downloads_and_decompresses_gzip():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {},
        state_store=InMemoryEmailWatcherState(),
    )

    payload = gzip.compress(b"raw email bytes")

    class DummyBody:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    class DummyClient:
        def get_object(self, Bucket, Key):
            return {"Body": DummyBody(payload), "ContentEncoding": "gzip"}

    raw = watcher._download_object(DummyClient(), "ses/inbound/test.eml.gz")
    assert raw == b"raw email bytes"
