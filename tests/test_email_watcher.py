import gzip
import json
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
            ses_inbound_prefix="ses/inbound/",
            ses_inbound_s3_uri=None,
            s3_bucket_name="bucket",
            ses_inbound_queue_url=None,
            ses_inbound_queue_wait_seconds=2,
            ses_inbound_queue_max_messages=10,
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
        self.ddl_statements = []

    def get_db_connection(self):
        return DummyNick._DummyConn(self)


class DummyQueue:
    def __init__(self, messages):
        self._messages = list(messages)
        self.deleted = []
        self.receive_calls = 0

    def receive_message(self, **kwargs):
        self.receive_calls += 1
        if self._messages:
            return {"Messages": [self._messages.pop(0)]}
        return {}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append((QueueUrl, ReceiptHandle))


def test_email_watcher_bootstraps_negotiation_tables():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        message_loader=lambda limit=None: [],
        state_store=InMemoryEmailWatcherState(),
    )

    ddl_statements = "\n".join(nick.ddl_statements)
    assert "CREATE TABLE IF NOT EXISTS proc.rfq_targets" in ddl_statements
    assert "CREATE TABLE IF NOT EXISTS proc.negotiation_sessions" in ddl_statements


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


def test_email_watcher_stops_after_matching_filters():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    messages = [
        {
            "id": "msg-a",
            "subject": "Re: RFQ-20240101-aaaa1111",
            "body": "Quoted price 1200",
            "from": "first@example.com",
            "rfq_id": "RFQ-20240101-aaaa1111",
        },
        {
            "id": "msg-b",
            "subject": "Re: RFQ-20240101-target",
            "body": "Quoted price 950",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-target",
        },
        {
            "id": "msg-c",
            "subject": "Re: RFQ-20240101-extra",
            "body": "Quoted price 800",
            "from": "other@example.com",
            "rfq_id": "RFQ-20240101-extra",
        },
    ]

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        message_loader=lambda limit=None: list(messages),
        state_store=InMemoryEmailWatcherState(),
    )

    results = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-target"})

    assert len(results) == 2
    assert results[-1]["rfq_id"] == "RFQ-20240101-target"
    assert len(supplier_agent.contexts) == 2
    assert watcher.state_store.get("msg-c") is None


def test_email_watcher_returns_cached_processed_message_for_filters():
    nick = DummyNick()
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()
    messages = [
        {
            "id": "msg-cache",
            "subject": "Re: RFQ-20240101-cache",
            "body": "Quoted price 500",
            "from": "supplier@example.com",
            "rfq_id": "RFQ-20240101-cache",
        }
    ]

    def loader(limit=None):
        if messages:
            return [messages.pop(0)]
        return []

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        message_loader=loader,
        state_store=InMemoryEmailWatcherState(),
    )

    first_batch = watcher.poll_once()

    assert len(first_batch) == 1
    assert first_batch[0]["rfq_id"] == "RFQ-20240101-cache"
    assert first_batch[0]["message_body"].startswith("Quoted price")
    assert len(supplier_agent.contexts) == 1

    supplier_agent.contexts.clear()

    cached_batch = watcher.poll_once(match_filters={"rfq_id": "RFQ-20240101-cache"})

    assert len(cached_batch) == 1
    assert cached_batch[0]["rfq_id"] == "RFQ-20240101-cache"
    assert cached_batch[0]["message_body"].startswith("Quoted price")
    assert supplier_agent.contexts == []


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


def test_email_watcher_builds_prefix_variants_for_mailbox():
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

    assert "ses/inbound/" in watcher._prefixes
    assert "ses/inbound/supplierconnect@procwise.co.uk/" in watcher._prefixes
    assert "supplierconnect@procwise.co.uk/" in watcher._prefixes


def test_email_watcher_processes_queue_messages_and_acknowledges(monkeypatch):
    nick = DummyNick()
    nick.settings.ses_inbound_queue_url = "https://sqs.eu-west-1.amazonaws.com/123/queue"
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-QUEUE", "target_price": 900},
        state_store=InMemoryEmailWatcherState(),
    )

    raw_email = b"Subject: Re: RFQ-20240101-abcd1234\nFrom: supplier@example.com\n\nQuoted price 950"
    sns_payload = {
        "Type": "Notification",
        "Message": json.dumps(
            {
                "receipt": {
                    "action": {
                        "bucketName": "bucket",
                        "objectKey": "ses/inbound/message-1.eml",
                    }
                },
                "mail": {"messageId": "msg-1"},
            }
        ),
    }

    queue = DummyQueue(
        [
            {
                "MessageId": "sqs-1",
                "ReceiptHandle": "rh-1",
                "Body": json.dumps(sns_payload),
            }
        ]
    )

    monkeypatch.setattr(watcher, "_get_sqs_client", lambda: queue)
    monkeypatch.setattr(watcher, "_get_s3_client", lambda: object())
    monkeypatch.setattr(watcher, "_load_from_s3", lambda limit=None: [])
    monkeypatch.setattr(
        watcher,
        "_download_object",
        lambda client, key, bucket=None: raw_email if key == "ses/inbound/message-1.eml" else None,
    )

    results = watcher.poll_once()

    assert len(results) == 1
    assert queue.deleted == [(nick.settings.ses_inbound_queue_url, "rh-1")]
    assert watcher.state_store.get("ses/inbound/message-1.eml")["status"] == "processed"


def test_email_watcher_leaves_queue_message_on_processing_error(monkeypatch):
    nick = DummyNick()
    nick.settings.ses_inbound_queue_url = "https://sqs.eu-west-1.amazonaws.com/123/queue"
    supplier_agent = StubSupplierInteractionAgent()
    negotiation_agent = StubNegotiationAgent()

    watcher = SESEmailWatcher(
        nick,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        metadata_provider=lambda _: {"supplier_id": "SUP-QUEUE"},
        state_store=InMemoryEmailWatcherState(),
    )

    raw_email = b"Subject: Re: RFQ-20240101-deadbeef\nFrom: supplier@example.com\n\nQuoted price 1200"
    sns_payload = {
        "Type": "Notification",
        "Message": json.dumps(
            {
                "receipt": {
                    "action": {
                        "bucketName": "bucket",
                        "objectKey": "ses/inbound/message-error.eml",
                    }
                },
                "mail": {"messageId": "msg-err"},
            }
        ),
    }

    queue = DummyQueue(
        [
            {
                "MessageId": "sqs-err",
                "ReceiptHandle": "rh-err",
                "Body": json.dumps(sns_payload),
            }
        ]
    )

    monkeypatch.setattr(watcher, "_get_sqs_client", lambda: queue)
    monkeypatch.setattr(watcher, "_get_s3_client", lambda: object())
    monkeypatch.setattr(watcher, "_load_from_s3", lambda limit=None: [])
    monkeypatch.setattr(
        watcher,
        "_download_object",
        lambda client, key, bucket=None: raw_email if key == "ses/inbound/message-error.eml" else None,
    )

    def explode(_message):
        raise RuntimeError("boom")

    monkeypatch.setattr(watcher, "_process_message", explode)

    results = watcher.poll_once()

    assert results == []
    assert queue.deleted == []
    assert watcher.state_store.get("ses/inbound/message-error.eml") is None


def test_email_watcher_uses_s3_uri_for_bucket_and_prefix():
    nick = DummyNick()
    nick.settings.s3_bucket_name = None
    nick.settings.ses_inbound_bucket = None
    nick.settings.ses_inbound_s3_uri = "s3://procwisemvp/emails/"

    watcher = SESEmailWatcher(
        nick,
        message_loader=lambda limit=None: [],
        state_store=InMemoryEmailWatcherState(),
    )

    assert watcher.bucket == "procwisemvp"
    assert "emails/" in watcher._prefixes
