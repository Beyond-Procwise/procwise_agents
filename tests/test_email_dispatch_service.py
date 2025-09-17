import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.email_dispatch_service import EmailDispatchService


class InMemoryDraftStore:
    def __init__(self):
        self.rows = {}
        self.next_id = 1

    def add(self, record):
        record = dict(record)
        record.setdefault("id", self.next_id)
        self.rows[record["id"]] = record
        self.next_id += 1
        return record["id"]

    def get_latest(self, rfq_id):
        candidates = [row for row in self.rows.values() if row["rfq_id"] == rfq_id]
        if not candidates:
            return None
        # Order by sent flag then thread index then id to mirror SQL
        candidates.sort(key=lambda row: (row["sent"], -row["thread_index"], -row["id"]))
        row = candidates[0]
        return (
            row["id"],
            row["rfq_id"],
            row.get("supplier_id"),
            row.get("supplier_name"),
            row["subject"],
            row["body"],
            row["sent"],
            row.get("recipient_email"),
            row.get("contact_level", 0),
            row.get("thread_index", 1),
            row.get("payload"),
            row.get("sender"),
            row.get("sent_on"),
        )

    def update(self, draft_id, **updates):
        if draft_id in self.rows:
            self.rows[draft_id].update(updates)


class DummyCursor:
    def __init__(self, store):
        self.store = store
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT id, rfq_id"):
            rfq_id = params[0]
            row = self.store.get_latest(rfq_id)
            self._result = [row] if row else []
        elif normalized.startswith("UPDATE proc.draft_rfq_emails SET"):
            sent_flag, subject, body, recipient, contact_level, payload_json, sent_bool, draft_id = params
            updates = {
                "sent": bool(sent_flag),
                "subject": subject,
                "body": body,
                "recipient_email": recipient,
                "contact_level": contact_level,
                "payload": payload_json,
            }
            if sent_bool:
                updates["sent_on"] = "now"
            self.store.update(draft_id, **updates)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected query: {query}")

    def fetchone(self):
        return self._result[0] if self._result else None


class DummyConnection:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return DummyCursor(self.store)

    def commit(self):
        pass


class DummyNick:
    def __init__(self, store):
        self.settings = SimpleNamespace(ses_default_sender="sender@example.com")
        self._store = store

    def get_db_connection(self):
        return DummyConnection(self._store)


def test_email_dispatch_service_sends_and_updates_status(monkeypatch):
    store = InMemoryDraftStore()
    draft_payload = {
        "rfq_id": "RFQ-UNIT",
        "subject": "RFQ RFQ-UNIT – Request for Quotation",
        "body": "<p>Hello</p>",
        "receiver": "buyer@example.com",
        "recipients": ["buyer@example.com"],
        "sender": "sender@example.com",
        "thread_index": 1,
        "contact_level": 1,
        "sent_status": False,
    }
    draft_id = store.add(
        {
            "rfq_id": "RFQ-UNIT",
            "supplier_id": "S1",
            "supplier_name": "Acme",
            "subject": draft_payload["subject"],
            "body": draft_payload["body"],
            "sent": False,
            "recipient_email": None,
            "contact_level": 0,
            "thread_index": 1,
            "sender": "sender@example.com",
            "payload": json.dumps(draft_payload),
            "sent_on": None,
        }
    )

    nick = DummyNick(store)
    service = EmailDispatchService(nick)

    sent_args = {}

    def fake_send(subject, body, recipients, sender, attachments=None):
        sent_args.update(
            {
                "subject": subject,
                "body": body,
                "recipients": recipients,
                "sender": sender,
                "attachments": attachments,
            }
        )
        return True

    monkeypatch.setattr(service.email_service, "send_email", fake_send)

    result = service.send_draft("RFQ-UNIT")

    assert result["sent"] is True
    assert result["recipients"] == ["buyer@example.com"]
    assert result["subject"].startswith("RFQ RFQ-UNIT")
    assert result["draft"]["sent_status"] is True
    assert result["body"].startswith("<!-- RFQ-ID: RFQ-UNIT -->")
    assert store.rows[draft_id]["sent"] is True
    assert store.rows[draft_id]["recipient_email"] == "buyer@example.com"
    assert json.loads(store.rows[draft_id]["payload"])["sent_status"] is True
    assert sent_args["recipients"] == ["buyer@example.com"]
    assert sent_args["sender"] == "sender@example.com"
    assert sent_args["body"].startswith("<!-- RFQ-ID: RFQ-UNIT -->")
