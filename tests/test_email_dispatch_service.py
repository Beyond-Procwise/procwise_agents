import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.email_drafting_agent import DEFAULT_RFQ_SUBJECT
from services.email_dispatch_service import EmailDispatchService
from services.email_service import EmailSendResult
from utils.email_markers import extract_rfq_id, extract_run_id, split_hidden_marker


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


class InMemoryActionStore:
    def __init__(self):
        self.rows = {}

    def get(self, action_id):
        return self.rows.get(action_id)

    def update(self, action_id, payload):
        self.rows[action_id] = payload


class DummyCursor:
    def __init__(self, draft_store, action_store):
        self.store = draft_store
        self.action_store = action_store
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
            (
                sent_flag,
                subject,
                body,
                recipient,
                contact_level,
                supplier_id,
                supplier_name,
                rfq_value,
                payload_json,
                sent_bool,
                draft_id,
            ) = params
            updates = {
                "sent": bool(sent_flag),
                "subject": subject,
                "body": body,
                "recipient_email": recipient,
                "contact_level": contact_level,
                "payload": payload_json,
            }
            updates["supplier_id"] = supplier_id
            updates["supplier_name"] = supplier_name
            updates["rfq_id"] = rfq_value
            if sent_bool:
                updates["sent_on"] = "now"
            self.store.update(draft_id, **updates)
        elif normalized.startswith("SELECT process_output FROM proc.action"):
            action_id = params[0]
            payload = self.action_store.get(action_id)
            self._result = [(payload,)] if payload is not None else []
        elif normalized.startswith("UPDATE proc.action SET process_output"):
            payload_json, action_id = params
            self.action_store.update(action_id, payload_json)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected query: {query}")

    def fetchone(self):
        return self._result[0] if self._result else None


class DummyConnection:
    def __init__(self, draft_store, action_store):
        self.store = draft_store
        self.action_store = action_store

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return DummyCursor(self.store, self.action_store)

    def commit(self):
        pass


class DummyNick:
    def __init__(self, draft_store, action_store):
        self.settings = SimpleNamespace(ses_default_sender="sender@example.com")
        self._store = draft_store
        self.action_store = action_store

    def get_db_connection(self):
        return DummyConnection(self._store, self.action_store)


def test_email_dispatch_service_sends_and_updates_status(monkeypatch):
    store = InMemoryDraftStore()
    draft_payload = {
        "rfq_id": "RFQ-UNIT",
        "subject": DEFAULT_RFQ_SUBJECT,
        "body": "<p>Hello</p>",
        "receiver": "buyer@example.com",
        "recipients": ["buyer@example.com"],
        "sender": "sender@example.com",
        "thread_index": 1,
        "contact_level": 1,
        "sent_status": False,
        "action_id": "action-1",
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

    action_store = InMemoryActionStore()
    action_store.update(
        "action-1",
        json.dumps({"drafts": [draft_payload], "rfq_id": "RFQ-UNIT", "sent_status": False}),
    )

    nick = DummyNick(store, action_store)
    service = EmailDispatchService(nick)

    sent_args = {}
    recorded_thread = {}

    def fake_record_thread(conn, message_id, rfq_id, supplier_id, recipients):
        recorded_thread.update(
            {
                "message_id": message_id,
                "rfq_id": rfq_id,
                "supplier_id": supplier_id,
                "recipients": list(recipients),
            }
        )

    def fake_send(subject, body, recipients, sender, attachments=None, **kwargs):
        sent_args.update(
            {
                "subject": subject,
                "body": body,
                "recipients": recipients,
                "sender": sender,
                "attachments": attachments,
                "headers": kwargs.get("headers"),
            }
        )
        return EmailSendResult(True, "<message-id-1>")

    monkeypatch.setattr(service.email_service, "send_email", fake_send)
    monkeypatch.setattr(service, "_record_thread_mapping", fake_record_thread)

    result = service.send_draft("RFQ-UNIT")

    assert result["sent"] is True
    assert result["recipients"] == ["buyer@example.com"]
    assert result["subject"] == DEFAULT_RFQ_SUBJECT
    assert result["draft"]["sent_status"] is True
    assert result["message_id"] == "<message-id-1>"
    assert sent_args["headers"]["X-Procwise-RFQ-ID"] == "RFQ-UNIT"
    assert result["draft"]["dispatch_metadata"]["rfq_id"] == "RFQ-UNIT"
    assert recorded_thread == {
        "message_id": "<message-id-1>",
        "rfq_id": "RFQ-UNIT",
        "supplier_id": "S1",
        "recipients": ["buyer@example.com"],
    }
    body_comment, body_visible = split_hidden_marker(result["body"])
    assert body_comment and extract_rfq_id(body_comment) == "RFQ-UNIT"
    assert "RFQ-UNIT" not in body_visible
    assert result["draft"]["dispatch_metadata"].get("dispatch_token")
    assert result["draft"]["dispatch_metadata"].get("run_id")
    assert (
        result["draft"]["dispatch_metadata"].get("dispatch_token")
        == result["draft"]["dispatch_metadata"].get("run_id")
    )
    assert result["draft"].get("dispatch_run_id") == result["draft"]["dispatch_metadata"]["run_id"]
    comment_run_id = extract_run_id(body_comment)
    assert comment_run_id == result["draft"]["dispatch_metadata"]["run_id"]
    assert store.rows[draft_id]["sent"] is True
    assert store.rows[draft_id]["recipient_email"] == "buyer@example.com"
    assert json.loads(store.rows[draft_id]["payload"])["sent_status"] is True
    updated_action = json.loads(action_store.get("action-1"))
    assert updated_action["drafts"][0]["sent_status"] == "True"
    assert updated_action["sent_status"] == "True"
    assert sent_args["recipients"] == ["buyer@example.com"]
    assert sent_args["sender"] == "sender@example.com"
    sent_comment, sent_visible = split_hidden_marker(sent_args["body"])
    assert sent_comment and extract_rfq_id(sent_comment) == "RFQ-UNIT"
    assert extract_run_id(sent_comment) == result["draft"]["dispatch_metadata"]["run_id"]
    assert "RFQ-UNIT" not in sent_visible
