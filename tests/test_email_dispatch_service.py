import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_backend_scheduler_stub = types.ModuleType("services.backend_scheduler")


class _DummyScheduler:
    @classmethod
    def ensure(cls, *_args, **_kwargs):
        return cls()

    def notify_email_dispatch(self, *args, **kwargs):  # pragma: no cover - stub
        return None


_backend_scheduler_stub.BackendScheduler = _DummyScheduler
sys.modules.setdefault("services.backend_scheduler", _backend_scheduler_stub)

DEFAULT_RFQ_SUBJECT = "Sourcing Request – Pricing Discussion"

from services.email_dispatch_service import EmailDispatchService, DraftNotFoundError
from services.email_service import EmailSendResult
from repositories import workflow_email_tracking_repo
from utils.email_tracking import extract_tracking_metadata, extract_unique_id_from_body


class InMemoryDraftStore:
    def __init__(self):
        self.rows = {}
        self.next_id = 1

    def add(self, record):
        record = dict(record)
        record.setdefault("id", self.next_id)
        record.setdefault("review_status", "PENDING")
        self.rows[record["id"]] = record
        self.next_id += 1
        return record["id"]

    def _row_to_tuple(self, row):
        if row is None:
            return None
        return (
            row["id"],
            row["rfq_id"],
            row.get("supplier_id"),
            row.get("supplier_name"),
            row.get("subject"),
            row.get("body"),
            row.get("sent", False),
            row.get("review_status", "PENDING"),
            row.get("recipient_email"),
            row.get("contact_level", 0),
            row.get("thread_index", 1),
            row.get("payload"),
            row.get("sender"),
            row.get("sent_on"),
            row.get("workflow_id"),
            row.get("run_id"),
            row.get("unique_id"),
            row.get("mailbox"),
            row.get("dispatch_run_id"),
            row.get("dispatched_at"),
        )

    def get_latest(self, identifier):
        candidates = [
            row
            for row in self.rows.values()
            if row.get("unique_id") == identifier or row["rfq_id"] == identifier
        ]
        if not candidates:
            return None
        # Order by sent flag then thread index then id to mirror SQL
        candidates.sort(key=lambda row: (row["sent"], -row["thread_index"], -row["id"]))
        return self._row_to_tuple(candidates[0])

    def get_latest_by_workflow(self, workflow_id, *, unsent_only=False):
        candidates = [
            row for row in self.rows.values() if row.get("workflow_id") == workflow_id
        ]
        if unsent_only:
            candidates = [row for row in candidates if not row.get("sent")]
        if not candidates:
            return None

        if unsent_only:
            candidates.sort(
                key=lambda row: (
                    0
                    if str(row.get("review_status", "")).upper() in {"APPROVED", "SAVED"}
                    else 1,
                    -row.get("thread_index", 0),
                    -row["id"],
                )
            )
        else:
            candidates.sort(
                key=lambda row: (
                    row.get("sent", False),
                    -row.get("thread_index", 0),
                    -row["id"],
                )
            )
        return self._row_to_tuple(candidates[0])

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
            if "WHERE workflow_id = %s AND (sent = FALSE OR sent IS NULL)" in normalized:
                workflow_id = params[0]
                row = self.store.get_latest_by_workflow(workflow_id, unsent_only=True)
            elif "WHERE workflow_id = %s" in normalized:
                workflow_id = params[0]
                row = self.store.get_latest_by_workflow(workflow_id, unsent_only=False)
            else:
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
                workflow_id,
                run_id,
                unique_id,
                mailbox,
                dispatch_run_id,
                review_toggle,
                dispatched_toggle,
                sent_toggle,
                draft_id,
            ) = params
            updates = {
                "sent": bool(sent_flag),
                "subject": subject,
                "body": body,
                "recipient_email": recipient,
                "contact_level": contact_level,
                "payload": payload_json,
                "workflow_id": workflow_id,
                "run_id": run_id,
                "unique_id": unique_id,
                "mailbox": mailbox,
                "dispatch_run_id": dispatch_run_id,
            }
            updates["supplier_id"] = supplier_id
            updates["supplier_name"] = supplier_name
            updates["rfq_id"] = rfq_value
            if review_toggle:
                updates["review_status"] = "SENT"
            if sent_toggle:
                updates["sent_on"] = "now"
            self.store.update(draft_id, **updates)
        elif normalized.startswith("CREATE TABLE IF NOT EXISTS proc.email_dispatch_chains"):
            return
        elif normalized.startswith("ALTER TABLE proc.email_dispatch_chains"):
            return
        elif normalized.startswith("INSERT INTO proc.email_dispatch_chains"):
            return
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
    unique_id = "PROC-WF-UNIT-12345"
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
        "review_status": "APPROVED",
        "action_id": "action-1",
        "workflow_id": "wf-test-1",
        "unique_id": unique_id,
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
                "workflow_id": "wf-test-1",
                "unique_id": unique_id,
            }
        )

    action_store = InMemoryActionStore()
    action_store.update(
        "action-1",
        json.dumps(
            {
                "drafts": [draft_payload],
                "rfq_id": "RFQ-UNIT",
                "unique_id": unique_id,
                "sent_status": False,
            }
        ),
    )

    nick = DummyNick(store, action_store)
    service = EmailDispatchService(nick)

    sent_args = {}
    recorded_thread = {}

    def fake_record_thread(conn, message_id, unique_id_value, supplier_id, recipients):
        recorded_thread.update(
            {
                "message_id": message_id,
                "unique_id": unique_id_value,
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
    assert result["unique_id"] == unique_id
    assert result["recipients"] == ["buyer@example.com"]
    assert result["subject"] == DEFAULT_RFQ_SUBJECT
    assert result["draft"]["sent_status"] is True
    assert result["message_id"] == "<message-id-1>"
    assert "X-Procwise-RFQ-ID" not in sent_args["headers"]
    unique_id = extract_unique_id_from_body(result["body"])
    assert unique_id
    assert sent_args["headers"]["X-Procwise-Unique-Id"] == unique_id
    if sent_args["headers"].get("X-Procwise-Workflow-Id"):
        assert (
            sent_args["headers"]["X-Procwise-Workflow-Id"]
            == result["draft"]["dispatch_metadata"].get("workflow_id")
        )
    assert result["draft"]["dispatch_metadata"]["unique_id"] == unique_id
    assert recorded_thread == {
        "message_id": "<message-id-1>",
        "unique_id": unique_id,
        "supplier_id": "S1",
        "recipients": ["buyer@example.com"],
    }
    metadata = extract_tracking_metadata(result["body"])
    assert metadata is not None
    assert metadata.unique_id == unique_id
    assert metadata.workflow_id == result["draft"]["dispatch_metadata"].get("workflow_id")
    assert metadata.run_id == result["draft"]["dispatch_metadata"].get("run_id")
    assert result["draft"]["dispatch_metadata"].get("dispatch_token")
    assert result["draft"]["dispatch_metadata"].get("run_id")
    assert (
        result["draft"]["dispatch_metadata"].get("dispatch_token")
        == result["draft"]["dispatch_metadata"].get("run_id")
    )
    assert result["draft"].get("dispatch_run_id") == result["draft"]["dispatch_metadata"]["run_id"]
    visible_body = result["body"].split("-->", 2)[-1]
    assert "RFQ-UNIT" not in visible_body
    assert store.rows[draft_id]["sent"] is True
    assert store.rows[draft_id]["recipient_email"] == "buyer@example.com"
    assert json.loads(store.rows[draft_id]["payload"])["sent_status"] is True
    updated_action = json.loads(action_store.get("action-1"))
    assert updated_action["drafts"][0]["sent_status"] == "True"
    assert updated_action["sent_status"] == "True"
    assert sent_args["recipients"] == ["buyer@example.com"]
    assert sent_args["sender"] == "sender@example.com"
    sent_unique_id = extract_unique_id_from_body(sent_args["body"])
    assert sent_unique_id == unique_id
    sent_metadata = extract_tracking_metadata(sent_args["body"])
    assert sent_metadata is not None
    assert sent_metadata.unique_id == unique_id
    assert sent_metadata.workflow_id == metadata.workflow_id
    assert sent_metadata.run_id == metadata.run_id
    visible_sent = sent_args["body"].split("-->", 2)[-1]
    assert "RFQ-UNIT" not in visible_sent

    workflow_email_tracking_repo.init_schema()
    stored_rows = workflow_email_tracking_repo.load_workflow_rows(
        workflow_id=metadata.workflow_id,
    )
    assert any(
        row.unique_id == unique_id and row.message_id == "<message-id-1>"
        for row in stored_rows
    )


def test_dispatch_from_context_resolves_workflow_and_normalises(monkeypatch):
    store = InMemoryDraftStore()
    workflow_id = "wf-context-42"
    unique_unsent = "PROC-WF-CONTEXT-001"
    unsent_payload = {
        "rfq_id": "RFQ-CONTEXT",
        "subject": "Stored Subject",
        "body": "<p>Stored Body</p>",
        "recipients": [" buyer@example.com ", "Buyer@example.com"],
        "sender": "stored@example.com",
        "thread_index": 2,
        "workflow_id": workflow_id,
        "unique_id": unique_unsent,
    }

    store.add(
        {
            "rfq_id": "RFQ-CONTEXT",
            "supplier_id": "SUP-CTX",
            "supplier_name": "Context Corp",
            "subject": unsent_payload["subject"],
            "body": unsent_payload["body"],
            "sent": False,
            "recipient_email": None,
            "contact_level": 0,
            "thread_index": 2,
            "sender": unsent_payload["sender"],
            "payload": json.dumps(unsent_payload),
            "workflow_id": workflow_id,
            "unique_id": unique_unsent,
            "review_status": "APPROVED",
        }
    )

    store.add(
        {
            "rfq_id": "RFQ-CONTEXT",
            "supplier_id": "SUP-CTX",
            "supplier_name": "Context Corp",
            "subject": "Older Subject",
            "body": "Older Body",
            "sent": True,
            "recipient_email": "old@example.com",
            "contact_level": 0,
            "thread_index": 1,
            "sender": "old@example.com",
            "payload": json.dumps({"unique_id": "PROC-WF-CONTEXT-OLD"}),
            "workflow_id": workflow_id,
            "unique_id": "PROC-WF-CONTEXT-OLD",
            "review_status": "SENT",
        }
    )

    nick = DummyNick(store, InMemoryActionStore())
    service = EmailDispatchService(nick)

    captured = {}

    def fake_send_draft(
        identifier,
        recipients=None,
        sender=None,
        subject_override=None,
        body_override=None,
        attachments=None,
    ):
        captured.update(
            {
                "identifier": identifier,
                "recipients": recipients,
                "sender": sender,
                "subject_override": subject_override,
                "body_override": body_override,
                "attachments": attachments,
            }
        )
        return {
            "identifier": identifier,
            "recipients": recipients,
            "sender": sender,
            "subject_override": subject_override,
            "body_override": body_override,
            "attachments": attachments,
        }

    monkeypatch.setattr(service, "send_draft", fake_send_draft)

    overrides = {
        "recipients": " Buyer@example.com ",
        "sender": "  sender2@example.com ",
        "subject_override": "  Negotiation Update  ",
        "body_override": "  <p>Body</p>  ",
        "attachments": [(b"data", "quote.pdf")],
    }

    result = service.dispatch_from_context(workflow_id, overrides=overrides)

    assert captured["identifier"] == unique_unsent
    assert captured["recipients"] == ["Buyer@example.com"]
    assert captured["sender"] == "sender2@example.com"
    assert captured["subject_override"] == "Negotiation Update"
    assert captured["body_override"] == "<p>Body</p>"
    assert captured["attachments"] == overrides["attachments"]

    assert result["identifier"] == unique_unsent
    assert result["recipients"] == ["Buyer@example.com"]
    assert result["sender"] == "sender2@example.com"
    assert result["subject_override"] == "Negotiation Update"
    assert result["body_override"] == "<p>Body</p>"


def test_dispatch_from_context_raises_when_workflow_only_has_sent(monkeypatch):
    store = InMemoryDraftStore()
    workflow_id = "wf-context-sent"
    store.add(
        {
            "rfq_id": "RFQ-SENT",
            "supplier_id": "SUP-SENT",
            "supplier_name": "Sent Supplier",
            "subject": "Sent Subject",
            "body": "<p>Sent Body</p>",
            "sent": True,
            "recipient_email": "old@example.com",
            "contact_level": 0,
            "thread_index": 1,
            "sender": "old@example.com",
            "payload": json.dumps({"unique_id": "PROC-WF-SENT"}),
            "workflow_id": workflow_id,
            "unique_id": "PROC-WF-SENT",
            "review_status": "SENT",
        }
    )

    nick = DummyNick(store, InMemoryActionStore())
    service = EmailDispatchService(nick)

    def _unexpected_send(*_args, **_kwargs):  # pragma: no cover - defensive
        raise AssertionError("send_draft should not be called")

    monkeypatch.setattr(service, "send_draft", _unexpected_send)

    with pytest.raises(DraftNotFoundError):
        service.dispatch_from_context(workflow_id)
