import json
import os
import sys
import types
from types import SimpleNamespace
from typing import List

os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_backend_scheduler_stub = types.ModuleType("services.backend_scheduler")


class _BackendSchedulerProxy:
    notifications: List[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.notifications.clear()

    @classmethod
    def ensure(cls, agent_nick):
        class _Notifier:
            def notify_email_dispatch(self, workflow_id: str):
                if workflow_id:
                    cls.notifications.append(workflow_id)

        return _Notifier()


_backend_scheduler_stub.BackendScheduler = _BackendSchedulerProxy
sys.modules.setdefault("services.backend_scheduler", _backend_scheduler_stub)

from services.email_dispatch_service import EmailDispatchService
from services.email_service import EmailSendResult
from repositories import workflow_email_tracking_repo
from utils.email_tracking import extract_tracking_metadata, extract_unique_id_from_body


DEFAULT_RFQ_SUBJECT = "Sourcing Request – Pricing Discussion"


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
            row.get("workflow_id"),
            row.get("run_id"),
            row.get("unique_id"),
            row.get("mailbox"),
            row.get("dispatch_run_id"),
            row.get("dispatched_at"),
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
                workflow_id,
                run_id,
                unique_id,
                mailbox,
                dispatch_run_id,
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
    _BackendSchedulerProxy.reset()
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

    dispatch_context = {
        "workflow_id": "wf-test-1",
        "unique_id": unique_id,
        "dispatch_key": "neg-round-1",
    }
    result = service.send_draft(
        "RFQ-UNIT",
        workflow_dispatch_context=dispatch_context,
    )

    assert result["sent"] is True
    assert result["unique_id"] == unique_id
    assert result["recipients"] == ["buyer@example.com"]
    assert result["subject"] == DEFAULT_RFQ_SUBJECT
    assert result["draft"]["sent_status"] is True
    assert result["message_id"] == "<message-id-1>"
    headers_lower = {k.lower(): v for k, v in sent_args["headers"].items()}
    assert "x-procwise-rfq-id" not in headers_lower
    unique_id = extract_unique_id_from_body(result["body"])
    assert unique_id
    assert headers_lower["x-procwise-unique-id"] == unique_id
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
    assert result["workflow_email"] is True
    assert result["workflow_context"] == dispatch_context
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
    assert _BackendSchedulerProxy.notifications == [metadata.workflow_id]


def test_email_dispatch_service_records_workflow_even_when_flag_false(monkeypatch):
    _BackendSchedulerProxy.reset()
    store = InMemoryDraftStore()
    unique_id = "PROC-WF-NOLOG-001"
    draft_payload = {
        "rfq_id": "RFQ-NOLOG",
        "subject": DEFAULT_RFQ_SUBJECT,
        "body": "<p>Skip</p>",
        "receiver": "buyer@example.com",
        "recipients": ["buyer@example.com"],
        "sender": "sender@example.com",
        "thread_index": 1,
        "contact_level": 1,
        "sent_status": False,
        "action_id": "action-2",
        "workflow_id": "wf-no-track",
        "unique_id": unique_id,
    }
    store.add(
        {
            "rfq_id": "RFQ-NOLOG",
            "supplier_id": "S2",
            "supplier_name": "NoTrack",
            "subject": draft_payload["subject"],
            "body": draft_payload["body"],
            "sent": False,
            "recipient_email": None,
            "contact_level": 0,
            "thread_index": 1,
            "sender": "sender@example.com",
            "payload": json.dumps(draft_payload),
            "sent_on": None,
            "workflow_id": "wf-no-track",
            "unique_id": unique_id,
        }
    )

    action_store = InMemoryActionStore()
    action_store.update(
        "action-2",
        json.dumps(
            {
                "drafts": [draft_payload],
                "rfq_id": "RFQ-NOLOG",
                "unique_id": unique_id,
                "sent_status": False,
            }
        ),
    )

    nick = DummyNick(store, action_store)
    service = EmailDispatchService(nick)

    def fake_send(subject, body, recipients, sender, attachments=None, **kwargs):
        return EmailSendResult(True, "<message-id-nolog>")

    monkeypatch.setattr(service.email_service, "send_email", fake_send)

    result = service.send_draft(
        "RFQ-NOLOG",
        is_workflow_email=False,
        workflow_dispatch_context={
            "workflow_id": "wf-no-track",
            "unique_id": unique_id,
        },
    )

    assert result["workflow_email"] is False

    workflow_email_tracking_repo.init_schema()
    stored_rows = workflow_email_tracking_repo.load_workflow_rows(
        workflow_id="wf-no-track"
    )
    assert any(row.unique_id == unique_id for row in stored_rows)
    assert _BackendSchedulerProxy.notifications == ["wf-no-track"]
