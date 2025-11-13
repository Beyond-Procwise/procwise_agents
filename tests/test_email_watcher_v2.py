from __future__ import annotations

import imaplib
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentOutput, AgentStatus
from decimal import Decimal

from repositories import (
    draft_rfq_emails_repo,
    email_dispatch_repo,
    supplier_interaction_repo,
    supplier_response_repo,
    workflow_email_tracking_repo,
    workflow_lifecycle_repo,
)
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from services import email_watcher as email_watcher_v2
from services.email_watcher import EmailResponse, EmailWatcherV2, _parse_email
from services import supplier_response_coordinator
from utils.email_tracking import (
    embed_unique_id_in_email_body,
    extract_unique_id_from_body,
    generate_unique_email_id,
)


class StubSupplierAgent:
    def __init__(self) -> None:
        self.contexts: List[object] = []

    def execute(self, context):
        self.contexts.append(context)
        action = context.input_data.get("action") if isinstance(context.input_data, dict) else None
        if action == "await_workflow_batch":
            workflow_id = context.input_data.get("workflow_id")
            pending = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
            first_row = pending[0] if pending else {}
            unique_id = first_row.get("unique_id") or workflow_id
            response = {
                "unique_id": unique_id,
                "workflow_id": workflow_id,
                "supplier_id": first_row.get("supplier_id"),
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "workflow_id": workflow_id,
                    "batch_ready": True,
                    "expected_responses": 1,
                    "collected_responses": 1,
                    "supplier_responses": [response],
                    "supplier_responses_batch": [response],
                    "supplier_responses_count": 1,
                    "unique_ids": [unique_id],
                    "batch_metadata": {
                        "expected": 1,
                        "collected": 1,
                        "ready": True,
                    },
                    "negotiation_batch": True,
                },
                next_agents=["NegotiationAgent"],
            )
        return AgentOutput(status=AgentStatus.SUCCESS, data={}, next_agents=[])


class StubNegotiationAgent:
    def __init__(self) -> None:
        self.contexts: List[object] = []

    def execute(self, context):
        self.contexts.append(context)
        return AgentOutput(status=AgentStatus.SUCCESS, data={})


def test_generate_unique_email_id_produces_prefixed_ids():
    workflow_id = "wf-test"
    supplier_id = "sup-001"
    ids = {generate_unique_email_id(workflow_id, supplier_id) for _ in range(20)}
    assert all(uid.startswith("PROC-WF-") for uid in ids)
    assert len(ids) == 20


def test_embed_unique_id_is_recoverable():
    uid = "PROC-WF-ABC123DEF456"
    body = embed_unique_id_in_email_body("Hello Supplier", uid)
    assert uid in body
    assert extract_unique_id_from_body(body) == uid


def test_default_fetcher_reads_all_unseen_emails(monkeypatch):
    class _FakeIMAP:
        def __init__(self) -> None:
            self.search_calls: List[Tuple[Any, ...]] = []
            self.fetch_calls: List[int] = []

        def select(self, mailbox, readonly=True):
            assert mailbox == "INBOX"
            assert readonly is True

        def uid(self, command, *args):
            if command == "SEARCH":
                self.search_calls.append(tuple(args))
                if "UNSEEN" in args:
                    return "OK", [b"1 2 3"]
                if "ALL" in args:
                    return "OK", [b""]
                return "OK", [b""]
            if command == "FETCH":
                uid_value = int(args[0])
                self.fetch_calls.append(uid_value)
                internal = f'{uid_value} (INTERNALDATE "01-Feb-2024 12:0{uid_value}:00 +0000")'.encode()
                message = EmailMessage()
                message["Message-ID"] = f"<msg-{uid_value}>"
                message["Subject"] = f"Test {uid_value}"
                message["Date"] = f"Thu, 1 Feb 2024 12:0{uid_value}:00 +0000"
                message.set_content(f"Body {uid_value}")
                return "OK", [(internal, message.as_bytes())]
            return "NO", []

        def close(self):
            return None

        def logout(self):
            return None

    fake_client = _FakeIMAP()
    monkeypatch.setattr(
        email_watcher_v2,
        "_imap_client",
        lambda *args, **kwargs: fake_client,
    )

    since = datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc)
    responses, last_uid = email_watcher_v2._default_fetcher(
        host="imap.test",
        username="user",
        password="secret",
        mailbox="INBOX",
        since=since,
    )

    assert {resp.message_id for resp in responses} == {
        "msg-1",
        "msg-2",
        "msg-3",
    }
    assert last_uid == 3
    assert any("UNSEEN" in call for call in fake_client.search_calls)
    assert [resp.received_at for resp in responses] == sorted(
        resp.received_at for resp in responses
    )


def test_register_sent_email_includes_dispatch_timestamp(monkeypatch):
    captured: Dict[str, Any] = {}

    def _capture_outbound(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_interaction_repo, "init_schema", lambda: None)
    monkeypatch.setattr(
        supplier_interaction_repo,
        "register_outbound",
        _capture_outbound,
    )

    dispatched_at = datetime(2024, 5, 1, 9, 30, tzinfo=timezone.utc)

    email_watcher_v2.register_sent_email(
        agent_nick="watcher",
        workflow_id="wf-meta",
        supplier_id="sup-1",
        supplier_email="supplier@example.com",
        unique_id="uid-123",
        dispatched_at=dispatched_at,
    )

    row = captured["row"]
    assert row.processed_at == dispatched_at
    assert row.metadata.get("agent") == "watcher"
    assert row.metadata.get("dispatched_at") == dispatched_at.isoformat()


def test_email_watcher_v2_matches_unique_id_and_triggers_agent(tmp_path):
    workflow_id = "wf-integration-test"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    supplier_agent = StubSupplierAgent()
    negotiation_agent = StubNegotiationAgent()

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-1")
    original_subject = "Quote request"
    original_message_id = "<msg-001>"

    responses = [
        EmailResponse(
            unique_id=unique_id,
            supplier_id="sup-1",
            supplier_email="supplier@example.com",
            from_address="supplier@example.com",
            message_id="<reply-001>",
            subject=f"Re: {original_subject}",
            body=embed_unique_id_in_email_body("Thanks for the opportunity", unique_id),
            received_at=now + timedelta(minutes=5),
            in_reply_to=(original_message_id,),
            rfq_id="RFQ-2024-0001",
        )
    ]

    class _Fetcher:
        def __init__(self, payload: List[EmailResponse]):
            self.payload = payload
            self.calls = 0

        def __call__(self, *, since, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return list(self.payload)
            return []

    fetcher = _Fetcher(responses)

    watcher = EmailWatcherV2(
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-1",
                "supplier_email": "supplier@example.com",
                "message_id": original_message_id,
                "subject": original_subject,
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == 1
    assert result["workflow_status"] == "responses_complete"
    assert result["expected_responses"] == 1
    assert result["timeout_reached"] is False
    assert supplier_agent.contexts, "Supplier agent should be invoked"
    assert fetcher.calls >= 1

    context = supplier_agent.contexts[0]
    assert context.input_data.get("workflow_status") == "responses_complete"
    assert context.input_data.get("rfq_id") == "RFQ-2024-0001"
    assert context.input_data["email_headers"].get("rfq_id") == "RFQ-2024-0001"

    rows = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert rows == []

    all_rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
    assert len(all_rows) == 1
    assert all_rows[0].get("rfq_id") == "RFQ-2024-0001"


def test_email_watcher_matches_thread_headers_without_unique_id(tmp_path):
    workflow_id = "wf-thread-only"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    supplier_agent = StubSupplierAgent()
    negotiation_agent = StubNegotiationAgent()

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-1")
    original_subject = "Quote request"
    original_message_id = "<msg-002>"

    responses = [
        EmailResponse(
            unique_id=None,
            supplier_id=None,
            supplier_email="supplier@example.com",
            from_address="supplier@example.com",
            message_id="<reply-002>",
            subject=f"Re: {original_subject}",
            body="Thank you for the opportunity",
            received_at=now + timedelta(minutes=7),
            in_reply_to=("msg-002",),
            references=("msg-002",),
            rfq_id=None,
        )
    ]

    class _Fetcher:
        def __init__(self, payload: List[EmailResponse]):
            self.payload = payload
            self.calls = 0

        def __call__(self, *, since, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return list(self.payload)
            return []

    fetcher = _Fetcher(responses)

    watcher = EmailWatcherV2(
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-1",
                "supplier_email": "supplier@example.com",
                "message_id": original_message_id,
                "subject": original_subject,
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == 1
    assert result["workflow_status"] == "responses_complete"
    assert result["expected_responses"] == 1
    assert result["timeout_reached"] is False
    assert supplier_agent.contexts, "Supplier agent should be invoked"

    rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
    assert len(rows) == 1
    assert rows[0].get("response_message_id") == "<reply-002>"
    assert rows[0].get("unique_id") == unique_id

def test_email_watcher_waits_for_all_drafted_suppliers_before_activation(monkeypatch):
    workflow_id = "wf-drafted-gap"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(
        workflow_id, "awaiting_responses"
    )

    unique_ids = ["UID-A", "UID-B", "UID-C"]
    supplier_map = {uid: f"SUP-{index}" for index, uid in enumerate(unique_ids, start=1)}

    def fake_draft_lookup(*, workflow_id: str, run_id=None):  # type: ignore[override]
        assert workflow_id == "wf-drafted-gap"
        return set(unique_ids), dict(supplier_map), None

    dispatch_calls = {"count": 0}

    def fake_dispatch_count(workflow_id: str) -> int:
        assert workflow_id == "wf-drafted-gap"
        dispatch_calls["count"] += 1
        return 2

    monkeypatch.setattr(
        draft_rfq_emails_repo,
        "expected_unique_ids_and_last_dispatch",
        fake_draft_lookup,
    )
    monkeypatch.setattr(
        email_dispatch_repo,
        "count_completed_supplier_dispatches",
        fake_dispatch_count,
    )

    watcher = EmailWatcherV2(
        supplier_agent=None,
        negotiation_agent=None,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=lambda since: [],
        sleep=lambda _: None,
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is False
    assert result["workflow_status"] == "partial_timeout"
    assert result["timeout_reached"] is True
    assert result["expected_responses"] == len(unique_ids)
    assert sorted(result["pending_unique_ids"]) == sorted(unique_ids)
    assert set(result["pending_suppliers"]) == set(supplier_map.values())
    assert dispatch_calls["count"] >= 1
    assert workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id) == []


def test_email_watcher_starts_even_when_supplier_agent_inactive():
    workflow_id = "wf-wait-for-supplier"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-inactive")

    current_time = now - timedelta(seconds=1)

    def now_fn() -> datetime:
        nonlocal current_time
        current_time = current_time + timedelta(seconds=1)
        return current_time

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=lambda **_: [],
        sleep=lambda _: None,
        now=now_fn,
        response_idle_timeout_seconds=2,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-inactive",
                "supplier_email": "inactive@example.com",
                "message_id": "<dispatch-inactive>",
                "subject": "Pending request",
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is False
    assert result["workflow_status"] == "partial_timeout"
    assert result["timeout_reached"] is True
    lifecycle = workflow_lifecycle_repo.get_lifecycle(workflow_id)
    assert lifecycle is not None
    assert lifecycle.get("watcher_status") == "stopped"
    metadata = lifecycle.get("metadata") or {}
    assert metadata.get("stop_reason") == "partial_timeout"
    assert metadata.get("timeout_reached") is True


def test_email_watcher_continues_polling_until_all_responses():
    workflow_id = "wf-multi-response"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    base_time = datetime.now(timezone.utc)
    supplier_ids = ["sup-a", "sup-b", "sup-c"]
    unique_ids = [generate_unique_email_id(workflow_id, sid) for sid in supplier_ids]
    original_subject = "Pricing request"

    dispatch_payloads = []
    responses = []
    for index, (supplier_id, unique_id) in enumerate(zip(supplier_ids, unique_ids)):
        message_id = f"<dispatch-{index}>"
        dispatch_payloads.append(
            {
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "supplier_email": f"{supplier_id}@example.com",
                "message_id": message_id,
                "subject": f"{original_subject} #{index + 1}",
                "dispatched_at": base_time,
            }
        )
        responses.append(
            EmailResponse(
                unique_id=unique_id,
                supplier_id=supplier_id,
                supplier_email=f"{supplier_id}@example.com",
                from_address=f"{supplier_id}@example.com",
                message_id=f"<reply-{index}>",
                subject=f"Re: {original_subject}",
                body=embed_unique_id_in_email_body(
                    f"Response {index + 1} for {supplier_id}", unique_id
                ),
                received_at=base_time + timedelta(minutes=index + 1),
                in_reply_to=(message_id,),
                rfq_id=f"RFQ-{index + 1:04d}",
            )
        )

    class SequenceFetcher:
        def __init__(self, batches: List[List[EmailResponse]]):
            self._batches = batches
            self.calls = 0

        def __call__(self, *, since, **_kwargs):
            _ = since
            if self.calls < len(self._batches):
                payload = self._batches[self.calls]
            else:
                payload = []
            self.calls += 1
            return list(payload)

    response_batches = [
        [],
        [responses[0]],
        [],
        [],
        [responses[1]],
        [],
        [],
        [],
        [responses[2]],
        [],
    ]
    fetcher = SequenceFetcher(response_batches)

    current_time = base_time

    def fake_now() -> datetime:
        return current_time

    def fake_sleep(seconds: float) -> None:
        nonlocal current_time
        current_time += timedelta(seconds=seconds)

    watcher = EmailWatcherV2(
        supplier_agent=None,
        negotiation_agent=None,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=fake_sleep,
        now=fake_now,
        max_total_wait_seconds=1800,
    )

    watcher.register_workflow_dispatch(workflow_id, dispatch_payloads)

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == len(unique_ids)
    assert result["expected_responses"] == len(unique_ids)
    assert result["timeout_reached"] is False
    assert all(uid in result["matched_responses"] for uid in unique_ids)
    assert fetcher.calls > len(unique_ids)


def test_email_watcher_recovers_from_imap_errors():
    workflow_id = "wf-imap-error"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    base_time = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "supplier-error")
    dispatch_message_id = "<dispatch-error>"

    current_time = base_time - timedelta(seconds=2)

    def now_fn() -> datetime:
        nonlocal current_time
        current_time = current_time + timedelta(seconds=2)
        return current_time

    response_email = EmailResponse(
        unique_id=unique_id,
        supplier_id="supplier-error",
        supplier_email="supplier-error@example.com",
        from_address="supplier-error@example.com",
        message_id="<reply-error>",
        subject="Re: Pricing request",
        body="Here is our quote",
        received_at=base_time + timedelta(seconds=10),
        in_reply_to=(dispatch_message_id,),
        rfq_id="RFQ-ERROR-01",
        body_text="Here is our quote",
    )

    fetch_calls = {"count": 0}

    def fetcher(**_: Any):
        fetch_calls["count"] += 1
        if fetch_calls["count"] == 1:
            raise imaplib.IMAP4.abort("session reset")
        return [response_email]

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=now_fn,
        response_idle_timeout_seconds=30,
        max_total_wait_seconds=120,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "supplier-error",
                "supplier_email": "supplier-error@example.com",
                "message_id": dispatch_message_id,
                "subject": "Pricing request",
                "dispatched_at": base_time,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert fetch_calls["count"] >= 2
    assert result["complete"] is True
    assert result["responded_count"] == 1
    assert supplier_agent.contexts, "Supplier agent should run on RFQ fallback"
    context = supplier_agent.contexts[0]
    assert context.input_data.get("rfq_id") == rfq_identifier
    assert context.input_data.get("unique_id") == unique_id


def test_wait_for_responses_applies_adaptive_backoff(monkeypatch):
    from services import email_watcher as email_watcher_module

    workflow_id = "wf-backoff"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

    class FakeClock:
        def __init__(self) -> None:
            self.current = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def now(self) -> datetime:
            return self.current

        def advance(self, seconds: float) -> None:
            self.current += timedelta(seconds=seconds)

    clock = FakeClock()

    async def immediate(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(email_watcher_module.asyncio, "to_thread", immediate)

    durations: List[float] = []

    def fake_sleep(seconds: float) -> None:
        durations.append(seconds)
        clock.advance(seconds)

    class EmptyFetcher:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *, since: datetime) -> List[EmailResponse]:
            self.calls += 1
            return []

    fetcher = EmptyFetcher()

    watcher = EmailWatcherV2(
        supplier_agent=None,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=3,
        email_fetcher=fetcher,
        sleep=fake_sleep,
        now=clock.now,
        poll_backoff_factor=2.0,
        poll_jitter_seconds=0.0,
        poll_max_interval_seconds=4,
    )

    unique_id = generate_unique_email_id(workflow_id, "sup-backoff")
    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-backoff",
                "supplier_email": "contact@vendor.example",
                "message_id": "<dispatch-backoff>",
                "subject": "RFQ",
                "dispatched_at": clock.now(),
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is False
    assert result["status"] == "max_attempts_exceeded"
    assert result.get("timeout_reason") == "max_attempts"
    assert result.get("poll_attempts") == 3
    assert fetcher.calls == 3
    assert durations == [1.0, 2.0]
