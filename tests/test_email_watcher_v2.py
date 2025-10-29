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
    supplier_response_repo,
    workflow_email_tracking_repo,
    workflow_lifecycle_repo,
)
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from services import email_watcher_v2
from services.email_watcher_v2 import EmailResponse, EmailWatcherV2, _parse_email
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
    assert result["workflow_status"] == "awaiting_dispatch_confirmation"
    assert result["expected_responses"] == len(unique_ids)
    assert sorted(result["pending_unique_ids"]) == sorted(unique_ids)
    assert set(result["pending_suppliers"]) == set(supplier_map.values())
    assert dispatch_calls["count"] >= 1


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
    assert result["workflow_status"] == "responses_complete"
    rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
    assert len(rows) == 1
    assert rows[0]["response_message_id"].strip("<>") == "reply-error"


def test_email_watcher_stops_after_inactivity_timeout(monkeypatch):
    workflow_id = "wf-timeout"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(
        workflow_id, "awaiting_responses"
    )

    unique_id = generate_unique_email_id(workflow_id, "sup-timeout")

    class _FakeBus:
        def __init__(self) -> None:
            self.events: List[Tuple[str, Dict[str, Any]]] = []

        def publish(self, event: str, payload: Dict[str, Any]) -> None:
            self.events.append((event, dict(payload)))

    fake_bus = _FakeBus()
    monkeypatch.setattr(email_watcher_v2, "get_event_bus", lambda: fake_bus)

    current_time = datetime.now(timezone.utc)

    def fake_now() -> datetime:
        return current_time

    def fake_sleep(seconds: float) -> None:
        nonlocal current_time
        current_time += timedelta(seconds=seconds)

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=10,
        max_poll_attempts=1,
        email_fetcher=lambda **_: [],
        sleep=fake_sleep,
        now=fake_now,
        response_idle_timeout_seconds=120,
        max_total_wait_seconds=3600,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-timeout",
                "supplier_email": "timeout@example.com",
                "message_id": "<timeout-dispatch>",
                "subject": "Timeout scenario",
                "dispatched_at": current_time,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["workflow_status"] == "partial_timeout"
    assert result["timeout_reached"] is True
    assert result["complete"] is False
    assert result["pending_unique_ids"] == [unique_id]
    assert result["pending_suppliers"] == ["sup-timeout"]
    assert any(event == "responses_timeout" for event, _ in fake_bus.events)


def test_email_watcher_stops_when_negotiation_completed():
    workflow_id = "wf-negotiation-complete"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-negotiated")

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=lambda **_: [],
        sleep=lambda _: workflow_lifecycle_repo.record_negotiation_status(
            workflow_id, "completed"
        ),
        now=lambda: now,
        max_total_wait_seconds=30,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-negotiated",
                "supplier_email": "negotiated@example.com",
                "message_id": "<dispatch-negotiated>",
                "subject": "Request for negotiation",
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is False
    assert result["workflow_status"] == "negotiation_completed_pending_responses"
    assert result["timeout_reached"] is False
    lifecycle = workflow_lifecycle_repo.get_lifecycle(workflow_id)
    assert lifecycle is not None
    assert lifecycle.get("watcher_status") == "stopped"
    assert lifecycle.get("negotiation_status") == "completed"
    metadata = lifecycle.get("metadata") or {}
    assert metadata.get("stop_reason") == "negotiation_completed"
    assert metadata.get("negotiation_completed") is True
    assert metadata.get("status") == "incomplete"


def test_email_watcher_flags_completed_negotiation_with_pending_responses():
    workflow_id = "wf-negotiation-pending-responses"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(
        workflow_id, "awaiting_responses"
    )
    workflow_lifecycle_repo.record_negotiation_status(workflow_id, "completed")

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-outstanding")

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=lambda **_: [],
        sleep=lambda _: None,
        now=lambda: now,
        max_total_wait_seconds=30,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-outstanding",
                "supplier_email": "pending@example.com",
                "message_id": "<dispatch-pending>",
                "subject": "Request for negotiation",
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is False
    assert result["workflow_status"] == "negotiation_completed_pending_responses"
    assert result["pending_unique_ids"] == [unique_id]
    assert result["pending_suppliers"] == ["sup-outstanding"]

def test_email_watcher_v2_matches_legacy_bracketed_message_id(tmp_path):
    workflow_id = "wf-legacy-thread"
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
    unique_id = generate_unique_email_id(workflow_id, "sup-legacy")
    legacy_message_id = "<legacy-msg-001@procwise>"

    workflow_email_tracking_repo.record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            WorkflowDispatchRow(
                workflow_id=workflow_id,
                unique_id=unique_id,
                supplier_id="sup-legacy",
                supplier_email="legacy@example.com",
                message_id=legacy_message_id,
                subject="Legacy Quote Request",
                dispatched_at=now - timedelta(minutes=15),
            )
        ],
    )

    responses = [
        EmailResponse(
            unique_id=None,
            supplier_id="sup-legacy",
            supplier_email="legacy@example.com",
            from_address="legacy@example.com",
            message_id="reply-legacy-001",
            subject="Re: Legacy Quote Request",
            body="Legacy response body",
            received_at=now,
            in_reply_to=("legacy-msg-001@procwise",),
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
        max_poll_attempts=1,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == 1
    assert result["workflow_status"] == "responses_complete"
    assert result["expected_responses"] == 1
    assert result["timeout_reached"] is False

    rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
    assert len(rows) == 1
    persisted = rows[0]
    assert persisted.get("original_message_id") == "<legacy-msg-001@procwise>"
    assert persisted.get("response_message_id") == "reply-legacy-001"

def test_parse_email_recovers_tracking_headers():
    message = EmailMessage()
    message["Subject"] = "Re: RFQ details"
    message["From"] = "Acme Quotes <quotes@acme.example>"
    message["To"] = "buyer@procwise.test"
    message["X-ProcWise-Unique-ID"] = "UID-HEADER12345"
    message["X-ProcWise-Workflow-ID"] = "wf-123"
    message["X-ProcWise-Supplier-ID"] = "sup-123"
    message["X-ProcWise-RFQ-ID"] = "RFQ-12345"
    message["X-ProcWise-Round"] = "0"
    message.set_content("Thank you for the opportunity")

    parsed = _parse_email(message.as_bytes())

    assert parsed.unique_id == "UID-HEADER12345"
    assert parsed.workflow_id == "wf-123"
    assert parsed.supplier_id == "sup-123"
    assert parsed.rfq_id == "RFQ-12345"
    assert parsed.round_number == 0


def test_parse_email_extracts_plain_and_html_bodies():
    message = EmailMessage()
    message["Subject"] = "Re: Pricing"
    message["From"] = "quotes@example.com"
    message["To"] = "buyer@example.com"
    message.set_content("Plain body text")
    message.add_alternative("<p>Plain body text</p>", subtype="html")

    parsed = _parse_email(message.as_bytes())

    assert parsed.body == "Plain body text"
    assert parsed.body_text == "Plain body text"
    assert "<p>Plain body text</p>" in (parsed.body_html or "")


def test_email_watcher_matches_using_supplier_id_when_threshold_not_met(tmp_path):
    workflow_id = "wf-header-fallback"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    supplier_agent = StubSupplierAgent()

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-55")

    responses = [
        EmailResponse(
            unique_id=None,
            supplier_id="sup-55",
            supplier_email="reply@alt-domain.example",
            from_address="reply@alt-domain.example",
            message_id="<reply-alt>",
            subject="Follow up",
            body="Looking forward to working together",
            received_at=now + timedelta(minutes=2),
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
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
        match_threshold=0.7,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-55",
                "supplier_email": "contact@acme.example",
                "message_id": "<msg-55>",
                "subject": "RFQ opportunity",
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
    assert supplier_agent.contexts, "Supplier agent should still be invoked via fallback"


def test_email_watcher_matches_using_rfq_id_when_unique_id_missing(tmp_path):
    workflow_id = "wf-rfq-fallback"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    supplier_agent = StubSupplierAgent()

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-99")
    rfq_identifier = "RFQ-2024-0099"

    responses = [
        EmailResponse(
            unique_id=None,
            supplier_id=None,
            supplier_email="reply@vendor.example",
            from_address="reply@vendor.example",
            message_id="<reply-rfq>",
            subject=f"Re: Opportunity {rfq_identifier}",
            body="Here is our quote",
            received_at=now + timedelta(minutes=4),
            rfq_id=rfq_identifier,
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
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
        match_threshold=0.8,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "rfq_id": rfq_identifier,
                "supplier_email": "contact@vendor.example",
                "message_id": "<dispatch-rfq>",
                "subject": f"Initial request {rfq_identifier}",
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
    assert supplier_agent.contexts, "Supplier agent should run on RFQ fallback"
    context = supplier_agent.contexts[0]
    assert context.input_data.get("workflow_status") == "responses_complete"
    assert context.input_data.get("rfq_id") == rfq_identifier
    assert context.input_data.get("unique_id") == unique_id


def test_email_watcher_does_not_dedup_distinct_message_ids_with_same_fingerprint(tmp_path):
    workflow_id = "wf-fingerprint"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    workflow_lifecycle_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)
    workflow_lifecycle_repo.reset_workflow(workflow_id)
    workflow_lifecycle_repo.record_supplier_agent_status(workflow_id, "awaiting_responses")

    supplier_agent = StubSupplierAgent()

    now = datetime.now(timezone.utc)
    unique_a = generate_unique_email_id(workflow_id, "sup-a")
    unique_b = generate_unique_email_id(workflow_id, "sup-b")

    subject = "Re: Pricing update"
    from_address = "aggregator@example.com"

    responses = [
        EmailResponse(
            unique_id=unique_a,
            supplier_id="sup-a",
            supplier_email=from_address,
            from_address=from_address,
            message_id="<reply-a>",
            subject=subject,
            body="Quote details A",
            received_at=now + timedelta(seconds=30),
        ),
        EmailResponse(
            unique_id=unique_b,
            supplier_id="sup-b",
            supplier_email=from_address,
            from_address=from_address,
            message_id="<reply-b>",
            subject=subject,
            body="Quote details B",
            received_at=now + timedelta(seconds=70),
        ),
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
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
        match_threshold=0.8,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_a,
                "supplier_id": "sup-a",
                "supplier_email": from_address,
                "message_id": "<dispatch-a>",
                "subject": subject,
                "dispatched_at": now,
            },
            {
                "unique_id": unique_b,
                "supplier_id": "sup-b",
                "supplier_email": from_address,
                "message_id": "<dispatch-b>",
                "subject": subject,
                "dispatched_at": now,
            },
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == 2
    assert result["expected_responses"] == 2
    assert result["workflow_status"] == "responses_complete"
    assert sorted(result["matched_responses"].keys()) == sorted([unique_a, unique_b])
    assert supplier_agent.contexts, "Supplier agent should receive completion context"


def test_email_fetcher_tracks_last_seen_uid(monkeypatch):
    workflow_id = "wf-last-uid"
    now = datetime.now(timezone.utc)

    response = EmailResponse(
        unique_id=generate_unique_email_id(workflow_id, "sup-late"),
        supplier_id="sup-late",
        supplier_email="late@example.com",
        from_address="late@example.com",
        message_id="<reply-last-uid>",
        subject="Re: Update",
        body="Responding with updated pricing",
        received_at=now,
    )

    calls = []

    def fetcher(**kwargs):
        calls.append(kwargs)
        if kwargs.get("last_seen_uid") is None:
            return {"responses": [response], "last_uid": 5123}
        return []

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": response.unique_id,
                "supplier_id": "sup-late",
                "supplier_email": "late@example.com",
                "message_id": "<dispatch-last-uid>",
                "subject": "Request",
                "dispatched_at": now - timedelta(minutes=5),
            }
        ],
    )

    first_batch = watcher._fetch_emails(now, workflow_ids=[workflow_id])
    assert first_batch == [response]
    assert watcher._imap_last_seen_uid == 5123

    second_batch = watcher._fetch_emails(now, workflow_ids=[workflow_id])
    assert second_batch == []
    assert len(calls) >= 2
    assert calls[0].get("last_seen_uid") is None
    assert calls[1].get("last_seen_uid") == 5123
    assert calls[0].get("workflow_ids") == [workflow_id]
    assert calls[1].get("workflow_ids") == [workflow_id]


def test_default_fetcher_returns_responses_before_last_seen(monkeypatch):
    now = datetime(2025, 10, 29, 9, 45, tzinfo=timezone.utc)

    message = EmailMessage()
    message["Subject"] = "Re: ProcWise RFQ"
    message["From"] = "supplier@example.com"
    message["To"] = "buyer@example.com"
    message["Message-ID"] = "<reply-before-last@procwise>"
    message.set_content("Pricing attached")
    raw_bytes = message.as_bytes()

    class _FakeIMAP:
        def __init__(self) -> None:
            self.selected = False
            self.closed = False

        def select(self, mailbox, readonly=True):  # pragma: no cover - trivial
            self.selected = True
            return "OK", [b"1"]

        def uid(self, command, charset, *criteria):
            if command == "SEARCH":
                tokens = [
                    c.decode() if isinstance(c, bytes) else str(c)
                    for c in criteria
                ]
                if "UID" in tokens:
                    return "OK", [b""]
                return "OK", [b"150"]
            if command == "FETCH":
                uid_token = charset
                if isinstance(uid_token, bytes):
                    uid = int(uid_token.decode())
                else:
                    uid = int(str(uid_token))
                if uid != 150:
                    return "OK", []
                internal = b'1 (INTERNALDATE "29-Oct-2025 09:40:00 +0000" RFC822 {0})'
                return "OK", [(internal, raw_bytes)]
            raise AssertionError(f"Unsupported command: {command}")

        def close(self):  # pragma: no cover - defensive
            self.closed = True

        def logout(self):  # pragma: no cover - defensive
            self.closed = True

    monkeypatch.setattr(
        email_watcher_v2,
        "_imap_client",
        lambda *args, **kwargs: _FakeIMAP(),
    )

    responses, last_uid = email_watcher_v2._default_fetcher(
        host="imap.test",
        username="user@test",
        password="secret",
        mailbox="INBOX",
        since=now,
        last_seen_uid=400,
        workflow_ids=["wf-backfill"],
    )

    assert len(responses) == 1
    response = responses[0]
    assert response.message_id == "reply-before-last@procwise"
    assert response.received_at is not None
    assert last_uid == 400


@pytest.fixture(autouse=True)
def _stub_response_coordinator(monkeypatch):
    class _Coordinator:
        def __init__(self) -> None:
            self.registered = []
            self.recorded = []

        def register_expected_responses(
            self,
            workflow_id,
            unique_ids,
            expected_count,
            *,
            round_number=None,
        ):
            self.registered.append(
                (workflow_id, list(unique_ids), expected_count, round_number)
            )

        def record_response(self, workflow_id, unique_id, *, round_number=None):
            self.recorded.append((workflow_id, unique_id, round_number))

        def await_completion(  # pragma: no cover - stub
            self, workflow_id, timeout, *, round_number=None
        ):
            return None

        def clear(self, workflow_id, *, round_number=None):  # pragma: no cover - stub
            return None

    coordinator = _Coordinator()
    monkeypatch.setattr(
        supplier_response_coordinator,
        "get_supplier_response_coordinator",
        lambda: coordinator,
    )
    return coordinator
