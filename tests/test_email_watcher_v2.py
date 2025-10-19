from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Dict, List, Set

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentOutput, AgentStatus
from decimal import Decimal

from repositories import supplier_response_repo, workflow_email_tracking_repo
from services.email_watcher_v2 import (
    EmailDispatchRecord,
    EmailResponse,
    EmailWatcherV2,
    _parse_email,
)
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


class QueueingFetcher:
    def __init__(self) -> None:
        self._payloads: List[List[EmailResponse]] = []

    def push(self, responses: List[EmailResponse]) -> None:
        self._payloads.append(list(responses))

    def __call__(self, *, since):
        if self._payloads:
            return self._payloads.pop(0)
        return []


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
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

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

        def __call__(self, *, since):
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
    assert supplier_agent.contexts, "Supplier agent should be invoked"
    assert fetcher.calls >= 1

    context = supplier_agent.contexts[0]
    assert context.input_data.get("rfq_id") == "RFQ-2024-0001"
    assert context.input_data["email_headers"].get("rfq_id") == "RFQ-2024-0001"

    rows = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert rows == []


def test_email_watcher_waits_for_all_expected_dispatches(monkeypatch):
    workflow_id = "wf-await-dispatch"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

    supplier_agent = StubSupplierAgent()

    now = datetime.now(timezone.utc)
    unique_id_primary = generate_unique_email_id(workflow_id, "sup-1")
    unique_id_secondary = generate_unique_email_id(workflow_id, "sup-2")

    sent_unique_ids = set()

    def fake_expectations(*, workflow_id: str, run_id=None):
        assert workflow_id == "wf-await-dispatch"
        pending = {unique_id_primary, unique_id_secondary} - sent_unique_ids
        return (
            {unique_id_primary, unique_id_secondary},
            {unique_id_primary: "sup-1", unique_id_secondary: "sup-2"},
            now,
            pending,
        )

    monkeypatch.setattr(
        "services.email_watcher_v2.draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch_with_pending",
        fake_expectations,
    )

    pending_actions: List[Callable[[], None]] = []

    def custom_sleep(_seconds: float) -> None:
        if pending_actions:
            action = pending_actions.pop(0)
            action()

    class _Fetcher:
        def __init__(self) -> None:
            self.calls = 0
            self.dispatch_counts: List[int] = []

        def __call__(self, *, since):
            self.calls += 1
            rows = workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id)
            self.dispatch_counts.append(len(rows))
            return []

    fetcher = _Fetcher()

    watcher = EmailWatcherV2(
        supplier_agent=supplier_agent,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=3,
        email_fetcher=fetcher,
        sleep=custom_sleep,
        now=lambda: now,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id_primary,
                "supplier_id": "sup-1",
                "supplier_email": "primary@example.com",
                "message_id": "<msg-001>",
                "subject": "Initial RFQ",
                "dispatched_at": now,
            }
        ],
    )
    sent_unique_ids.add(unique_id_primary)

    def _register_secondary():
        watcher.register_workflow_dispatch(
            workflow_id,
            [
                {
                    "unique_id": unique_id_secondary,
                    "supplier_id": "sup-2",
                    "supplier_email": "secondary@example.com",
                    "message_id": "<msg-002>",
                    "subject": "Follow-up RFQ",
                    "dispatched_at": now + timedelta(minutes=1),
                }
            ],
        )
        sent_unique_ids.add(unique_id_secondary)

    pending_actions.append(_register_secondary)

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["dispatched_count"] == 2
    assert fetcher.dispatch_counts, "Fetcher should have been invoked"
    assert fetcher.dispatch_counts[0] == 2, "Fetcher should poll only after all dispatches are recorded"
    assert not pending_actions, "All pending dispatch actions should have executed"


def test_email_watcher_waits_for_pending_unsent_dispatches(monkeypatch):
    workflow_id = "wf-await-unsent"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

    supplier_agent = StubSupplierAgent()

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-unsent")

    pending_unsent = {unique_id}

    def fake_expectations(*, workflow_id: str, run_id=None):
        assert workflow_id == "wf-await-unsent"
        return ({unique_id}, {unique_id: "sup-unsent"}, now, set(pending_unsent))

    monkeypatch.setattr(
        "services.email_watcher_v2.draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch_with_pending",
        fake_expectations,
    )

    pending_updates: List[Callable[[], None]] = []

    def custom_sleep(_seconds: float) -> None:
        if pending_updates:
            action = pending_updates.pop(0)
            action()

    class _Fetcher:
        def __init__(self) -> None:
            self.calls = 0
            self.pending_snapshots: List[Set[str]] = []

        def __call__(self, *, since):
            self.calls += 1
            self.pending_snapshots.append(set(pending_unsent))
            return []

    fetcher = _Fetcher()

    watcher = EmailWatcherV2(
        supplier_agent=supplier_agent,
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=3,
        email_fetcher=fetcher,
        sleep=custom_sleep,
        now=lambda: now,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": unique_id,
                "supplier_id": "sup-unsent",
                "supplier_email": "pending@example.com",
                "message_id": "<msg-unsent>",
                "subject": "Pending dispatch",
                "dispatched_at": now,
            }
        ],
    )

    pending_updates.append(lambda: pending_unsent.clear())

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["dispatched_count"] == 1
    assert fetcher.calls >= 1
    assert all(snapshot == set() for snapshot in fetcher.pending_snapshots)
    assert pending_unsent == set()


def test_parse_email_recovers_tracking_headers():
    message = EmailMessage()
    message["Subject"] = "Re: RFQ details"
    message["From"] = "Acme Quotes <quotes@acme.example>"
    message["To"] = "buyer@procwise.test"
    message["X-Procwise-Unique-Id"] = "UID-HEADER12345"
    message["X-Procwise-Workflow-Id"] = "wf-123"
    message["X-Procwise-Supplier-Id"] = "sup-123"
    message["X-Procwise-RFQ-ID"] = "RFQ-12345"
    message.set_content("Thank you for the opportunity")

    parsed = _parse_email(message.as_bytes())

    assert parsed.unique_id == "UID-HEADER12345"
    assert parsed.workflow_id == "wf-123"
    assert parsed.supplier_id == "sup-123"
    assert parsed.rfq_id == "RFQ-12345"


def test_email_watcher_matches_using_supplier_id_when_threshold_not_met(tmp_path):
    workflow_id = "wf-header-fallback"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

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

        def __call__(self, *, since):
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
    assert supplier_agent.contexts, "Supplier agent should still be invoked via fallback"


def test_email_watcher_does_not_match_when_only_rfq_id_present(tmp_path):
    workflow_id = "wf-rfq-fallback"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

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

        def __call__(self, *, since):
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

    assert result["complete"] is False
    assert result["responded_count"] == 0
    assert not supplier_agent.contexts, "Supplier agent should not run without a unique_id match"


def test_email_watcher_refreshes_tracker_when_dispatch_logged_after_start(monkeypatch):
    workflow_id = "wf-refresh-tracker"
    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-late")
    original_subject = "Pricing request"
    original_message_id = "<msg-late>"

    response = EmailResponse(
        unique_id=unique_id,
        supplier_id="sup-late",
        supplier_email="supplier@example.com",
        from_address="supplier@example.com",
        message_id="<reply-late>",
        subject=f"Re: {original_subject}",
        body=embed_unique_id_in_email_body("Thank you", unique_id),
        received_at=now + timedelta(minutes=3),
        in_reply_to=(original_message_id.strip("<>"),),
    )

    monkeypatch.setattr("services.email_watcher_v2.tracking_repo.init_schema", lambda: None)
    monkeypatch.setattr("services.email_watcher_v2.supplier_response_repo.init_schema", lambda: None)
    monkeypatch.setattr("services.email_watcher_v2.tracking_repo.load_workflow_rows", lambda **_: [])
    monkeypatch.setattr("services.email_watcher_v2.tracking_repo.record_dispatches", lambda **_: None)
    monkeypatch.setattr("services.email_watcher_v2.tracking_repo.mark_response", lambda **_: None)
    monkeypatch.setattr("services.email_watcher_v2.supplier_response_repo.insert_response", lambda row: None)
    monkeypatch.setattr("services.email_watcher_v2.supplier_response_repo.fetch_pending", lambda **_: [])
    monkeypatch.setattr("services.email_watcher_v2.supplier_response_repo.delete_responses", lambda **_: None)

    call_log: List[str] = []
    pending_dispatches: List[EmailDispatchRecord] = [
        EmailDispatchRecord(
            unique_id=unique_id,
            supplier_id="sup-late",
            supplier_email="supplier@example.com",
            message_id=original_message_id,
            subject=original_subject,
            thread_headers={"in_reply_to": (original_message_id.strip("<>"),)},
            dispatched_at=now,
        )
    ]

    def custom_refresh(self, tracker):
        if pending_dispatches:
            tracker.register_dispatches(list(pending_dispatches))
            pending_dispatches.clear()
            call_log.append("refresh_with_dispatch")
        else:
            call_log.append("refresh")

    monkeypatch.setattr(EmailWatcherV2, "_refresh_tracker_from_repo", custom_refresh)

    def fetcher(*, since):
        call_log.append("fetch")
        return [response]

    watcher = EmailWatcherV2(
        supplier_agent=StubSupplierAgent(),
        negotiation_agent=StubNegotiationAgent(),
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    tracker = watcher._trackers[workflow_id]
    assert call_log.count("fetch") >= 1
    assert "refresh_with_dispatch" in call_log
    assert tracker.dispatched_count == 1
    assert unique_id in tracker.email_records
    assert result["dispatched_count"] == 1


def test_email_watcher_retains_responses_without_supplier_agent(tmp_path):
    workflow_id = "wf-no-agent"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-handoff")

    responses = [
        EmailResponse(
            unique_id=unique_id,
            supplier_id="sup-handoff",
            supplier_email="reply@example.com",
            from_address="reply@example.com",
            message_id="<reply-handoff>",
            subject="Our proposal",
            body=embed_unique_id_in_email_body("Here is the quote", unique_id),
            received_at=now + timedelta(minutes=1),
        )
    ]

    class _Fetcher:
        def __init__(self, payload: List[EmailResponse]):
            self.payload = payload
            self.calls = 0

        def __call__(self, *, since):
            self.calls += 1
            if self.calls == 1:
                return list(self.payload)
            return []

    fetcher = _Fetcher(responses)

    watcher = EmailWatcherV2(
        supplier_agent=None,
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
                "supplier_id": "sup-handoff",
                "supplier_email": "reply@example.com",
                "message_id": "<msg-handoff>",
                "subject": "Initial quote",
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == 1
    rows = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert rows and rows[0]["unique_id"] == unique_id


def test_email_watcher_supplier_agent_factory_per_workflow():
    base_now = datetime.now(timezone.utc)
    fetcher = QueueingFetcher()
    created_agents: Dict[str, List[StubSupplierAgent]] = {}

    class WorkflowAwareAgent(StubSupplierAgent):
        def __init__(self, workflow: str) -> None:
            super().__init__()
            self.workflow = workflow

        def execute(self, context):
            self.contexts.append(context)
            assert context.workflow_id == self.workflow
            unique_id = context.input_data.get("unique_id")
            data = {
                "workflow_id": self.workflow,
                "batch_ready": True,
                "expected_responses": 1 if unique_id else 0,
                "collected_responses": 1 if unique_id else 0,
                "supplier_responses": (
                    [{"unique_id": unique_id, "workflow_id": self.workflow}]
                    if unique_id
                    else []
                ),
                "unique_ids": [unique_id] if unique_id else [],
                "batch_metadata": {
                    "expected": 1 if unique_id else 0,
                    "collected": 1 if unique_id else 0,
                    "ready": True,
                },
            }
            return AgentOutput(status=AgentStatus.SUCCESS, data=data)

    def factory(workflow_id: str) -> StubSupplierAgent:
        agent = WorkflowAwareAgent(workflow_id)
        created_agents.setdefault(workflow_id, []).append(agent)
        return agent

    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()

    watcher = EmailWatcherV2(
        supplier_agent_factory=factory,
        negotiation_agent=StubNegotiationAgent(),
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=1,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: base_now,
    )

    workflows = ["wf-factory-alpha", "wf-factory-beta"]
    for index, workflow_id in enumerate(workflows):
        supplier_response_repo.reset_workflow(workflow_id=workflow_id)
        workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

        unique_id = generate_unique_email_id(workflow_id, f"sup-{index}")
        original_message_id = f"<msg-{index}>"
        watcher.register_workflow_dispatch(
            workflow_id,
            [
                {
                    "unique_id": unique_id,
                    "supplier_id": f"sup-{index}",
                    "supplier_email": f"supplier{index}@example.com",
                    "message_id": original_message_id,
                    "subject": "Quote request",
                    "dispatched_at": base_now,
                }
            ],
        )

        fetcher.push(
            [
                EmailResponse(
                    unique_id=unique_id,
                    supplier_id=f"sup-{index}",
                    supplier_email=f"supplier{index}@example.com",
                    from_address=f"supplier{index}@example.com",
                    message_id=f"<reply-{index}>",
                    subject="Re: Quote request",
                    body=embed_unique_id_in_email_body(
                        "Appreciate the opportunity", unique_id
                    ),
                    received_at=base_now + timedelta(minutes=index + 1),
                    in_reply_to=(original_message_id,),
                )
            ]
        )

        result = watcher.wait_and_collect_responses(workflow_id)
        assert result["responded_count"] == 1

    for workflow_id in workflows:
        agents = created_agents.get(workflow_id, [])
        assert len(agents) == 1
        agent = agents[0]
        assert agent.contexts, "Factory agent should receive processing context"
        assert all(context.workflow_id == workflow_id for context in agent.contexts)

