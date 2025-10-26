from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentOutput, AgentStatus
from decimal import Decimal

from repositories import supplier_response_repo, workflow_email_tracking_repo
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from services.email_watcher_v2 import EmailResponse, EmailWatcherV2, _parse_email
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

    all_rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
    assert len(all_rows) == 1
    assert all_rows[0].get("rfq_id") == "RFQ-2024-0001"


def test_email_watcher_v2_matches_legacy_bracketed_message_id(tmp_path):
    workflow_id = "wf-legacy-thread"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

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
        max_poll_attempts=1,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["complete"] is True
    assert result["responded_count"] == 1

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


def test_email_watcher_matches_using_rfq_id_when_unique_id_missing(tmp_path):
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

    assert result["complete"] is True
    assert result["responded_count"] == 1
    assert supplier_agent.contexts, "Supplier agent should run on RFQ fallback"
    context = supplier_agent.contexts[0]
    assert context.input_data.get("rfq_id") == rfq_identifier
    assert context.input_data.get("unique_id") == unique_id
