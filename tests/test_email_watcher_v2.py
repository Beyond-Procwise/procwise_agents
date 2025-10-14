from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.base_agent import AgentOutput, AgentStatus
from decimal import Decimal

from repositories import supplier_response_repo, workflow_email_tracking_repo
from services.email_watcher_v2 import EmailResponse, EmailWatcherV2
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
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "batch_ready": True,
                    "expected_responses": 1,
                    "collected_responses": 1,
                    "supplier_responses": [
                        {
                            "unique_id": context.input_data.get("workflow_id"),
                        }
                    ],
                },
                next_agents=[],
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
    assert all(uid.startswith("UID-") for uid in ids)
    assert len(ids) == 20


def test_embed_unique_id_is_recoverable():
    uid = "UID-ABC1234567890"
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

    message_contexts = [
        ctx
        for ctx in supplier_agent.contexts
        if isinstance(getattr(ctx, "input_data", None), dict)
        and "email_headers" in ctx.input_data
    ]
    assert message_contexts, "Supplier agent should process email context"
    context = message_contexts[0]
    headers = context.input_data["email_headers"]
    assert headers["unique_id"] == unique_id
    assert headers["received_time"] is not None
    assert "Thanks for the opportunity" in context.input_data["message"]

    rows = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert rows == []

    stored = supplier_response_repo.fetch_all(workflow_id=workflow_id)
    assert len(stored) == 1
    record = stored[0]
    assert record["processed"] is True
    assert "Thanks for the opportunity" in record["response_text"]
    assert record["response_message_id"] == "<reply-001>"
    assert (
        str(record.get("original_message_id", ""))
        .replace("<", "")
        .replace(">", "")
        == str(original_message_id).replace("<", "").replace(">", "")
    )
    confidence = record.get("match_confidence")
    if confidence is not None:
        assert Decimal(str(confidence)) >= Decimal("0.50")


def test_email_watcher_v2_persists_thread_headers_between_instances():
    workflow_id = "wf-restart"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

    now = datetime.now(timezone.utc)
    original_subject = "RFQ 1234"
    original_message_id = "msg-root"

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=lambda since: [],
        sleep=lambda _: None,
        now=lambda: now,
    )

    watcher.register_workflow_dispatch(
        workflow_id,
        [
            {
                "unique_id": "uid-123",
                "supplier_id": "sup-1",
                "supplier_email": "supplier@example.com",
                "message_id": original_message_id,
                "subject": original_subject,
                "dispatched_at": now,
                "thread_headers": {
                    "references": ["thread-1", original_message_id],
                    "in_reply_to": [original_message_id],
                },
            }
        ],
    )

    stored_rows = workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id)
    assert stored_rows[0].thread_headers
    assert original_message_id in stored_rows[0].thread_headers.get("in_reply_to", ())

    responses = [
        EmailResponse(
            unique_id=None,
            supplier_id="sup-1",
            supplier_email="supplier@example.com",
            from_address="supplier@example.com",
            message_id="reply-1",
            subject=f"Re: {original_subject}",
            body="Hello",
            received_at=now + timedelta(minutes=10),
            in_reply_to=(original_message_id,),
            references=("thread-1",),
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

    restart_watcher = EmailWatcherV2(
        dispatch_wait_seconds=0,
        poll_interval_seconds=1,
        max_poll_attempts=2,
        email_fetcher=fetcher,
        sleep=lambda _: None,
        now=lambda: now,
    )

    result = restart_watcher.wait_and_collect_responses(workflow_id)

    assert result["responded_count"] == 1
    assert result["matched_responses"]
    matched = next(iter(result["matched_responses"].values()))
    assert matched.message_id == "reply-1"
