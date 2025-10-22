from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from enum import Enum
from pathlib import Path
from typing import List, Optional
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


_orchestration_module = sys.modules.setdefault("orchestration", types.ModuleType("orchestration"))
prompt_engine_stub = types.ModuleType("orchestration.prompt_engine")


class PromptEngine:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def render(self, *_, **__):
        return {}


prompt_engine_stub.PromptEngine = PromptEngine
sys.modules["orchestration.prompt_engine"] = prompt_engine_stub
setattr(_orchestration_module, "prompt_engine", prompt_engine_stub)


class AgentStatus(str, Enum):
    SUCCESS = "success"


@dataclass
class AgentOutput:
    status: AgentStatus
    data: dict
    next_agents: Optional[List[str]] = None

from repositories import supplier_response_repo, workflow_email_tracking_repo
from repositories.supplier_response_repo import SupplierResponseRow
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
            if not pending:
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={
                        "workflow_id": workflow_id,
                        "batch_ready": False,
                        "expected_responses": 0,
                        "collected_responses": 0,
                        "supplier_responses": [],
                        "supplier_responses_batch": [],
                        "supplier_responses_count": 0,
                        "unique_ids": [],
                        "batch_metadata": {"expected": 0, "collected": 0, "ready": False},
                        "negotiation_batch": False,
                    },
                    next_agents=[],
                )

            latest_by_uid = {}
            for row in pending:
                uid = row.get("unique_id") or workflow_id
                latest_by_uid[uid] = row

            unique_ids = list(latest_by_uid.keys())
            responses = []
            for uid in unique_ids:
                row = latest_by_uid[uid]
                responses.append(
                    {
                        "unique_id": uid,
                        "workflow_id": workflow_id,
                        "supplier_id": row.get("supplier_id"),
                    }
                )
            expected = len(unique_ids)
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "workflow_id": workflow_id,
                    "batch_ready": True,
                    "expected_responses": expected,
                    "collected_responses": expected,
                    "supplier_responses": responses,
                    "supplier_responses_batch": responses,
                    "supplier_responses_count": expected,
                    "unique_ids": unique_ids,
                    "batch_metadata": {
                        "expected": expected,
                        "collected": expected,
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


def test_email_watcher_tracks_multiple_responses_for_same_unique_id(monkeypatch):
    workflow_id = "wf-double-response"
    supplier_response_repo.init_schema()
    workflow_email_tracking_repo.init_schema()
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)
    workflow_email_tracking_repo.reset_workflow(workflow_id=workflow_id)

    supplier_agent = StubSupplierAgent()
    negotiation_agent = StubNegotiationAgent()

    now = datetime.now(timezone.utc)
    unique_id = generate_unique_email_id(workflow_id, "sup-2")
    original_subject = "Request details"
    original_message_id = "<orig-message>"

    first_body = embed_unique_id_in_email_body("Thanks for reaching out", unique_id)
    second_body = embed_unique_id_in_email_body("Updated pricing attached", unique_id)

    responses = [
        EmailResponse(
            unique_id=unique_id,
            supplier_id="sup-2",
            supplier_email="supplier@example.com",
            from_address="supplier@example.com",
            message_id="<reply-first>",
            subject=f"Re: {original_subject}",
            body=first_body,
            received_at=now + timedelta(minutes=3),
            in_reply_to=(original_message_id,),
        ),
        EmailResponse(
            unique_id=unique_id,
            supplier_id="sup-2",
            supplier_email="supplier@example.com",
            from_address="supplier@example.com",
            message_id="<reply-second>",
            subject=f"Re: {original_subject}",
            body=second_body,
            received_at=now + timedelta(minutes=9),
            in_reply_to=(original_message_id,),
        ),
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

    insert_calls: List[SupplierResponseRow] = []
    original_insert = supplier_response_repo.insert_response

    def _tracking_insert(row: SupplierResponseRow) -> None:
        insert_calls.append(row)
        return original_insert(row)

    monkeypatch.setattr(supplier_response_repo, "insert_response", _tracking_insert)

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
                "supplier_id": "sup-2",
                "supplier_email": "supplier@example.com",
                "message_id": original_message_id,
                "subject": original_subject,
                "dispatched_at": now,
            }
        ],
    )

    result = watcher.wait_and_collect_responses(workflow_id)

    assert result["responded_count"] == 1
    assert unique_id in result["response_history"]
    assert len(result["response_history"][unique_id]) == 2

    assert len(insert_calls) == 2
    assert insert_calls[-1].response_text == second_body

    assert supplier_agent.contexts, "Supplier agent should receive the batched response"
    context = supplier_agent.contexts[-1]
    assert context.input_data.get("body") == second_body
    history = context.input_data.get("response_history")
    assert isinstance(history, list) and len(history) == 2
    assert history[-1]["body"] == second_body

    assert fetcher.calls >= 1


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
