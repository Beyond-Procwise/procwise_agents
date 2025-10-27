from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from repositories import supplier_interaction_repo, supplier_response_repo
from repositories.supplier_interaction_repo import SupplierInteractionRow
from repositories.workflow_email_tracking_repo import (
    WorkflowDispatchRow,
    record_dispatches,
    reset_workflow as reset_tracking,
)
import services.db as db
from services.email_watcher_v2 import (
    ContinuousEmailWatcher,
    EmailResponse,
    embed_unique_id_in_email_body,
    generate_unique_email_id,
    get_supplier_responses,
    mark_interaction_processed,
    register_sent_email,
)


def _fail_connect(*_args, **_kwargs):
    raise psycopg2.OperationalError("forced test fallback")


psycopg2.connect = _fail_connect
db.psycopg2 = psycopg2
db._pg_dsn = lambda: None


@pytest.fixture(autouse=True)
def _force_fake_db(monkeypatch):
    monkeypatch.setattr("services.db.psycopg2.connect", _fail_connect)


def setup_module(_module):
    supplier_interaction_repo.init_schema()
    supplier_interaction_repo.reset()
    supplier_response_repo.init_schema()


def teardown_module(_module):
    supplier_interaction_repo.reset()
    supplier_response_repo.reset_workflow(workflow_id="wf-watch")
    supplier_response_repo.reset_workflow(workflow_id="wf-header")
    supplier_response_repo.reset_workflow(workflow_id="wf-continuous")
    supplier_response_repo.reset_workflow(workflow_id="wf-fallback")
    reset_tracking(workflow_id="wf-watch")
    reset_tracking(workflow_id="wf-header")
    reset_tracking(workflow_id="wf-continuous")
    reset_tracking(workflow_id="wf-fallback")


def test_register_and_process_supplier_response():
    supplier_interaction_repo.reset()
    workflow_id = "wf-integration"
    supplier_id = "sup-123"
    unique_id = generate_unique_email_id(workflow_id, supplier_id, round_number=1)

    register_sent_email(
        "drafting-agent",
        workflow_id,
        supplier_id,
        "quotes@example.com",
        unique_id,
        round_number=1,
        interaction_type="initial",
        subject="RFQ request",
        rfq_id="RFQ-2024-0001",
    )

    outbound_dict = supplier_interaction_repo.lookup_outbound(unique_id)
    assert outbound_dict is not None
    outbound = SupplierInteractionRow(
        workflow_id=outbound_dict["workflow_id"],
        unique_id=outbound_dict["unique_id"],
        supplier_id=outbound_dict.get("supplier_id"),
        supplier_email=outbound_dict.get("supplier_email"),
        round_number=outbound_dict.get("round_number", 1),
        direction="outbound",
        interaction_type=outbound_dict.get("interaction_type", "initial"),
        status=outbound_dict.get("status", "pending"),
        subject=outbound_dict.get("subject"),
        body=outbound_dict.get("body"),
        message_id=outbound_dict.get("message_id"),
        in_reply_to=outbound_dict.get("in_reply_to"),
        references=outbound_dict.get("references"),
        rfq_id="RFQ-2024-0001",
        metadata=outbound_dict.get("metadata"),
    )

    supplier_interaction_repo.record_inbound_response(
        outbound=outbound,
        message_id="<reply-001>",
        subject="Re: RFQ request",
        body="We can support this request",
        from_address="quotes@example.com",
        received_at=datetime.now(timezone.utc),
        in_reply_to=[],
        references=[],
        rfq_id="RFQ-2024-0001",
    )

    responses = get_supplier_responses("supplier-agent", workflow_id)
    assert len(responses) == 1
    assert responses[0]["status"] == "received"

    mark_interaction_processed("supplier-agent", workflow_id, unique_id)
    processed = get_supplier_responses("supplier-agent", workflow_id, status="processed")
    assert len(processed) == 1
    assert processed[0]["status"] == "processed"


def test_continuous_watcher_records_unique_id_response():
    supplier_interaction_repo.reset()
    supplier_response_repo.reset_workflow(workflow_id="wf-watch")
    reset_tracking(workflow_id="wf-watch")
    workflow_id = "wf-watch"
    unique_id = generate_unique_email_id(workflow_id, "sup-9", round_number=1)

    register_sent_email(
        "negotiation-agent",
        workflow_id,
        "sup-9",
        "contact@supplier.test",
        unique_id,
        round_number=2,
        interaction_type="negotiation",
        subject="Round 2 counter",
        rfq_id="RFQ-2024-0099",
    )

    watcher = ContinuousEmailWatcher(
        "watcher-agent",
        poll_interval_seconds=5,
        email_fetcher=lambda **_: [],
        sleep_fn=lambda *_: None,
        now_fn=lambda: datetime.now(timezone.utc),
    )

    response = EmailResponse(
        unique_id=unique_id,
        supplier_id="sup-9",
        supplier_email="contact@supplier.test",
        from_address="contact@supplier.test",
        message_id="<reply-123>",
        subject="Re: Round 2 counter",
        body="We agree to the price",
        received_at=datetime.now(timezone.utc),
        in_reply_to=("<msg-123>",),
        references=(),
        workflow_id=workflow_id,
        rfq_id="RFQ-2024-0099",
    )

    dispatch_index, _ = watcher._build_dispatch_index(workflow_id, None)
    watcher._handle_response(response, dispatch_index, {unique_id})

    responses = get_supplier_responses("negotiation-agent", workflow_id)
    assert len(responses) == 1
    assert responses[0]["subject"].startswith("Re: Round 2")

    pending = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert len(pending) == 1
    assert pending[0]["response_message_id"] == "<reply-123>"


def test_continuous_watcher_matches_via_headers_and_persists():
    supplier_interaction_repo.reset()
    workflow_id = "wf-header"
    reset_tracking(workflow_id=workflow_id)
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)

    unique_id = generate_unique_email_id(workflow_id, "sup-hdr", round_number=1)
    dispatched_at = datetime(2024, 3, 14, 12, 0, tzinfo=timezone.utc)

    register_sent_email(
        "negotiation-agent",
        workflow_id,
        "sup-hdr",
        "contact@supplier.test",
        unique_id,
        round_number=1,
        interaction_type="initial",
        subject="Round 1",
        message_id="<dispatch-001>",
    )

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            WorkflowDispatchRow(
                workflow_id=workflow_id,
                unique_id=unique_id,
                dispatch_key="dispatch-001",
                supplier_id="sup-hdr",
                supplier_email="contact@supplier.test",
                message_id="<dispatch-001>",
                subject="Round 1",
                dispatched_at=dispatched_at,
                responded_at=None,
                response_message_id=None,
                matched=False,
                thread_headers={
                    "in_reply_to": ["<dispatch-001>"],
                    "references": [],
                },
            )
        ],
    )

    response = EmailResponse(
        unique_id=None,
        supplier_id="sup-hdr",
        supplier_email="contact@supplier.test",
        from_address="Quotes <contact@supplier.test>",
        message_id="<reply-001>",
        subject="Re: Round 1",
        body="We received your request.",
        received_at=dispatched_at + timedelta(minutes=5),
        in_reply_to=("<dispatch-001>",),
        references=(),
        workflow_id=workflow_id,
        rfq_id=None,
        headers={"In-Reply-To": ("<dispatch-001>",)},
    )

    watcher = ContinuousEmailWatcher(
        "watcher-agent",
        poll_interval_seconds=1,
        poll_jitter_seconds=0,
        email_fetcher=lambda **_: [response],
        sleep_fn=lambda *_: None,
        now_fn=lambda: dispatched_at + timedelta(minutes=5, seconds=1),
    )

    dispatch_index, tracking_rows = watcher._build_dispatch_index(workflow_id, None)
    stored = watcher.run_once(workflow_id, dispatch_index, tracking_rows)

    assert stored
    entry = stored[0]
    assert entry["unique_id"] == unique_id
    assert entry["match_score"] >= watcher.match_threshold
    assert "in-reply-to" in entry["matched_on"]

    pending = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert len(pending) == 1
    pending_row = pending[0]
    assert pending_row["response_message_id"] == "<reply-001>"
    evidence = pending_row.get("match_evidence") or []
    if isinstance(evidence, str):
        evidence = json.loads(evidence)
    assert "in-reply-to" in evidence


def test_run_once_defaults_to_remaining_dispatch_when_low_confidence():
    supplier_interaction_repo.reset()
    workflow_id = "wf-fallback"
    reset_tracking(workflow_id=workflow_id)
    supplier_response_repo.reset_workflow(workflow_id=workflow_id)

    unique_id = generate_unique_email_id(workflow_id, "sup-fallback", round_number=1)
    dispatched_at = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)

    register_sent_email(
        "negotiation-agent",
        workflow_id,
        "sup-fallback",
        "contact@supplier.test",
        unique_id,
        round_number=1,
        interaction_type="initial",
        subject="RFQ Request",
    )

    record_dispatches(
        workflow_id=workflow_id,
        dispatches=[
            WorkflowDispatchRow(
                workflow_id=workflow_id,
                unique_id=unique_id,
                dispatch_key="dispatch-fallback",
                supplier_id="sup-fallback",
                supplier_email="contact@supplier.test",
                message_id="<dispatch-fallback>",
                subject="RFQ Request",
                dispatched_at=dispatched_at,
                responded_at=None,
                response_message_id=None,
                matched=False,
                thread_headers={},
            )
        ],
    )

    response = EmailResponse(
        unique_id=None,
        supplier_id=None,
        supplier_email=None,
        from_address="updates@other.test",
        message_id="<fallback-reply>",
        subject="Availability Update",
        body="Hello team, attaching our availability soon.",
        received_at=dispatched_at + timedelta(minutes=15),
        in_reply_to=(),
        references=(),
        workflow_id=workflow_id,
        rfq_id=None,
        headers={},
    )

    watcher = ContinuousEmailWatcher(
        "watcher-agent",
        poll_interval_seconds=1,
        poll_jitter_seconds=0,
        email_fetcher=lambda **_: [response],
        sleep_fn=lambda *_: None,
        now_fn=lambda: dispatched_at + timedelta(minutes=15, seconds=1),
    )

    dispatch_index, tracking_rows = watcher._build_dispatch_index(workflow_id, None)
    stored = watcher.run_once(workflow_id, dispatch_index, tracking_rows)

    assert stored, "fallback response should be stored"
    entry = stored[0]
    assert entry["unique_id"] == unique_id
    assert entry["match_score"] >= watcher.match_threshold
    assert "sole_pending_dispatch" in entry["matched_on"]

    pending = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
    assert len(pending) == 1
    assert pending[0]["response_message_id"] == "<fallback-reply>"
    evidence = pending[0].get("match_evidence") or []
    if isinstance(evidence, str):
        evidence = json.loads(evidence)
    assert "sole_pending_dispatch" in evidence


def test_run_continuously_waits_for_all_responses(monkeypatch):
    def _fail_connect(_dsn):
        raise psycopg2.OperationalError("forced test fallback")

    monkeypatch.setattr("services.db.psycopg2.connect", _fail_connect)

    supplier_interaction_repo.reset()
    supplier_response_repo.reset_workflow(workflow_id="wf-continuous")
    workflow_id = "wf-continuous"
    unique_a = generate_unique_email_id(workflow_id, "sup-a", round_number=1)
    unique_b = generate_unique_email_id(workflow_id, "sup-b", round_number=1)

    register_sent_email(
        "negotiation-agent",
        workflow_id,
        "sup-a",
        "a@supplier.test",
        unique_a,
        round_number=1,
        interaction_type="initial",
        subject="Round 1",
    )

    register_sent_email(
        "negotiation-agent",
        workflow_id,
        "sup-b",
        "b@supplier.test",
        unique_b,
        round_number=1,
        interaction_type="initial",
        subject="Round 1",
    )

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    now_calls = {"count": -1}
    moments = [base_time + timedelta(seconds=idx) for idx in range(20)]

    def now_fn() -> datetime:
        now_calls["count"] = min(now_calls["count"] + 1, len(moments) - 1)
        return moments[now_calls["count"]]

    fetch_calls = {"count": -1}

    response_one = EmailResponse(
        unique_id=unique_a,
        supplier_id="sup-a",
        supplier_email="a@supplier.test",
        from_address="a@supplier.test",
        message_id="<msg-a>",
        subject="Re: Round 1",
        body="First reply",
        received_at=base_time + timedelta(seconds=5),
        in_reply_to=("<dispatch-a>",),
        references=(),
        workflow_id=workflow_id,
    )

    response_two = EmailResponse(
        unique_id=unique_b,
        supplier_id="sup-b",
        supplier_email="b@supplier.test",
        from_address="b@supplier.test",
        message_id="<msg-b>",
        subject="Re: Round 1",
        body="Second reply",
        received_at=base_time + timedelta(seconds=15),
        in_reply_to=("<dispatch-b>",),
        references=(),
        workflow_id=workflow_id,
    )

    def fetcher(**_kwargs):
        fetch_calls["count"] += 1
        if fetch_calls["count"] == 0:
            return [response_one]
        if fetch_calls["count"] == 2:
            return [response_two]
        return []

    watcher = ContinuousEmailWatcher(
        "watcher-agent",
        poll_interval_seconds=1,
        poll_jitter_seconds=0,
        email_fetcher=fetcher,
        sleep_fn=lambda *_args: None,
        now_fn=now_fn,
    )

    result = watcher.run_continuously(workflow_id)

    assert result["status"] == "complete"
    assert result["expected_responses"] == 2
    assert result["responses_received"] == 2
    assert set(result["matched_dispatch_ids"]) == {unique_a, unique_b}
    assert not result["unmatched_inbound"]

def test_embed_unique_id_round_trip():
    uid = generate_unique_email_id("wf-roundtrip", "sup-42", round_number=3)
    body = embed_unique_id_in_email_body("Thank you", uid)
    assert uid in body
