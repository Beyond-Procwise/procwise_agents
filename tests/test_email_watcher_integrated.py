from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from repositories import supplier_interaction_repo
from repositories.supplier_interaction_repo import SupplierInteractionRow
from services.email_watcher_integrated import (
    ContinuousEmailWatcher,
    embed_unique_id_in_email_body,
    generate_unique_email_id,
    get_supplier_responses,
    mark_interaction_processed,
    register_sent_email,
)
from services.email_watcher_v2 import EmailResponse


def setup_module(_module):
    supplier_interaction_repo.init_schema()
    supplier_interaction_repo.reset()


def teardown_module(_module):
    supplier_interaction_repo.reset()


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

    watcher._handle_response(response)

    responses = get_supplier_responses("negotiation-agent", workflow_id)
    assert len(responses) == 1
    assert responses[0]["subject"].startswith("Re: Round 2")


def test_embed_unique_id_round_trip():
    uid = generate_unique_email_id("wf-roundtrip", "sup-42", round_number=3)
    body = embed_unique_id_in_email_body("Thank you", uid)
    assert uid in body
