from datetime import datetime, timezone
from decimal import Decimal

from repositories import supplier_response_repo
from utils.supplier_response_helper import store_supplier_response


def test_store_supplier_response_persists_row(monkeypatch):
    captured = {}

    def fake_insert(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    now = datetime.now(timezone.utc)
    row = store_supplier_response(
        workflow_id="wf-1",
        unique_id="uid-1",
        supplier_id="sup-1",
        supplier_email="supplier@example.com",
        body_text="Thanks for the opportunity",
        body_html="<p>Thanks for the opportunity</p>",
        received_at=now,
        response_time=Decimal("12.5"),
        response_message_id="mid-1",
        response_subject="Re: RFQ",
        response_from="contact@example.com",
        original_message_id="orig-1",
        original_subject="RFQ",
        match_confidence=Decimal("0.95"),
        processed=True,
    )

    assert captured["row"] is row
    assert row.response_text == "Thanks for the opportunity"
    assert row.response_body == "<p>Thanks for the opportunity</p>"
    assert row.processed is True
    assert row.response_time == Decimal("12.5")
    assert row.supplier_email == "supplier@example.com"


def test_store_supplier_response_falls_back_to_available_body(monkeypatch):
    captured = {}

    def fake_insert(row):
        captured["row"] = row

    monkeypatch.setattr(supplier_response_repo, "insert_response", fake_insert)

    now = datetime.now(timezone.utc)
    row = store_supplier_response(
        workflow_id="wf-2",
        unique_id="uid-2",
        supplier_id=None,
        supplier_email=None,
        body_text=None,
        body_html="<p>Hello world</p>",
        received_at=now,
        response_time=None,
        response_message_id=None,
        response_subject=None,
        response_from=None,
        original_message_id=None,
        original_subject=None,
        match_confidence=None,
        processed=False,
    )

    assert captured["row"] is row
    assert row.response_text == "<p>Hello world</p>"
    assert row.response_body == "<p>Hello world</p>"
    assert row.processed is False
