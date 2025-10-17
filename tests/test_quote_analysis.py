import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..")))

from email_thread import EmailThread
from models.quote_analysis import (
    SupplierScorecard,
    build_analytics_context,
    build_quote_snapshot,
    build_scorecard,
)


def _build_thread_with_offer(content: str) -> EmailThread:
    thread = EmailThread(supplier_id="SUP-001")
    thread.current_round = 3
    thread.add_message("supplier_response_round_3", content, "ACT-123")
    return thread


def test_quote_snapshot_extracts_price_terms_and_lead_time():
    content = (
        "We can extend a 5% discount and supply for USD 1250 per lot. "
        "Delivery in 10 days, payment terms Net 45, and recycled materials used."
    )
    thread = _build_thread_with_offer(content)
    snapshot = build_quote_snapshot(thread)

    assert snapshot.price == pytest.approx(1250.0)
    assert snapshot.currency == "USD"
    assert snapshot.payment_terms_days == 45
    assert snapshot.lead_time_days == 10
    assert any("discount" in concession.lower() for concession in snapshot.concessions)
    assert any("recycled" in flag.lower() for flag in snapshot.sustainability_flags)


def test_scorecard_generates_reasonable_weighted_total():
    content = "Offering USD 980 with delivery in 7 days and Net 60 payment terms."
    thread = _build_thread_with_offer(content)
    snapshot = build_quote_snapshot(thread)
    context = build_analytics_context([snapshot])
    scorecard = build_scorecard(snapshot, thread, context)

    assert isinstance(scorecard, SupplierScorecard)
    assert 0.0 <= scorecard.weighted_total() <= 100.0
    assert 0.0 <= scorecard.price_competitiveness <= 100.0
