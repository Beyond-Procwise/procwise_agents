from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.risk_intelligence import SupplierRiskSignal
from repositories.supplier_risk_repo import SupplierRiskSignalRepository, init_schema
from services import db
from services.event_bus import EventBus
from services.risk_intelligence_service import (
    PredictiveRiskModel,
    RiskIntelligenceService,
)


@pytest.fixture(autouse=True)
def _reset_fake_store():
    os.environ.setdefault("PYTEST_CURRENT_TEST", "risk-intelligence-tests")
    store = db._fake_store()
    store.ensure_tables()
    store.supplier_risk_signals = []  # type: ignore[attr-defined]
    store.supplier_risk_scores = {}  # type: ignore[attr-defined]
    return


def test_repository_round_trip():
    init_schema()
    repo = SupplierRiskSignalRepository()
    signal = SupplierRiskSignal(
        supplier_id="supplier-1",
        signal_type="financial",
        severity=0.7,
        source="dnb",
        payload={"rating": "BBB"},
        occurred_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    repo.record_signal(signal)

    results = repo.fetch_signals("supplier-1")
    assert len(results) == 1
    fetched = results[0]
    assert fetched.signal_type == "financial"
    assert fetched.payload["rating"] == "BBB"


def test_evaluate_supplier_triggers_event_when_threshold_breached():
    init_schema()
    repo = SupplierRiskSignalRepository()
    bus = EventBus()
    service = RiskIntelligenceService(
        repository=repo,
        model=PredictiveRiskModel(),
        event_bus=bus,
        risk_threshold=0.45,
        history_limit=20,
    )

    recent_signal = SupplierRiskSignal(
        supplier_id="supplier-42",
        signal_type="geopolitical",
        severity=0.9,
        source="achilles",
        payload={"region": "EMEA"},
        occurred_at=datetime.now(timezone.utc) - timedelta(hours=12),
    )
    trailing_signal = SupplierRiskSignal(
        supplier_id="supplier-42",
        signal_type="financial",
        severity=0.6,
        source="moodys",
        payload={"trend": "negative"},
        occurred_at=datetime.now(timezone.utc) - timedelta(days=5),
    )
    repo.bulk_record([recent_signal, trailing_signal])

    captured = []
    bus.subscribe(
        "supplier.risk.threshold.exceeded", lambda payload: captured.append(payload)
    )

    metrics = {
        "on_time_delivery_rate": 0.65,
        "quality_score": 0.7,
        "anomaly_index": 0.55,
        "resilience_index": 0.4,
    }

    assessment = service.evaluate_supplier("supplier-42", metrics)

    assert assessment.score >= 0.45
    assert captured, "Expected threshold event to be published"
    payload = captured[0]
    assert payload["supplier_id"] == "supplier-42"
    assert pytest.approx(payload["score"], rel=1e-6) == assessment.score

    latest = service.latest_score("supplier-42")
    assert latest is not None
    assert latest.model_version == assessment.model_version
    assert pytest.approx(latest.score, rel=1e-6) == assessment.score
