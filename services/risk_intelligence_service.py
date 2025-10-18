from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from models.risk_intelligence import PredictiveRiskScore, SupplierRiskSignal
from repositories.supplier_risk_repo import SupplierRiskSignalRepository
from services.event_bus import EventBus, get_event_bus

logger = logging.getLogger(__name__)


class RiskSignalAdapter:
    """Base class for external risk signal adapters."""

    name = "adapter"

    def fetch(self, supplier_id: str) -> Iterable[SupplierRiskSignal]:  # pragma: no cover - interface
        raise NotImplementedError


class PredictiveRiskModel:
    """Lightweight forward-looking risk model combining signals and performance."""

    def __init__(self, *, model_version: str = "risk-intel-v0.1", half_life_hours: float = 168.0) -> None:
        self.model_version = model_version
        self.half_life_hours = max(1.0, half_life_hours)

    def evaluate(
        self,
        supplier_metrics: Dict[str, float],
        signals: Sequence[SupplierRiskSignal],
    ) -> Dict[str, float]:
        """Return score components for downstream aggregation."""

        now = datetime.now(timezone.utc)
        total_decay = 0.0
        weighted_severity = 0.0
        for signal in signals:
            age_hours = max(
                0.0,
                (now - signal.occurred_at).total_seconds() / 3600.0,
            )
            decay = 0.5 ** (age_hours / self.half_life_hours)
            severity = max(0.0, min(1.0, signal.severity))
            weighted_severity += severity * decay
            total_decay += decay
        signal_component = weighted_severity / total_decay if total_decay else 0.0

        on_time = _clamp_metric(supplier_metrics.get("on_time_delivery_rate", 1.0))
        quality = _clamp_metric(supplier_metrics.get("quality_score", 1.0))
        anomaly = _clamp_metric(supplier_metrics.get("anomaly_index", 0.0))
        resilience = _clamp_metric(supplier_metrics.get("resilience_index", 0.5))

        performance_component = (
            (1.0 - on_time) * 0.4
            + (1.0 - quality) * 0.25
            + anomaly * 0.25
            + (1.0 - resilience) * 0.1
        )

        combined = (signal_component * 0.55) + (performance_component * 0.45)
        logistic = 1.0 / (1.0 + math.exp(-8.0 * (combined - 0.5)))
        score = _clamp_metric(logistic)

        return {
            "score": score,
            "signal_component": signal_component,
            "performance_component": performance_component,
            "signals_considered": float(len(signals)),
        }


def _clamp_metric(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if value != value:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, float(value)))


class RiskIntelligenceService:
    """Coordinates signal ingestion, predictive scoring, and event publishing."""

    def __init__(
        self,
        repository: Optional[SupplierRiskSignalRepository] = None,
        *,
        model: Optional[PredictiveRiskModel] = None,
        event_bus: Optional[EventBus] = None,
        risk_threshold: float = 0.7,
        history_limit: int = 100,
    ) -> None:
        self.repository = repository or SupplierRiskSignalRepository()
        self.model = model or PredictiveRiskModel()
        self.event_bus = event_bus or get_event_bus()
        self.risk_threshold = _clamp_metric(risk_threshold)
        self.history_limit = max(10, history_limit)

    def ingest_from_adapters(
        self,
        supplier_id: str,
        adapters: Iterable[RiskSignalAdapter],
    ) -> List[SupplierRiskSignal]:
        recorded: List[SupplierRiskSignal] = []
        for adapter in adapters:
            try:
                fetched = list(adapter.fetch(supplier_id))
            except Exception:
                logger.exception("Risk adapter %s failed for supplier=%s", adapter.name, supplier_id)
                continue
            if not fetched:
                continue
            self.repository.bulk_record(fetched)
            recorded.extend(fetched)
        return recorded

    def evaluate_supplier(
        self,
        supplier_id: str,
        supplier_metrics: Dict[str, float],
        *,
        signals: Optional[Sequence[SupplierRiskSignal]] = None,
    ) -> PredictiveRiskScore:
        if signals is None:
            signals = self.repository.fetch_signals(supplier_id, limit=self.history_limit)
        else:
            signals = list(signals)

        components = self.model.evaluate(supplier_metrics, signals)
        score_value = components["score"]
        computed_at = datetime.now(timezone.utc)
        feature_summary = {
            "signal_component": components["signal_component"],
            "performance_component": components["performance_component"],
            "signals_considered": components["signals_considered"],
            "supplier_metrics": {k: _clamp_metric(v) for k, v in supplier_metrics.items()},
        }
        assessment = PredictiveRiskScore(
            supplier_id=supplier_id,
            score=score_value,
            model_version=self.model.model_version,
            computed_at=computed_at,
            feature_summary=feature_summary,
        )
        self.repository.save_predictive_score(assessment)

        if score_value >= self.risk_threshold:
            payload = {
                "supplier_id": supplier_id,
                "score": score_value,
                "threshold": self.risk_threshold,
                "model_version": self.model.model_version,
                "signals_considered": components["signals_considered"],
                "signal_component": components["signal_component"],
                "performance_component": components["performance_component"],
            }
            logger.info(
                "Risk threshold exceeded for supplier=%s score=%.3f threshold=%.3f",
                supplier_id,
                score_value,
                self.risk_threshold,
            )
            self.event_bus.publish("supplier.risk.threshold.exceeded", payload)

        return assessment

    def latest_score(self, supplier_id: str) -> Optional[PredictiveRiskScore]:
        return self.repository.get_latest_score(supplier_id)
