from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


def _ensure_timezone(value: datetime) -> datetime:
    """Normalise datetimes to timezone aware UTC values."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SupplierRiskSignal:
    """Structured representation of an external supplier risk datapoint."""

    supplier_id: str
    signal_type: str
    severity: float
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self) -> Dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "signal_type": self.signal_type,
            "severity": float(self.severity),
            "source": self.source,
            "payload": dict(self.payload),
            "occurred_at": _ensure_timezone(self.occurred_at),
        }


@dataclass(frozen=True)
class PredictiveRiskScore:
    """Result of a predictive risk evaluation for a supplier."""

    supplier_id: str
    score: float
    model_version: str
    computed_at: datetime
    feature_summary: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "score": float(self.score),
            "model_version": self.model_version,
            "computed_at": _ensure_timezone(self.computed_at),
            "feature_summary": dict(self.feature_summary),
        }
