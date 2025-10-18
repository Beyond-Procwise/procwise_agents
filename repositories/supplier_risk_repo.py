from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from models.risk_intelligence import PredictiveRiskScore, SupplierRiskSignal
from services.db import get_conn

DDL_SQL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.supplier_risk_signals (
    id SERIAL PRIMARY KEY,
    supplier_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    severity NUMERIC(5, 2) NOT NULL,
    source TEXT NOT NULL,
    payload JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_supplier_risk_signals_supplier
ON proc.supplier_risk_signals (supplier_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS proc.supplier_risk_scores (
    supplier_id TEXT PRIMARY KEY,
    score NUMERIC(6, 3) NOT NULL,
    model_version TEXT NOT NULL,
    feature_summary JSONB NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def init_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        for statement in filter(None, (stmt.strip() for stmt in DDL_SQL.split(";"))):
            cur.execute(statement)
        cur.close()


def _normalise_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SupplierRiskSignalRepository:
    """Data access helpers for supplier risk telemetry."""

    def record_signal(self, signal: SupplierRiskSignal) -> None:
        record = signal.to_record()
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO proc.supplier_risk_signals
                (supplier_id, signal_type, severity, source, payload, occurred_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    record["supplier_id"],
                    record["signal_type"],
                    record["severity"],
                    record["source"],
                    json.dumps(record["payload"]) if record["payload"] else None,
                    _normalise_dt(record["occurred_at"]),
                ),
            )
            cur.close()

    def bulk_record(self, signals: Iterable[SupplierRiskSignal]) -> None:
        batch = list(signals)
        if not batch:
            return
        with get_conn() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO proc.supplier_risk_signals
                (supplier_id, signal_type, severity, source, payload, occurred_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        signal.supplier_id,
                        signal.signal_type,
                        signal.severity,
                        signal.source,
                        json.dumps(signal.payload) if signal.payload else None,
                        _normalise_dt(signal.occurred_at),
                    )
                    for signal in batch
                ],
            )
            cur.close()

    def fetch_signals(
        self,
        supplier_id: str,
        *,
        limit: int = 50,
    ) -> List[SupplierRiskSignal]:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT supplier_id, signal_type, severity, source, payload, occurred_at
                FROM proc.supplier_risk_signals
                WHERE supplier_id=%s
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (supplier_id, limit),
            )
            rows = cur.fetchall()
            cur.close()

        signals: List[SupplierRiskSignal] = []
        for supplier_id, signal_type, severity, source, payload, occurred_at in rows:
            payload_dict = json.loads(payload) if payload else {}
            occurred = _normalise_dt(occurred_at)
            signals.append(
                SupplierRiskSignal(
                    supplier_id=supplier_id,
                    signal_type=signal_type,
                    severity=float(severity),
                    source=source,
                    payload=payload_dict,
                    occurred_at=occurred,
                )
            )
        return signals

    def save_predictive_score(self, score: PredictiveRiskScore) -> None:
        record = score.to_record()
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO proc.supplier_risk_scores
                (supplier_id, score, model_version, feature_summary, computed_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (supplier_id) DO UPDATE SET
                    score = EXCLUDED.score,
                    model_version = EXCLUDED.model_version,
                    feature_summary = EXCLUDED.feature_summary,
                    computed_at = EXCLUDED.computed_at
                """,
                (
                    record["supplier_id"],
                    record["score"],
                    record["model_version"],
                    json.dumps(record["feature_summary"]),
                    _normalise_dt(record["computed_at"]),
                ),
            )
            cur.close()

    def get_latest_score(self, supplier_id: str) -> Optional[PredictiveRiskScore]:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT supplier_id, score, model_version, feature_summary, computed_at
                FROM proc.supplier_risk_scores
                WHERE supplier_id=%s
                LIMIT 1
                """,
                (supplier_id,),
            )
            row = cur.fetchone()
            cur.close()

        if not row:
            return None

        supplier_id, score, model_version, feature_summary, computed_at = row
        summary_dict: Dict[str, float] = json.loads(feature_summary) if feature_summary else {}
        return PredictiveRiskScore(
            supplier_id=supplier_id,
            score=float(score),
            model_version=model_version,
            computed_at=_normalise_dt(computed_at),
            feature_summary=summary_dict,
        )
