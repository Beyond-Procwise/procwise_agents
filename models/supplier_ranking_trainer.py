"""Lightweight ML training utilities for the supplier ranking workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.reference_loader import load_reference_dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    label_column: str
    learning_rate: float
    epochs: int
    tolerance: float
    min_samples: int
    blend_ratio: float


def _load_config() -> TrainingConfig:
    reference = load_reference_dataset("supplier_scoring_reference")
    training = reference.get("training", {}) if isinstance(reference, dict) else {}
    return TrainingConfig(
        label_column=str(training.get("label_column", "is_preferred_supplier")),
        learning_rate=float(training.get("learning_rate", 0.05)),
        epochs=int(training.get("epochs", 400)),
        tolerance=float(training.get("tolerance", 1e-5)),
        min_samples=int(training.get("min_samples", 6)),
        blend_ratio=float(training.get("blend_ratio", 0.5)),
    )


class SupplierRankingTrainer:
    """Train a simple logistic model to refine supplier ranking weights."""

    def __init__(self) -> None:
        self.config = _load_config()
        self._last_weights: Dict[str, float] = {}

    @property
    def last_weights(self) -> Dict[str, float]:
        return dict(self._last_weights)

    def _prepare_features(
        self, df: pd.DataFrame, metrics: Sequence[str]
    ) -> Optional[tuple[np.ndarray, np.ndarray, List[str]]]:
        if df.empty:
            return None

        label_col = self.config.label_column
        if label_col not in df.columns:
            return None

        label_series = df[label_col]
        if label_series.isna().all():
            return None

        label_numeric = pd.to_numeric(
            label_series.replace(
                {
                    True: 1,
                    False: 0,
                    "true": 1,
                    "yes": 1,
                    "preferred": 1,
                    "y": 1,
                    "1": 1,
                    "0": 0,
                    "no": 0,
                    "false": 0,
                    "n": 0,
                    None: np.nan,
                }
            ),
            errors="coerce",
        )

        valid_mask = (~label_numeric.isna()).copy()
        feature_columns: List[np.ndarray] = []
        feature_names: List[str] = []

        for metric in metrics:
            score_col = f"{metric}_score"
            raw_col = metric
            if score_col in df.columns:
                column = pd.to_numeric(df[score_col], errors="coerce")
            elif raw_col in df.columns:
                column = pd.to_numeric(df[raw_col], errors="coerce")
            else:
                continue

            valid_mask &= ~column.isna()
            feature_columns.append(column)
            feature_names.append(metric)

        if not feature_columns:
            return None

        if isinstance(valid_mask, pd.Series):
            mask = valid_mask.to_numpy(dtype=bool)
        else:
            mask = np.asarray(valid_mask, dtype=bool)

        if mask.sum() < self.config.min_samples:
            return None

        X = np.column_stack([col.to_numpy()[mask] for col in feature_columns])
        y = label_numeric.to_numpy()[mask]

        if np.unique(y).size < 2:
            return None

        ones = np.ones((X.shape[0], 1))
        X_augmented = np.hstack([ones, X])
        return X_augmented, y.astype(float), feature_names

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    def _run_gradient_descent(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        weights = np.zeros(X.shape[1], dtype=float)
        lr = self.config.learning_rate

        for _ in range(self.config.epochs):
            predictions = self._sigmoid(X @ weights)
            gradient = X.T @ (predictions - y) / y.size
            weights -= lr * gradient
            if np.linalg.norm(gradient) < self.config.tolerance:
                break

        feature_weights = {name: abs(weight) for name, weight in zip(feature_names, weights[1:])}
        total = sum(feature_weights.values())
        if total <= 0:
            return {}
        return {name: value / total for name, value in feature_weights.items()}

    def train(self, df: pd.DataFrame, metrics: Iterable[str]) -> Dict[str, float]:
        prepared = self._prepare_features(df, list(metrics))
        if prepared is None:
            logger.debug("Supplier ranking trainer: insufficient data to train")
            return {}

        X, y, feature_names = prepared
        weights = self._run_gradient_descent(X, y, feature_names)
        self._last_weights = weights
        return dict(weights)

    def blend_with_policy(
        self, policy_weights: Dict[str, float], learned_weights: Dict[str, float]
    ) -> Dict[str, float]:
        if not learned_weights:
            return dict(policy_weights)

        blend = max(0.0, min(1.0, self.config.blend_ratio))
        combined: Dict[str, float] = {}
        union_keys = set(policy_weights) | set(learned_weights)
        for key in union_keys:
            policy_value = policy_weights.get(key, 0.0)
            learned_value = learned_weights.get(key, 0.0)
            combined[key] = blend * learned_value + (1.0 - blend) * policy_value

        total = sum(value for value in combined.values() if value > 0)
        if total <= 0:
            return {key: value for key, value in policy_weights.items() if value > 0}
        return {key: value / total for key, value in combined.items() if value > 0}

    def build_training_snapshot(
        self,
        df: pd.DataFrame,
        metrics: Sequence[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        if df.empty:
            return None

        label_col = self.config.label_column
        required_columns = [label_col]
        feature_columns: List[str] = []
        for metric in metrics:
            score_col = f"{metric}_score"
            if score_col in df.columns:
                feature_columns.append(score_col)
            elif metric in df.columns:
                feature_columns.append(metric)
        if not feature_columns or label_col not in df.columns:
            return None

        cols = ["supplier_id"] + feature_columns + [label_col]
        available_cols = [col for col in cols if col in df.columns]
        snapshot = df[available_cols].dropna(subset=[label_col], how="any")
        if snapshot.empty:
            return None

        records = snapshot.to_dict("records")
        snapshot: Dict[str, Any] = {
            "metrics": list(metrics),
            "rows": records,
            "label_column": label_col,
        }
        if weights:
            snapshot["policy_weights"] = {
                key: float(value) for key, value in weights.items() if value is not None
            }
        if self._last_weights:
            snapshot["learned_weights"] = dict(self._last_weights)
        return snapshot

    def train_from_records(
        self, records: Sequence[Dict[str, Any]], metrics: Sequence[str], label_column: str
    ) -> Dict[str, float]:
        if not records:
            return {}

        df = pd.DataFrame(records)
        if label_column not in df.columns:
            return {}

        prepared = self._prepare_features(df, metrics)
        if prepared is None:
            return {}
        X, y, feature_names = prepared
        weights = self._run_gradient_descent(X, y, feature_names)
        self._last_weights = weights
        return dict(weights)

