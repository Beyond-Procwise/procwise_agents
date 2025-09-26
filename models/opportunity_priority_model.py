"""Self-contained clustering model for opportunity prioritisation."""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np

from utils.reference_loader import load_reference_dataset

logger = logging.getLogger(__name__)


class OpportunityPriorityModel:
    """Assign a machine-learned priority score to opportunity findings."""

    def __init__(self) -> None:
        config = load_reference_dataset("opportunity_priority_config")
        self.cluster_count = max(2, int(config.get("cluster_count", 3)))
        self.max_iterations = int(config.get("max_iterations", 50))
        self.tolerance = float(config.get("tolerance", 1e-4))

    def _initial_centroids(self, values: np.ndarray) -> np.ndarray:
        quantiles = np.linspace(0, 1, self.cluster_count + 2)[1:-1]
        centroids = [np.quantile(values, q) for q in quantiles]
        centroids = np.unique(np.concatenate(([values.min()], centroids, [values.max()])))
        if centroids.size < self.cluster_count:
            padding = np.linspace(values.min(), values.max(), self.cluster_count)
            return padding
        return centroids[: self.cluster_count]

    def _kmeans(self, values: np.ndarray) -> np.ndarray:
        centroids = self._initial_centroids(values)
        centroids = centroids.astype(float)

        for _ in range(self.max_iterations):
            distances = np.abs(values[:, None] - centroids[None, :])
            assignments = distances.argmin(axis=1)
            new_centroids = centroids.copy()
            for idx in range(self.cluster_count):
                members = values[assignments == idx]
                if members.size:
                    new_centroids[idx] = members.mean()
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= self.tolerance:
                break
        return assignments

    def assign_scores(self, impacts: Iterable[float]) -> List[float]:
        raw_values = [0.0 if val is None else float(val) for val in impacts]
        impact_array = np.array(raw_values, dtype=float)
        if impact_array.size == 0:
            return []
        if np.allclose(impact_array, impact_array[0]):
            return [1.0] * impact_array.size
        if impact_array.size < self.cluster_count:
            ranks = np.argsort(np.argsort(-impact_array))
            return list(1.0 - ranks / max(1, impact_array.size - 1))

        assignments = self._kmeans(impact_array)
        centroids = []
        for idx in range(self.cluster_count):
            members = impact_array[assignments == idx]
            centroid = members.mean() if members.size else float("nan")
            centroids.append((idx, centroid))
        centroids.sort(key=lambda item: item[1], reverse=True)
        rank_map = {cluster: rank for rank, (cluster, _) in enumerate(centroids)}
        max_rank = max(rank_map.values()) or 1
        scores = [1.0 - (rank_map.get(cluster, max_rank) / max_rank) for cluster in assignments]
        return scores

