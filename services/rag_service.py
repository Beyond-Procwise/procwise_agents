import logging
import os
import uuid
from typing import List, Dict, Optional
from types import SimpleNamespace

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from qdrant_client import models
from utils.gpu import configure_gpu, load_cross_encoder

configure_gpu()


logger = logging.getLogger(__name__)


class RAGService:
    """Handles embedding upserts and retrieval for RAG workflows."""

    def __init__(self, agent_nick, cross_encoder_cls=None):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.client = agent_nick.qdrant_client
        self.embedder = agent_nick.embedding_model
        # Local FAISS and BM25 indexes to complement Qdrant
        self._faiss_index = None
        self._doc_vectors: List[np.ndarray] = []
        self._documents: List[Dict] = []  # payloads with content
        self._bm25 = None
        self._bm25_corpus: List[List[str]] = []
        if cross_encoder_cls is None:
            from sentence_transformers import CrossEncoder
            cross_encoder_cls = CrossEncoder
        model_name = getattr(
            self.settings,
            "reranker_model",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
        )
        self._reranker = load_cross_encoder(
            model_name, cross_encoder_cls, getattr(self.agent_nick, "device", None)
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into whitespace-aware chunks with overlap."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        configured_max = getattr(self.settings, "rag_chunk_chars", max_chars)
        configured_overlap = getattr(self.settings, "rag_chunk_overlap", overlap)
        max_chars = max(configured_max, 200)
        overlap = max(0, min(configured_overlap, max_chars - 1))
        step = max_chars - overlap if max_chars > overlap else max_chars
        return [cleaned[i : i + max_chars] for i in range(0, len(cleaned), step)]

    def upsert_texts(self, texts: List[str], metadata: Optional[Dict] = None):
        """Encode and upsert texts into Qdrant, FAISS and BM25."""
        points: List[models.PointStruct] = []
        metadata = metadata or {}
        for text in texts:
            chunks = self._chunk_text(text)
            if not chunks:
                continue
            vectors = self.embedder.encode(
                chunks, normalize_embeddings=True, show_progress_bar=False
            )
            record_id = metadata.get("record_id", str(uuid.uuid4()))
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                payload = {"content": chunk, "chunk_id": idx, **metadata}
                point_id = self._build_point_id(record_id, idx)
                vec = np.array(vector, dtype="float32")
                points.append(
                    models.PointStruct(id=point_id, vector=vec.tolist(), payload=payload)
                )

                # --- Update local FAISS/BM25 indexes ---
                self._doc_vectors.append(vec)
                self._documents.append({"id": point_id, **payload})
                self._bm25_corpus.append(chunk.lower().split())

        if self._doc_vectors:
            dim = len(self._doc_vectors[0])
            if self._faiss_index is None:
                index = faiss.IndexFlatIP(dim)
                index = self._maybe_init_gpu_index(index)
                self._faiss_index = index
            self._faiss_index.add(np.vstack(self._doc_vectors))
            self._bm25 = BM25Okapi(self._bm25_corpus)

        if points:
            self.client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )

    def _build_point_id(self, record_id: str, chunk_idx: int) -> str:
        """Create a Qdrant-compatible point ID for the given record chunk."""

        namespace_uuid = self._normalise_uuid(record_id)
        chunk_uuid = uuid.uuid5(namespace_uuid, str(chunk_idx))
        return str(chunk_uuid)

    @staticmethod
    def _normalise_uuid(value: str) -> uuid.UUID:
        """Return a UUID, deriving one deterministically when needed."""

        try:
            return uuid.UUID(str(value))
        except (ValueError, AttributeError, TypeError):
            return uuid.uuid5(uuid.NAMESPACE_URL, str(value))

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[models.Filter] = None,
    ):
        """Retrieve and rerank documents for the given query."""
        candidates: Dict[str, SimpleNamespace] = {}

        # --- FAISS semantic search ---
        if self._faiss_index is not None and self._doc_vectors:
            q_vec = self.embedder.encode(query, normalize_embeddings=True)
            q_vec = np.array([q_vec], dtype="float32")
            scores, ids = self._faiss_index.search(q_vec, top_k * 5)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                candidates[doc["id"]] = SimpleNamespace(
                    id=doc["id"], payload=doc, score=float(score)
                )

        # --- BM25 lexical search ---
        if self._bm25 is not None:
            tokens = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokens)
            for idx, score in sorted(
                enumerate(bm25_scores), key=lambda x: x[1], reverse=True
            )[: top_k * 5]:
                doc = self._documents[idx]
                existing = candidates.get(doc["id"])
                if existing is None or score > existing.score:
                    candidates[doc["id"]] = SimpleNamespace(
                        id=doc["id"], payload=doc, score=float(score)
                    )

        hits = list(candidates.values())

        if hits and filters is not None:
            hits = [h for h in hits if self._payload_matches_filters(h.payload, filters)]
            if not hits:
                candidates = {}

        # --- Fallback to Qdrant if local indexes empty ---
        if not hits:
            q_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
            search_params = models.SearchParams(hnsw_ef=256, exact=False)
            hits = self.client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=q_vec,
                query_filter=filters,
                limit=top_k * 5,
                with_payload=True,
                with_vectors=False,
                search_params=search_params,
            )
            if not hits:
                exact_params = models.SearchParams(hnsw_ef=256, exact=True)
                hits = self.client.search(
                    collection_name=self.settings.qdrant_collection_name,
                    query_vector=q_vec,
                    query_filter=filters,
                    limit=top_k * 5,
                    with_payload=True,
                    with_vectors=False,
                    search_params=exact_params,
                )
            if not hits:
                return []

        # --- Re-rank with cross-encoder ---
        pairs = [
            (query, h.payload.get("content", h.payload.get("summary", "")))
            for h in hits
        ]
        scores = self._reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _maybe_init_gpu_index(self, index: "faiss.Index"):
        """Attempt to move ``index`` to GPU when available.

        FAISS exposes GPU helpers only in GPU-enabled builds. Some
        environments (including CI or developer laptops) only have the CPU
        variant installed; attempting to access ``StandardGpuResources`` in
        those cases raises :class:`AttributeError`. To keep the agentic
        workflow robust we detect this condition and gracefully fall back to
        the CPU index while logging the reason.
        """

        if getattr(self.agent_nick, "device", "cpu") != "cuda":
            return index

        if not hasattr(faiss, "StandardGpuResources"):
            logger.warning(
                "FAISS GPU resources are unavailable in this environment; "
                "falling back to CPU index."
            )
            return index

        try:  # pragma: no cover - depends on GPU libraries
            resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:  # pragma: no cover - hardware specific
            logger.warning(
                "Failed to initialise FAISS GPU resources, falling back to CPU "
                "index: %s",
                exc,
            )
            return index

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------
    def _payload_matches_filters(
        self, payload: Dict, filters: Optional[models.Filter]
    ) -> bool:
        if filters is None:
            return True

        if getattr(filters, "must", None):
            for condition in filters.must:
                if not self._payload_matches_condition(payload, condition):
                    return False

        if getattr(filters, "must_not", None):
            for condition in filters.must_not:
                if self._payload_matches_condition(payload, condition):
                    return False

        if getattr(filters, "should", None):
            should_conditions = [
                condition for condition in filters.should if condition is not None
            ]
            if should_conditions and not any(
                self._payload_matches_condition(payload, condition)
                for condition in should_conditions
            ):
                return False

        return True

    def _payload_matches_condition(self, payload: Dict, condition) -> bool:
        if condition is None:
            return True

        if isinstance(condition, models.Filter):
            return self._payload_matches_filters(payload, condition)

        if isinstance(condition, models.FieldCondition):
            value = payload.get(condition.key)

            match = getattr(condition, "match", None)
            if match is not None:
                if isinstance(match, models.MatchValue):
                    return value == match.value
                if isinstance(match, models.MatchAny):
                    options = getattr(match, "any", None) or []
                    return value in set(options)
                match_except = getattr(models, "MatchExcept", None)
                if match_except is not None and isinstance(match, match_except):
                    excluded = (
                        getattr(match, "except_values", None)
                        or getattr(match, "except_", None)
                        or []
                    )
                    return value not in set(excluded)

            range_ = getattr(condition, "range", None)
            if range_ is not None and value is not None:
                try:
                    return self._value_in_range(float(value), range_)
                except (TypeError, ValueError):
                    return False

            return False

        return False

    @staticmethod
    def _value_in_range(value: float, range_: models.Range) -> bool:
        lower_inclusive = getattr(range_, "gte", None)
        lower_exclusive = getattr(range_, "gt", None)
        upper_inclusive = getattr(range_, "lte", None)
        upper_exclusive = getattr(range_, "lt", None)

        if lower_inclusive is not None and value < lower_inclusive:
            return False
        if lower_exclusive is not None and value <= lower_exclusive:
            return False
        if upper_inclusive is not None and value > upper_inclusive:
            return False
        if upper_exclusive is not None and value >= upper_exclusive:
            return False

        return True
