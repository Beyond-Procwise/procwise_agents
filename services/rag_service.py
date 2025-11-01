import importlib
import importlib.util
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from types import SimpleNamespace

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.gpu import configure_gpu, load_cross_encoder
from services.semantic_cache import SemanticCacheManager
from services.document_extractor import LayoutAwareParser
from services.semantic_chunker import SemanticChunker

_TRAINING_ROOT = (
    Path(__file__).resolve().parent.parent
    / "resources"
    / "training"
    / "rag"
)
_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
_PREFERENCE_WEIGHTS_PATH = _TRAINING_ROOT / "preference_weights.json"

configure_gpu()


logger = logging.getLogger(__name__)


class RAGService:
    """Handles embedding upserts and retrieval for RAG workflows."""

    def __init__(self, agent_nick, cross_encoder_cls=None):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.client = agent_nick.qdrant_client
        self.embedder = agent_nick.embedding_model
        self.primary_collection = getattr(
            self.settings,
            "qdrant_collection_name",
            "procwise_document_embeddings",
        )
        self.uploaded_collection = getattr(
            self.settings,
            "uploaded_documents_collection_name",
            "uploaded_documents",
        )
        self.static_policy_collection = getattr(
            self.settings,
            "static_policy_collection_name",
            "static_policy",
        )
        self.learning_collection = getattr(
            self.settings,
            "learning_collection_name",
            "learning",
        )
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
            "BAAI/bge-reranker-large",
        )
        self._reranker = load_cross_encoder(
            model_name, cross_encoder_cls, getattr(self.agent_nick, "device", None)
        )
        self._semantic_cache = SemanticCacheManager(
            self.settings, namespace="rag_service"
        )
        self._preference_weights_path = _PREFERENCE_WEIGHTS_PATH
        self._preference_weights = self._load_preference_weights()
        self._rrf_k = 60
        self._layout_parser = LayoutAwareParser()
        self._chunker = SemanticChunker(settings=self.settings)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def reload_preference_weights(self) -> None:
        """Reload retrieval preference weights from disk."""

        self._preference_weights = self._load_preference_weights()

    def _load_preference_weights(self) -> Dict[str, Dict[str, float]]:
        defaults: Dict[str, Dict[str, float]] = {
            "collection": {},
            "document_type": {},
        }
        defaults["collection"][self.primary_collection] = 0.12
        defaults["collection"][self.uploaded_collection] = 0.08
        if self.static_policy_collection:
            defaults["collection"].setdefault(self.static_policy_collection, 0.14)
        defaults["document_type"].setdefault("policy", 0.1)

        path_obj = Path(self._preference_weights_path)
        if not path_obj.exists():
            return defaults

        try:
            loaded = json.loads(path_obj.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "Falling back to default preference weights due to parse failure",
                exc_info=True,
            )
            return defaults

        if not isinstance(loaded, dict):
            return defaults

        result = {
            "collection": dict(defaults["collection"]),
            "document_type": dict(defaults["document_type"]),
        }

        for section_key in ("collection", "document_type"):
            section = loaded.get(section_key)
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    continue
                result[section_key][key] = float(value)

        return result

    def _chunk_text(self, text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk text using the structure-aware chunker for previews."""
        del max_chars, overlap  # legacy parameters retained for compatibility
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        structured = self._layout_parser.from_text(cleaned, scanned=False)
        title_hint = None
        if structured.elements:
            first = structured.elements[0]
            candidate = getattr(first, "text", None)
            if candidate:
                title_hint = candidate.strip()
        chunk_seed = {
            "document_type": "General",
            "source_type": "Upload",
            "title": title_hint or "Preview",
        }
        chunks = self._chunker.build_from_structured(
            structured,
            document_type="General",
            base_metadata=chunk_seed,
            title_hint=chunk_seed["title"],
            default_section="document_overview",
        )
        if not chunks:
            return [cleaned]
        return [chunk.content for chunk in chunks]

    def _embedding_dimension(self) -> int:
        """Determine the embedding dimensionality of the active encoder."""

        getter = getattr(self.embedder, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                dimension = int(getter())
                if dimension > 0:
                    return dimension
            except Exception:  # pragma: no cover - defensive logging only
                logger.debug("Failed to query embedding dimension from model", exc_info=True)

        probe = self.embedder.encode("dimension probe", normalize_embeddings=True)
        if isinstance(probe, np.ndarray):
            if probe.ndim == 1:
                return int(probe.shape[0])
            if probe.ndim == 2:
                return int(probe.shape[1])
        if isinstance(probe, list):
            if probe and isinstance(probe[0], (float, int)):
                return len(probe)
            if probe and isinstance(probe[0], (list, tuple, np.ndarray)):
                first = probe[0]
                return len(first)
        raise RuntimeError("Unable to determine embedding dimension for Qdrant collection initialisation")

    def ensure_collection(self, collection_name: Optional[str] = None) -> None:
        """Ensure the specified Qdrant collection exists with the right vector size."""

        if self.client is None:
            raise RuntimeError("Qdrant client is not configured on AgentNick")

        target = collection_name or self.primary_collection

        try:
            self.client.get_collection(collection_name=target)
            return
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) != 404:
                return
        except Exception:
            try:
                collections = self.client.get_collections().collections
            except Exception:
                collections = []
            if any(getattr(col, "name", None) == target for col in collections):
                return

        dimension = self._embedding_dimension()
        try:
            self.client.create_collection(
                collection_name=target,
                vectors_config=models.VectorParams(
                    size=int(dimension),
                    distance=models.Distance.COSINE,
                ),
            )
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) != 409:
                raise
        except Exception:  # pragma: no cover - defensive logging only
            logger.warning("Failed to ensure Qdrant collection %s", target, exc_info=True)

    def upsert_payloads(
        self,
        payloads: List[Dict[str, Any]],
        text_representation_key: str = "content",
        collection_name: Optional[str] = None,
    ):
        """Encode and upsert structured payloads into Qdrant, FAISS and BM25."""

        if not payloads:
            return

        resolved_collection = collection_name or self.primary_collection
        points: List[models.PointStruct] = []
        texts_for_embedding: List[str] = []
        payloads_for_storage: List[Dict[str, Any]] = []

        for payload in payloads:
            if not isinstance(payload, dict):
                continue

            structured_payload = payload.get("payload")
            if isinstance(structured_payload, dict):
                base_payload = dict(structured_payload)
            else:
                base_payload = dict(payload)
            base_payload.pop("payload", None)

            record_id = (
                base_payload.get("record_id")
                or payload.get("record_id")
                or str(uuid.uuid4())
            )
            base_payload["record_id"] = record_id

            override_collection = self._extract_source_collection(base_payload)
            if override_collection:
                resolved_collection = override_collection

            if text_representation_key in base_payload:
                base_payload.pop(text_representation_key, None)

            text_to_embed = payload.get(text_representation_key)
            if text_to_embed is None and text_representation_key != "content":
                text_to_embed = base_payload.get(text_representation_key)
            if not isinstance(text_to_embed, str) or not text_to_embed.strip():
                summary_fallback = payload.get("text_summary")
                if isinstance(summary_fallback, str) and summary_fallback.strip():
                    text_to_embed = summary_fallback
            if not isinstance(text_to_embed, str) or not text_to_embed.strip():
                try:
                    text_to_embed = json.dumps(base_payload, ensure_ascii=False)
                except (TypeError, ValueError):
                    text_to_embed = ""

            chunks = self._chunk_text(text_to_embed)
            if not chunks:
                continue

            for idx, chunk in enumerate(chunks):
                chunk_payload = dict(base_payload)
                chunk_payload["chunk_id"] = idx
                chunk_payload["chunk_index"] = idx
                if (
                    "content" in chunk_payload
                    and chunk_payload.get("content") != chunk
                ):
                    chunk_payload.setdefault(
                        "_rag_source_content", chunk_payload.get("content")
                    )
                chunk_payload["content"] = chunk
                if (
                    "text_summary" in chunk_payload
                    and chunk_payload.get("text_summary") != chunk
                ):
                    chunk_payload.setdefault(
                        "_rag_source_text_summary",
                        chunk_payload.get("text_summary"),
                    )
                chunk_payload["text_summary"] = chunk
                chunk_payload.setdefault("collection_name", resolved_collection)
                texts_for_embedding.append(chunk)
                payloads_for_storage.append(chunk_payload)

        if not texts_for_embedding:
            return

        vectors = self.embedder.encode(
            texts_for_embedding, normalize_embeddings=True, show_progress_bar=False
        )

        new_vectors: List[np.ndarray] = []
        for idx, (vector, payload) in enumerate(zip(vectors, payloads_for_storage)):
            point_id = self._build_point_id(
                payload.get("record_id", str(uuid.uuid4())), payload.get("chunk_id", idx)
            )
            vec = np.array(vector, dtype="float32")
            points.append(
                models.PointStruct(id=point_id, vector=vec.tolist(), payload=payload)
            )

            # --- Update local FAISS/BM25 indexes ---
            new_vectors.append(vec)
            self._doc_vectors.append(vec)
            self._documents.append({"id": point_id, **payload})
            content_value = payload.get("content")
            if isinstance(content_value, str):
                self._bm25_corpus.append(content_value.lower().split())
            else:
                self._bm25_corpus.append([])

        if new_vectors:
            dim = len(new_vectors[0])
            if self._faiss_index is None:
                index = faiss.IndexFlatIP(dim)
                index = self._maybe_init_gpu_index(index)
                self._faiss_index = index
            stacked = np.vstack(new_vectors)
            self._faiss_index.add(stacked)
            if self._bm25_corpus:
                self._bm25 = BM25Okapi(self._bm25_corpus)

        collection_name = (
            resolved_collection
            or getattr(self.settings, "qdrant_collection_name", None)
        )

        if points and collection_name:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

    def upsert_texts(self, texts: List[str], metadata: Optional[Dict] = None):
        """Backward compatible wrapper around :meth:`upsert_payloads`."""

        metadata = metadata or {}
        payloads: List[Dict[str, Any]] = []
        for text in texts:
            payloads.append({**metadata, "content": text})

        self.upsert_payloads(payloads, text_representation_key="content")

    @staticmethod
    def _extract_source_collection(payload: Dict[str, Any]) -> Optional[str]:
        """Return a collection override specified by the payload, if any."""

        candidate = payload.get("source_collection")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        nested = payload.get("payload")
        if isinstance(nested, dict):
            nested_candidate = nested.get("source_collection")
            if isinstance(nested_candidate, str) and nested_candidate.strip():
                return nested_candidate.strip()
        return None

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
        *,
        session_hint: Optional[str] = None,
        memory_fragments: Optional[List[str]] = None,
        policy_mode: bool = False,
    ):
        """Retrieve and rerank documents for the given query."""
        if not query or not query.strip():
            return []

        base_query = re.sub(r"\s+", " ", query or "").strip()
        if not base_query:
            return []

        hint_text = self._compose_hint_text(session_hint, memory_fragments)
        rewritten_query, auto_conditions = self._rewrite_query(base_query)
        combined_filter = self._merge_filters(filters, auto_conditions)
        policy_mode = bool(policy_mode or self._looks_like_policy_query(base_query, hint_text))
        search_text = rewritten_query or base_query
        if hint_text:
            search_text = f"{search_text}\n\nConversation context: {hint_text}".strip()
        candidates: Dict[str, SimpleNamespace] = {}

        query_matrix: Optional[np.ndarray] = None
        query_vector: Optional[List[float]] = None

        def _ensure_query_vectors() -> Tuple[np.ndarray, List[float]]:
            nonlocal query_matrix, query_vector
            if query_matrix is not None and query_vector is not None:
                return query_matrix, query_vector

            encoded = self.embedder.encode(search_text, normalize_embeddings=True)
            arr = (
                encoded
                if isinstance(encoded, np.ndarray)
                else np.array(encoded, dtype="float32")
            )
            if arr.ndim == 1:
                matrix = arr.reshape(1, -1).astype("float32")
            else:
                matrix = np.array(arr, dtype="float32")
            vector = matrix[0].astype("float32").tolist()
            query_matrix, query_vector = matrix, vector
            return query_matrix, query_vector

        use_cache = combined_filter is None and not auto_conditions and not hint_text
        cached_entries: List[Dict[str, Any]] = []
        if use_cache:
            cached_entries = self._semantic_cache.get_cached_queries(search_text)
            direct_cache = self._rebuild_cached_hits(
                cached_entries, search_text, top_k
            )
            if direct_cache is not None:
                return direct_cache

        prefetch = max(50, top_k * 8)
        weights = {"dense": 1.0, "sparse": 0.75, "qdrant": 1.0, "cache": 0.6}

        def _register_candidate(
            source: str,
            payload: Dict[str, Any],
            doc_id: Any,
            raw_score: float,
            rank: int,
            weight: float,
        ) -> None:
            if not isinstance(payload, dict) or doc_id is None:
                return
            doc_id_str = str(doc_id)
            collection = payload.get("collection_name", self.primary_collection)
            key = f"{collection}:{doc_id_str}"
            entry = candidates.get(key)
            if entry is None:
                entry = SimpleNamespace(
                    id=doc_id_str,
                    payload=dict(payload),
                    score_components={},
                    aggregated=0.0,
                )
                candidates[key] = entry
            rrf = weight * (1.0 / (self._rrf_k + rank))
            entry.score_components[source] = {
                "raw": float(raw_score),
                "rank": int(rank),
                "rrf": float(rrf),
            }
            entry.aggregated += float(rrf)
            entry.payload.setdefault("_query_rewrite", rewritten_query or base_query)
            if hint_text:
                entry.payload.setdefault("_query_context", hint_text)

        if cached_entries:
            self._augment_candidates_from_cache(
                cached_entries, _register_candidate, weights["cache"]
            )

        if self._faiss_index is not None and self._doc_vectors:
            matrix, _ = _ensure_query_vectors()
            scores, ids = self._faiss_index.search(matrix, prefetch)
            for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                payload = dict(doc)
                payload.setdefault(
                    "collection_name", doc.get("collection_name", self.primary_collection)
                )
                _register_candidate(
                    "dense", payload, doc.get("id"), float(score), rank, weights["dense"]
                )

        if self._bm25 is not None and self._documents:
            tokens = rewritten_query.lower().split()
            bm25_scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                enumerate(bm25_scores), key=lambda item: item[1], reverse=True
            )[:prefetch]
            for rank, (idx, score) in enumerate(ranked, 1):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                payload = dict(doc)
                payload.setdefault(
                    "collection_name", doc.get("collection_name", self.primary_collection)
                )
                _register_candidate(
                    "sparse", payload, doc.get("id"), float(score), rank, weights["sparse"]
                )

        collections_to_query: Tuple[Tuple[str, Optional[models.Filter]], ...] = tuple()
        if self.client is not None:
            _, vector = _ensure_query_vectors()
            search_params = models.SearchParams(hnsw_ef=256, exact=False)

            def _search_collection(
                name: str,
                *,
                query_filter: Optional[models.Filter],
            ) -> List[SimpleNamespace]:
                if not name:
                    return []

                try:
                    raw_hits = self.client.search(
                        collection_name=name,
                        query_vector=vector,
                        query_filter=query_filter,
                        limit=prefetch,
                        with_payload=True,
                        with_vectors=False,
                        search_params=search_params,
                    )
                except UnexpectedResponse as exc:
                    if getattr(exc, "status_code", None) == 404:
                        return []
                    logger.warning(
                        "Qdrant search failed for collection %s", name, exc_info=True
                    )
                    return []
                except Exception:
                    logger.warning(
                        "Qdrant search failed for collection %s", name, exc_info=True
                    )
                    return []

                if not raw_hits:
                    try:
                        raw_hits = self.client.search(
                            collection_name=name,
                            query_vector=vector,
                            query_filter=query_filter,
                            limit=prefetch,
                            with_payload=True,
                            with_vectors=False,
                            search_params=models.SearchParams(
                                hnsw_ef=256, exact=True
                            ),
                        )
                    except UnexpectedResponse as exc:
                        if getattr(exc, "status_code", None) == 404:
                            return []
                        logger.warning(
                            "Exact Qdrant search failed for collection %s", name, exc_info=True
                        )
                        return []
                    except Exception:
                        logger.warning(
                            "Exact Qdrant search failed for collection %s", name, exc_info=True
                        )
                        return []

                wrapped: List[SimpleNamespace] = []
                for hit in raw_hits or []:
                    payload = dict(getattr(hit, "payload", {}) or {})
                    payload.setdefault("collection_name", name)
                    wrapped.append(
                        SimpleNamespace(
                            id=str(getattr(hit, "id", payload.get("record_id"))),
                            payload=payload,
                            score=float(getattr(hit, "score", 0.0)),
                        )
                    )
                return wrapped

            base_collections: List[Tuple[str, Optional[models.Filter]]] = []
            seen_collections: Set[str] = set()
            ordered_collections: List[Optional[str]] = [
                self.primary_collection,
                self.uploaded_collection,
                self.static_policy_collection,
                self.learning_collection,
            ]
            if policy_mode and self.static_policy_collection in ordered_collections:
                ordered_collections = [
                    self.static_policy_collection,
                    self.primary_collection,
                    self.uploaded_collection,
                    self.learning_collection,
                ]

            for candidate in ordered_collections:
                if not candidate or candidate in seen_collections:
                    continue
                seen_collections.add(candidate)
                base_collections.append((candidate, combined_filter))

            collections_to_query = tuple(base_collections)

            for name, collection_filter in collections_to_query:
                hits = _search_collection(name, query_filter=collection_filter)
                for rank, hit in enumerate(hits[:prefetch], 1):
                    payload = dict(hit.payload or {})
                    payload.setdefault("collection_name", name)
                    _register_candidate(
                        "qdrant",
                        payload,
                        hit.id,
                        float(getattr(hit, "score", 0.0)),
                        rank,
                        weights["qdrant"],
                    )

        if not candidates:
            return []

        ranked_candidates = sorted(
            candidates.values(), key=lambda item: item.aggregated, reverse=True
        )
        hits = ranked_candidates[:prefetch]
        if not hits:
            return []

        pairs = [
            (
                rewritten_query,
                h.payload.get(
                    "text_summary",
                    h.payload.get("content", h.payload.get("summary", "")),
                ),
            )
            for h in hits
        ]
        scores = self._reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        def _priority_bonus(hit: SimpleNamespace) -> float:
            payload = getattr(hit, "payload", {}) or {}
            collection = payload.get("collection_name", self.primary_collection)
            document_type = str(payload.get("document_type", "")).lower()
            weights_map = getattr(self, "_preference_weights", {}) or {}
            collection_weights = weights_map.get("collection", {}) or {}
            document_weights = weights_map.get("document_type", {}) or {}

            bonus = float(collection_weights.get(collection, 0.0))
            if collection == self.static_policy_collection:
                bonus = max(bonus, 0.12)
                if policy_mode:
                    bonus += 0.22

            doc_bonus = float(document_weights.get(document_type, 0.0))
            if document_type == "policy":
                doc_bonus = max(doc_bonus, 0.08)
                if policy_mode:
                    doc_bonus += 0.15

            return float(bonus + doc_bonus)

        scored_hits: List[Tuple[SimpleNamespace, float, float]] = []
        for hit, score in zip(hits, scores):
            base_score = float(score)
            combined = base_score + hit.aggregated + _priority_bonus(hit)
            scored_hits.append((hit, combined, base_score))

        if not scored_hits:
            return []

        scored_hits.sort(key=lambda item: item[1], reverse=True)

        if top_k <= 0:
            return []

        selected = scored_hits[:top_k]

        def _collection_name(hit_obj: SimpleNamespace) -> str:
            payload = getattr(hit_obj, "payload", {}) or {}
            return payload.get("collection_name", self.primary_collection)

        required_collections: Tuple[str, ...] = tuple(
            name for name, _ in collections_to_query if name
        )

        available_by_collection: Dict[str, List[Tuple[SimpleNamespace, float, float]]] = {}
        for record in scored_hits:
            collection = _collection_name(record[0])
            available_by_collection.setdefault(collection, []).append(record)

        def _has_collection(collection: str) -> bool:
            return any(_collection_name(hit) == collection for hit, _, _ in selected)

        for collection in required_collections:
            candidates_for_collection = available_by_collection.get(collection)
            if not candidates_for_collection or _has_collection(collection):
                continue
            replacement_candidate = candidates_for_collection[0]
            if len(selected) < top_k:
                selected.append(replacement_candidate)
            else:
                lowest_idx = min(range(len(selected)), key=lambda idx: selected[idx][1])
                selected[lowest_idx] = replacement_candidate

        if policy_mode and self.static_policy_collection:
            static_hits = [
                record
                for record in scored_hits
                if _collection_name(record[0]) == self.static_policy_collection
            ]
            if static_hits:
                max_static = min(len(static_hits), max(1, top_k // 2))
                for idx in range(max_static):
                    candidate = static_hits[idx]
                    if candidate not in selected:
                        if len(selected) < top_k:
                            selected.append(candidate)
                        else:
                            lowest_idx = min(
                                range(len(selected)), key=lambda i: selected[i][1]
                            )
                            selected[lowest_idx] = candidate

        deduped: List[Tuple[SimpleNamespace, float, float]] = []
        seen_keys: Set[Tuple[str, str]] = set()
        for hit, combined, base in sorted(selected, key=lambda item: item[1], reverse=True):
            payload = getattr(hit, "payload", {}) or {}
            key = (str(getattr(hit, "id", payload.get("record_id"))), _collection_name(hit))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append((hit, combined, base))

        results = [hit for hit, _, _ in deduped[:top_k]]

        if use_cache:
            cache_payload: List[Dict[str, Any]] = []
            for hit, combined, base in deduped[:top_k]:
                payload = self._normalise_payload_for_cache(
                    getattr(hit, "payload", {}) or {}
                )
                cache_payload.append(
                    {
                        "id": getattr(hit, "id", payload.get("record_id")),
                        "collection_name": payload.get(
                            "collection_name", self.primary_collection
                        ),
                        "payload": payload,
                        "score": base,
                        "combined_score": combined,
                        "aggregated_score": getattr(hit, "aggregated", 0.0),
                    }
                )
            self._semantic_cache.set_query_results(search_text, cache_payload)

        return results

    def _rewrite_query(
        self, query: str
    ) -> Tuple[str, List[models.FieldCondition]]:
        normalised = re.sub(r"\s+", " ", query or "").strip()
        if not normalised:
            return "", []

        normalised = re.sub(
            r"(?i)\bpo[-\s]?(\d{3,})\b",
            lambda match: f"purchase order {match.group(1)}",
            normalised,
        )

        conditions: List[models.FieldCondition] = []

        def _ensure_condition(field: str, value: str) -> None:
            for condition in conditions:
                if getattr(condition, "key", None) == field:
                    match_obj = getattr(condition, "match", None)
                    if isinstance(match_obj, models.MatchValue) and match_obj.value == value:
                        return
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
            )

        lowered = normalised.lower()
        if "policy" in lowered:
            _ensure_condition("source_type", "Policy")
        if "invoice" in lowered:
            _ensure_condition("source_type", "Invoice")
        if "purchase order" in lowered:
            _ensure_condition("source_type", "PO")
        if "quote" in lowered:
            _ensure_condition("source_type", "Quote")

        rounds = re.findall(r"\bround(?:\s+|#)(\d+)\b", lowered)
        for round_id in rounds:
            _ensure_condition("round_id", round_id)

        years = sorted(set(re.findall(r"\b(20[0-4]\d|19\d{2})\b", lowered)))
        for year in years:
            conditions.append(
                models.FieldCondition(
                    key="effective_date", match=models.MatchText(text=year)
                )
            )

        synonyms: Dict[str, List[str]] = {
            "payment terms": ["net terms", "discount period"],
            "lead time": ["delivery time", "turnaround"],
            "escalation": ["escalation clause"],
        }
        expansions: List[str] = []
        for phrase, alternatives in synonyms.items():
            if phrase in lowered:
                expansions.extend(alternatives)

        if expansions:
            unique = " ".join(sorted(set(expansions)))
            normalised = f"{normalised} {unique}".strip()

        return normalised, conditions

    def _compose_hint_text(
        self,
        session_hint: Optional[str],
        memory_fragments: Optional[List[str]],
    ) -> str:
        hints: List[str] = []
        if session_hint:
            snippet = re.sub(r"\s+", " ", session_hint).strip()
            if snippet:
                hints.append(snippet)
        if memory_fragments:
            for fragment in memory_fragments:
                cleaned = re.sub(r"\s+", " ", str(fragment or "")).strip()
                if cleaned:
                    hints.append(cleaned)
        return " ".join(hints)

    def _looks_like_policy_query(
        self, query: str, hint_text: Optional[str]
    ) -> bool:
        combined = " ".join(
            part.strip().lower()
            for part in (query or "", hint_text or "")
            if part and part.strip()
        )
        if not combined:
            return False

        policy_tokens = (
            "policy",
            "policies",
            "approved supplier",
            "supplier approval",
            "compliance",
            "procurement rule",
            "sourcing rule",
            "maverick",
            "non-approved",
            "delegation of authority",
        )
        return any(token in combined for token in policy_tokens)

    def _merge_filters(
        self,
        base_filter: Optional[models.Filter],
        extra_conditions: List[models.FieldCondition],
    ) -> Optional[models.Filter]:
        if not extra_conditions:
            return base_filter
        if base_filter is None:
            return models.Filter(must=list(extra_conditions))

        merged_must = list(getattr(base_filter, "must", []) or [])
        merged_must.extend(extra_conditions)
        merged_must_not = list(getattr(base_filter, "must_not", []) or [])
        merged_should = list(getattr(base_filter, "should", []) or [])
        return models.Filter(
            must=merged_must,
            must_not=merged_must_not if merged_must_not else None,
            should=merged_should if merged_should else None,
        )

    def create_langchain_retriever_tool(
        self,
        name: str = "procwise_rag_retriever",
        description: Optional[str] = None,
        top_k: int = 5,
    ):
        """Expose this service as a LangChain retriever tool."""

        if importlib.util.find_spec("langchain_core") is None:
            raise RuntimeError(
                "LangChain integrations requested but 'langchain-core' is missing."
            )
        if importlib.util.find_spec("langchain.tools") is None:
            raise RuntimeError(
                "LangChain tool support requires the 'langchain' extra dependencies."
            )

        retriever_module = importlib.import_module("langchain_core.retrievers")
        documents_module = importlib.import_module("langchain_core.documents")
        tools_module = importlib.import_module("langchain.tools")

        base_retriever_cls = getattr(retriever_module, "BaseRetriever")
        document_cls = getattr(documents_module, "Document")
        create_retriever_tool = getattr(tools_module, "create_retriever_tool")

        rag_service = self

        class _RAGServiceRetriever(base_retriever_cls):
            def __init__(self, limit: int):
                super().__init__()
                self._limit = max(1, int(limit))

            def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs):
                hits = rag_service.search(query, top_k=self._limit)
                documents: List[Any] = []
                for hit in hits:
                    payload: Dict[str, Any]
                    if hasattr(hit, "payload") and isinstance(hit.payload, dict):
                        payload = dict(hit.payload)
                    elif isinstance(hit, dict):
                        payload = dict(hit)
                    else:
                        payload = {"content": str(hit)}
                    text = (
                        payload.get("content")
                        or payload.get("text_summary")
                        or payload.get("summary")
                        or ""
                    )
                    documents.append(
                        document_cls(page_content=str(text), metadata=payload)
                    )
                return documents

            async def _aget_relevant_documents(  # type: ignore[override]
                self,
                query: str,
                *,
                run_manager=None,
                **kwargs,
            ):
                return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)

        tool_description = description or (
            "Retrieve procurement knowledge base documents from Qdrant to "
            "ground supplier negotiations and policy checks."
        )
        retriever = _RAGServiceRetriever(limit=top_k)
        return create_retriever_tool(retriever, name=name, description=tool_description)

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

    # ------------------------------------------------------------------
    # LangCache integration helpers
    # ------------------------------------------------------------------
    def _rebuild_cached_hits(
        self,
        cached_entries: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> Optional[List[SimpleNamespace]]:
        rebuilt: List[SimpleNamespace] = []
        for entry in cached_entries:
            response = entry.get("response")
            if not isinstance(response, str):
                continue
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                continue
            hits = payload.get("hits")
            if not isinstance(hits, list):
                continue
            similarity = float(entry.get("similarity") or 0.0)
            if entry.get("prompt") == query and similarity >= 0.995:
                for hit in hits[:top_k]:
                    rebuilt_hit = self._convert_cache_hit(hit)
                    if rebuilt_hit is not None:
                        collection_name = (
                            getattr(rebuilt_hit, "payload", {}) or {}
                        ).get("collection_name")
                        if (
                            collection_name
                            and collection_name == self.learning_collection
                        ):
                            continue
                        rebuilt.append(rebuilt_hit)
                if rebuilt:
                    return rebuilt[:top_k]
        return None

    def _augment_candidates_from_cache(
        self,
        cached_entries: List[Dict[str, Any]],
        register: Callable[[str, Dict[str, Any], Any, float, int, float], None],
        weight: float,
    ) -> None:
        for entry in cached_entries:
            response = entry.get("response")
            if not isinstance(response, str):
                continue
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                continue
            hits = payload.get("hits")
            if not isinstance(hits, list):
                continue
            similarity = float(entry.get("similarity") or 0.0)
            similarity_weight = max(0.45, min(1.0, similarity))
            effective_weight = weight * similarity_weight
            for rank, cached_hit in enumerate(hits, 1):
                rebuilt = self._convert_cache_hit(cached_hit)
                if rebuilt is None:
                    continue
                score = float(
                    cached_hit.get("combined_score", cached_hit.get("score", 0.0))
                )
                register(
                    "cache",
                    rebuilt.payload,
                    rebuilt.id,
                    score,
                    rank,
                    effective_weight,
                )

    def _convert_cache_hit(self, cached_hit: Dict[str, Any]) -> Optional[SimpleNamespace]:
        if not isinstance(cached_hit, dict):
            return None
        payload = cached_hit.get("payload")
        if not isinstance(payload, dict):
            return None
        payload = dict(payload)
        collection = cached_hit.get(
            "collection_name", payload.get("collection_name", self.primary_collection)
        )
        payload.setdefault("collection_name", collection)
        hit_id = cached_hit.get("id") or payload.get("record_id")
        if hit_id is None:
            return None
        score = float(cached_hit.get("combined_score", cached_hit.get("score", 0.0)))
        return SimpleNamespace(id=str(hit_id), payload=payload, score=score)

    def _normalise_payload_for_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def _normalise(value: Any):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                return [_normalise(item) for item in value]
            if isinstance(value, tuple):
                return [_normalise(item) for item in value]
            if isinstance(value, dict):
                return {str(key): _normalise(val) for key, val in value.items()}
            return str(value)

        try:
            json.dumps(payload)
            return payload
        except (TypeError, ValueError):
            return _normalise(payload)

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
