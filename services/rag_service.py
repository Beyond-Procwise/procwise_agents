import importlib
import importlib.util
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set
from types import SimpleNamespace

import hashlib

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.gpu import configure_gpu, load_cross_encoder
from services.semantic_cache import SemanticCacheManager
from services.document_extractor import LayoutAwareParser
from services.semantic_chunker import SemanticChunker, SemanticChunk

DISALLOWED_METADATA_KEYS: Set[str] = {
    "effective_date",
    "supplier",
    "doc_version",
    "round_id",
}

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
        self._payload_index_cache: Dict[str, Set[str]] = {}
        self._prefetch_limit = max(
            int(getattr(self.settings, "rag_prefetch_limit", 48) or 1), 1
        )
        self._reranker_batch_size = max(
            int(getattr(self.settings, "rag_reranker_batch_size", 32) or 1), 1
        )
        self._reranker_max_chars = max(
            int(getattr(self.settings, "rag_reranker_max_chars", 1400) or 256), 256
        )
        self._rerank_cache_size = max(
            int(getattr(self.settings, "rag_reranker_cache_size", 384) or 0), 0
        )
        self._rerank_cache: "OrderedDict[str, float]" = OrderedDict()
        self._rerank_cache_lock = threading.Lock()
        self._qdrant_search_workers = max(
            int(getattr(self.settings, "rag_qdrant_search_workers", 4) or 1), 1
        )
        self._qdrant_hnsw_ef = max(
            int(getattr(self.settings, "rag_qdrant_search_ef", 160) or 1), 16
        )
        self._qdrant_search_params = models.SearchParams(
            hnsw_ef=self._qdrant_hnsw_ef,
            exact=False,
        )
        self._qdrant_exact_search_params = models.SearchParams(
            hnsw_ef=self._qdrant_hnsw_ef,
            exact=True,
        )

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

    def _chunk_text(
        self, text: str, max_chars: int = 1000, overlap: int = 200
    ) -> List[SemanticChunk]:
        """Chunk text using the structure-aware chunker for previews.

        The ``max_chars`` and ``overlap`` parameters are retained for backward
        compatibility with older call sites. They now map to the runtime
        configuration derived from :class:`SemanticChunker`.
        """

        del max_chars, overlap  # legacy parameters retained for compatibility
        cleaned = (text or "").strip()
        if not cleaned:
            return []

        try:
            structured = self._layout_parser.from_text(cleaned, scanned=False)
        except Exception:
            logger.debug("Layout parsing failed; using fallback chunking", exc_info=True)
            structured = None

        chunks: List[SemanticChunk] = []
        title_hint = None
        if structured and structured.elements:
            first = structured.elements[0]
            candidate = getattr(first, "text", None)
            if candidate:
                title_hint = candidate.strip()
        chunk_seed = {
            "document_type": "General",
            "source_type": "Upload",
            "title": title_hint or "Preview",
        }

        if structured:
            try:
                chunks = self._chunker.build_from_structured(
                    structured,
                    document_type="General",
                    base_metadata=chunk_seed,
                    title_hint=chunk_seed["title"],
                    default_section="document_overview",
                )
            except Exception:
                logger.debug(
                    "Semantic chunking failed; falling back to heuristic chunker",
                    exc_info=True,
                )
                chunks = []

        if not chunks or self._requires_fallback(chunks):
            return self._fallback_chunk_text(cleaned, seed_metadata=chunk_seed)

        return chunks

    def _requires_fallback(self, chunks: List[SemanticChunk]) -> bool:
        """Determine whether the semantic chunker output needs refinement."""

        if not chunks:
            return True

        if len(chunks) > 1:
            return False

        char_limit = self._chunk_char_limit()
        return len(chunks[0].content or "") > int(char_limit * 1.2)

    def _chunk_char_limit(self) -> int:
        """Return the configured character limit for fallback chunking."""

        approx_chars = getattr(self.settings, "rag_chunk_chars", None)
        try:
            approx_value = int(approx_chars) if approx_chars else 0
        except (TypeError, ValueError):
            approx_value = 0
        if approx_value <= 0:
            approx_value = 1800
        return max(900, min(approx_value, 2400))

    def _chunk_overlap_chars(self) -> int:
        """Return the configured overlap (in characters) for fallback chunks."""

        overlap_value = getattr(self.settings, "rag_chunk_overlap", None)
        try:
            overlap = int(overlap_value) if overlap_value else 0
        except (TypeError, ValueError):
            overlap = 0
        if overlap <= 0:
            overlap = int(self._chunk_char_limit() * 0.1)
        return max(120, min(overlap, int(self._chunk_char_limit() * 0.4)))

    def _fallback_chunk_text(
        self, text: str, *, seed_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SemanticChunk]:
        """Fallback heuristic chunker when structural parsing fails.

        This splitter groups paragraphs and long sentences into overlapping
        windows so that downstream retrieval receives semantically coherent
        spans rather than single large blocks of text.
        """

        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []

        char_limit = self._chunk_char_limit()
        overlap_chars = self._chunk_overlap_chars()

        paragraphs = [
            part.strip()
            for part in re.split(r"\n{2,}", text)
            if part and part.strip()
        ]
        if not paragraphs:
            paragraphs = [cleaned]

        expanded_parts: List[str] = []
        for paragraph in paragraphs:
            expanded_parts.extend(self._split_paragraph(paragraph, char_limit))

        chunks: List[SemanticChunk] = []
        buffer: List[str] = []
        buffer_len = 0
        idx = 0
        while idx < len(expanded_parts):
            part = expanded_parts[idx]
            part_len = len(part)
            if buffer and buffer_len + part_len + 2 > char_limit:
                chunk_text = self._normalise_chunk("\n\n".join(buffer))
                if chunk_text:
                    chunks.append(
                        SemanticChunk(
                            content=chunk_text,
                            metadata=self._fallback_metadata(
                                seed_metadata, len(chunks)
                            ),
                        )
                    )
                buffer, buffer_len = self._build_overlap_buffer(
                    buffer, overlap_chars
                )
                continue

            buffer.append(part)
            buffer_len += part_len + 2
            idx += 1

        if buffer:
            chunk_text = self._normalise_chunk("\n\n".join(buffer))
            if chunk_text:
                chunks.append(
                    SemanticChunk(
                        content=chunk_text,
                        metadata=self._fallback_metadata(seed_metadata, len(chunks)),
                    )
                )

        return chunks

    def _split_paragraph(self, paragraph: str, limit: int) -> List[str]:
        """Split long paragraphs into sentence-based windows."""

        paragraph = paragraph.strip()
        if len(paragraph) <= limit:
            return [paragraph]

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return [paragraph[i : i + limit] for i in range(0, len(paragraph), limit)]

        parts: List[str] = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            if current and current_len + sentence_len + 1 > limit:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sentence)
            current_len += sentence_len + 1
        if current:
            parts.append(" ".join(current))
        return parts

    def _build_overlap_buffer(
        self, buffer: List[str], overlap_chars: int
    ) -> Tuple[List[str], int]:
        if not buffer or overlap_chars <= 0:
            return [], 0

        tail = buffer[-1]
        overlap_text = self._tail_sentences(tail, overlap_chars)
        if not overlap_text:
            return [], 0
        return [overlap_text], len(overlap_text)

    def _tail_sentences(self, text: str, limit: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return text[-limit:].strip()

        collected: List[str] = []
        total = 0
        for sentence in reversed(sentences):
            collected.insert(0, sentence)
            total += len(sentence) + 1
            if total >= limit:
                break
        return " ".join(collected).strip()

    def _normalise_chunk(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _fallback_metadata(
        self, seed_metadata: Optional[Dict[str, Any]], index: int
    ) -> Dict[str, Any]:
        metadata = dict(seed_metadata or {})
        metadata.setdefault("section", "document_overview")
        metadata.setdefault("section_path", "document_overview")
        metadata.setdefault("content_type", "paragraph")
        metadata.setdefault("chunk_strategy", "fallback")
        metadata.setdefault("fallback_index", index)
        return metadata

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
            self._refresh_payload_index_cache(target)
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
            self._payload_index_cache.pop(target, None)
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) != 409:
                raise
        except Exception:  # pragma: no cover - defensive logging only
            logger.warning("Failed to ensure Qdrant collection %s", target, exc_info=True)

        self._refresh_payload_index_cache(target)

    def _refresh_payload_index_cache(self, collection_name: str) -> None:
        if not self.client or not collection_name:
            return
        get_collection = getattr(self.client, "get_collection", None)
        if get_collection is None:
            return
        try:
            info = get_collection(collection_name=collection_name)
        except Exception:
            return

        schema = getattr(info, "payload_schema", {}) or {}
        if isinstance(schema, dict):
            self._payload_index_cache[collection_name] = set(schema.keys())

    def _ensure_payload_index(
        self,
        collection_name: Optional[str],
        field_name: Optional[str],
        schema: models.PayloadSchemaType = models.PayloadSchemaType.KEYWORD,
    ) -> None:
        if not collection_name or not field_name or not self.client:
            return

        create_index = getattr(self.client, "create_payload_index", None)
        if create_index is None:
            return

        cached = self._payload_index_cache.setdefault(collection_name, set())
        if field_name in cached:
            return

        if not cached:
            self._refresh_payload_index_cache(collection_name)
            cached = self._payload_index_cache.setdefault(collection_name, set())
            if field_name in cached:
                return

        try:
            create_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
            cached.add(field_name)
        except UnexpectedResponse as exc:
            status = getattr(exc, "status_code", None)
            if status in (400, 409):
                cached.add(field_name)
            else:  # pragma: no cover - diagnostic logging only
                logger.debug(
                    "Failed to create payload index for %s.%s", collection_name, field_name, exc_info=True
                )
        except Exception:  # pragma: no cover - diagnostic logging only
            logger.debug(
                "Failed to ensure payload index for %s.%s", collection_name, field_name, exc_info=True
            )

    def _collect_filter_fields(self, query_filter: Optional[models.Filter]) -> Set[str]:
        if query_filter is None:
            return set()

        fields: Set[str] = set()
        stack: List[Any] = [query_filter]
        while stack:
            current = stack.pop()
            if isinstance(current, models.FieldCondition):
                key = getattr(current, "key", None)
                if isinstance(key, str) and key.strip():
                    fields.add(key.strip())
                continue

            if isinstance(current, models.Filter):
                for attr in ("must", "must_not", "should"):
                    items = getattr(current, attr, None) or []
                    for item in items:
                        if item is not None:
                            stack.append(item)

        return fields

    def _query_qdrant_collection(
        self,
        collection_name: str,
        vector: Sequence[float],
        *,
        query_filter: Optional[models.Filter],
        limit: int,
    ) -> List[SimpleNamespace]:
        if not collection_name or self.client is None:
            return []

        self._ensure_filter_indexes(collection_name, query_filter)

        search_kwargs = dict(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        def _copy_params(params: Optional[models.SearchParams]) -> Optional[models.SearchParams]:
            if params is None:
                return None
            copier = getattr(params, "model_copy", None)
            if callable(copier):
                try:
                    return copier(deep=True)
                except Exception:
                    return params
            return params

        try:
            raw_hits = self.client.search(
                **search_kwargs,
                search_params=_copy_params(self._qdrant_search_params),
            )
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                return []
            logger.warning(
                "Qdrant search failed for collection %s", collection_name, exc_info=True
            )
            return []
        except Exception:
            logger.warning(
                "Qdrant search failed for collection %s", collection_name, exc_info=True
            )
            return []

        if not raw_hits:
            try:
                raw_hits = self.client.search(
                    **search_kwargs,
                    search_params=_copy_params(self._qdrant_exact_search_params),
                )
            except UnexpectedResponse as exc:
                if getattr(exc, "status_code", None) == 404:
                    return []
                logger.warning(
                    "Exact Qdrant search failed for collection %s",
                    collection_name,
                    exc_info=True,
                )
                return []
            except Exception:
                logger.warning(
                    "Exact Qdrant search failed for collection %s",
                    collection_name,
                    exc_info=True,
                )
                return []

        wrapped: List[SimpleNamespace] = []
        for hit in raw_hits or []:
            payload = dict(getattr(hit, "payload", {}) or {})
            payload.setdefault("collection_name", collection_name)
            wrapped.append(
                SimpleNamespace(
                    id=str(getattr(hit, "id", payload.get("record_id"))),
                    payload=payload,
                    score=float(getattr(hit, "score", 0.0)),
                )
            )

        return wrapped

    def _search_collections_parallel(
        self,
        requests: Sequence[Tuple[str, Optional[models.Filter]]],
        vector: Sequence[float],
        *,
        limit: int,
    ) -> Dict[str, List[SimpleNamespace]]:
        if not requests or self.client is None:
            return {}

        tasks: List[Tuple[str, Optional[models.Filter]]] = [
            (name, query_filter)
            for name, query_filter in requests
            if isinstance(name, str) and name
        ]
        if not tasks:
            return {}

        max_workers = min(len(tasks), max(self._qdrant_search_workers, 1))
        results: Dict[str, List[SimpleNamespace]] = {}

        if max_workers <= 1:
            for name, query_filter in tasks:
                results[name] = self._query_qdrant_collection(
                    name,
                    vector,
                    query_filter=query_filter,
                    limit=limit,
                )
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._query_qdrant_collection,
                    name,
                    vector,
                    query_filter=query_filter,
                    limit=limit,
                ): name
                for name, query_filter in tasks
            }

            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result()
                except Exception:
                    logger.warning(
                        "Parallel Qdrant search failed for collection %s",
                        name,
                        exc_info=True,
                    )
                    results[name] = []

        return results

    def _prepare_reranker_text(self, payload: Dict[str, Any]) -> str:
        """Extract a concise text snippet for cross-encoder reranking."""

        if not isinstance(payload, dict):
            return ""

        parts: List[str] = []

        highlights = payload.get("highlights")
        if isinstance(highlights, str) and highlights.strip():
            parts.append(highlights.strip())
        elif isinstance(highlights, (list, tuple, set)):
            highlight_bits = [
                str(item).strip()
                for item in highlights
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
            if highlight_bits:
                parts.append(" ".join(highlight_bits))

        text_keys = (
            "text_summary",
            "summary",
            "chunk_text",
            "content",
            "text",
            "description",
        )
        for key in text_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
            if isinstance(value, (list, tuple)):
                merged = " ".join(
                    str(item).strip()
                    for item in value
                    if isinstance(item, (str, int, float)) and str(item).strip()
                ).strip()
                if merged:
                    parts.append(merged)
                    break

        if not parts:
            fallback_bits: List[str] = []
            for key in ("title", "document_type", "source_type", "record_id"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    fallback_bits.append(value.strip())
            if fallback_bits:
                parts.append(" - ".join(fallback_bits))

        combined = " ".join(parts).strip()
        if not combined:
            return ""

        max_chars = self._reranker_max_chars
        if len(combined) <= max_chars:
            return combined

        truncated = combined[:max_chars].rsplit(" ", 1)[0].strip()
        if not truncated:
            truncated = combined[:max_chars].strip()
        return truncated + "…"

    def _build_rerank_cache_key(
        self, query: Optional[str], document: Optional[str]
    ) -> Optional[str]:
        if self._rerank_cache_size <= 0:
            return None
        if not query or not document:
            return None
        try:
            query_text = str(query).strip()
            document_text = str(document).strip()
        except Exception:
            return None
        if not query_text or not document_text:
            return None
        trimmed_document = document_text[: self._reranker_max_chars]
        payload = f"{query_text}\u241f{trimmed_document}"
        try:
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _batched_reranker_predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Score ``pairs`` using the cross-encoder in small batches.

        Results are memoised in a bounded LRU cache so repeated or highly
        similar queries avoid recomputing expensive cross-encoder scores.
        """

        if not pairs:
            return []

        cache_keys: List[Optional[str]] = [
            self._build_rerank_cache_key(query, document) for query, document in pairs
        ]
        scores: List[Optional[float]] = [None] * len(pairs)

        if self._rerank_cache_size > 0:
            with self._rerank_cache_lock:
                for idx, key in enumerate(cache_keys):
                    if key is None:
                        continue
                    cached = self._rerank_cache.get(key)
                    if cached is None:
                        continue
                    # Refresh the LRU position
                    self._rerank_cache.move_to_end(key)
                    scores[idx] = float(cached)

        missing_pairs: List[Tuple[str, str]] = []
        missing_indices: List[int] = []
        for idx, (pair, cached_score) in enumerate(zip(pairs, scores)):
            if cached_score is None:
                missing_pairs.append(pair)
                missing_indices.append(idx)

        if missing_pairs:
            batch_size = max(self._reranker_batch_size, 1)
            computed_scores: List[float] = []
            for start in range(0, len(missing_pairs), batch_size):
                chunk = list(missing_pairs[start : start + batch_size])
                if not chunk:
                    continue
                chunk_scores = self._reranker.predict(chunk)
                if hasattr(chunk_scores, "tolist"):
                    chunk_scores = chunk_scores.tolist()
                computed_scores.extend(float(score) for score in chunk_scores)

            for idx, score in zip(missing_indices, computed_scores):
                scores[idx] = float(score)

            if self._rerank_cache_size > 0 and computed_scores:
                with self._rerank_cache_lock:
                    for idx, score in zip(missing_indices, computed_scores):
                        key = cache_keys[idx]
                        if key is None:
                            continue
                        self._rerank_cache[key] = float(score)
                        while len(self._rerank_cache) > self._rerank_cache_size:
                            self._rerank_cache.popitem(last=False)

        return [float(score) if score is not None else 0.0 for score in scores]

    def _ensure_filter_indexes(
        self, collection_name: Optional[str], query_filter: Optional[models.Filter]
    ) -> None:
        if not collection_name or query_filter is None:
            return

        try:
            fields = self._collect_filter_fields(query_filter)
        except Exception:  # pragma: no cover - diagnostic logging only
            logger.debug(
                "Failed to inspect filter for collection %s", collection_name, exc_info=True
            )
            return

        for field in fields:
            self._ensure_payload_index(collection_name, field)

    def _purge_disallowed_metadata(self, collection_name: str) -> None:
        if not DISALLOWED_METADATA_KEYS:
            return
        try:
            selector = models.FilterSelector(filter=models.Filter(must=[]))
            self.client.delete_payload(
                collection_name=collection_name,
                keys=list(DISALLOWED_METADATA_KEYS),
                points=selector,
                wait=False,
            )
        except Exception:
            logger.debug(
                "Failed to purge disallowed metadata for collection %s",
                collection_name,
                exc_info=True,
            )

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
            for key in DISALLOWED_METADATA_KEYS:
                base_payload.pop(key, None)

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

                if isinstance(chunk, SemanticChunk):
                    chunk_text = chunk.content
                    chunk_metadata = chunk.metadata or {}
                else:
                    chunk_text = str(chunk)
                    chunk_metadata = {}

                chunk_text = (chunk_text or "").strip()
                if not chunk_text:
                    continue

                for key, value in (chunk_metadata or {}).items():
                    if value in (None, "", [], {}):
                        continue
                    chunk_payload.setdefault(key, value)

                if (
                    "content" in chunk_payload
                    and chunk_payload.get("content") != chunk_text
                ):
                    chunk_payload.setdefault(
                        "_rag_source_content", chunk_payload.get("content")
                    )
                chunk_payload["content"] = chunk_text
                if (
                    "text_summary" in chunk_payload
                    and chunk_payload.get("text_summary") != chunk_text
                ):
                    chunk_payload.setdefault(
                        "_rag_source_text_summary",
                        chunk_payload.get("text_summary"),
                    )
                chunk_payload["text_summary"] = chunk_text
                chunk_payload.setdefault("collection_name", resolved_collection)
                texts_for_embedding.append(chunk_text)
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
            self._purge_disallowed_metadata(collection_name)
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
        nltk_features: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
    ):
        """Retrieve and rerank documents for the given query."""
        if not query or not query.strip():
            return []

        base_query = re.sub(r"\s+", " ", query or "").strip()
        if not base_query:
            return []

        hint_text = self._compose_hint_text(session_hint, memory_fragments)
        feature_keywords: List[str] = []
        feature_phrases: List[str] = []
        nltk_features: Optional[Dict[str, Any]] = None
        if isinstance(nltk_features, dict):
            raw_keywords = nltk_features.get("keywords", [])
            raw_phrases = nltk_features.get("key_phrases", [])
            feature_keywords = [
                str(item).strip()
                for item in raw_keywords
                if isinstance(item, str) and str(item).strip()
            ]
            feature_phrases = [
                str(item).strip()
                for item in raw_phrases
                if isinstance(item, str) and str(item).strip()
            ]
            if feature_keywords or feature_phrases:
                logger.debug(
                    "Applying NLTK feature hints: keywords=%s phrases=%s",
                    feature_keywords[:8],
                    feature_phrases[:5],
                )

        (
            rewritten_query,
            auto_conditions,
            focus_document_types,
        ) = self._rewrite_query(base_query)
        combined_filter = self._merge_filters(filters, auto_conditions)
        combined_filter, session_key = self._separate_session_scope(
            combined_filter, session_id
        )
        policy_mode = bool(policy_mode or self._looks_like_policy_query(base_query, hint_text))
        search_text = rewritten_query or base_query
        focus_document_types = {
            token.strip().lower()
            for token in (focus_document_types or [])
            if isinstance(token, str) and token.strip()
        }
        feature_hint_parts = feature_keywords[:6] + feature_phrases[:4]
        if feature_hint_parts:
            features_line = ", ".join(dict.fromkeys(feature_hint_parts))
            search_text = f"{search_text}\n\nFocus terms: {features_line}".strip()
            if hint_text:
                hint_text = f"{hint_text} | focus terms: {features_line}"
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

        prefetch_limit = max(self._prefetch_limit, top_k)
        prefetch = min(prefetch_limit, max(32, top_k * 6))
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
            normalised_payload = {
                key: value
                for key, value in payload.items()
                if key not in DISALLOWED_METADATA_KEYS
            }
            doc_id_str = str(doc_id)
            collection = normalised_payload.get(
                "collection_name", self.primary_collection
            )
            chunk_marker = normalised_payload.get("chunk_id")
            if chunk_marker is None:
                chunk_marker = normalised_payload.get("chunk_index")
            if chunk_marker is None:
                chunk_marker = normalised_payload.get("chunk_hash")
            key = (
                f"{collection}:{doc_id_str}:{chunk_marker}"
                if chunk_marker is not None
                else f"{collection}:{doc_id_str}"
            )
            entry = candidates.get(key)
            if entry is None:
                entry = SimpleNamespace(
                    id=(
                        f"{doc_id_str}:{chunk_marker}"
                        if chunk_marker is not None
                        else doc_id_str
                    ),
                    payload=dict(normalised_payload),
                    score_components={},
                    aggregated=0.0,
                    chunk_id=chunk_marker,
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
            if feature_keywords:
                tokens.extend(
                    keyword.lower()
                    for keyword in feature_keywords
                    if keyword.lower() not in tokens
                )
            if feature_phrases:
                for phrase in feature_phrases:
                    for part in phrase.lower().split():
                        if part not in tokens:
                            tokens.append(part)
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
        base_collections: List[Tuple[str, Optional[models.Filter]]] = []

        def _filter_signature(value: Optional[models.Filter]) -> str:
            if value is None:
                return "∅"
            try:
                return value.model_dump_json(sort_keys=True)
            except Exception:
                try:
                    return json.dumps(value.model_dump(), sort_keys=True)
                except Exception:
                    return repr(value)

        seen_pairs: Set[Tuple[str, str]] = set()

        def _append_collection(
            name: Optional[str], filt: Optional[models.Filter]
        ) -> None:
            if not name:
                return
            signature = _filter_signature(filt)
            key = (name, signature)
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            base_collections.append((name, filt))

        forced_names: List[str] = []
        for candidate in collections or []:
            try:
                cleaned_name = str(candidate).strip()
            except Exception:
                cleaned_name = ""
            if cleaned_name:
                forced_names.append(cleaned_name)
        forced_collections: Tuple[str, ...] = tuple(forced_names)

        session_specific_filter: Optional[models.Filter] = None
        if session_key:
            session_specific_filter = self._merge_filters(
                combined_filter,
                [
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_key),
                    )
                ],
            )
        if forced_collections:
            for name in forced_collections:
                if (
                    session_specific_filter is not None
                    and name == self.uploaded_collection
                ):
                    _append_collection(name, session_specific_filter)
                else:
                    _append_collection(name, combined_filter)
        else:
            if session_specific_filter is not None:
                _append_collection(self.uploaded_collection, session_specific_filter)

            if policy_mode:
                ordered_candidates: List[Tuple[Optional[str], Optional[models.Filter]]] = [
                    (self.static_policy_collection, combined_filter),
                    (self.uploaded_collection, combined_filter),
                    (self.primary_collection, combined_filter),
                    (self.learning_collection, combined_filter),
                ]
            else:
                ordered_candidates = [
                    (self.uploaded_collection, combined_filter),
                    (self.primary_collection, combined_filter),
                    (self.static_policy_collection, combined_filter),
                    (self.learning_collection, combined_filter),
                ]

            for name, filt in ordered_candidates:
                if session_specific_filter is not None and name == self.uploaded_collection:
                    continue
                _append_collection(name, filt)

        collections_to_query = tuple(base_collections)

        if collections_to_query and self.client is not None:
            _, vector = _ensure_query_vectors()
            batched_hits = self._search_collections_parallel(
                collections_to_query,
                vector,
                limit=prefetch,
            )
            for name, _ in collections_to_query:
                hits = batched_hits.get(name, [])
                for rank, hit in enumerate(hits[:prefetch], 1):
                    payload = dict(getattr(hit, "payload", {}) or {})
                    payload.setdefault("collection_name", name)
                    _register_candidate(
                        "qdrant",
                        payload,
                        getattr(hit, "id", payload.get("record_id")),
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

        pair_query = search_text if search_text else rewritten_query
        pairs = []
        for h in hits:
            payload = getattr(h, "payload", {}) or {}
            doc_text = self._prepare_reranker_text(payload)
            pairs.append((pair_query, doc_text))

        try:
            scores = self._batched_reranker_predict(pairs)
        except Exception:  # pragma: no cover - defensive logging only
            logger.warning(
                "Cross-encoder reranking failed; falling back to aggregated scores",
                exc_info=True,
            )
            scores = [0.0 for _ in pairs]

        if len(scores) < len(hits):
            deficit = len(hits) - len(scores)
            scores.extend([scores[-1] if scores else 0.0] * deficit)

        def _matches_focus_payload(payload: Dict[str, Any]) -> bool:
            if not focus_document_types:
                return False
            if not isinstance(payload, dict):
                return False
            text_bits: List[str] = []
            for key in ("document_type", "source_type", "source_category"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text_bits.append(value.lower())
            label_fields = payload.get("tags") or payload.get("labels")
            if isinstance(label_fields, (list, tuple, set)):
                for item in label_fields:
                    if isinstance(item, str) and item.strip():
                        text_bits.append(item.lower())
            if not text_bits:
                return False
            merged = " ".join(text_bits)
            return any(token in merged for token in focus_document_types)

        def _priority_bonus(hit: SimpleNamespace) -> float:
            payload = getattr(hit, "payload", {}) or {}
            collection = payload.get("collection_name", self.primary_collection)
            document_type = str(payload.get("document_type", "")).lower()
            source_type = str(payload.get("source_type", "")).lower()
            weights_map = getattr(self, "_preference_weights", {}) or {}
            collection_weights = weights_map.get("collection", {}) or {}
            document_weights = weights_map.get("document_type", {}) or {}

            collection_bonus = float(collection_weights.get(collection, 0.0))
            doc_bonus = float(document_weights.get(document_type, 0.0))

            is_policy_doc = (
                document_type == "policy"
                or "policy" in document_type
                or source_type == "policy"
                or "policy" in source_type
            )
            matches_focus = _matches_focus_payload(payload)

            if collection == self.static_policy_collection and not policy_mode:
                collection_bonus = min(collection_bonus, 0.08)
                doc_bonus = min(doc_bonus, 0.05)

            if policy_mode and (
                collection == self.static_policy_collection or is_policy_doc
            ):
                collection_bonus = max(collection_bonus, 0.14)
                doc_bonus = max(doc_bonus, 0.1)

            if focus_document_types:
                if matches_focus:
                    collection_bonus = max(collection_bonus, 0.12) + 0.12
                    if document_type:
                        doc_bonus = max(doc_bonus, 0.1)
                elif not policy_mode and (
                    collection == self.static_policy_collection or is_policy_doc
                ):
                    collection_bonus = min(collection_bonus, 0.02)
                    doc_bonus = 0.0

            return float(collection_bonus + doc_bonus)

        scored_hits: List[Tuple[SimpleNamespace, float, float]] = []
        rerank_weight = 1.35
        context_weight = 0.65
        for hit, score in zip(hits, scores):
            base_score = float(score)
            aggregated_score = float(getattr(hit, "aggregated", 0.0))
            combined = (
                base_score * rerank_weight
                + aggregated_score * context_weight
                + _priority_bonus(hit)
            )
            hit.rerank_score = base_score
            hit.combined_score = combined
            hit.context_score = aggregated_score
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

        def _matches_focus(hit_obj: SimpleNamespace) -> bool:
            return _matches_focus_payload(getattr(hit_obj, "payload", {}) or {})

        if focus_document_types and not any(
            _matches_focus(hit) for hit, _, _ in selected
        ):
            for candidate in scored_hits:
                if _matches_focus(candidate[0]):
                    if len(selected) < top_k:
                        selected.append(candidate)
                    else:
                        lowest_idx = min(
                            range(len(selected)), key=lambda i: selected[i][1]
                        )
                        selected[lowest_idx] = candidate
                    break

        deduped: List[Tuple[SimpleNamespace, float, float]] = []
        seen_keys: Set[Tuple[str, str, Any]] = set()
        for hit, combined, base in sorted(selected, key=lambda item: item[1], reverse=True):
            payload = getattr(hit, "payload", {}) or {}
            chunk_marker = payload.get("chunk_id")
            if chunk_marker is None:
                chunk_marker = payload.get("chunk_index")
            if chunk_marker is None:
                chunk_marker = payload.get("chunk_hash")
            key = (
                str(getattr(hit, "id", payload.get("record_id"))),
                _collection_name(hit),
                chunk_marker,
            )
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
    ) -> Tuple[str, List[models.FieldCondition], Set[str]]:
        normalised = re.sub(r"\s+", " ", query or "").strip()
        if not normalised:
            return "", [], set()

        normalised = re.sub(
            r"(?i)\bpo[-\s]?(\d{3,})\b",
            lambda match: f"purchase order {match.group(1)}",
            normalised,
        )

        conditions: List[models.FieldCondition] = []
        focus_tokens: Set[str] = set()

        def _ensure_condition(field: str, value: str) -> None:
            for condition in conditions:
                if getattr(condition, "key", None) == field:
                    match_obj = getattr(condition, "match", None)
                    if isinstance(match_obj, models.MatchValue) and match_obj.value == value:
                        return
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
            )

        def _mark_focus(*values: str) -> None:
            for value in values:
                if isinstance(value, str) and value.strip():
                    focus_tokens.add(value.strip().lower())

        lowered = normalised.lower()
        if "policy" in lowered:
            _ensure_condition("source_type", "Policy")
            _mark_focus("policy")
        if "invoice" in lowered:
            _ensure_condition("source_type", "Invoice")
            _mark_focus("invoice")
        if "purchase order" in lowered:
            _ensure_condition("source_type", "PO")
            _mark_focus("purchase order", "po")
        elif re.search(r"\bpo\b", lowered):
            _ensure_condition("source_type", "PO")
            _mark_focus("purchase order", "po")
        if "quote" in lowered:
            _ensure_condition("source_type", "Quote")
            _mark_focus("quote", "quotation")
        if "quotation" in lowered:
            _ensure_condition("source_type", "Quote")
            _mark_focus("quote", "quotation")
        if "contract" in lowered:
            _ensure_condition("source_type", "Contract")
            _mark_focus("contract", "agreement")
        if "agreement" in lowered:
            _ensure_condition("source_type", "Contract")
            _mark_focus("contract", "agreement")

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

        return normalised, conditions, focus_tokens

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
        """Detect policy-focused prompts without over-weighting conversation memory."""

        query_text = " ".join(
            part.strip().lower() for part in (query or "",) if part and part.strip()
        )
        if not query_text:
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
        if any(token in query_text for token in policy_tokens):
            return True

        if not hint_text:
            return False

        history_text = hint_text.strip().lower()
        if not history_text:
            return False

        reinforcing_tokens = (
            "policy exception",
            "policy breach",
            "approval matrix",
            "delegation of authority",
            "compliance escalation",
        )
        soft_cues = ("approval", "compliance", "policy")
        return any(token in history_text for token in reinforcing_tokens) and any(
            cue in query_text for cue in soft_cues
        )

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

    def _separate_session_scope(
        self,
        active_filter: Optional[models.Filter],
        session_id: Optional[str],
    ) -> Tuple[Optional[models.Filter], Optional[str]]:
        """Detach session-specific constraints from the shared filter space."""

        session_token = self._normalise_session_token(session_id)
        if active_filter is None:
            return None, session_token

        def _extract_from_condition(condition: Any) -> Tuple[bool, Optional[str]]:
            if isinstance(condition, models.FieldCondition):
                key = getattr(condition, "key", None)
                if key and str(key).strip().lower() == "session_id":
                    match_obj = getattr(condition, "match", None)
                    candidate: Optional[str] = None
                    if isinstance(match_obj, models.MatchValue):
                        candidate = self._normalise_session_token(match_obj.value)
                    elif isinstance(match_obj, models.MatchAny):
                        for item in getattr(match_obj, "any", []) or []:
                            candidate = self._normalise_session_token(item)
                            if candidate:
                                break
                    return True, candidate
            if isinstance(condition, dict):
                key = str(condition.get("key") or condition.get("field") or "").strip().lower()
                if key == "session_id":
                    value = condition.get("value")
                    if value is None:
                        match_payload = condition.get("match")
                        if isinstance(match_payload, dict):
                            value = match_payload.get("value") or match_payload.get("any")
                    candidate = None
                    if isinstance(value, list):
                        for item in value:
                            candidate = self._normalise_session_token(item)
                            if candidate:
                                break
                    else:
                        candidate = self._normalise_session_token(value)
                    return True, candidate
            return False, None

        def _process_conditions(items: Optional[Sequence[Any]]) -> List[Any]:
            nonlocal session_token
            processed: List[Any] = []
            for condition in list(items or []):
                should_remove, found = _extract_from_condition(condition)
                if should_remove:
                    if not session_token and found:
                        session_token = found
                    elif found:
                        session_token = session_token or found
                    continue
                processed.append(condition)
            return processed

        must_conditions = _process_conditions(getattr(active_filter, "must", None))
        must_not_conditions = _process_conditions(getattr(active_filter, "must_not", None))
        should_conditions = _process_conditions(getattr(active_filter, "should", None))

        if not (must_conditions or must_not_conditions or should_conditions):
            return None, session_token

        cleaned_filter = models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
            should=should_conditions or None,
        )
        return cleaned_filter, session_token

    @staticmethod
    def _normalise_session_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

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

        filtered = {
            key: value
            for key, value in (payload or {}).items()
            if key not in DISALLOWED_METADATA_KEYS
        }
        try:
            json.dumps(filtered)
            return filtered
        except (TypeError, ValueError):
            return _normalise(filtered)

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
