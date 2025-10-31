import importlib
import importlib.util
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
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

        target_collection = collection_name or self.primary_collection
        points: List[models.PointStruct] = []
        texts_for_embedding: List[str] = []
        payloads_for_storage: List[Dict[str, Any]] = []
        target_collection: Optional[str] = None

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

            if target_collection is None:
                target_collection = self._extract_source_collection(base_payload)

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
                chunk_payload.setdefault("collection_name", target_collection)
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
            target_collection
            or getattr(self.settings, "qdrant_collection_name", None)
        )

        if points and collection_name:
            self.client.upsert(
                collection_name=target_collection,
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
    ):
        """Retrieve and rerank documents for the given query."""
        candidates: Dict[str, SimpleNamespace] = {}

        query_matrix: Optional[np.ndarray] = None
        query_vector: Optional[List[float]] = None

        def _ensure_query_vectors() -> Tuple[np.ndarray, List[float]]:
            nonlocal query_matrix, query_vector
            if query_matrix is not None and query_vector is not None:
                return query_matrix, query_vector

            encoded = self.embedder.encode(query, normalize_embeddings=True)
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

        # --- FAISS semantic search ---
        if self._faiss_index is not None and self._doc_vectors:
            matrix, _ = _ensure_query_vectors()
            scores, ids = self._faiss_index.search(matrix, top_k * 5)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                key = f"{doc.get('collection_name', self.primary_collection)}:{doc['id']}"
                existing = candidates.get(key)
                if existing is None or float(score) > existing.score:
                    candidates[key] = SimpleNamespace(
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
                key = f"{doc.get('collection_name', self.primary_collection)}:{doc['id']}"
                existing = candidates.get(key)
                if existing is None or score > existing.score:
                    candidates[key] = SimpleNamespace(
                        id=doc["id"], payload=doc, score=float(score)
                    )

        # --- Fallback to Qdrant if local indexes empty ---
        if not candidates and self.client is not None:
            _, vector = _ensure_query_vectors()
            search_params = models.SearchParams(hnsw_ef=256, exact=False)

            def _search_collection(name: str) -> List[SimpleNamespace]:
                if not name:
                    return []

                try:
                    raw_hits = self.client.search(
                        collection_name=name,
                        query_vector=vector,
                        query_filter=filters,
                        limit=top_k * 5,
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
                            query_filter=filters,
                            limit=top_k * 5,
                            with_payload=True,
                            with_vectors=False,
                            search_params=models.SearchParams(hnsw_ef=256, exact=True),
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

            for collection in {
                self.primary_collection,
                self.uploaded_collection,
            }:
                for hit in _search_collection(collection):
                    key = f"{hit.payload.get('collection_name', collection)}:{hit.id}"
                    existing = candidates.get(key)
                    if existing is None or hit.score > existing.score:
                        candidates[key] = hit

        if not candidates:
            return []

        hits = list(candidates.values())

        # --- Re-rank with cross-encoder ---
        pairs = [
            (
                query,
                h.payload.get(
                    "text_summary",
                    h.payload.get(
                        "content",
                        h.payload.get("summary", ""),
                    ),
                ),
            )
            for h in hits
        ]
        scores = self._reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]

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
