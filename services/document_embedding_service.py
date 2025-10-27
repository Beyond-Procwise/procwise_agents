"""Utilities for embedding uploaded documents into Qdrant collections."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from qdrant_client import models

logger = logging.getLogger(__name__)

try:  # Optional heavy dependency
    from unstructured.partition.auto import partition  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    partition = None  # type: ignore


@dataclass
class EmbeddedDocument:
    """Structured response for embedded document uploads."""

    document_id: str
    collection: str
    chunk_count: int
    metadata: Dict[str, Any]


class DocumentEmbeddingService:
    """Create and query embeddings for ad-hoc uploaded documents."""

    def __init__(
        self,
        agent_nick,
        *,
        collection_name: str = "uploaded_documents",
        chat_completion_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.agent_nick = agent_nick
        self.collection_name = collection_name
        self.settings = getattr(agent_nick, "settings", None)
        self.client = getattr(agent_nick, "qdrant_client", None)
        self.embedder = getattr(agent_nick, "embedding_model", None)
        self._chat_completion_fn = chat_completion_fn
        if self.client is None or self.embedder is None:
            raise ValueError("AgentNick must provide Qdrant client and embedding model")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_document(
        self,
        *,
        filename: str,
        file_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddedDocument:
        """Extract text, chunk, embed, and persist a document."""

        if not file_bytes:
            raise ValueError("Uploaded document is empty")

        document_id = str(uuid.uuid4())
        upload_ts = datetime.utcnow().isoformat(timespec="seconds")
        metadata_payload = {
            "document_id": document_id,
            "filename": filename,
            "file_extension": Path(filename).suffix.lower() or "",
            "uploaded_at": upload_ts,
        }
        if metadata:
            metadata_payload.update(metadata)

        text = self._extract_text(file_bytes=file_bytes, filename=filename)
        if not text.strip():
            raise ValueError("Unable to extract textual content from document")

        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Document extraction produced no embeddable chunks")

        vectors = self._encode_chunks(chunks)
        if not vectors:
            raise ValueError("Embedding model returned no vectors")

        vector_size = len(vectors[0])
        self._ensure_collection(vector_size)

        points: List[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = dict(metadata_payload)
            payload.update({"chunk_id": idx, "content": chunk})
            point_id = str(uuid.uuid5(uuid.UUID(document_id), str(idx)))
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return EmbeddedDocument(
            document_id=document_id,
            collection=self.collection_name,
            chunk_count=len(points),
            metadata=metadata_payload,
        )

    def query(
        self,
        query: str,
        *,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate a grounded answer."""

        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        self._ensure_collection_ready()

        query_vector = self._encode_query(query)
        if query_vector is None:
            raise ValueError("Unable to generate embedding for query")

        conditions: List[models.FieldCondition] = []
        if document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=max(1, min(top_k, 10)),
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter,
        )

        contexts: List[Dict[str, Any]] = []
        for hit in search_results:
            payload = getattr(hit, "payload", {}) or {}
            contexts.append(
                {
                    "document_id": payload.get("document_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "content": payload.get("content", ""),
                    "score": getattr(hit, "score", None),
                    "filename": payload.get("filename"),
                    "uploaded_at": payload.get("uploaded_at"),
                }
            )

        answer = self._generate_answer(query, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
            "collection": self.collection_name,
        }

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _extract_text(self, *, file_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md", ""}:
            try:
                return file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                logger.debug("Failed UTF-8 decode for %s", filename, exc_info=True)

        if partition is not None:
            try:
                with NamedTemporaryFile(suffix=suffix or ".bin", delete=True) as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    elements = partition(filename=tmp.name)
                    text_parts = [
                        getattr(element, "text", "")
                        for element in elements
                        if getattr(element, "text", None)
                    ]
                    combined = "\n".join(text_parts)
                    if combined.strip():
                        return combined
            except Exception:
                logger.exception("unstructured partition failed for %s", filename)

        # Fallback: attempt UTF-8 decode regardless of file type
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            logger.warning("Failed to decode document %s", filename)
            return ""

    def _chunk_text(self, text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        if self.settings is not None:
            max_chars = max(getattr(self.settings, "rag_chunk_chars", max_chars), 200)
            configured_overlap = getattr(self.settings, "rag_chunk_overlap", overlap)
            overlap = max(0, min(configured_overlap, max_chars - 1))
        step = max_chars - overlap if max_chars > overlap else max_chars
        return [cleaned[i : i + max_chars] for i in range(0, len(cleaned), step)]

    def _encode_chunks(self, chunks: Iterable[str]) -> List[List[float]]:
        chunk_list = [chunk for chunk in chunks if chunk.strip()]
        if not chunk_list:
            return []
        vectors = self.embedder.encode(
            chunk_list,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(vectors, np.ndarray):
            return vectors.astype("float32").tolist()
        return [list(map(float, vec)) for vec in vectors]

    def _encode_query(self, query: str) -> Optional[List[float]]:
        try:
            vector = self.embedder.encode(
                query,
                normalize_embeddings=True,
            )
        except TypeError:
            vector = self.embedder.encode(query)  # type: ignore[call-arg]
        except Exception:
            logger.exception("Failed to embed query for uploaded documents")
            return None
        if vector is None:
            return None
        if isinstance(vector, np.ndarray):
            return vector.astype("float32").tolist()
        if hasattr(vector, "tolist"):
            return [float(v) for v in vector.tolist()]
        return [float(v) for v in vector]

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------
    def _ensure_collection_ready(self) -> None:
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            vector_size = self._infer_vector_size()
            if vector_size is None:
                raise ValueError("Unable to determine embedding vector size")
            self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            payload_schema = getattr(collection_info, "payload_schema", {}) or {}
            self._ensure_payload_indexes(payload_schema)
            return
        except Exception:
            logger.info(
                "Creating Qdrant collection '%s' for uploaded documents",
                self.collection_name,
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._ensure_payload_indexes({})

    def _ensure_payload_indexes(self, schema: Dict[str, Any]) -> None:
        indexed_fields = set(schema.keys()) if schema else set()
        required = {
            "document_id",
            "filename",
            "file_extension",
            "uploaded_at",
        }
        for field in required:
            if field in indexed_fields:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception:
                logger.debug("Failed to create payload index for %s", field, exc_info=True)

    def _infer_vector_size(self) -> Optional[int]:
        getter = getattr(self.embedder, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                size = int(getter())
                if size > 0:
                    return size
            except Exception:
                logger.debug("Embedding model dimension lookup failed", exc_info=True)
        try:
            probe = self.embedder.encode("uploaded document warmup", normalize_embeddings=True)
        except TypeError:
            probe = self.embedder.encode("uploaded document warmup")  # type: ignore[call-arg]
        except Exception:
            logger.exception("Failed to probe embedding vector size")
            return None
        if probe is None:
            return None
        if isinstance(probe, np.ndarray):
            return int(probe.size)
        if hasattr(probe, "__len__"):
            return int(len(probe))
        return None

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------
    def _generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for item in contexts:
            content = item.get("content", "")
            if content:
                context_blocks.append(
                    f"Document {item.get('document_id')} chunk {item.get('chunk_id')}:\n{content}"
                )
        if not context_blocks:
            return "No relevant content found for the provided query."

        prompt = (
            "Answer the user's question using ONLY the provided document excerpts. "
            "Cite specific details from the excerpts when relevant."
        )
        context_text = "\n\n".join(context_blocks)
        user_content = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"

        chat_fn = self._chat_completion_fn
        if chat_fn is None:
            try:
                import ollama  # type: ignore

                def _default_chat(**kwargs):
                    return ollama.chat(**kwargs)

                chat_fn = _default_chat
            except Exception:  # pragma: no cover - Ollama optional
                chat_fn = None

        if chat_fn is not None:
            try:
                model_name = getattr(self.settings, "rag_model", None)
                if not model_name:
                    model_name = getattr(self.settings, "extraction_model", "")
                options = {}
                ollama_opts = getattr(self.agent_nick, "ollama_options", None)
                if callable(ollama_opts):
                    options = ollama_opts()
                response = chat_fn(
                    model=model_name or "llama3",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_content},
                    ],
                    options=options,
                )
                message = response.get("message", {}) if isinstance(response, dict) else {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception:
                logger.exception("LLM generation failed for uploaded document query")

        # Fallback deterministic summary
        return context_blocks[0]
