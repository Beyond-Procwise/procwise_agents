"""Utilities for embedding uploaded documents into Qdrant collections."""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import models

from services.document_extractor import LayoutAwareParser
from services.semantic_chunker import SemanticChunker
from services.semantic_cache import SemanticCacheManager

DISALLOWED_METADATA_KEYS: set[str] = set()

logger = logging.getLogger(__name__)

try:  # Optional heavy dependency
    from unstructured.partition.auto import partition  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    partition = None  # type: ignore

try:  # Optional PDF dependency for deterministic fallback extraction
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    pdfplumber = None  # type: ignore

try:  # Optional PyMuPDF dependency for additional PDF fallback
    import fitz  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency fallback
    fitz = None  # type: ignore


@dataclass
class EmbeddedDocument:
    """Structured response for embedded document uploads."""

    document_id: str
    collection: str
    chunk_count: int
    metadata: Dict[str, Any]


@dataclass
class ExtractedContent:
    """Container for raw text extracted from uploaded files."""

    text: str
    method: str
    metadata: Dict[str, Any]


@dataclass
class DocumentChunk:
    """Represents a semantically meaningful chunk of a document."""

    content: str
    metadata: Dict[str, Any]


class DocumentEmbeddingService:
    """Create and query embeddings for ad-hoc uploaded documents."""

    def __init__(
        self,
        agent_nick,
        *,
        collection_name: str = "uploaded_documents",
        rag_service_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.agent_nick = agent_nick
        self.collection_name = collection_name
        self.settings = getattr(agent_nick, "settings", None)
        self.client = getattr(agent_nick, "qdrant_client", None)
        self.embedder = getattr(agent_nick, "embedding_model", None)
        self._rag_service_factory = rag_service_factory
        self._rag_service: Optional[Any] = None
        self._rag_service_failed = False
        self._layout_parser = LayoutAwareParser()
        self._chunker = SemanticChunker(settings=self.settings)
        self._semantic_cache = (
            SemanticCacheManager(self.settings, namespace="document_embeddings")
            if self.settings is not None
            else SemanticCacheManager(None, namespace="document_embeddings")
        )
        self._embedding_model_name = self._resolve_embedding_model_name()
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
        base_name = Path(filename).name
        metadata_payload = {
            "document_id": document_id,
            "filename": filename,
            "file_extension": Path(filename).suffix.lower() or "",
            "doc_name": base_name,
            "uploaded_at": upload_ts,
        }
        if metadata:
            metadata_payload.update(metadata)
        for disallowed in DISALLOWED_METADATA_KEYS:
            metadata_payload.pop(disallowed, None)

        extracted = self._extract_text(file_bytes=file_bytes, filename=filename)
        text = extracted.text
        if not text.strip():
            raise ValueError("Unable to extract textual content from document")

        metadata_payload["extraction_method"] = extracted.method
        metadata_payload.update(extracted.metadata)
        for disallowed in DISALLOWED_METADATA_KEYS:
            metadata_payload.pop(disallowed, None)

        document_type = metadata_payload.get("document_type")
        chunks = self._build_document_chunks(
            text,
            filename=filename,
            document_type_hint=document_type,
            extraction_metadata=extracted.metadata,
            base_metadata=metadata_payload,
        )
        if not chunks:
            raise ValueError("Document extraction produced no embeddable chunks")

        first_chunk_metadata = chunks[0].metadata if chunks else {}
        metadata_payload.setdefault(
            "document_type", first_chunk_metadata.get("document_type")
        )
        metadata_payload.setdefault(
            "source_type", first_chunk_metadata.get("source_type")
        )
        metadata_payload.setdefault("title", first_chunk_metadata.get("title"))
        vectors = self._encode_chunks(chunks)
        if not vectors:
            raise ValueError("Embedding model returned no vectors")

        vector_size = len(vectors[0])
        self._ensure_collection(vector_size)
        self._remove_disallowed_payload_fields(self.collection_name)

        chunk_vector_pairs = list(zip(chunks, vectors))
        points: List[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(chunk_vector_pairs):
            payload = dict(metadata_payload)
            payload.update(
                {
                    key: value
                    for key, value in chunk.metadata.items()
                    if key not in DISALLOWED_METADATA_KEYS
                }
            )
            payload.update(
                {
                    "chunk_id": idx,
                    "chunk_index": idx,
                    "content": chunk.content,
                    "embedding_model": self._embedding_model_name,
                }
            )
            point_id = str(uuid.uuid5(uuid.UUID(document_id), str(idx)))
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        self._propagate_to_rag([chunk for chunk, _ in chunk_vector_pairs], metadata_payload)

        return EmbeddedDocument(
            document_id=document_id,
            collection=self.collection_name,
            chunk_count=len(points),
            metadata=metadata_payload,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _extract_text(self, *, file_bytes: bytes, filename: str) -> ExtractedContent:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md", ""}:
            try:
                return ExtractedContent(
                    text=file_bytes.decode("utf-8", errors="ignore"),
                    method="direct_decode",
                    metadata={"content_type": "text"},
                )
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
                        element_types = {
                            getattr(element, "category", getattr(element, "type", "unknown"))
                            for element in elements
                        }
                        return ExtractedContent(
                            text=combined,
                            method="unstructured_partition",
                            metadata={
                                "content_type": self._detect_content_category(filename),
                                "detected_elements": sorted(element_types),
                            },
                        )
            except Exception:
                logger.exception("unstructured partition failed for %s", filename)

        if suffix == ".pdf" and pdfplumber is not None:
            try:
                with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                    combined = "\n".join(part for part in pages if part)
                    if combined.strip():
                        metadata = {
                            "content_type": "pdf",
                            "pdf_page_count": len(pdf.pages),
                        }
                        return ExtractedContent(
                            text=combined,
                            method="pdfplumber_fallback",
                            metadata=metadata,
                        )
            except Exception:
                logger.exception("pdfplumber fallback failed for %s", filename)

        if suffix == ".pdf" and fitz is not None:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    pages = [page.get_text("text") or "" for page in doc]
                    combined = "\n".join(part for part in pages if part)
                    if combined.strip():
                        metadata = {
                            "content_type": "pdf",
                            "pdf_page_count": doc.page_count,
                            "pymupdf_extraction": True,
                        }
                        return ExtractedContent(
                            text=combined,
                            method="pymupdf_fallback",
                            metadata=metadata,
                        )
            except Exception:
                logger.exception("pymupdf fallback failed for %s", filename)

        # Fallback: attempt UTF-8 decode regardless of file type
        try:
            return ExtractedContent(
                text=file_bytes.decode("utf-8", errors="ignore"),
                method="raw_decode",
                metadata={"content_type": self._detect_content_category(filename)},
            )
        except Exception:
            logger.warning("Failed to decode document %s", filename)
            return ExtractedContent(text="", method="raw_decode", metadata={})

    def _detect_content_category(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".pdf"}:
            return "pdf"
        if suffix in {".doc", ".docx"}:
            return "docx"
        if suffix in {".ppt", ".pptx"}:
            return "pptx"
        if suffix in {".xls", ".xlsx"}:
            return "spreadsheet"
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            return "image"
        return "text"

    def _build_document_chunks(
        self,
        text: str,
        *,
        filename: str,
        document_type_hint: Optional[str],
        extraction_metadata: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        structured = self._layout_parser.from_text(
            text, scanned=self._looks_like_scanned(text, extraction_metadata)
        )
        doc_type = self._infer_document_type(text, document_type_hint)
        chunk_metadata_seed = {
            "doc_name": base_metadata.get("doc_name", Path(filename).stem),
            "document_type": doc_type,
            "title": base_metadata.get("title")
            or extraction_metadata.get("title")
            or Path(filename).stem,
            "source_type": base_metadata.get("source_type")
            or self._derive_source_type(doc_type, base_metadata.get("source_type")),
        }
        for disallowed in DISALLOWED_METADATA_KEYS:
            chunk_metadata_seed.pop(disallowed, None)

        chunks = self._chunker.build_from_structured(
            structured,
            document_type=doc_type,
            base_metadata=chunk_metadata_seed,
            title_hint=chunk_metadata_seed["title"],
            default_section="document_overview",
        )

        if not chunks:
            return []

        return [DocumentChunk(content=chunk.content, metadata=chunk.metadata) for chunk in chunks]

    def _looks_like_scanned(self, text: str, metadata: Dict[str, Any]) -> bool:
        if metadata.get("content_type") == "image":
            return True
        characters = [char for char in text if char.strip()]
        if not characters:
            return False
        non_ascii = sum(1 for char in characters if ord(char) > 126)
        ratio = non_ascii / len(characters)
        return ratio > 0.25

    def _encode_chunks(self, chunks: Sequence[DocumentChunk]) -> List[List[float]]:
        if not chunks:
            return []

        vectors: Dict[int, List[float]] = {}
        encode_batches: List[str] = []
        encode_indices: List[int] = []

        for idx, chunk in enumerate(chunks):
            text = chunk.content.strip()
            if not text:
                continue
            cached = self._semantic_cache.get_embedding(text, self._embedding_model_name)
            if cached is not None:
                vectors[idx] = [float(value) for value in cached]
                continue
            encode_batches.append(text)
            encode_indices.append(idx)

        if encode_batches:
            raw_vectors = self.embedder.encode(
                encode_batches,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if isinstance(raw_vectors, np.ndarray):
                encoded = raw_vectors.astype("float32").tolist()
            else:
                encoded = [list(map(float, vec)) for vec in raw_vectors]

            for idx, vector in zip(encode_indices, encoded):
                vectors[idx] = vector
            for text, vector in zip(encode_batches, encoded):
                self._semantic_cache.set_embedding(text, vector, self._embedding_model_name)

        ordered: List[List[float]] = []
        for idx in range(len(chunks)):
            vector = vectors.get(idx)
            if vector is not None:
                ordered.append(vector)

        return ordered

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
            "doc_name",
            "document_type",
            "content_type",
            "section",
            "section_path",
            "source_type",
            "title",
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

    def _remove_disallowed_payload_fields(self, collection_name: str) -> None:
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
                "Failed to purge disallowed metadata keys from %s", collection_name,
                exc_info=True,
            )

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
    # RAG propagation helpers
    # ------------------------------------------------------------------
    def _resolve_rag_service(self) -> Optional[Any]:
        if self._rag_service_failed:
            return None
        if self._rag_service is not None:
            return self._rag_service
        try:
            if self._rag_service_factory is not None:
                self._rag_service = self._rag_service_factory(self.agent_nick)
            else:
                from services.rag_service import RAGService  # type: ignore

                self._rag_service = RAGService(self.agent_nick)
        except Exception:
            logger.exception("Failed to initialise RAGService for uploaded documents")
            self._rag_service_failed = True
            return None
        return self._rag_service

    def _propagate_to_rag(
        self, chunks: Sequence[DocumentChunk], metadata: Dict[str, Any]
    ) -> None:
        if not chunks:
            return
        rag_service = self._resolve_rag_service()
        if rag_service is None:
            return

        payloads: List[Dict[str, Any]] = []
        document_id = metadata.get("document_id")
        for idx, chunk in enumerate(chunks):
            if not chunk.content.strip():
                continue
            payload = dict(metadata)
            for key in DISALLOWED_METADATA_KEYS:
                payload.pop(key, None)
            payload["record_id"] = document_id or metadata.get("record_id") or str(uuid.uuid4())
            payload["chunk_id"] = idx
            payload.setdefault("document_type", "uploaded_document")
            payload["source_collection"] = self.collection_name
            payload.update(chunk.metadata)
            payload["content"] = chunk.content
            payloads.append(payload)

        if not payloads:
            return

        try:
            rag_service.upsert_payloads(payloads, text_representation_key="content")
        except Exception:
            logger.exception(
                "Failed to propagate uploaded document chunks to primary RAG store"
            )

    def _derive_source_type(
        self, document_type: Optional[str], explicit: Optional[str]
    ) -> str:
        if explicit:
            return str(explicit)
        mapping = {
            "invoice": "Invoice",
            "purchase_order": "PO",
            "purchase order": "PO",
            "po": "PO",
            "contract": "Contract",
            "quote": "Quote",
            "policy": "Policy",
        }
        key = (document_type or "uploaded_document").replace("_", " ").lower()
        return mapping.get(key, "Upload")

    def _infer_document_type(
        self, text: str, document_type_hint: Optional[str] = None
    ) -> str:
        if document_type_hint:
            return document_type_hint

        lowered = text.lower()
        keyword_map: Dict[str, Tuple[Tuple[str, ...], ...]] = {
            "Invoice": (
                ("invoice",),
                ("amount", "due"),
                ("invoice", "number"),
                ("total", "tax"),
            ),
            "Purchase_Order": (
                ("purchase", "order"),
                ("po", "number"),
                ("ship", "to"),
            ),
            "Contract": (
                ("agreement",),
                ("contract",),
                ("term", "length"),
            ),
            "Quote": (
                ("quote",),
                ("quotation",),
                ("valid", "until"),
            ),
        }

        best_type = "Contract"
        best_score = -math.inf
        for doc_type, patterns in keyword_map.items():
            score = 0.0
            for pattern in patterns:
                if all(token in lowered for token in pattern):
                    score += len(pattern) * 1.5
            if score > best_score:
                best_type = doc_type
                best_score = score

        return best_type

    def _resolve_embedding_model_name(self) -> str:
        if self.settings is not None:
            configured = getattr(self.settings, "embedding_model", None)
            if isinstance(configured, str) and configured:
                return configured
        candidate = getattr(self.embedder, "model_name", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        return self.embedder.__class__.__name__
