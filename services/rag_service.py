import os
import uuid
from typing import List, Dict, Optional

import torch
from qdrant_client import models


# Ensure GPU is enabled when available
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class RAGService:
    """Handles embedding upserts and retrieval for RAG workflows."""

    def __init__(self, agent_nick, cross_encoder_cls=None):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.client = agent_nick.qdrant_client
        self.embedder = agent_nick.embedding_model
        if cross_encoder_cls is None:
            from sentence_transformers import CrossEncoder
            cross_encoder_cls = CrossEncoder
        model_name = getattr(self.settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranker = cross_encoder_cls(model_name, device=self.agent_nick.device)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into whitespace-aware chunks with overlap."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        step = max_chars - overlap if max_chars > overlap else max_chars
        return [cleaned[i : i + max_chars] for i in range(0, len(cleaned), step)]

    def upsert_texts(self, texts: List[str], metadata: Optional[Dict] = None):
        """Encode and upsert texts into Qdrant."""
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
                point_id = f"{record_id}_{idx}"
                points.append(
                    models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
                )
        if points:
            self.client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=points,
                wait=True,
            )

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
        query_vec = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()
        search_params = models.SearchParams(hnsw_ef=256, exact=False)
        hits = self.client.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_vec,
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
                query_vector=query_vec,
                query_filter=filters,
                limit=top_k * 5,
                with_payload=True,
                with_vectors=False,
                search_params=exact_params,
            )
        if not hits:
            limit=top_k * 3,
            with_payload=True,
            with_vectors=False,
            search_params=models.SearchParams(hnsw_ef=256, exact=False),

        if not hits:
            return []
        pairs = [
            (query, h.payload.get("content", h.payload.get("summary", "")))
            for h in hits
        ]
        scores = self._reranker.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]
