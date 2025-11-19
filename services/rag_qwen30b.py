import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from config.settings import settings
from services.lmstudio_client import get_lmstudio_client
from utils.gpu import configure_gpu, load_cross_encoder

logger = logging.getLogger(__name__)

configure_gpu()

# Defaults and env-driven configuration
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
RAG_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "8"))
RAG_PER_DOC_CAP = int(os.getenv("RAG_PER_DOC_CAP", "2"))
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "12000"))
COMPRESS_PER_CHUNK_CHARS = int(os.getenv("COMPRESS_PER_CHUNK_CHARS", "1200"))
RERANK_MODEL = os.getenv("RERANK_MODEL", os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "16"))
RERANK_FP16 = bool(int(os.getenv("RERANK_FP16", "1")))
LMSTUDIO_MODEL = os.getenv(
    "LMSTUDIO_MODEL",
    getattr(settings, "lmstudio_chat_model", "microsoft/phi-4-reasoning-plus"),
)
RAG_EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "16"))

# Hallucination refusal exact sentence
REFUSAL_SENTENCE = "I don't have enough information in the provided documents."

# similarity threshold for semantic dedupe
SEMANTIC_DEDUPE_THRESHOLD = 0.92

# MMR lambda
MMR_LAMBDA = 0.75

# helper types
Chunk = Dict[str, Any]


def _norm_text_for_hash(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode to ascii, lowercase
    nf = unicodedata.normalize("NFKD", text)
    ascii_text = nf.encode("ASCII", "ignore").decode("ASCII")
    # lowercase
    s = ascii_text.lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # remove punctuation except alnum and spaces
    s = re.sub(r"[^0-9a-z ]+", "", s)
    return s


def _hash_text(text: str) -> str:
    norm = _norm_text_for_hash(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _hashed_token_vector(text: str, dims: int = 64) -> Optional[np.ndarray]:
    """Fallback lightweight vector using hashing trick when dense embedding missing."""
    if not text:
        return None
    tokens = re.findall(r"[0-9a-z]+", text.lower())
    if not tokens:
        return None
    vec = np.zeros(dims, dtype=np.float32)
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8, person=b"rag_hash").digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1 if (digest[4] & 1) == 0 else -1
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RAGQwen30b:
    """High-precision RAG pipeline that enforces strict hallucination control and
    uses qwen3-30b-procwise via Ollama for generation.

    The class expects an `agent_nick` which exposes:
      - settings (config object)
      - qdrant_client
      - embedding_model (with .encode)
      - device (optional)

    The reranker uses a cross-encoder which is initialised via utils.gpu.load_cross_encoder.
    """

    def __init__(self, agent_nick, cross_encoder_cls=None):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.qdrant = agent_nick.qdrant_client
        self.embedder = agent_nick.embedding_model
        self.device = getattr(agent_nick, "device", None)
        self.primary_collection = getattr(self.settings, "qdrant_collection_name", "procwise_document_embeddings")
        self.uploaded_collection = getattr(self.settings, "uploaded_documents_collection_name", "uploaded_documents")
        # load reranker
        if cross_encoder_cls is None:
            # defer import to preserve tests flexibility
            from sentence_transformers import CrossEncoder

            cross_encoder_cls = CrossEncoder
        self._reranker = self._init_reranker(cross_encoder_cls)
        # Telemetry
        self.telemetry = {
            "dense_count": 0,
            "reranked_count": 0,
            "deduped_count": 0,
            "capped_count": 0,
            "packed_chars": 0,
        }
        self._embed_batch_size = max(int(RAG_EMBED_BATCH), 1)

    def _init_reranker(self, cross_encoder_cls):
        """Load cross-encoder with an 8s circuit breaker before falling back to CPU."""
        target_device = self.device or "cpu"
        if target_device == "cpu":
            return self._load_reranker_once(cross_encoder_cls, "cpu")
        # try device twice if each attempt exceeds 8s
        for attempt in range(2):
            reranker, duration = self._load_reranker_once(cross_encoder_cls, target_device, return_duration=True)
            if reranker is None:
                continue
            if duration > 8.0 and attempt == 0:
                logger.warning("Cross-encoder init on %s took %.2fs (>8s); retrying once", target_device, duration)
                continue
            return reranker
        # fallback to CPU
        logger.warning("Falling back to CPU for cross-encoder after slow/failed attempts on %s", target_device)
        reranker = self._load_reranker_once(cross_encoder_cls, "cpu")
        if reranker is None:
            logger.exception("Cross-encoder init failed permanently; pipeline will degrade")
        return reranker

    def _load_reranker_once(self, cross_encoder_cls, device, return_duration: bool = False):
        start = time.time()
        try:
            reranker = load_cross_encoder(RERANK_MODEL, cross_encoder_cls, device)
        except Exception:
            logger.exception("Failed to load cross-encoder on %s", device)
            reranker = None
        duration = time.time() - start
        if return_duration:
            return reranker, duration
        return reranker

    # -------------------- Retrieval & rerank --------------------------------
    def retrieve_and_rerank(self, query: str, top_k: int = RAG_TOP_K, rerank_top_n: int = RAG_RERANK_TOP_N, vector_name: Optional[str] = None, collections: Optional[List[str]] = None) -> List[Chunk]:
        """Retrieve candidates from Qdrant and rerank using cross-encoder. Returns list of chunks with added fields:
        - rerank_score
        - embedding (if available or computed)

        If `collections` is provided, search each collection and merge hits (useful for uploaded docs).
        """
        # Encode query for dense search
        try:
            q_vec = self.embedder.encode(query, normalize_embeddings=True)
        except Exception:
            # fallback: single-dim dummy
            q_vec = [0.0]

        search_collections = collections or [self.primary_collection]
        named_vector = vector_name or QDRANT_VECTOR_NAME
        all_hits = []
        for coll in search_collections:
            try:
                hits = self.qdrant.search(
                    collection_name=coll,
                    query_vector=q_vec,
                    limit=top_k,
                    vector_name=named_vector,
                )
            except Exception:
                logger.exception("Qdrant search failed for collection %s; continuing", coll)
                hits = []
            for h in hits or []:
                # tag collection name into payload so downstream code can prioritise
                try:
                    if getattr(h, 'payload', None) is not None:
                        h.payload = dict(getattr(h, 'payload') or {})
                        h.payload.setdefault('collection_name', coll)
                except Exception:
                    pass
            all_hits.extend(hits or [])

        dense = []
        for h in all_hits or []:
            payload = getattr(h, "payload", {}) if h is not None else {}
            content = payload.get("content") or payload.get("text") or payload.get("summary") or ""
            chunk = {
                "id": getattr(h, "id", None) or payload.get("record_id") or None,
                "doc_id": payload.get("record_id") or payload.get("doc_id") or payload.get("document_id") or payload.get("documentid") or payload.get("filename") or payload.get("file_name") or None,
                "content": content,
                "payload": payload,
                "score": getattr(h, "score", None) or 0.0,
                "vector": payload.get("embedding") if isinstance(payload.get("embedding"), (list, tuple, np.ndarray)) else None,
            }
            dense.append(chunk)
        self.telemetry["dense_count"] = len(dense)

        # Rerank using cross-encoder pairs (query, chunk_text)
        if not dense:
            return []
        texts = [c["content"] for c in dense]
        pairs = [[query, t] for t in texts]
        rerank_scores: List[float]
        if self._reranker is None:
            # fallback: use qdrant score
            rerank_scores = [float(c.get("score") or 0.0) for c in dense]
        else:
            # batch predict
            rerank_scores = []
            try:
                for i in range(0, len(pairs), RERANK_BATCH):
                    batch = pairs[i : i + RERANK_BATCH]
                    scores = list(self._reranker.predict(batch))
                    rerank_scores.extend([float(s) for s in scores])
            except Exception:
                logger.exception("Reranker failed during prediction; falling back to qdrant scores")
                rerank_scores = [float(c.get("score") or 0.0) for c in dense]
        for idx, chunk in enumerate(dense):
            chunk["rerank_score"] = rerank_scores[idx] if idx < len(rerank_scores) else float(chunk.get("score") or 0.0)
        dense_sorted = sorted(dense, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        self.telemetry["reranked_count"] = len(dense_sorted)
        return dense_sorted[: rerank_top_n]

    # -------------------- Embedding helpers ---------------------------------
    def _encode_batch(self, texts: Sequence[str]) -> List[Optional[np.ndarray]]:
        """Encode ``texts`` in a single call when possible.

        This avoids issuing many sequential embedding requests which can add
        significant latency when the embedder is backed by a remote service or
        large local model. The helper is defensive so the pipeline gracefully
        degrades to returning ``None`` placeholders if batch encoding fails.
        """

        if not texts:
            return []

        cleaned_texts = [t if isinstance(t, str) else "" for t in texts]
        try:
            embeddings = self.embedder.encode(
                cleaned_texts,
                normalize_embeddings=True,
                batch_size=self._embed_batch_size,
            )
        except TypeError:
            embeddings = self.embedder.encode(cleaned_texts, normalize_embeddings=True)
        except Exception:
            logger.exception("Batch embedding failed; returning empty vectors")
            return [None for _ in cleaned_texts]

        if embeddings is None:
            return [None for _ in cleaned_texts]

        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1 and len(cleaned_texts) == 1:
                return [np.asarray(embeddings, dtype=np.float32)]
            if embeddings.ndim == 2 and embeddings.shape[0] == len(cleaned_texts):
                return [np.asarray(row, dtype=np.float32) for row in embeddings]

        result: List[Optional[np.ndarray]] = []
        try:
            for emb in embeddings:
                if emb is None:
                    result.append(None)
                else:
                    result.append(np.asarray(emb, dtype=np.float32))
        except Exception:
            logger.exception("Failed to normalise embedding outputs; using None placeholders")
            return [None for _ in cleaned_texts]

        if len(result) != len(cleaned_texts):
            if len(result) < len(cleaned_texts):
                result.extend([None] * (len(cleaned_texts) - len(result)))
            else:
                result = result[: len(cleaned_texts)]

        return result

    # -------------------- Deduplication ------------------------------------
    def hash_dedupe(self, candidates: List[Chunk]) -> List[Chunk]:
        seen_hashes: Set[str] = set()
        out: List[Chunk] = []
        for c in candidates:
            h = _hash_text(c.get("content", ""))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out.append(c)
        return out

    def semantic_dedupe(self, candidates: List[Chunk], threshold: float = SEMANTIC_DEDUPE_THRESHOLD, cross_doc_top: int = 20) -> List[Chunk]:
        """Perform semantic dedupe.
        1) Within-doc dedupe (fast): compare embeddings or compute temporarily
        2) Cross-doc dedupe among top `cross_doc_top` candidates
        """
        # Precompute embeddings where missing using batch encoding to avoid per-chunk latency
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        for idx, chunk in enumerate(candidates):
            vector = chunk.get("vector")
            if vector is None:
                missing_indices.append(idx)
                missing_texts.append(chunk.get("content", ""))
            else:
                try:
                    chunk["vector"] = np.asarray(vector, dtype=np.float32)
                except Exception:
                    missing_indices.append(idx)
                    missing_texts.append(chunk.get("content", ""))

        if missing_indices:
            embeddings = self._encode_batch(missing_texts)
            for pos, emb in zip(missing_indices, embeddings):
                candidates[pos]["vector"] = emb

        # Within-doc dedupe: keep highest rerank_score per similar group
        by_doc: Dict[Any, List[Chunk]] = {}
        for c in candidates:
            doc = c.get("doc_id") or "_no_doc"
            by_doc.setdefault(doc, []).append(c)
        kept: List[Chunk] = []
        for doc_id, list_chunks in by_doc.items():
            # sort descending rerank
            list_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            flags = [True] * len(list_chunks)
            for i in range(len(list_chunks)):
                if not flags[i]:
                    continue
                a = list_chunks[i]
                for j in range(i + 1, len(list_chunks)):
                    if not flags[j]:
                        continue
                    b = list_chunks[j]
                    # quick hash equality check
                    if _hash_text(a.get("content", "")) == _hash_text(b.get("content", "")):
                        flags[j] = False
                        continue
                    # semantic sim if embeddings present
                    if a.get("vector") is not None and b.get("vector") is not None:
                        sim = _cosine_sim(a.get("vector"), b.get("vector"))
                        if sim >= threshold:
                            flags[j] = False
                if flags[i]:
                    kept.append(list_chunks[i])
        # Cross-doc dedupe among top candidates
        # sort kept by rerank_score
        kept.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_for_cross = kept[: min(len(kept), cross_doc_top)]
        suppressed: Set[int] = set()
        for i in range(len(top_for_cross)):
            if i in suppressed:
                continue
            a = top_for_cross[i]
            for j in range(i + 1, len(top_for_cross)):
                if j in suppressed:
                    continue
                b = top_for_cross[j]
                if (a.get("doc_id") or "") == (b.get("doc_id") or ""):
                    continue
                if a.get("vector") is None or b.get("vector") is None:
                    continue
                sim = _cosine_sim(a.get("vector"), b.get("vector"))
                if sim >= threshold:
                    # drop the lower rerank_score (j)
                    suppressed.add(j)
        filtered_top = [c for idx, c in enumerate(top_for_cross) if idx not in suppressed]
        # combine filtered_top with the remainder beyond cross_doc_top
        remainder = kept[min(len(kept), cross_doc_top) :]
        result = filtered_top + remainder
        self.telemetry["deduped_count"] = len(result)
        return result

    # -------------------- Per-doc cap --------------------------------------
    def per_doc_cap(self, candidates: List[Chunk], max_per_doc: int = RAG_PER_DOC_CAP) -> List[Chunk]:
        out: List[Chunk] = []
        counts: Dict[Any, int] = {}
        available_docs = {c.get("doc_id") for c in candidates if c.get("doc_id")}
        target_min_docs = min(3, len(available_docs))
        for c in candidates:
            doc = c.get("doc_id") or "_no_doc"
            cnt = counts.get(doc, 0)
            if cnt < max_per_doc:
                out.append(c)
                counts[doc] = cnt + 1
            else:
                # skip for now
                continue
        # Ensure at least target_min_docs distinct docs present if available
        distinct_in_out = {c.get("doc_id") for c in out if c.get("doc_id")}
        if len(distinct_in_out) < target_min_docs and target_min_docs > 0:
            # try to add missing doc entries from candidates in rerank order
            for c in candidates:
                doc = c.get("doc_id")
                if not doc or doc in distinct_in_out:
                    continue
                # allow one extra for this doc
                out.append(c)
                distinct_in_out.add(doc)
                if len(distinct_in_out) >= target_min_docs:
                    break
        self.telemetry["capped_count"] = len(out)
        return out

    # -------------------- Extractive compression ----------------------------
    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        # Robust rule-based splitter: split on sentence end followed by space and capital/number
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9£])", text.strip())
        segs = [s.strip() for s in raw if s and s.strip()]
        return segs

    def compress_chunk(self, query: str, chunk: Chunk, per_chunk_chars: int = COMPRESS_PER_CHUNK_CHARS) -> Chunk:
        """Extractively compress chunk content preserving citation metadata.
        Adds compressed_text and sentence_spans (if available) to returned chunk dict.
        """
        text = chunk.get("content") or ""
        sentences = self._split_sentences(text)
        if not sentences:
            chunk["compressed_text"] = text[:per_chunk_chars]
            return chunk
        # if there are <=3 short sentences, skip compression
        short_count = sum(1 for s in sentences if len(s) < 120)
        if len(sentences) <= 3 and short_count >= len(sentences):
            chunk["compressed_text"] = text[:per_chunk_chars]
            return chunk
        orig_len = len(text)
        # Score sentences with reranker: pairs (query, sentence)
        pairs = [[query, s] for s in sentences]
        scores = []
        if self._reranker is None:
            # fallback: use simple heuristics (tf-like)
            for s in sentences:
                scores.append(float(min(1.0, 0.1 + 0.9 * len(set(query.split()) & set(s.split())) / (len(set(s.split())) + 1))))
        else:
            try:
                scores = []
                for i in range(0, len(pairs), RERANK_BATCH):
                    batch = pairs[i : i + RERANK_BATCH]
                    batch_scores = list(self._reranker.predict(batch))
                    scores.extend([float(s) for s in batch_scores])
            except Exception:
                logger.exception("Sentence reranker failed; falling back to naive scoring")
                scores = [0.0 for _ in sentences]
        # Pair sentences with scores
        sent_objs = [{"text": s, "score": scores[i] if i < len(scores) else 0.0, "idx": i} for i, s in enumerate(sentences)]
        # Greedy MMR selection
        selected: List[Dict[str, Any]] = []
        remaining = sent_objs.copy()
        # Precompute sentence embeddings for similarity in MMR (use embedder)
        sent_embs = self._encode_batch(sentences)
        # sort remaining by score desc
        remaining.sort(key=lambda x: x["score"], reverse=True)
        total_chars = 0
        # guard: if there are very many sentences, limit to top 50 by score for speed
        remaining = remaining[:50]
        while remaining and total_chars < per_chunk_chars:
            if not selected:
                choice = remaining.pop(0)
                selected.append(choice)
                total_chars += len(choice["text"]) + 1
                continue
            # select candidate maximizing lambda*rel - (1-lambda)*maxsim
            best_idx = -1
            best_val = -1e9
            for idx, cand in enumerate(remaining):
                rel = cand["score"]
                # compute max similarity with already selected
                cand_emb = sent_embs[cand["idx"]] if cand["idx"] < len(sent_embs) else None
                maxsim = 0.0
                if cand_emb is not None:
                    for sel in selected:
                        sel_emb = sent_embs[sel["idx"]] if sel["idx"] < len(sent_embs) else None
                        if sel_emb is None:
                            continue
                        maxsim = max(maxsim, _cosine_sim(cand_emb, sel_emb))
                val = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * maxsim
                if val > best_val:
                    best_val = val
                    best_idx = idx
            if best_idx == -1:
                break
            choice = remaining.pop(best_idx)
            if total_chars + len(choice["text"]) + 1 > per_chunk_chars:
                break
            selected.append(choice)
            total_chars += len(choice["text"]) + 1
        if not selected:
            # fallback: take the top sentence(s)
            selected = remaining[:1]
        compressed_text = " ".join(s["text"] for s in selected)
        # Truncate to per_chunk_chars
        if len(compressed_text) > per_chunk_chars:
            compressed_text = compressed_text[:per_chunk_chars]

        # If compression did not reduce sufficiently (target at least 30% reduction),
        # enforce a stricter cap: target_max = min(per_chunk_chars, int(orig_len * 0.7)).
        target_max = min(per_chunk_chars, int(orig_len * 0.7))
        if target_max > 0 and len(compressed_text) > target_max:
            # Greedily keep highest-scoring selected sentences until under target_max
            sel_sorted = sorted(selected, key=lambda x: x.get("score", 0.0), reverse=True)
            kept = []
            kept_len = 0
            for s in sel_sorted:
                if not kept:
                    kept.append(s)
                    kept_len += len(s["text"]) + 1
                    continue
                if kept_len + len(s["text"]) + 1 > target_max:
                    continue
                kept.append(s)
                kept_len += len(s["text"]) + 1
            if not kept:
                # ensure at least the top sentence
                kept = [sel_sorted[0]]
            # preserve original order of sentences
            kept_idx = {s["idx"] for s in kept}
            final_ordered = [s for s in selected if s["idx"] in kept_idx]
            compressed_text = " ".join(s["text"] for s in final_ordered)
            if len(compressed_text) > target_max:
                compressed_text = compressed_text[:target_max]

        chunk["compressed_text"] = compressed_text
        # preserve citation spans if present
        chunk["sentence_spans"] = None
        return chunk

    # -------------------- Context packing ----------------------------------
    def pack_context(self, compressed_chunks: List[Chunk], max_chars: int = RAG_CONTEXT_MAX_CHARS, ensure_min_docs: int = 3) -> Tuple[str, List[Chunk]]:
        packed_blocks: List[Dict[str, Any]] = []
        packed_chars = 0
        doc_counts: Dict[str, int] = defaultdict(int)
        available_docs = [c.get("doc_id") or "unknown" for c in compressed_chunks]
        target_min_docs = min(ensure_min_docs, len({d for d in available_docs if d and d != "unknown"}))

        for chunk in compressed_chunks:
            remaining = max_chars - packed_chars
            block, block_len = self._format_block(chunk, remaining)
            if not block:
                if remaining <= 0:
                    break
                continue
            packed_blocks.append({"doc": chunk.get("doc_id") or "unknown", "block": block, "chunk": chunk, "chars": block_len})
            packed_chars += block_len
            doc_counts[chunk.get("doc_id") or "unknown"] += 1
            if packed_chars >= max_chars:
                break

        # If we failed to capture enough distinct docs but have more available, try to inject extras.
        included_docs = {blk["doc"] for blk in packed_blocks if blk["doc"] and blk["doc"] != "unknown"}
        if len(included_docs) < target_min_docs:
            missing_docs = [
                doc
                for doc in ({c.get("doc_id") or "unknown" for c in compressed_chunks} - included_docs)
                if doc and doc != "unknown"
            ]
            for doc in missing_docs:
                chunk = next((c for c in compressed_chunks if (c.get("doc_id") or "unknown") == doc), None)
                if chunk is None:
                    continue
                remaining = max_chars - packed_chars
                block, block_len = self._format_block(chunk, remaining)
                if not block:
                    # try freeing space by dropping the last block from an overrepresented doc
                    drop_idx = next(
                        (idx for idx in reversed(range(len(packed_blocks))) if doc_counts[packed_blocks[idx]["doc"]] > 1),
                        None,
                    )
                    if drop_idx is None:
                        continue
                    removed = packed_blocks.pop(drop_idx)
                    packed_chars -= removed["chars"]
                    doc_counts[removed["doc"]] -= 1
                    if doc_counts[removed["doc"]] == 0:
                        doc_counts.pop(removed["doc"], None)
                        if removed["doc"] != "unknown":
                            included_docs.discard(removed["doc"])
                    remaining = max_chars - packed_chars
                    block, block_len = self._format_block(chunk, remaining)
                    if not block:
                        continue
                packed_blocks.append({"doc": doc, "block": block, "chunk": chunk, "chars": block_len})
                packed_chars += block_len
                doc_counts[doc] += 1
                if doc != "unknown":
                    included_docs.add(doc)
                if len(included_docs) >= target_min_docs:
                    break

        self.telemetry["packed_chars"] = packed_chars
        context_str = "".join(blk["block"] for blk in packed_blocks)
        ordered_top_docs: List[str] = []
        for blk in packed_blocks:
            doc = blk["doc"]
            if doc and doc not in ordered_top_docs:
                ordered_top_docs.append(doc)
            if len(ordered_top_docs) == 5:
                break
        logger.info("RAG pack: packed_chars=%d top_docs=%s", packed_chars, ordered_top_docs)
        return context_str, [blk["chunk"] for blk in packed_blocks]

    def _format_block(self, chunk: Chunk, remaining_chars: int) -> Tuple[str, int]:
        if remaining_chars <= 0:
            return "", 0
        text = chunk.get("compressed_text") or chunk.get("content") or ""
        if not text:
            return "", 0
        doc = chunk.get("doc_id") or "unknown"
        md = chunk.get("payload") or {}
        if md.get("line_start") is not None and md.get("line_end") is not None:
            span = f"{doc}({md.get('line_start')}-{md.get('line_end')})"
        else:
            span = f"{doc}"
        prefix = f"[{span}] "
        suffix = "\n\n"
        available_for_text = remaining_chars - len(prefix) - len(suffix)
        if available_for_text <= 0:
            return "", 0
        trimmed_text = text[:available_for_text]
        if not trimmed_text.strip():
            return "", 0
        block = f"{prefix}{trimmed_text}{suffix}"
        return block, len(block)

    # -------------------- Full pipeline ------------------------------------
    def answer(self, query: str, *, ensure_min_docs: int = 3, collections: Optional[List[str]] = None) -> Dict[str, Any]:
        start = time.time()
        # Detect purchase order / invoice queries and adjust retrieval accordingly
        financial_q = False
        qlow = (query or "").lower()
        if re.search(r"\b(invoice|invoices|purchase order|purchase orders|po\b|po number|po no\b|purchase-order|p\.o\.)", qlow):
            financial_q = True

        # Stage A: retrieve & rerank
        retrieval_collections: Optional[List[str]] = None
        if collections:
            retrieval_collections = list(collections)
        elif financial_q:
            retrieval_collections = [self.primary_collection, self.uploaded_collection]

        top_k = RAG_TOP_K
        rerank_top_n = RAG_RERANK_TOP_N
        if financial_q and not collections:
            # broaden retrieval for invoice/PO related queries when caller does not override collections
            top_k = max(RAG_TOP_K, 30)
            rerank_top_n = max(RAG_RERANK_TOP_N, 20)

        dense = self.retrieve_and_rerank(
            query,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            collections=retrieval_collections,
        )
        # Stage B1: hash dedupe
        after_hash = self.hash_dedupe(dense)
        # Stage B2: semantic dedupe
        after_sem = self.semantic_dedupe(after_hash, threshold=SEMANTIC_DEDUPE_THRESHOLD)
        # Stage C: per-doc cap
        if financial_q:
            # allow a slightly higher per-doc cap for transactional docs to include more line-level evidence
            after_cap = self.per_doc_cap(after_sem, max_per_doc=max(RAG_PER_DOC_CAP, 3))
        else:
            after_cap = self.per_doc_cap(after_sem, max_per_doc=RAG_PER_DOC_CAP)
        # Stage D: compress
        compressed = [self.compress_chunk(query, c, per_chunk_chars=COMPRESS_PER_CHUNK_CHARS) for c in after_cap]
        # Stage E: pack context with doc diversity guarantee
        ctx, used_chunks = self.pack_context(compressed, max_chars=RAG_CONTEXT_MAX_CHARS, ensure_min_docs=ensure_min_docs)
        # If no chunks were kept
        if not used_chunks:
            # Pass explicit token to generator context to force refusal
            ctx = "NO RELEVANT CONTEXT"
        # Stage F: generation via LM Studio
        model_name = (
            getattr(self.settings, "lmstudio_chat_model", None)
            or LMSTUDIO_MODEL
        )
        options = {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": -1,
        }
        prompt_payload = {"query": query, "context": ctx}
        answer_text = ""
        sources = []
        client = get_lmstudio_client()
        system_prompt = (
            "You are ProcWise's sourcing copilot. "
            "Given the JSON payload containing a query and supporting context, "
            "draft a concise answer grounded only in that context. "
            "Always finish with a 'Sources:' line listing the unique document_ids "
            "referenced in the answer separated by commas. "
            "If the context indicates 'NO RELEVANT CONTEXT', respond with a "
            "brief refusal followed by 'Sources:'."
        )
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, ensure_ascii=False),
                },
            ]
            gen = client.chat(
                model=model_name,
                messages=messages,
                options=options,
            )
            answer_text = (
                gen.get("message", {}).get("content")
                or gen.get("response")
                or ""
            )
        except Exception:
            logger.exception("LM Studio generation failed")
            answer_text = ""

        # If generator returned empty or NO RELEVANT CONTEXT, enforce refusal sentence
        if not answer_text or ctx == "NO RELEVANT CONTEXT":
            answer_text = REFUSAL_SENTENCE + "\n\nSources: "
            sources = []
        else:
            # Extract Sources: line if present; otherwise compute from used_chunks
            # Ensure Sources: only contains doc_ids present
            if "Sources:" in answer_text:
                # naive extraction of line after Sources:
                try:
                    tail = answer_text.split("Sources:", 1)[1]
                    found = re.findall(r"[A-Za-z0-9_.\-]+", tail)
                    sources = [d for d in found if d]
                except Exception:
                    sources = []
            else:
                sources = [c.get("doc_id") for c in used_chunks if c.get("doc_id")]
            # append Sources: if missing
            if "Sources:" not in answer_text:
                unique_sources = []
                for s in sources:
                    if s and s not in unique_sources:
                        unique_sources.append(s)
                if unique_sources:
                    answer_text = answer_text.rstrip() + "\n\nSources: " + ", ".join(unique_sources)
                else:
                    answer_text = answer_text.rstrip() + "\n\nSources:"

        elapsed = time.time() - start
        diagnostics = {
            "dense": self.telemetry.get("dense_count", 0),
            "after_rerank": self.telemetry.get("reranked_count", 0),
            "after_dedupe": self.telemetry.get("deduped_count", 0),
            "after_cap": self.telemetry.get("capped_count", 0),
            "packed_chars": self.telemetry.get("packed_chars", 0),
            "elapsed_seconds": elapsed,
        }
        # top doc ids
        doc_counts: Dict[str, int] = {}
        for c in used_chunks:
            did = c.get("doc_id") or "unknown"
            doc_counts[did] = doc_counts.get(did, 0) + 1
        top_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        diagnostics["top_docs"] = [d for d, _ in top_docs]

        result = {
            "answer": answer_text,
            "sources": [s for s in sources if s],
            "diagnostics": diagnostics,
            "used_context_chunks": used_chunks,
        }
        return result
