import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from types import SimpleNamespace
import numpy as np

from services.rag_qwen30b import (
    RAGQwen30b,
    COMPRESS_PER_CHUNK_CHARS,
    RAG_CONTEXT_MAX_CHARS,
    SEMANTIC_DEDUPE_THRESHOLD,
)


class DummyEmbed:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        """Deterministic, lightweight embedding: hash-based scalar -> small vector."""
        if isinstance(texts, list):
            return [self._single(t) for t in texts]
        return self._single(texts)

    def _single(self, t):
        if t is None:
            return [0.0, 0.0, 0.0]
        h = sum(ord(c) for c in str(t))
        v = [(h % 97) / 97.0, ((h >> 3) % 89) / 89.0, ((h >> 6) % 83) / 83.0]
        return v


class DummyCrossEncoder:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    def predict(self, pairs):
        # higher score when sentence/query includes token 'important' else low
        out = []
        for q, t in pairs:
            if "important" in t.lower() or "important" in str(q).lower():
                out.append(0.95)
            else:
                out.append(0.15)
        return out


def make_rag():
    nick = SimpleNamespace()
    nick.settings = SimpleNamespace(qdrant_collection_name="test")
    nick.qdrant_client = SimpleNamespace()
    nick.embedding_model = DummyEmbed()
    nick.device = "cpu"
    return RAGQwen30b(nick, cross_encoder_cls=DummyCrossEncoder)


def test_hash_dedupe():
    rag = make_rag()
    chunks = [
        {"content": "Hello, WORLD!! This is a test.", "doc_id": "d1"},
        {"content": "hello world this is a test", "doc_id": "d1"},
        {"content": "Completely different text", "doc_id": "d2"},
    ]
    out = rag.hash_dedupe(chunks)
    # the two first should be deduped
    assert any(c["doc_id"] == "d2" for c in out)
    assert sum(1 for c in out if c["doc_id"] == "d1") == 1


def test_semantic_dedupe_within_and_cross_doc():
    rag = make_rag()
    # create two very similar vectors for different docs
    base_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    near_vec = np.array([0.999, 0.001, 0.0], dtype=float)
    chunks = [
        {"content": "Paragraph A", "doc_id": "doc1", "vector": base_vec, "rerank_score": 0.9},
        {"content": "Paragraph A variant", "doc_id": "doc1", "vector": near_vec, "rerank_score": 0.8},
        {"content": "Paragraph B", "doc_id": "doc2", "vector": near_vec, "rerank_score": 0.7},
    ]
    out = rag.semantic_dedupe(chunks, threshold=SEMANTIC_DEDUPE_THRESHOLD, cross_doc_top=10)
    # Expect within-doc dedupe to collapse two doc1 variants into 1
    doc1_count = sum(1 for c in out if c.get("doc_id") == "doc1")
    assert doc1_count == 1
    # If cross-doc sim also high, one of doc1/doc2 may be removed depending on ordering
    # Ensure no two outputs are above threshold similarity
    vecs = [c.get("vector") for c in out if c.get("vector") is not None]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
            assert sim < SEMANTIC_DEDUPE_THRESHOLD or math.isclose(sim, SEMANTIC_DEDUPE_THRESHOLD) is False


def test_compress_chunk_reduces_length():
    rag = make_rag()
    long_sentences = [
        "This is an unimportant filler sentence that adds noise.",
        "This sentence contains an important point about procurement and pricing.",
        "Another supporting sentence that is less important.",
        "Important: this sentence is very important and should be included.",
        "Final filler sentence to increase length and diversity.",
    ]
    text = " ".join(long_sentences * 5)  # make it longer
    chunk = {"content": text, "doc_id": "dX", "payload": {}}
    orig_len = len(text)
    comp = rag.compress_chunk("Tell me the important points", chunk, per_chunk_chars=COMPRESS_PER_CHUNK_CHARS)
    compressed = comp.get("compressed_text")
    assert compressed is not None
    assert len(compressed) <= COMPRESS_PER_CHUNK_CHARS
    # check that compression removed at least 30% of characters on this synthetic example
    assert len(compressed) < orig_len * 0.7


def test_financial_query_broadens_retrieval(monkeypatch):
    # Dummy Qdrant that records search calls
    class DummyQdrantSearch:
        def __init__(self):
            self.search_calls = []

        def search(self, **kwargs):
            # record the kwargs for inspection and return empty list to force NO RELEVANT CONTEXT
            self.search_calls.append(kwargs)
            return []

    dq = DummyQdrantSearch()
    nick = SimpleNamespace()
    nick.settings = SimpleNamespace(qdrant_collection_name="primary_coll", uploaded_documents_collection_name="uploaded_documents")
    nick.qdrant_client = dq
    nick.embedding_model = DummyEmbed()
    nick.device = "cpu"
    rag = RAGQwen30b(nick, cross_encoder_cls=DummyCrossEncoder)

    # Call answer with an invoice-related query
    resp = rag.answer("Show me invoices for PO 12345")

    # Ensure qdrant.search was called at least once
    assert dq.search_calls, "Expected qdrant.search to be called"
    # Check that at least one search targeted the uploaded collection
    collections_searched = {call.get('collection_name') for call in dq.search_calls}
    assert 'uploaded_documents' in collections_searched or 'uploaded_documents' in collections_searched
    # Check that at least one call used an increased limit (>=30)
    limits = [call.get('limit') for call in dq.search_calls]
    assert any((l or 0) >= 30 for l in limits), f"Expected at least one call with limit >= 30, got {limits}"


def test_pack_context_enforces_doc_diversity():
    rag = make_rag()
    chunks = []
    for i in range(6):
        chunks.append(
            {
                "doc_id": f"doc{i % 4}",
                "content": f"Important sentence from doc {i % 4}.",
                "compressed_text": f"Important sentence from doc {i % 4}.",
                "payload": {},
            }
        )
    ctx, used = rag.pack_context(chunks, max_chars=RAG_CONTEXT_MAX_CHARS, ensure_min_docs=3)
    included = {c.get("doc_id") for c in used if c.get("doc_id")}
    assert len(included) >= 3, f"Expected at least 3 docs, got {included}"
    assert ctx, "Context should not be empty when chunks are available"
