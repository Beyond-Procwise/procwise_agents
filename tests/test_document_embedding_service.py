import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.document_embedding_service import DocumentEmbeddingService


class DummyEmbedder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return np.array([float(len(texts)), 0.0, 1.0], dtype="float32")
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), 0.0, 1.0])
        return np.array(vectors, dtype="float32")


class DummyQdrantClient:
    def __init__(self):
        self.collections = {}
        self.upserts = []

    def get_collection(self, collection_name: str):
        info = self.collections.get(collection_name)
        if not info:
            raise RuntimeError("missing collection")
        return SimpleNamespace(payload_schema=info.get("schema"))

    def create_collection(self, collection_name: str, vectors_config):
        self.collections[collection_name] = {"size": vectors_config.size, "schema": {}}

    def create_payload_index(self, collection_name: str, field_name: str, field_schema, wait: bool = True):
        info = self.collections.setdefault(collection_name, {"size": 3, "schema": {}})
        info.setdefault("schema", {})[field_name] = field_schema

    def upsert(self, collection_name: str, points, wait: bool = True):
        self.upserts.append({"collection": collection_name, "points": points})
        self.collections.setdefault(collection_name, {"size": 3, "schema": {}})

    def search(self, *args, **kwargs):
        raise AssertionError("Search should not be invoked in embedding tests")


class DummyRAGService:
    def __init__(self):
        self.upserts: list[tuple[list[dict], str]] = []

    def upsert_payloads(self, payloads, text_representation_key="content"):
        self.upserts.append((payloads, text_representation_key))


@pytest.fixture()
def service():
    client = DummyQdrantClient()
    embedder = DummyEmbedder()
    rag_service = DummyRAGService()
    settings = SimpleNamespace(rag_chunk_chars=50, rag_chunk_overlap=10, rag_model="mock-model")
    agent = SimpleNamespace(
        qdrant_client=client,
        embedding_model=embedder,
        settings=settings,
        ollama_options=lambda: {"temperature": 0.1},
    )

    svc = DocumentEmbeddingService(
        agent,
        rag_service_factory=lambda _: rag_service,
    )
    return svc, client, rag_service


def test_embed_document_creates_collection_and_upserts(service):
    svc, client, rag_service = service
    result = svc.embed_document(filename="sample.txt", file_bytes=b"Hello world", metadata={"mime_type": "text/plain"})
    assert result.collection == "uploaded_documents"
    assert result.chunk_count > 0
    assert client.upserts, "Expected upsert call"
    point = client.upserts[0]["points"][0]
    assert point.payload["filename"] == "sample.txt"
    assert point.payload["mime_type"] == "text/plain"
    assert client.collections["uploaded_documents"]["schema"].keys() >= {
        "document_id",
        "filename",
        "file_extension",
        "uploaded_at",
    }
    assert rag_service.upserts, "Expected propagation to RAG"
    forwarded_payloads, key = rag_service.upserts[0]
    assert key == "content"
    assert forwarded_payloads[0]["source_collection"] == "uploaded_documents"
    assert forwarded_payloads[0]["document_id"] == result.document_id


def test_embed_document_handles_rag_failure(service):
    svc, _, rag_service = service

    def failing_upsert(*args, **kwargs):
        raise RuntimeError("boom")

    rag_service.upsert_payloads = failing_upsert

    result = svc.embed_document(filename="policy.txt", file_bytes=b"Always collect two bids")
    assert result.chunk_count > 0
