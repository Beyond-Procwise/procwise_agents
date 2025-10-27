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

    def search(self, collection_name: str, query_vector, limit: int, with_payload: bool, with_vectors: bool, query_filter=None):
        points = []
        if self.upserts:
            points = self.upserts[-1]["points"][:limit]
        results = []
        for point in points:
            results.append(SimpleNamespace(payload=point.payload, score=0.42))
        return results


@pytest.fixture()
def service():
    client = DummyQdrantClient()
    embedder = DummyEmbedder()
    settings = SimpleNamespace(rag_chunk_chars=50, rag_chunk_overlap=10, rag_model="mock-model")
    agent = SimpleNamespace(
        qdrant_client=client,
        embedding_model=embedder,
        settings=settings,
        ollama_options=lambda: {"temperature": 0.1},
    )
    def fake_chat(**kwargs):
        return {"message": {"content": "Mock grounded answer"}}
    svc = DocumentEmbeddingService(agent, chat_completion_fn=fake_chat)
    return svc, client


def test_embed_document_creates_collection_and_upserts(service):
    svc, client = service
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


def test_query_returns_grounded_answer(service):
    svc, _ = service
    svc.embed_document(filename="note.txt", file_bytes=b"Procurement policy requires two quotes.")
    response = svc.query("What does the policy require?", top_k=2)
    assert response["answer"] == "Mock grounded answer"
    assert response["contexts"]
    assert response["contexts"][0]["document_id"]
