import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.documents import router as documents_router


class _DummyBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


class _DummyS3Client:
    def __init__(self, objects: Dict[str, bytes], bucket: str) -> None:
        self._objects = objects
        self._bucket = bucket

    def get_object(self, Bucket: str, Key: str):  # noqa: N802
        assert Bucket == self._bucket
        assert Key in self._objects, f"Unexpected key {Key}"
        return {"Body": _DummyBody(self._objects[Key])}

    def list_objects_v2(self, Bucket: str, Prefix: str, **kwargs):  # noqa: N802
        assert Bucket == self._bucket
        matched = [key for key in self._objects if key.startswith(Prefix)]
        return {"Contents": [{"Key": key} for key in matched], "IsTruncated": False}


class _Settings:
    s3_bucket_name = "procwisemvp"


class _DummyAgentNick:
    def __init__(self, objects: Dict[str, bytes], db_path: Path) -> None:
        self.settings = _Settings()
        self._db_path = db_path
        self._objects = objects

    @contextmanager
    def reserve_s3_connection(self):
        yield _DummyS3Client(self._objects, self.settings.s3_bucket_name)

    def get_db_connection(self):
        conn = sqlite3.connect(self._db_path)
        return conn

    def ollama_options(self):  # pragma: no cover - deterministic default
        return {}


@pytest.fixture
def api_app(tmp_path):
    invoice_text = "\n".join(
        [
            "Invoice Number: INV-3001",
            "Vendor: Example Supplier",
            "Invoice Date: 2024-04-01",
            "Due Date: 2024-04-30",
            "Invoice Total: 250.00",
            "Item Description    Qty    Unit Price    Line Total",
            "Consulting Hours    5      50            250",
        ]
    )
    second_invoice_text = "\n".join(
        [
            "Invoice Number: INV-3002",
            "Vendor: Example Supplier",
            "Invoice Date: 2024-05-01",
            "Due Date: 2024-05-30",
            "Invoice Total: 125.00",
            "Item Description    Qty    Unit Price    Line Total",
            "Managed Services    5      25            125",
        ]
    )
    objects = {
        "incoming/april/invoice-3001.txt": invoice_text.encode("utf-8"),
        "incoming/april/invoice-3002.txt": second_invoice_text.encode("utf-8"),
    }
    db_path = tmp_path / "extraction.sqlite"

    app = FastAPI()
    app.include_router(documents_router)
    app.state.agent_nick = _DummyAgentNick(objects, db_path)

    client = TestClient(app)
    return client, db_path, objects


def test_extract_document_from_s3_endpoint(api_app):
    client, db_path, objects = api_app
    s3_path = "incoming/april/"

    response = client.post(
        "/documents/extract-from-s3",
        json={"s3_path": s3_path, "metadata": {"custom": "value"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    documents = body["documents"]
    assert len(documents) == len(objects)
    first_payload = body["document"]
    assert first_payload == documents[0]
    assert {doc["metadata"]["s3_object_key"] for doc in documents} == set(objects.keys())
    assert all(doc["document_type"] == "Invoice" for doc in documents)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM "proc.raw_invoice" ORDER BY rowid').fetchall()
        assert len(rows) == len(objects)
        headers = [json.loads(row["header_json"]) for row in rows]
        invoice_ids = {header["invoice_id"] for header in headers}
        assert invoice_ids == {"INV-3001", "INV-3002"}
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            assert metadata["s3_bucket"] == "procwisemvp"
            assert metadata["s3_object_key"] in objects


def test_embed_document_without_user_id(api_app, monkeypatch):
    client, *_ = api_app

    class DummyRAGService:
        def __init__(self) -> None:
            self.created_collections = []
            self.uploaded_collection = "uploaded_documents"

        def ensure_collection(self, name: str) -> None:
            self.created_collections.append(name)

    class DummyPipeline:
        def __init__(self) -> None:
            self.rag = DummyRAGService()
            self.activations = []

        def activate_uploaded_context(self, document_ids, *, metadata=None, session_id=None):
            self.activations.append(
                {
                    "document_ids": list(document_ids or []),
                    "metadata": dict(metadata or {}),
                    "session_id": session_id,
                }
            )

    pipeline = DummyPipeline()
    client.app.state.rag_pipeline = pipeline

    calls: Dict[str, Dict] = {}

    class DummyEmbeddingService:
        def __init__(self, agent_nick, *, collection_name: str, **_kwargs):
            calls["init"] = {"collection_name": collection_name, "agent_nick": agent_nick}

        def embed_document(self, *, filename: str, file_bytes: bytes, metadata: Dict[str, str]):
            calls["embed"] = {
                "filename": filename,
                "file_bytes": file_bytes,
                "metadata": dict(metadata),
            }
            return SimpleNamespace(
                document_id="doc-001",
                collection="uploaded_documents",
                chunk_count=2,
                metadata={
                    "filename": filename,
                    "doc_name": filename,
                },
            )

    monkeypatch.setattr(
        "api.routers.documents.DocumentEmbeddingService",
        DummyEmbeddingService,
    )

    response = client.post(
        "/document/embed-document",
        files={"files": ("invoice.txt", b"Invoice details", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["total_documents"] == 1
    assert body["total_chunks"] == 2
    assert body["failed"] == []

    assert "embed" in calls
    embed_metadata = calls["embed"]["metadata"]
    assert embed_metadata["mime_type"] == "text/plain"
    assert embed_metadata["ingestion_source"] == "document_embed_endpoint"
    assert "uploaded_by" not in embed_metadata

    assert pipeline.rag.created_collections == ["uploaded_documents"]
    assert len(pipeline.activations) == 1
    activation = pipeline.activations[0]
    assert activation["document_ids"] == ["doc-001"]
    assert activation["metadata"] == {
        "filenames": ["invoice.txt"],
        "total_chunks": 2,
    }
    assert activation["session_id"] is None

    processed = body["processed"][0]
    assert processed["document_id"] == "doc-001"
    assert processed["collection"] == "uploaded_documents"
    assert processed["chunk_count"] == 2
    assert processed["metadata"]["filename"] == "invoice.txt"
