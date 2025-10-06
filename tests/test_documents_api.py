import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

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
    def __init__(self, payload: bytes, expected_key: str) -> None:
        self._payload = payload
        self._expected_key = expected_key

    def get_object(self, Bucket: str, Key: str):  # noqa: N802
        assert Bucket == "unit-test-bucket"
        assert Key == self._expected_key
        return {"Body": _DummyBody(self._payload)}


class _Settings:
    s3_bucket_name = "unit-test-bucket"


class _DummyAgentNick:
    def __init__(self, payload: bytes, db_path: Path, object_key: str) -> None:
        self.settings = _Settings()
        self._payload = payload
        self._db_path = db_path
        self._object_key = object_key

    @contextmanager
    def reserve_s3_connection(self):
        yield _DummyS3Client(self._payload, self._object_key)

    def get_db_connection(self):
        conn = sqlite3.connect(self._db_path)
        return conn

    def ollama_options(self):  # pragma: no cover - deterministic default
        return {}


@pytest.fixture
def api_app(tmp_path):
    document_text = "\n".join(
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
    object_key = "incoming/invoice.txt"
    payload = document_text.encode("utf-8")
    db_path = tmp_path / "extraction.sqlite"

    app = FastAPI()
    app.include_router(documents_router)
    app.state.agent_nick = _DummyAgentNick(payload, db_path, object_key)

    client = TestClient(app)
    return client, db_path, object_key


def test_extract_document_from_s3_endpoint(api_app):
    client, db_path, object_key = api_app

    response = client.post(
        "/documents/extract-from-s3",
        json={"object_key": object_key, "metadata": {"custom": "value"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    payload = body["document"]
    assert payload["document_type"] == "Invoice"
    assert payload["header"]["invoice_id"] == "INV-3001"
    assert payload["metadata"]["s3_object_key"] == object_key

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM "proc.raw_invoice"').fetchone()
        assert row is not None
        header = json.loads(row["header_json"])
        assert header["invoice_id"] == "INV-3001"
        tables = json.loads(row["tables_json"])
        assert tables and tables[0]["rows"][0]["item_description"].lower() == "consulting hours"
