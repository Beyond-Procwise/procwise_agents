import os
import sys
import types
from typing import Any

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.supplier_relationship_service import SupplierRelationshipService


class DummySettings:
    qdrant_collection_name = "supplier-relationships"


class DummyNick:
    def __init__(self, client):
        self.settings = DummySettings()
        self.qdrant_client = client


class DummyPoint:
    def __init__(self, payload):
        self.payload = payload


@pytest.fixture(autouse=True)
def stub_rag_service(monkeypatch):
    module = types.SimpleNamespace(RAGService=lambda *_: None)
    monkeypatch.setattr(
        "services.supplier_relationship_service.rag_module",
        module,
    )


def test_fetch_relationship_handles_missing_index(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        def scroll(self, *_, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise Exception("Bad request: Index required but not found for \"supplier_id\"")

            payload = {
                "supplier_id": "S1",
                "supplier_name": "Acme Corp",
                "supplier_name_normalized": "acme corp",
            }
            return [DummyPoint(payload)], None

    client = DummyClient()
    service = SupplierRelationshipService(DummyNick(client))

    results = service.fetch_relationship(supplier_id="S1")

    assert results == [
        {
            "supplier_id": "S1",
            "supplier_name": "Acme Corp",
            "supplier_name_normalized": "acme corp",
        }
    ]
    assert client.calls >= 2


def test_fetch_relationship_missing_index_without_match(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        def scroll(self, *_, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise Exception("Bad request: Index required but not found for \"supplier_name\"")

            payload = {
                "supplier_id": "S2",
                "supplier_name": "Beta",
                "supplier_name_normalized": "beta",
            }
            return [DummyPoint(payload)], None

    client = DummyClient()
    service = SupplierRelationshipService(DummyNick(client))

    results = service.fetch_relationship(supplier_id="S1")

    assert results == []
    assert client.calls >= 2


def test_service_creates_missing_payload_indexes(monkeypatch):
    created: list[tuple[str, Any]] = []

    class DummyCollection:
        def __init__(self):
            self.payload_schema = {"document_type": {"type": "keyword"}}

    class DummyClient:
        def get_collection(self, collection_name):
            assert collection_name == "supplier-relationships"
            return DummyCollection()

        def create_payload_index(self, *, collection_name, field_name, field_schema, wait):
            assert wait is True
            created.append((collection_name, field_name, field_schema))

    client = DummyClient()
    SupplierRelationshipService(DummyNick(client))

    created_fields = {field for _, field, _ in created}
    assert created_fields == {"supplier_id", "supplier_name_normalized"}
