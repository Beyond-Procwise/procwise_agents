from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from services.static_policy_loader import StaticPolicyLoader


class _DummyBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _DummyPaginator:
    def __init__(self, entries):
        self._entries = entries

    def paginate(self, **kwargs):  # pragma: no cover - simple iterator wrapper
        sanitized = []
        for entry in self._entries:
            sanitized.append({k: v for k, v in entry.items() if k != "Body"})
        yield {"Contents": sanitized}


class DummyS3Client:
    def __init__(self, bucket: str, entries):
        self.bucket = bucket
        self._entries = list(entries)
        self._body_map = {entry["Key"]: entry["Body"] for entry in self._entries}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _DummyPaginator(self._entries)

    def get_object(self, Bucket: str, Key: str):
        assert Bucket == self.bucket
        body = self._body_map[Key]
        return {"Body": _DummyBody(body)}


class DummyQdrantClient:
    def __init__(self) -> None:
        self.upserts = []
        self.deleted = []
        self.created_collections = []
        self.created_indexes = []

    def get_collection(self, collection_name: str):
        raise RuntimeError("collection missing")

    def create_collection(self, collection_name: str, **kwargs):
        self.created_collections.append(collection_name)

    def create_payload_index(self, collection_name: str, field_name: str, **kwargs):
        self.created_indexes.append((collection_name, field_name))

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def scroll(self, **kwargs):
        target = None
        scroll_filter = kwargs.get("scroll_filter")
        if scroll_filter and getattr(scroll_filter, "must", None):
            condition = scroll_filter.must[0]
            target = getattr(getattr(condition, "match", None), "value", None)
        for batch in self.upserts:
            for point in batch.get("points", []):
                payload = getattr(point, "payload", {})
                if target and payload.get("source_path") == target:
                    return ([SimpleNamespace(payload=payload)], None)
        return ([], None)

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


class DummyEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, list):
            return [[float(idx + 1)] for idx, _ in enumerate(texts)]
        return [0.1]

    def get_sentence_embedding_dimension(self):  # pragma: no cover - deterministic
        return 1


def _build_agent_nick(s3_client, qdrant_client):
    settings = SimpleNamespace(
        static_policy_collection_name="static_policy",
        static_policy_s3_prefix="Static Policy/",
        static_policy_auto_ingest=True,
        s3_bucket_name="procwisemvp",
    )
    return SimpleNamespace(
        settings=settings,
        s3_client=s3_client,
        qdrant_client=qdrant_client,
        embedding_model=DummyEmbedder(),
    )


def test_static_policy_loader_ingests_and_skips():
    body = b"Procurement Policy\nSection 1\nClause A"
    entries = [
        {
            "Key": "Static Policy/policy.txt",
            "Size": len(body),
            "ETag": '"etag-123"',
            "LastModified": datetime(2024, 1, 1, 0, 0),
            "Body": body,
        }
    ]
    s3_client = DummyS3Client("procwisemvp", entries)
    qdrant_client = DummyQdrantClient()
    agent_nick = _build_agent_nick(s3_client, qdrant_client)

    loader = StaticPolicyLoader(agent_nick)
    summary = loader.sync_static_policy()

    assert summary["ingested"] == 1
    assert qdrant_client.upserts
    point_payloads = [point.payload for point in qdrant_client.upserts[0]["points"]]
    assert all(payload["document_type"] == "policy" for payload in point_payloads)
    assert all(payload["official_policy"] for payload in point_payloads)
    assert {payload["collection_name"] for payload in point_payloads} == {"static_policy"}

    second_summary = loader.sync_static_policy()
    assert second_summary["skipped"] == 1
    assert not qdrant_client.deleted


def test_static_policy_loader_force_refresh_triggers_delete():
    body = b"Policy Update\nSection A\nClause 1"
    entries = [
        {
            "Key": "Static Policy/policy.txt",
            "Size": len(body),
            "ETag": '"etag-456"',
            "LastModified": datetime(2024, 2, 2, 0, 0),
            "Body": body,
        }
    ]
    s3_client = DummyS3Client("procwisemvp", entries)
    qdrant_client = DummyQdrantClient()
    agent_nick = _build_agent_nick(s3_client, qdrant_client)

    loader = StaticPolicyLoader(agent_nick)
    loader.sync_static_policy()
    assert not qdrant_client.deleted

    forced_summary = loader.sync_static_policy(force=True)
    assert forced_summary["ingested"] == 1
    assert qdrant_client.deleted
