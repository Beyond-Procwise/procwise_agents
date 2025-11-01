import os
import sys
from datetime import datetime
from types import SimpleNamespace


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.static_policy_loader import StaticPolicyLoader  # noqa: E402


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


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(text: str) -> bytes:
    escaped = _escape_pdf_text(text)
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET\n".encode("latin-1")

    objects = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        (
            4,
            b"<< /Length %d >>\n" % len(stream)
            + b"stream\n"
            + stream
            + b"endstream",
        ),
        (5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]

    header = b"%PDF-1.4\n"
    body_parts = []
    offsets = []
    cursor = len(header)
    for obj_number, payload in objects:
        entry = (
            f"{obj_number} 0 obj\n".encode("latin-1")
            + payload
            + b"\nendobj\n"
        )
        body_parts.append(entry)
        offsets.append(cursor)
        cursor += len(entry)

    xref_offset = cursor
    xref_header = f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    xref_entries = [b"0000000000 65535 f \n"]
    for offset in offsets:
        xref_entries.append(f"{offset:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1")
        + f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )

    return header + b"".join(body_parts) + xref_header + b"".join(xref_entries) + trailer


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


def test_static_policy_loader_ingests_pdf_with_pdfplumber_fallback():
    pdf_bytes = _build_simple_pdf("Policy Clause 1")
    entries = [
        {
            "Key": "Static Policy/policy.pdf",
            "Size": len(pdf_bytes),
            "ETag": '"etag-pdf"',
            "LastModified": datetime(2024, 3, 3, 0, 0),
            "Body": pdf_bytes,
        }
    ]
    s3_client = DummyS3Client("procwisemvp", entries)
    qdrant_client = DummyQdrantClient()
    agent_nick = _build_agent_nick(s3_client, qdrant_client)

    loader = StaticPolicyLoader(agent_nick)
    summary = loader.sync_static_policy(force=True)

    assert summary["ingested"] == 1
    assert qdrant_client.upserts
    payloads = [point.payload for point in qdrant_client.upserts[-1]["points"]]
    assert all(payload["extraction_method"] == "pdfplumber_fallback" for payload in payloads)
    assert all(payload.get("pdf_page_count") == 1 for payload in payloads)
    assert any("Clause 1" in payload["content"] for payload in payloads)
