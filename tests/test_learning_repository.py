import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.learning_repository import LearningRepository


class DummyEmbedder:
    def encode(self, texts, **kwargs):
        count = len(texts)
        return np.ones((count, 4), dtype=np.float32)


class FakeQdrantClient:
    def __init__(self):
        self.collections = {}
        self.upserts = []
        self.scroll_batches = []

    def get_collection(self, collection_name):
        if collection_name not in self.collections:
            raise Exception("missing")
        return SimpleNamespace(payload_schema=self.collections[collection_name])

    def create_collection(self, collection_name, vectors_config):  # pragma: no cover - invoked once
        self.collections[collection_name] = {}

    def create_payload_index(self, collection_name, field_name, field_schema, wait=True):
        self.collections.setdefault(collection_name, {})[field_name] = field_schema

    def upsert(self, collection_name, points, wait=True):
        self.upserts.append((collection_name, points))

    def search(self, *args, **kwargs):  # pragma: no cover - not exercised in tests
        return []

    def scroll(
        self,
        collection_name,
        scroll_filter=None,
        limit=10,
        offset=None,
        with_payload=True,
        with_vectors=False,
    ):
        batch = list(self.scroll_batches)
        self.scroll_batches = []
        return batch, None


def _build_repo():
    client = FakeQdrantClient()
    repo = LearningRepository(
        SimpleNamespace(
            settings=SimpleNamespace(
                learning_collection_name="learning", vector_size=4
            ),
            qdrant_client=client,
            embedding_model=DummyEmbedder(),
        )
    )
    return repo, client


def test_record_email_learning_filters_sensitive_metadata():
    repo, client = _build_repo()
    draft = {
        "rfq_id": "RFQ-001",
        "supplier_id": "SUP-1",
        "subject": "Test",
        "body": "<p>internal analysis</p>",
        "contact_level": 1,
        "recipients": ["test@example.com"],
    }
    context = {
        "intent": "RFQ_DRAFT",
        "document_origin": "digital",
        "target_price": 12.5,
    }

    repo.record_email_learning(workflow_id="wf-1", draft=draft, context=context)

    assert client.upserts, "expected learning to be upserted"
    _, points = client.upserts[-1]
    payload = points[0].payload
    metadata = payload["metadata"]
    assert "body" not in metadata
    assert metadata["intent"] == "RFQ_DRAFT"
    assert metadata["document_origin"] == "digital"
    assert "analysis" not in payload["summary"].lower()


def test_record_negotiation_learning_handles_scanned_and_digital_samples():
    repo, client = _build_repo()
    draft = {
        "rfq_id": "RFQ-002",
        "supplier_id": "SUP-2",
        "subject": "Update",
        "contact_level": 0,
        "recipients": [],
    }

    for origin in ("scanned", "scanned", "digital", "digital"):
        context = {"document_origin": origin, "intent": "RFQ_DRAFT"}
        repo.record_email_learning(workflow_id="wf-2", draft=draft, context=context)

    origins = [
        point.payload["metadata"].get("document_origin")
        for _, points in client.upserts
        for point in points
    ]
    assert origins.count("scanned") == 2
    assert origins.count("digital") == 2


def test_build_context_snapshot_reads_scroll_results():
    repo, client = _build_repo()
    client.scroll_batches = [
        SimpleNamespace(
            payload={
                "document_type": "learning",
                "summary": "Source: email_drafting_agent",
                "workflow_id": "wf-9",
            }
        )
    ]

    snapshot = repo.build_context_snapshot(workflow_id="wf-9", limit=1)

    assert snapshot is not None
    assert snapshot["count"] == 1
    assert snapshot["events"][0]["workflow_id"] == "wf-9"


def test_record_model_plan_upserts_vectorised_payload():
    repo, client = _build_repo()

    metadata = {"plan_version": "v1", "dataset_counts": {"sft": 320, "dpo": 240}}
    plan_text = "Step 1: curate SFT data. Step 2: run DPO."

    point_id = repo.record_model_plan(
        model_name="phi4-joshi",
        plan_text=plan_text,
        plan_metadata=metadata,
        tags=["humanization", "phi4"],
    )

    assert point_id is not None
    assert client.upserts, "expected model plan to be persisted"
    _, points = client.upserts[-1]
    payload = points[0].payload

    assert payload["document_type"] == "model_plan"
    assert payload["model_name"] == "phi4-joshi"
    assert payload["plan_text"] == plan_text
    assert sorted(payload["tags"]) == ["humanization", "phi4"]
    assert payload["metadata"]["plan_version"] == "v1"
    assert payload["metadata"]["model_name"] == "phi4-joshi"


def test_fetch_model_plans_returns_sorted_payloads():
    repo, client = _build_repo()

    newer = SimpleNamespace(
        payload={
            "document_type": "model_plan",
            "model_name": "phi4-joshi",
            "plan_text": "newer",
            "created_at": "2024-11-12T10:00:00+00:00",
        }
    )
    older = SimpleNamespace(
        payload={
            "document_type": "model_plan",
            "model_name": "phi4-joshi",
            "plan_text": "older",
            "created_at": "2024-11-11T10:00:00+00:00",
        }
    )
    unrelated = SimpleNamespace(
        payload={
            "document_type": "model_plan",
            "model_name": "other-model",
            "plan_text": "skip",
            "created_at": "2024-11-10T10:00:00+00:00",
        }
    )
    client.scroll_batches = [newer, older, unrelated]

    plans = repo.fetch_model_plans(model_name="phi4-joshi", limit=2)

    assert [plan["plan_text"] for plan in plans] == ["newer", "older"]
