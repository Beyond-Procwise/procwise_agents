import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fine_tune_dataset import (
    FineTuneDatasetBuilder,
    FineTuneSample,
    export_samples_jsonl,
)


class DummyLearningRepository:
    def __init__(self, records):
        self._records = records
        self.calls = []

    def get_recent_learnings(self, *, workflow_id=None, limit=20):
        self.calls.append({"workflow_id": workflow_id, "limit": limit})
        return list(self._records)


class DummyRAGService:
    def __init__(self, hits):
        self._hits = hits
        self.queries = []

    def search(self, query, top_k=3, filters=None):
        self.queries.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self._hits)


def _negotiation_record(**overrides):
    metadata = {
        "rfq_id": "RFQ-1001",
        "supplier_id": "SUP-9",
        "supplier_name": "Acme Components",
        "strategy": "counter",
        "counter_price": "95.00",
        "target_price": "90.00",
        "asks": ["expedite shipping"],
        "lead_time_request": "5 days",
        "awaiting_response": True,
        "supplier_reply_registered": False,
    }
    metadata.update(overrides.pop("metadata", {}))
    record = {
        "summary": "Counter proposal recommended for RFQ-1001",
        "event_type": "negotiation_round",
        "metadata": metadata,
    }
    record.update(overrides)
    return record


def test_builder_generates_samples_with_context(tmp_path):
    hits = [
        SimpleNamespace(
            payload={"content": "Previous counter at 97.00 accepted", "document_type": "learning"},
            score=0.91,
        ),
        SimpleNamespace(
            payload={"summary": "Supplier delivered 5 days early last cycle"},
            score=0.73,
        ),
    ]
    repo = DummyLearningRepository([_negotiation_record()])
    rag = DummyRAGService(hits)

    builder = FineTuneDatasetBuilder(repo, rag_service=rag, max_context_snippets=2)
    samples = builder.build_negotiation_samples(limit=5)

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, FineTuneSample)
    assert "RFQ-1001" in sample.prompt
    assert "Previous counter" in sample.prompt
    assert "Supplier delivered" in sample.prompt
    assert "counter_price" in sample.completion
    assert json.loads(sample.completion)["strategy"] == "counter"

    path = tmp_path / "negotiation.jsonl"
    builder.export_jsonl(samples, path)
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["prompt"] == sample.prompt
    assert payload["metadata"]["supplier_id"] == "SUP-9"

    assert rag.queries and rag.queries[0]["top_k"] == 2
    assert repo.calls and repo.calls[0]["limit"] == 5


def test_builder_handles_missing_context(tmp_path):
    repo = DummyLearningRepository([_negotiation_record(metadata={"strategy": "accept"})])
    builder = FineTuneDatasetBuilder(repo, rag_service=None)

    samples = builder.build_negotiation_samples(limit=1)
    assert len(samples) == 1
    sample = samples[0]
    assert "No matching retrievals" in sample.prompt
    assert json.loads(sample.completion)["strategy"] == "accept"

    alt_path = tmp_path / "dataset.jsonl"
    export_samples_jsonl(samples, alt_path)
    exported = json.loads(alt_path.read_text(encoding="utf-8").splitlines()[0])
    assert exported["completion"] == sample.completion


def test_builder_filters_non_negotiation_events():
    repo = DummyLearningRepository([
        _negotiation_record(),
        {"event_type": "email_draft_generated", "metadata": {"subject": "Hi"}},
        "invalid",
    ])
    builder = FineTuneDatasetBuilder(repo)

    samples = builder.build_negotiation_samples(limit=10)
    assert len(samples) == 1

