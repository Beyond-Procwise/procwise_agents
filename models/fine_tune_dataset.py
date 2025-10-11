"""Utilities for assembling fine-tuning datasets from workflow learnings.

The :mod:`services.learning_repository` already captures a compact, structured
representation of high quality workflow outcomes inside Qdrant.  This module
bridges those summaries with supervised fine-tuning by transforming the
metadata and associated retrieval context into prompt/response pairs that can
be fed directly into a LoRA/QLoRA training loop.  The helpers are intentionally
framework agnostic so that datasets can be consumed by HuggingFace ``Trainer``
pipelines, lightweight JSONL exporters, or custom model adaptation scripts.

The builder focuses on negotiation learnings for now because they contain rich
metadata (strategy, counter, lead-time requests) that map cleanly to a JSON
supervision target.  Additional builders can be layered on later for email
drafting or extraction agents once those workflows record higher fidelity
outputs in the learning store.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FineTuneSample:
    """Simple container capturing a single fine-tuning record."""

    prompt: str
    completion: str
    metadata: Dict[str, Any]


def _normalise_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


class FineTuneDatasetBuilder:
    """Build prompt/response pairs from learning repository snapshots."""

    def __init__(
        self,
        learning_repository: Any,
        *,
        rag_service: Optional[Any] = None,
        max_context_snippets: int = 3,
    ) -> None:
        self.learning_repository = learning_repository
        self.rag_service = rag_service
        self.max_context_snippets = max(1, int(max_context_snippets or 1))

    # ------------------------------------------------------------------
    # Negotiation learnings
    # ------------------------------------------------------------------
    def build_negotiation_samples(
        self,
        *,
        limit: int = 50,
        workflow_id: Optional[str] = None,
    ) -> List[FineTuneSample]:
        """Convert negotiation learnings into JSON-supervised records."""

        if self.learning_repository is None:
            return []

        try:
            records = self.learning_repository.get_recent_learnings(
                workflow_id=workflow_id, limit=limit
            )
        except Exception:
            return []

        samples: List[FineTuneSample] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if record.get("event_type") != "negotiation_round":
                continue

            metadata = dict(record.get("metadata") or {})
            if not metadata:
                continue

            prompt = self._compose_negotiation_prompt(record, metadata)
            completion = self._compose_negotiation_completion(metadata)
            if not prompt or not completion:
                continue
            samples.append(
                FineTuneSample(prompt=prompt, completion=completion, metadata=metadata)
            )

        return samples

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def export_jsonl(self, samples: Iterable[FineTuneSample], output_path: Path) -> Path:
        """Persist a sequence of samples to JSONL for training frameworks."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                payload = {
                    "prompt": sample.prompt,
                    "completion": sample.completion,
                    "metadata": sample.metadata,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compose_negotiation_prompt(
        self, record: Dict[str, Any], metadata: Dict[str, Any]
    ) -> str:
        metadata_lines = [
            f"- {key}: {_normalise_value(metadata[key]).strip()}"
            for key in sorted(metadata.keys())
            if _normalise_value(metadata[key]).strip()
        ]
        if not metadata_lines:
            return ""

        context_lines = self._retrieve_context(metadata)
        if not context_lines:
            context_lines = ["- No matching retrievals"]

        header = record.get("summary") or "Negotiation learning captured"
        metadata_block = "\n".join(metadata_lines)
        context_block = "\n".join(context_lines)
        prompt = textwrap.dedent(
            f"""
            ### Negotiation Snapshot
            {header}

            ### Metadata
            {metadata_block}

            ### Retrieved Context
            {context_block}

            ### Task
            Produce a JSON object describing the next negotiation action with the keys
            strategy, counter_price, target_price, asks, lead_time_request,
            awaiting_response, and supplier_reply_registered. Ground the response in
            the metadata and retrieved insights when available.
            """
        ).strip()
        return prompt

    def _compose_negotiation_completion(self, metadata: Dict[str, Any]) -> str:
        keys = [
            "strategy",
            "counter_price",
            "target_price",
            "asks",
            "lead_time_request",
            "awaiting_response",
            "supplier_reply_registered",
        ]
        payload = {
            key: metadata[key]
            for key in keys
            if key in metadata and metadata[key] is not None
        }
        if not payload:
            return ""
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _retrieve_context(self, metadata: Dict[str, Any]) -> List[str]:
        if self.rag_service is None:
            return []

        query_terms = [
            str(metadata.get(key)).strip()
            for key in (
                "rfq_id",
                "supplier_id",
                "supplier_name",
                "strategy",
                "target_price",
                "counter_price",
            )
            if metadata.get(key)
        ]
        query = " ".join(term for term in query_terms if term)
        if not query:
            query = "negotiation learning"

        try:
            hits = self.rag_service.search(
                query,
                top_k=self.max_context_snippets,
                filters={"document_type": "learning"},
            )
        except Exception:
            return []

        context_lines: List[str] = []
        for hit in hits or []:
            payload = getattr(hit, "payload", None)
            score = getattr(hit, "score", None)
            if isinstance(payload, dict):
                snippet = payload.get("content") or payload.get("summary")
            else:
                snippet = None
            if not snippet:
                continue
            if score is not None:
                context_lines.append(f"- ({score:.2f}) {snippet}")
            else:
                context_lines.append(f"- {snippet}")
        return context_lines


def export_samples_jsonl(
    samples: Iterable[FineTuneSample], output_path: Path
) -> Path:
    """Convenience wrapper mirroring :meth:`FineTuneDatasetBuilder.export_jsonl`."""

    builder = FineTuneDatasetBuilder(learning_repository=None)
    return builder.export_jsonl(samples, output_path)


__all__ = [
    "FineTuneSample",
    "FineTuneDatasetBuilder",
    "export_samples_jsonl",
]

