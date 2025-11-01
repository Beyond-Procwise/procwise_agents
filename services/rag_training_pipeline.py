"""Layered training pipeline for the ProcWise RAG agent.

The trainer orchestrates a multi-stage curriculum:

Layer 0
    Ensure collections and baseline retrieval settings are in place so the
    agent always has a healthy Qdrant foundation.
Layer 1
    Ingest new procurement documents (POs, RFQs, contracts, invoices) while
    removing sensitive identifiers before embedding them with
    :class:`services.rag_service.RAGService`.
Layer 2
    Capture instruction-following samples into an instruction-tuning dataset
    that downstream QLoRA/LoRA jobs can consume.
Layer 3
    Update preference weights so retrieval results reflect human-in-the-loop
    preferences gathered from negotiation and sourcing reviews.
Layer 4
    Refresh tone and citation guidelines so answers stay conversational,
    cite sources, and gracefully decline when knowledge is unavailable.

The pipeline runs inside the model training dispatch endpoint, keeping agent
learning aligned with existing scheduler and governance hooks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from models.context_trainer import ConversationDatasetWriter
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class LayeredRAGTrainer:
    """Coordinate layered improvements for the ProcWise RAG agent."""

    _SENSITIVE_KEYS = {
        "supplier_id",
        "rfq_id",
        "invoice_id",
        "po_id",
        "contract_id",
        "workflow_id",
        "quote_id",
    }

    def __init__(self, agent_nick, rag_service: Optional[RAGService] = None) -> None:
        self.agent_nick = agent_nick
        self.rag_service = rag_service or RAGService(agent_nick)
        self.training_dir = (
            Path(__file__).resolve().parent.parent
            / "resources"
            / "training"
            / "rag"
        )
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir = self.training_dir / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.instruction_dataset_path = self.training_dir / "instruction_dataset.jsonl"
        self.preference_path = self.training_dir / "preference_weights.json"
        self.citation_path = self.training_dir / "citation_guidelines.json"
        self._conversation_writer = ConversationDatasetWriter(
            str(self.training_dir / "instruction_conversations")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all training layers and return a structured summary."""

        payload = payload or {}
        logger.info("Starting layered RAG training with payload keys: %s", list(payload.keys()))

        summary: Dict[str, Any] = {
            "layer0": self._run_layer_zero(),
            "layer1": self._run_layer_one(
                payload.get("documents"), payload.get("corpus_path")
            ),
            "layer2": self._run_layer_two(payload.get("instruction_examples")),
            "layer3": self._run_layer_three(payload.get("preference_pairs")),
            "layer4": self._run_layer_four(payload.get("citation_examples")),
        }

        # Reload preference weights so subsequent retrieval calls immediately
        # honour the updated calibration.
        try:
            self.rag_service.reload_preference_weights()
        except Exception:  # pragma: no cover - defensive logging only
            logger.exception("Failed to reload RAG preference weights after training")

        logger.info("Layered RAG training completed: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------
    def _run_layer_zero(self) -> Dict[str, Any]:
        """Ensure Qdrant collections exist for all procurement corpora."""

        collections = {
            "primary": self.rag_service.primary_collection,
            "uploaded": self.rag_service.uploaded_collection,
            "static_policy": getattr(
                self.rag_service, "static_policy_collection", None
            ),
            "learning": self.rag_service.learning_collection,
        }
        ensured: Dict[str, bool] = {}
        for label, name in collections.items():
            if not name:
                continue
            try:
                self.rag_service.ensure_collection(name)
            except Exception:
                ensured[label] = False
                logger.exception("Failed to ensure Qdrant collection %s", name)
            else:
                ensured[label] = True
        ensured["timestamp"] = datetime.utcnow().isoformat()
        return ensured

    def _run_layer_one(
        self,
        documents: Optional[Iterable[Any]],
        corpus_path: Optional[str],
    ) -> Dict[str, Any]:
        """Sanitise and ingest procurement documents into the RAG store."""

        normalised: List[Dict[str, Any]] = []

        for entry in documents or []:
            doc = self._normalise_document(entry)
            if doc:
                normalised.append(doc)

        if corpus_path:
            corpus_root = Path(corpus_path)
            if corpus_root.exists():
                for path in corpus_root.rglob("*.json"):
                    try:
                        content = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        logger.exception("Failed to parse corpus file %s", path)
                        continue
                    if isinstance(content, list):
                        for item in content:
                            doc = self._normalise_document(item)
                            if doc:
                                normalised.append(doc)
                    else:
                        doc = self._normalise_document(content)
                        if doc:
                            normalised.append(doc)
                for path in corpus_root.rglob("*.txt"):
                    try:
                        text = path.read_text(encoding="utf-8")
                    except Exception:
                        logger.exception("Failed to read corpus text %s", path)
                        continue
                    doc = self._normalise_document({"content": text, "metadata": {"document_type": path.suffix.strip('.') or "text"}})
                    if doc:
                        normalised.append(doc)

        ingested_count = 0
        storage_path: Optional[str] = None
        if normalised:
            try:
                self.rag_service.upsert_payloads(normalised)
                ingested_count = len(normalised)
                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                output_path = self.corpus_dir / f"corpus-{timestamp}.jsonl"
                with output_path.open("w", encoding="utf-8") as handle:
                    for doc in normalised:
                        handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
                storage_path = str(output_path)
            except Exception:
                logger.exception("Failed to upsert procurement documents into RAG")

        return {
            "ingested_documents": ingested_count,
            "stored_dataset": storage_path,
        }

    def _run_layer_two(self, instruction_examples: Optional[Iterable[Any]]) -> Dict[str, Any]:
        """Persist instruction-following samples for downstream fine-tuning."""

        if not instruction_examples:
            return {
                "records": 0,
                "dataset_path": str(self.instruction_dataset_path),
            }

        existing_prompts: set[str] = set()
        if self.instruction_dataset_path.exists():
            for line in self.instruction_dataset_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                prompt = record.get("prompt")
                if isinstance(prompt, str):
                    existing_prompts.add(prompt)

        appended = 0
        with self.instruction_dataset_path.open("a", encoding="utf-8") as handle:
            for example in instruction_examples:
                prompt, completion = self._extract_instruction_pair(example)
                if not prompt or not completion or prompt in existing_prompts:
                    continue
                payload = {
                    "prompt": prompt,
                    "completion": completion,
                    "metadata": {"captured_on": datetime.utcnow().isoformat()},
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._conversation_writer.write_record(
                    context_text=prompt,
                    response_text=completion,
                    metadata={"source": "layered_rag_instruction"},
                )
                existing_prompts.add(prompt)
                appended += 1

        return {
            "records": appended,
            "dataset_path": str(self.instruction_dataset_path),
        }

    def _run_layer_three(
        self, preference_pairs: Optional[Iterable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Update retrieval preference weights using preference comparisons."""

        weights = self._load_preference_weights()
        updated_collections: set[str] = set()
        updated_document_types: set[str] = set()

        for pair in preference_pairs or []:
            preferred = self._normalise_preference(pair.get("preferred"))
            rejected = self._normalise_preference(pair.get("rejected"))
            if preferred:
                collection = preferred.get("collection_name")
                doc_type = preferred.get("document_type")
                if collection:
                    weights["collection"][collection] = round(
                        weights["collection"].get(collection, 0.0) + 0.02, 4
                    )
                    updated_collections.add(collection)
                if doc_type:
                    weights["document_type"][doc_type] = round(
                        weights["document_type"].get(doc_type, 0.0) + 0.01, 4
                    )
                    updated_document_types.add(doc_type)
            if rejected:
                collection = rejected.get("collection_name")
                doc_type = rejected.get("document_type")
                if collection:
                    weights["collection"][collection] = max(
                        0.0, round(weights["collection"].get(collection, 0.0) - 0.015, 4)
                    )
                    updated_collections.add(collection)
                if doc_type:
                    weights["document_type"][doc_type] = max(
                        0.0,
                        round(weights["document_type"].get(doc_type, 0.0) - 0.008, 4),
                    )
                    updated_document_types.add(doc_type)

        try:
            self.preference_path.write_text(
                json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to persist updated preference weights")

        return {
            "updated_collections": sorted(updated_collections),
            "updated_document_types": sorted(updated_document_types),
            "preference_path": str(self.preference_path),
        }

    def _run_layer_four(
        self, citation_examples: Optional[Iterable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Refresh tone and citation guidelines for structured answers."""

        guidelines = self._load_citation_guidelines()

        for example in citation_examples or []:
            if not isinstance(example, dict):
                continue
            acknowledgement = example.get("acknowledgement")
            summary_intro = example.get("summary_intro")
            actions_lead = example.get("actions_lead")
            fallback = example.get("fallback")
            follow_ups = example.get("follow_ups")

            if isinstance(acknowledgement, str) and acknowledgement.strip():
                guidelines.setdefault("acknowledgements", [])
                if acknowledgement not in guidelines["acknowledgements"]:
                    guidelines["acknowledgements"].append(acknowledgement.strip())
            if isinstance(summary_intro, str) and summary_intro.strip():
                guidelines["summary_intro"] = summary_intro.strip()
            if isinstance(actions_lead, str) and actions_lead.strip():
                guidelines["actions_lead"] = actions_lead.strip()
            if isinstance(fallback, str) and fallback.strip():
                guidelines["fallback"] = fallback.strip()
            if isinstance(follow_ups, list):
                cleaned = [str(item).strip() for item in follow_ups if str(item).strip()]
                if cleaned:
                    existing = guidelines.setdefault("default_follow_ups", [])
                    for item in cleaned:
                        if item not in existing:
                            existing.append(item)

        try:
            self.citation_path.write_text(
                json.dumps(guidelines, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to persist citation guidelines")

        return {
            "acknowledgements": guidelines.get("acknowledgements", []),
            "summary_intro": guidelines.get("summary_intro"),
            "fallback": guidelines.get("fallback"),
            "citation_path": str(self.citation_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_document(self, entry: Any) -> Optional[Dict[str, Any]]:
        if entry is None:
            return None
        if isinstance(entry, str):
            text = entry
            metadata: Dict[str, Any] = {}
        elif isinstance(entry, dict):
            text = entry.get("text") or entry.get("content")
            metadata = entry.get("metadata") or entry.get("payload") or {}
            if not text and isinstance(entry.get("pages"), list):
                text = "\n".join(str(page) for page in entry["pages"] if page)
        else:
            return None

        if not isinstance(text, str) or not text.strip():
            return None

        metadata = self._sanitize_metadata(metadata)
        return {"content": text, "payload": metadata}

    def _sanitize_metadata(self, metadata: Any) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        clean: Dict[str, Any] = {}
        for key, value in metadata.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if any(sensitive in key_lower for sensitive in self._SENSITIVE_KEYS):
                continue
            clean[key_str] = value
        return clean

    def _extract_instruction_pair(self, example: Any) -> tuple[str, str]:
        if not isinstance(example, dict):
            return "", ""
        prompt = example.get("prompt") or example.get("question") or ""
        completion = example.get("completion") or example.get("answer") or ""
        return prompt.strip(), completion.strip()

    def _load_preference_weights(self) -> Dict[str, Dict[str, float]]:
        defaults = {
            "collection": {
                self.rag_service.primary_collection: 0.12,
                self.rag_service.uploaded_collection: 0.08,
            },
            "document_type": {"policy": 0.1},
        }
        static_collection = getattr(self.rag_service, "static_policy_collection", None)
        if static_collection:
            defaults["collection"][static_collection] = 0.14
        learning_collection = getattr(self.rag_service, "learning_collection", None)
        if learning_collection:
            defaults["collection"].setdefault(learning_collection, 0.02)
        if not self.preference_path.exists():
            return defaults
        try:
            loaded = json.loads(self.preference_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Unable to parse existing preference weights; using defaults")
            return defaults
        if not isinstance(loaded, dict):
            return defaults
        result = {
            "collection": dict(defaults["collection"]),
            "document_type": dict(defaults["document_type"]),
        }
        for key in ("collection", "document_type"):
            section = loaded.get(key)
            if isinstance(section, dict):
                for sub_key, value in section.items():
                    if isinstance(sub_key, str) and isinstance(value, (int, float)):
                        result[key][sub_key] = float(value)
        return result

    def _normalise_preference(self, preference: Any) -> Dict[str, str]:
        if not isinstance(preference, dict):
            return {}
        collection = preference.get("collection_name")
        doc_type = preference.get("document_type")
        metadata = self._sanitize_metadata(preference)
        if isinstance(collection, str):
            metadata["collection_name"] = collection
        if isinstance(doc_type, str):
            metadata["document_type"] = doc_type.lower()
        return metadata

    def _load_citation_guidelines(self) -> Dict[str, Any]:
        defaults = {
            "acknowledgements": [
                "Thanks for flagging this — here's what I can confirm.",
                "Appreciate the context. Here's the current view.",
            ],
            "summary_intro": "Current highlights from the knowledge base:",
            "actions_lead": "Let me know if you want me to escalate, refresh the data, or prep outreach notes.",
            "fallback": "I do not have that information as per my knowledge.",
            "default_follow_ups": [
                "Would you like me to surface the related purchase order or contract details?",
                "Should I queue a supplier relationship summary for review?",
                "Do you want me to capture this question for the next procurement sync?",
            ],
        }
        if not self.citation_path.exists():
            return defaults
        try:
            loaded = json.loads(self.citation_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Unable to parse existing citation guidelines; using defaults")
            return defaults
        if isinstance(loaded, dict):
            defaults.update(loaded)
        return defaults


__all__ = ["LayeredRAGTrainer"]
