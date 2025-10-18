"""Conversation memory helpers for ProcWise agents.

This module maintains a thin layer over :class:`services.rag_service.RAGService`
so agents can persist and retrieve snippets of conversation history.  The
service is intentionally defensive - when Qdrant, embedding models or the RAG
stack are unavailable the helpers degrade gracefully, returning empty
responses instead of raising exceptions.  This keeps unit tests isolated from
vector infrastructure while ensuring production environments gain richer
context windows for agents.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import models

from services import rag_service as rag_module

logger = logging.getLogger(__name__)


def _safe_strip(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > 2000:
        text = text[:1997] + "..."
    return text


@dataclass
class RetrievedMessage:
    """Normalised representation of a retrieved conversation fragment."""

    content: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationMemoryService:
    """Persist and retrieve conversational snippets for workflow context."""

    def __init__(
        self,
        agent_nick: Any,
        *,
        rag_service: Optional[rag_module.RAGService] = None,
    ) -> None:
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        self._rag_service = rag_service
        self._seen_message_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_rag(self) -> Optional[rag_module.RAGService]:
        rag = self._rag_service
        if rag is not None:
            return rag
        try:
            rag = rag_module.RAGService(self.agent_nick)
        except Exception:  # pragma: no cover - dependency failures handled gracefully
            logger.debug("Conversation memory failed to initialise RAGService", exc_info=True)
            return None
        self._rag_service = rag
        return rag

    def _build_metadata(
        self,
        workflow_id: Optional[str],
        message_id: str,
        entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        speaker = entry.get("speaker") or entry.get("from_address")
        metadata = {
            "document_type": "conversation_message",
            "workflow_id": workflow_id,
            "message_id": message_id,
            "speaker": speaker,
        }
        if entry.get("document_origin"):
            metadata["document_origin"] = entry.get("document_origin")
        if entry.get("rfq_id"):
            metadata["rfq_id"] = entry.get("rfq_id")
        if entry.get("supplier_id"):
            metadata["supplier_id"] = entry.get("supplier_id")
        if entry.get("negotiation_round"):
            metadata["negotiation_round"] = entry.get("negotiation_round")
        metadata["record_id"] = f"conversation::{workflow_id or 'global'}::{message_id}"
        return metadata

    def _format_entry(self, entry: Dict[str, Any]) -> str:
        speaker = entry.get("speaker") or entry.get("from_address") or entry.get("role")
        subject = entry.get("subject")
        round_info = entry.get("negotiation_round") or entry.get("round")
        parts: List[str] = []
        if speaker:
            parts.append(f"Speaker: {_safe_strip(speaker)}")
        if subject:
            parts.append(f"Subject: {_safe_strip(subject)}")
        if round_info is not None:
            parts.append(f"Round: {round_info}")
        origin = entry.get("document_origin")
        if origin:
            parts.append(f"Origin: {_safe_strip(origin)}")
        message = (
            entry.get("message_body")
            or entry.get("message")
            or entry.get("negotiation_message")
            or entry.get("summary")
        )
        if isinstance(message, (list, tuple)):
            message = " ".join(str(item) for item in message if item)
        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if message:
            parts.append(_safe_strip(message))
        supplier_output = entry.get("supplier_output")
        if isinstance(supplier_output, dict):
            summary = json.dumps(supplier_output, ensure_ascii=False)
            parts.append(f"Supplier output: {summary}")
        negotiation_output = entry.get("negotiation_output")
        if isinstance(negotiation_output, dict):
            summary = json.dumps(negotiation_output, ensure_ascii=False)
            parts.append(f"Negotiation output: {summary}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(
        self,
        workflow_id: Optional[str],
        entries: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rag = self._resolve_rag()
        if rag is None:
            return []

        stored_metadata: List[Dict[str, Any]] = []
        payloads_to_upsert: List[Dict[str, Any]] = []
        dedupe_keys: List[str] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            message_id = _safe_strip(entry.get("message_id") or str(uuid.uuid4()))
            if not message_id:
                message_id = str(uuid.uuid4())
            dedupe_key = f"{workflow_id}:{message_id}"
            if dedupe_key in self._seen_message_ids:
                continue
            text_for_embedding = self._format_entry(entry)
            if not text_for_embedding:
                continue
            metadata = self._build_metadata(workflow_id, message_id, entry)
            full_payload = {**entry, **metadata, "content": text_for_embedding}
            payloads_to_upsert.append(full_payload)
            stored_metadata.append(metadata)
            dedupe_keys.append(dedupe_key)

        if not payloads_to_upsert:
            return stored_metadata

        try:
            rag.upsert_payloads(payloads_to_upsert, text_representation_key="content")
        except Exception:  # pragma: no cover - underlying storage errors already logged
            logger.exception("Failed to store conversation entry for workflow %s", workflow_id)
            return []

        for dedupe_key in dedupe_keys:
            self._seen_message_ids.add(dedupe_key)
        return stored_metadata

    def retrieve(
        self,
        workflow_id: Optional[str],
        query: str,
        *,
        limit: int = 5,
    ) -> List[RetrievedMessage]:
        rag = self._resolve_rag()
        if rag is None:
            return []
        query = _safe_strip(query)
        if not query:
            return []

        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="document_type", match=models.MatchValue(value="conversation_message")
            )
        ]
        if workflow_id:
            must_conditions.append(
                models.FieldCondition(
                    key="workflow_id", match=models.MatchValue(value=workflow_id)
                )
            )
        query_filter = models.Filter(must=must_conditions)
        try:
            hits = rag.search(query, top_k=limit, filters=query_filter)
        except Exception:  # pragma: no cover - defensive search fallback
            logger.exception("Conversation memory retrieval failed")
            return []

        results: List[RetrievedMessage] = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            content = payload.get("content") or payload.get("summary")
            if not content:
                continue
            score = getattr(hit, "score", None)
            results.append(
                RetrievedMessage(
                    content=_safe_strip(content),
                    score=float(score) if score is not None else None,
                    metadata=payload,
                )
            )
        return results

