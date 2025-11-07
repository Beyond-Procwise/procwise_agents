"""Centralised repository for vectorised workflow learnings.

The repository maintains a dedicated Qdrant collection that stores metadata
based learnings produced across workflows.  Only structured metadata and
high-level summaries are embedded – raw email bodies, internal scoring logic
and any sensitive calculations are explicitly filtered out.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from qdrant_client import models


logger = logging.getLogger(__name__)


_EXCLUDED_METADATA_KEYS: tuple[str, ...] = (
    "body",
    "raw_body",
    "raw_email",
    "analysis",
    "score",
    "scores",
    "scoring",
    "calculation",
    "calculations",
    "formula",
    "explanation",
    "debug",
)


class LearningRepository:
    """Helper that coordinates the ``learning`` Qdrant collection."""

    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        self.qdrant_client = getattr(agent_nick, "qdrant_client", None)
        self.embedder = getattr(agent_nick, "embedding_model", None)
        self.collection_name: str = getattr(
            self.settings, "learning_collection_name", "learning"
        )
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def record_learning(
        self,
        *,
        workflow_id: Optional[str],
        source: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Persist a metadata learning point into Qdrant."""

        if self.qdrant_client is None or self.embedder is None:
            logger.debug("Learning repository disabled (missing client or embedder)")
            return None

        safe_metadata = self._normalise_metadata(metadata)
        if workflow_id:
            safe_metadata.setdefault("workflow_id", workflow_id)

        summary = self._build_summary(source, event_type, safe_metadata)
        if not summary:
            logger.debug("Skipping learning record for %s – empty summary", event_type)
            return None

        vector = self._encode_text(summary)
        if vector is None:
            return None

        payload: Dict[str, Any] = {
            "document_type": "learning",
            "source": source,
            "event_type": event_type,
            "workflow_id": workflow_id,
            "summary": summary,
            "metadata": safe_metadata,
        }

        if tags:
            cleaned_tags = [
                str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()
            ]
            if cleaned_tags:
                payload["tags"] = sorted(set(cleaned_tags))

        point_id = str(uuid.uuid4())
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)

        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            )
        except Exception:
            logger.exception("Failed to persist learning record for %s", event_type)
            return None

        return point_id

    def record_model_plan(
        self,
        *,
        model_name: str,
        plan_text: str,
        plan_metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Persist a lightweight fine-tuning plan for a model.

        The record is stored in the learning collection so that agents can
        retrieve the latest humanisation playbook directly from Qdrant.
        """

        if self.qdrant_client is None or self.embedder is None:
            logger.debug("Model plan repository disabled (missing client or embedder)")
            return None

        cleaned_plan = str(plan_text or "").strip()
        if not cleaned_plan:
            logger.debug("Skipping model plan for %s – empty body", model_name)
            return None

        metadata = self._normalise_metadata(plan_metadata)
        metadata["model_name"] = model_name

        vector = self._encode_text(cleaned_plan)
        if vector is None:
            return None

        payload: Dict[str, Any] = {
            "document_type": "model_plan",
            "model_name": model_name,
            "plan_text": cleaned_plan,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if tags:
            cleaned_tags = [
                str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()
            ]
            if cleaned_tags:
                payload["tags"] = sorted(set(cleaned_tags))

        point_id = str(uuid.uuid4())
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)

        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            )
        except Exception:
            logger.exception("Failed to persist model plan for %s", model_name)
            return None

        return point_id

    def record_email_learning(
        self,
        *,
        workflow_id: Optional[str],
        draft: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[str]:
        metadata = {
            "rfq_id": draft.get("rfq_id"),
            "supplier_id": draft.get("supplier_id"),
            "supplier_name": draft.get("supplier_name"),
            "subject": draft.get("subject"),
            "contact_level": draft.get("contact_level"),
            "recipients": draft.get("recipients"),
            "intent": context.get("intent") or draft.get("metadata", {}).get("intent"),
            "document_origin": context.get("document_origin"),
            "negotiation_round": draft.get("thread_index")
            or draft.get("metadata", {}).get("round"),
            "target_price": context.get("target_price"),
            "current_offer": context.get("current_offer"),
        }
        tags = [context.get("document_origin") or "digital"]
        return self.record_learning(
            workflow_id=workflow_id,
            source="email_drafting_agent",
            event_type="email_draft_generated",
            metadata=metadata,
            tags=tags,
        )

    def record_negotiation_learning(
        self,
        *,
        workflow_id: Optional[str],
        rfq_id: Optional[str],
        supplier_id: Optional[str],
        decision: Dict[str, Any],
        state: Dict[str, Any],
        awaiting_response: bool,
        supplier_reply_registered: bool,
    ) -> Optional[str]:
        metadata = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "strategy": decision.get("strategy"),
            "counter_price": decision.get("counter_price"),
            "target_price": decision.get("target_price"),
            "asks": decision.get("asks"),
            "lead_time_request": decision.get("lead_time_request"),
            "round": decision.get("round"),
            "awaiting_response": awaiting_response,
            "supplier_reply_count": state.get("supplier_reply_count"),
            "supplier_reply_registered": supplier_reply_registered,
        }
        return self.record_learning(
            workflow_id=workflow_id,
            source="negotiation_agent",
            event_type="negotiation_round",
            metadata=metadata,
            tags=["negotiation"],
        )

    def record_workflow_learning(
        self,
        *,
        workflow_id: str,
        workflow_name: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        opportunities = result.get("opportunities") or {}
        metadata = {
            "workflow_name": workflow_name,
            "status": result.get("status"),
            "total_suppliers": result.get("supplier_count"),
            "opportunity_categories": list(opportunities.keys()) if isinstance(opportunities, dict) else [],
        }
        return self.record_learning(
            workflow_id=workflow_id,
            source="workflow",
            event_type="workflow_complete",
            metadata=metadata,
            tags=[workflow_name or "workflow"],
        )

    def get_recent_learnings(
        self,
        *,
        workflow_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if self.qdrant_client is None:
            return []

        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="document_type", match=models.MatchValue(value="learning")
            )
        ]
        if workflow_id:
            must_conditions.append(
                models.FieldCondition(
                    key="workflow_id", match=models.MatchValue(value=workflow_id)
                )
            )

        query_filter = models.Filter(must=must_conditions)
        results: List[Dict[str, Any]] = []
        offset = None

        while len(results) < limit:
            batch_size = min(64, limit - len(results))
            try:
                batch, offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                logger.exception("Failed to scroll learning collection")
                break

            for point in batch or []:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    results.append(payload)
            if offset is None:
                break

        return results

    def build_context_snapshot(
        self, *, workflow_id: Optional[str], limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        records = self.get_recent_learnings(workflow_id=workflow_id, limit=limit)
        if not records:
            return None
        return {
            "workflow_id": workflow_id,
            "count": len(records),
            "events": records,
        }

    def fetch_model_plans(
        self,
        *,
        model_name: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored model plans for a given model name."""

        client = self.qdrant_client
        if client is None:
            return []

        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="document_type", match=models.MatchValue(value="model_plan")
            ),
            models.FieldCondition(
                key="model_name", match=models.MatchValue(value=model_name)
            ),
        ]

        query_filter = models.Filter(must=must_conditions)
        results: List[Dict[str, Any]] = []
        offset = None

        while len(results) < limit:
            batch_size = min(64, limit - len(results))
            try:
                batch, offset = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                logger.exception("Failed to scroll model plans for %s", model_name)
                break

            for point in batch or []:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict) and payload.get("document_type") == "model_plan":
                    results.append(payload)
            if offset is None:
                break

        results.sort(
            key=lambda item: item.get("created_at") or "",
            reverse=True,
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        client = self.qdrant_client
        if client is None:
            return
        try:
            collection = client.get_collection(collection_name=self.collection_name)
            schema = getattr(collection, "payload_schema", {}) or {}
            self._ensure_indexes(schema)
            return
        except Exception:
            logger.debug("Creating learning collection %s", self.collection_name)

        try:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=int(getattr(self.settings, "vector_size", 768)),
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception:
            logger.exception("Failed to create learning collection %s", self.collection_name)
            return

        try:
            client.get_collection(collection_name=self.collection_name)
            self._ensure_indexes({})
        except Exception:
            logger.exception("Failed to verify learning collection %s", self.collection_name)

    def _ensure_indexes(self, schema: Dict[str, Any]) -> None:
        client = self.qdrant_client
        if client is None:
            return
        existing = set(schema.keys()) if schema else set()
        required = {
            "document_type": models.PayloadSchemaType.KEYWORD,
            "source": models.PayloadSchemaType.KEYWORD,
            "event_type": models.PayloadSchemaType.KEYWORD,
            "workflow_id": models.PayloadSchemaType.KEYWORD,
            "rfq_id": models.PayloadSchemaType.KEYWORD,
            "supplier_id": models.PayloadSchemaType.KEYWORD,
            "source_type": models.PayloadSchemaType.KEYWORD,
            "model_name": models.PayloadSchemaType.KEYWORD,
        }

        for field, schema_type in required.items():
            if field in existing:
                continue
            try:
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                    wait=True,
                )
            except Exception:
                logger.debug("Payload index %s already exists", field, exc_info=True)

    def _normalise_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        if not isinstance(metadata, dict):
            return cleaned
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if any(excluded in lowered for excluded in _EXCLUDED_METADATA_KEYS):
                continue
            serialised = self._serialise_value(value)
            if serialised is None:
                continue
            cleaned[key] = serialised
        return cleaned

    def _serialise_value(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            collapsed = [self._serialise_value(item) for item in value]
            return [item for item in collapsed if item is not None]
        if isinstance(value, dict):
            return {
                str(key): self._serialise_value(val)
                for key, val in value.items()
                if self._serialise_value(val) is not None
            }
        text = str(value).strip()
        if not text:
            return None
        if len(text) > 512:
            text = text[:509] + "..."
        return text

    def _build_summary(self, source: str, event_type: str, metadata: Dict[str, Any]) -> str:
        parts = [f"Source: {source}", f"Event: {event_type}"]
        for key in sorted(metadata.keys()):
            value = metadata[key]
            if isinstance(value, (list, tuple)):
                display = ", ".join(str(item) for item in value[:5])
            elif isinstance(value, dict):
                display = json.dumps(value, ensure_ascii=False)[:200]
            else:
                display = str(value)
            parts.append(f"{key}: {display}")
        return "; ".join(parts)

    def _encode_text(self, text: str) -> Optional[List[float]]:
        if self.embedder is None:
            return None
        try:
            vectors = self.embedder.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            logger.exception("Failed to encode learning summary")
            return None
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 2:
            array = array[0]
        if array.ndim != 1:
            logger.debug("Unexpected embedding shape for learning summary: %s", array.shape)
            return None
        return array.astype(np.float32).tolist()

