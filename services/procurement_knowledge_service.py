"""Utilities for sourcing procurement-specific external knowledge for gpt-oss.

The service surfaces procurement market intelligence briefs that can be
embedded into Qdrant and retrieved by RAG-enabled agents.  The dataset is
limited to procurement topics and is sanitised to ensure no customer
information is introduced or manipulated.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcurementBrief:
    """Container describing a procurement intelligence snippet."""

    identifier: str
    title: str
    summary: str
    source: Optional[str] = None
    region: Optional[str] = None

    def to_payload(self) -> Dict[str, str]:
        """Return a payload suitable for vector storage."""

        payload = {
            "record_id": self.identifier,
            "document_type": "external_procurement_brief",
            "title": self.title,
            "summary": self.summary,
        }
        if self.source:
            payload["source"] = self.source
        if self.region:
            payload["region"] = self.region
        return payload


class ProcurementKnowledgeService:
    """Loads procurement-only external knowledge for gpt-oss workflows."""

    def __init__(self, agent_nick, *, knowledge_path: Optional[str] = None):
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        default_path = None
        if self.settings is not None:
            default_path = getattr(
                self.settings,
                "procurement_knowledge_path",
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "resources",
                    "reference_data",
                    "procurement_market_intelligence.json",
                ),
            )
        self.knowledge_path = knowledge_path or default_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_briefs(self) -> List[ProcurementBrief]:
        """Return procurement-focused briefs from the configured data source."""

        if not self.knowledge_path:
            logger.debug("No procurement knowledge path configured; skipping load")
            return []
        path = os.path.abspath(self.knowledge_path)
        if not os.path.exists(path):
            logger.warning("Procurement knowledge file '%s' was not found", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw_entries = json.load(handle)
        except Exception:
            logger.exception("Failed to load procurement knowledge from %s", path)
            return []

        briefs: List[ProcurementBrief] = []
        for entry in raw_entries:
            brief = self._to_brief(entry)
            if brief is not None:
                briefs.append(brief)
        return briefs

    def embed_briefs(self, briefs: Iterable[ProcurementBrief]) -> None:
        """Embed procurement briefs into the shared RAG store."""

        from services.rag_service import RAGService

        rag = RAGService(self.agent_nick)
        texts: List[str] = []
        for brief in briefs:
            payload = brief.to_payload()
            # Ensure customer-identifying fields are stripped from payloads.
            payload.pop("customer", None)
            payload.pop("customer_id", None)
            payload.pop("customer_name", None)
            texts.append(
                f"{brief.title}: {brief.summary} (Source: {brief.source or 'unknown'})"
            )
        if not texts:
            return
        rag.upsert_texts(texts, metadata={"document_type": "external_procurement_brief"})
        logger.info("Embedded %d procurement briefs for external knowledge", len(texts))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_brief(self, entry: Dict[str, object]) -> Optional[ProcurementBrief]:
        """Validate and normalise JSON payloads into :class:`ProcurementBrief`."""

        if not isinstance(entry, dict):
            return None
        identifier = str(entry.get("id") or entry.get("identifier") or "").strip()
        title = str(entry.get("title") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        if not identifier or not title or not summary:
            return None
        brief = ProcurementBrief(
            identifier=identifier,
            title=title,
            summary=summary,
            source=self._clean_value(entry.get("source")),
            region=self._clean_value(entry.get("region")),
        )
        return brief

    @staticmethod
    def _clean_value(value: Optional[object]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


__all__ = ["ProcurementKnowledgeService", "ProcurementBrief"]
