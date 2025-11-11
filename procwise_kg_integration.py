"""Integration utilities for ProcWise services."""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from email.utils import formatdate
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from hybrid_query_engine import HybridQueryEngine, build_default_engine

LOGGER = logging.getLogger("procwise_kg_integration")
logging.basicConfig(
    level=os.environ.get("PROCWISE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@dataclass
class NegotiationTriggerResult:
    should_trigger_negotiation: bool
    received: int
    expected: int
    recommendation: str


class WorkflowSyncService:
    """Provides workflow state checks backed by Postgres and the knowledge graph."""

    def __init__(self, postgres_dsn: str, engine: Optional[HybridQueryEngine] = None) -> None:
        self._postgres_dsn = postgres_dsn
        self._conn = psycopg2.connect(postgres_dsn, cursor_factory=RealDictCursor)
        self._conn.autocommit = True
        self._engine = engine or build_default_engine()

    def should_trigger_negotiation(self, workflow_id: str, rfq_id: Optional[str], min_responses: int) -> NegotiationTriggerResult:
        received = self._count_responses(workflow_id, rfq_id)
        should_trigger = received >= min_responses
        if should_trigger:
            recommendation = "✅ Proceed with the next negotiation round; response threshold met."
        else:
            remaining = max(min_responses - received, 0)
            recommendation = f"⏳ Wait for {remaining} more supplier response{'s' if remaining != 1 else ''} before proceeding."
        LOGGER.info(
            "Negotiation trigger check: workflow=%s rfq=%s received=%s expected=%s -> %s",
            workflow_id,
            rfq_id,
            received,
            min_responses,
            should_trigger,
        )
        return NegotiationTriggerResult(
            should_trigger_negotiation=should_trigger,
            received=received,
            expected=min_responses,
            recommendation=recommendation,
        )

    def _count_responses(self, workflow_id: str, rfq_id: Optional[str]) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT sr.response_id) AS responses
                FROM supplier_responses sr
                INNER JOIN negotiation_sessions ns ON ns.negotiation_session_id = sr.negotiation_session_id
                WHERE ns.workflow_id = %s
                  AND (%s IS NULL OR ns.rfq_reference = %s)
                """,
                (workflow_id, rfq_id, rfq_id),
            )
            row = cur.fetchone()
        return int(row["responses"]) if row and row.get("responses") is not None else 0


class EmailThreadingUtilities:
    """Constructs deterministic threading headers for outbound emails."""

    DOMAIN = os.environ.get("PROCWISE_EMAIL_DOMAIN", "procwise.local")

    @classmethod
    def build_outbound_headers(
        cls,
        workflow_id: str,
        thread_token: Optional[str] = None,
        parent_message_id: Optional[str] = None,
    ) -> Dict[str, str]:
        token = thread_token or cls.generate_thread_token(workflow_id)
        message_id = cls._compose_message_id(workflow_id, token)
        headers = {
            "Message-ID": message_id,
            "Date": formatdate(localtime=True),
            "X-Procwise-Workflow": workflow_id,
            "X-Procwise-Thread": token,
        }
        if parent_message_id:
            headers["In-Reply-To"] = parent_message_id
            headers["References"] = f"{parent_message_id} {message_id}"
        else:
            headers["References"] = message_id
        return headers

    @classmethod
    def generate_thread_token(cls, workflow_id: str) -> str:
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"procwise:{workflow_id}")
        return uuid.uuid5(namespace, datetime.utcnow().isoformat()).hex[:16]

    @classmethod
    def _compose_message_id(cls, workflow_id: str, token: str) -> str:
        return f"<{token}.{workflow_id}@{cls.DOMAIN}>"

    @classmethod
    def parse_incoming_headers(cls, message_id: str, references: Optional[str]) -> Dict[str, str]:
        tokens = (references or "").split()
        if message_id:
            tokens.append(message_id)
        workflow_token = None
        workflow_id = None
        for token in tokens:
            if token.startswith("<") and token.endswith(">") and "@" in token:
                local_part = token[1:-1].split("@", 1)[0]
                pieces = local_part.split(".")
                if len(pieces) == 2:
                    workflow_token = pieces[0]
                    workflow_id = pieces[1]
                    break
        return {
            "workflow_id": workflow_id,
            "thread_token": workflow_token,
        }


class StrategyGenerator:
    """Produces negotiation strategies using the knowledge graph and semantic search."""

    def __init__(self, engine: HybridQueryEngine) -> None:
        self._engine = engine

    def generate(self, workflow_id: str, rfq_id: Optional[str], context: Optional[str] = None) -> Dict[str, Any]:
        graph_context = self._engine._graph_service.negotiation_summary(rfq_reference=rfq_id, workflow_id=workflow_id)
        semantic_neighbors = self._engine.find_similar_negotiations(rfq_id=rfq_id, text=context or "Negotiation context")
        prompt_payload = {
            "workflow_id": workflow_id,
            "rfq_id": rfq_id,
            "negotiation_summary": graph_context,
            "semantic_neighbors": semantic_neighbors,
            "current_context": context,
        }
        prompt = (
            "You are assisting a procurement negotiator. Using the structured data provided, craft a recommended strategy "
            "covering counter-offers, negotiation levers, talking points, and an estimated win probability. Return JSON with "
            "keys counter_offers (list), levers (list), talking_points (list), win_probability (0-1 float)."
        )
        response = self._engine._ollama.chat(
            prompt=json.dumps(prompt_payload),
            system=prompt,
            format="json",
        )
        try:
            strategy = json.loads(response)
            strategy.setdefault('counter_offers', [])
            strategy.setdefault('levers', [])
            strategy.setdefault('talking_points', [])
        except json.JSONDecodeError:
            LOGGER.error("Strategy generation failed, response: %s", response)
            raise
        strategy["win_probability"] = float(strategy.get("win_probability", 0.0))
        return strategy


class ProcwiseIntegrationFacade:
    """High-level entry points for ProcWise services."""

    def __init__(self, postgres_dsn: Optional[str] = None, engine: Optional[HybridQueryEngine] = None) -> None:
        self._engine = engine or build_default_engine()
        dsn = postgres_dsn or os.environ.get("POSTGRES_DSN")
        if not dsn:
            raise RuntimeError("Postgres DSN required for integration facade")
        self._workflow_service = WorkflowSyncService(dsn, engine=self._engine)
        self._strategy_generator = StrategyGenerator(self._engine)

    def should_trigger_negotiation(self, workflow_id: str, rfq_id: Optional[str], min_responses: int) -> Dict[str, Any]:
        result = self._workflow_service.should_trigger_negotiation(workflow_id, rfq_id, min_responses)
        return {
            "should_trigger_negotiation": result.should_trigger_negotiation,
            "received": result.received,
            "expected": result.expected,
            "recommendation": result.recommendation,
        }

    def generate_strategy(self, workflow_id: str, rfq_id: Optional[str], context: Optional[str] = None) -> Dict[str, Any]:
        return self._strategy_generator.generate(workflow_id, rfq_id, context=context)

    def ask(self, text: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._engine.ask(text, filters=filters)

    def get_email_headers(self, workflow_id: str, parent_message_id: Optional[str] = None) -> Dict[str, str]:
        return EmailThreadingUtilities.build_outbound_headers(workflow_id, parent_message_id=parent_message_id)

    def map_reply(self, message_id: str, references: Optional[str]) -> Dict[str, str]:
        return EmailThreadingUtilities.parse_incoming_headers(message_id, references)
