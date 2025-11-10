"""Integration helpers for the procurement knowledge graph components."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Optional

from config.settings import settings
from procurement_knowledge_graph.config import AppConfig
from procurement_knowledge_graph.procwise_kg_integration import (
    ProcWiseKnowledgeGraphAgent,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_agent() -> ProcWiseKnowledgeGraphAgent:
    config = AppConfig.from_settings(settings)
    logger.debug(
        "Initialising ProcWise knowledge graph agent (Neo4j=%s, Qdrant=%s)",
        config.neo4j.uri,
        f"{config.qdrant.host}:{config.qdrant.port}",
    )
    return ProcWiseKnowledgeGraphAgent(config)


def get_agent() -> ProcWiseKnowledgeGraphAgent:
    """Return the cached knowledge graph agent instance."""

    return _get_agent()


def should_trigger_negotiation(
    workflow_id: str,
    *,
    rfq_id: Optional[str] = None,
    min_responses: int = 3,
) -> Dict[str, Any]:
    """Proxy to the knowledge graph negotiation gating logic."""

    agent = get_agent()
    return agent.should_trigger_negotiation(
        workflow_id,
        rfq_id=rfq_id,
        min_responses_required=min_responses,
    )


def reset_agent_cache() -> None:
    """Clear the cached agent instance for testability."""

    _get_agent.cache_clear()  # type: ignore[attr-defined]


__all__ = [
    "get_agent",
    "should_trigger_negotiation",
    "reset_agent_cache",
]
