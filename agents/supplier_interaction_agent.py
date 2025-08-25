import logging

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class SupplierInteractionAgent(BaseAgent):
    """Normalise supplier messages and decide next routing step."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()

    def run(self, context: AgentContext) -> AgentOutput:
        message = context.input_data.get("message")
        supplier_id = context.input_data.get("supplier_id")
        if not message:
            return AgentOutput(status=AgentStatus.FAILED, data={}, error="message not provided")

        # Gather supplier profile from Postgres
        supplier_tier = "unknown"
        try:
            with self.agent_nick.get_db_connection() as conn:  # pragma: no cover - network
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT tier FROM suppliers WHERE supplier_id = %s", (supplier_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        supplier_tier = row[0]
        except Exception:  # pragma: no cover - best effort logging
            logger.exception("failed to fetch supplier profile")

        # Search Qdrant for related context
        references: list[str] = []
        try:
            vector = self.agent_nick.embedding_model.encode(message).tolist()
            hits = self.agent_nick.qdrant_client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=vector,
                limit=1,
            )
            references = [h.payload.get("document_type") for h in hits]
        except Exception:  # pragma: no cover - external dependency
            logger.exception("qdrant search failed")

        normalized = {"text": message, "supplier_tier": supplier_tier}
        decision = (
            "NegotiationAgent" if "offer" in message.lower() else "EmailDraftingAgent"
        )
        transcript = [message]

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "normalized_message": normalized,
                "routing_decision": decision,
                "transcript": transcript,
                "references": references,
            },
            next_agents=[decision],
        )
