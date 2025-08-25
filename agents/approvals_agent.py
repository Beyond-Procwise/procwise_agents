import logging

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class ApprovalsAgent(BaseAgent):
    """Determine approval decisions based on simple thresholds."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()

    def run(self, context: AgentContext) -> AgentOutput:
        amount = context.input_data.get("amount")
        supplier_id = context.input_data.get("supplier_id")
        if amount is None:
            return AgentOutput(status=AgentStatus.FAILED, data={}, error="amount not provided")

        # Retrieve approval threshold from Postgres
        threshold = context.input_data.get("threshold", 1000)
        try:
            with self.agent_nick.get_db_connection() as conn:  # pragma: no cover - network
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT threshold FROM approval_policies WHERE supplier_id = %s",
                        (supplier_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        threshold = row[0]
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to fetch approval policy")

        decision = "approve" if amount <= threshold else "escalate"
        log = f"Amount {amount} {'within' if amount <= threshold else 'above'} threshold {threshold}"

        # Query Qdrant for supporting evidence
        references: list[str] = []
        try:
            vector = self.agent_nick.embedding_model.encode(str(amount)).tolist()
            hits = self.agent_nick.qdrant_client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=vector,
                limit=1,
            )
            references = [h.payload.get("document_type") for h in hits]
        except Exception:  # pragma: no cover - external dependency
            logger.exception("qdrant search failed")

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "amount": amount,
                "threshold": threshold,
                "decision": decision,
                "decision_log": log,
                "references": references,
            },
            next_agents=[],
        )
