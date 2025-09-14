import logging

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class NegotiationAgent(BaseAgent):
    """Generate supplier counter proposals using an LLM.

    This agent encapsulates the negotiation flow described in the business
    requirements.  It consumes supplier proposals and negotiation context and
    produces structured counter-proposal options alongside a recommended message
    ready for a drafting agent.
    """

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("NegotiationAgent starting with input %s", context.input_data)
        supplier = context.input_data.get("supplier")
        current_offer = context.input_data.get("current_offer")
        target_price = context.input_data.get("target_price")
        rfq_id = context.input_data.get("rfq_id")
        round_no = int(context.input_data.get("round", 1))

        if supplier is None or current_offer is None or target_price is None:
            """Gracefully handle missing negotiation inputs.

            Some flows may branch to the negotiation agent even when the
            necessary fields are absent (e.g. no quote returned).  Rather than
            failing the workflow we return a success with an explanatory
            message so downstream steps can decide how to proceed.
            """
            logger.warning("NegotiationAgent missing input data; skipping negotiation")
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={
                    "supplier": supplier,
                    "counter_proposals": [],
                    "strategy": None,
                    "savings_score": 0.0,
                    "decision_log": "no negotiation data provided",
                    "message": "",
                    "transcript": [],
                    "references": [],
                },
                pass_fields={
                    "supplier": supplier,
                    "counter_proposals": [],
                    "strategy": None,
                    "savings_score": 0.0,
                    "decision_log": "no negotiation data provided",
                    "message": "",
                    "transcript": [],
                    "references": [],
                },
                next_agents=[],
            )

        # Retrieve negotiation strategy from Postgres
        strategy = "counter"
        try:
            with self.agent_nick.get_db_connection() as conn:  # pragma: no cover - network
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT strategy FROM negotiation_strategies WHERE supplier = %s",
                        (supplier,),
                    )
                    row = cur.fetchone()
                    if row:
                        strategy = row[0]
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to fetch negotiation strategy")

        prompt = (
            f"You are negotiating with supplier {supplier} for RFQ {rfq_id}. "
            f"This is round {round_no}. Their current offer is {current_offer}. "
            f"Craft a concise professional counter-proposal aiming for {target_price}."
        )

        logger.debug("NegotiationAgent prompt: %s", prompt)
        response = self.call_ollama(prompt=prompt)
        message = response.get("response", "").strip()
        logger.info("NegotiationAgent generated message: %s", message)

        # Retrieve relevant references from Qdrant
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

        savings_score = 0.0
        try:
            savings_score = (current_offer - target_price) / float(current_offer)
        except Exception:  # pragma: no cover - defensive maths errors
            pass

        decision_log = (
            f"Targeting {target_price} against current offer {current_offer} from {supplier} for {rfq_id}."
        )
        counter_options = [{"price": target_price, "terms": None, "bundle": None}]

        self._store_session(rfq_id, supplier, round_no, target_price)

        data = {
            "supplier": supplier,
            "rfq_id": rfq_id,
            "round": round_no,
            "counter_proposals": counter_options,
            "strategy": strategy,
            "savings_score": savings_score,
            "decision_log": decision_log,
            "message": message,
            "transcript": [message],
            "references": references,
        }
        logger.debug("NegotiationAgent output: %s", data)
        logger.info(
            "NegotiationAgent produced %d counter proposals", len(counter_options)
        )
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=data,
            pass_fields=data,
            next_agents=["EmailDraftingAgent"],
        )

    def _store_session(self, rfq_id: str, supplier: str, round_no: int, counter_price: float) -> None:
        """Persist negotiation round details."""
        if not rfq_id or not supplier:
            return
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.negotiation_sessions
                            (rfq_id, supplier_id, round, counter_offer, created_on)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (rfq_id, supplier, round_no, counter_price),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store negotiation session")
