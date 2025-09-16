import logging
from typing import Dict, Optional, Tuple

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
        self.policy_engine = getattr(agent_nick, "policy_engine", None)
        self._negotiation_counts: Dict[Tuple[str, Optional[str]], int] = {}

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("NegotiationAgent starting with input %s", context.input_data)
        supplier = context.input_data.get("supplier")
        current_offer = context.input_data.get("current_offer")
        target_price = context.input_data.get("target_price")
        rfq_id = context.input_data.get("rfq_id")
        round_no = int(context.input_data.get("round", 1))
        item_reference = (
            context.input_data.get("item_id")
            or context.input_data.get("item_description")
            or context.input_data.get("primary_item")
        )

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

        negotiation_key = (str(supplier), str(item_reference or rfq_id))
        db_rounds = self._count_existing_rounds(rfq_id, supplier)
        in_memory_rounds = self._negotiation_counts.get(negotiation_key, 0)
        total_rounds = max(db_rounds, in_memory_rounds)
        if total_rounds >= 3:
            logger.info(
                "Negotiation limit reached for supplier %s and item %s", supplier, item_reference
            )
            data = {
                "supplier": supplier,
                "rfq_id": rfq_id,
                "round": round_no,
                "counter_proposals": [],
                "strategy": None,
                "savings_score": 0.0,
                "decision_log": "negotiation limit reached",
                "message": "",
                "transcript": [],
                "references": [],
                "negotiation_allowed": False,
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=data,
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

        prompt = self._build_prompt(
            supplier,
            rfq_id,
            round_no,
            current_offer,
            target_price,
            item_reference,
            context.input_data,
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
        self._negotiation_counts[negotiation_key] = total_rounds + 1

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
            "item_reference": item_reference,
            "negotiation_allowed": True,
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

    def _build_prompt(
        self,
        supplier: str,
        rfq_id: str,
        round_no: int,
        current_offer: float,
        target_price: float,
        item_reference: Optional[str],
        context: Dict,
    ) -> str:
        guidelines: list[str] = []
        if self.policy_engine is not None:
            policies = getattr(self.policy_engine, "opportunity_policies", [])
            for policy in policies:
                if not policy.get("is_active", True):
                    continue
                description = policy.get("description")
                suggestion = (
                    policy.get("details", {})
                    .get("rules", {})
                    .get("output_suggestion")
                )
                if description:
                    guidelines.append(description)
                if suggestion:
                    guidelines.append(suggestion.replace("{supplier_name}", str(supplier)))
        supplier_response = context.get("response_text")
        if supplier_response:
            guidelines.append(f"Supplier response: {supplier_response}")

        prompt = (
            f"You are negotiating with supplier {supplier} for RFQ {rfq_id}. "
            f"This is round {round_no}. Their current offer is {current_offer}. "
            f"Craft a concise professional counter-proposal aiming for {target_price}."
        )
        if item_reference:
            prompt += f" The discussion concerns item {item_reference}."
        if guidelines:
            prompt += " Consider the following context:\n- " + "\n- ".join(guidelines)
        prompt += " Keep recommendations within policy limits and propose next steps."
        return prompt

    def _count_existing_rounds(self, rfq_id: Optional[str], supplier: Optional[str]) -> int:
        if not rfq_id or not supplier:
            return 0
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM proc.negotiation_sessions WHERE rfq_id = %s AND supplier_id = %s",
                        (rfq_id, supplier),
                    )
                    row = cur.fetchone()
                    if row:
                        return int(row[0])
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to count negotiation rounds")
        return 0

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
