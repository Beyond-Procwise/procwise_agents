import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.supplier_relationship_service import SupplierRelationshipService
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
        logger.info("NegotiationAgent starting")
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
                    "interaction_type": "negotiation",
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
                    "interaction_type": "negotiation",
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
                "interaction_type": "negotiation",
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

        supplier_context = self._load_supplier_context(supplier, context.input_data)

        prompt = self._build_prompt(
            supplier,
            rfq_id,
            round_no,
            current_offer,
            target_price,
            item_reference,
            context.input_data,
            supplier_context,
        )

        logger.debug("NegotiationAgent prompt prepared")
        response = self.call_ollama(prompt=prompt)
        if response.get("error"):
            logger.error(
                "NegotiationAgent LLM call returned error for supplier=%s, rfq_id=%s: %s",
                supplier,
                rfq_id,
                response.get("error"),
            )
        message = response.get("response", "").strip()
        if not message:
            logger.warning(
                "NegotiationAgent received empty response from model for supplier=%s, rfq_id=%s; using fallback",
                supplier,
                rfq_id,
            )
            message = self._build_fallback_message(
                supplier_context,
                supplier,
                rfq_id,
                current_offer,
                target_price,
                item_reference,
                context.input_data,
            )
        logger.info("NegotiationAgent generated negotiation guidance")

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

        decision_log = self._build_decision_log(
            supplier,
            rfq_id,
            current_offer,
            target_price,
            supplier_context,
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
            "interaction_type": "negotiation",
        }
        logger.debug(
            "NegotiationAgent output compiled with keys: %s",
            sorted(data.keys()),
        )
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
        supplier_context: Dict[str, Any],
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

        supplier_profile = supplier_context.get("profile") or {}
        supplier_name = (
            supplier_profile.get("supplier_name")
            or context.get("supplier_name")
            or str(supplier)
        )

        context_lines = self._build_context_lines(
            supplier_context,
            current_offer,
            target_price,
            item_reference,
            context,
        )
        if context_lines:
            guidelines = context_lines + guidelines

        prompt = (
            f"You are negotiating with supplier {supplier_name} for RFQ {rfq_id}. "
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

    def _load_supplier_context(
        self, supplier: Optional[str], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"profile": {}, "spend": {}, "contracts": []}
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return summary

        supplier_id = supplier or context.get("supplier_id")
        supplier_name_hint = (
            context.get("supplier_name")
            or context.get("supplier_company")
            or context.get("supplier")
        )

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    profile = None
                    if supplier_id:
                        cur.execute(
                            """
                            SELECT supplier_id, supplier_name, risk_score, default_currency,
                                   delivery_lead_time_days, is_preferred_supplier,
                                   contact_name_1, contact_email_1
                            FROM proc.supplier
                            WHERE supplier_id = %s
                            LIMIT 1
                            """,
                            (supplier_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            profile = self._row_to_dict(cur, row)

                    if profile is None and supplier_name_hint:
                        cur.execute(
                            """
                            SELECT supplier_id, supplier_name, risk_score, default_currency,
                                   delivery_lead_time_days, is_preferred_supplier,
                                   contact_name_1, contact_email_1
                            FROM proc.supplier
                            WHERE LOWER(supplier_name) = LOWER(%s)
                            LIMIT 1
                            """,
                            (supplier_name_hint,),
                        )
                        row = cur.fetchone()
                        if row:
                            profile = self._row_to_dict(cur, row)

                    if profile:
                        summary["profile"] = profile
                        supplier_name_hint = profile.get("supplier_name") or supplier_name_hint
                        supplier_id = profile.get("supplier_id") or supplier_id

                    if supplier_name_hint:
                        cur.execute(
                            """
                            SELECT
                                COALESCE(SUM(total_amount), 0) AS total_amount,
                                COUNT(*) AS po_count,
                                COALESCE(AVG(total_amount), 0) AS avg_order_value,
                                MAX(order_date) AS last_order_date
                            FROM proc.purchase_order_agent
                            WHERE LOWER(supplier_name) = LOWER(%s)
                            """,
                            (supplier_name_hint,),
                        )
                        row = cur.fetchone()
                        if row:
                            summary["spend"] = self._row_to_dict(cur, row)

                    if supplier_id:
                        cur.execute(
                            """
                            SELECT contract_id, contract_title, contract_end_date,
                                   total_contract_value
                            FROM proc.contracts
                            WHERE supplier_id = %s
                            ORDER BY contract_end_date DESC NULLS LAST
                            LIMIT 3
                            """,
                            (supplier_id,),
                        )
                        rows = cur.fetchall()
                        if rows:
                            columns = [desc[0] for desc in cur.description]
                            summary["contracts"] = [
                                {columns[i]: row[i] for i in range(len(columns))}
                                for row in rows
                            ]
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to load supplier negotiation context")

        relationship_service = SupplierRelationshipService(self.agent_nick)
        relationship_payloads = relationship_service.fetch_relationship(
            supplier_id=supplier_id,
            supplier_name=supplier_name_hint,
            limit=1,
        )
        if relationship_payloads:
            summary["relationship"] = relationship_payloads[0]
            summary["relationships"] = relationship_payloads

        return summary

    def _row_to_dict(self, cursor, row) -> Dict[str, Any]:
        columns = [desc[0] for desc in cursor.description]
        return {columns[i]: row[i] for i in range(len(columns))}

    def _build_context_lines(
        self,
        supplier_context: Dict[str, Any],
        current_offer: Optional[float],
        target_price: Optional[float],
        item_reference: Optional[str],
        context: Dict[str, Any],
    ) -> List[str]:
        lines: List[str] = []
        profile = supplier_context.get("profile") or {}
        currency = profile.get("default_currency") or context.get("currency")

        risk_score = profile.get("risk_score")
        if risk_score:
            lines.append(f"Supplier risk rating: {risk_score}.")

        lead_time = profile.get("delivery_lead_time_days")
        if lead_time:
            lines.append(f"Average lead time {lead_time} days in supplier record.")

        if profile.get("is_preferred_supplier"):
            lines.append("Supplier is designated as preferred in master data.")

        spend = supplier_context.get("spend") or {}
        total_amount = spend.get("total_amount")
        total_text = self._format_currency(total_amount, currency)
        if total_text:
            lines.append(f"Historic purchase orders total {total_text}.")

        po_count = spend.get("po_count")
        try:
            count_value = int(po_count) if po_count is not None else None
        except (TypeError, ValueError):
            count_value = None
        if count_value:
            lines.append(f"Recorded {count_value} purchase orders to date.")

        last_order = spend.get("last_order_date")
        if last_order:
            if hasattr(last_order, "isoformat"):
                order_text = last_order.isoformat()
            else:
                order_text = str(last_order)
            if order_text:
                lines.append(f"Most recent order issued on {order_text}.")

        contracts = supplier_context.get("contracts") or []
        if contracts:
            contract = contracts[0]
            end_date = contract.get("contract_end_date")
            if hasattr(end_date, "isoformat"):
                end_text = end_date.isoformat()
            else:
                end_text = str(end_date) if end_date else ""
            value_text = self._format_currency(
                contract.get("total_contract_value"), currency
            )
            snippet = "Latest contract"
            if contract.get("contract_id"):
                snippet += f" {contract['contract_id']}"
            if end_text:
                snippet += f" ends {end_text}"
            if value_text:
                snippet += f" valued at {value_text}"
            lines.append(snippet + ".")

        benchmark = context.get("benchmark_price")
        benchmark_text = self._format_currency(benchmark, currency)
        if benchmark_text:
            lines.append(f"Internal benchmark price is {benchmark_text}.")

        try:
            if current_offer and target_price and float(current_offer) > 0:
                savings_ratio = (float(current_offer) - float(target_price)) / float(current_offer)
                if savings_ratio > 0:
                    lines.append(
                        f"Achieving the target unlocks approximately {savings_ratio * 100:.1f}% savings."
                    )
        except Exception:
            pass

        if item_reference and context.get("item_summary"):
            lines.append(f"Item {item_reference}: {context.get('item_summary')}")

        additional_points = context.get("negotiation_notes")
        if isinstance(additional_points, list):
            for note in additional_points:
                if isinstance(note, str) and note.strip():
                    lines.append(note.strip())
        elif isinstance(additional_points, str) and additional_points.strip():
            lines.append(additional_points.strip())

        relationship = supplier_context.get("relationship") or {}
        coverage = relationship.get("coverage_ratio")
        if isinstance(coverage, (int, float)) and coverage:
            lines.append(
                f"Supplier data coverage across contracts, purchase orders, invoices and quotes stands at {coverage:.2f}."
            )
        statements = relationship.get("relationship_statements")
        if isinstance(statements, list):
            for statement in statements:
                if isinstance(statement, str) and statement.strip():
                    lines.append(statement.strip())

        return lines

    def _build_decision_log(
        self,
        supplier: Optional[str],
        rfq_id: Optional[str],
        current_offer: Optional[float],
        target_price: Optional[float],
        supplier_context: Dict[str, Any],
    ) -> str:
        base = (
            f"Targeting {target_price} against current offer {current_offer} "
            f"from {supplier} for {rfq_id}."
        )

        extras: List[str] = []
        profile = supplier_context.get("profile") or {}
        spend = supplier_context.get("spend") or {}
        currency = profile.get("default_currency")
        spend_text = self._format_currency(spend.get("total_amount"), currency)
        if spend_text:
            extras.append(f"Historic spend snapshot: {spend_text}.")

        contracts = supplier_context.get("contracts") or []
        if contracts:
            contract = contracts[0]
            end_date = contract.get("contract_end_date")
            if hasattr(end_date, "isoformat"):
                end_text = end_date.isoformat()
            else:
                end_text = str(end_date) if end_date else ""
            if end_text:
                extras.append(f"Latest contract milestone ends {end_text}.")

        if extras:
            return " ".join([base] + extras)
        return base

    def _build_fallback_message(
        self,
        supplier_context: Dict[str, Any],
        supplier: Optional[str],
        rfq_id: Optional[str],
        current_offer: Optional[float],
        target_price: Optional[float],
        item_reference: Optional[str],
        context: Dict[str, Any],
    ) -> str:
        profile = supplier_context.get("profile") or {}
        currency = profile.get("default_currency") or context.get("currency")
        supplier_name = (
            profile.get("supplier_name")
            or context.get("supplier_name")
            or str(supplier or "supplier")
        )

        target_text = self._format_currency(target_price, currency) or str(target_price)
        current_text = self._format_currency(current_offer, currency) or str(current_offer)

        opening = f"Thank you for the quotation provided for RFQ {rfq_id}."
        if item_reference:
            opening += f" We appreciate your proposal covering {item_reference}."

        request = ""
        if target_text:
            request = f"To progress the evaluation we are aiming for pricing closer to {target_text}."
        if current_text:
            request += f" Your current proposal of {current_text} provides a helpful baseline for this conversation."

        spend = supplier_context.get("spend") or {}
        spend_text = self._format_currency(spend.get("total_amount"), currency)
        if spend_text:
            request += (
                f" Historic collaboration with {supplier_name} totals {spend_text}, and we hope to continue this partnership with competitive terms."
            )

        closing = (
            "Please let us know if you can revise the pricing or suggest alternative value drivers such as lead time or service enhancements."
        )

        return " ".join(part for part in (opening, request, closing) if part)

    def _format_currency(self, value: Any, currency: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None
        code = (currency or "GBP").upper()
        symbol = "£" if code == "GBP" else "$" if code == "USD" else ""
        formatted = f"{amount:,.2f}"
        if symbol:
            return f"{symbol}{formatted}"
        return f"{formatted} {code}"
