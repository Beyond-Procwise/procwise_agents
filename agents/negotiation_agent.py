import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

COMPOSE_SYSTEM_PROMPT = (
    "You are a procurement negotiator. Be concise, professional, and specific.\n"
    "Keep emails under 140 words. Include currency symbols. Use bullet points for lists.\n"
    "Do not invent numbers not provided in the JSON. Always reference the RFQ id."
)

POLISH_SYSTEM_PROMPT = (
    "Polish the email for warmth and clarity without changing numbers or commitments.\n"
    "Cap at 140 words. Keep bullet points. British English."
)


def decide_strategy(
    price: Optional[float],
    target: Optional[float],
    lead_weeks: Optional[float] = None,
    constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministic negotiation logic shared across tests and runtime."""

    if price is None or target is None:
        return {
            "strategy": "clarify",
            "counter_price": None,
            "asks": [
                "Confirm unit price, currency, tiered price @ 100/250/500."
            ],
            "lead_time_request": None,
            "rationale": "Price missing; request structured quote.",
        }

    gap = (price - target) / target
    asks: List[str] = []
    lead_req = None

    if lead_weeks and lead_weeks > 3:
        lead_req = "≤ 2 weeks or split shipment (20–30% now, balance later)"
        asks.append("Split shipment or expedite option")

    if gap <= 0:
        counter = round(min(price, target * 0.99), 2)
        return {
            "strategy": "accept-with-sweetener",
            "counter_price": counter,
            "asks": asks
            + ["Net-30 (or 1–2% discount)", "Volume price break @ 100/250/500"],
            "lead_time_request": lead_req,
            "rationale": "Offer meets or beats target; seek small concession.",
        }

    if gap <= 0.10:
        counter = round((price + target) / 2, 2)
        return {
            "strategy": "midpoint",
            "counter_price": counter,
            "asks": asks + ["Volume price break @ 100/250/500"],
            "lead_time_request": lead_req,
            "rationale": "Narrow gap; split the difference with rationale.",
        }

    counter = round(max(target, price * 0.88), 2)
    return {
        "strategy": "anchor-lower",
        "counter_price": counter,
        "asks": asks
        + ["Volume price break @ 100/250/500", "Alternative part/brand if needed"],
        "lead_time_request": lead_req,
        "rationale": "Wide gap; anchor lower and open multiple paths.",
    }


class NegotiationAgent(BaseAgent):
    """Generate deterministic negotiation decisions and LLM-composed emails."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self._negotiation_counts: Dict[Tuple[str, Optional[str]], int] = {}

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("NegotiationAgent starting")

        supplier = context.input_data.get("supplier")
        rfq_id = context.input_data.get("rfq_id")
        round_no = int(context.input_data.get("round", 1))
        item_reference = (
            context.input_data.get("item_id")
            or context.input_data.get("item_description")
            or context.input_data.get("primary_item")
        )

        price_raw = (
            context.input_data.get("current_offer")
            if context.input_data.get("current_offer") is not None
            else context.input_data.get("price")
        )
        target_raw = context.input_data.get("target_price")
        currency = self._normalise_currency(
            context.input_data.get("currency")
            or context.input_data.get("current_offer_currency")
            or context.input_data.get("price_currency")
        )
        currency_conf = self._coerce_float(
            context.input_data.get("currency_confidence")
            or context.input_data.get("current_offer_currency_confidence")
        )

        price = self._coerce_float(price_raw)
        target_price = self._coerce_float(target_raw)

        if currency_conf is not None and currency_conf < 0.5:
            logger.warning(
                "Currency confidence %.2f below threshold; withholding price data", currency_conf
            )
            currency = None
            price = None

        lead_weeks = self._parse_lead_weeks(
            context.input_data.get("lead_time_weeks")
            or context.input_data.get("lead_time")
            or context.input_data.get("lead_time_days")
        )

        constraints = self._ensure_list(context.input_data.get("constraints"))

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
                "decision": decide_strategy(None, None),
                "negotiation_allowed": False,
                "message": "",
                "interaction_type": "negotiation",
            }
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=data,
                next_agents=[],
            )

        decision = decide_strategy(price, target_price, lead_weeks=lead_weeks, constraints=constraints)
        decision.setdefault("strategy", "clarify")
        decision.setdefault("counter_price", None)
        decision.setdefault("asks", [])
        decision.setdefault("lead_time_request", None)
        decision.setdefault("rationale", "")
        decision["round"] = round_no

        supplier_snippets = self._collect_supplier_snippets(context.input_data)
        price_breaks = self._ensure_list(context.input_data.get("price_breaks"))
        moq = self._coerce_float(context.input_data.get("moq"))
        incoterms = context.input_data.get("incoterms")
        payment_terms = context.input_data.get("payment_terms")

        email_body = ""
        compose_payload = {
            "rfq_id": rfq_id,
            "supplier": supplier,
            "target_price": target_price,
            "current_offer": price,
            "lead_time_weeks": lead_weeks,
            "currency": currency,
            "decision": decision,
            "moq": moq,
            "incoterms": incoterms,
            "payment_terms": payment_terms,
            "price_breaks": price_breaks,
        }

        compose_messages = [
            {"role": "system", "content": COMPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_compose_prompt(compose_payload, supplier_snippets),
            },
        ]

        compose_response = self.call_ollama(
            model="llama3.2",
            messages=compose_messages,
            options={"temperature": 0.3, "top_p": 0.9, "num_ctx": 4096},
        )
        email_body = self._extract_message(compose_response)

        if not email_body:
            logger.warning(
                "NegotiationAgent compose model returned empty response for supplier=%s rfq_id=%s",
                supplier,
                rfq_id,
            )
            email_body = self._build_fallback_email(rfq_id, decision, currency)

        should_polish = bool(
            context.input_data.get("polish_email")
            or getattr(self.settings, "negotiation_polish_enabled", False)
        )
        if should_polish and email_body:
            polish_messages = [
                {"role": "system", "content": POLISH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Original:\n<<<\n{email_body}\n<<<",
                },
            ]
            polish_response = self.call_ollama(
                model="gemma3",
                messages=polish_messages,
                options={"temperature": 0.2, "num_ctx": 2048, "max_tokens": 260},
            )
            polished = self._extract_message(polish_response)
            if polished:
                email_body = polished

        savings_score = 0.0
        if price and price > 0 and target_price is not None:
            try:
                savings_score = (price - target_price) / float(price)
            except ZeroDivisionError:
                savings_score = 0.0

        counter_options: List[Dict[str, Any]] = []
        counter_price = decision.get("counter_price")
        if counter_price is not None:
            counter_options.append({"price": counter_price, "terms": None, "bundle": None})

        if counter_price is not None and supplier and rfq_id:
            self._store_session(rfq_id, supplier, round_no, counter_price)

        self._negotiation_counts[negotiation_key] = total_rounds + 1

        decision_log = self._build_decision_log(supplier, rfq_id, price, target_price, decision)
        email_subject = self._build_subject(rfq_id, supplier)

        data = {
            "supplier": supplier,
            "rfq_id": rfq_id,
            "round": round_no,
            "counter_proposals": counter_options,
            "decision": decision,
            "savings_score": savings_score,
            "decision_log": decision_log,
            "message": email_body,
            "email_subject": email_subject,
            "email_body": email_body,
            "supplier_snippets": supplier_snippets,
            "negotiation_allowed": True,
            "interaction_type": "negotiation",
        }

        logger.info(
            "NegotiationAgent generated strategy '%s' for supplier=%s rfq_id=%s",
            decision.get("strategy"),
            supplier,
            rfq_id,
        )

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=data,
            pass_fields=data,
            next_agents=["EmailDraftingAgent"],
        )

    def _build_compose_prompt(
        self, payload: Dict[str, Any], supplier_snippets: List[str]
    ) -> str:
        body = ["Context (JSON):", json.dumps(payload, ensure_ascii=False, default=self._json_default)]
        if supplier_snippets:
            body.append("Supplier snippets:")
            for snippet in supplier_snippets:
                body.append(f"- {snippet}")
        body.append(
            "Write a short email to the supplier proposing the counter-offer."
            " Include: counter price (if present), requested lead time (if any),"
            " and 1–2 alternative paths from decision. Ask for confirmation within 3 days."
        )
        return "\n".join(body)

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

    def _collect_supplier_snippets(self, payload: Dict[str, Any]) -> List[str]:
        snippets: List[str] = []
        for key in (
            "supplier_snippets",
            "snippets",
            "highlights",
            "supplier_highlights",
            "response_text",
            "message",
            "raw_email",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                snippets.extend(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                snippets.append(value.strip())
        return snippets[:5]

    def _build_fallback_email(
        self, rfq_id: Optional[str], decision: Dict[str, Any], currency: Optional[str]
    ) -> str:
        parts = [f"Thank you for your response on RFQ {rfq_id}."]
        counter_price = decision.get("counter_price")
        if counter_price is not None:
            parts.append(
                f"We would like to align on pricing at {self._format_currency(counter_price, currency)}."
            )
        lead_request = decision.get("lead_time_request")
        if lead_request:
            parts.append(f"Please confirm if the following lead time works: {lead_request}.")
        asks = decision.get("asks") or []
        if asks:
            parts.append("Key requests: " + "; ".join(asks))
        parts.append("Could you confirm within 3 days? Many thanks.")
        return " ".join(filter(None, parts))

    def _build_decision_log(
        self,
        supplier: Optional[str],
        rfq_id: Optional[str],
        price: Optional[float],
        target_price: Optional[float],
        decision: Dict[str, Any],
    ) -> str:
        base = (
            f"Strategy={decision.get('strategy')} counter={decision.get('counter_price')}"
            f" target={target_price} current={price} supplier={supplier} rfq={rfq_id}."
        )
        rationale = decision.get("rationale")
        if rationale:
            return f"{base} {rationale}"
        return base

    def _build_subject(self, rfq_id: Optional[str], supplier: Optional[str]) -> str:
        rfq_text = rfq_id or "RFQ"
        supplier_text = supplier or "Supplier"
        return f"Re: {rfq_text} – Updated terms for {supplier_text}"

    def _format_currency(self, value: Optional[float], currency: Optional[str]) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return ""
        code = (currency or "GBP").upper()
        symbol = "£" if code == "GBP" else "$" if code == "USD" else "€" if code == "EUR" else "₹" if code == "INR" else ""
        formatted = f"{amount:,.2f}"
        return f"{symbol}{formatted}" if symbol else f"{formatted} {code}"

    def _normalise_currency(self, value: Any) -> Optional[str]:
        if not value:
            return None
        if isinstance(value, str):
            trimmed = value.strip().upper()
            if len(trimmed) == 3:
                return trimmed
        return None

    def _parse_lead_weeks(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
            return number if number <= 12 else round(number / 7.0, 2)
        except ValueError:
            pass
        lowered = text.lower()
        digits = "".join(ch for ch in lowered if (ch.isdigit() or ch == "."))
        try:
            numeric = float(digits)
        except ValueError:
            return None
        if "week" in lowered:
            return numeric
        if "day" in lowered or "business" in lowered:
            return round(numeric / 7.0, 2)
        return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _ensure_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _extract_message(self, response: Dict[str, Any]) -> str:
        if not response:
            return ""
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        content = response.get("response")
        if isinstance(content, str):
            return content.strip()
        return ""

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
