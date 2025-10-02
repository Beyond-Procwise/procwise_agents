import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

MAX_SUPPLIER_REPLIES = 3
FINAL_OFFER_PATTERNS = (
    "best and final",
    "best final",
    "best offer we can",
    "best price we can",
    "cannot go lower",
    "final offer",
    "final price",
    "final quotation",
    "last price",
    "lowest we can do",
    "our best price",
    "rock bottom",
    "take it or leave it",
    "ultimatum",
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
            "asks": ["Confirm unit price, currency, tiered price @ 100/250/500."],
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
    """Generate deterministic negotiation decisions and coordinate counter drafting."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self._state_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("NegotiationAgent starting")

        supplier = context.input_data.get("supplier") or context.input_data.get("supplier_id")
        rfq_id = context.input_data.get("rfq_id")
        raw_round = context.input_data.get("round", 1)

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
        supplier_snippets = self._collect_supplier_snippets(context.input_data)
        supplier_message = self._coerce_text(
            context.input_data.get("supplier_message")
            or context.input_data.get("response_text")
            or context.input_data.get("message")
        )

        decision = decide_strategy(price, target_price, lead_weeks=lead_weeks, constraints=constraints)
        decision.setdefault("strategy", "clarify")
        decision.setdefault("counter_price", None)
        decision.setdefault("asks", [])
        decision.setdefault("lead_time_request", None)
        decision.setdefault("rationale", "")

        state, _ = self._load_session_state(rfq_id, supplier)
        message_id = context.input_data.get("message_id")
        supplier_reply_registered = False
        if message_id and state.get("last_supplier_msg_id") != message_id:
            previously_awaiting = bool(state.get("awaiting_response", False))
            if previously_awaiting:
                state["supplier_reply_count"] = int(state.get("supplier_reply_count", 0)) + 1
            else:
                state["supplier_reply_count"] = int(state.get("supplier_reply_count", 0))
            state["last_supplier_msg_id"] = message_id
            state["awaiting_response"] = False
            supplier_reply_registered = True

        final_offer_reason = self._detect_final_offer(supplier_message, supplier_snippets)
        should_continue, new_status, halt_reason = self._should_continue(
            state, supplier_reply_registered, final_offer_reason
        )

        round_no = int(state.get("current_round", 1))
        if isinstance(raw_round, (int, float)):
            round_no = max(round_no, int(raw_round))
        decision["round"] = round_no

        savings_score = 0.0
        if price and price > 0 and target_price is not None:
            try:
                savings_score = (price - target_price) / float(price)
            except ZeroDivisionError:
                savings_score = 0.0

        decision_log = self._build_decision_log(supplier, rfq_id, price, target_price, decision)

        draft_records: List[Dict[str, Any]] = []

        if not should_continue:
            state["status"] = new_status
            awaiting_before = bool(state.get("awaiting_response", False))
            if awaiting_before and halt_reason == "Awaiting supplier response.":
                state["awaiting_response"] = True
            else:
                state["awaiting_response"] = False
            stop_message = self._build_stop_message(new_status, halt_reason, round_no)
            decision.setdefault("status_reason", halt_reason)
            self._save_session_state(rfq_id, supplier, state)
            negotiation_open = new_status not in {"COMPLETED", "EXHAUSTED"}
            data = {
                "supplier": supplier,
                "rfq_id": rfq_id,
                "round": round_no,
                "counter_proposals": [],
                "decision": decision,
                "savings_score": savings_score,
                "decision_log": decision_log,
                "message": stop_message,
                "email_subject": None,
                "email_body": None,
                "supplier_snippets": supplier_snippets,
                "negotiation_allowed": negotiation_open,
                "interaction_type": "negotiation",
                "session_state": self._public_state(state),
                "currency": currency,
                "current_offer": price,
                "target_price": target_price,
                "supplier_message": supplier_message,
                "drafts": draft_records,
                "sent_status": False,
            }
            logger.info(
                "NegotiationAgent halted negotiation for supplier=%s rfq_id=%s reason=%s",
                supplier,
                rfq_id,
                halt_reason,
            )
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=data,
                next_agents=[],
            )

        negotiation_message = self._build_summary(
            rfq_id,
            decision,
            price,
            target_price,
            currency,
            round_no,
        )

        counter_options: List[Dict[str, Any]] = []
        counter_price = decision.get("counter_price")
        if counter_price is not None:
            counter_options.append({"price": counter_price, "terms": None, "bundle": None})

        if rfq_id and supplier:
            self._store_session(rfq_id, supplier, round_no, counter_price)

        draft_payload = {
            "intent": "NEGOTIATION_COUNTER",
            "rfq_id": rfq_id,
            "supplier_id": supplier,
            "current_offer": price_raw,
            "current_offer_numeric": price,
            "target_price": target_price,
            "counter_price": counter_price,
            "currency": currency,
            "decision": decision,
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "round": round_no,
            "supplier_reply_count": state.get("supplier_reply_count", 0),
            "supplier_message": supplier_message,
            "supplier_snippets": supplier_snippets,
            "from_address": context.input_data.get("from_address"),
            "negotiation_message": negotiation_message,
        }

        draft_metadata = {
            "counter_price": counter_price,
            "target_price": target_price,
            "current_offer": price,
            "round": round_no,
            "supplier_reply_count": state.get("supplier_reply_count", 0),
            "strategy": decision.get("strategy"),
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "intent": "NEGOTIATION_COUNTER",
        }

        draft_stub = {
            "rfq_id": rfq_id,
            "supplier_id": supplier,
            "intent": "NEGOTIATION_COUNTER",
            "metadata": draft_metadata,
            "negotiation_message": negotiation_message,
            "counter_proposals": counter_options,
            "sent_status": False,
            "thread_index": round_no,
        }
        if currency:
            draft_stub["currency"] = currency
        if supplier_snippets:
            draft_stub["supplier_snippets"] = supplier_snippets
        draft_stub["payload"] = draft_payload
        draft_records.append(draft_stub)

        state["status"] = "ACTIVE"
        state["awaiting_response"] = True
        state["current_round"] = round_no + 1
        state["last_email_sent_at"] = datetime.now(timezone.utc)
        self._save_session_state(rfq_id, supplier, state)

        data = {
            "supplier": supplier,
            "rfq_id": rfq_id,
            "round": round_no,
            "counter_proposals": counter_options,
            "decision": decision,
            "savings_score": savings_score,
            "decision_log": decision_log,
            "message": negotiation_message,
            "email_subject": None,
            "email_body": None,
            "supplier_snippets": supplier_snippets,
            "negotiation_allowed": True,
            "interaction_type": "negotiation",
            "intent": "NEGOTIATION_COUNTER",
            "draft_payload": draft_payload,
            "drafts": draft_records,
            "session_state": self._public_state(state),
            "currency": currency,
            "current_offer": price,
            "target_price": target_price,
            "supplier_message": supplier_message,
            "sent_status": False,
        }

        logger.info(
            "NegotiationAgent prepared counter round %s for supplier=%s rfq_id=%s", round_no, supplier, rfq_id
        )

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=data,
            pass_fields=data,
            next_agents=["EmailDraftingAgent"],
        )

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------
    def _default_state(self) -> Dict[str, Any]:
        return {
            "supplier_reply_count": 0,
            "current_round": 1,
            "status": "ACTIVE",
            "awaiting_response": False,
            "last_supplier_msg_id": None,
            "last_agent_msg_id": None,
            "last_email_sent_at": None,
        }

    def _public_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "supplier_reply_count": int(state.get("supplier_reply_count", 0)),
            "current_round": int(state.get("current_round", 1)),
            "status": state.get("status", "ACTIVE"),
            "awaiting_response": bool(state.get("awaiting_response", False)),
        }

    def _load_session_state(
        self, rfq_id: Optional[str], supplier: Optional[str]
    ) -> Tuple[Dict[str, Any], bool]:
        if not rfq_id or not supplier:
            return self._default_state(), False
        key = (str(rfq_id), str(supplier))
        if key in self._state_cache:
            return dict(self._state_cache[key]), True

        state = self._default_state()
        exists = False
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT supplier_reply_count, current_round, status, awaiting_response,
                               last_supplier_msg_id, last_agent_msg_id, last_email_sent_at
                          FROM proc.negotiation_session_state
                         WHERE rfq_id = %s AND supplier_id = %s
                        """,
                        (rfq_id, supplier),
                    )
                    row = cur.fetchone()
                    if row:
                        (
                            state["supplier_reply_count"],
                            state["current_round"],
                            state["status"],
                            state["awaiting_response"],
                            state["last_supplier_msg_id"],
                            state["last_agent_msg_id"],
                            state["last_email_sent_at"],
                        ) = row
                        exists = True
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to load negotiation session state")
        self._state_cache[key] = dict(state)
        return dict(state), exists

    def _save_session_state(self, rfq_id: Optional[str], supplier: Optional[str], state: Dict[str, Any]) -> None:
        if not rfq_id or not supplier:
            return
        key = (str(rfq_id), str(supplier))
        self._state_cache[key] = dict(state)
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.negotiation_session_state (
                            rfq_id,
                            supplier_id,
                            supplier_reply_count,
                            current_round,
                            status,
                            awaiting_response,
                            last_supplier_msg_id,
                            last_agent_msg_id,
                            last_email_sent_at,
                            updated_on
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (rfq_id, supplier_id) DO UPDATE SET
                            supplier_reply_count = EXCLUDED.supplier_reply_count,
                            current_round = EXCLUDED.current_round,
                            status = EXCLUDED.status,
                            awaiting_response = EXCLUDED.awaiting_response,
                            last_supplier_msg_id = EXCLUDED.last_supplier_msg_id,
                            last_agent_msg_id = EXCLUDED.last_agent_msg_id,
                            last_email_sent_at = EXCLUDED.last_email_sent_at,
                            updated_on = NOW()
                        """,
                        (
                            rfq_id,
                            supplier,
                            int(state.get("supplier_reply_count", 0)),
                            int(state.get("current_round", 1)),
                            state.get("status", "ACTIVE"),
                            bool(state.get("awaiting_response", False)),
                            state.get("last_supplier_msg_id"),
                            state.get("last_agent_msg_id"),
                            state.get("last_email_sent_at"),
                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist negotiation session state")

    # ------------------------------------------------------------------
    # Decision support helpers
    # ------------------------------------------------------------------
    def _detect_final_offer(
        self, supplier_message: Optional[str], supplier_snippets: List[str]
    ) -> Optional[str]:
        texts: List[str] = []
        if supplier_message:
            texts.append(supplier_message)
        texts.extend(snippet for snippet in supplier_snippets if snippet)
        for text in texts:
            lowered = text.lower()
            for pattern in FINAL_OFFER_PATTERNS:
                if pattern in lowered:
                    return f"Supplier indicated final offer via phrase '{pattern}'."
        return None

    def _should_continue(
        self,
        state: Dict[str, Any],
        supplier_reply_registered: bool,
        final_offer_reason: Optional[str],
    ) -> Tuple[bool, str, str]:
        status = state.get("status", "ACTIVE")
        if status in {"COMPLETED", "EXHAUSTED"}:
            return False, status, f"Session already {status.lower()}."
        if final_offer_reason:
            return False, "COMPLETED", final_offer_reason
        replies = int(state.get("supplier_reply_count", 0))
        if replies >= MAX_SUPPLIER_REPLIES:
            return False, "EXHAUSTED", "Supplier reply cap reached."
        if state.get("awaiting_response") and not supplier_reply_registered:
            return False, status or "ACTIVE", "Awaiting supplier response."
        return True, "ACTIVE", ""

    def _build_summary(
        self,
        rfq_id: Optional[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
    ) -> str:
        parts: List[str] = []
        rfq_text = rfq_id or "RFQ"
        parts.append(f"Round {round_no} plan for {rfq_text}: {decision.get('strategy')}")
        counter_price = decision.get("counter_price")
        if counter_price is not None:
            parts.append(
                f"Counter at {self._format_currency(counter_price, currency)} against supplier offer"
            )
        elif price is not None:
            parts.append(
                f"Seek clarification before accepting {self._format_currency(price, currency)}"
            )
        if target_price is not None and price is not None:
            parts.append(
                f"Target {self._format_currency(target_price, currency)} vs offer {self._format_currency(price, currency)}"
            )
        lead_time = decision.get("lead_time_request")
        if lead_time:
            parts.append(f"Lead time ask: {lead_time}")
        asks = decision.get("asks") or []
        if asks:
            parts.append("Key asks: " + "; ".join(asks))
        return ". ".join(parts)

    def _build_stop_message(self, status: str, reason: str, round_no: int) -> str:
        status_text = status.capitalize()
        reason_text = reason or "No further action required."
        return f"Negotiation {status_text} after round {round_no}: {reason_text}"

    def _store_session(self, rfq_id: str, supplier: str, round_no: int, counter_price: Optional[float]) -> None:
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

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _format_currency(self, value: Optional[float], currency: Optional[str]) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return ""
        code = (currency or "GBP").upper()
        symbol = (
            "£"
            if code == "GBP"
            else "$"
            if code == "USD"
            else "€"
            if code == "EUR"
            else "₹"
            if code == "INR"
            else ""
        )
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

    def _coerce_text(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        return None

    def _ensure_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
