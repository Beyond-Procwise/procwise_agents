import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from agents.email_drafting_agent import EmailDraftingAgent

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from agents.supplier_interaction_agent import SupplierInteractionAgent
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
        self._email_agent: Optional[EmailDraftingAgent] = None
        self._supplier_agent: Optional["SupplierInteractionAgent"] = None

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
            if halt_reason == "Awaiting supplier response.":
                state["awaiting_response"] = True
            else:
                state["awaiting_response"] = False
            stop_message = self._build_stop_message(new_status, halt_reason, round_no)
            decision.setdefault("status_reason", halt_reason)
            self._save_session_state(rfq_id, supplier, state)
            self._record_learning_snapshot(
                context,
                rfq_id,
                supplier,
                decision,
                state,
                bool(state.get("awaiting_response", False)),
                supplier_reply_registered,
            )
            awaiting_now = bool(state.get("awaiting_response", False))
            negotiation_open = (
                new_status not in {"COMPLETED", "EXHAUSTED"} and not awaiting_now
            )
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
                "awaiting_response": awaiting_now,
            }
            logger.info(
                "NegotiationAgent halted negotiation for supplier=%s rfq_id=%s reason=%s",
                supplier,
                rfq_id,
                halt_reason,
            )
            pass_fields = dict(data)
            next_agents: List[str] = []
            if awaiting_now:
                watch_fields = self._build_supplier_watch_fields(
                    context=context,
                    rfq_id=rfq_id,
                    supplier=supplier,
                    drafts=draft_records,
                    state=state,
                )
                if watch_fields:
                    pass_fields.update(watch_fields)
                    next_agents.append("SupplierInteractionAgent")
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=pass_fields,
                next_agents=next_agents,
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

        try:
            supplier_reply_count = int(state.get("supplier_reply_count", 0))
        except Exception:
            supplier_reply_count = 0

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
            "supplier_reply_count": supplier_reply_count,
            "supplier_message": supplier_message,
            "supplier_snippets": supplier_snippets,
            "from_address": context.input_data.get("from_address"),
            "negotiation_message": negotiation_message,
        }

        recipients = self._collect_recipient_candidates(context)
        if recipients:
            draft_payload.setdefault("recipients", recipients)

        draft_metadata = {
            "counter_price": counter_price,
            "target_price": target_price,
            "current_offer": price,
            "round": round_no,
            "supplier_reply_count": supplier_reply_count,
            "strategy": decision.get("strategy"),
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "intent": "NEGOTIATION_COUNTER",
        }

        email_action_id: Optional[str] = None
        email_subject: Optional[str] = None
        email_body: Optional[str] = None
        draft_records: List[Dict[str, Any]] = []
        next_agents: List[str] = []

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
        if recipients:
            draft_stub["recipients"] = recipients

        email_payload = self._build_email_agent_payload(
            context,
            draft_payload,
            decision,
            state,
            negotiation_message,
        )

        email_output: Optional[AgentOutput] = None
        fallback_payload: Optional[Dict[str, Any]] = None
        if email_payload and supplier and rfq_id:
            email_output = self._invoke_email_drafting_agent(context, email_payload)
            fallback_payload = dict(email_payload)
        if email_output and email_output.status == AgentStatus.SUCCESS:
            email_data = email_output.data or {}
            email_action_id = email_output.action_id or email_data.get("action_id")
            email_subject = email_data.get("subject")
            email_body = email_data.get("body")
            drafts_payload = email_data.get("drafts")
            if isinstance(drafts_payload, list) and drafts_payload:
                for draft in drafts_payload:
                    if not isinstance(draft, dict):
                        continue
                    draft_copy = dict(draft)
                    if email_action_id:
                        draft_copy.setdefault("email_action_id", email_action_id)
                    draft_records.append(draft_copy)
            else:
                draft_records.append(dict(draft_stub))
        else:
            draft_records.append(dict(draft_stub))
            next_agents = ["EmailDraftingAgent"]
            if fallback_payload is None:
                # Reconstruct a payload compatible with EmailDraftingAgent expectations.
                fallback_payload = self._build_email_agent_payload(
                    context,
                    draft_payload,
                    decision,
                    state,
                    negotiation_message,
                )

        state["status"] = "ACTIVE"
        state["awaiting_response"] = True
        state["current_round"] = round_no + 1
        state["last_email_sent_at"] = datetime.now(timezone.utc)
        if email_action_id:
            state["last_agent_msg_id"] = email_action_id
        self._save_session_state(rfq_id, supplier, state)
        self._record_learning_snapshot(
            context,
            rfq_id,
            supplier,
            decision,
            state,
            True,
            supplier_reply_registered,
        )
        cache_key: Optional[Tuple[str, str]] = None
        if rfq_id and supplier:
            cache_key = (str(rfq_id), str(supplier))
        cached_state = self._state_cache.get(cache_key) if cache_key else None
        public_state = self._public_state(cached_state or state)
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
            "session_state": public_state,
            "currency": currency,
            "current_offer": price,
            "target_price": target_price,
            "supplier_message": supplier_message,
            "sent_status": False,
            "awaiting_response": True,
        }

        if email_subject:
            data["email_subject"] = email_subject
        if email_body:
            data["email_body"] = email_body
        if email_action_id:
            data["email_action_id"] = email_action_id
        pass_fields: Dict[str, Any] = dict(data)

        supplier_watch_fields = self._build_supplier_watch_fields(
            context=context,
            rfq_id=rfq_id,
            supplier=supplier,
            drafts=draft_records,
            state=state,
        )
        supplier_responses: List[Dict[str, Any]] = []
        if supplier_watch_fields:
            pass_fields.update(supplier_watch_fields)
            await_response = bool(supplier_watch_fields.get("await_response"))
            if await_response and not next_agents:
                wait_results = self._await_supplier_responses(
                    context=context,
                    watch_payload=supplier_watch_fields,
                    state=state,
                )
                if wait_results is None:
                    logger.error(
                        "Supplier responses not received before timeout for rfq_id=%s supplier=%s",
                        rfq_id,
                        supplier,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_id,
                        "round": round_no,
                        "decision": decision,
                        "message": "Supplier response not received before timeout.",
                    }
                    return AgentOutput(
                        status=AgentStatus.FAILED,
                        data=error_payload,
                        error="supplier response timeout",
                    )
                supplier_responses = [res for res in wait_results if isinstance(res, dict)]
                if not supplier_responses:
                    logger.error(
                        "No supplier responses received while waiting for rfq_id=%s supplier=%s",
                        rfq_id,
                        supplier,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_id,
                        "round": round_no,
                        "decision": decision,
                        "message": "Missing supplier responses after wait.",
                    }
                    return AgentOutput(
                        status=AgentStatus.FAILED,
                        data=error_payload,
                        error="supplier response missing",
                    )
                known_ids: Set[str] = set()
                current_message = self._coerce_text(context.input_data.get("message_id"))
                if current_message:
                    known_ids.add(current_message.lower())
                previous_recorded = self._coerce_text(state.get("last_supplier_msg_id"))
                if previous_recorded:
                    known_ids.add(previous_recorded.lower())

                new_responses: List[Dict[str, Any]] = []
                for response in supplier_responses:
                    message_token = self._coerce_text(
                        response.get("message_id") or response.get("id")
                    )
                    if message_token and message_token.lower() in known_ids:
                        continue
                    if message_token:
                        known_ids.add(message_token.lower())
                    new_responses.append(response)

                increment_count = len(new_responses)
                if increment_count:
                    state["supplier_reply_count"] = int(
                        state.get("supplier_reply_count", 0)
                    ) + increment_count
                    last_reply = new_responses[-1]
                    last_message_id = last_reply.get("message_id") or last_reply.get("id")
                    if last_message_id:
                        state["last_supplier_msg_id"] = last_message_id
                elif not supplier_reply_registered:
                    # Ensure at least the last observed response is recorded.
                    last_reply = supplier_responses[-1]
                    last_message_id = last_reply.get("message_id") or last_reply.get("id")
                    if last_message_id:
                        state["last_supplier_msg_id"] = last_message_id
                supplier_responses = new_responses or supplier_responses
                state["awaiting_response"] = False
                self._save_session_state(rfq_id, supplier, state)
                public_state = self._public_state(state)
                data["session_state"] = public_state
                data["awaiting_response"] = False
                pass_fields["session_state"] = public_state
                pass_fields.pop("await_response", None)
                pass_fields.pop("await_all_responses", None)
                self._record_learning_snapshot(
                    context,
                    rfq_id,
                    supplier,
                    decision,
                    state,
                    False,
                    bool(new_responses) or supplier_reply_registered,
                )
            else:
                if "SupplierInteractionAgent" not in next_agents:
                    next_agents.append("SupplierInteractionAgent")
        if supplier_responses:
            data["supplier_responses"] = supplier_responses
            pass_fields["supplier_responses"] = supplier_responses

        if next_agents:
            merge_payload = dict(fallback_payload or {})
            merge_payload.setdefault("intent", "NEGOTIATION_COUNTER")
            merge_payload.setdefault("decision", decision)
            merge_payload["session_state"] = public_state
            if rfq_id is not None:
                merge_payload.setdefault("rfq_id", rfq_id)
            if supplier is not None:
                merge_payload.setdefault("supplier_id", supplier)
            supplier_name = context.input_data.get("supplier_name")
            if supplier_name:
                merge_payload.setdefault("supplier_name", supplier_name)
            merge_payload.setdefault("round", round_no)
            merge_payload.setdefault("counter_price", counter_price)
            merge_payload.setdefault("target_price", target_price)
            merge_payload.setdefault("current_offer", price)
            merge_payload.setdefault("current_offer_numeric", price)
            if currency:
                merge_payload.setdefault("currency", currency)
            merge_payload.setdefault("asks", decision.get("asks", []))
            merge_payload.setdefault("lead_time_request", decision.get("lead_time_request"))
            merge_payload.setdefault("rationale", decision.get("rationale"))
            merge_payload.setdefault("negotiation_message", negotiation_message)
            if supplier_message:
                merge_payload.setdefault("supplier_message", supplier_message)
            merge_payload.setdefault("supplier_reply_count", supplier_reply_count)
            if recipients:
                merge_payload.setdefault("recipients", recipients)
            sender = context.input_data.get("sender")
            if sender and not merge_payload.get("sender"):
                merge_payload["sender"] = sender

            sync_keys = {
                "supplier_id",
                "supplier_name",
                "rfq_id",
                "counter_price",
                "target_price",
                "current_offer",
                "current_offer_numeric",
                "currency",
                "asks",
                "lead_time_request",
                "rationale",
                "negotiation_message",
                "supplier_message",
                "supplier_reply_count",
                "session_state",
                "intent",
                "round",
                "recipients",
                "sender",
            }
            for key in sync_keys:
                if key in merge_payload:
                    data[key] = merge_payload[key]

            pass_fields.update(merge_payload)
        logger.info(
            "NegotiationAgent prepared counter round %s for supplier=%s rfq_id=%s", round_no, supplier, rfq_id
        )

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=data,
            pass_fields=pass_fields,
            next_agents=next_agents,
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
            return False, "AWAITING_SUPPLIER", "Awaiting supplier response."
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

    def _build_supplier_watch_fields(
        self,
        *,
        context: AgentContext,
        rfq_id: Optional[str],
        supplier: Optional[str],
        drafts: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not rfq_id or not supplier:
            return None

        candidate_drafts: List[Dict[str, Any]] = []
        for source in (drafts, context.input_data.get("drafts")):
            if not source:
                continue
            if isinstance(source, dict):
                source = [source]
            if not isinstance(source, list):
                continue
            for entry in source:
                if isinstance(entry, dict):
                    candidate_drafts.append(dict(entry))

        if not candidate_drafts:
            candidate_drafts.append({"rfq_id": rfq_id, "supplier_id": supplier})

        poll_interval = getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)

        watch_payload: Dict[str, Any] = {
            "await_response": True,
            "message": "",
            "body": "",
            "drafts": candidate_drafts,
            "rfq_id": rfq_id,
            "supplier_id": supplier,
            "response_poll_interval": poll_interval,
        }

        batch_limit = getattr(self.agent_nick.settings, "email_response_batch_limit", None)
        if batch_limit:
            try:
                batch_value = int(batch_limit)
                if batch_value > 0:
                    watch_payload["response_batch_limit"] = batch_value
            except Exception:  # pragma: no cover - defensive
                logger.debug("Invalid email_response_batch_limit=%s", batch_limit)

        timeout_setting = getattr(self.agent_nick.settings, "email_response_timeout_seconds", None)
        if timeout_setting:
            try:
                timeout_value = int(timeout_setting)
                if timeout_value > 0:
                    watch_payload["response_timeout"] = timeout_value
            except Exception:  # pragma: no cover - defensive
                logger.debug("Invalid email_response_timeout_seconds=%s", timeout_setting)

        if len(watch_payload["drafts"]) > 1:
            watch_payload["await_all_responses"] = True

        reply_count = state.get("supplier_reply_count")
        if reply_count is not None:
            watch_payload["supplier_reply_count"] = reply_count

        contact = context.input_data.get("from_address")
        if contact:
            watch_payload.setdefault("from_address", contact)

        sender = context.input_data.get("sender")
        if sender:
            watch_payload.setdefault("sender", sender)

        return watch_payload

    def _await_supplier_responses(
        self,
        *,
        context: AgentContext,
        watch_payload: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[List[Optional[Dict[str, Any]]]]:
        if not watch_payload.get("await_response"):
            return []

        try:
            supplier_agent = self._get_supplier_agent()
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("Unable to initialise supplier interaction agent for wait")
            return None

        timeout_default = getattr(
            self.agent_nick.settings, "email_response_timeout_seconds", 900
        )
        poll_default = getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)
        batch_default = getattr(self.agent_nick.settings, "email_response_batch_limit", 5)

        timeout_raw = watch_payload.get("response_timeout") or timeout_default
        poll_raw = watch_payload.get("response_poll_interval") or poll_default
        batch_raw = watch_payload.get("response_batch_limit") or batch_default

        timeout = self._positive_int(timeout_raw, fallback=timeout_default)
        poll_interval = self._positive_int(poll_raw, fallback=poll_default)
        batch_limit = self._positive_int(batch_raw, fallback=batch_default)

        draft_entries: List[Dict[str, Any]] = []
        for draft in watch_payload.get("drafts", []):
            if isinstance(draft, dict):
                draft_entries.append(dict(draft))

        if not draft_entries:
            fallback_entry = {
                "rfq_id": watch_payload.get("rfq_id"),
                "supplier_id": watch_payload.get("supplier_id"),
            }
            if fallback_entry.get("rfq_id") or fallback_entry.get("supplier_id"):
                draft_entries.append(fallback_entry)

        if not draft_entries:
            logger.warning("No draft context available while awaiting supplier response")
            return None

        await_all = bool(watch_payload.get("await_all_responses") and len(draft_entries) > 1)

        try:
            if await_all:
                return supplier_agent.wait_for_multiple_responses(
                    draft_entries,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    enable_negotiation=False,
                )

            target = draft_entries[0]
            recipient_hint = target.get("receiver") or target.get("recipient_email")
            if not recipient_hint:
                recipients_field = target.get("recipients")
                if isinstance(recipients_field, list) and recipients_field:
                    recipient_hint = recipients_field[0]
                elif isinstance(recipients_field, str):
                    recipient_hint = recipients_field

            return [
                supplier_agent.wait_for_response(
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    rfq_id=target.get("rfq_id"),
                    supplier_id=target.get("supplier_id"),
                    subject_hint=target.get("subject"),
                    from_address=recipient_hint,
                    enable_negotiation=False,
                )
            ]
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed while waiting for supplier responses (rfq_id=%s, supplier=%s)",
                watch_payload.get("rfq_id"),
                watch_payload.get("supplier_id"),
            )
            return None

    def _get_supplier_agent(self) -> "SupplierInteractionAgent":
        if self._supplier_agent is None:
            from agents.supplier_interaction_agent import SupplierInteractionAgent

            self._supplier_agent = SupplierInteractionAgent(self.agent_nick)
        return self._supplier_agent

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

    def _record_learning_snapshot(
        self,
        context: AgentContext,
        rfq_id: Optional[str],
        supplier: Optional[str],
        decision: Dict[str, Any],
        state: Dict[str, Any],
        awaiting_response: bool,
        supplier_reply_registered: bool,
    ) -> None:
        repository = getattr(self, "learning_repository", None)
        if repository is None:
            return
        try:
            repository.record_negotiation_learning(
                workflow_id=getattr(context, "workflow_id", None),
                rfq_id=rfq_id,
                supplier_id=supplier,
                decision=decision,
                state=state,
                awaiting_response=awaiting_response,
                supplier_reply_registered=supplier_reply_registered,
            )
        except Exception:
            logger.debug(
                "Failed to capture negotiation learning for %s/%s", rfq_id, supplier,
                exc_info=True,
            )

    def _collect_recipient_candidates(self, context: AgentContext) -> List[str]:
        seen: Set[str] = set()
        recipients: List[str] = []
        payload = context.input_data

        def _append(candidate: Any) -> None:
            email = self._coerce_text(candidate)
            if not email:
                return
            key = email.lower()
            if key in seen:
                return
            seen.add(key)
            recipients.append(email)

        for key in (
            "recipients",
            "recipient_email",
            "recipient",
            "supplier_contact_email",
            "supplier_email",
            "email",
            "contact_email",
            "contact_email_1",
            "contact_email_2",
        ):
            value = payload.get(key)
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _append(item)
            else:
                _append(value)

        contacts = payload.get("supplier_contacts")
        if isinstance(contacts, list):
            for contact in contacts:
                if not isinstance(contact, dict):
                    continue
                _append(contact.get("email"))
                _append(contact.get("contact_email"))

        drafts = payload.get("drafts")
        if isinstance(drafts, list):
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                _append(draft.get("recipient_email"))
                _append(draft.get("receiver"))
                draft_recipients = draft.get("recipients")
                if isinstance(draft_recipients, list):
                    for item in draft_recipients:
                        _append(item)

        return recipients

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

    def _positive_int(self, value: Any, *, fallback: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return fallback
        return parsed if parsed > 0 else fallback

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

    def _build_email_agent_payload(
        self,
        context: AgentContext,
        draft_payload: Dict[str, Any],
        decision: Dict[str, Any],
        state: Dict[str, Any],
        negotiation_message: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(draft_payload, dict):
            return None
        payload = dict(draft_payload)
        payload.setdefault("decision", decision)
        payload.setdefault("session_state", self._public_state(state))
        payload.setdefault("negotiation_message", negotiation_message)
        payload.setdefault("supplier_reply_count", state.get("supplier_reply_count", 0))
        supplier_name = context.input_data.get("supplier_name")
        if supplier_name:
            payload.setdefault("supplier_name", supplier_name)
        contact = context.input_data.get("from_address")
        if contact:
            payload.setdefault("recipients", [contact])
        payload.setdefault("intent", "NEGOTIATION_COUNTER")
        return payload

    def _invoke_email_drafting_agent(
        self,
        parent_context: AgentContext,
        payload: Dict[str, Any],
    ) -> Optional[AgentOutput]:
        try:
            if self._email_agent is None:
                self._email_agent = EmailDraftingAgent(self.agent_nick)
            email_context = AgentContext(
                workflow_id=parent_context.workflow_id,
                agent_id="EmailDraftingAgent",
                user_id=parent_context.user_id,
                input_data=payload,
                parent_agent=parent_context.agent_id,
                routing_history=list(parent_context.routing_history),
            )
            return self._email_agent.execute(email_context)
        except Exception:
            logger.exception("Failed to invoke EmailDraftingAgent for negotiation counter")
            return None

