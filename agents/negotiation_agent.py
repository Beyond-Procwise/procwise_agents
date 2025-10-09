import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from agents.email_drafting_agent import EmailDraftingAgent, DEFAULT_NEGOTIATION_SUBJECT
from agents.negotiation_pricer import NegotiationContext, SupplierSignals, plan_counter

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from agents.supplier_interaction_agent import SupplierInteractionAgent
from utils.gpu import configure_gpu

from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------
# Tunables & feature toggles
# -----------------------------
MAX_SUPPLIER_REPLIES = int(os.getenv("NEG_MAX_SUPPLIER_REPLIES", "3"))
LLM_ENABLED = os.getenv("NEG_ENABLE_LLM", "1").strip() not in {"0", "false", "False"}
LLM_MODEL = os.getenv("NEG_LLM_MODEL", "llama3.2:latest")
COST_OF_CAPITAL_APR = float(os.getenv("NEG_COST_OF_CAPITAL_APR", "0.12"))
LEAD_TIME_VALUE_PCT_PER_WEEK = float(os.getenv("NEG_LT_VALUE_PCT_PER_WEEK", "0.01"))
AGGRESSIVE_FIRST_COUNTER_PCT = float(os.getenv("NEG_FIRST_COUNTER_AGGR_PCT", "0.12"))
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

LEVER_CATEGORIES = {
    "COMMERCIAL",
    "OPERATIONAL",
    "RISK",
    "STRATEGIC",
    "RELATIONAL",
}

TRADE_OFF_HINTS = {
    "Commercial": "May require volume commitments, stepped pricing, or altered cash flow.",
    "Operational": "May reduce supplier flexibility or need shared planning resources.",
    "Risk": "Could introduce legal negotiation overhead or stricter enforcement costs.",
    "Strategic": "Requires executive sponsorship and potential co-investment or exclusivity.",
    "Relational": "Demands governance time and tighter alignment of internal stakeholders.",
}

PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "resources" / "reference_data" / "negotiation_playbook.json"


def compute_decision(
    payload: Dict[str, Any],
    supplier_message_text: str = "",
    offer_prev: Optional[float] = None,
) -> Dict[str, Any]:
    ask_disc_raw = payload.get("ask_early_pay_disc", 0.02)
    try:
        ask_disc = float(ask_disc_raw) if ask_disc_raw is not None else None
    except (TypeError, ValueError):
        ask_disc = None

    walkaway_raw = payload.get("walkaway_price")
    try:
        walkaway = float(walkaway_raw) if walkaway_raw is not None else None
    except (TypeError, ValueError):
        walkaway = None

    current_offer = float(payload["current_offer"])
    target_price = float(payload["target_price"])
    round_idx = int(payload.get("round", 1))

    ctx = NegotiationContext(
        current_offer=current_offer,
        target_price=target_price,
        round_index=round_idx,
        currency=payload.get("currency"),
        aggressiveness=0.75,
        leverage=0.6,
        urgency=0.3,
        risk_buffer_pct=0.06,
        min_abs_buffer=3.0,
        step_pct_of_gap=0.12,
        min_abs_step=4.0,
        max_rounds=int(payload.get("max_rounds", 3)),
        walkaway_price=walkaway,
        ask_early_pay_disc=ask_disc,
        ask_lead_time_keep=bool(payload.get("ask_lead_time_keep", True)),
    )
    signals = SupplierSignals(
        offer_prev=offer_prev,
        offer_new=current_offer,
        message_text=supplier_message_text or "",
    )
    return plan_counter(ctx, signals)


def decide_strategy(
    payload: Dict[str, Any],
    *,
    lead_weeks: Optional[float] = None,
    constraints: Optional[List[str]] = None,
    supplier_message: Optional[str] = None,
    offer_prev: Optional[float] = None,
) -> Dict[str, Any]:
    """Plan price counter proposals using deterministic negotiation heuristics."""

    current_offer = payload.get("current_offer")
    target = payload.get("target_price")
    if current_offer is None or target is None:
        return {
            "strategy": "clarify",
            "counter_price": None,
            "asks": ["Confirm unit price, currency, tiered price @ 100/250/500."],
            "lead_time_request": None,
            "rationale": "Price missing; request structured quote.",
        }

    plan = compute_decision(payload, supplier_message_text=supplier_message or "", offer_prev=offer_prev)

    decision: Dict[str, Any] = {
        "strategy": plan.get("decision", "counter"),
        "counter_price": plan.get("counter_price"),
        "asks": plan.get("asks", []),
        "lead_time_request": plan.get("lead_time_request"),
        "rationale": plan.get("message", ""),
        "decision_origin": "plan_counter",
        "price_plan_locked": True,
    }

    if lead_weeks and lead_weeks > 3:
        lead_req = plan.get("lead_time_request") or "≤ 2 weeks or split shipment (20–30% now, balance later)"
        decision["lead_time_request"] = lead_req
        if plan.get("asks") is None:
            decision["asks"] = []
        decision["asks"] = list(dict.fromkeys((decision.get("asks") or []) + ["Split shipment or expedite option"]))

    if constraints:
        decision.setdefault("constraints", constraints)

    decision.setdefault("decision_log", plan.get("log", []))
    return decision


class NegotiationAgent(BaseAgent):
    """Generate deterministic negotiation decisions and coordinate counter drafting."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self._state_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._email_agent: Optional[EmailDraftingAgent] = None
        self._supplier_agent: Optional["SupplierInteractionAgent"] = None
        self._state_schema_checked = False
        self._playbook_cache: Optional[Dict[str, Any]] = None

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

        # Load state before subject normalization (bug fix)
        state, _ = self._load_session_state(rfq_id, supplier)

        round_no = int(state.get("current_round", 1))
        if isinstance(raw_round, (int, float)):
            try:
                round_no = max(round_no, int(raw_round))
            except (TypeError, ValueError):
                round_no = max(round_no, 1)
        state["current_round"] = max(round_no, 1)

        incoming_subject = self._coerce_text(context.input_data.get("subject"))
        base_candidate = self._normalise_base_subject(incoming_subject)
        if base_candidate and not state.get("base_subject"):
            state["base_subject"] = base_candidate

        signals = self._extract_negotiation_signals(
            supplier_message=supplier_message,
            snippets=supplier_snippets,
            market=context.input_data.get("market_context") or {},
            performance=context.input_data.get("supplier_performance") or {},
        )

        zopa = self._estimate_zopa(
            price=price,
            target=target_price,
            history=context.input_data.get("history") or {},
            benchmarks=context.input_data.get("benchmarks") or {},
            should_cost=context.input_data.get("should_cost"),
            signals=signals,
        )

        pricing_payload: Dict[str, Any] = {
            "current_offer": price,
            "target_price": target_price,
            "round": raw_round,
            "currency": currency,
            "max_rounds": context.input_data.get("max_rounds") or 3,
            "walkaway_price": self._coerce_float(context.input_data.get("walkaway_price")),
            "ask_early_pay_disc": context.input_data.get("ask_early_pay_disc", 0.02),
            "ask_lead_time_keep": context.input_data.get("ask_lead_time_keep", True),
        }
        offer_prev = self._coerce_float(
            context.input_data.get("previous_offer")
            or context.input_data.get("supplier_previous_offer")
            or context.input_data.get("last_supplier_offer")
        )

        decision = decide_strategy(
            pricing_payload,
            lead_weeks=lead_weeks,
            constraints=constraints,
            supplier_message=supplier_message,
            offer_prev=offer_prev,
        )
        decision = self._adaptive_strategy(
            base_decision=decision,
            zopa=zopa,
            signals=signals,
            round_hint=raw_round,
            lead_weeks=lead_weeks,
            target_price=target_price,
            price=price,
        )
        decision.setdefault("strategy", "clarify")
        decision.setdefault("counter_price", None)
        decision.setdefault("asks", [])
        decision.setdefault("lead_time_request", None)
        decision.setdefault("rationale", "")

        playbook_context = self._resolve_playbook_context(context, decision)
        play_recommendations = playbook_context.get("plays", [])
        if play_recommendations:
            decision["play_recommendations"] = play_recommendations
            descriptor = playbook_context.get("descriptor")
            if descriptor:
                decision["playbook_descriptor"] = descriptor
            style_used = playbook_context.get("style")
            if style_used:
                decision["playbook_style"] = style_used
            lever_focus = playbook_context.get("lever_priorities")
            if lever_focus:
                decision.setdefault("lever_priorities", lever_focus)
            examples = playbook_context.get("examples")
            if examples:
                decision["playbook_examples"] = examples
        else:
            decision.setdefault("play_recommendations", [])

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

        decision["round"] = round_no

        final_offer_reason = self._detect_final_offer(supplier_message, supplier_snippets)
        should_continue, new_status, halt_reason = self._should_continue(
            state, supplier_reply_registered, final_offer_reason
        )

        savings_score = 0.0
        if price and price > 0 and target_price is not None:
            try:
                savings_score = (price - target_price) / float(price)
            except ZeroDivisionError:
                savings_score = 0.0

        decision_log = self._build_decision_log(supplier, rfq_id, price, target_price, decision)

        draft_records: List[Dict[str, Any]] = []

        counter_options: List[Dict[str, Any]] = []

        if not should_continue:
            state["status"] = new_status
            state["awaiting_response"] = halt_reason == "Awaiting supplier response."
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
                "counter_proposals": counter_options,
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
                "play_recommendations": play_recommendations,
                "playbook_descriptor": playbook_context.get("descriptor"),
                "playbook_examples": playbook_context.get("examples"),
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

        optimized = self._optimize_multi_issue(
            price=price,
            currency=currency,
            target=target_price,
            lead_weeks=lead_weeks,
            weights=context.input_data.get("weights") or {},
            policy=context.input_data.get("policy_guidance") or {},
            constraints=context.input_data.get("hard_constraints") or {},
            zopa=zopa,
            signals=signals,
            round_no=round_no,
        )
        overrides = optimized.get("decision_overrides", {}) or {}
        if decision.get("price_plan_locked"):
            overrides = {k: v for k, v in overrides.items() if k != "counter_price"}
        decision.update(overrides)
        counter_options = optimized.get("counter_options") or []
        if not counter_options and decision.get("counter_price") is not None:
            counter_options = [{"price": decision["counter_price"], "terms": None, "bundle": None}]

        negotiation_message = self._build_summary(
            rfq_id,
            decision,
            price,
            target_price,
            currency,
            round_no,
            playbook_context=playbook_context,
        )

        if play_recommendations:
            negotiation_message = self._append_playbook_recommendations(
                negotiation_message,
                play_recommendations,
                playbook_context,
            )

        if rfq_id and supplier:
            self._store_session(rfq_id, supplier, round_no, decision.get("counter_price"))

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
            "counter_price": decision.get("counter_price"),
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
            "play_recommendations": play_recommendations,
            "playbook_descriptor": playbook_context.get("descriptor"),
            "playbook_examples": playbook_context.get("examples"),
            "counter_options": counter_options,
        }

        draft_payload.setdefault("subject", self._format_negotiation_subject(state))

        recipients = self._collect_recipient_candidates(context)
        if recipients:
            draft_payload.setdefault("recipients", recipients)

        draft_metadata = {
            "counter_price": decision.get("counter_price"),
            "target_price": target_price,
            "current_offer": price,
            "round": round_no,
            "supplier_reply_count": supplier_reply_count,
            "strategy": decision.get("strategy"),
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "intent": "NEGOTIATION_COUNTER",
            "play_recommendations": play_recommendations,
        }

        email_action_id: Optional[str] = None
        email_subject: Optional[str] = None
        email_body: Optional[str] = None
        draft_records: List[Dict[str, Any]] = []
        next_agents: List[str] = []

        subject_seed = self._coerce_text(draft_payload.get("subject"))
        if not subject_seed:
            subject_seed = self._format_negotiation_subject(state)

        draft_stub = {
            "rfq_id": rfq_id,
            "supplier_id": supplier,
            "intent": "NEGOTIATION_COUNTER",
            "metadata": draft_metadata,
            "negotiation_message": negotiation_message,
            "counter_proposals": counter_options,
            "sent_status": False,
            "thread_index": round_no,
            "subject": subject_seed,
        }
        if currency:
            draft_stub["currency"] = currency
        if supplier_snippets:
            draft_stub["supplier_snippets"] = supplier_snippets
        if play_recommendations:
            draft_stub["play_recommendations"] = play_recommendations
        descriptor = playbook_context.get("descriptor")
        if descriptor:
            draft_stub["playbook_descriptor"] = descriptor
        draft_stub["payload"] = draft_payload
        if recipients:
            draft_stub["recipients"] = recipients
        if negotiation_message:
            draft_stub.setdefault("body", negotiation_message)

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

        if email_subject:
            base_from_email = self._normalise_base_subject(email_subject)
            if base_from_email and not state.get("base_subject"):
                state["base_subject"] = base_from_email
        elif subject_seed and not state.get("base_subject"):
            fallback_base = self._normalise_base_subject(subject_seed)
            if fallback_base:
                state["base_subject"] = fallback_base

        if email_body and not state.get("initial_body"):
            state["initial_body"] = email_body
        elif not state.get("initial_body") and negotiation_message:
            state["initial_body"] = negotiation_message

        if not email_subject:
            email_subject = subject_seed
        if not email_body and negotiation_message:
            email_body = negotiation_message

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
            "play_recommendations": play_recommendations,
            "playbook_descriptor": playbook_context.get("descriptor"),
            "playbook_examples": playbook_context.get("examples"),
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
            merge_payload.setdefault("counter_price", decision.get("counter_price"))
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
            if play_recommendations:
                merge_payload.setdefault("play_recommendations", play_recommendations)
            descriptor = playbook_context.get("descriptor")
            if descriptor:
                merge_payload.setdefault("playbook_descriptor", descriptor)

            if counter_options:
                merge_payload.setdefault("counter_options", counter_options)

            for key, value in merge_payload.items():
                data[key] = value

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
    # NEW: Intelligence helpers
    # ------------------------------------------------------------------
    def _extract_negotiation_signals(
        self,
        *,
        supplier_message: Optional[str],
        snippets: List[str],
        market: Dict[str, Any],
        performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = " ".join([t for t in ([supplier_message] + snippets) if t])[:8000]
        signals: Dict[str, Any] = {
            "finality_hint": False,
            "capacity_tight": False,
            "moq": None,
            "payment_terms_hint": None,
            "delivery_flex": None,
            "alt_part_offered": False,
            "tone": "neutral",
            "concession_band_pct": None,
        }

        lowered = text.lower() if text else ""
        if any(p in lowered for p in ("capacity", "backlog", "constrained")):
            signals["capacity_tight"] = True
        if "moq" in lowered:
            m = re.search(r"moq[^0-9]*([0-9]{2,})", lowered)
            if m:
                try:
                    signals["moq"] = int(m.group(1))
                except Exception:
                    pass
        if any(x in lowered for x in ("net-30", "net30", "net 30", "net-45", "net 45", "early payment")):
            signals["payment_terms_hint"] = "tradeable"
        if any(x in lowered for x in ("expedite", "split shipment", "partial", "air freight")):
            signals["delivery_flex"] = "possible"
        if any(x in lowered for x in ("alternate", "alternative", "equivalent", "substitute", "brand b")):
            signals["alt_part_offered"] = True
        if any(x in lowered for x in ("cannot go lower", "final", "last price", "our best price", "rock bottom")):
            signals["finality_hint"] = True
            signals["tone"] = "firm"

        if LLM_ENABLED and text:
            try:  # pragma: no cover - optional dependency
                import ollama  # type: ignore

                prompt = (
                    "Extract JSON with keys: tone (firm/flexible/neutral), finality_hint (bool), "
                    "capacity_tight (bool), moq (int or null), payment_terms_hint (tradeable/fixed/null), "
                    "delivery_flex (possible/unlikely/null), concession_band_pct (float or null). Only return JSON.\n\n"
                    f"Text:\n{text}"
                )
                resp = ollama.generate(model=LLM_MODEL, prompt=prompt, options={"temperature": 0.1})
                content = resp.get("response") or ""
                if "{" in content and "}" in content:
                    content = content[content.find("{") : content.rfind("}") + 1]
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        for key in signals:
                            if key in parsed and parsed[key] is not None:
                                signals[key] = parsed[key]
            except Exception:
                logger.debug("LLM signal extraction skipped/failed", exc_info=True)

        signals["market"] = market or {}
        signals["performance"] = performance or {}
        return signals

    def _estimate_zopa(
        self,
        *,
        price: Optional[float],
        target: Optional[float],
        history: Dict[str, Any],
        benchmarks: Dict[str, Any],
        should_cost: Optional[float],
        signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        buyer_max = None
        if target is not None and price is not None:
            buyer_max = min(target, price)
        elif target is not None:
            buyer_max = target
        elif price is not None:
            buyer_max = price

        candidates: List[float] = []
        if should_cost is not None and should_cost > 0:
            candidates.append(float(should_cost))
        bench_low = self._coerce_float(benchmarks.get("p10") or benchmarks.get("low"))
        if bench_low:
            candidates.append(bench_low)
        hist_min = self._coerce_float(history.get("min_accepted_price"))
        if hist_min:
            candidates.append(hist_min)
        supplier_floor = min(candidates) if candidates else (price * 0.85 if price else None)

        if supplier_floor:
            if signals.get("capacity_tight"):
                supplier_floor *= 1.03
            if signals.get("tone") == "firm":
                supplier_floor *= 1.01

        concession = signals.get("concession_band_pct") or 0.05
        entry = None
        if price:
            entry = round(price * (1 - max(0.03, min(0.12, concession))), 2)

        return {
            "buyer_max": float(buyer_max) if buyer_max is not None else None,
            "supplier_floor": float(supplier_floor) if supplier_floor is not None else None,
            "entry_counter": entry,
        }

    def _adaptive_strategy(
        self,
        *,
        base_decision: Dict[str, Any],
        zopa: Dict[str, Any],
        signals: Dict[str, Any],
        round_hint: Any,
        lead_weeks: Optional[float],
        target_price: Optional[float],
        price: Optional[float],
    ) -> Dict[str, Any]:
        decision = dict(base_decision)
        price_locked = bool(base_decision.get("price_plan_locked"))
        current_round = int(round_hint) if isinstance(round_hint, (int, float)) else 1

        if signals.get("finality_hint"):
            decision["strategy"] = "package-trade"
            decision["asks"] = (decision.get("asks") or []) + [
                "Consider payment-terms trade (early pay discount vs Net45/60)",
                "Bundle volume commitment for stepped pricing",
            ]
            if lead_weeks and lead_weeks > 3:
                decision["lead_time_request"] = "Split shipment or expedite slot if possible"
            return decision

        buyer_max = zopa.get("buyer_max")
        supplier_floor = zopa.get("supplier_floor")
        entry = zopa.get("entry_counter")
        if not price_locked and price and target_price and entry:
            if current_round == 1:
                decision["counter_price"] = min(entry, (price + target_price) / 2)
            else:
                if buyer_max and supplier_floor and supplier_floor < buyer_max:
                    midpoint = (buyer_max + supplier_floor) / 2
                    decision["counter_price"] = round(
                        min(decision.get("counter_price") or entry, midpoint), 2
                    )
                else:
                    decision["counter_price"] = round(
                        min(decision.get("counter_price") or entry, (price + target_price) / 2), 2
                    )

        asks = decision.get("asks", [])
        if signals.get("payment_terms_hint") == "tradeable":
            asks.append("Offer early payment for additional 1–2% discount")
        if signals.get("delivery_flex") == "possible" and (lead_weeks or 0) > 3:
            asks.append("Split shipment or expedite window swap")
        decision["asks"] = list(dict.fromkeys(asks))
        decision["price_plan_locked"] = price_locked
        return decision

    def _optimize_multi_issue(
        self,
        *,
        price: Optional[float],
        currency: Optional[str],
        target: Optional[float],
        lead_weeks: Optional[float],
        weights: Dict[str, Any],
        policy: Dict[str, Any],
        constraints: Dict[str, Any],
        zopa: Dict[str, Any],
        signals: Dict[str, Any],
        round_no: int,
    ) -> Dict[str, Any]:
        w_price = float(weights.get("price", 0.5))
        w_delivery = float(weights.get("delivery", 0.2))
        w_risk = float(weights.get("risk", 0.2))
        w_terms = float(weights.get("terms", 0.1))
        normalizer = max(w_price + w_delivery + w_risk + w_terms, 1e-9)
        w_price, w_delivery, w_risk, w_terms = [w / normalizer for w in (w_price, w_delivery, w_risk, w_terms)]

        _ = policy, constraints  # reserved for future constraint-aware adjustments

        buyer_max = zopa.get("buyer_max")
        supplier_floor = zopa.get("supplier_floor")
        entry = zopa.get("entry_counter")

        price_candidates: List[float] = []
        if target and price:
            price_candidates.extend(
                [
                    round(max(supplier_floor or 0, target * 0.98), 2),
                    round(max(supplier_floor or 0, (price + target) / 2 * 0.97), 2),
                ]
            )
        if entry:
            price_candidates.append(round(entry, 2))
        if price and target:
            price_candidates.append(
                round(max(supplier_floor or 0, price * (1 - AGGRESSIVE_FIRST_COUNTER_PCT)), 2)
            )

        sanitized: List[float] = []
        for candidate in price_candidates:
            if candidate is None:
                continue
            value = candidate
            if buyer_max and value > buyer_max:
                value = buyer_max
            if supplier_floor and value < supplier_floor:
                value = supplier_floor
            sanitized.append(round(value, 2))
        price_candidates = sorted(set(sanitized))

        daily_rate = COST_OF_CAPITAL_APR / 365.0
        term_pkgs = [
            {"label": "Net15 with 2% disc", "apr_equiv": -0.02},
            {"label": "Net30 std", "apr_equiv": 0.0},
            {"label": "Net45 std", "apr_equiv": daily_rate * 15},
            {"label": "Net60 std", "apr_equiv": daily_rate * 30},
        ]

        if lead_weeks is not None and lead_weeks > 3:
            lead_packages = [{"label": "≤2w or split shipment", "weeks_gain": (lead_weeks - 2)}]
        else:
            lead_packages = [{"label": "as quoted", "weeks_gain": 0}]

        volume_tiers = [None, 100, 250, 500]

        option_grid: List[Dict[str, Any]] = []
        for price_option in price_candidates[:5]:
            for term_option in term_pkgs:
                for lead_option in lead_packages:
                    for tier in volume_tiers:
                        option_grid.append(
                            {
                                "price": price_option,
                                "terms": term_option,
                                "lead": lead_option,
                                "volume": tier,
                            }
                        )

        best_option = None
        best_score = -1e9
        for option in option_grid:
            counter_price = option["price"]
            price_score = 0.0
            if target and counter_price:
                price_score = (target - counter_price) / max(target, 1e-9)

            terms_score = -float(option["terms"]["apr_equiv"])

            delivery_score = 0.0
            weeks_gain = float(option["lead"].get("weeks_gain") or 0)
            delivery_score = weeks_gain * LEAD_TIME_VALUE_PCT_PER_WEEK

            risk_score = 0.0
            performance = signals.get("performance") or {}
            on_time = self._coerce_float(
                performance.get("on_time_delivery")
                or performance.get("delivery_score")
                or performance.get("otif")
            )
            if on_time is not None and on_time < 0.9:
                risk_score += 0.02
            if signals.get("capacity_tight") and "split" in option["lead"]["label"].lower():
                risk_score += 0.02

            total_score = (
                (w_price * price_score)
                + (w_terms * terms_score)
                + (w_delivery * delivery_score)
                + (w_risk * risk_score)
            )

            if total_score > best_score:
                best_score = total_score
                best_option = option

        decision_overrides: Dict[str, Any] = {}
        counter_options: List[Dict[str, Any]] = []
        if best_option:
            decision_overrides["counter_price"] = round(best_option["price"], 2)
            asks = decision_overrides.get("asks", [])
            if "≤2w" in best_option["lead"]["label"]:
                asks.append("≤ 2 weeks or split shipment (20–30% now, balance later)")
            if "2% disc" in best_option["terms"]["label"]:
                asks.append("2% discount for Net15 / early payment")
            if best_option["volume"]:
                asks.append(f"Tiers @ {best_option['volume']}/250/500 for stepped pricing")
            decision_overrides["asks"] = list(dict.fromkeys(asks))
            counter_options.append(
                {
                    "price": round(best_option["price"], 2),
                    "terms": best_option["terms"]["label"],
                    "bundle": {
                        "lead_time": best_option["lead"]["label"],
                        "volume_tier": best_option["volume"],
                    },
                }
            )

        return {"decision_overrides": decision_overrides, "counter_options": counter_options}

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
            "base_subject": None,
            "initial_body": None,
        }

    def _public_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "supplier_reply_count": int(state.get("supplier_reply_count", 0)),
            "current_round": int(state.get("current_round", 1)),
            "status": state.get("status", "ACTIVE"),
            "awaiting_response": bool(state.get("awaiting_response", False)),
        }

    def _ensure_state_schema(self) -> None:
        if getattr(self, "_state_schema_checked", False):
            return
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            self._state_schema_checked = True
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS base_subject TEXT"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS initial_body TEXT"
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.debug("failed to ensure negotiation state schema", exc_info=True)
        finally:
            self._state_schema_checked = True

    @staticmethod
    def _normalise_base_subject(subject: Optional[str]) -> Optional[str]:
        if subject is None:
            return None
        if not isinstance(subject, str):
            try:
                subject = str(subject)
            except Exception:
                return None
        trimmed = subject.strip()
        if not trimmed:
            return None
        trimmed = re.sub(r"(?i)^(re|fw|fwd):\s*", "", trimmed)
        cleaned = EmailDraftingAgent._strip_rfq_identifier_tokens(trimmed)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("-–: ")
        return cleaned or None

    def _format_negotiation_subject(self, state: Dict[str, Any]) -> str:
        base = self._coerce_text(state.get("base_subject"))
        if base:
            if re.match(r"(?i)^(re|fw|fwd):", base.strip()):
                return base.strip()
            return f"Re: {base}".strip()
        return DEFAULT_NEGOTIATION_SUBJECT


    def _load_session_state(
        self, rfq_id: Optional[str], supplier: Optional[str]
    ) -> Tuple[Dict[str, Any], bool]:
        if not rfq_id or not supplier:
            return self._default_state(), False
        self._ensure_state_schema()
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
                               last_supplier_msg_id, last_agent_msg_id, last_email_sent_at,
                               base_subject, initial_body
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
                            base_subject,
                            initial_body,
                        ) = row
                        if isinstance(base_subject, (bytes, bytearray, memoryview)):
                            try:
                                base_subject = base_subject.decode("utf-8", errors="ignore")
                            except Exception:
                                base_subject = str(base_subject)
                        if isinstance(initial_body, (bytes, bytearray, memoryview)):
                            try:
                                initial_body = initial_body.decode("utf-8", errors="ignore")
                            except Exception:
                                initial_body = str(initial_body)
                        state["base_subject"] = base_subject
                        state["initial_body"] = initial_body
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
        self._ensure_state_schema()
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
                            base_subject,
                            initial_body,
                            updated_on
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (rfq_id, supplier_id) DO UPDATE SET
                            supplier_reply_count = EXCLUDED.supplier_reply_count,
                            current_round = EXCLUDED.current_round,
                            status = EXCLUDED.status,
                            awaiting_response = EXCLUDED.awaiting_response,
                            last_supplier_msg_id = EXCLUDED.last_supplier_msg_id,
                            last_agent_msg_id = EXCLUDED.last_agent_msg_id,
                            last_email_sent_at = EXCLUDED.last_email_sent_at,
                            base_subject = EXCLUDED.base_subject,
                            initial_body = EXCLUDED.initial_body,
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
                            state.get("base_subject"),
                            state.get("initial_body"),
                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist negotiation session state")

    # ------------------------------------------------------------------
    # Negotiation playbook helpers
    # ------------------------------------------------------------------

    def _resolve_playbook_context(
        self, context: AgentContext, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        playbook = self._load_playbook()
        if not playbook:
            return {"plays": [], "lever_priorities": []}

        supplier_type = self._normalise_supplier_type(
            context.input_data.get("supplier_type")
            or context.input_data.get("supplier_segment")
            or decision.get("supplier_type")
        )
        negotiation_style = self._normalise_negotiation_style(
            context.input_data.get("negotiation_style")
            or context.input_data.get("style")
            or decision.get("negotiation_style")
        )

        if not supplier_type or supplier_type not in playbook:
            return {"plays": [], "lever_priorities": []}

        supplier_entry = playbook.get(supplier_type, {})
        styles = supplier_entry.get("styles", {})
        if not negotiation_style or negotiation_style not in styles:
            return {
                "plays": [],
                "descriptor": supplier_entry.get("descriptor"),
                "examples": supplier_entry.get("examples", []),
                "lever_priorities": [],
                "style": negotiation_style,
                "supplier_type": supplier_type,
            }

        lever_priorities = self._resolve_lever_priorities(
            context,
            styles[negotiation_style],
        )

        if not lever_priorities:
            lever_priorities = list(styles[negotiation_style].keys())

        policy_guidance = self._extract_policy_guidance(context)
        supplier_performance = context.input_data.get("supplier_performance")
        if not isinstance(supplier_performance, dict):
            supplier_performance = {}
        market_context = context.input_data.get("market_context")
        if not isinstance(market_context, dict):
            market_context = {}

        plays: List[Dict[str, Any]] = []
        style_plays = styles[negotiation_style]
        for lever in lever_priorities:
            lever_plays = style_plays.get(lever)
            if not isinstance(lever_plays, list):
                continue
            for idx, play_text in enumerate(lever_plays):
                if not isinstance(play_text, str) or not play_text.strip():
                    continue
                base_score = 1.0 + (idx * 0.01)
                policy_score, policy_notes = self._score_policy_alignment(lever, policy_guidance)
                performance_score, performance_notes = self._score_supplier_performance(
                    lever, supplier_performance
                )
                market_score, market_notes = self._score_market_context(lever, market_context)
                total_score = base_score + policy_score + performance_score + market_score
                rationale = self._compose_play_rationale(
                    supplier_entry.get("descriptor"),
                    negotiation_style,
                    lever,
                    policy_notes,
                    performance_notes,
                    market_notes,
                )
                plays.append(
                    {
                        "supplier_type": supplier_type,
                        "style": negotiation_style,
                        "lever": lever,
                        "play": play_text.strip(),
                        "score": round(total_score, 4),
                        "policy_alignment": policy_notes,
                        "performance_signals": performance_notes,
                        "market_signals": market_notes,
                        "rationale": rationale,
                        "trade_offs": TRADE_OFF_HINTS.get(
                            lever, "Monitor implementation impact across stakeholders."
                        ),
                    }
                )

        plays.sort(key=lambda item: (-item["score"], item.get("lever", ""), item.get("play", "")))
        top_plays = plays[:10]

        return {
            "plays": top_plays,
            "descriptor": supplier_entry.get("descriptor"),
            "examples": supplier_entry.get("examples", []),
            "style": negotiation_style,
            "supplier_type": supplier_type,
            "lever_priorities": lever_priorities,
        }

    def _resolve_lever_priorities(self, context: AgentContext, style_plays: Dict[str, Any]) -> List[str]:
        raw_candidates = (
            context.input_data.get("lever_priorities")
            or context.input_data.get("lever_focus")
            or context.input_data.get("lever_preferences")
            or context.input_data.get("preferred_levers")
        )
        priorities: List[str] = []
        for candidate in self._ensure_list(raw_candidates):
            lever = self._normalise_lever_category(candidate)
            if lever and lever in style_plays and lever not in priorities:
                priorities.append(lever)
        if priorities:
            return priorities
        # Fall back to context-provided comma separated string
        if isinstance(raw_candidates, str):
            for chunk in raw_candidates.split(","):
                lever = self._normalise_lever_category(chunk)
                if lever and lever in style_plays and lever not in priorities:
                    priorities.append(lever)
        return priorities

    def _append_playbook_recommendations(
        self,
        summary: str,
        plays: List[Dict[str, Any]],
        playbook_context: Dict[str, Any],
    ) -> str:
        lines: List[str] = [summary.strip()]
        descriptor = playbook_context.get("descriptor")
        style = playbook_context.get("style")
        supplier_type = playbook_context.get("supplier_type")
        lever_priorities = playbook_context.get("lever_priorities") or []
        if descriptor or style or supplier_type:
            lines.append("")
        if descriptor:
            lines.append(f"Playbook focus: {descriptor}")
        if style or supplier_type or lever_priorities:
            lever_text = ", ".join(lever_priorities[:3]) if lever_priorities else "balanced"
            context_bits: List[str] = []
            if supplier_type:
                context_bits.append(f"Supplier type: {supplier_type}")
            if style:
                context_bits.append(f"Style: {style}")
            context_bits.append(f"Lever focus: {lever_text}")
            lines.append("; ".join(context_bits))
        if plays:
            lines.append("")
            lines.append("Recommended plays from negotiation playbook:")
            for idx, play in enumerate(plays[:3], 1):
                if not isinstance(play, dict):
                    continue
                lever = play.get("lever")
                description = play.get("play")
                rationale = play.get("rationale")
                trade_offs = play.get("trade_offs")
                entry = f"{idx}. "
                if lever:
                    entry += f"[{lever}] "
                if description:
                    entry += str(description)
                if rationale:
                    entry += f" — Rationale: {rationale}"
                if trade_offs:
                    entry += f" Trade-off: {trade_offs}"
                lines.append(entry)
        examples = playbook_context.get("examples") or []
        if examples:
            lines.append("")
            lines.append("Examples:")
            for example in examples[:2]:
                lines.append(f"- {example}")
        return "\n".join(line for line in lines if line is not None)

    def _load_playbook(self) -> Dict[str, Any]:
        if self._playbook_cache is not None:
            return self._playbook_cache
        try:
            with PLAYBOOK_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            # Normalise lever keys to expected casing for faster lookups.
            for supplier_type, entry in data.items():
                styles = entry.get("styles")
                if not isinstance(styles, dict):
                    continue
                for style_key, style_entry in list(styles.items()):
                    if not isinstance(style_entry, dict):
                        continue
                    normalised_style_entry: Dict[str, Any] = {}
                    for lever_key, plays in style_entry.items():
                        lever_name = self._normalise_lever_category(lever_key)
                        if not lever_name:
                            continue
                        normalised_style_entry[lever_name] = plays
                    styles[style_key] = normalised_style_entry
            self._playbook_cache = data
            return data
        except FileNotFoundError:
            logger.warning("Negotiation playbook file missing at %s", PLAYBOOK_PATH)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to load negotiation playbook from %s", PLAYBOOK_PATH)
        self._playbook_cache = {}
        return {}

    def _normalise_supplier_type(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        mapping = {
            "transactional": "Transactional",
            "leverage": "Leverage",
            "strategic": "Strategic",
            "bottleneck": "Bottleneck",
        }
        for key, canonical in mapping.items():
            if key in lowered:
                return canonical
        return None

    def _normalise_negotiation_style(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        mapping = {
            "competitive": "Competitive",
            "collaborative": "Collaborative",
            "principled": "Principled",
            "accommodating": "Accommodating",
            "compromising": "Compromising",
        }
        for key, canonical in mapping.items():
            if key in lowered:
                return canonical
        return None

    def _normalise_lever_category(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        cleaned = re.sub(r"[^a-zA-Z]+", " ", text).strip().upper()
        if not cleaned:
            return None
        for category in LEVER_CATEGORIES:
            if cleaned == category or category in cleaned.split():
                return category.title()
        return None

    def _extract_policy_guidance(self, context: AgentContext) -> Dict[str, Set[str]]:
        guidance: Dict[str, Set[str]] = {
            "preferred": set(),
            "required": set(),
            "restricted": set(),
            "discouraged": set(),
        }

        def ingest(bucket: str, values: Any) -> None:
            if not values:
                return
            for item in self._ensure_list(values):
                lever = self._normalise_lever_category(item)
                if lever:
                    guidance[bucket].add(lever)

        def parse_blob(blob: Any) -> None:
            if blob is None:
                return
            if isinstance(blob, dict):
                for key, value in blob.items():
                    if isinstance(value, (dict, list)):
                        parse_blob(value)
                        continue
                    lowered = str(key).lower()
                    if lowered in {"preferred", "preferred_levers", "allowed_levers", "focus_levers", "priority_levers"}:
                        ingest("preferred", value)
                    elif lowered in {"required", "required_levers", "mandated_levers", "must_have"}:
                        ingest("required", value)
                    elif lowered in {"restricted", "restricted_levers", "blocked_levers", "prohibited_levers"}:
                        ingest("restricted", value)
                    elif lowered in {"discouraged", "discouraged_levers", "avoid_levers"}:
                        ingest("discouraged", value)
                    else:
                        lever = self._normalise_lever_category(value)
                        if lever:
                            guidance["preferred"].add(lever)
            elif isinstance(blob, list):
                for item in blob:
                    parse_blob(item)
            elif isinstance(blob, str):
                tokens = re.split(r"[,;/]", blob)
                for token in tokens:
                    lever = self._normalise_lever_category(token)
                    if lever:
                        guidance["preferred"].add(lever)

        parse_blob(context.input_data.get("policy_guidance"))
        parse_blob(context.input_data.get("policy_constraints"))
        parse_blob(context.input_data.get("policy_preferences"))
        for policy in self._ensure_list(context.input_data.get("policies")):
            parse_blob(policy)
        return guidance

    def _score_policy_alignment(
        self, lever: str, guidance: Dict[str, Set[str]]
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        score = 0.0
        if lever in guidance.get("required", set()):
            score += 0.6
            notes.append("Required by policy")
        if lever in guidance.get("preferred", set()):
            score += 0.3
            notes.append("Policy prefers this lever")
        if lever in guidance.get("discouraged", set()):
            score -= 0.3
            notes.append("Policy discourages this lever")
        if lever in guidance.get("restricted", set()):
            score -= 0.7
            notes.append("Policy restricts this lever")
        return score, notes

    def _score_supplier_performance(
        self, lever: str, performance: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        if not performance:
            return score, notes

        def _coerce_float_metric(*keys: str) -> Optional[float]:
            for key in keys:
                value = performance.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        on_time = _coerce_float_metric("on_time_delivery", "on_time", "delivery_score", "otif")
        if on_time is not None:
            if on_time < 0.9 and lever in {"Operational", "Risk"}:
                score += 0.4 if lever == "Operational" else 0.2
                notes.append("On-time delivery below 90%")
            elif on_time > 0.97 and lever == "Operational":
                score -= 0.1
                notes.append("Delivery reliability already strong")

        defect_rate = _coerce_float_metric("quality_incidents", "defect_rate", "return_rate")
        if defect_rate is not None and defect_rate > 0:
            if lever == "Risk":
                score += 0.3
            notes.append("Quality issues detected")

        esg_score = _coerce_float_metric("esg_score", "sustainability_score")
        if esg_score is not None and esg_score < 0.6 and lever == "Strategic":
            score += 0.2
            notes.append("ESG performance lagging")

        collaboration_score = _coerce_float_metric("relationship_score", "collaboration_index")
        if collaboration_score is not None and collaboration_score < 0.6 and lever == "Relational":
            score += 0.3
            notes.append("Relationship maturity low")

        innovation_score = _coerce_float_metric("innovation_score", "co_innovation_index")
        if innovation_score is not None and innovation_score < 0.5 and lever == "Strategic":
            score += 0.25
            notes.append("Innovation potential needs reinforcement")

        return score, notes

    def _score_market_context(
        self, lever: str, market: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        score = 0.0
        notes: List[str] = []
        if not market:
            return score, notes

        supply_risk = market.get("supply_risk") or market.get("supply_risk_level")
        if isinstance(supply_risk, str) and supply_risk.strip().lower() in {"high", "elevated", "tight"}:
            if lever == "Risk":
                score += 0.3
            notes.append("Market supply risk elevated")

        demand_trend = market.get("demand_trend") or market.get("demand")
        if isinstance(demand_trend, str) and demand_trend.strip().lower() in {"rising", "high"}:
            if lever == "Commercial":
                score += 0.2
            notes.append("Demand is rising")

        inflation = market.get("inflation") or market.get("price_trend")
        if isinstance(inflation, str) and inflation.strip().lower() in {"inflationary", "increasing"}:
            if lever == "Commercial":
                score += 0.25
            notes.append("Prices trending upward")

        capacity = market.get("capacity") or market.get("capacity_constraints")
        if isinstance(capacity, str) and capacity.strip().lower() in {"limited", "constrained"}:
            if lever in {"Operational", "Strategic"}:
                score += 0.2
            notes.append("Capacity constraints present")

        esg_pressure = market.get("esg_pressure") or market.get("regulatory_focus")
        if isinstance(esg_pressure, str) and esg_pressure.strip().lower() in {"high", "tightening"}:
            if lever == "Strategic":
                score += 0.2
            notes.append("ESG expectations increasing")

        return score, notes

    def _compose_play_rationale(
        self,
        descriptor: Optional[str],
        style: Optional[str],
        lever: str,
        policy_notes: List[str],
        performance_notes: List[str],
        market_notes: List[str],
    ) -> str:
        segments: List[str] = []
        if descriptor:
            segments.append(str(descriptor))
        if style:
            segments.append(f"Supports {style.lower()} posture on the {lever.lower()} lever.")
        else:
            segments.append(f"Targets the {lever.lower()} lever.")
        if policy_notes:
            segments.append("Policy: " + "; ".join(policy_notes))
        if performance_notes:
            segments.append("Performance: " + "; ".join(performance_notes))
        if market_notes:
            segments.append("Market: " + "; ".join(market_notes))
        return " ".join(segments)

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
        *,
        playbook_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        rfq_text = rfq_id or "RFQ"
        strategy = decision.get("strategy")
        lines: List[str] = []
        header = f"Round {round_no} plan for {rfq_text}"
        if strategy:
            header += f": {strategy}"
        lines.append(header)

        descriptor: Optional[str] = None
        supplier_type: Optional[str] = None
        negotiation_style: Optional[str] = None
        lever_priorities: List[str] = []
        if playbook_context:
            descriptor_candidate = playbook_context.get("descriptor")
            descriptor = str(descriptor_candidate).strip() if descriptor_candidate else None
            supplier_candidate = playbook_context.get("supplier_type")
            supplier_type = str(supplier_candidate).strip() if supplier_candidate else None
            style_candidate = playbook_context.get("style")
            negotiation_style = (
                str(style_candidate).strip() if isinstance(style_candidate, str) else None
            )
            lever_values = playbook_context.get("lever_priorities")
            if isinstance(lever_values, (list, tuple)):
                lever_priorities = [
                    str(value).strip()
                    for value in lever_values
                    if isinstance(value, str) and value.strip()
                ]

        if supplier_type or descriptor:
            descriptor_parts: List[str] = []
            if supplier_type:
                descriptor_parts.append(f"Supplier category: {supplier_type}")
            if descriptor:
                descriptor_parts.append(descriptor)
            if descriptor_parts:
                lines.append("- " + " — ".join(descriptor_parts))

        if negotiation_style:
            if lever_priorities:
                lever_text = ", ".join(lever_priorities[:3])
            else:
                lever_text = "balanced levers"
            lines.append(
                f"- Negotiation style: {negotiation_style}; focus levers: {lever_text}"
            )

        counter_price = decision.get("counter_price")
        formatted_offer: Optional[str] = None
        if price is not None:
            formatted_offer = self._format_currency(price, currency)

        if counter_price is not None:
            counter_text = self._format_currency(counter_price, currency)
            if formatted_offer:
                lines.append(
                    f"- Counter at {counter_text} against supplier offer {formatted_offer}"
                )
            else:
                lines.append(f"- Counter target: {counter_text}")
        elif formatted_offer:
            lines.append(f"- Seek clarification before accepting {formatted_offer}")

        if target_price is not None and price is not None:
            target_text = self._format_currency(target_price, currency)
            delta = price - target_price
            try:
                delta_text = self._format_currency(delta, currency)
            except Exception:
                delta_text = f"{delta:0.2f}"
            lines.append(
                f"- Target {target_text} vs offer {formatted_offer} (gap {delta_text})"
            )

        lead_time = decision.get("lead_time_request")
        if lead_time:
            lines.append(f"- Lead time ask: {lead_time}")

        asks = decision.get("asks") or []
        if asks:
            lines.append("- Key asks: " + "; ".join(str(item) for item in asks if item))

        return "\n".join(lines)

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

        workflow_hint = getattr(context, "workflow_id", None)
        if workflow_hint:
            for draft in candidate_drafts:
                if isinstance(draft, dict) and not draft.get("workflow_id"):
                    draft["workflow_id"] = workflow_hint

        watch_payload: Dict[str, Any] = {
            "await_response": True,
            "message": "",
            "body": "",
            "drafts": candidate_drafts,
            "rfq_id": rfq_id,
            "supplier_id": supplier,
            "response_poll_interval": poll_interval,
            "workflow_id": workflow_hint,
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

            draft_action_id = None
            for key in ("action_id", "draft_action_id", "email_action_id"):
                candidate = target.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    draft_action_id = candidate.strip()
                    break
            workflow_hint = target.get("workflow_id") or watch_payload.get("workflow_id")
            if not workflow_hint and isinstance(target.get("metadata"), dict):
                meta_workflow = target["metadata"].get("workflow_id") or target["metadata"].get("process_workflow_id")
                if isinstance(meta_workflow, str) and meta_workflow.strip():
                    workflow_hint = meta_workflow.strip()
            dispatch_run_id = None
            for key in ("dispatch_run_id", "run_id"):
                candidate = target.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    dispatch_run_id = candidate.strip()
                    break
            if dispatch_run_id is None and isinstance(target.get("metadata"), dict):
                meta_run = target["metadata"].get("dispatch_run_id") or target["metadata"].get("run_id")
                if isinstance(meta_run, str) and meta_run.strip():
                    dispatch_run_id = meta_run.strip()

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
                    draft_action_id=draft_action_id,
                    workflow_id=workflow_hint,
                    dispatch_run_id=dispatch_run_id,
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
        logger.debug(
            "Negotiation learning capture skipped (rfq_id=%s, supplier=%s)", rfq_id, supplier
        )
        return

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
        plays = decision.get("play_recommendations") or []
        if plays:
            top_snippets: List[str] = []
            for play in plays[:3]:
                if not isinstance(play, dict):
                    continue
                lever = play.get("lever") or play.get("category")
                description = play.get("play") or play.get("description")
                if not description:
                    continue
                if lever:
                    top_snippets.append(f"{lever}: {description}")
                else:
                    top_snippets.append(str(description))
            if top_snippets:
                base = f"{base} Plays={' | '.join(top_snippets)}."
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

