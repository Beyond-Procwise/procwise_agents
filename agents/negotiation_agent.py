import json
import logging
import math
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast

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

MARKET_REVIEW_THRESHOLD = float(os.getenv("NEG_MARKET_REVIEW_PCT", "0.2"))
MARKET_ESCALATION_THRESHOLD = float(os.getenv("NEG_MARKET_ESCALATION_PCT", "0.4"))
MAX_VOLUME_LIMIT = float(os.getenv("NEG_MAX_VOLUME_LIMIT", "1000"))
MAX_TERM_DAYS = int(os.getenv("NEG_MAX_TERM_DAYS", "120"))

DEFAULT_NEGOTIATION_MESSAGE_TEMPLATE = "{header}\n{details}{context_sections}"

BATCH_INPUT_KEYS = ("negotiation_batch", "supplier_responses_batch", "batch_responses")
BATCH_SHARED_KEYS = (
    "shared_context",
    "shared_payload",
    "batch_defaults",
    "shared_fields",
    "defaults",
)
BATCH_EXCLUDE_KEYS = {
    "negotiation_batch",
    "supplier_responses_batch",
    "batch_responses",
    "shared_context",
    "shared_payload",
    "batch_defaults",
    "shared_fields",
    "defaults",
    "batch_metadata",
    "batch_results",
    "batch_summary",
    "agentic_plan",
    "pass_fields",
    "results",
    "drafts",
    "supplier_responses",
}


@dataclass
class NegotiationIdentifier:
    workflow_id: str
    session_reference: str
    supplier_id: str
    round_number: int = 1

    def __post_init__(self) -> None:
        self.workflow_id = self._normalise(self.workflow_id, fallback_prefix="WF")
        self.session_reference = self._normalise(
            self.session_reference, fallback_prefix="WF"
        )
        self.supplier_id = self._normalise(self.supplier_id, fallback_prefix="SUP")
        try:
            self.round_number = int(self.round_number) if self.round_number else 1
        except Exception:
            self.round_number = 1

    @staticmethod
    def _normalise(value: Optional[str], *, fallback_prefix: str = "") -> str:
        if isinstance(value, str):
            token = value.strip()
        elif value is None:
            token = ""
        else:
            token = str(value).strip()
        if not token:
            return f"{fallback_prefix}-{uuid.uuid4().hex[:12].upper()}" if fallback_prefix else ""
        return token

    @property
    def unique_key(self) -> str:
        return f"{self.workflow_id}:{self.supplier_id}:{self.round_number}"

    @property
    def thread_key(self) -> str:
        return f"{self.workflow_id}:{self.supplier_id}"


@dataclass
class EmailThreadState:
    thread_id: str
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    subject_base: str = ""

    def to_headers(self, round_number: int) -> Dict[str, Any]:
        message_id = f"<{uuid.uuid4()}@procwise.co.uk>"
        headers: Dict[str, Any] = {"Message-ID": message_id}
        if self.references:
            headers["References"] = " ".join(self.references[-10:])
        if self.in_reply_to:
            headers["In-Reply-To"] = self.in_reply_to
        subject = self.subject_base or DEFAULT_NEGOTIATION_SUBJECT
        if round_number > 1 and subject:
            if subject.lower().startswith("re:"):
                headers["Subject"] = subject
            else:
                headers["Subject"] = f"Re: {subject}".strip()
        elif subject:
            headers["Subject"] = subject
        return headers

    def update_after_send(self, message_id: Optional[str]) -> None:
        token = self._normalise_token(message_id)
        if not token:
            return
        if not self.thread_id:
            self.thread_id = token
        if token not in self.references:
            self.references.append(token)
        self.in_reply_to = token

    def update_after_receive(self, message_id: Optional[str]) -> None:
        token = self._normalise_token(message_id)
        if not token:
            return
        if token not in self.references:
            self.references.append(token)
        self.in_reply_to = token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "in_reply_to": self.in_reply_to,
            "references": list(self.references),
            "subject_base": self.subject_base,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], *, fallback_subject: str) -> "EmailThreadState":
        thread_id = str(data.get("thread_id") or f"<{uuid.uuid4()}@procwise.co.uk>")
        references = data.get("references") if isinstance(data.get("references"), list) else []
        return EmailThreadState(
            thread_id=thread_id,
            in_reply_to=data.get("in_reply_to"),
            references=[str(item) for item in references if item],
            subject_base=str(data.get("subject_base") or fallback_subject or DEFAULT_NEGOTIATION_SUBJECT),
        )

    @staticmethod
    def _normalise_token(token: Optional[str]) -> Optional[str]:
        if isinstance(token, str):
            value = token.strip()
        elif token is None:
            value = ""
        else:
            value = str(token).strip()
        return value or None


class ResponseMatcher:
    """Match supplier responses to the registered negotiation rounds."""

    def __init__(self, db_connection_factory: Optional[Any]) -> None:
        self.db = db_connection_factory
        self._pending_responses: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def register_expected_response(
        self,
        identifier: NegotiationIdentifier,
        email_action_id: Optional[str],
        expected_unique_id: str,
        timeout_seconds: int = 900,
    ) -> None:
        payload = {
            "round": identifier.round_number,
            "email_action_id": email_action_id,
            "unique_id": expected_unique_id,
            "registered_at": datetime.now(timezone.utc),
            "timeout_at": datetime.now(timezone.utc) + timedelta(seconds=max(timeout_seconds, 60)),
            "thread_key": identifier.thread_key,
        }
        with self._lock:
            bucket = self._pending_responses.setdefault(identifier.workflow_id, {})
            bucket[identifier.supplier_id] = payload

    def match_response(
        self,
        workflow_id: str,
        supplier_id: str,
        message_id: Optional[str],
        response_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not workflow_id or not supplier_id:
            return None
        with self._lock:
            workflow_bucket = self._pending_responses.get(workflow_id)
            if not workflow_bucket:
                return None
            expected = workflow_bucket.get(supplier_id)
            if not expected:
                return None
            if datetime.now(timezone.utc) > expected.get("timeout_at", datetime.now(timezone.utc)):
                workflow_bucket.pop(supplier_id, None)
                return None
            workflow_bucket.pop(supplier_id, None)
        payload = dict(expected)
        payload["response_data"] = response_data or {}
        payload["message_id"] = message_id
        payload["matched_at"] = datetime.now(timezone.utc)
        return payload

    def get_pending_count(self, workflow_id: str) -> int:
        with self._lock:
            bucket = self._pending_responses.get(workflow_id) or {}
            return len(bucket)


@dataclass
class NegotiationPositions:
    start: Optional[float]
    desired: Optional[float]
    no_deal: Optional[float]
    supplier_offer: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def serialise(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "desired": self.desired,
            "no_deal": self.no_deal,
            "supplier_offer": self.supplier_offer,
            "history": list(self.history),
        }

    def snapshot_for_next_round(
        self, counter_price: Optional[float], round_no: int
    ) -> Dict[str, Any]:
        history = list(self.history)

        def _append(entry_type: str, value: Optional[float]) -> None:
            if value is None:
                return
            record = {
                "round": round_no,
                "type": entry_type,
                "value": value,
            }
            if not any(
                existing.get("round") == record["round"]
                and existing.get("type") == record["type"]
                and self._is_close(existing.get("value"), record["value"])
                for existing in history
            ):
                history.append(record)

        _append("supplier_offer", self.supplier_offer)
        _append("counter", counter_price)

        next_start = counter_price if counter_price is not None else self.start

        return {
            "start": next_start,
            "desired": self.desired,
            "no_deal": self.no_deal,
            "supplier_offer": self.supplier_offer,
            "history": history,
            "last_counter": counter_price if counter_price is not None else next_start,
        }

    @staticmethod
    def _is_close(value_a: Any, value_b: Any, *, tolerance: float = 1e-6) -> bool:
        try:
            return abs(float(value_a) - float(value_b)) <= tolerance
        except (TypeError, ValueError):
            return False


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

    AGENTIC_PLAN_STEPS = (
        "Analyse supplier offers, historical spend, and negotiation constraints for the active RFQ session.",
        "Compute pricing and terms strategy including counter targets, concessions, and playbook actions.",
        "Coordinate email drafting and state persistence to deliver the counter proposal and next steps.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self._state_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._email_agent: Optional[EmailDraftingAgent] = None
        self._supplier_agent: Optional["SupplierInteractionAgent"] = None
        self._state_schema_checked = False
        self._playbook_cache: Optional[Dict[str, Any]] = None
        self._state_lock = threading.RLock()
        self._email_agent_lock = threading.Lock()
        self._supplier_agent_lock = threading.Lock()
        self._negotiation_session_state_identifier_column: Optional[str] = None
        self._negotiation_sessions_identifier_column: Optional[str] = None
        self.response_matcher = ResponseMatcher(getattr(self.agent_nick, "get_db_connection", None))

        # Feature flag for enhanced negotiation message composition
        self._use_enhanced_messages = (
            os.getenv("NEG_USE_ENHANCED_MESSAGES", "false").lower() == "true"
        )

    def _resolve_session_id(self, context: AgentContext) -> Optional[str]:
        """Derive the canonical negotiation session identifier."""

        if not context:
            return None

        for key in ("workflow_id", "session_id", "process_id"):
            candidate = getattr(context, key, None)
            text = self._coerce_text(candidate)
            if text:
                return text

        payload = context.input_data if isinstance(context.input_data, dict) else {}
        for key in ("workflow_id", "session_id", "process_id"):
            candidate = payload.get(key)
            text = self._coerce_text(candidate)
            if text:
                return text

        responses = payload.get("supplier_responses")
        if isinstance(responses, list):
            for entry in responses:
                if not isinstance(entry, dict):
                    continue
                uid = self._coerce_text(entry.get("unique_id"))
                if uid:
                    return uid

        return None

    def _resolve_session_reference(self, context: AgentContext) -> Optional[str]:
        """Resolve a display/reference token for downstream compatibility."""

        payload = context.input_data if isinstance(context.input_data, dict) else {}
        responses = payload.get("supplier_responses")
        if isinstance(responses, list):
            for entry in responses:
                if not isinstance(entry, dict):
                    continue
                for key in ("session_reference", "unique_id", "workflow_id", "session_id", "process_id"):
                    candidate = self._coerce_text(entry.get(key))
                    if candidate:
                        return candidate

        for key in ("session_reference", "unique_id"):
            candidate = self._coerce_text(payload.get(key))
            if candidate:
                return candidate

        return self._coerce_text(getattr(context, "workflow_id", None))

    def _resolve_negotiation_identifier(self, context: AgentContext) -> NegotiationIdentifier:
        payload = context.input_data if isinstance(context.input_data, dict) else {}

        workflow_id = self._coerce_text(getattr(context, "workflow_id", None))
        if not workflow_id:
            raise ValueError(
                "NegotiationAgent requires workflow_id on context; none was provided"
            )

        payload_workflow_id = self._coerce_text(payload.get("workflow_id")) if isinstance(payload, dict) else None
        if payload_workflow_id and payload_workflow_id != workflow_id:
            logger.warning(
                "Workflow ID mismatch between context (%s) and payload (%s); using context value",
                workflow_id,
                payload_workflow_id,
            )

        supplier_id: Optional[str] = None
        for key in ("supplier_id", "supplier", "supplier_name"):
            candidate = payload.get(key)
            text = self._coerce_text(candidate)
            if text:
                supplier_id = text
                break
        if not supplier_id:
            responses = payload.get("supplier_responses")
            if isinstance(responses, list):
                for entry in responses:
                    if not isinstance(entry, dict):
                        continue
                    candidate = self._coerce_text(
                        entry.get("supplier_id") or entry.get("supplier") or entry.get("supplier_name")
                    )
                    if candidate:
                        supplier_id = candidate
                        break
        if not supplier_id:
            supplier_id = f"SUP-{uuid.uuid4().hex[:10].upper()}"

        session_reference: Optional[str] = None
        for key in ("session_reference", "unique_id", "conversation_id", "thread_id"):
            candidate = payload.get(key)
            text = self._coerce_text(candidate)
            if text:
                session_reference = text
                break
        if not session_reference:
            session_reference = f"{workflow_id}-{supplier_id}"

        round_number_raw = payload.get("round")
        try:
            round_number = int(round_number_raw) if round_number_raw is not None else 1
        except Exception:
            round_number = 1

        identifier = NegotiationIdentifier(
            workflow_id=workflow_id,
            session_reference=session_reference,
            supplier_id=supplier_id,
            round_number=round_number,
        )

        if isinstance(payload, dict):
            payload.setdefault("workflow_id", identifier.workflow_id)
            payload.setdefault("session_reference", identifier.session_reference)
            payload.setdefault("unique_id", identifier.session_reference)
            payload.setdefault("supplier_id", identifier.supplier_id)
            payload.setdefault("supplier", identifier.supplier_id)
            payload.setdefault("round", identifier.round_number)

        return identifier

    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("NegotiationAgent starting")

        batch_entries, shared_context = self._extract_batch_inputs(context.input_data)
        if batch_entries:
            return self._run_batch_negotiations(context, batch_entries, shared_context)

        return self._run_single_negotiation(context)

    def _extract_batch_inputs(
        self, payload: Optional[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not isinstance(payload, dict):
            return [], {}

        responses = payload.get("supplier_responses")
        if isinstance(responses, list):
            response_batch = [entry for entry in responses if isinstance(entry, dict)]
        else:
            response_batch = []

        if response_batch:
            shared: Dict[str, Any] = {"negotiation_batch": True}
            for key in BATCH_SHARED_KEYS:
                candidate = payload.get(key)
                if isinstance(candidate, dict):
                    shared.update(candidate)
            return response_batch, shared

        batch: List[Dict[str, Any]] = []
        for key in BATCH_INPUT_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                batch = [entry for entry in value if isinstance(entry, dict)]
                if batch:
                    break

        if not batch:
            return [], {}

        shared: Dict[str, Any] = {}
        for key in BATCH_SHARED_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                shared.update(candidate)

        return batch, shared

    def _coerce_batch_defaults(
        self, base_payload: Dict[str, Any], shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        if isinstance(base_payload, dict):
            for key, value in base_payload.items():
                if key in BATCH_EXCLUDE_KEYS:
                    continue
                defaults[key] = value
        if isinstance(shared_context, dict):
            defaults.update(shared_context)
        return defaults

    def _prepare_batch_payload(
        self, defaults: Dict[str, Any], entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if isinstance(defaults, dict):
            payload.update(defaults)
        payload.update(entry)
        for key in BATCH_EXCLUDE_KEYS:
            payload.pop(key, None)
        supplier_id = payload.get("supplier_id") or payload.get("supplier")
        if supplier_id is not None:
            payload.setdefault("supplier_id", supplier_id)
            payload.setdefault("supplier", supplier_id)
        return payload

    def _execute_batch_entry(
        self, parent_context: AgentContext, payload: Dict[str, Any]
    ) -> AgentOutput:
        workflow_id = payload.get("workflow_id") or parent_context.workflow_id or "negotiation-batch"
        user_id = payload.get("user_id") or parent_context.user_id or "system"
        sub_context = AgentContext(
            workflow_id=str(workflow_id),
            agent_id=parent_context.agent_id,
            user_id=str(user_id),
            input_data=dict(payload),
            parent_agent=parent_context.agent_id,
            routing_history=list(parent_context.routing_history),
        )
        return self.execute(sub_context)

    def _run_batch_negotiations(
        self,
        context: AgentContext,
        batch_entries: List[Dict[str, Any]],
        shared_context: Dict[str, Any],
    ) -> AgentOutput:
        defaults = self._coerce_batch_defaults(context.input_data, shared_context)
        prepared_entries = [
            self._prepare_batch_payload(defaults, entry) for entry in batch_entries if isinstance(entry, dict)
        ]

        if not prepared_entries:
            empty_data = {
                "negotiation_batch": True,
                "results": [],
                "drafts": [],
                "successful_suppliers": [],
                "failed_suppliers": [],
            }
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=empty_data,
                    pass_fields={"negotiation_batch": True, "batch_results": []},
                ),
            )

        workflow_id = shared_context.get("workflow_id") or getattr(context, "workflow_id", None)
        expected_total_raw = (
            shared_context.get("expected_count")
            or shared_context.get("expected_responses")
            or shared_context.get("expected_email_count")
        )
        expected_total: Optional[int] = None
        if expected_total_raw is not None:
            try:
                expected_total = int(expected_total_raw)
            except Exception:
                logger.debug(
                    "Unable to coerce expected_total=%s for workflow_id=%s", expected_total_raw, workflow_id
                )
                expected_total = None
        if expected_total is not None and expected_total > 0 and len(prepared_entries) < expected_total:
            logger.error(
                "Batch negotiation received incomplete entries for workflow=%s (got=%s, expected=%s)",
                workflow_id,
                len(prepared_entries),
                expected_total,
            )
            error_payload = {
                "negotiation_batch": True,
                "workflow_id": workflow_id,
                "expected_count": expected_total,
                "received_count": len(prepared_entries),
                "results": [],
                "error": f"Incomplete batch: received {len(prepared_entries)}/{expected_total} entries",
            }
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data=error_payload,
                    pass_fields=error_payload,
                    error="incomplete_batch",
                ),
            )

        if len(prepared_entries) == 1:
            try:
                return self._execute_batch_entry(context, prepared_entries[0])
            except Exception as exc:  # pragma: no cover - propagate as failure
                logger.exception("NegotiationAgent batch execution failed", exc_info=True)
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={
                            "negotiation_batch": True,
                            "results": [
                                {
                                    "supplier_id": prepared_entries[0].get("supplier_id"),
                                    "session_reference": prepared_entries[0].get("session_reference")
                                    or prepared_entries[0].get("unique_id"),
                                    "status": AgentStatus.FAILED.value,
                                    "error": str(exc),
                                }
                            ],
                        },
                        error=str(exc),
                    ),
                )

        try:
            configured_workers = getattr(self.agent_nick.settings, "negotiation_parallel_workers", None)
            if configured_workers is not None:
                try:
                    configured_workers = int(configured_workers)
                except Exception:
                    configured_workers = None
            max_workers = configured_workers if configured_workers and configured_workers > 0 else None
        except Exception:
            max_workers = None

        if max_workers is None:
            cpu_default = os.cpu_count() or 4
            max_workers = max(1, cpu_default)

        unique_suppliers: Set[str] = set()
        for entry in prepared_entries:
            supplier_id = entry.get("supplier_id") or entry.get("supplier")
            if supplier_id is None:
                continue
            try:
                supplier_key = str(supplier_id).strip()
            except Exception:
                continue
            if supplier_key:
                unique_suppliers.add(supplier_key)

        desired_workers = max(1, len(unique_suppliers)) if unique_suppliers else len(prepared_entries)
        if desired_workers <= 0:
            desired_workers = len(prepared_entries)

        max_workers = max(desired_workers, max_workers)
        max_workers = min(max_workers, len(prepared_entries))

        futures: List[Tuple[Dict[str, Any], Any]] = []
        aggregated_results: List[Dict[str, Any]] = []
        drafts: List[Dict[str, Any]] = []
        failed_records: List[Dict[str, Any]] = []
        success_ids: List[str] = []
        supplier_map: Dict[str, Dict[str, Any]] = {}
        next_agents: Set[str] = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for payload in prepared_entries:
                futures.append((payload, executor.submit(self._execute_batch_entry, context, payload)))

            for payload, future in futures:
                supplier_id = payload.get("supplier_id") or payload.get("supplier")
                session_reference = (
                    payload.get("session_reference")
                    or payload.get("unique_id")
                )
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive aggregation
                    error_message = str(exc)
                    record = {
                        "supplier_id": supplier_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "status": AgentStatus.FAILED.value,
                        "error": error_message,
                    }
                    aggregated_results.append(record)
                    failed_records.append(record)
                    continue

                record_reference = (
                    result.data.get("session_reference")
                    or result.data.get("unique_id")
                    or session_reference
                )
                record = {
                    "supplier_id": result.data.get("supplier")
                    or supplier_id
                    or payload.get("supplier_id"),
                    "session_reference": record_reference,
                    "unique_id": record_reference,
                    "status": result.status.value,
                    "output": result.data,
                    "pass_fields": result.pass_fields,
                    "next_agents": list(result.next_agents),
                }
                if result.error:
                    record["error"] = result.error
                if result.action_id:
                    record["action_id"] = result.action_id

                aggregated_results.append(record)

                supplier_key = record.get("supplier_id")
                if supplier_key:
                    supplier_map[str(supplier_key)] = record

                if result.status == AgentStatus.SUCCESS:
                    if supplier_key:
                        success_ids.append(str(supplier_key))
                    result_drafts = result.data.get("drafts")
                    if isinstance(result_drafts, list):
                        for draft in result_drafts:
                            if isinstance(draft, dict):
                                drafts.append(draft)
                    next_agents.update(result.next_agents)
                else:
                    failed_records.append(
                        {
                            "supplier_id": supplier_key,
                            "session_reference": record.get("session_reference"),
                            "status": record.get("status"),
                            "error": record.get("error"),
                        }
                    )

        any_success = any(record["status"] == AgentStatus.SUCCESS.value for record in aggregated_results)
        any_failure = any(record["status"] != AgentStatus.SUCCESS.value for record in aggregated_results)

        overall_status = AgentStatus.SUCCESS
        error_text = None
        if any_failure and not any_success:
            overall_status = AgentStatus.FAILED
            error_text = "All negotiation rounds failed"

        if success_ids:
            try:
                success_ids = list(dict.fromkeys(success_ids))
            except Exception:
                success_ids = [sid for sid in success_ids if sid]

        data = {
            "negotiation_batch": True,
            "results": aggregated_results,
            "drafts": drafts,
            "successful_suppliers": success_ids,
            "failed_suppliers": failed_records,
            "results_by_supplier": supplier_map,
            "batch_size": len(aggregated_results),
        }

        pass_fields = {
            "negotiation_batch": True,
            "batch_results": aggregated_results,
        }
        if drafts:
            pass_fields["drafts"] = drafts

        return self._with_plan(
            context,
            AgentOutput(
                status=overall_status,
                data=data,
                pass_fields=pass_fields,
                next_agents=sorted(next_agents),
                error=error_text,
            ),
        )

    def _run_single_negotiation(self, context: AgentContext) -> AgentOutput:

        payload = context.input_data if isinstance(context.input_data, dict) else {}
        identifier = self._resolve_negotiation_identifier(context)
        supplier = identifier.supplier_id
        session_reference = identifier.session_reference
        workflow_id = identifier.workflow_id
        session_id = workflow_id

        rfq_id: Optional[str] = None
        if isinstance(payload, dict):
            for key in ("rfq_id", "rfqId", "rfq", "rfq_reference", "rfqReference"):
                candidate = payload.get(key)
                text = self._coerce_text(candidate)
                if text:
                    rfq_id = text
                    break

        state_identifier: Optional[str] = self._coerce_text(workflow_id)
        if not state_identifier:
            for candidate in (rfq_id, session_reference, session_id):
                text = self._coerce_text(candidate)
                if text:
                    state_identifier = text
                    break
        if not state_identifier:
            state_identifier = identifier.workflow_id
        else:
            identifier.workflow_id = state_identifier

        if rfq_id and isinstance(context.input_data, dict):
            context.input_data.setdefault("rfq_id", rfq_id)
        rfq_value = rfq_id or state_identifier
        raw_round = context.input_data.get("round", identifier.round_number)

        state_identifier = identifier.workflow_id or state_identifier

        price_raw = (
            context.input_data.get("current_offer")
            if context.input_data.get("current_offer") is not None
            else context.input_data.get("price")
        )

        normalised_inputs, validation_issues = self._normalise_negotiation_inputs(
            context.input_data
        )

        price = normalised_inputs.get("current_offer")
        target_price = normalised_inputs.get("target_price")
        walkaway_price = normalised_inputs.get("walkaway_price")
        currency = normalised_inputs.get("currency")
        lead_weeks = normalised_inputs.get("lead_time_weeks")
        volume_units = normalised_inputs.get("volume_units")
        term_days = normalised_inputs.get("term_days")
        valid_until = normalised_inputs.get("valid_until")
        market_floor = normalised_inputs.get("market_floor_price")

        currency_conf = self._coerce_float(
            context.input_data.get("currency_confidence")
            or context.input_data.get("current_offer_currency_confidence")
        )
        if currency_conf is not None and currency_conf < 0.5:
            logger.warning(
                "Currency confidence %.2f below threshold; withholding price data", currency_conf
            )
            currency = None
            price = None

        constraints = self._ensure_list(context.input_data.get("constraints"))
        supplier_snippets = self._collect_supplier_snippets(context.input_data)
        supplier_message = self._coerce_text(
            context.input_data.get("supplier_message")
            or context.input_data.get("response_text")
            or context.input_data.get("message")
        )

        # Load state before subject normalization (bug fix)
        state, _ = self._load_session_state(state_identifier, supplier)
        state["workflow_id"] = identifier.workflow_id
        state["session_reference"] = session_reference

        round_no = int(state.get("current_round", 1))
        if isinstance(raw_round, (int, float)):
            try:
                round_no = max(round_no, int(raw_round))
            except (TypeError, ValueError):
                round_no = max(round_no, 1)
        state["current_round"] = max(round_no, 1)
        identifier.round_number = round_no

        previous_positions = state.get("positions") if isinstance(state.get("positions"), dict) else None
        previous_counter_hint = (
            context.input_data.get("agent_previous_offer")
            or context.input_data.get("previous_counter_price")
            or (
                previous_positions.get("last_counter")
                if isinstance(previous_positions, dict)
                else None
            )
            or state.get("last_counter_price")
        )
        previous_counter = self._coerce_float(previous_counter_hint)
        positions = self._build_positions(
            supplier_offer=price,
            target_price=target_price,
            walkaway_price=walkaway_price,
            previous_counter=previous_counter,
            previous_positions=previous_positions,
            round_no=round_no,
        )
        target_price = positions.desired
        walkaway_price = positions.no_deal
        price = (
            positions.supplier_offer
            if positions.supplier_offer is not None
            else positions.start
        )

        incoming_subject = self._coerce_text(context.input_data.get("subject"))
        base_candidate = self._normalise_base_subject(incoming_subject)
        if base_candidate and not state.get("base_subject"):
            state["base_subject"] = base_candidate

        thread_state = self._load_thread_state(
            identifier, state, subject_hint=incoming_subject
        )

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
            "walkaway_price": walkaway_price,
            "ask_early_pay_disc": context.input_data.get("ask_early_pay_disc", 0.02),
            "ask_lead_time_keep": context.input_data.get("ask_lead_time_keep", True),
        }
        pricing_payload["start_position"] = positions.start
        pricing_payload["desired_position"] = positions.desired
        pricing_payload["no_deal_position"] = positions.no_deal

        offer_prev = normalised_inputs.get("previous_offer")
        if offer_prev is None and positions.history:
            for entry in reversed(positions.history):
                if entry.get("type") != "supplier_offer":
                    continue
                value = self._coerce_float(entry.get("value"))
                if value is None:
                    continue
                if positions.supplier_offer is not None and NegotiationPositions._is_close(
                    value, positions.supplier_offer
                ):
                    continue
                offer_prev = value
                break

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
        base_plan_message = decision.get("rationale")
        decision.setdefault("strategy", "clarify")
        decision.setdefault("counter_price", None)
        decision.setdefault("asks", [])
        decision.setdefault("lead_time_request", None)
        decision.setdefault("rationale", "")
        decision["closing_round"] = round_no >= 3
        decision["counter_price"] = self._respect_positions(
            decision.get("counter_price"), positions
        )
        decision["positions"] = positions.serialise()
        if validation_issues:
            decision["validation_issues"] = validation_issues
        if volume_units is not None:
            decision["volume_units"] = volume_units
        if term_days is not None:
            decision["term_days"] = term_days
        if valid_until:
            decision["valid_until"] = valid_until
        if market_floor is not None:
            decision["market_floor_price"] = market_floor
        decision["plan_counter_message"] = base_plan_message

        outlier_result = self._detect_outliers(
            supplier_offer=positions.supplier_offer,
            target_price=positions.desired,
            walkaway_price=positions.no_deal,
            market_floor=market_floor,
            volume_units=volume_units,
            term_days=term_days,
        )
        if outlier_result.get("alerts"):
            decision.setdefault("alerts", [])
            for alert in outlier_result["alerts"]:
                if alert not in decision["alerts"]:
                    decision["alerts"].append(alert)
            decision["outlier_alerts"] = outlier_result["alerts"]
        if outlier_result.get("requires_review"):
            decision["strategy"] = "review"
            decision.setdefault("flags", {})
            decision["flags"]["review_recommended"] = True
            decision["recommendation"] = (
                outlier_result.get("recommendation") or "query_for_human_review"
            )
            asks = list(decision.get("asks", []))
            asks.append("Please justify the requested pricing/terms for internal review.")
            decision["asks"] = list(dict.fromkeys(asks))
            if not outlier_result.get("human_override"):
                decision["counter_price"] = self._respect_positions(
                    decision.get("counter_price"), positions
                )
        if outlier_result.get("human_override"):
            decision.setdefault("flags", {})
            decision["flags"]["human_override_required"] = True
            decision["human_override_required"] = True
            decision["counter_price"] = None

        structured_rationale = self._compose_rationale(
            positions=positions,
            decision=decision,
            currency=currency,
            lead_weeks=lead_weeks,
            volume_units=volume_units,
            term_days=term_days,
            outlier_message=outlier_result.get("message"),
            validation_issues=validation_issues,
        )
        if base_plan_message and base_plan_message not in structured_rationale:
            decision["rationale"] = f"{structured_rationale} {base_plan_message}".strip()
        else:
            decision["rationale"] = structured_rationale

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

        message_id = self._coerce_text(context.input_data.get("message_id"))
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
            if thread_state:
                thread_state.update_after_receive(message_id)
            if self.response_matcher and identifier.workflow_id and supplier:
                matched = self.response_matcher.match_response(
                    identifier.workflow_id,
                    supplier,
                    message_id,
                    context.input_data if isinstance(context.input_data, dict) else {},
                )
                if matched:
                    matched_list = state.setdefault("matched_responses", [])
                    if isinstance(matched_list, list):
                        matched_list.append(matched)

        decision["round"] = round_no
        state["positions"] = positions.snapshot_for_next_round(
            decision.get("counter_price"), round_no
        )
        state["last_counter_price"] = decision.get("counter_price")

        final_offer_reason = self._detect_final_offer(supplier_message, supplier_snippets)
        should_continue, new_status, halt_reason = self._should_continue(
            state, supplier_reply_registered, final_offer_reason
        )

        if outlier_result.get("human_override"):
            should_continue = False
            new_status = "PAUSED"
            halt_reason = outlier_result.get("message") or "Human override required."
            decision.setdefault("status_reason", halt_reason)
        elif outlier_result.get("requires_review") and outlier_result.get("message"):
            decision.setdefault("status_reason", outlier_result.get("message"))

        savings_score = 0.0
        if price and price > 0 and target_price is not None:
            try:
                savings_score = (price - target_price) / float(price)
            except ZeroDivisionError:
                savings_score = 0.0

        decision_log = self._build_decision_log(
            supplier,
            session_reference,
            price,
            target_price,
            decision,
        )

        draft_records: List[Dict[str, Any]] = []

        counter_options: List[Dict[str, Any]] = []

        sent_message_id: Optional[str] = None

        if not should_continue:
            state["status"] = new_status
            state["awaiting_response"] = halt_reason == "Awaiting supplier response."
            stop_message = self._build_stop_message(new_status, halt_reason, round_no)
            decision.setdefault("status_reason", halt_reason)
            self._persist_thread_state(state, thread_state)
            self._save_session_state(identifier.workflow_id, supplier, state)
            self._record_learning_snapshot(
                context,
                identifier.workflow_id or state_identifier or session_id,
                supplier,
                decision,
                state,
                bool(state.get("awaiting_response", False)),
                supplier_reply_registered,
                rfq_id=rfq_value,
            )
            awaiting_now = bool(state.get("awaiting_response", False))
            negotiation_open = (
                new_status not in {"COMPLETED", "EXHAUSTED"}
                and not awaiting_now
                and decision.get("strategy") != "review"
                and not decision.get("human_override_required")
            )
            data = {
                "supplier": supplier,
                "rfq_id": rfq_value,
                "session_reference": session_reference,
                "unique_id": session_reference,
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
                "positions": decision.get("positions"),
                "validation_issues": decision.get("validation_issues"),
                "outlier_alerts": decision.get("outlier_alerts"),
                "volume_units": volume_units,
                "term_days": term_days,
                "valid_until": valid_until,
                "flags": decision.get("flags"),
                "market_floor_price": market_floor,
                "normalised_inputs": normalised_inputs,
            }
            logger.info(
                "NegotiationAgent halted negotiation for supplier=%s session_id=%s reason=%s",
                supplier,
                session_id,
                halt_reason,
            )
            pass_fields = dict(data)
            next_agents: List[str] = []
            if awaiting_now:
                watch_fields = self._build_supplier_watch_fields(
                    context=context,
                    session_reference=session_reference,
                    supplier=supplier,
                    drafts=draft_records,
                    state=state,
                    workflow_id=session_id,
                    rfq_id=rfq_value,
                )
                if watch_fields:
                    pass_fields.update(watch_fields)
                    next_agents.append("SupplierInteractionAgent")
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=data,
                    pass_fields=pass_fields,
                    next_agents=next_agents,
                ),
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

        supplier_name = context.input_data.get("supplier_name")
        supplier_identifier = context.input_data.get("supplier_id") or supplier
        procurement_summary = self._retrieve_procurement_summary(
            supplier_id=supplier_identifier,
            supplier_name=supplier_name,
        )
        rag_snippets = self._collect_vector_snippets(
            context=context,
            supplier_name=supplier_name or supplier_identifier,
            workflow_reference=session_reference,
        )

        negotiation_message = self._build_summary(
            context,
            session_reference,
            decision,
            price,
            target_price,
            currency,
            round_no,
            supplier=supplier,
            supplier_snippets=supplier_snippets,
            supplier_message=supplier_message,
            playbook_context=playbook_context,
            signals=signals,
            zopa=zopa,
            procurement_summary=procurement_summary,
            rag_snippets=rag_snippets,
        )

        if play_recommendations:
            negotiation_message = self._append_playbook_recommendations(
                negotiation_message,
                play_recommendations,
                playbook_context,
            )

        if state_identifier and supplier:
            self._store_session(
                state_identifier, supplier, round_no, decision.get("counter_price")
            )

        try:
            supplier_reply_count = int(state.get("supplier_reply_count", 0))
        except Exception:
            supplier_reply_count = 0

        draft_payload = {
            "intent": "NEGOTIATION_COUNTER",
            "session_reference": session_reference,
            "unique_id": session_reference,
            "supplier_id": supplier,
            "rfq_id": rfq_value,
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
            "closing_round": round_no >= 3,
            "supplier_reply_count": supplier_reply_count,
            "supplier_message": supplier_message,
            "supplier_snippets": supplier_snippets,
            "from_address": context.input_data.get("from_address"),
            "negotiation_message": negotiation_message,
            "play_recommendations": play_recommendations,
            "playbook_descriptor": playbook_context.get("descriptor"),
            "playbook_examples": playbook_context.get("examples"),
            "counter_options": counter_options,
            "positions": decision.get("positions"),
            "validation_issues": decision.get("validation_issues"),
            "outlier_alerts": decision.get("outlier_alerts"),
            "flags": decision.get("flags"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
            "normalised_inputs": normalised_inputs,
            "human_override_required": decision.get("human_override_required", False),
        }

        draft_payload.setdefault("subject", self._format_negotiation_subject(state))

        recipients = self._collect_recipient_candidates(context)
        if recipients:
            draft_payload.setdefault("recipients", recipients)

        computed_thread_headers: Optional[Dict[str, Any]] = None
        if thread_state:
            computed_thread_headers = thread_state.to_headers(round_no)

        thread_headers: Optional[Any] = None
        if isinstance(context.input_data, dict):
            incoming_thread_headers = context.input_data.get("thread_headers")
            if isinstance(incoming_thread_headers, dict):
                thread_headers = dict(incoming_thread_headers)
            else:
                thread_headers = incoming_thread_headers
        if not thread_headers:
            cached_thread_headers = state.get("last_thread_headers")
            if isinstance(cached_thread_headers, dict):
                thread_headers = dict(cached_thread_headers)
            else:
                thread_headers = cached_thread_headers
        if not thread_headers and computed_thread_headers:
            thread_headers = dict(computed_thread_headers)
        if isinstance(thread_headers, dict) and thread_headers:
            draft_payload.setdefault("thread_headers", thread_headers)
            state["last_thread_headers"] = dict(thread_headers)
        elif thread_headers:
            draft_payload.setdefault("thread_headers", thread_headers)
            state["last_thread_headers"] = thread_headers
        elif computed_thread_headers:
            draft_payload.setdefault("thread_headers", dict(computed_thread_headers))
            state["last_thread_headers"] = dict(computed_thread_headers)

        draft_metadata = {
            "counter_price": decision.get("counter_price"),
            "target_price": target_price,
            "current_offer": price,
            "round": round_no,
            "closing_round": round_no >= 3,
            "supplier_reply_count": supplier_reply_count,
            "strategy": decision.get("strategy"),
            "asks": decision.get("asks", []),
            "lead_time_request": decision.get("lead_time_request"),
            "rationale": decision.get("rationale"),
            "intent": "NEGOTIATION_COUNTER",
            "play_recommendations": play_recommendations,
            "positions": decision.get("positions"),
            "validation_issues": decision.get("validation_issues"),
            "outlier_alerts": decision.get("outlier_alerts"),
            "flags": decision.get("flags"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
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
            "session_reference": session_reference,
            "unique_id": session_reference,
            "supplier_id": supplier,
            "rfq_id": rfq_value,
            "intent": "NEGOTIATION_COUNTER",
            "metadata": draft_metadata,
            "negotiation_message": negotiation_message,
            "counter_proposals": counter_options,
            "sent_status": False,
            "thread_index": round_no,
            "subject": subject_seed,
        }
        if thread_headers:
            draft_stub["thread_headers"] = thread_headers
        draft_stub["closing_round"] = round_no >= 3
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
        if email_payload and supplier and session_reference:
            email_output = self._invoke_email_drafting_agent(context, email_payload)
            fallback_payload = dict(email_payload)
        if email_output and email_output.status == AgentStatus.SUCCESS:
            email_data = email_output.data or {}
            email_action_id = email_output.action_id or email_data.get("action_id")
            email_subject = email_data.get("subject")
            email_body = email_data.get("body")
            candidate_headers = email_data.get("thread_headers")
            if isinstance(candidate_headers, dict):
                header_message_id = self._coerce_text(
                    candidate_headers.get("Message-ID")
                    or candidate_headers.get("message_id")
                )
                if header_message_id:
                    sent_message_id = header_message_id
            drafts_payload = email_data.get("drafts")
            if isinstance(drafts_payload, list) and drafts_payload:
                for draft in drafts_payload:
                    if not isinstance(draft, dict):
                        continue
                    draft_copy = dict(draft)
                    if email_action_id:
                        draft_copy.setdefault("email_action_id", email_action_id)
                    if not sent_message_id:
                        candidate_id = self._coerce_text(
                            draft_copy.get("message_id")
                            or draft_copy.get("Message-ID")
                        )
                        if not candidate_id:
                            headers_payload = draft_copy.get("headers") or draft_copy.get("thread_headers")
                            if isinstance(headers_payload, dict):
                                candidate_id = self._coerce_text(
                                    headers_payload.get("Message-ID")
                                    or headers_payload.get("message_id")
                                )
                        if candidate_id:
                            sent_message_id = candidate_id
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
        subject_candidate = email_subject or draft_payload.get("subject")
        if subject_candidate and not state.get("base_subject"):
            base_from_email = self._normalise_base_subject(subject_candidate)
            if base_from_email:
                state["base_subject"] = base_from_email

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

        final_thread_headers = draft_payload.get("thread_headers")
        if isinstance(final_thread_headers, dict):
            if thread_state:
                subject_hint = final_thread_headers.get("Subject")
                base_subject_hint = self._normalise_base_subject(subject_hint)
                if base_subject_hint:
                    thread_state.subject_base = base_subject_hint
            if not sent_message_id:
                sent_message_id = self._coerce_text(
                    final_thread_headers.get("Message-ID")
                    or final_thread_headers.get("message_id")
                )
        elif isinstance(final_thread_headers, str) and not sent_message_id:
            sent_message_id = self._coerce_text(final_thread_headers)

        if sent_message_id and thread_state:
            thread_state.update_after_send(sent_message_id)

        state["status"] = "ACTIVE"
        state["awaiting_response"] = True
        state["current_round"] = round_no + 1
        state["last_email_sent_at"] = datetime.now(timezone.utc)
        if email_action_id:
            state["last_agent_msg_id"] = email_action_id
        self._persist_thread_state(state, thread_state)
        self._save_session_state(identifier.workflow_id, supplier, state)
        self._record_learning_snapshot(
            context,
            identifier.workflow_id or state_identifier or session_id,
            supplier,
            decision,
            state,
            True,
            supplier_reply_registered,
            rfq_id=rfq_value,
        )
        if self.response_matcher and supplier:
            try:
                self.response_matcher.register_expected_response(
                    identifier=NegotiationIdentifier(
                        workflow_id=identifier.workflow_id,
                        session_reference=session_reference,
                        supplier_id=supplier,
                        round_number=round_no,
                    ),
                    email_action_id=email_action_id,
                    expected_unique_id=session_reference,
                    timeout_seconds=int(context.input_data.get("response_timeout", 900))
                    if isinstance(context.input_data, dict)
                    else 900,
                )
            except Exception:
                logger.debug("Failed to register expected response", exc_info=True)
        cache_key: Optional[Tuple[str, str]] = None
        if state_identifier and supplier:
            cache_key = (str(state_identifier), str(supplier))
        cached_state = self._get_cached_state(cache_key)
        public_state = self._public_state(cached_state or state)
        data = {
            "supplier": supplier,
            "rfq_id": rfq_value,
            "session_reference": session_reference,
            "unique_id": session_reference,
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
            "positions": decision.get("positions"),
            "validation_issues": decision.get("validation_issues"),
            "outlier_alerts": decision.get("outlier_alerts"),
            "flags": decision.get("flags"),
            "volume_units": volume_units,
            "term_days": term_days,
            "valid_until": valid_until,
            "market_floor_price": market_floor,
            "normalised_inputs": normalised_inputs,
            "thread_state": thread_state.as_dict() if thread_state else None,
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
            workflow_id=workflow_id,
            supplier=supplier,
            drafts=draft_records,
            state=state,
            session_reference=session_reference,
            rfq_id=rfq_value,
        )
        supplier_responses: List[Dict[str, Any]] = []
        if supplier_watch_fields:
            pass_fields.update(supplier_watch_fields)
            await_response = bool(supplier_watch_fields.get("await_response"))
            awaiting_email_drafting = "EmailDraftingAgent" in next_agents
            if await_response and not awaiting_email_drafting:
                wait_results = self._await_supplier_responses(
                    context=context,
                    watch_payload=supplier_watch_fields,
                    state=state,
                )
                watch_unique_ids = [
                    self._coerce_text(entry.get("unique_id"))
                    for entry in supplier_watch_fields.get("drafts", [])
                    if isinstance(entry, dict) and self._coerce_text(entry.get("unique_id"))
                ]
                if wait_results is None:
                    logger.error(
                        "Supplier responses not received before timeout (workflow_id=%s supplier=%s unique_ids=%s)",
                        workflow_id,
                        supplier,
                        watch_unique_ids or None,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_value,
                        "workflow_id": workflow_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "round": round_no,
                        "decision": decision,
                        "message": "Supplier response not received before timeout.",
                        "unique_ids": watch_unique_ids,
                    }
                    return self._with_plan(
                        context,
                        AgentOutput(
                            status=AgentStatus.FAILED,
                            data=error_payload,
                            error="supplier response timeout",
                        ),
                    )
                wait_thread_headers: Optional[Dict[str, Any]] = None
                if isinstance(wait_results, dict):
                    candidate_headers = wait_results.get("thread_headers")
                    if isinstance(candidate_headers, dict) and candidate_headers:
                        wait_thread_headers = dict(candidate_headers)
                elif isinstance(wait_results, list):
                    for candidate in wait_results:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_headers = candidate.get("thread_headers")
                        if isinstance(candidate_headers, dict) and candidate_headers:
                            wait_thread_headers = dict(candidate_headers)
                            break
                if wait_thread_headers:
                    state["last_thread_headers"] = dict(wait_thread_headers)
                    wait_message_id = self._coerce_text(
                        wait_thread_headers.get("Message-ID")
                        or wait_thread_headers.get("message_id")
                    )
                    if wait_message_id:
                        message_id = wait_message_id
                        if thread_state:
                            thread_state.update_after_receive(wait_message_id)
                supplier_responses = [res for res in wait_results if isinstance(res, dict)]
                if not supplier_responses:
                    logger.error(
                        "No supplier responses received while waiting (workflow_id=%s supplier=%s unique_ids=%s)",
                        workflow_id,
                        supplier,
                        watch_unique_ids or None,
                    )
                    error_payload = {
                        "supplier": supplier,
                        "rfq_id": rfq_value,
                        "workflow_id": workflow_id,
                        "session_reference": session_reference,
                        "unique_id": session_reference,
                        "round": round_no,
                        "decision": decision,
                        "message": "Missing supplier responses after wait.",
                        "unique_ids": watch_unique_ids,
                    }
                    return self._with_plan(
                        context,
                        AgentOutput(
                            status=AgentStatus.FAILED,
                            data=error_payload,
                            error="supplier response missing",
                        ),
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
                        if thread_state:
                            thread_state.update_after_receive(message_token)
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
                self._persist_thread_state(state, thread_state)
                self._save_session_state(identifier.workflow_id, supplier, state)
                public_state = self._public_state(state)
                data["session_state"] = public_state
                data["awaiting_response"] = False
                pass_fields["session_state"] = public_state
                pass_fields.pop("await_response", None)
                pass_fields.pop("await_all_responses", None)
                self._record_learning_snapshot(
                    context,
                    identifier.workflow_id or state_identifier or session_id,
                    supplier,
                    decision,
                    state,
                    False,
                    bool(new_responses) or supplier_reply_registered,
                    rfq_id=rfq_value,
                )
            else:
                if await_response and "SupplierInteractionAgent" not in next_agents:
                    next_agents.append("SupplierInteractionAgent")
        if supplier_responses:
            data["supplier_responses"] = supplier_responses
            pass_fields["supplier_responses"] = supplier_responses

        if next_agents:
            merge_payload = dict(fallback_payload or {})
            merge_payload.setdefault("intent", "NEGOTIATION_COUNTER")
            merge_payload.setdefault("decision", decision)
            merge_payload["session_state"] = public_state
            if session_reference is not None:
                merge_payload.setdefault("session_reference", session_reference)
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
            "NegotiationAgent prepared counter round %s for supplier=%s session_reference=%s",
            round_no,
            supplier,
            session_reference,
        )

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=pass_fields,
                next_agents=next_agents,
            ),
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
        price_val = self._coerce_float(price)
        target_val = self._coerce_float(target)

        buyer_max = None
        if target_val is not None and price_val is not None:
            buyer_max = min(target_val, price_val)
        elif target_val is not None:
            buyer_max = target_val
        elif price_val is not None:
            buyer_max = price_val

        buyer_max = self._validate_buyer_max(buyer_max)

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

        buyer_max = self._validate_buyer_max(zopa.get("buyer_max"))
        supplier_floor = self._coerce_float(zopa.get("supplier_floor"))
        entry = self._coerce_float(zopa.get("entry_counter"))
        if not price_locked and price and target_price and entry:
            if current_round == 1:
                decision["counter_price"] = min(entry, (price + target_price) / 2)
            else:
                if (
                    buyer_max is not None
                    and supplier_floor is not None
                    and supplier_floor < buyer_max
                ):
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

        buyer_max = self._validate_buyer_max(zopa.get("buyer_max"))
        supplier_floor = self._coerce_float(zopa.get("supplier_floor"))
        entry = self._coerce_float(zopa.get("entry_counter"))

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
            if buyer_max is not None and value > buyer_max:
                value = buyer_max
            if supplier_floor is not None and value < supplier_floor:
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
            "last_thread_headers": None,
            "positions": {},
            "last_counter_price": None,
            "thread_state": None,
            "workflow_id": None,
            "session_reference": None,
        }

    def _public_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "supplier_reply_count": int(state.get("supplier_reply_count", 0)),
            "current_round": int(state.get("current_round", 1)),
            "status": state.get("status", "ACTIVE"),
            "awaiting_response": bool(state.get("awaiting_response", False)),
        }

    def _load_thread_state(
        self,
        identifier: NegotiationIdentifier,
        state: Dict[str, Any],
        *,
        subject_hint: Optional[str] = None,
    ) -> EmailThreadState:
        fallback_subject = self._normalise_base_subject(state.get("base_subject"))
        if not fallback_subject:
            fallback_subject = self._normalise_base_subject(subject_hint)
        if not fallback_subject:
            fallback_subject = DEFAULT_NEGOTIATION_SUBJECT

        thread_payload = state.get("thread_state")
        if isinstance(thread_payload, dict) and thread_payload:
            thread_state = EmailThreadState.from_dict(
                thread_payload, fallback_subject=fallback_subject
            )
        else:
            thread_state = EmailThreadState(
                thread_id=f"<{uuid.uuid4()}@procwise.co.uk>",
                subject_base=fallback_subject,
            )

        if not thread_state.subject_base:
            thread_state.subject_base = fallback_subject

        if not thread_state.references:
            base_reference = state.get("last_agent_msg_id") or state.get("last_supplier_msg_id")
            token = EmailThreadState._normalise_token(base_reference)
            if token and token not in thread_state.references:
                thread_state.references.append(token)

        state["thread_state"] = thread_state.as_dict()
        state["workflow_id"] = identifier.workflow_id
        state["session_reference"] = identifier.session_reference
        return thread_state

    @staticmethod
    def _persist_thread_state(
        state: Dict[str, Any], thread_state: Optional[EmailThreadState]
    ) -> None:
        if thread_state is None:
            return
        state["thread_state"] = thread_state.as_dict()

    def _get_cached_state(
        self, cache_key: Optional[Tuple[str, str]]
    ) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        with self._state_lock:
            cached = self._state_cache.get(cache_key)
            return dict(cached) if isinstance(cached, dict) else None

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
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS thread_state JSONB"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS workflow_id VARCHAR(255)"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS session_reference VARCHAR(255)"
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.debug("failed to ensure negotiation state schema", exc_info=True)
        finally:
            self._state_schema_checked = True

    def _get_identifier_column(
        self,
        table: str,
        *,
        default: str = "workflow_id",
        fallback: str = "rfq_id",
        alt_fallback: str = "session_id",
    ) -> str:
        attr_name = f"_{table}_identifier_column"
        cached = getattr(self, attr_name, None)
        if isinstance(cached, str) and cached:
            return cached

        column = default
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if callable(get_conn):
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT column_name
                              FROM information_schema.columns
                             WHERE table_schema = 'proc'
                               AND table_name = %s
                               AND column_name = %s
                            """,
                            (table, default),
                        )
                        if cur.fetchone():
                            column = default
                        else:
                            cur.execute(
                                """
                                SELECT column_name
                                  FROM information_schema.columns
                                 WHERE table_schema = 'proc'
                                   AND table_name = %s
                                   AND column_name = %s
                                """,
                                (table, fallback),
                            )
                            if cur.fetchone():
                                column = fallback
                            else:
                                cur.execute(
                                    """
                                    SELECT column_name
                                      FROM information_schema.columns
                                     WHERE table_schema = 'proc'
                                       AND table_name = %s
                                       AND column_name = %s
                                    """,
                                    (table, alt_fallback),
                                )
                                if cur.fetchone():
                                    column = alt_fallback
                    conn.commit()
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "failed to detect identifier column for %s", table, exc_info=True
                )

        setattr(self, attr_name, column)
        return column

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
        self, session_id: Optional[str], supplier: Optional[str]
    ) -> Tuple[Dict[str, Any], bool]:
        identifier = self._coerce_text(session_id)
        supplier_id = self._coerce_text(supplier)
        if not identifier or not supplier_id:
            return self._default_state(), False
        self._ensure_state_schema()
        key = (identifier, supplier_id)
        with self._state_lock:
            if key in self._state_cache:
                return dict(self._state_cache[key]), True

        state = self._default_state()
        exists = False
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    column = self._get_identifier_column("negotiation_session_state")
                    cur.execute(
                        f"""
                        SELECT supplier_reply_count, current_round, status, awaiting_response,
                               last_supplier_msg_id, last_agent_msg_id, last_email_sent_at,
                               base_subject, initial_body, thread_state, workflow_id, session_reference
                          FROM proc.negotiation_session_state
                         WHERE {column} = %s AND supplier_id = %s
                        """,
                        (identifier, supplier_id),
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
                            thread_state_raw,
                            workflow_value,
                            session_reference_value,
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
                        if thread_state_raw:
                            if isinstance(thread_state_raw, (bytes, bytearray, memoryview)):
                                try:
                                    thread_state_raw = thread_state_raw.tobytes().decode("utf-8")
                                except Exception:
                                    thread_state_raw = None
                            if isinstance(thread_state_raw, str):
                                try:
                                    state["thread_state"] = json.loads(thread_state_raw)
                                except Exception:
                                    state["thread_state"] = None
                            elif isinstance(thread_state_raw, dict):
                                state["thread_state"] = dict(thread_state_raw)
                        if workflow_value:
                            state["workflow_id"] = self._coerce_text(workflow_value)
                        if session_reference_value:
                            state["session_reference"] = self._coerce_text(
                                session_reference_value
                            )
                        exists = True
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to load negotiation session state")
        with self._state_lock:
            self._state_cache[key] = dict(state)
        return dict(state), exists

    def _save_session_state(
        self, session_id: Optional[str], supplier: Optional[str], state: Dict[str, Any]
    ) -> None:
        identifier = self._coerce_text(session_id)
        supplier_id = self._coerce_text(supplier)
        if not identifier or not supplier_id:
            return
        key = (identifier, supplier_id)
        with self._state_lock:
            self._state_cache[key] = dict(state)
        self._ensure_state_schema()
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    column = self._get_identifier_column("negotiation_session_state")
                    columns = [
                        column,
                        "supplier_id",
                        "supplier_reply_count",
                        "current_round",
                        "status",
                        "awaiting_response",
                        "last_supplier_msg_id",
                        "last_agent_msg_id",
                        "last_email_sent_at",
                        "base_subject",
                        "initial_body",
                    ]
                    values: List[Any] = [
                        identifier,
                        supplier_id,
                        int(state.get("supplier_reply_count", 0)),
                        int(state.get("current_round", 1)),
                        state.get("status", "ACTIVE"),
                        bool(state.get("awaiting_response", False)),
                        state.get("last_supplier_msg_id"),
                        state.get("last_agent_msg_id"),
                        state.get("last_email_sent_at"),
                        state.get("base_subject"),
                        state.get("initial_body"),
                    ]
                    updates = [
                        "supplier_reply_count = EXCLUDED.supplier_reply_count",
                        "current_round = EXCLUDED.current_round",
                        "status = EXCLUDED.status",
                        "awaiting_response = EXCLUDED.awaiting_response",
                        "last_supplier_msg_id = EXCLUDED.last_supplier_msg_id",
                        "last_agent_msg_id = EXCLUDED.last_agent_msg_id",
                        "last_email_sent_at = EXCLUDED.last_email_sent_at",
                        "base_subject = EXCLUDED.base_subject",
                        "initial_body = EXCLUDED.initial_body",
                    ]

                    thread_state_payload = state.get("thread_state")
                    if thread_state_payload is not None:
                        try:
                            thread_state_serialised = json.dumps(
                                thread_state_payload, ensure_ascii=False, default=str
                            )
                        except Exception:
                            thread_state_serialised = None
                        columns.append("thread_state")
                        values.append(thread_state_serialised)
                        updates.append("thread_state = EXCLUDED.thread_state")

                    workflow_value = state.get("workflow_id") or (
                        identifier if column != "workflow_id" else None
                    )
                    if workflow_value and column != "workflow_id":
                        columns.append("workflow_id")
                        values.append(workflow_value)
                        updates.append("workflow_id = EXCLUDED.workflow_id")

                    session_reference_value = state.get("session_reference")
                    if session_reference_value:
                        columns.append("session_reference")
                        values.append(session_reference_value)
                        updates.append("session_reference = EXCLUDED.session_reference")

                    updated_on = datetime.now(timezone.utc)
                    columns.append("updated_on")
                    values.append(updated_on)
                    updates.append("updated_on = EXCLUDED.updated_on")

                    placeholders = ", ".join(["%s"] * len(values))
                    column_list = ", ".join(columns)
                    update_clause = ", ".join(updates)

                    cur.execute(
                        f"""
                        INSERT INTO proc.negotiation_session_state ({column_list})
                        VALUES ({placeholders})
                        ON CONFLICT ({column}, supplier_id) DO UPDATE SET {update_clause}
                        """,
                        tuple(values),
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
        """Append structured playbook recommendations to the negotiation summary."""

        if not plays:
            return summary

        lines: List[str] = [summary.rstrip()]

        recommendation_lines: List[str] = []
        lever_hints: List[str] = []
        for play in plays[:4]:
            if not isinstance(play, dict):
                continue
            lever = str(play.get("lever", "")).strip()
            description = str(play.get("play", "")).strip()
            if description:
                if lever:
                    recommendation_lines.append(f"- {lever}: {description}")
                else:
                    recommendation_lines.append(f"- {description}")
            if lever:
                lookup_key = lever.strip().title()
                hint = TRADE_OFF_HINTS.get(lookup_key)
                if hint:
                    lever_hints.append(f"- {lookup_key}: {hint}")

        if recommendation_lines:
            lines.append("\nRecommended plays:")
            lines.extend(recommendation_lines)

        if lever_hints:
            lines.append("\nTrade-off considerations:")
            lines.extend(lever_hints[:3])

        return "\n".join(lines)

    def _compose_negotiation_message(
        self,
        *,
        context: AgentContext,
        decision: Dict[str, Any],
        positions: NegotiationPositions,
        round_no: int,
        currency: Optional[str],
        supplier: Optional[str],
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        zopa: Dict[str, Any],
        playbook_context: Optional[Dict[str, Any]],
        procurement_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Compose a human-like, strategically crafted negotiation message."""

        counter_price = decision.get("counter_price")
        current_offer = positions.supplier_offer
        target_price = positions.desired
        strategy = decision.get("strategy", "counter")

        # Determine negotiation tone based on round and context
        tone = self._determine_negotiation_tone(round_no, signals, strategy)

        # Build opening that establishes rapport
        opening = self._craft_opening(round_no, supplier_message, signals, tone)

        # Acknowledge supplier's position (reciprocity principle)
        acknowledgment = self._craft_acknowledgment(
            current_offer, supplier_message, signals, currency, round_no
        )

        # Present our position with reasoning (anchoring with justification)
        position_statement = self._craft_position_statement(
            counter_price=counter_price,
            current_offer=current_offer,
            target_price=target_price,
            currency=currency,
            round_no=round_no,
            signals=signals,
            zopa=zopa,
            procurement_summary=procurement_summary,
        )

        # Weave in playbook recommendations naturally
        value_proposition = self._craft_value_proposition(
            playbook_context=playbook_context,
            decision=decision,
            round_no=round_no,
            tone=tone,
        )

        # Create collaborative asks (frame as mutual benefit)
        collaborative_asks = self._craft_collaborative_asks(
            decision=decision,
            round_no=round_no,
            signals=signals,
            tone=tone,
        )

        # Closing that maintains momentum
        closing = self._craft_closing(round_no, strategy, tone)

        # Assemble the complete message
        message_parts = [
            opening,
            acknowledgment,
            position_statement,
            value_proposition,
            collaborative_asks,
            closing,
        ]

        return "\n\n".join(part for part in message_parts if part)

    def _determine_negotiation_tone(
        self, round_no: int, signals: Dict[str, Any], strategy: str
    ) -> str:
        """Determine the appropriate negotiation tone."""

        if strategy in ("accept", "decline"):
            return "decisive"

        if signals.get("finality_hint"):
            return "collaborative-firm"

        if signals.get("capacity_tight"):
            return "understanding-assertive"

        if round_no == 1:
            return "exploratory-confident"
        elif round_no == 2:
            return "focused-collaborative"
        else:
            return "closure-oriented"

    def _craft_opening(
        self,
        round_no: int,
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        tone: str,
    ) -> str:
        """Craft a rapport-building opening."""

        # Acknowledge their response if this isn't the first round
        if round_no > 1 and supplier_message:
            if signals.get("finality_hint"):
                return (
                    "Thank you for taking the time to provide such a detailed response. "
                    "I appreciate your transparency about your position, and I'd like to explore "
                    "whether there's a way we can structure this to work for both parties."
                )
            elif signals.get("capacity_tight"):
                return (
                    "I appreciate you sharing the challenges around capacity and lead times. "
                    "Understanding your operational constraints helps us find creative solutions "
                    "that work within those parameters."
                )
            else:
                return (
                    "Thank you for your continued engagement on this. I've reviewed your proposal "
                    "and believe we're making good progress toward an agreement that benefits both sides."
                )

        # First round opening
        if tone == "exploratory-confident":
            return (
                "Thank you for submitting your proposal. I've had a chance to review the details "
                "and would like to discuss how we might align this with our project requirements "
                "and budget parameters."
            )

        return "I'd like to continue our discussion on finding the right structure for this partnership."

    def _craft_acknowledgment(
        self,
        current_offer: Optional[float],
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        currency: Optional[str],
        round_no: int,
    ) -> str:
        """Acknowledge supplier's position (builds reciprocity)."""

        if round_no == 1 or not supplier_message:
            return ""

        acknowledgments: List[str] = []

        # Acknowledge specific points they raised
        if signals.get("capacity_tight"):
            acknowledgments.append(
                "I understand the capacity constraints you mentioned, and we're certainly "
                "mindful of the current market dynamics affecting lead times."
            )

        if signals.get("moq"):
            moq_value = signals.get("moq")
            acknowledgments.append(
                f"Regarding your minimum order quantity of {moq_value} units, "
                "we're confident our volumes can support that threshold."
            )

        if signals.get("payment_terms_hint"):
            acknowledgments.append(
                "Your flexibility on payment terms is noted and appreciated—"
                "that's definitely an area where we can find mutual value."
            )

        if not acknowledgments and current_offer:
            offer_text = self._format_currency(current_offer, currency)
            acknowledgments.append(
                f"Your quoted price of {offer_text} reflects the quality and service level "
                "you bring, which we certainly value."
            )

        return " ".join(acknowledgments) if acknowledgments else ""

    def _craft_position_statement(
        self,
        *,
        counter_price: Optional[float],
        current_offer: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        signals: Dict[str, Any],
        zopa: Dict[str, Any],
        procurement_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Present our position with strategic justification."""

        if counter_price is None:
            return (
                "Before we can move forward, I'd like to better understand the cost drivers "
                "behind your proposal. Could you provide a breakdown of the key components "
                "affecting the pricing structure?"
            )

        counter_text = self._format_currency(counter_price, currency)
        offer_text = self._format_currency(current_offer, currency) if current_offer else ""

        # Build justification based on available data
        justifications: List[str] = []

        # Use market intelligence if available
        supplier_floor = zopa.get("supplier_floor")

        if supplier_floor and current_offer:
            try:
                market_gap_pct = (current_offer - supplier_floor) / supplier_floor
                if market_gap_pct > 0.15:
                    justifications.append(
                        "Based on our market analysis and benchmarking across similar products, "
                        "we're seeing a meaningful opportunity to optimize the pricing structure"
                    )
            except (TypeError, ZeroDivisionError):
                pass

        # Use procurement history if available
        if procurement_summary and procurement_summary.get("metrics"):
            metrics = procurement_summary["metrics"]
            avg_value = metrics.get("average_transaction_value")
            if avg_value and counter_price:
                try:
                    if abs(counter_price - avg_value) / avg_value < 0.20:
                        avg_text = self._format_currency(avg_value, currency)
                        justifications.append(
                            f"Our historical spend on comparable items has averaged around {avg_text}, "
                            "which helps frame our expectations for this engagement"
                        )
                except (TypeError, ZeroDivisionError):
                    pass

        # Frame based on round
        if round_no == 1:
            intro = (
                f"To align with our project budget and market benchmarks, "
                f"I'd like to propose we structure this around {counter_text}. "
            )
        elif round_no == 2:
            gap_direction = "narrowing" if current_offer and counter_price and current_offer > counter_price else "refining"
            intro = (
                f"As we continue {gap_direction} the gap, I believe {counter_text} represents "
                f"a fair midpoint that reflects both the value you're delivering and our budget realities. "
            )
        else:  # Round 3+
            intro = (
                f"To help us reach closure, I'm proposing {counter_text} as our target. "
                f"This represents our best position given the total business case, "
            )

        # Add strongest justification
        if justifications:
            position = intro + justifications[0] + "."
        else:
            position = intro + "and I believe it sets us up for a successful long-term relationship."

        # Add gap context if material
        if current_offer and counter_price and target_price:
            try:
                movement_from_offer = ((current_offer - counter_price) / current_offer) * 100
                if movement_from_offer > 8:
                    position += (
                        f" This represents a meaningful movement from your {offer_text} quote, "
                        f"and demonstrates our commitment to finding common ground."
                    )
            except (TypeError, ZeroDivisionError):
                pass

        return position

    def _craft_value_proposition(
        self,
        *,
        playbook_context: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
        round_no: int,
        tone: str,
    ) -> str:
        """Weave playbook recommendations into natural value propositions."""

        if not playbook_context or not playbook_context.get("plays"):
            return self._craft_generic_value_proposition(decision, round_no)

        plays = playbook_context["plays"][:3]  # Top 3 plays
        supplier_type = playbook_context.get("supplier_type")

        value_statements: List[str] = []

        # Group plays by lever for coherent messaging
        plays_by_lever: Dict[str, List[Dict[str, Any]]] = {}
        for play in plays:
            lever = play.get("lever", "Other")
            if lever not in plays_by_lever:
                plays_by_lever[lever] = []
            plays_by_lever[lever].append(play)

        # Craft integrated value propositions
        for lever, lever_plays in list(plays_by_lever.items())[:2]:  # Focus on top 2 levers
            if lever == "Commercial":
                value_statements.append(
                    self._weave_commercial_play(lever_plays, round_no, tone)
                )
            elif lever == "Operational":
                value_statements.append(
                    self._weave_operational_play(lever_plays, round_no, tone)
                )
            elif lever == "Strategic":
                value_statements.append(
                    self._weave_strategic_play(lever_plays, round_no, tone)
                )
            elif lever == "Risk":
                value_statements.append(
                    self._weave_risk_play(lever_plays, round_no, tone)
                )
            elif lever == "Relational":
                value_statements.append(
                    self._weave_relational_play(lever_plays, round_no, tone)
                )

        if not value_statements:
            return self._craft_generic_value_proposition(decision, round_no)

        # Add context about partnership type
        intro = ""
        if supplier_type and round_no <= 2:
            if supplier_type == "Strategic":
                intro = "Given the strategic nature of this relationship, "
            elif supplier_type == "Leverage":
                intro = "To maximize the value of this partnership, "
            elif supplier_type == "Bottleneck":
                intro = "Understanding the critical nature of supply continuity, "

        return intro + " ".join(value_statements)

    def _weave_commercial_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave commercial plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Volume-based plays
        if any("volume" in p.lower() or "tier" in p.lower() for p in play_texts):
            return (
                "I'd like to explore how we can structure volume commitments to unlock "
                "better unit economics for both sides. If we can commit to tier-based volumes—"
                "say, 250 units initially with pathways to 500+—there may be room to optimize "
                "the pricing structure while giving you better demand visibility."
            )

        # Early payment plays
        if any("early payment" in p.lower() or "net-15" in p.lower() for p in play_texts):
            return (
                "One area where we can create immediate value is payment terms. "
                "If we can accelerate payment to net-15, would that open up opportunities "
                "for a 2-3% discount? This improves your cash flow while reducing our total cost."
            )

        # Bundling plays
        if any("bundle" in p.lower() or "consolidate" in p.lower() for p in play_texts):
            return (
                "We're also looking at how we might consolidate spend across multiple categories "
                "or adjacent products. If we can bundle this with other requirements coming through "
                "our pipeline, there could be meaningful volume synergies that benefit both parties."
            )

        # Generic commercial
        return (
            "I believe there's room to structure the commercial terms in a way that creates "
            "value for both organizations—whether through volume commitments, payment optimization, "
            "or longer-term price stability."
        )

    def _weave_operational_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave operational plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Delivery/lead time plays
        if any("delivery" in p.lower() or "lead time" in p.lower() for p in play_texts):
            return (
                "On the operational side, delivery timing is critical for our project schedule. "
                "If you can guarantee delivery within 2 weeks or offer split shipments "
                "(perhaps 30% upfront, balance within 4 weeks), that would significantly de-risk "
                "our production timeline and justify the investment."
            )

        # Planning/forecasting plays
        if any("forecast" in p.lower() or "planning" in p.lower() for p in play_texts):
            return (
                "We're happy to share our demand forecasts and collaborate on supply planning "
                "to give you better visibility. This integrated approach typically helps both sides "
                "reduce buffer stock and improve fulfillment rates."
            )

        # SLA/service level plays
        if any("sla" in p.lower() or "priority" in p.lower() for p in play_texts):
            return (
                "To ensure this partnership meets both our operational needs, I'd like to define "
                "clear service levels around delivery performance and fulfillment rates. "
                "Having those guardrails helps us plan effectively and holds both parties accountable."
            )

        # Generic operational
        return (
            "From an operational standpoint, we're looking for reliability and responsiveness "
            "in fulfillment. If we can align on clear delivery commitments and planning cadences, "
            "that creates a strong foundation for the partnership."
        )

    def _weave_strategic_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave strategic plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Innovation/co-development plays
        if any("innovation" in p.lower() or "co-develop" in p.lower() for p in play_texts):
            return (
                "Beyond the immediate transaction, I see potential for deeper collaboration "
                "on product development and innovation. If we can align on a shared roadmap "
                "for the next 12-24 months, there may be opportunities to co-invest in capabilities "
                "that benefit both our organizations."
            )

        # ESG/sustainability plays
        if any("esg" in p.lower() or "sustainab" in p.lower() for p in play_texts):
            return (
                "Sustainability is increasingly important to our stakeholders. "
                "If we can incorporate ESG metrics and carbon reduction targets into our partnership, "
                "that strengthens the strategic case and may unlock internal budget flexibility."
            )

        # Long-term commitment plays
        if any("long-term" in p.lower() or "multi-year" in p.lower() for p in play_texts):
            return (
                "We're thinking about this as a multi-year partnership rather than a one-off transaction. "
                "If you're open to a longer-term commitment with price stability mechanisms, "
                "we can structure this in a way that gives you revenue predictability while securing "
                "our supply chain."
            )

        # Generic strategic
        return (
            "Strategically, we're looking to build partnerships that go beyond transactional relationships. "
            "If we can align on shared objectives and longer-term value creation, "
            "that opens up different ways to structure the commercial terms."
        )

    def _weave_risk_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave risk plays into natural language."""

        play_texts = [p.get("play", "") for p in plays]

        # Warranty plays
        if any("warranty" in p.lower() for p in play_texts):
            return (
                "Given the mission-critical nature of these components, warranty coverage is important. "
                "If you can extend warranty to 2-3 years at no additional cost, "
                "that reduces our total cost of ownership and makes the business case stronger."
            )

        # Service credit plays
        if any("service credit" in p.lower() or "sla penalt" in p.lower() for p in play_texts):
            return (
                "To manage performance risk, I'd like to incorporate service-level agreements "
                "with appropriate credits if delivery or quality targets aren't met. "
                "This ensures we have recourse mechanisms without damaging the relationship."
            )

        # Dual-sourcing plays
        if any("dual-sourc" in p.lower() or "alternative" in p.lower() for p in play_texts):
            return (
                "From a risk management perspective, we're evaluating dual-sourcing strategies "
                "to protect against supply disruption. If you can offer competitive terms and strong SLAs, "
                "you'd be well-positioned as our primary supplier with the volumes that come with that."
            )

        # Generic risk
        return (
            "We need to ensure appropriate risk mitigation through warranty coverage, "
            "performance guarantees, and clear remediation processes. "
            "Building these protections into the agreement benefits both parties."
        )

    def _weave_relational_play(
        self, plays: List[Dict[str, Any]], round_no: int, tone: str
    ) -> str:
        """Weave relational plays into natural language."""

        return (
            "Looking beyond this specific engagement, we value suppliers who can grow with us "
            "and become trusted partners. If this initial project goes well, there are "
            "significant opportunities for expanded business across our organization. "
            "That long-term potential should factor into how we structure the initial terms."
        )

    def _craft_generic_value_proposition(
        self, decision: Dict[str, Any], round_no: int
    ) -> str:
        """Fallback value proposition when playbook unavailable."""

        asks = decision.get("asks", [])
        if not asks:
            return ""

        if round_no <= 2:
            return (
                "To make this work within our budget parameters, I'd like to explore "
                "areas where we can create mutual value—whether through volume commitments, "
                "payment terms optimization, or longer-term partnership structures."
            )
        else:
            return (
                "As we work toward closure, I believe there are still opportunities "
                "to optimize the total package through creative structuring of terms, "
                "payment schedules, and service commitments."
            )

    def _craft_collaborative_asks(
        self,
        *,
        decision: Dict[str, Any],
        round_no: int,
        signals: Dict[str, Any],
        tone: str,
    ) -> str:
        """Frame asks as collaborative opportunities."""

        asks = decision.get("asks", [])
        lead_time_request = decision.get("lead_time_request")

        if not asks and not lead_time_request:
            return ""

        # Frame based on tone
        if tone in ("decisive", "closure-oriented"):
            intro = "To finalize this agreement, there are a few specific elements I'd like to confirm:"
        else:
            intro = "To help us structure the best possible outcome, I'd like to explore a few areas:"

        formatted_asks: List[str] = []

        # Convert asks into questions/proposals rather than demands
        for ask in asks[:4]:  # Limit to top 4
            ask_text = str(ask).strip()

            # Rephrase common asks to be more collaborative
            if "volume" in ask_text.lower() or "tier" in ask_text.lower():
                formatted_asks.append(
                    "• What volume thresholds would unlock better unit economics? "
                    "We're confident we can hit meaningful tiers if the pricing justifies it."
                )
            elif "payment" in ask_text.lower() and ("early" in ask_text.lower() or "discount" in ask_text.lower()):
                formatted_asks.append(
                    "• Would accelerated payment (net-15) create value for you? "
                    "We'd be happy to explore if that opens up pricing flexibility."
                )
            elif "lead time" in ask_text.lower() or "delivery" in ask_text.lower():
                formatted_asks.append(
                    "• Can you confirm the delivery timeline and whether there's flexibility "
                    "for partial shipments if that helps manage both our schedules?"
                )
            elif "warranty" in ask_text.lower():
                formatted_asks.append(
                    "• What warranty coverage is included, and is there room to extend that "
                    "as part of the overall package?"
                )
            elif "breakdown" in ask_text.lower() or "cost" in ask_text.lower():
                formatted_asks.append(
                    "• Would you be open to sharing a high-level cost breakdown? "
                    "That transparency helps us justify the investment internally."
                )
            elif "alternative" in ask_text.lower() or "spec" in ask_text.lower():
                formatted_asks.append(
                    "• Are there alternative specifications or components that could reduce cost "
                    "while still meeting our performance requirements?"
                )
            else:
                # Generic reframe
                formatted_asks.append(f"• {ask_text}")

        # Add lead time if specified
        if lead_time_request and not any("lead time" in fa.lower() for fa in formatted_asks):
            formatted_asks.append(
                f"• Regarding timing: {lead_time_request.lower()} would be ideal for our project schedule."
            )

        if not formatted_asks:
            return ""

        return intro + "\n\n" + "\n".join(formatted_asks)

    def _craft_closing(self, round_no: int, strategy: str, tone: str) -> str:
        """Create a closing that maintains momentum."""

        if strategy == "accept":
            return (
                "If we can align on these final points, I'm ready to move forward quickly "
                "and get the paperwork in motion. Looking forward to your thoughts."
            )

        if strategy == "decline":
            return (
                "I appreciate your engagement throughout this process. Given where we've landed, "
                "I'll need to take this back to my team for further discussion. "
                "I'll circle back if our parameters change."
            )

        if round_no >= 3:
            return (
                "I'm hopeful we can find common ground here. This represents our best position "
                "given all the factors at play. I'd appreciate your thoughts on whether we can "
                "make this work, and I'm happy to jump on a call if that would be helpful "
                "to talk through any remaining sticking points."
            )

        if round_no == 2:
            return (
                "I believe we're getting close to something that works for both sides. "
                "Let me know your thoughts on this structure, and we can refine from there. "
                "Happy to discuss any concerns or questions you might have."
            )

        # Round 1 or default
        return (
            "I'd welcome your perspective on this proposal. If you have questions or "
            "want to discuss any of these points in more detail, I'm happy to set up "
            "a quick call. Looking forward to your response."
        )

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
        context: AgentContext,
        session_reference: Optional[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        *,
        supplier: Optional[str] = None,
        supplier_snippets: Optional[List[str]] = None,
        supplier_message: Optional[str] = None,
        playbook_context: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
        zopa: Optional[Dict[str, Any]] = None,
        procurement_summary: Optional[Dict[str, Any]] = None,
        rag_snippets: Optional[List[str]] = None,
    ) -> str:
        """Build negotiation message using skilled negotiation techniques."""

        positions = self._build_positions_from_decision(
            decision, price, target_price, round_no
        )

        if self._use_enhanced_messages and hasattr(
            self, "_compose_negotiation_message"
        ):
            try:
                return self._compose_negotiation_message(
                    context=context,
                    decision=decision,
                    positions=positions,
                    round_no=round_no,
                    currency=currency,
                    supplier=supplier,
                    supplier_message=supplier_message,
                    signals=signals or {},
                    zopa=zopa or {},
                    playbook_context=playbook_context,
                    procurement_summary=procurement_summary,
                )
            except Exception as exc:
                logger.warning(
                    "Enhanced message composer failed: %s, using fallback",
                    exc,
                    exc_info=True,
                )

        return self._build_summary_fallback(
            context=context,
            session_reference=session_reference,
            decision=decision,
            price=price,
            target_price=target_price,
            currency=currency,
            round_no=round_no,
            supplier=supplier,
            supplier_snippets=supplier_snippets,
            supplier_message=supplier_message,
            playbook_context=playbook_context,
            signals=signals,
            zopa=zopa,
            procurement_summary=procurement_summary,
            rag_snippets=rag_snippets,
        )

    def _build_positions_from_decision(
        self,
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        round_no: int,
    ) -> NegotiationPositions:
        """Build NegotiationPositions object from decision data."""

        positions_dict = decision.get("positions", {})
        if isinstance(positions_dict, dict):
            history = positions_dict.get("history", [])
        else:
            history = []

        return NegotiationPositions(
            start=decision.get("start_position") or positions_dict.get("start"),
            desired=decision.get("desired_position")
            or positions_dict.get("desired")
            or target_price,
            no_deal=decision.get("no_deal_position") or positions_dict.get("no_deal"),
            supplier_offer=price,
            history=history if isinstance(history, list) else [],
        )

    def _build_summary_fallback(
        self,
        context: AgentContext,
        session_reference: Optional[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        *,
        supplier: Optional[str] = None,
        supplier_snippets: Optional[List[str]] = None,
        supplier_message: Optional[str] = None,
        playbook_context: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
        zopa: Optional[Dict[str, Any]] = None,
        procurement_summary: Optional[Dict[str, Any]] = None,
        rag_snippets: Optional[List[str]] = None,
    ) -> str:
        """Fallback basic summary builder (original implementation)."""

        workflow_reference = session_reference or context.input_data.get("workflow_id")
        if not workflow_reference:
            workflow_reference = getattr(context, "workflow_id", None)
        reference_text = str(workflow_reference).strip() if workflow_reference else "workflow"

        strategy = decision.get("strategy")
        lines: List[str] = []
        header = f"Round {round_no} plan for {reference_text}"
        if strategy:
            header += f": {strategy}"
        lines.append(header)

        if round_no >= 3:
            lines.append(
                "- Closing round: request the supplier's best and final quote and confirm readiness to award"
            )

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

    def _compose_negotiation_message(
        self,
        *,
        context: AgentContext,
        decision: Dict[str, Any],
        positions: NegotiationPositions,
        round_no: int,
        currency: Optional[str],
        supplier: Optional[str],
        supplier_message: Optional[str],
        signals: Dict[str, Any],
        zopa: Dict[str, Any],
        playbook_context: Optional[Dict[str, Any]],
        procurement_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Compose a human-like, strategically crafted negotiation message."""

        counter_price = decision.get("counter_price")
        current_offer = positions.supplier_offer
        target_price = positions.desired
        strategy = decision.get("strategy", "counter")

        sections: List[str] = []

        opening = self._craft_opening_simple(round_no, supplier_message, signals)
        if opening:
            sections.append(opening)

        if round_no > 1 and supplier_message:
            acknowledgment = self._craft_acknowledgment_simple(
                current_offer, signals, currency
            )
            if acknowledgment:
                sections.append(acknowledgment)

        position = self._craft_position_simple(
            counter_price, current_offer, target_price, currency, round_no
        )
        if position:
            sections.append(position)

        if playbook_context and playbook_context.get("plays"):
            value_prop = self._craft_value_proposition_simple(playbook_context, round_no)
            if value_prop:
                sections.append(value_prop)

        asks_section = self._craft_asks_simple(decision)
        if asks_section:
            sections.append(asks_section)

        closing = self._craft_closing_simple(round_no, strategy)
        if closing:
            sections.append(closing)

        return "\n\n".join(sections)

    def _craft_opening_simple(
        self, round_no: int, supplier_message: Optional[str], signals: Dict[str, Any]
    ) -> str:
        """Craft a simple opening."""
        if round_no > 1 and supplier_message:
            return (
                "Thank you for your response. I've reviewed your proposal and would like to discuss "
                "how we can align on the details."
            )
        return (
            "Thank you for your proposal. I'd like to discuss the terms to ensure we can reach an "
            "agreement that works for both parties."
        )

    def _craft_acknowledgment_simple(
        self, current_offer: Optional[float], signals: Dict[str, Any], currency: Optional[str]
    ) -> str:
        """Craft simple acknowledgment."""
        if signals.get("capacity_tight"):
            return (
                "I understand the capacity constraints you mentioned, and we're mindful of the current "
                "market dynamics."
            )
        if current_offer:
            offer_text = self._format_currency(current_offer, currency)
            return (
                f"Your quoted price of {offer_text} reflects the quality you provide, which we value."
            )
        return ""

    def _craft_position_simple(
        self,
        counter_price: Optional[float],
        current_offer: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
    ) -> str:
        """Craft simple position statement."""
        if counter_price is None:
            return "Before we proceed, could you provide more details on the cost breakdown?"

        counter_text = self._format_currency(counter_price, currency)

        if round_no == 1:
            return (
                f"To align with our project budget, I'd like to propose we structure this around {counter_text}."
            )
        if round_no == 2:
            return (
                f"As we work toward agreement, I believe {counter_text} represents a fair position that reflects both the value "
                "and our budget constraints."
            )
        return (
            f"To help us reach closure, I'm proposing {counter_text} as our target. This represents our best position given the "
            "business case."
        )

    def _craft_value_proposition_simple(
        self, playbook_context: Dict[str, Any], round_no: int
    ) -> str:
        """Craft simple value proposition from playbook."""
        plays = playbook_context.get("plays", [])[:2]

        if not plays:
            return ""

        statements: List[str] = []
        for play in plays:
            if not isinstance(play, dict):
                continue

            description = play.get("play", "")
            lowered = description.lower()
            if "volume" in lowered or "tier" in lowered:
                statements.append(
                    "We'd like to explore volume commitments that could unlock better pricing for both sides."
                )
            elif "payment" in lowered and "early" in lowered:
                statements.append(
                    "We can offer accelerated payment terms if that helps with pricing flexibility."
                )
            elif "warranty" in lowered:
                statements.append(
                    "Extended warranty coverage would strengthen the business case on our side."
                )

        return " ".join(statements[:2]) if statements else ""

    def _craft_asks_simple(self, decision: Dict[str, Any]) -> str:
        """Craft simple asks section."""
        asks = decision.get("asks", [])
        if not asks:
            return ""

        asks_list = [f"• {ask}" for ask in asks[:4] if ask]
        if not asks_list:
            return ""

        return "To help structure the best outcome:\n" + "\n".join(asks_list)

    def _craft_closing_simple(self, round_no: int, strategy: str) -> str:
        """Craft simple closing."""
        if strategy == "accept":
            return (
                "If we can align on these final points, I'm ready to move forward. Looking forward to your thoughts."
            )

        if round_no >= 3:
            return (
                "I'm hopeful we can find common ground. Please let me know your thoughts, and I'm happy to discuss further if needed."
            )

        return (
            "I'd welcome your feedback on this proposal. Happy to discuss any questions you might have."
        )

    def _get_prompt_template(self, context: AgentContext) -> str:
        template: Optional[str] = None
        prompt_ids: List[int] = []

        prompts_payload = context.input_data.get("prompts")
        if isinstance(prompts_payload, list):
            for prompt in prompts_payload:
                if not isinstance(prompt, dict):
                    continue
                candidate = prompt.get("prompt_template") or prompt.get("template")
                if candidate:
                    template = str(candidate)
                    break
                pid = prompt.get("promptId") or prompt.get("prompt_id")
                try:
                    if pid is not None:
                        prompt_ids.append(int(pid))
                except (TypeError, ValueError):
                    continue

        if template:
            return template

        prompt_engine = getattr(self, "prompt_engine", None)
        if prompt_engine is not None:
            for pid in prompt_ids:
                prompt_entry = prompt_engine.get_prompt(pid)
                if prompt_entry and prompt_entry.get("template"):
                    return str(prompt_entry["template"])

            for prompt_entry in prompt_engine.prompts_for_agent(self.__class__.__name__):
                candidate = prompt_entry.get("template")
                if candidate:
                    return str(candidate)

        return DEFAULT_NEGOTIATION_MESSAGE_TEMPLATE

    def _build_prompt_context(
        self,
        *,
        context: AgentContext,
        header: str,
        lines: List[str],
        decision: Dict[str, Any],
        price: Optional[float],
        target_price: Optional[float],
        currency: Optional[str],
        round_no: int,
        supplier: Optional[str],
        supplier_snippets: List[str],
        supplier_message: Optional[str],
        playbook_context: Optional[Dict[str, Any]],
        signals: Optional[Dict[str, Any]],
        zopa: Optional[Dict[str, Any]],
        procurement_summary: Optional[Dict[str, Any]],
        rag_snippets: Optional[List[str]],
        rfq_reference: Optional[str] = None,
    ) -> Dict[str, str]:
        summary_lines = lines[1:]
        details = "\n".join(summary_lines)
        full_summary = "\n".join(lines)

        supplier_name = context.input_data.get("supplier_name")
        supplier_identifier = context.input_data.get("supplier_id") or supplier or ""

        current_offer_raw = context.input_data.get("current_offer")
        counter_price = decision.get("counter_price")

        current_offer_formatted = (
            self._format_currency(price, currency) if price is not None else ""
        )
        target_price_formatted = (
            self._format_currency(target_price, currency) if target_price is not None else ""
        )
        counter_price_formatted = (
            self._format_currency(counter_price, currency) if counter_price is not None else ""
        )

        supplier_snippet_text = "\n".join(
            f"- {snippet}" for snippet in supplier_snippets if isinstance(snippet, str) and snippet
        )

        procurement_lines = ""
        procurement_metrics = ""
        procurement_count = ""
        if procurement_summary:
            lines_payload = procurement_summary.get("summary_lines") or []
            if isinstance(lines_payload, list):
                unique_lines: List[str] = []
                for entry in lines_payload:
                    text = str(entry).strip()
                    if text and text not in unique_lines:
                        unique_lines.append(text)
                if unique_lines:
                    procurement_lines = "\n".join(f"- {text}" for text in unique_lines[:5])
            metrics_payload = procurement_summary.get("metrics")
            if metrics_payload:
                procurement_metrics = self._serialise_for_prompt(metrics_payload)
            record_count = procurement_summary.get("record_count")
            if record_count is not None:
                procurement_count = str(record_count)

        rag_text = ""
        if rag_snippets:
            rag_entries = [
                str(snippet).strip()
                for snippet in rag_snippets
                if isinstance(snippet, str) and snippet.strip()
            ]
            if rag_entries:
                rag_text = "\n".join(f"- {snippet}" for snippet in rag_entries)

        policy_payload = context.input_data.get("policies") or context.policy_context
        policy_text = self._serialise_for_prompt(policy_payload)
        performance_text = self._serialise_for_prompt(
            context.input_data.get("supplier_performance")
        )
        market_text = self._serialise_for_prompt(context.input_data.get("market_context"))

        context_sections: List[str] = []
        if procurement_lines:
            context_sections.append("\n\nContext – Procurement summary:\n" + procurement_lines)
        if procurement_metrics:
            context_sections.append("\n\nProcurement metrics:\n" + procurement_metrics)
        if rag_text:
            context_sections.append("\n\nRetrieved knowledge snippets:\n" + rag_text)
        if policy_text:
            context_sections.append("\n\nPolicy guidance:\n" + policy_text)
        if performance_text:
            context_sections.append("\n\nSupplier performance:\n" + performance_text)
        if market_text:
            context_sections.append("\n\nMarket context:\n" + market_text)

        decision_positions = self._serialise_for_prompt(decision.get("positions"))
        decision_flags = self._serialise_for_prompt(decision.get("flags"))
        decision_outliers = self._serialise_for_prompt(decision.get("outlier_alerts"))
        decision_validation = self._serialise_for_prompt(decision.get("validation_issues"))

        prompt_values: Dict[str, str] = {
            "header": header,
            "details": details,
            "summary_lines": details,
            "full_summary": full_summary,
            "context_sections": "".join(context_sections),
            "session_reference": str(rfq_reference or ""),
            "round_number": str(round_no),
            "strategy": str(decision.get("strategy") or ""),
            "supplier_id": str(supplier_identifier),
            "supplier_name": str(supplier_name or supplier_identifier or ""),
            "supplier_snippets": supplier_snippet_text,
            "supplier_message": str(supplier_message or ""),
            "current_offer": self._serialise_for_prompt(current_offer_raw),
            "current_offer_formatted": current_offer_formatted,
            "target_price": self._serialise_for_prompt(target_price),
            "target_price_formatted": target_price_formatted,
            "counter_price": self._serialise_for_prompt(counter_price),
            "counter_price_formatted": counter_price_formatted,
            "currency": str(currency or ""),
            "asks": self._serialise_for_prompt(decision.get("asks")),
            "lead_time_request": str(decision.get("lead_time_request") or ""),
            "decision_rationale": str(decision.get("rationale") or ""),
            "decision_plan_message": str(decision.get("plan_counter_message") or ""),
            "decision_positions": decision_positions,
            "decision_flags": decision_flags,
            "decision_outliers": decision_outliers,
            "decision_validation": decision_validation,
            "signals_summary": self._serialise_for_prompt(signals),
            "zopa_summary": self._serialise_for_prompt(zopa),
            "playbook_context": self._serialise_for_prompt(playbook_context),
            "policy_context": self._serialise_for_prompt(context.policy_context),
            "policy_guidance": policy_text,
            "supplier_performance": performance_text,
            "market_context": market_text,
            "task_profile": self._serialise_for_prompt(context.task_profile),
            "knowledge_base": self._serialise_for_prompt(context.knowledge_base),
            "workflow_id": str(context.workflow_id),
            "agent_id": str(context.agent_id),
            "agentic_plan": "\n".join(self.AGENTIC_PLAN_STEPS),
            "procurement_summary_lines": procurement_lines,
            "procurement_summary_metrics": procurement_metrics,
            "procurement_summary_record_count": procurement_count,
            "rag_snippets": rag_text,
            "previous_email_subject": str(
                context.input_data.get("previous_email_subject") or ""
            ),
            "thread_headers": self._serialise_for_prompt(
                context.input_data.get("thread_headers")
            ),
            "supplier_replies": self._serialise_for_prompt(
                context.input_data.get("supplier_replies")
                or context.input_data.get("supplier_responses")
            ),
            "supplier_reply_count": str(
                context.input_data.get("supplier_reply_count")
                or context.input_data.get("supplier_responses_count")
                or ""
            ),
            "positions": decision_positions,
        }

        return prompt_values

    def _apply_prompt_template(
        self, template: str, values: Dict[str, str], lines: List[str]
    ) -> str:
        safe_values: defaultdict[str, str] = defaultdict(str)
        for key, value in values.items():
            if value is None:
                continue
            safe_values[key] = value

        rendered: str = ""
        try:
            rendered = template.format_map(safe_values).strip()
        except Exception:
            logger.debug("Negotiation prompt template formatting failed", exc_info=True)

        if not rendered:
            rendered = "\n".join(lines)

        return rendered

    def _serialise_for_prompt(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            try:
                return str(value)
            except Exception:
                return ""

    def _retrieve_procurement_summary(
        self,
        *,
        supplier_id: Optional[str],
        supplier_name: Optional[str],
    ) -> Dict[str, Any]:
        engine = getattr(self.agent_nick, "query_engine", None)
        if engine is None or not hasattr(engine, "fetch_procurement_flow"):
            return {}

        try:
            df = engine.fetch_procurement_flow(
                supplier_ids=[supplier_id] if supplier_id else None,
                supplier_names=[supplier_name.lower()] if supplier_name else None,
            )
        except Exception:
            logger.debug(
                "NegotiationAgent: procurement flow retrieval failed", exc_info=True
            )
            return {}

        if df is None:
            return {}

        records: List[Dict[str, Any]] = []
        try:
            records = df.to_dict(orient="records")  # type: ignore[attr-defined]
        except Exception:
            if isinstance(df, list):
                records = [dict(row) for row in df if isinstance(row, dict)]

        if not records:
            return {}

        summary_lines: List[str] = []
        totals: List[float] = []
        currency_hint: Optional[str] = None

        for row in records[:10]:
            if not isinstance(row, dict):
                continue
            currency_hint = (
                row.get("currency")
                or row.get("default_currency")
                or row.get("invoice_currency")
                or currency_hint
            )
            amount = (
                row.get("total_amount")
                or row.get("total")
                or row.get("invoice_total_incl_tax")
                or row.get("order_total")
            )
            amount_value = self._coerce_float(amount)
            po_id = (
                row.get("po_id")
                or row.get("purchase_order_number")
                or row.get("purchase_order_id")
            )
            invoice_id = row.get("invoice_id") or row.get("invoice_number")
            order_date = (
                row.get("order_date")
                or row.get("requested_date")
                or row.get("invoice_date")
            )
            if amount_value is not None:
                totals.append(float(amount_value))
            reference = po_id or invoice_id
            if reference and amount_value is not None:
                amount_text = self._format_currency(amount_value, currency_hint)
                if order_date:
                    summary_lines.append(f"{reference} – {amount_text} on {order_date}")
                else:
                    summary_lines.append(f"{reference} – {amount_text}")

        metrics: Dict[str, Any] = {}
        if totals:
            total_spend = sum(totals)
            average_value = total_spend / len(totals)
            metrics["total_spend"] = total_spend
            metrics["average_transaction_value"] = average_value
            metrics["total_spend_formatted"] = self._format_currency(
                total_spend, currency_hint
            )
            metrics["average_transaction_value_formatted"] = self._format_currency(
                average_value, currency_hint
            )

        unique_lines: List[str] = []
        for entry in summary_lines:
            if entry and entry not in unique_lines:
                unique_lines.append(entry)

        return {
            "summary_lines": unique_lines[:5],
            "metrics": metrics,
            "record_count": len(records),
        }

    def _collect_vector_snippets(
        self,
        *,
        context: AgentContext,
        supplier_name: Optional[str],
        workflow_reference: Optional[str],
    ) -> List[str]:
        query_tokens: List[str] = []
        if supplier_name:
            query_tokens.append(str(supplier_name))
        category = (
            context.input_data.get("product_category")
            or context.input_data.get("category")
            or context.input_data.get("product_type")
        )
        if isinstance(category, str) and category.strip():
            query_tokens.append(category.strip())
        description = context.input_data.get("rfq_description") or context.input_data.get(
            "item_description"
        )
        if isinstance(description, str) and description.strip():
            query_tokens.append(description.strip())
        if workflow_reference:
            query_tokens.append(str(workflow_reference))

        query = " ".join(query_tokens[:3]).strip()
        if not query:
            return []

        results = self.vector_search(query, top_k=3)
        snippets: List[str] = []
        for result in results or []:
            payload = getattr(result, "payload", None)
            if isinstance(payload, dict):
                snippet = (
                    payload.get("text")
                    or payload.get("content")
                    or payload.get("summary")
                    or payload.get("body")
                )
                if snippet:
                    text = str(snippet).strip()
                    if text:
                        snippets.append(text)
                continue

            if isinstance(result, dict):
                payload_data = result.get("payload")
                if isinstance(payload_data, dict):
                    snippet = (
                        payload_data.get("text")
                        or payload_data.get("content")
                        or payload_data.get("summary")
                    )
                    if snippet:
                        text = str(snippet).strip()
                        if text:
                            snippets.append(text)
                            continue
                snippet = (
                    result.get("text")
                    or result.get("content")
                    or result.get("summary")
                    or result.get("body")
                )
                if snippet:
                    text = str(snippet).strip()
                    if text:
                        snippets.append(text)

        return snippets

    def _build_supplier_watch_fields(
        self,
        *,
        context: AgentContext,
        workflow_id: Optional[str],
        supplier: Optional[str],
        drafts: List[Dict[str, Any]],
        state: Dict[str, Any],
        session_reference: Optional[str] = None,
        rfq_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not workflow_id:
            return None

        candidate_drafts: List[Dict[str, Any]] = []
        observed_unique_ids: List[str] = []
        unique_id_set: Set[str] = set()

        def _record_unique_id(raw_value: Any) -> Optional[str]:
            text = self._coerce_text(raw_value)
            if text and text not in unique_id_set:
                unique_id_set.add(text)
                observed_unique_ids.append(text)
            return text

        for source in (drafts, context.input_data.get("drafts")):
            if not source:
                continue
            if isinstance(source, dict):
                source = [source]
            if not isinstance(source, list):
                continue
            for entry in source:
                if isinstance(entry, dict):
                    draft_copy = dict(entry)
                    metadata_source = draft_copy.get("metadata")
                    metadata = (
                        dict(metadata_source)
                        if isinstance(metadata_source, dict)
                        else {}
                    )
                    if rfq_id and not draft_copy.get("rfq_id"):
                        draft_copy["rfq_id"] = rfq_id
                    if rfq_id and not metadata.get("rfq_id"):
                        metadata["rfq_id"] = rfq_id
                    if metadata:
                        draft_copy["metadata"] = metadata
                    unique_value = _record_unique_id(draft_copy.get("unique_id"))
                    if not unique_value:
                        for key in (
                            "dispatch_run_id",
                            "run_id",
                            "email_action_id",
                            "action_id",
                            "draft_id",
                        ):
                            unique_value = _record_unique_id(
                                draft_copy.get(key) or metadata.get(key)
                            )
                            if unique_value:
                                break
                    if not unique_value and metadata:
                        for key in ("unique_id", "message_id", "id"):
                            unique_value = _record_unique_id(metadata.get(key))
                            if unique_value:
                                break
                    if not unique_value and session_reference:
                        unique_value = _record_unique_id(session_reference)
                    if unique_value:
                        draft_copy.setdefault("unique_id", unique_value)
                    candidate_drafts.append(draft_copy)

        filtered_drafts: List[Dict[str, Any]] = []
        dropped_due_to_missing_supplier: List[Dict[str, Any]] = []

        for entry in candidate_drafts:
            if not isinstance(entry, dict):
                continue

            supplier_hint: Optional[Any] = entry.get("supplier_id") or entry.get("supplier")
            if not supplier_hint and isinstance(entry.get("metadata"), dict):
                meta = cast(Dict[str, Any], entry["metadata"])
                supplier_hint = meta.get("supplier_id") or meta.get("supplier")

            supplier_text = self._coerce_text(supplier_hint)
            if supplier_text:
                filtered_drafts.append(entry)
                continue

            dropped_due_to_missing_supplier.append(entry)

        if filtered_drafts:
            candidate_drafts = filtered_drafts
        elif dropped_due_to_missing_supplier:
            logger.info(
                "Ignoring %s draft(s) without supplier identifiers while preparing watch list",
                len(dropped_due_to_missing_supplier),
            )
            candidate_drafts = []

        if not candidate_drafts:
            fallback_entry = {"workflow_id": workflow_id}
            if rfq_id:
                fallback_entry["rfq_id"] = rfq_id
            if supplier:
                fallback_entry["supplier_id"] = supplier
            if session_reference:
                fallback_entry["session_reference"] = session_reference
                fallback_entry.setdefault("unique_id", session_reference)
            candidate_drafts.append(fallback_entry)

        poll_interval = getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)

        workflow_hint = workflow_id or getattr(context, "workflow_id", None)
        if workflow_hint:
            for draft in candidate_drafts:
                if isinstance(draft, dict) and not draft.get("workflow_id"):
                    draft["workflow_id"] = workflow_hint
                if isinstance(draft, dict) and session_reference:
                    draft.setdefault("session_reference", session_reference)
                    draft.setdefault("unique_id", session_reference)
                if isinstance(draft, dict) and rfq_id and not draft.get("rfq_id"):
                    draft["rfq_id"] = rfq_id

        if candidate_drafts:
            unique_id_set = set()
            observed_unique_ids = []
            for draft in candidate_drafts:
                if not isinstance(draft, dict):
                    continue
                metadata = (
                    draft.get("metadata")
                    if isinstance(draft.get("metadata"), dict)
                    else {}
                )
                unique_value = _record_unique_id(draft.get("unique_id"))
                if not unique_value:
                    for key in (
                        "dispatch_run_id",
                        "run_id",
                        "email_action_id",
                        "action_id",
                        "draft_id",
                    ):
                        unique_value = _record_unique_id(
                            draft.get(key) or metadata.get(key)
                        )
                        if unique_value:
                            break
                if not unique_value and metadata:
                    for key in ("unique_id", "message_id", "id"):
                        unique_value = _record_unique_id(metadata.get(key))
                        if unique_value:
                            break
                if not unique_value and session_reference:
                    unique_value = _record_unique_id(session_reference)
                if unique_value:
                    draft.setdefault("unique_id", unique_value)

        expected_count = max(len(candidate_drafts), len(observed_unique_ids)) or 1

        watch_payload: Dict[str, Any] = {
            "await_response": True,
            "message": "",
            "body": "",
            "drafts": candidate_drafts,
            "supplier_id": supplier,
            "response_poll_interval": poll_interval,
            "workflow_id": workflow_hint,
            "expected_dispatch_count": expected_count,
            "expected_email_count": expected_count,
        }

        if rfq_id:
            watch_payload["rfq_id"] = rfq_id

        if session_reference:
            watch_payload.setdefault("session_reference", session_reference)
            watch_payload.setdefault("unique_id", session_reference)

        if observed_unique_ids:
            watch_payload["unique_ids"] = list(observed_unique_ids)
            watch_payload["expected_unique_ids"] = list(observed_unique_ids)

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

        if len(watch_payload["drafts"]) > 1 or expected_count > 1:
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

        expected_dispatch = self._positive_int(
            watch_payload.get("expected_dispatch_count"), fallback=0
        )
        expected_email_count = self._positive_int(
            watch_payload.get("expected_email_count"), fallback=0
        )
        unique_expectations = [
            self._coerce_text(value)
            for value in self._ensure_list(
                watch_payload.get("expected_unique_ids")
                or watch_payload.get("unique_ids")
            )
            if self._coerce_text(value)
        ]
        expected_total = max(
            expected_dispatch,
            expected_email_count,
            len(unique_expectations),
        )

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
        if expected_total and batch_limit < expected_total:
            batch_limit = expected_total

        draft_entries: List[Dict[str, Any]] = []
        for draft in watch_payload.get("drafts", []):
            if isinstance(draft, dict):
                draft_copy = dict(draft)
                draft_unique = self._coerce_text(draft_copy.get("unique_id"))
                if not draft_unique and unique_expectations:
                    for candidate in unique_expectations:
                        if not candidate:
                            continue
                        if any(
                            self._coerce_text(entry.get("unique_id")) == candidate
                            for entry in draft_entries
                        ):
                            continue
                        draft_unique = candidate
                        break
                if draft_unique:
                    draft_copy.setdefault("unique_id", draft_unique)
                draft_entries.append(draft_copy)

        workflow_hint_raw = watch_payload.get("workflow_id") or getattr(
            context, "workflow_id", None
        )

        if not draft_entries:
            fallback_entry = {}
            if workflow_hint_raw:
                fallback_entry["workflow_id"] = workflow_hint_raw
            supplier_hint = watch_payload.get("supplier_id") or watch_payload.get("supplier")
            if supplier_hint:
                fallback_entry["supplier_id"] = supplier_hint
            if fallback_entry:
                draft_entries.append(fallback_entry)

        if not draft_entries:
            logger.warning("No draft context available while awaiting supplier response")
            return None

        workflow_hint = self._coerce_text(workflow_hint_raw)

        if draft_entries:
            logger.info(
                "Awaiting supplier responses for workflow=%s with %s drafts",
                workflow_hint or workflow_hint_raw,
                len(draft_entries),
            )
            for idx, entry in enumerate(draft_entries[:5]):
                logger.info(
                    "  Draft %s: unique=%s supplier=%s workflow=%s",
                    idx,
                    entry.get("unique_id"),
                    entry.get("supplier_id"),
                    entry.get("workflow_id"),
                )

        if draft_entries and len(draft_entries) > 1:
            draft_workflows = {
                self._coerce_text(entry.get("workflow_id"))
                for entry in draft_entries
                if self._coerce_text(entry.get("workflow_id"))
            }
            if len(draft_workflows) > 1:
                logger.error(
                    "CRITICAL: Multiple workflows in draft batch! workflows=%s",
                    sorted(draft_workflows),
                )
            elif len(draft_workflows) == 1 and workflow_hint:
                draft_workflow = draft_workflows.pop()
                if draft_workflow != workflow_hint:
                    logger.error(
                        "Workflow mismatch: context has %s but drafts have %s",
                        workflow_hint,
                        draft_workflow,
                    )

        if workflow_hint and draft_entries:
            filtered_entries: List[Dict[str, Any]] = []
            dropped_entries: List[Dict[str, Any]] = []
            for entry in draft_entries:
                entry_workflow = self._coerce_text(entry.get("workflow_id"))
                if entry_workflow and entry_workflow != workflow_hint:
                    dropped_entries.append(entry)
                    continue
                filtered_entries.append(entry)

            if dropped_entries:
                logger.warning(
                    "Discarding %s drafts due to workflow mismatch with %s",
                    len(dropped_entries),
                    workflow_hint,
                )
                for idx, entry in enumerate(dropped_entries[:5]):
                    logger.warning(
                        "  Dropped draft %s: unique=%s supplier=%s workflow=%s",
                        idx,
                        entry.get("unique_id"),
                        entry.get("supplier_id"),
                        entry.get("workflow_id"),
                    )
            if filtered_entries:
                draft_entries = filtered_entries
            elif dropped_entries:
                adopted_workflows = {
                    self._coerce_text(entry.get("workflow_id"))
                    for entry in dropped_entries
                    if self._coerce_text(entry.get("workflow_id"))
                }
                if len(adopted_workflows) == 1:
                    adopted_workflow = adopted_workflows.pop()
                    logger.warning(
                        "Adopting draft workflow_id=%s in place of context workflow_id=%s",
                        adopted_workflow,
                        workflow_hint,
                    )
                    workflow_hint = adopted_workflow
                    draft_entries = dropped_entries
                else:
                    logger.error(
                        "All provided drafts conflicted with workflow=%s; aborting await",
                        workflow_hint,
                    )
                    return None

        await_all = bool(watch_payload.get("await_all_responses") and len(draft_entries) > 1)
        tracked_unique_ids = sorted(
            {
                self._coerce_text(entry.get("unique_id"))
                for entry in draft_entries
                if self._coerce_text(entry.get("unique_id"))
            }
        )
        if unique_expectations and not tracked_unique_ids:
            tracked_unique_ids = sorted({value for value in unique_expectations if value})
        if expected_total and expected_total < len(draft_entries):
            expected_total = len(draft_entries)
        elif not expected_total:
            expected_total = len(draft_entries)

        try:
            if await_all:
                results = supplier_agent.wait_for_multiple_responses(
                    draft_entries,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    enable_negotiation=False,
                )
                valid_results = [res for res in results if isinstance(res, dict)]
                if expected_total and len(valid_results) < expected_total:
                    logger.warning(
                        "Awaited %s supplier responses but only received %s (workflow_id=%s unique_ids=%s)",
                        expected_total,
                        len(valid_results),
                        workflow_hint,
                        tracked_unique_ids or unique_expectations or None,
                    )
                    return None
                if unique_expectations:
                    observed = {
                        self._coerce_text(res.get("unique_id"))
                        for res in valid_results
                        if self._coerce_text(res.get("unique_id"))
                    }
                    missing = [
                        value
                        for value in unique_expectations
                        if value and value not in observed
                    ]
                    if missing:
                        logger.warning(
                            "Missing supplier responses for unique_ids=%s (workflow_id=%s)",
                            missing,
                            workflow_hint,
                        )
                        return None
                return results

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
            workflow_hint = workflow_hint or target.get("workflow_id")
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

            if not workflow_hint:
                logger.warning(
                    "Cannot await supplier response without workflow_id for supplier=%s",
                    target.get("supplier_id"),
                )
                return None

            return [
                supplier_agent.wait_for_response(
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    supplier_id=target.get("supplier_id"),
                    subject_hint=target.get("subject"),
                    from_address=recipient_hint,
                    enable_negotiation=False,
                    draft_action_id=draft_action_id,
                    workflow_id=workflow_hint,
                    dispatch_run_id=dispatch_run_id,
                    unique_id=target.get("unique_id"),
                )
            ]
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed while waiting for supplier responses (workflow_id=%s, supplier=%s)",
                workflow_hint,
                watch_payload.get("supplier_id"),
            )
            return None

    def _get_supplier_agent(self) -> "SupplierInteractionAgent":
        if self._supplier_agent is None:
            with self._supplier_agent_lock:
                if self._supplier_agent is None:
                    from agents.supplier_interaction_agent import SupplierInteractionAgent

                    self._supplier_agent = SupplierInteractionAgent(self.agent_nick)
        return self._supplier_agent

    def _build_stop_message(self, status: str, reason: str, round_no: int) -> str:
        status_text = status.capitalize()
        reason_text = reason or "No further action required."
        return f"Negotiation {status_text} after round {round_no}: {reason_text}"

    def _store_session(
        self, session_id: str, supplier: str, round_no: int, counter_price: Optional[float]
    ) -> None:
        """Persist negotiation round details."""
        identifier = self._coerce_text(session_id)
        supplier_id = self._coerce_text(supplier)
        if not identifier or not supplier_id:
            return
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    column = self._get_identifier_column("negotiation_sessions")
                    cur.execute(
                        f"""
                        INSERT INTO proc.negotiation_sessions
                            ({column}, supplier_id, round, counter_offer, created_on)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (identifier, supplier_id, round_no, counter_price),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store negotiation session")

    def _record_learning_snapshot(
        self,
        context: AgentContext,
        session_id: Optional[str] = None,
        supplier: Optional[str] = None,
        decision: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        awaiting_response: bool = False,
        supplier_reply_registered: bool = False,
        rfq_id: Optional[str] = None,
    ) -> None:
        _ = self._coerce_text(rfq_id) or self._coerce_text(session_id)
        logger.debug(
            "Negotiation learning capture skipped (session_id=%s, supplier=%s)",
            session_id,
            supplier,
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
        session_reference: Optional[str],
        price: Optional[float],
        target_price: Optional[float],
        decision: Dict[str, Any],
    ) -> str:
        base = (
            f"Strategy={decision.get('strategy')} counter={decision.get('counter_price')}"
            f" target={target_price} current={price} supplier={supplier} reference={session_reference}."
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
    def _normalise_negotiation_inputs(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        normalised: Dict[str, Any] = {}
        issues: List[str] = []

        price_sources = [
            payload.get("current_offer"),
            payload.get("price"),
            payload.get("supplier_offer"),
        ]
        target_sources = [payload.get("target_price"), payload.get("target")]
        walkaway_sources = [payload.get("walkaway_price"), payload.get("walkaway"), payload.get("no_deal_price")]
        previous_offer_sources = [
            payload.get("previous_offer"),
            payload.get("supplier_previous_offer"),
            payload.get("last_supplier_offer"),
            payload.get("offer_prev"),
        ]

        def _pick_first(values: List[Any], parser) -> Optional[float]:
            for candidate in values:
                parsed = parser(candidate)
                if parsed is not None:
                    return parsed
            return None

        normalised["current_offer"] = _pick_first(price_sources, self._parse_money)
        normalised["target_price"] = _pick_first(target_sources, self._parse_money)
        normalised["walkaway_price"] = _pick_first(walkaway_sources, self._parse_money)
        normalised["previous_offer"] = _pick_first(previous_offer_sources, self._parse_money)

        currency = (
            payload.get("currency")
            or payload.get("current_offer_currency")
            or payload.get("price_currency")
        )
        normalised["currency"] = self._normalise_currency(currency)

        lead_sources = [
            payload.get("lead_time_weeks"),
            payload.get("lead_time"),
            payload.get("lead_time_days"),
        ]
        lead_weeks = None
        for candidate in lead_sources:
            lead_weeks = self._parse_lead_weeks(candidate)
            if lead_weeks is not None:
                break
        normalised["lead_time_weeks"] = lead_weeks

        volume_sources = [
            payload.get("volume"),
            payload.get("volume_commitment"),
            payload.get("order_volume"),
            payload.get("order_quantity"),
            payload.get("quantity"),
            payload.get("units"),
        ]
        normalised["volume_units"] = _pick_first(volume_sources, self._parse_quantity)

        term_sources = [
            payload.get("term_days"),
            payload.get("payment_terms"),
            payload.get("payment_term"),
            payload.get("terms"),
        ]
        normalised["term_days"] = _pick_first(term_sources, self._parse_term_days)

        valid_until_sources = [
            payload.get("valid_until"),
            payload.get("offer_valid_until"),
            payload.get("validity"),
            payload.get("expiration_date"),
            payload.get("expiry_date"),
        ]
        valid_until = None
        for candidate in valid_until_sources:
            valid_until = self._parse_date(candidate)
            if valid_until is not None:
                break
        normalised["valid_until"] = valid_until

        reference_prices = self._extract_reference_prices(payload)
        normalised.update(reference_prices)

        essential_labels = {
            "current_offer": "Supplier offer",
            "target_price": "Target price",
        }
        for field, label in essential_labels.items():
            if normalised.get(field) is None:
                issues.append(f"{label} missing or invalid")

        return normalised, issues

    def _parse_money(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            cleaned = re.sub(r"[\s,]", "", text)
            match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    def _parse_quantity(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    def _parse_term_days(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidate = int(round(float(value)))
            return candidate if candidate > 0 else None
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            if not numbers:
                return None
            try:
                numeric = float(numbers[0])
            except ValueError:
                return None
            if "week" in text and numeric > 0:
                return int(round(numeric * 7))
            return int(round(numeric)) if numeric > 0 else None
        return None

    def _parse_date(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for fmt in (
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%d %b %Y",
                "%b %d, %Y",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    return dt.replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    continue
            try:
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                return None
        return None

    def _extract_reference_prices(self, payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
        candidates: List[float] = []

        def _ingest(value: Any) -> None:
            parsed = self._parse_money(value)
            if parsed is not None:
                candidates.append(parsed)

        for key in ("benchmarks", "market_context", "market_prices", "history"):
            source = payload.get(key)
            if isinstance(source, dict):
                for sub_key, sub_value in source.items():
                    lowered = str(sub_key).lower()
                    if any(
                        token in lowered
                        for token in ("price", "p10", "p25", "low", "min", "floor")
                    ):
                        _ingest(sub_value)
            elif isinstance(source, list):
                for item in source:
                    if isinstance(item, dict):
                        for sub_value in item.values():
                            _ingest(sub_value)
                    else:
                        _ingest(item)

        lowest_market = min(candidates) if candidates else None
        return {"market_floor_price": lowest_market}

    def _build_positions(
        self,
        *,
        supplier_offer: Optional[float],
        target_price: Optional[float],
        walkaway_price: Optional[float],
        previous_counter: Optional[float],
        previous_positions: Optional[Dict[str, Any]],
        round_no: int,
    ) -> NegotiationPositions:
        history: List[Dict[str, Any]] = []
        if isinstance(previous_positions, dict):
            stored_history = previous_positions.get("history")
            if isinstance(stored_history, list):
                history = [
                    entry
                    for entry in stored_history
                    if isinstance(entry, dict) and entry.get("round") is not None
                ]
        desired = self._coerce_float(target_price)
        if desired is None and isinstance(previous_positions, dict):
            desired = self._coerce_float(previous_positions.get("desired"))
        no_deal = self._coerce_float(walkaway_price)
        if no_deal is None and isinstance(previous_positions, dict):
            no_deal = self._coerce_float(previous_positions.get("no_deal"))

        candidate_starts: List[Optional[float]] = []
        candidate_starts.append(self._coerce_float(previous_counter))
        if isinstance(previous_positions, dict):
            candidate_starts.append(self._coerce_float(previous_positions.get("last_counter")))
            candidate_starts.append(self._coerce_float(previous_positions.get("start")))
        candidate_starts.append(self._coerce_float(supplier_offer))

        start_value: Optional[float] = None
        for candidate in candidate_starts:
            if candidate is not None:
                start_value = candidate
                break

        positions = NegotiationPositions(
            start=start_value,
            desired=desired,
            no_deal=no_deal,
            supplier_offer=self._coerce_float(supplier_offer),
            history=history,
        )

        if positions.supplier_offer is not None:
            positions.history.append(
                {
                    "round": round_no,
                    "type": "supplier_offer",
                    "value": positions.supplier_offer,
                }
            )

        return positions

    def _respect_positions(
        self, counter: Optional[float], positions: NegotiationPositions
    ) -> Optional[float]:
        if counter is None:
            return None
        try:
            candidate = float(counter)
        except (TypeError, ValueError):
            return None

        if positions.desired is not None:
            try:
                candidate = max(candidate, float(positions.desired))
            except (TypeError, ValueError):
                pass
        if positions.no_deal is not None:
            try:
                candidate = min(candidate, float(positions.no_deal))
            except (TypeError, ValueError):
                pass
        if positions.start is not None:
            try:
                candidate = min(candidate, float(positions.start))
            except (TypeError, ValueError):
                pass

        return round(candidate, 2)

    def _detect_outliers(
        self,
        *,
        supplier_offer: Optional[float],
        target_price: Optional[float],
        walkaway_price: Optional[float],
        market_floor: Optional[float],
        volume_units: Optional[float],
        term_days: Optional[int],
    ) -> Dict[str, Any]:
        alerts: List[str] = []
        requires_review = False
        human_override = False
        review_recommendation: Optional[str] = None
        rationale_notes: List[str] = []

        def _percentage_gap(reference: Optional[float], offer: Optional[float]) -> Optional[float]:
            try:
                if reference is None or offer is None or reference <= 0:
                    return None
                return (reference - offer) / reference
            except (TypeError, ValueError):
                return None

        market_gap = _percentage_gap(market_floor, supplier_offer)
        walkaway_gap = _percentage_gap(walkaway_price, supplier_offer)

        if market_gap is not None and market_gap >= MARKET_REVIEW_THRESHOLD:
            requires_review = True
            review_recommendation = "query_for_human_review"
            alerts.append(
                f"Supplier offer is {market_gap * 100:.1f}% below market reference {market_floor:.2f}."
            )
            rationale_notes.append("Requested price is materially below market benchmarks; seek justification.")
            if market_gap >= MARKET_ESCALATION_THRESHOLD:
                human_override = True
                rationale_notes.append(
                    "Supplier offer breaches escalation threshold relative to market floor."
                )

        if walkaway_gap is not None and walkaway_gap >= MARKET_REVIEW_THRESHOLD:
            requires_review = True
            review_recommendation = review_recommendation or "query_for_human_review"
            alerts.append(
                f"Supplier offer is {walkaway_gap * 100:.1f}% below walk-away price {walkaway_price:.2f}."
            )
            rationale_notes.append(
                "Requested price undercuts internal walk-away guardrail; confirm intent before proceeding."
            )
            if walkaway_gap >= MARKET_ESCALATION_THRESHOLD:
                human_override = True
                rationale_notes.append(
                    "The requested price is more than 20% below our walk-away price; escalation required."
                )

        if volume_units is not None and volume_units > MAX_VOLUME_LIMIT:
            requires_review = True
            review_recommendation = review_recommendation or "query_for_human_review"
            alerts.append(
                f"Requested volume {volume_units:.0f} exceeds configured limit {MAX_VOLUME_LIMIT:.0f}."
            )
            rationale_notes.append("Request supplier rationale for above-capacity volume.")
            if volume_units > MAX_VOLUME_LIMIT * 1.5:
                human_override = True
                rationale_notes.append("Volume exceeds escalation ceiling; seek human approval.")

        if term_days is not None and term_days > MAX_TERM_DAYS:
            requires_review = True
            review_recommendation = review_recommendation or "query_for_human_review"
            alerts.append(
                f"Requested payment term {term_days} days exceeds policy limit {MAX_TERM_DAYS} days."
            )
            rationale_notes.append("Payment term exceeds policy; confirm via human review.")
            if term_days > MAX_TERM_DAYS * 2:
                human_override = True
                rationale_notes.append("Payment term far exceeds tolerance; human intervention required.")

        message = None
        if human_override:
            for note in rationale_notes[::-1]:
                if "escalation required" in note.lower():
                    message = note
                    break
            message = message or "Escalation required before proceeding."
        elif requires_review and rationale_notes:
            message = rationale_notes[-1]

        return {
            "alerts": alerts,
            "requires_review": requires_review,
            "human_override": human_override,
            "recommendation": review_recommendation,
            "message": message,
        }

    def _compose_rationale(
        self,
        *,
        positions: NegotiationPositions,
        decision: Dict[str, Any],
        currency: Optional[str],
        lead_weeks: Optional[float],
        volume_units: Optional[float],
        term_days: Optional[int],
        outlier_message: Optional[str],
        validation_issues: List[str],
    ) -> str:
        counter_price = decision.get("counter_price")
        parts: List[str] = []

        def _format(amount: Optional[float]) -> Optional[str]:
            if amount is None:
                return None
            text = self._format_currency(amount, currency)
            return text or f"{amount:0.2f}"

        start_text = _format(positions.start)
        target_text = _format(positions.desired)
        counter_text = _format(counter_price)

        if start_text and target_text and counter_text:
            rationale = (
                f"Because the starting price is {start_text} and the target is {target_text}, "
                f"we counter at {counter_text}"
            )
        elif counter_text:
            rationale = f"We counter at {counter_text} based on the available pricing guardrails"
        else:
            rationale = "Pricing inputs incomplete; request clarification before committing to a counter"

        additions: List[str] = []
        if term_days:
            additions.append(f"{int(term_days)}-day term")
        if volume_units:
            additions.append(f"{int(round(volume_units))}-unit volume")
        if lead_weeks:
            additions.append(f"{lead_weeks:.1f}-week lead time request")
        if additions and counter_text:
            rationale += " with " + " and ".join(additions)

        parts.append(rationale + ".")

        if validation_issues:
            issue_text = "; ".join(sorted(set(validation_issues)))
            parts.append(f"Data validation flagged: {issue_text}.")

        if outlier_message:
            parts.append(outlier_message)

        return " ".join(part for part in parts if part)

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

    def _validate_buyer_max(self, value: Any) -> Optional[float]:
        parsed = self._coerce_float(value)
        if parsed is None:
            return None
        if not math.isfinite(parsed):
            return None
        if parsed <= 0:
            return None
        return parsed

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
        decision_payload = dict(decision) if isinstance(decision, dict) else {}
        workflow_id = getattr(context, "workflow_id", None)
        if workflow_id:
            payload.setdefault("workflow_id", workflow_id)
        else:
            logger.warning("NegotiationAgent building email payload without workflow_id on context")

        logger.info(
            "Building email drafting payload for supplier",
            extra={
                "workflow_id": workflow_id,
                "supplier_id": payload.get("supplier_id") or payload.get("supplier"),
            },
        )

        thread_headers = (
            context.input_data.get("thread_headers")
            if isinstance(context.input_data, dict)
            else None
        )
        if not thread_headers:
            thread_headers = payload.get("thread_headers")
        if not thread_headers:
            cached_thread_headers = state.get("last_thread_headers")
            if isinstance(cached_thread_headers, dict):
                thread_headers = dict(cached_thread_headers)
            else:
                thread_headers = cached_thread_headers
        if thread_headers:
            payload.setdefault("thread_headers", thread_headers)
            decision_payload.setdefault("thread", thread_headers)
        payload.setdefault("decision", decision_payload)
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
                with self._email_agent_lock:
                    if self._email_agent is None:
                        self._email_agent = EmailDraftingAgent(self.agent_nick)
            email_agent = self._email_agent
            email_context = parent_context.create_child_context(
                "EmailDraftingAgent", payload
            )
            if email_context.workflow_id != parent_context.workflow_id:
                raise RuntimeError(
                    "Email drafting child context did not inherit workflow_id"
                )
            logger.info(
                "Invoking EmailDraftingAgent",
                extra={
                    "workflow_id": email_context.workflow_id,
                    "parent_agent": parent_context.agent_id,
                },
            )
            return email_agent.execute(email_context) if email_agent else None
        except Exception:
            logger.exception("Failed to invoke EmailDraftingAgent for negotiation counter")
            return None

