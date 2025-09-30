from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import requests
from agents.base_agent import AgentContext, AgentOutput, AgentStatus
# -----------------------
# Config & small utilities
# -----------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_COMPOSE = os.getenv("EMAIL_DRAFT_MODEL", "llama3.2")
MODEL_POLISH = os.getenv("EMAIL_POLISH_MODEL", "gemma3")  # optional
ENABLE_POLISH = os.getenv("EMAIL_POLISH_ENABLED", "false").strip().lower() == "true"


def _post_ollama(path: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(f"{OLLAMA_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _chat(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.9,
    num_ctx: int = 4096,
    max_tokens: Optional[int] = None,
) -> str:
    options: Dict[str, Any] = {"temperature": temperature, "top_p": top_p, "num_ctx": num_ctx}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
        "stream": False,
    }
    response = _post_ollama("/api/chat", payload)
    return response.get("message", {}).get("content", "").strip()


def _to_html(text: str) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    lines = [line.rstrip() for line in escaped.splitlines()]
    html_lines: List[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "•")):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped.lstrip('-• ').strip()}</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        if stripped:
            html_lines.append(f"<p>{line}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines) if html_lines else f"<p>{escaped}</p>"


def _first_line_subject_fallback(body: str, default: str) -> str:
    for line in body.splitlines():
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            if subject:
                return subject
    return default


def _trim_subject_prefixes(value: str) -> str:
    return re.sub(r"^\s*(re|fw|fwd)\s*:\s*", "", value, flags=re.I)


# -----------------------
# Input contracts
# -----------------------


@dataclass
class ThreadHeaders:
    message_id: Optional[str] = None
    references: Optional[List[str]] = None


@dataclass
class DecisionContext:
    rfq_id: str
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    to: Optional[str] = None
    cc: Optional[List[str]] = None
    current_offer: Optional[float] = None
    currency: Optional[str] = None
    lead_time_weeks: Optional[float] = None
    target_price: Optional[float] = None
    round: Optional[int] = 1
    strategy: str = "clarify"
    counter_price: Optional[float] = None
    asks: Optional[List[str]] = None
    lead_time_request: Optional[str] = None
    rationale: Optional[str] = None
    thread: Optional[ThreadHeaders] = None

    def to_public_json(self) -> dict:
        return {
            "rfq_id": self.rfq_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "current_offer": self.current_offer,
            "currency": self.currency,
            "lead_time_weeks": self.lead_time_weeks,
            "target_price": self.target_price,
            "round": self.round,
            "strategy": self.strategy,
            "counter_price": self.counter_price,
            "asks": self.asks or [],
            "lead_time_request": self.lead_time_request,
            "rationale": self.rationale,
        }


# -----------------------
# Agent Output contract
# -----------------------


def _build_output(decision: DecisionContext, subject: str, body_text: str) -> dict:
    body_text = body_text.strip()
    subject = subject.strip()
    headers = {"X-Procwise-RFQ-ID": decision.rfq_id}
    if decision.thread and decision.thread.message_id:
        headers["In-Reply-To"] = decision.thread.message_id
    if decision.thread and decision.thread.references:
        headers["References"] = " ".join(decision.thread.references)

    return {
        "rfq_id": decision.rfq_id,
        "to": decision.to,
        "cc": decision.cc or [],
        "subject": subject,
        "text": body_text,
        "html": _to_html(body_text),
        "headers": headers,
        "metadata": {
            "supplier_id": decision.supplier_id,
            "round": decision.round,
            "strategy": decision.strategy,
            "counter_price": decision.counter_price,
        },
    }


# -----------------------
# System prompts
# -----------------------

SYSTEM_COMPOSE = (
    "You are a procurement negotiator. Use British English. Be concise and specific.\n"
    "Rules:\n"
    "- Keep the email under 140 words.\n"
    "- Use bullet points for multiple asks.\n"
    "- Do NOT invent numbers not provided in the JSON context.\n"
    "- Always reference the RFQ id.\n"
    "- Be polite and professional.\n"
)

USER_COMPOSE_TEMPLATE = """Context (JSON):
{ctx}

Write a short email to the supplier with:
- a clear subject line,
- brief opening referencing the RFQ,
- counter price (if present) with currency,
- requested lead time (if present),
- 1–2 bullet-pointed asks from the list,
- a polite close asking for confirmation within 3 days.
"""

SYSTEM_POLISH = (
    "Polish the following email for warmth and clarity without changing any numbers or commitments.\n"
    "Keep it under 140 words. Preserve bullet points. Use British English."
)

USER_POLISH_TEMPLATE = """Original:
<<<
{email}
<<<
"""

SYSTEM_PROMPT_MODE = (
    "You are a procurement email drafter. Use British English. Be concise and specific.\n"
    "Keep the output under 140 words. Use bullet points for lists. Reference the RFQ id if provided."
)

USER_PROMPT_MODE_TEMPLATE = """Draft a supplier email using the instruction below.

Instruction:
<<<
{prompt}
>>>

If an RFQ id appears, include it in the subject and opening line.
"""


# -----------------------
# The Agent
# -----------------------


class EmailDraftingAgent:
    """Email drafting helper backed by an Ollama chat model."""

    def __init__(
        self,
        compose_model: Optional[str] = MODEL_COMPOSE,
        polish_model: Optional[str] = MODEL_POLISH,
        *_,
        **kwargs,
    ) -> None:
        # Backwards compatibility for legacy initialisation ``EmailDraftingAgent(agent_nick)``.
        if compose_model is not None and not isinstance(compose_model, str):
            kwargs.setdefault("agent_nick", compose_model)
            compose_model = MODEL_COMPOSE
        if polish_model is not None and not isinstance(polish_model, str):
            kwargs.setdefault("agent_nick", polish_model)
            polish_model = MODEL_POLISH

        self.compose_model = compose_model or MODEL_COMPOSE
        self.polish_model = (polish_model or MODEL_POLISH) if ENABLE_POLISH else None
        self.agent_nick = kwargs.get("agent_nick")

    # Connected mode (NegotiationAgent -> EmailDraftingAgent)
    def from_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        context = self._coerce_decision(decision)
        ctx_json = json.dumps(context.to_public_json(), ensure_ascii=False)
        body = _chat(
            self.compose_model,
            SYSTEM_COMPOSE,
            USER_COMPOSE_TEMPLATE.format(ctx=ctx_json),
            temperature=0.3,
            top_p=0.9,
            num_ctx=4096,
        )

        if self.polish_model:
            body = _chat(
                self.polish_model,
                SYSTEM_POLISH,
                USER_POLISH_TEMPLATE.format(email=body),
                temperature=0.2,
                top_p=0.9,
                num_ctx=2048,
                max_tokens=260,
            )

        default_subject = _trim_subject_prefixes(
            f"{context.rfq_id} – Counter Offer & Next Steps"
        )
        subject = _first_line_subject_fallback(body, default_subject)
        body = re.sub(r"(?i)^subject:.*\n", "", body).strip()

        return _build_output(context, subject, body)

    # Prompt mode (free form)
    def from_prompt(self, prompt: str, rfq_id: Optional[str] = None) -> Dict[str, Any]:
        body = _chat(
            self.compose_model,
            SYSTEM_PROMPT_MODE,
            USER_PROMPT_MODE_TEMPLATE.format(prompt=prompt),
            temperature=0.3,
            top_p=0.9,
            num_ctx=4096,
        )

        if self.polish_model:
            body = _chat(
                self.polish_model,
                SYSTEM_POLISH,
                USER_POLISH_TEMPLATE.format(email=body),
                temperature=0.2,
                top_p=0.9,
                num_ctx=2048,
                max_tokens=260,
            )

        if not rfq_id:
            match = re.search(r"(RFQ[-\s:_]*[A-Za-z0-9\-]+)", body, flags=re.I)
            rfq_id = match.group(1).replace(" ", "") if match else "RFQ"

        subject = _first_line_subject_fallback(
            body, _trim_subject_prefixes(f"{rfq_id} – Follow-up")
        )
        body = re.sub(r"(?i)^subject:.*\n", "", body).strip()

        context = DecisionContext(rfq_id=rfq_id)
        return _build_output(context, subject, body)

    # Legacy orchestrator compatibility -------------------------------------------------
    def execute(self, context: AgentContext) -> AgentOutput:
        """Legacy orchestration shim.

        The refreshed drafting helper exposes :meth:`from_decision` and
        :meth:`from_prompt` for direct usage.  Legacy flows, however, still call
        :meth:`execute` with an :class:`AgentContext`.  This adapter keeps those
        flows functional by extracting drafting inputs from ``context`` and
        delegating to the lightweight helpers.
        """

        input_data: Dict[str, Any] = getattr(context, "input_data", {}) or {}

        drafts: List[Dict[str, Any]] = []
        errors: List[str] = []

        for decision in self._extract_decisions(input_data):
            try:
                drafts.append(self.from_decision(decision))
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"decision draft failed: {exc}")

        if not drafts:
            for prompt_text, rfq_id in self._extract_prompts(input_data):
                try:
                    drafts.append(self.from_prompt(prompt_text, rfq_id=rfq_id))
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"prompt draft failed: {exc}")

        status = AgentStatus.SUCCESS if drafts or not errors else AgentStatus.FAILED
        payload: Dict[str, Any] = {"drafts": drafts}
        if errors:
            payload["errors"] = errors

        pass_fields = dict(input_data)
        pass_fields["drafts"] = drafts

        return AgentOutput(status=status, data=payload, pass_fields=pass_fields)

    # -----------------------
    # helpers
    # -----------------------
    def _coerce_decision(self, payload: Dict[str, Any]) -> DecisionContext:
        thread_info = payload.get("thread") or {}
        thread = ThreadHeaders(
            message_id=thread_info.get("message_id"),
            references=thread_info.get("references") or None,
        )
        asks = payload.get("asks") or []

        return DecisionContext(
            rfq_id=str(payload.get("rfq_id")),
            supplier_id=payload.get("supplier_id"),
            supplier_name=payload.get("supplier_name"),
            to=payload.get("to"),
            cc=payload.get("cc") or [],
            current_offer=_num_or_none(payload.get("current_offer")),
            currency=payload.get("currency"),
            lead_time_weeks=_num_or_none(payload.get("lead_time_weeks")),
            target_price=_num_or_none(payload.get("target_price")),
            round=int(payload.get("round") or 1),
            strategy=str(payload.get("strategy") or "clarify"),
            counter_price=_num_or_none(payload.get("counter_price")),
            asks=[str(item) for item in asks],
            lead_time_request=payload.get("lead_time_request"),
            rationale=payload.get("rationale"),
            thread=thread,
        )

    # ------------------------------------------------------------------
    # Legacy context helpers
    # ------------------------------------------------------------------

    def _extract_decisions(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalise decision-like payloads for :meth:`from_decision`."""

        decisions: List[Dict[str, Any]] = []

        def _append(candidate: Optional[Dict[str, Any]]) -> None:
            if not isinstance(candidate, dict):
                return
            if not candidate.get("rfq_id"):
                return
            decisions.append(self._normalise_decision_payload(candidate))

        _append(payload.get("decision"))

        decision_list = payload.get("decisions")
        if isinstance(decision_list, Sequence) and not isinstance(decision_list, (str, bytes)):
            for item in decision_list:
                _append(item if isinstance(item, dict) else None)

        ranking = payload.get("ranking")
        if isinstance(ranking, Sequence) and not isinstance(ranking, (str, bytes)):
            for entry in ranking:
                if isinstance(entry, dict):
                    _append(entry.get("decision"))

        # Fallback: treat the full payload as a decision when it resembles the
        # negotiation agent's pass-through structure.
        if not decisions and payload.get("rfq_id"):
            _append(payload)

        return decisions

    def _extract_prompts(self, payload: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
        """Normalise prompt style drafting requests."""

        prompts: List[Tuple[str, Optional[str]]] = []

        direct_prompt = payload.get("prompt") or payload.get("email_prompt")
        if isinstance(direct_prompt, str) and direct_prompt.strip():
            prompts.append((direct_prompt.strip(), payload.get("rfq_id")))

        prompt_entries = payload.get("prompts")
        if isinstance(prompt_entries, Sequence) and not isinstance(prompt_entries, (str, bytes)):
            for entry in prompt_entries:
                if isinstance(entry, dict):
                    text = entry.get("prompt") or entry.get("template")
                    if isinstance(text, str) and text.strip():
                        prompts.append((text.strip(), entry.get("rfq_id") or payload.get("rfq_id")))
                elif isinstance(entry, str) and entry.strip():
                    prompts.append((entry.strip(), payload.get("rfq_id")))

        return prompts

    def _normalise_decision_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract recognised decision fields with sensible fallbacks."""

        decision: Dict[str, Any] = {"rfq_id": str(payload.get("rfq_id"))}

        def _first_non_empty(keys: Iterable[str]) -> Optional[Any]:
            for key in keys:
                if key not in payload:
                    continue
                value = payload.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    candidate = value.strip()
                    if candidate:
                        return candidate
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    sequence = [item for item in value if item]
                    if sequence:
                        return sequence
                else:
                    return value
            return None

        recipients = payload.get("recipients")
        if isinstance(recipients, Sequence) and not isinstance(recipients, (str, bytes)):
            recipients_list = [str(item).strip() for item in recipients if str(item).strip()]
        else:
            recipients_list = []
            if isinstance(recipients, str) and recipients.strip():
                recipients_list = [recipients.strip()]

        to_address: Optional[str] = _first_non_empty(
            ["to", "recipient", "recipient_email", "supplier_email"]
        )
        if not to_address and recipients_list:
            to_address = recipients_list[0]
            recipients_list = recipients_list[1:]

        if to_address:
            decision["to"] = to_address
        if recipients_list:
            decision["cc"] = recipients_list

        cc_value = payload.get("cc")
        if isinstance(cc_value, Sequence) and not isinstance(cc_value, (str, bytes)):
            decision["cc"] = [str(item).strip() for item in cc_value if str(item).strip()]
        elif isinstance(cc_value, str) and cc_value.strip():
            decision["cc"] = [chunk.strip() for chunk in cc_value.split(",") if chunk.strip()]

        for key in (
            "supplier_id",
            "supplier_name",
            "currency",
            "lead_time_weeks",
            "target_price",
            "round",
            "strategy",
            "counter_price",
            "lead_time_request",
            "rationale",
        ):
            if key in payload and payload.get(key) is not None:
                decision[key] = payload.get(key)

        if "current_offer" in payload and payload.get("current_offer") is not None:
            decision["current_offer"] = payload.get("current_offer")

        if "counter_price" not in decision:
            counter_opts = payload.get("counter_proposals")
            if isinstance(counter_opts, Sequence) and not isinstance(counter_opts, (str, bytes)):
                for option in counter_opts:
                    if isinstance(option, dict) and option.get("price") is not None:
                        decision["counter_price"] = option.get("price")
                        break

        asks_value = payload.get("asks")
        if isinstance(asks_value, Sequence) and not isinstance(asks_value, (str, bytes)):
            decision["asks"] = [str(item) for item in asks_value if str(item).strip()]
        elif isinstance(asks_value, str) and asks_value.strip():
            decision["asks"] = [chunk.strip() for chunk in asks_value.split("\n") if chunk.strip()]

        if "rationale" not in decision:
            rationale = _first_non_empty(["decision_log", "notes"])
            if rationale:
                decision["rationale"] = rationale

        thread_info = payload.get("thread")
        if not isinstance(thread_info, dict):
            thread_info = {
                "message_id": _first_non_empty(["in_reply_to", "message_id"]),
                "references": payload.get("references"),
            }
        if isinstance(thread_info, dict):
            decision["thread"] = {
                key: value
                for key, value in thread_info.items()
                if value
            }

        return decision


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None
