from __future__ import annotations

import json
import os
import re
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
            f"RFQ {context.rfq_id} – Counter Offer & Next Steps"
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
            body, _trim_subject_prefixes(f"RFQ {rfq_id} – Follow-up")
        )
        body = re.sub(r"(?i)^subject:.*\n", "", body).strip()

        context = DecisionContext(rfq_id=rfq_id)
        return _build_output(context, subject, body)

    # Legacy orchestrator compatibility -------------------------------------------------
    def execute(self, context: Any) -> Any:  # pragma: no cover - legacy flow support
        raise NotImplementedError(
            "EmailDraftingAgent.execute is not supported in the lightweight implementation."
        )

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


def _num_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None
