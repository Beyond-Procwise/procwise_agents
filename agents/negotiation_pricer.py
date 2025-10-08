"""Deterministic pricing helper for the negotiation agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

FINALITY_PATTERNS = (
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


@dataclass
class NegotiationContext:
    current_offer: float
    target_price: float
    round_index: int = 1
    currency: Optional[str] = None
    aggressiveness: float = 0.5
    leverage: float = 0.5
    urgency: float = 0.5
    risk_buffer_pct: float = 0.05
    min_abs_buffer: float = 0.0
    step_pct_of_gap: float = 0.1
    min_abs_step: float = 1.0
    max_rounds: int = 3
    walkaway_price: Optional[float] = None
    ask_early_pay_disc: Optional[float] = None
    ask_lead_time_keep: bool = True


@dataclass
class SupplierSignals:
    offer_prev: Optional[float] = None
    offer_new: Optional[float] = None
    message_text: str = ""


def _detect_finality(message: str) -> bool:
    lowered = (message or "").lower()
    return any(pattern in lowered for pattern in FINALITY_PATTERNS)


def _format_currency(amount: float, currency: Optional[str]) -> str:
    if currency:
        return f"{currency} {amount:,.2f}"
    return f"{amount:,.2f}"


def plan_counter(ctx: NegotiationContext, signals: SupplierSignals) -> Dict[str, object]:
    """Plan counter pricing and supporting asks for the negotiation agent."""

    log: List[str] = []
    asks: List[str] = []
    lead_time_request: Optional[str] = None

    if ctx.round_index > ctx.max_rounds:
        counter = min(ctx.current_offer, ctx.walkaway_price or ctx.current_offer)
        counter = round(counter, 2)
        message = (
            f"Round {ctx.round_index} exceeds configured max rounds; hold at "
            f"{_format_currency(counter, ctx.currency)} and focus on non-price levers."
        )
        log.append("Max rounds reached; maintaining prior position.")
        return {
            "decision": "hold",
            "counter_price": counter,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": False,
        }

    if ctx.current_offer <= 0 or ctx.target_price <= 0:
        message = "Offer or target missing/invalid; request structured pricing."
        asks.append("Confirm unit price, currency, tiered price @ 100/250/500.")
        log.append("Invalid numeric input detected while planning counter.")
        return {
            "decision": "clarify",
            "counter_price": None,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": False,
        }

    gap_value = ctx.current_offer - ctx.target_price
    finality = _detect_finality(signals.message_text)
    supplier_concession = False
    concession_amt = 0.0
    if signals.offer_prev is not None and signals.offer_new is not None:
        try:
            concession_amt = float(signals.offer_prev) - float(signals.offer_new)
            supplier_concession = concession_amt > max(0.5, abs(float(signals.offer_prev)) * 0.005)
        except (TypeError, ValueError):
            concession_amt = 0.0
            supplier_concession = False

    if gap_value <= 0:
        counter = round(min(ctx.current_offer, ctx.target_price), 2)
        message = (
            "Supplier already at/below target; accept while requesting a minor sweetener."
        )
        log.append("Offer meets target; recommending soft acceptance.")
        if ctx.ask_early_pay_disc:
            asks.append(
                f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount."
            )
        asks.append("Confirm lead time and packaging before sign-off.")
        if ctx.ask_lead_time_keep:
            lead_time_request = "Maintain committed lead time"
        return {
            "decision": "accept",
            "counter_price": counter,
            "asks": asks,
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": finality,
        }

    buffer = max(ctx.min_abs_buffer, ctx.target_price * ctx.risk_buffer_pct)
    pressure = max(0.0, min(1.0, 0.6 * ctx.aggressiveness + 0.25 * ctx.leverage + 0.15 * (1 - ctx.urgency)))
    gap_factor = 0.35 + 0.25 * ctx.aggressiveness + 0.15 * ctx.leverage
    anchor_buffer = min(buffer, max(0.0, gap_value) * gap_factor)
    anchor_buffer = max(anchor_buffer, min(buffer, ctx.min_abs_buffer))
    anchor_lift = max(0.0, gap_value) * 0.05
    base_counter = ctx.target_price + anchor_buffer + anchor_lift
    log.append(
        f"Base counter anchored at target + buffer: {_format_currency(base_counter, ctx.currency)}."
    )

    step_base = max(ctx.min_abs_step, max(0.0, gap_value) * ctx.step_pct_of_gap)
    rounds_over_anchor = max(ctx.round_index - 1, 0)
    if rounds_over_anchor:
        if supplier_concession:
            step_factor = max(0.4, 1 - 0.6 * pressure)
            log.append(
                "Supplier conceded last round; reducing counter step to maintain momentum."
            )
        else:
            step_factor = 1 + 0.6 * pressure
            log.append("No supplier concession; escalating counter by weighted step.")
        if finality:
            step_factor = min(step_factor, 0.5)
            log.append("Detected final-offer language; constraining price escalation.")
        step_total = step_base * step_factor * rounds_over_anchor
        if supplier_concession and concession_amt > 0:
            step_total = max(0.0, step_total - concession_amt * 0.5)
        base_counter += step_total
        log.append(
            f"Applied step of {_format_currency(step_total, ctx.currency)} across {rounds_over_anchor} round(s)."
        )

    counter = max(ctx.target_price, base_counter)
    if ctx.walkaway_price is not None:
        counter = min(counter, ctx.walkaway_price)
        log.append("Applied walkaway guardrail.")
    counter = min(counter, ctx.current_offer)
    counter = round(counter, 2)

    if ctx.ask_early_pay_disc:
        asks.append(
            f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount in exchange for lower price."
        )
    if ctx.ask_lead_time_keep:
        lead_time_request = "Maintain or improve current lead time commitment"
        asks.append("Hold existing lead time commitments.")

    message = (
        f"Round {ctx.round_index}: counter at {_format_currency(counter, ctx.currency)}"
        f" vs supplier offer {_format_currency(ctx.current_offer, ctx.currency)}."
    )
    if finality:
        message += " Supplier signalled finality; emphasise non-price levers."

    return {
        "decision": "counter",
        "counter_price": counter,
        "asks": list(dict.fromkeys(asks)),
        "lead_time_request": lead_time_request,
        "message": message,
        "log": log,
        "finality": finality,
    }
