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

    current_offer = float(ctx.current_offer)
    target_price = float(ctx.target_price)
    gap_value = current_offer - target_price
    gap_pct = gap_value / target_price if target_price else None
    finality = _detect_finality(signals.message_text)

    threshold = ctx.walkaway_price if ctx.walkaway_price is not None else target_price

    if finality:
        log.append("Supplier message contained final-offer language.")
        if threshold is not None and current_offer <= threshold:
            counter = round(min(current_offer, target_price), 2)
            message = (
                "Supplier signalled final offer within acceptable threshold; accept with a minor sweetener."
            )
            if ctx.ask_early_pay_disc:
                asks.append(
                    f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount on this final offer."
                )
            asks.append("Confirm net 30 terms and shipment schedule before closing.")
            if ctx.ask_lead_time_keep:
                lead_time_request = "Confirm committed lead time"
            return {
                "decision": "accept",
                "counter_price": counter,
                "asks": list(dict.fromkeys(asks)),
                "lead_time_request": lead_time_request,
                "message": message,
                "log": log,
                "finality": True,
            }

        message = (
            "Supplier marked this as a final offer but it remains above our no-deal threshold; pause and escalate."
        )
        log.append("Final offer rejected because it exceeds walk-away threshold.")
        return {
            "decision": "decline",
            "counter_price": None,
            "asks": [],
            "lead_time_request": None,
            "message": message,
            "log": log,
            "finality": True,
        }

    if gap_value <= 0:
        counter = round(min(current_offer, target_price), 2)
        message = "Supplier already at/below target; accept while requesting a minor sweetener."
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
            "asks": list(dict.fromkeys(asks)),
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": finality,
        }

    round_no = ctx.round_index
    counter_price: float

    if round_no <= 1:
        if gap_pct is not None and gap_pct > 0.10:
            counter_price = max(target_price, current_offer * 0.88)
            log.append("Round 1 anchor strategy applied with ~12% reduction from supplier offer.")
            asks.extend(
                [
                    "Volume-based discount for 250/500 unit tiers?",
                    "Improved payment terms (net 45 or early-pay option).",
                    "Explore alternative specs or components to lower cost.",
                ]
            )
        else:
            counter_price = (current_offer + target_price) / 2
            log.append("Round 1 gap within 10%; proposing midpoint counter.")
            asks.extend(
                [
                    "Hold quoted price for the full project timeline.",
                    "Include expedited production slot if volumes increase.",
                ]
            )
        message_intro = "Round 1 plan"
    elif round_no == 2:
        if gap_pct is not None and gap_pct <= 0.10:
            counter_price = (current_offer + target_price) / 2
            log.append("Round 2 midpoint strategy: splitting difference with supplier.")
        else:
            step = gap_value * 0.6
            counter_price = current_offer - step
            log.append(
                "Round 2 large gap: taking a firm step-down while highlighting business case."
            )
        asks.extend(
            [
                "Lock in pricing for 12-month demand forecast.",
                "Share cost breakdown to validate efficiency improvements.",
            ]
        )
        message_intro = "Round 2 plan"
    else:
        near_target = gap_pct is not None and gap_pct <= 0.02
        if near_target:
            counter_price = min(current_offer, target_price)
            log.append("Round 3 within 2% of target; recommend accept-with-sweetener.")
            asks.extend(
                [
                    "Confirm net 30 terms and include expedited shipping once per quarter.",
                    "Add volume break trigger at 300 units.",
                ]
            )
            message_intro = "Round 3 close-out"
            decision = "accept"
        else:
            step = gap_value * 0.7
            counter_price = max(target_price, current_offer - step)
            log.append("Round 3 still wide gap; countering firmly toward target.")
            asks.extend(
                [
                    "Provide quarterly business reviews to unlock further concessions.",
                    "Evaluate split shipments to optimise logistics cost.",
                ]
            )
            message_intro = "Round 3 plan"
            decision = "counter"

        counter_price = min(counter_price, current_offer)
        counter_price = round(counter_price, 2)

        if ctx.ask_early_pay_disc:
            asks.append(
                f"Consider {ctx.ask_early_pay_disc * 100:.1f}% early-pay discount if we prepay."
            )
        if ctx.ask_lead_time_keep and lead_time_request is None:
            lead_time_request = "Maintain committed lead time"
        message = (
            f"{message_intro}: {('accept' if near_target else 'counter')} at "
            f"{_format_currency(counter_price, ctx.currency)} vs supplier offer {_format_currency(current_offer, ctx.currency)}."
        )
        return {
            "decision": decision,
            "counter_price": counter_price,
            "asks": list(dict.fromkeys(asks)),
            "lead_time_request": lead_time_request,
            "message": message,
            "log": log,
            "finality": finality,
        }

    counter_price = min(counter_price, current_offer)
    counter_price = round(counter_price, 2)

    if ctx.ask_early_pay_disc:
        asks.append(
            f"Offer early payment for {ctx.ask_early_pay_disc * 100:.1f}% discount in exchange for lower price."
        )
    if ctx.ask_lead_time_keep:
        lead_time_request = "Maintain or improve current lead time commitment"
        asks.append("Hold existing lead time commitments.")

    message = (
        f"{message_intro}: counter at {_format_currency(counter_price, ctx.currency)}"
        f" vs supplier offer {_format_currency(current_offer, ctx.currency)}."
    )
    if round_no == 2:
        message += " Reinforcing total cost and partnership benefits for midpoint split."

    return {
        "decision": "counter",
        "counter_price": counter_price,
        "asks": list(dict.fromkeys(asks)),
        "lead_time_request": lead_time_request,
        "message": message,
        "log": log,
        "finality": finality,
    }
