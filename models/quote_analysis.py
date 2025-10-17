"""Utilities for analysing supplier quotes and negotiation outcomes.

This module extracts structured commercial signals from :class:`EmailThread`
objects so that downstream agents can evaluate supplier proposals without
relying on brittle prompt-only logic.  The heuristics focus on pricing,
payment terms, delivery commitments, sustainability cues, and risk language –
key themes highlighted by modern procurement platforms such as Oro Labs.

The helpers are intentionally lightweight so that they can execute inside a
trusted procurement environment without external dependencies while still
producing rich context for human-in-the-loop review.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from email_thread import EmailThread


_CURRENCY_SYMBOLS: Dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
}

_CURRENCY_PATTERN = re.compile(
    r"(?P<currency>USD|EUR|GBP|INR|AUD|CAD|SGD|CHF|JPY|NOK|SEK|DKK|[€$£₹])?\s*(?P<amount>\d+(?:[.,]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_LEAD_TIME_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>business\s+days?|days?|weeks?)",
    re.IGNORECASE,
)

_PAYMENT_TERMS_PATTERN = re.compile(r"net\s*(?P<days>\d+)", re.IGNORECASE)

_DISCOUNT_HINT_PATTERN = re.compile(
    r"(discount|reduc(?:e|tion)|drop(?:ped)?|savings?|concession|better price)",
    re.IGNORECASE,
)

_RISK_PHRASE_PATTERN = re.compile(
    r"(delay|risk|issue|concern|backlog|unavailable|unable|cannot|excess cost)",
    re.IGNORECASE,
)

_SUSTAINABILITY_PATTERN = re.compile(
    r"(sustainab|esg|green|recycl|carbon|ethical|inclusive|diversity)",
    re.IGNORECASE,
)

_INNOVATION_PATTERN = re.compile(
    r"(innovation|roadmap|co-?develop|automation|analytics|ai|machine learning)",
    re.IGNORECASE,
)


def _clean_amount(text: str) -> float:
    return float(text.replace(",", "").replace(" ", ""))


def _clamp(value: float, *, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _convert_lead_time(value: float, unit: str) -> int:
    unit_lower = unit.lower()
    if "week" in unit_lower:
        return int(round(value * 7))
    return int(round(value))


def _extract_currency(match: re.Match) -> Optional[str]:
    symbol = match.group("currency")
    if not symbol:
        return None
    symbol = symbol.strip()
    return _CURRENCY_SYMBOLS.get(symbol, symbol.upper())


@dataclass
class SupplierQuoteSnapshot:
    """Structured summary of the supplier's latest offer."""

    supplier_id: str
    final_offer_text: str
    round_number: int
    price: Optional[float] = None
    currency: Optional[str] = None
    payment_terms_days: Optional[int] = None
    lead_time_days: Optional[int] = None
    concessions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    sustainability_flags: List[str] = field(default_factory=list)
    innovation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "supplier_id": self.supplier_id,
            "round_number": self.round_number,
            "final_offer_text": self.final_offer_text,
            "price": self.price,
            "currency": self.currency,
            "payment_terms_days": self.payment_terms_days,
            "lead_time_days": self.lead_time_days,
            "concessions": list(self.concessions),
            "risks": list(self.risks),
            "sustainability_flags": list(self.sustainability_flags),
            "innovation_notes": list(self.innovation_notes),
        }


@dataclass
class QuoteAnalyticsContext:
    """Benchmark values derived from all supplier responses."""

    price_min: Optional[float]
    price_max: Optional[float]
    lead_min: Optional[int]
    lead_max: Optional[int]
    payment_max: Optional[int]


@dataclass
class SupplierScorecard:
    """Composite scoring model used for supplier ranking."""

    price_competitiveness: float
    commercial_terms: float
    responsiveness: float
    risk_posture: float
    sustainability_alignment: float
    narrative: str = ""

    PRICE_WEIGHT: float = 0.32
    TERMS_WEIGHT: float = 0.22
    RESPONSIVENESS_WEIGHT: float = 0.18
    RISK_WEIGHT: float = 0.18
    SUSTAINABILITY_WEIGHT: float = 0.10

    def weighted_total(self) -> float:
        return _clamp(
            self.price_competitiveness * self.PRICE_WEIGHT
            + self.commercial_terms * self.TERMS_WEIGHT
            + self.responsiveness * self.RESPONSIVENESS_WEIGHT
            + self.risk_posture * self.RISK_WEIGHT
            + self.sustainability_alignment * self.SUSTAINABILITY_WEIGHT
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "price_competitiveness": round(self.price_competitiveness, 2),
            "commercial_terms": round(self.commercial_terms, 2),
            "responsiveness": round(self.responsiveness, 2),
            "risk_posture": round(self.risk_posture, 2),
            "sustainability_alignment": round(self.sustainability_alignment, 2),
            "weighted_total": round(self.weighted_total(), 2),
            "narrative": self.narrative,
        }


def build_quote_snapshot(thread: "EmailThread") -> SupplierQuoteSnapshot:
    """Extract the latest supplier offer from an email thread."""

    supplier_id = getattr(thread, "supplier_id", "")
    final_round = getattr(thread, "current_round", 0) or 0
    final_message: Optional[Dict[str, object]] = None

    for message in reversed(getattr(thread, "messages", [])):
        message_type = str(message.get("message_type", "")).lower()
        if message_type.startswith("supplier_response"):
            final_message = message
            break

    if final_message is None:
        return SupplierQuoteSnapshot(
            supplier_id=supplier_id,
            final_offer_text="",
            round_number=final_round,
        )

    text = str(final_message.get("content", "")).strip()
    round_number = int(final_message.get("round") or final_round or 0)

    price: Optional[float] = None
    currency: Optional[str] = None
    for match in _CURRENCY_PATTERN.finditer(text):
        amount_raw = match.group("amount")
        if not amount_raw:
            continue
        start, end = match.span()
        prefix = text[max(0, start - 8) : start].lower()
        suffix = text[end : min(len(text), end + 6)].lower()
        if "net" in prefix:
            continue
        if any(unit in suffix for unit in ("day", "week")):
            continue
        try:
            price_candidate = _clean_amount(amount_raw)
        except ValueError:
            continue
        detected_currency = _extract_currency(match)
        if not detected_currency and not re.search(r"(usd|eur|gbp|aud|cad|price|offer)", prefix + suffix):
            # require an explicit currency cue when possible to avoid false positives
            continue
        price = price_candidate
        if detected_currency:
            currency = detected_currency
        # take the last recognised value assuming it refers to the final number

    payment_terms_days: Optional[int] = None
    payment_match = _PAYMENT_TERMS_PATTERN.search(text)
    if payment_match:
        try:
            payment_terms_days = int(payment_match.group("days"))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            payment_terms_days = None

    lead_time_days: Optional[int] = None
    for lead_match in _LEAD_TIME_PATTERN.finditer(text):
        value = float(lead_match.group("value"))
        lead_time_days = _convert_lead_time(value, lead_match.group("unit"))

    concessions: List[str] = []
    if _DISCOUNT_HINT_PATTERN.search(text):
        concessions.append("Commercial discount offered")
    if lead_time_days is not None:
        concessions.append(f"Committed lead time: {lead_time_days} days")
    if payment_terms_days is not None:
        concessions.append(f"Payment terms: Net {payment_terms_days}")

    risks: List[str] = []
    for sentence in _split_sentences(text):
        if _RISK_PHRASE_PATTERN.search(sentence):
            risks.append(sentence.strip())

    sustainability_flags: List[str] = []
    for sentence in _split_sentences(text):
        if _SUSTAINABILITY_PATTERN.search(sentence):
            sustainability_flags.append(sentence.strip())

    innovation_notes: List[str] = []
    for sentence in _split_sentences(text):
        if _INNOVATION_PATTERN.search(sentence):
            innovation_notes.append(sentence.strip())

    return SupplierQuoteSnapshot(
        supplier_id=supplier_id,
        final_offer_text=text,
        round_number=round_number,
        price=price,
        currency=currency,
        payment_terms_days=payment_terms_days,
        lead_time_days=lead_time_days,
        concessions=concessions,
        risks=risks,
        sustainability_flags=sustainability_flags,
        innovation_notes=innovation_notes,
    )


def _split_sentences(text: str) -> List[str]:
    fragments = re.split(r"(?<=[.!?])\s+", text)
    return [fragment for fragment in fragments if fragment.strip()]


def build_analytics_context(
    snapshots: Sequence[SupplierQuoteSnapshot],
) -> QuoteAnalyticsContext:
    prices = [snap.price for snap in snapshots if isinstance(snap.price, (int, float))]
    leads = [snap.lead_time_days for snap in snapshots if isinstance(snap.lead_time_days, int)]
    payments = [
        snap.payment_terms_days for snap in snapshots if isinstance(snap.payment_terms_days, int)
    ]
    return QuoteAnalyticsContext(
        price_min=min(prices) if prices else None,
        price_max=max(prices) if prices else None,
        lead_min=min(leads) if leads else None,
        lead_max=max(leads) if leads else None,
        payment_max=max(payments) if payments else None,
    )


def _score_price(snapshot: SupplierQuoteSnapshot, context: QuoteAnalyticsContext) -> float:
    if snapshot.price is None:
        return 65.0
    price_min = context.price_min
    price_max = context.price_max
    if price_min is None or price_max is None or math.isclose(price_min, price_max):
        base = 82.0
    else:
        spread = price_max - price_min
        relative = (snapshot.price - price_min) / spread if spread else 0.0
        base = 95.0 - (relative * 35.0)
    if any("discount" in concession.lower() for concession in snapshot.concessions):
        base += 4.0
    return _clamp(base)


def _score_terms(snapshot: SupplierQuoteSnapshot, context: QuoteAnalyticsContext) -> float:
    payment_score = 58.0
    if snapshot.payment_terms_days:
        benchmark = context.payment_max or snapshot.payment_terms_days
        if benchmark <= 0:
            benchmark = snapshot.payment_terms_days
        ratio = snapshot.payment_terms_days / benchmark
        payment_score = 60.0 + ratio * 35.0
    lead_score = 62.0
    if snapshot.lead_time_days:
        lead_min = context.lead_min or snapshot.lead_time_days
        lead_max = context.lead_max or snapshot.lead_time_days
        if lead_max == lead_min:
            lead_score = 78.0
        else:
            ratio = (snapshot.lead_time_days - lead_min) / (lead_max - lead_min)
            lead_score = 88.0 - ratio * 40.0
    return _clamp(0.6 * payment_score + 0.4 * lead_score)


def _score_responsiveness(thread: "EmailThread") -> float:
    supplier_rounds = {
        int(message.get("round", 0))
        for message in getattr(thread, "messages", [])
        if str(message.get("message_type", "")).lower().startswith("supplier_response")
    }
    total_rounds = max(supplier_rounds) if supplier_rounds else getattr(thread, "current_round", 0) or 0
    total_rounds = max(total_rounds, len(supplier_rounds))
    if total_rounds <= 0:
        return 60.0
    responsiveness_ratio = len(supplier_rounds) / total_rounds
    base = 58.0 + responsiveness_ratio * 37.0
    if len(supplier_rounds) >= total_rounds:
        base += 5.0
    return _clamp(base)


def _score_risk(snapshot: SupplierQuoteSnapshot) -> float:
    if not snapshot.risks:
        return 86.0
    penalties = 0.0
    for entry in snapshot.risks:
        if _RISK_PHRASE_PATTERN.search(entry):
            penalties += 14.0
        else:
            penalties += 8.0
    return _clamp(86.0 - penalties)


def _score_sustainability(snapshot: SupplierQuoteSnapshot) -> float:
    if snapshot.sustainability_flags:
        return 82.0
    if snapshot.innovation_notes:
        return 74.0
    return 48.0


def build_scorecard(
    snapshot: SupplierQuoteSnapshot,
    thread: "EmailThread",
    context: QuoteAnalyticsContext,
) -> SupplierScorecard:
    return SupplierScorecard(
        price_competitiveness=_score_price(snapshot, context),
        commercial_terms=_score_terms(snapshot, context),
        responsiveness=_score_responsiveness(thread),
        risk_posture=_score_risk(snapshot),
        sustainability_alignment=_score_sustainability(snapshot),
    )

