"""Quote evaluation agent for post-negotiation supplier analysis.

This agent consolidates negotiation transcripts, derives structured analytics,
and produces human-ready insight packs for the procurement reviewer.  The
design emphasises responsible AI usage: all generated copy is routed to a human
decision-maker and leverages only locally hosted LLMs specified in the
ProcWise deployment manifest.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Tuple

import ollama

from email_thread import EmailThread
from models.quote_analysis import (
    SupplierQuoteSnapshot,
    SupplierScorecard,
    build_analytics_context,
    build_quote_snapshot,
    build_scorecard,
)
from workflow_context_manager import WorkflowContextManager


class QuoteEvaluationAgent:
    """Evaluate final quotes after three negotiation rounds."""

    MODEL_CANDIDATES: Tuple[str, ...] = (
        "qwen3:30b",
        "mixtral:8x7b",
        "phi4:latest",
        "gemma3:1b-it-qat",
        "gemma3:latest",
        "mistral:latest",
        "gpt-oss:latest",
        "llama3.2:latest",
    )

    def __init__(self, context_manager: WorkflowContextManager):
        self.context = context_manager
        override = os.getenv("QUOTE_EVAL_MODEL")
        if override:
            self.model_preferences = (override, *self.MODEL_CANDIDATES)
        else:
            self.model_preferences = self.MODEL_CANDIDATES

    async def evaluate_quotes(self, threads: Dict[str, EmailThread]) -> Dict:
        snapshots: Dict[str, SupplierQuoteSnapshot] = {}
        for supplier_id, thread in threads.items():
            snapshots[supplier_id] = build_quote_snapshot(thread)

        analytics_context = build_analytics_context(list(snapshots.values()))

        evaluations: Dict[str, Dict] = {}
        ranking_material: List[Tuple[str, SupplierScorecard, SupplierQuoteSnapshot]] = []

        for supplier_id, thread in threads.items():
            snapshot = snapshots[supplier_id]
            scorecard = build_scorecard(snapshot, thread, analytics_context)
            narrative = await self._generate_summary(snapshot, scorecard, thread)
            scorecard.narrative = narrative

            evaluation_payload = {
                "supplier_id": supplier_id,
                "final_offer_for_review": snapshot.to_dict(),
                "scorecard": scorecard.to_dict(),
                "human_review": self._build_human_review_pack(snapshot, scorecard),
                "thread_overview": thread.get_full_thread(),
            }
            evaluations[supplier_id] = evaluation_payload
            ranking_material.append((supplier_id, scorecard, snapshot))

        ranked = self._rank_suppliers(ranking_material)

        return {
            "evaluations": evaluations,
            "ranking": ranked,
            "recommendation": ranked[0] if ranked else None,
            "executive_summary": self._build_executive_summary(ranked, evaluations),
            "requires_human_review": True,
        }

    def _rank_suppliers(
        self, ranking_material: List[Tuple[str, SupplierScorecard, SupplierQuoteSnapshot]]
    ) -> List[Dict[str, object]]:
        sorted_entries = sorted(
            ranking_material,
            key=lambda item: item[1].weighted_total(),
            reverse=True,
        )
        return [
            {
                "supplier_id": supplier_id,
                "weighted_score": round(scorecard.weighted_total(), 2),
                "price": snapshot.price,
                "currency": snapshot.currency,
                "lead_time_days": snapshot.lead_time_days,
                "payment_terms_days": snapshot.payment_terms_days,
            }
            for supplier_id, scorecard, snapshot in sorted_entries
        ]

    async def _generate_summary(
        self,
        snapshot: SupplierQuoteSnapshot,
        scorecard: SupplierScorecard,
        thread: EmailThread,
    ) -> str:
        prompt = f"""
You are a senior procurement negotiator preparing a handoff note for a human reviewer.
Use the supplier's latest quote and structured analytics below to craft a warm, human-like
summary that highlights commercial leverage and relationship considerations.

Supplier ID: {snapshot.supplier_id}
Negotiation Round: {snapshot.round_number}
Detected Price: {snapshot.price} {snapshot.currency or ''}
Payment Terms: {snapshot.payment_terms_days or 'n/a'} days
Lead Time Commitment: {snapshot.lead_time_days or 'n/a'} days
Concessions: {', '.join(snapshot.concessions) or 'None observed'}
Risks Observed: {', '.join(snapshot.risks) or 'None flagged'}
Sustainability & Innovation Notes: {', '.join(snapshot.sustainability_flags + snapshot.innovation_notes) or 'None mentioned'}

Scorecard (0-100):
- Price competitiveness: {scorecard.price_competitiveness:.1f}
- Commercial terms: {scorecard.commercial_terms:.1f}
- Responsiveness: {scorecard.responsiveness:.1f}
- Risk posture: {scorecard.risk_posture:.1f}
- Sustainability alignment: {scorecard.sustainability_alignment:.1f}
- Weighted total: {scorecard.weighted_total():.1f}

Email thread context:
{thread.get_full_thread()}

Respond with:
1. A headline sentence (<=18 words) capturing the negotiation stance.
2. 2-3 bullet points on key strengths or leverage.
3. 2 bullet points on watch-outs or required validations.
4. A suggested next action for the human reviewer in one sentence.
Keep the tone collaborative, intelligent, and procurement-savvy.
"""
        return await self._call_ollama(prompt)

    def _build_human_review_pack(
        self, snapshot: SupplierQuoteSnapshot, scorecard: SupplierScorecard
    ) -> Dict[str, object]:
        strengths: List[str] = []
        if snapshot.price is not None:
            strengths.append("Concrete commercial offer captured")
        if snapshot.payment_terms_days and snapshot.payment_terms_days >= 45:
            strengths.append(f"Extended payment terms (Net {snapshot.payment_terms_days})")
        if snapshot.concessions:
            strengths.extend(snapshot.concessions)
        if snapshot.sustainability_flags:
            strengths.append("Supplier highlighted sustainability/ESG credentials")

        watchouts: List[str] = list(snapshot.risks)
        if scorecard.risk_posture < 70.0 and not snapshot.risks:
            watchouts.append("Validate operational contingencies; limited risk coverage in supplier note")
        if snapshot.lead_time_days and snapshot.lead_time_days > 28:
            watchouts.append("Longer lead time than desired – confirm with demand planners")

        next_step = (
            "Recommend preparing award rationale pending final finance sign-off"
            if scorecard.weighted_total() >= 80.0
            else "Arrange follow-up call to probe commercial flex and mitigation options"
        )

        return {
            "headline_score": scorecard.weighted_total(),
            "strengths": strengths or ["Awaiting supplier concessions"],
            "watchouts": watchouts or ["No explicit risks cited; validate independently"],
            "recommended_next_step": next_step,
            "final_offer_text": snapshot.final_offer_text,
        }

    def _build_executive_summary(
        self, ranking: List[Dict[str, object]], evaluations: Dict[str, Dict]
    ) -> str:
        if not ranking:
            return "No supplier responses available for executive summary."
        top_entry = ranking[0]
        supplier_id = top_entry["supplier_id"]
        score = top_entry["weighted_score"]
        details = evaluations[supplier_id]["human_review"]
        return (
            f"Top supplier: {supplier_id} with weighted score {score}. "
            f"Key strengths: {', '.join(details['strengths'][:2])}. "
            f"Watch-outs: {', '.join(details['watchouts'][:1])}."
        )

    async def _call_ollama(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a procurement co-pilot that writes polished negotiation insights."},
            {"role": "user", "content": prompt},
        ]
        last_error: str = ""
        for model in self.model_preferences:
            try:
                if hasattr(ollama, "AsyncClient"):
                    client = ollama.AsyncClient()
                    response = await client.chat(model=model, messages=messages)
                else:  # pragma: no cover - compatibility path
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, lambda: ollama.chat(model=model, messages=messages)
                    )
                content = response.get("message", {}).get("content")
                if content:
                    return content.strip()
            except Exception as exc:  # pragma: no cover - defensive
                last_error = str(exc)
                continue
        return f"[LLM summary unavailable: {last_error or 'all local models failed'}]"
