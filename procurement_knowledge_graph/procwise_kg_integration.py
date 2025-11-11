from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .config import AppConfig, app_config_from_global_settings
from .hybrid_query_engine import HybridProcurementQueryEngine


class ProcWiseKnowledgeGraphAgent:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        if config is None:
            try:
                config = app_config_from_global_settings()
            except RuntimeError:
                config = AppConfig.from_env()
        self._config = config
        self.engine = HybridProcurementQueryEngine.from_config(self._config)

    @classmethod
    def from_settings(cls) -> "ProcWiseKnowledgeGraphAgent":
        return cls(app_config_from_global_settings())

    def close(self) -> None:
        self.engine.close()

    # ------------------------------------------------------------------
    # Method 1
    # ------------------------------------------------------------------
    def should_trigger_negotiation(
        self,
        workflow_id: str,
        *,
        rfq_id: Optional[str] = None,
        min_responses_required: int = 3,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"workflow_id": workflow_id}
        response_filters = ["workflow_id = :workflow_id", "COALESCE(processed, FALSE) = FALSE"]
        if rfq_id and self.engine.table_has_column("proc", "supplier_response", "rfq_id"):
            params["rfq_id"] = rfq_id
            response_filters.append("rfq_id = :rfq_id")
        response_query = f"""
            SELECT COUNT(DISTINCT unique_id) AS response_count
            FROM proc.supplier_response
            WHERE {' AND '.join(response_filters)}
        """

        invite_params: Dict[str, Any] = {"workflow_id": workflow_id}
        invite_filters = ["workflow_id = :workflow_id"]
        has_rfq = bool(rfq_id) and self.engine.table_has_column(
            "proc", "workflow_email_tracking", "rfq_id"
        )
        if has_rfq:
            invite_params["rfq_id"] = rfq_id
            invite_filters.append("rfq_id = :rfq_id")
        invite_query = f"""
            SELECT COUNT(DISTINCT unique_id) AS invited_count
            FROM proc.workflow_email_tracking
            WHERE {' AND '.join(invite_filters)}
        """

        response_count = self.engine.execute_scalar(response_query, params) or 0
        invited_count = self.engine.execute_scalar(invite_query, invite_params) or 0
        effective_threshold = min_responses_required
        if invited_count and invited_count < min_responses_required:
            effective_threshold = invited_count
        should_trigger = response_count >= max(effective_threshold, 1) and invited_count > 0
        all_suppliers_responded = invited_count > 0 and response_count >= invited_count
        recommendation = (
            "Proceed with negotiation"
            if should_trigger
            else (
                "Wait for additional supplier responses"
                if invited_count and response_count < effective_threshold
                else "Awaiting initial supplier responses"
            )
        )
        return {
            "workflow_id": workflow_id,
            "rfq_id": rfq_id,
            "response_count": int(response_count),
            "invited_count": int(invited_count),
            "should_trigger_negotiation": bool(should_trigger),
            "all_suppliers_responded": bool(all_suppliers_responded),
            "threshold_required": int(effective_threshold),
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Method 2
    # ------------------------------------------------------------------
    def get_negotiation_context(
        self,
        rfq_id: str,
        supplier_id: str,
        current_round: int,
    ) -> Dict[str, Any]:
        performance = self.engine.get_supplier_performance_metrics(supplier_id)
        state = self.engine.run_cypher(
            """
            MATCH (n:NegotiationSession {rfq_id: $rfq_id, supplier_id: $supplier_id})
            RETURN n.rfq_id AS rfq_id,
                   n.supplier_id AS supplier_id,
                   n.round AS round,
                   n.counter_offer AS counter_offer,
                   n.status AS status,
                   n.awaiting_response AS awaiting_response,
                   n.supplier_reply_count AS supplier_reply_count
            ORDER BY n.round DESC
            LIMIT 1
            """,
            {"rfq_id": rfq_id, "supplier_id": supplier_id},
        )
        negotiation_state = state[0] if state else None
        pattern = self.engine.analyze_negotiation_pattern(supplier_id)
        vector_context = f"RFQ {rfq_id} Supplier {supplier_id} Round {current_round}"
        similar = self.engine.find_similar_negotiations(vector_context)
        return {
            "performance": performance,
            "state": negotiation_state,
            "pattern_analysis": pattern,
            "similar_negotiations": similar,
        }

    # ------------------------------------------------------------------
    # Method 3
    # ------------------------------------------------------------------
    def generate_negotiation_strategy(
        self,
        rfq_id: str,
        supplier_id: str,
        target_price: float,
        current_quote: float,
    ) -> Dict[str, Any]:
        context = self.get_negotiation_context(rfq_id, supplier_id, current_round=1)
        performance = context["performance"]
        price_gap = current_quote - target_price
        gap_percent = (price_gap / current_quote * 100) if current_quote else 0
        similar = context["similar_negotiations"]
        pattern = context["pattern_analysis"]
        payload = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "target_price": target_price,
            "current_quote": current_quote,
            "price_gap": price_gap,
            "price_gap_percent": gap_percent,
            "performance": performance,
            "similar_negotiations": similar,
            "pattern_analysis": pattern,
        }
        llm_response = self.engine.generate_llm_json(
            system="Provide data-driven negotiation recommendations in JSON.",
            prompt=json.dumps(payload),
        )
        return {
            "context": payload,
            "strategy": llm_response,
        }

    # ------------------------------------------------------------------
    # Method 4
    # ------------------------------------------------------------------
    def track_negotiation_round(
        self,
        workflow_id: str,
        rfq_id: Optional[str],
        supplier_id: str,
        round_number: int,
        email_message_id: str,
        dispatched_at: str,
    ) -> bool:
        parts = [workflow_id, supplier_id, str(round_number)]
        if rfq_id:
            parts.insert(1, rfq_id)
        unique_id = "_".join(filter(None, parts))

        has_rfq_column = bool(rfq_id) and self.engine.table_has_column(
            "proc", "workflow_email_tracking", "rfq_id"
        )
        columns = ["workflow_id", "unique_id"]
        params: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "unique_id": unique_id,
            "supplier_id": supplier_id,
            "message_id": email_message_id,
            "dispatched_at": dispatched_at,
        }
        if has_rfq_column:
            columns.append("rfq_id")
            params["rfq_id"] = rfq_id
        columns.extend(["supplier_id", "message_id", "dispatched_at"])
        placeholders = [f":{column}" for column in columns]
        update_assignments = [
            "message_id = EXCLUDED.message_id",
            "dispatched_at = EXCLUDED.dispatched_at",
            "supplier_id = COALESCE(EXCLUDED.supplier_id, proc.workflow_email_tracking.supplier_id)",
        ]
        if has_rfq_column:
            update_assignments.append("rfq_id = EXCLUDED.rfq_id")
        query = f"""
            INSERT INTO proc.workflow_email_tracking (
                {', '.join(columns)}
            ) VALUES ({', '.join(placeholders)})
            ON CONFLICT (workflow_id, unique_id)
            DO UPDATE SET {', '.join(update_assignments)}
        """
        self.engine.execute_write(query, params)
        return True

    # ------------------------------------------------------------------
    # Method 5
    # ------------------------------------------------------------------
    def get_email_thread_context(self, rfq_id: str, supplier_id: str) -> Dict[str, Any]:
        return self.engine.get_email_thread(rfq_id, supplier_id)

    # ------------------------------------------------------------------
    # Method 6
    # ------------------------------------------------------------------
    def generate_counter_offer_email(
        self,
        rfq_id: str,
        supplier_id: str,
        counter_offer: float,
        round_number: int,
        justification_points: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        context = self.get_negotiation_context(rfq_id, supplier_id, round_number)
        thread = self.get_email_thread_context(rfq_id, supplier_id)
        supplier_metrics = context["performance"]
        payload = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "counter_offer": counter_offer,
            "round_number": round_number,
            "justification_points": justification_points or [],
            "supplier_metrics": supplier_metrics,
            "thread": thread,
        }
        llm_response = self.engine.generate_llm_json(
            system="Draft professional procurement emails. Return JSON with subject, body, tone, closing.",
            prompt=json.dumps(payload),
        )
        supplier_query = "SELECT contact_email FROM proc.supplier WHERE supplier_id = :supplier_id"
        df = self.engine.query_dataframe(supplier_query, {"supplier_id": supplier_id})
        supplier_email = df.iloc[0]["contact_email"] if not df.empty else None
        return {
            "subject": llm_response.get("subject", f"Counter-offer for RFQ {rfq_id}"),
            "body": llm_response.get("body", ""),
            "threading_headers": thread["threading_headers"],
            "supplier_email": supplier_email,
        }

    # ------------------------------------------------------------------
    # Method 7
    # ------------------------------------------------------------------
    def get_supplier_communication_preferences(self, supplier_id: str) -> Dict[str, Any]:
        stats = self.engine.get_supplier_response_stats(supplier_id)
        response_hours = stats.get("avg_response_hours")
        if response_hours is None:
            classification = "Unknown"
            recommendation = "Schedule follow-up within 48 hours"
        elif response_hours < 24:
            classification = "Very Responsive"
            recommendation = "Expect replies within a day; send follow-up after 24h if needed."
        elif response_hours < 72:
            classification = "Moderate"
            recommendation = "Allow 2-3 days before follow-up."
        else:
            classification = "Slow"
            recommendation = "Plan proactive reminders within 48 hours."
        return {
            "supplier_id": supplier_id,
            "avg_response_hours": response_hours,
            "classification": classification,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Method 8
    # ------------------------------------------------------------------
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        return self.engine.get_workflow_overview(workflow_id)

    # Convenience wrappers for analytics methods
    def get_supplier_performance_metrics(self, supplier_id: str) -> Dict[str, Any]:
        return self.engine.get_supplier_performance_metrics(supplier_id)

    def find_similar_negotiations(self, negotiation_context: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.engine.find_similar_negotiations(negotiation_context, top_k=top_k)

    def analyze_negotiation_pattern(self, supplier_id: str) -> Dict[str, Any]:
        return self.engine.analyze_negotiation_pattern(supplier_id)

    def natural_language_query(self, question: str) -> Dict[str, Any]:
        return self.engine.natural_language_query(question)


__all__ = ["ProcWiseKnowledgeGraphAgent"]
