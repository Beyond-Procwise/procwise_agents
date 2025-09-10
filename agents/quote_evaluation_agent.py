import logging
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import models

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

# Ensure GPU configuration is applied
configure_gpu()

logger = logging.getLogger(__name__)


class QuoteEvaluationAgent(BaseAgent):
    """Agent for retrieving and exposing quote data from Qdrant."""

    def run(self, context: AgentContext) -> AgentOutput:
        return self.process(context)

    def process(self, context: AgentContext) -> AgentOutput:
        """Fetch quotes and return only required commercial fields."""
        try:
            input_data = context.input_data
            product_category = (
                input_data.get("product_category")
                or input_data.get("product_type")
                or None
            )
            quotes = self._fetch_quotes(
                input_data.get("supplier_names", []),
                product_category,
            )

            if not quotes:
                # Absence of quotes should not be treated as a hard failure:
                # downstream agents may still proceed with an empty result set.
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={"quotes": [], "message": "No quotes found"},
                )

            simplified: List[Dict] = []
            for q in quotes:
                simplified.append(
                    {
                        "total_spend": q.get("total_spend", q.get("total_amount")),
                        "tenure": q.get("tenure") or q.get("payment_terms"),
                        "total_cost": q.get("total_cost", q.get("total_amount")),
                        "unit_price": q.get("unit_price", q.get("avg_unit_price")),
                        "volume": q.get("volume", q.get("line_items_count")),
                        "quote_file_s3_path": q.get("quote_file_s3_path") or q.get("s3_path"),
                    }
                )

            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=self._to_native({"quotes": simplified}),
            )
        except Exception as exc:
            logger.error("QuoteEvaluationAgent error: %s", exc)
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(exc),
            )

    def _fetch_quotes(
        self,
        supplier_names: List[str],
        product_category: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch quotes from Qdrant vector database."""

        def ensure_index(field_name: str) -> None:
            try:
                self.agent_nick.qdrant_client.create_payload_index(
                    collection_name=self.settings.qdrant_collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as ie:  # pragma: no cover - network dependent
                logger.debug("Index creation skipped for %s: %s", field_name, ie)

        if supplier_names:
            ensure_index("supplier_name")
        if product_category:
            ensure_index("product_category")

        def build_filter(include_supplier: bool = True) -> models.Filter:
            filters = [
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value="quote"),
                )
            ]
            if include_supplier and supplier_names:
                filters.append(
                    models.FieldCondition(
                        key="supplier_name",
                        match=models.MatchAny(any=supplier_names),
                    )
                )
            if product_category:
                filters.append(
                    models.FieldCondition(
                        key="product_category",
                        match=models.MatchValue(value=product_category),
                    )
                )
            return models.Filter(must=filters)

        def qdrant_scroll(q_filter: models.Filter):
            return self.agent_nick.qdrant_client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                scroll_filter=q_filter,
                with_payload=True,
                limit=100,
            )

        try:
            points, _ = qdrant_scroll(build_filter())
        except Exception as e:
            logger.warning(
                "Primary quote fetch failed (%s). Retrying without supplier filter.", e
            )
            try:
                points, _ = qdrant_scroll(build_filter(include_supplier=False))
            except Exception as e2:
                logger.error("Failed to fetch quotes from Qdrant: %s", e2)
                return []

        quotes: List[Dict] = []
        for point in points:
            payload = point.payload or {}
            quotes.append(
                {
                    "quote_id": payload.get("quote_id", point.id),
                    "supplier_name": payload.get("supplier_name", ""),
                    "total_spend": payload.get("total_spend", payload.get("total_amount", 0)),
                    "tenure": payload.get("tenure"),
                    "total_cost": payload.get("total_cost", payload.get("total_amount", 0)),
                    "unit_price": payload.get("unit_price", payload.get("avg_unit_price", 0)),
                    "volume": payload.get("volume", payload.get("line_items_count", 0)),
                    "quote_file_s3_path": payload.get("quote_file_s3_path")
                    or payload.get("s3_path"),
                    "payment_terms": payload.get("payment_terms"),
                    "avg_unit_price": payload.get("avg_unit_price"),
                    "line_items_count": payload.get("line_items_count"),
                    "total_amount": payload.get("total_amount"),
                }
            )
        return quotes

    def _to_native(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

