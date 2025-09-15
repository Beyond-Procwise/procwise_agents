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
        """Fetch quotes and return only required commercial fields or compare RFQ responses."""
        try:
            input_data = context.input_data
            logger.info("QuoteEvaluationAgent: starting with input %s", input_data)
            rfq_id = input_data.get("rfq_id")
            if rfq_id:
                quotes = self._get_responses_from_db(rfq_id)
                quotes_sorted = sorted(
                    quotes,
                    key=lambda q: (q.get("price", float("inf")), q.get("lead_time", "")),
                )
                best = quotes_sorted[0] if quotes_sorted else None
                next_agents = ["approvals"] if best else []
                logger.info(
                    "QuoteEvaluationAgent: retrieved %d quotes for rfq %s",
                    len(quotes_sorted),
                    rfq_id,
                )
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={"quotes": quotes_sorted, "best_quote": best},
                    pass_fields={"quotes": quotes_sorted, "best_quote": best},
                    next_agents=next_agents,
                )

            product_category = (
                input_data.get("product_category")
                or input_data.get("product_type")
                or None
            )
            weights = dict(
                input_data.get(
                    "weights", {"price": 0.5, "delivery": 0.3, "risk": 0.2}
                )
            )

            quotes = self._fetch_quotes(
                input_data.get("supplier_names", []),
                product_category,
            )
            logger.info(
                "QuoteEvaluationAgent: fetched %d quotes for category %s",
                len(quotes),
                product_category,
            )

            standardized_quotes = self._standardize_quotes(quotes, weights)

            if not quotes:
                # Absence of quotes should not be treated as a hard failure:
                # downstream agents may still proceed with an empty result set.
                empty_payload = self._to_native(
                    {
                        "quotes": standardized_quotes,
                        "weights": weights,
                        "message": "No quotes found",
                    }
                )
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=empty_payload,
                    pass_fields=empty_payload,
                )

            output_data = self._to_native(
                {"quotes": standardized_quotes, "weights": weights}
            )
            logger.debug("QuoteEvaluationAgent output: %s", output_data)
            logger.info(
                "QuoteEvaluationAgent returning %d supplier quotes",
                max(len(standardized_quotes) - 1, 0),
            )
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields=output_data,
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

    def _standardize_quotes(
        self, quotes: List[Dict], weights: Dict
    ) -> List[Dict]:
        """Produce the standard quote comparison output structure."""

        standardized: List[Dict] = [self._build_weighting_entry(weights)]
        for quote in quotes:
            standardized.append(self._standardize_quote(quote))
        return standardized

    @staticmethod
    def _build_weighting_entry(weights: Dict) -> Dict:
        """Create the leading weighting row for the comparison table."""

        return {
            "name": "weighting",
            "total_spend": weights.get("price", 0),
            "total_cost": weights.get("delivery", 0),
            "unit_price": weights.get("risk", 0),
            "quote_file_s3_path": None,
            "tenure": weights.get("tenure"),
            "volume": weights.get("volume", 0),
        }

    @staticmethod
    def _standardize_quote(quote: Dict) -> Dict:
        """Normalise quote payload keys into the standard output schema."""

        supplier_name = QuoteEvaluationAgent._coalesce(
            quote, ["supplier_name", "name"], None, strip_strings=True
        )
        if supplier_name is None:
            supplier_identifier = QuoteEvaluationAgent._coalesce(
                quote, ["supplier_id", "quote_id"], None, strip_strings=True
            )
            if supplier_identifier is not None:
                supplier_name = f"Supplier {supplier_identifier}"
            else:
                supplier_name = "Unknown supplier"

        return {
            "name": str(supplier_name),
            "total_spend": QuoteEvaluationAgent._coalesce(
                quote, ["total_spend", "total_amount"], 0
            ),
            "total_cost": QuoteEvaluationAgent._coalesce(
                quote, ["total_cost", "total_amount"], 0
            ),
            "unit_price": QuoteEvaluationAgent._coalesce(
                quote, ["unit_price", "avg_unit_price"], 0
            ),
            "quote_file_s3_path": QuoteEvaluationAgent._coalesce(
                quote, ["quote_file_s3_path", "s3_path"], None, strip_strings=False
            ),
            "tenure": QuoteEvaluationAgent._coalesce(
                quote, ["tenure", "payment_terms"], None
            ),
            "volume": QuoteEvaluationAgent._coalesce(
                quote, ["volume", "line_items_count"], 0
            ),
        }

    @staticmethod
    def _coalesce(
        source: Dict,
        keys: List[str],
        default,
        *,
        strip_strings: bool = True,
    ):
        """Return the first non-empty value from ``source`` for ``keys``."""

        for key in keys:
            if key not in source:
                continue
            value = source[key]
            if value is None:
                continue
            if strip_strings and isinstance(value, str):
                if not value.strip():
                    continue
            return value
        return default

    def _get_responses_from_db(self, rfq_id: str) -> List[Dict]:
        if not rfq_id:
            return []
        sql = (
            "SELECT supplier_id, price, lead_time, response_text FROM proc.supplier_responses WHERE rfq_id = %s"
        )
        try:
            with self.agent_nick.get_db_connection() as conn:
                df = None
                import pandas as pd

                df = pd.read_sql(sql, conn, params=(rfq_id,))
            return df.to_dict(orient="records") if df is not None else []
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to fetch supplier responses")
            return []

    def _to_native(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

