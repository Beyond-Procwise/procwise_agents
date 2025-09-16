import logging
from typing import Any, Dict, List, Optional

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
            raw_weights = input_data.get(
                "weights", {"price": 0.5, "delivery": 0.3, "risk": 0.2}
            )
            if isinstance(raw_weights, dict):
                weights = dict(raw_weights)
            else:
                weights = {"value": raw_weights}
            combined_weight = self._combine_weight_values(weights)
            weight_metadata = {
                "tenure": weights.get("tenure"),
                "volume": weights.get("volume"),
            }

            ranking_suppliers = self._extract_ranked_suppliers(input_data, limit=3)
            ranking_names = [
                entry["name"] for entry in ranking_suppliers if entry.get("name")
            ]
            ranking_ids = [
                entry["supplier_id"]
                for entry in ranking_suppliers
                if entry.get("supplier_id")
            ]
            explicit_suppliers = self._ensure_sequence(
                input_data.get("supplier_names", [])
            )
            explicit_supplier_ids = self._ensure_sequence(
                input_data.get("supplier_ids", [])
            )

            supplier_names_filter = ranking_names or explicit_suppliers
            supplier_ids_filter = ranking_ids or explicit_supplier_ids
            supplier_tokens = self._merge_supplier_tokens(
                supplier_names_filter, supplier_ids_filter
            )

            quotes = self._fetch_quotes(
                supplier_names_filter,
                supplier_ids_filter,
                product_category,
            )
            if supplier_tokens:
                quotes = self._filter_quotes_by_suppliers(quotes, supplier_tokens)
            if ranking_suppliers:
                quotes = self._order_quotes_by_rank(
                    quotes, ranking_suppliers, limit=3
                )
            logger.info(
                "QuoteEvaluationAgent: fetched %d quotes for category %s",
                len(quotes),
                product_category,
            )

            standardized_quotes = self._standardize_quotes(
                quotes, combined_weight, weight_metadata
            )

            if not quotes:
                # Absence of quotes should not be treated as a hard failure:
                # downstream agents may still proceed with an empty result set.
                empty_payload = self._to_native(
                    {
                        "quotes": standardized_quotes,
                        "weights": combined_weight,
                        "message": "No quotes found",
                    }
                )
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=empty_payload,
                    pass_fields=empty_payload,
                )

            output_data = self._to_native(
                {"quotes": standardized_quotes, "weights": combined_weight}
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
        supplier_ids: Optional[List[str]] = None,
        product_category: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch quotes from Qdrant vector database."""

        supplier_ids = supplier_ids or []

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
        if supplier_ids:
            ensure_index("supplier_id")
        if product_category:
            ensure_index("product_category")

        def build_filter(include_supplier: bool = True) -> models.Filter:
            filters = [
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value="quote"),
                )
            ]
            if include_supplier and (supplier_names or supplier_ids):
                supplier_conditions: List[models.FieldCondition] = []
                if supplier_names:
                    supplier_conditions.append(
                        models.FieldCondition(
                            key="supplier_name",
                            match=models.MatchAny(any=supplier_names),
                        )
                    )
                if supplier_ids:
                    supplier_conditions.append(
                        models.FieldCondition(
                            key="supplier_id",
                            match=models.MatchAny(any=supplier_ids),
                        )
                    )
                if len(supplier_conditions) == 1:
                    filters.append(supplier_conditions[0])
                elif supplier_conditions:
                    filters.append(
                        models.Filter(
                            should=supplier_conditions,
                            must=[],
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
            if not points and (supplier_names or supplier_ids):
                logger.info(
                    "QuoteEvaluationAgent: retrying quote fetch without supplier filter",
                )
                points, _ = qdrant_scroll(build_filter(include_supplier=False))
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
                    "supplier_id": payload.get("supplier_id"),
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
        self,
        quotes: List[Dict],
        combined_weight: float,
        weight_metadata: Dict,
    ) -> List[Dict]:
        """Produce the standard quote comparison output structure."""

        standardized: List[Dict] = [
            self._build_weighting_entry(combined_weight, weight_metadata)
        ]
        for quote in quotes:
            standardized.append(self._standardize_quote(quote))
        return standardized

    @staticmethod
    def _build_weighting_entry(
        combined_weight: float, metadata: Optional[Dict]
    ) -> Dict:
        """Create the leading weighting row for the comparison table."""

        meta = metadata or {}
        return {
            "name": "weighting",
            "total_spend": combined_weight,
            "total_cost": 0,
            "unit_price": 0,
            "quote_file_s3_path": None,
            "tenure": meta.get("tenure"),
            "volume": meta.get("volume", 0),
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
            "supplier_id": QuoteEvaluationAgent._coalesce(
                quote, ["supplier_id"], None, strip_strings=True
            ),
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

    @staticmethod
    def _combine_weight_values(weights: Dict) -> float:
        """Aggregate multi-factor weights into a single numeric value."""

        if not isinstance(weights, dict):
            try:
                return float(weights)
            except (TypeError, ValueError):
                return 0.0

        total = 0.0
        for key in ("price", "delivery", "risk", "value"):
            value = weights.get(key)
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue

        if total:
            return total

        numeric_values: List[float] = []
        for key, value in weights.items():
            if key in {"tenure", "volume"}:
                continue
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        return sum(numeric_values)

    @staticmethod
    def _ensure_sequence(value: Any) -> List[str]:
        """Normalise supplier inputs into a clean list of strings."""

        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            sequence = value
        else:
            sequence = [value]
        normalised: List[str] = []
        for item in sequence:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalised.append(text)
        return normalised

    def _merge_supplier_tokens(self, *sequences: List[str]) -> List[str]:
        """Combine supplier identifiers while preserving order and removing duplicates."""

        merged: List[str] = []
        seen = set()
        for sequence in sequences:
            if not sequence:
                continue
            for item in sequence:
                if item is None:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                normalised = self._normalize_supplier_identifier(text)
                if not normalised or normalised in seen:
                    continue
                seen.add(normalised)
                merged.append(text)
        return merged

    @staticmethod
    def _extract_ranked_suppliers(
        input_data: Dict, limit: int = 3
    ) -> List[Dict[str, Optional[str]]]:
        """Return ordered supplier records from ranking payloads."""

        ranking = input_data.get("ranking")
        if not ranking:
            return []
        if isinstance(ranking, dict):
            ranking_list = ranking.get("ranking")
            if isinstance(ranking_list, list):
                ranking = ranking_list
            else:
                return []
        if not isinstance(ranking, list):
            return []

        suppliers: List[Dict[str, Optional[str]]] = []
        for entry in ranking:
            if not isinstance(entry, dict):
                continue
            supplier_name = None
            for key in ("supplier_name", "name", "supplier"):
                candidate = entry.get(key)
                if candidate and str(candidate).strip():
                    supplier_name = str(candidate).strip()
                    break
            supplier_id = None
            for key in ("supplier_id", "id", "supplier_code"):
                candidate = entry.get(key)
                if candidate and str(candidate).strip():
                    supplier_id = str(candidate).strip()
                    break
            if not supplier_name and not supplier_id:
                continue
            suppliers.append({"name": supplier_name, "supplier_id": supplier_id})
            if limit and len(suppliers) >= limit:
                break
        return suppliers

    @staticmethod
    def _normalize_supplier_identifier(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    def _filter_quotes_by_suppliers(
        self, quotes: List[Dict], suppliers: List[str]
    ) -> List[Dict]:
        """Restrict quotes to those whose supplier matches the provided list."""

        normalized_targets = {
            token
            for supplier in suppliers
            if (token := self._normalize_supplier_identifier(supplier))
        }
        if not normalized_targets:
            return quotes

        filtered: List[Dict] = []
        for quote in quotes:
            identifiers = (
                quote.get("supplier_name"),
                quote.get("supplier_id"),
                quote.get("name"),
            )
            for identifier in identifiers:
                norm_identifier = self._normalize_supplier_identifier(identifier)
                if norm_identifier and norm_identifier in normalized_targets:
                    filtered.append(quote)
                    break
        return filtered

    def _order_quotes_by_rank(
        self,
        quotes: List[Dict],
        supplier_order: List[Dict[str, Optional[str]]],
        limit: int = 3,
    ) -> List[Dict]:
        """Order and limit quotes based on supplier ranking."""

        if not supplier_order:
            return quotes[:limit] if limit else quotes

        prioritized: List[Dict] = []
        seen_suppliers = set()
        for supplier in supplier_order:
            if not isinstance(supplier, dict):
                continue

            candidate_tokens = [
                token
                for key in ("supplier_id", "name")
                if (token := self._normalize_supplier_identifier(supplier.get(key)))
            ]
            if not candidate_tokens:
                continue

            supplier_quotes = []
            for quote in quotes:
                identifiers = [
                    self._normalize_supplier_identifier(quote.get("supplier_id")),
                    self._normalize_supplier_identifier(quote.get("supplier_name")),
                    self._normalize_supplier_identifier(quote.get("name")),
                ]
                if any(token and token in candidate_tokens for token in identifiers):
                    supplier_quotes.append(quote)

            if not supplier_quotes:
                continue

            supplier_quotes.sort(key=self._quote_priority)
            chosen = supplier_quotes[0]
            quote_key = self._quote_key(chosen)
            if quote_key and quote_key in seen_suppliers:
                continue
            if quote_key:
                seen_suppliers.add(quote_key)
            prioritized.append(chosen)
            if limit and len(prioritized) >= limit:
                break

        if limit:
            return prioritized[:limit]
        return prioritized

    def _quote_key(self, quote: Dict) -> Optional[str]:
        for key in ("supplier_id", "supplier_name", "name"):
            identifier = self._normalize_supplier_identifier(quote.get(key))
            if identifier:
                return identifier
        return None

    @staticmethod
    def _quote_priority(quote: Dict) -> float:
        """Provide a deterministic ordering for supplier quotes."""

        for key in ("total_cost", "total_spend", "unit_price"):
            value = quote.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float("inf")

