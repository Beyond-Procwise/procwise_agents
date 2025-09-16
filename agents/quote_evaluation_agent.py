import logging
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

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
            candidate_supplier_ids = self._ensure_sequence(
                input_data.get("supplier_candidates", [])
            )

            raw_supplier_names = self._merge_supplier_tokens(
                ranking_names,
                explicit_suppliers,
            )
            raw_supplier_ids = self._merge_supplier_tokens(
                ranking_ids,
                explicit_supplier_ids,
                candidate_supplier_ids,
            )
            supplier_name_queries = self._expand_supplier_names(raw_supplier_names)
            supplier_id_queries = self._expand_supplier_ids(raw_supplier_ids)
            supplier_tokens = self._merge_supplier_tokens(
                raw_supplier_names,
                raw_supplier_ids,
            )

            retrieval_strategy = "ranked_suppliers"

            db_quotes = self._fetch_quotes_from_database(
                supplier_name_queries,
                supplier_id_queries,
                product_category=product_category,
            )
            vector_quotes = self._fetch_quotes(
                supplier_name_queries,
                supplier_id_queries,
                product_category,
            )
            quotes = self._merge_quote_sources(db_quotes, vector_quotes)
            if db_quotes or vector_quotes:
                logger.info(
                    "QuoteEvaluationAgent: sourced %d database quotes and %d vector quotes",
                    len(db_quotes),
                    len(vector_quotes),
                )

            if not quotes and product_category:
                supplier_filters_present = bool(
                    supplier_name_queries or supplier_id_queries
                )
                if supplier_filters_present:
                    supplier_fallback_db = self._fetch_quotes_from_database(
                        supplier_name_queries,
                        supplier_id_queries,
                        product_category=None,
                    )
                    supplier_fallback_vector = self._fetch_quotes(
                        supplier_name_queries,
                        supplier_id_queries,
                        product_category=None,
                    )
                    supplier_fallback = self._merge_quote_sources(
                        supplier_fallback_db, supplier_fallback_vector
                    )
                    if supplier_fallback:
                        retrieval_strategy = "supplier_fallback"
                        quotes = supplier_fallback
                        logger.info(
                            "QuoteEvaluationAgent: widened search for suppliers %s/%s produced %d quotes",
                            raw_supplier_names,
                            raw_supplier_ids,
                            len(quotes),
                        )
                if not quotes:
                    category_fallback_db = self._fetch_quotes_from_database(
                        [], [], product_category=product_category
                    )
                    category_fallback_vector = self._fetch_quotes(
                        [], [], product_category
                    )
                    category_fallback = self._merge_quote_sources(
                        category_fallback_db, category_fallback_vector
                    )
                    if category_fallback:
                        retrieval_strategy = "category_fallback"
                        quotes = category_fallback
                        logger.info(
                            "QuoteEvaluationAgent: broadened to category '%s' and retrieved %d quotes",
                            product_category,
                            len(quotes),
                        )

            filtered_quotes = quotes

            if supplier_tokens:
                filtered_quotes = self._filter_quotes_by_suppliers(
                    quotes, supplier_tokens
                )
                if filtered_quotes or retrieval_strategy != "category_fallback":
                    quotes = filtered_quotes
                elif not filtered_quotes and retrieval_strategy == "category_fallback":
                    logger.info(
                        "QuoteEvaluationAgent: retaining category fallback quotes after supplier filter removed all results"
                    )

            if ranking_suppliers and retrieval_strategy != "category_fallback":
                quotes = self._order_quotes_by_rank(
                    quotes, ranking_suppliers, limit=3
                )
            logger.info(
                "QuoteEvaluationAgent: fetched %d quotes for category %s using %s",
                len(quotes),
                product_category,
                retrieval_strategy,
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
                        "retrieval_strategy": retrieval_strategy,
                    }
                )
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=empty_payload,
                    pass_fields=empty_payload,
                )

            output_data = self._to_native(
                {
                    "quotes": standardized_quotes,
                    "weights": combined_weight,
                    "retrieval_strategy": retrieval_strategy,
                }
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

    def _fetch_quotes_from_database(
        self,
        supplier_names: List[str],
        supplier_ids: List[str],
        product_category: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch quotes and detailed line information directly from Postgres."""

        connection_factory = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(connection_factory):
            return []

        try:
            with connection_factory() as conn:
                with conn.cursor() as cursor:
                    where_clauses: List[str] = []
                    params: List[Any] = []
                    if supplier_ids:
                        where_clauses.append("q.supplier_id = ANY(%s)")
                        params.append(supplier_ids)
                    if supplier_names:
                        lowered = [name.lower() for name in supplier_names]
                        where_clauses.append(
                            "LOWER(COALESCE(s.supplier_name, q.supplier_id)) = ANY(%s)"
                        )
                        params.append(lowered)

                    where_sql = ""
                    if where_clauses:
                        where_sql = "WHERE " + " OR ".join(f"({clause})" for clause in where_clauses)

                    params.append(50)
                    cursor.execute(
                        f"""
                        SELECT
                            q.quote_id,
                            q.supplier_id,
                            COALESCE(s.supplier_name, q.supplier_id) AS supplier_name,
                            q.buyer_id,
                            q.quote_date,
                            q.validity_date,
                            q.currency,
                            q.total_amount,
                            q.tax_percent,
                            q.tax_amount,
                            q.total_amount_incl_tax,
                            q.po_id,
                            q.country,
                            q.region,
                            q.ai_flag_required,
                            q.trigger_type,
                            q.trigger_context_description,
                            q.created_date,
                            q.created_by,
                            q.last_modified_by,
                            q.last_modified_date
                        FROM proc.quote_agent AS q
                        LEFT JOIN proc.supplier AS s ON s.supplier_id = q.supplier_id
                        {where_sql}
                        ORDER BY q.quote_date DESC NULLS LAST,
                                 q.created_date DESC NULLS LAST,
                                 q.quote_id
                        LIMIT %s
                        """,
                        tuple(params),
                    )
                    quote_rows = cursor.fetchall()
                    description = cursor.description or []
                    quote_columns = [desc[0] for desc in description]

                if not quote_rows:
                    return []

                quote_records = [dict(zip(quote_columns, row)) for row in quote_rows]
                quote_ids = [
                    quote.get("quote_id")
                    for quote in quote_records
                    if quote.get("quote_id")
                ]
                if not quote_ids:
                    return quote_records

                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            quote_id,
                            quote_line_id,
                            line_number,
                            item_id,
                            item_description,
                            quantity,
                            unit_of_measure,
                            unit_price,
                            line_total,
                            tax_percent,
                            tax_amount,
                            total_amount,
                            currency
                        FROM proc.quote_line_items_agent
                        WHERE quote_id = ANY(%s)
                        ORDER BY quote_id, line_number
                        """,
                        (quote_ids,),
                    )
                    line_rows = cursor.fetchall()
                    description = cursor.description or []
                    line_columns = [desc[0] for desc in description]

        except Exception:  # pragma: no cover - database connectivity
            logger.exception("QuoteEvaluationAgent failed to fetch quotes from database")
            return []

        line_map: Dict[Any, List[Dict]] = {}
        for row in line_rows:
            if not line_columns:
                continue
            record = dict(zip(line_columns, row))
            quote_id = record.get("quote_id")
            if quote_id is None:
                continue
            line_map.setdefault(quote_id, []).append(record)

        category_token = None
        if product_category:
            token = str(product_category).strip().lower()
            category_token = token or None

        matched_quotes: List[Dict] = []
        unmatched_quotes: List[Dict] = []

        for quote in quote_records:
            quote_id = quote.get("quote_id")
            items = line_map.get(quote_id, [])

            matched = False

            if category_token:
                matched = any(
                    category_token in str(item.get("item_description", "")).lower()
                    or category_token == str(item.get("item_id", "")).lower()
                    for item in items
                )

            quote["line_items"] = items
            quote["line_items_count"] = len(items)
            quote["category_match"] = bool(matched)

            if items:
                if not quote.get("currency"):
                    currency = next(
                        (item.get("currency") for item in items if item.get("currency")),
                        None,
                    )
                    if currency:
                        quote["currency"] = currency

                totals: List[float] = []
                for item in items:
                    if item.get("line_total") is not None:
                        totals.append(self._to_float(item.get("line_total")))
                    elif item.get("total_amount") is not None:
                        totals.append(self._to_float(item.get("total_amount")))
                if totals:
                    quote["total_line_amount"] = sum(totals)
                    quote.setdefault("total_amount", quote["total_line_amount"])
                    quote.setdefault("total_cost", quote.get("total_amount"))

                unit_prices = [
                    self._to_float(item.get("unit_price"))
                    for item in items
                    if item.get("unit_price") is not None
                ]
                if unit_prices:
                    avg_price = sum(unit_prices) / len(unit_prices)
                    quote["avg_unit_price"] = avg_price
                    quote.setdefault("unit_price", avg_price)

                quote.setdefault("volume", len(items))

            target = matched_quotes if matched else unmatched_quotes
            target.append(quote)

        enriched = matched_quotes + unmatched_quotes

        if category_token:
            logger.debug(
                "QuoteEvaluationAgent: %d of %d quotes matched category token '%s'",
                len(matched_quotes),
                len(enriched),
                category_token,
            )


        logger.debug(
            "QuoteEvaluationAgent: retrieved %d quotes from database", len(enriched)
        )
        return enriched

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

        line_items = quote.get("line_items")
        if not isinstance(line_items, list):
            line_items = []

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
            "quote_id": QuoteEvaluationAgent._coalesce(
                quote, ["quote_id"], None, strip_strings=True
            ),
            "currency": QuoteEvaluationAgent._coalesce(
                quote, ["currency"], None, strip_strings=True
            ),
            "quote_date": QuoteEvaluationAgent._coalesce(
                quote, ["quote_date"], None, strip_strings=False
            ),
            "validity_date": QuoteEvaluationAgent._coalesce(
                quote, ["validity_date"], None, strip_strings=False
            ),
            "line_items": line_items,
            "line_items_count": len(line_items),
            "avg_unit_price": QuoteEvaluationAgent._coalesce(
                quote, ["avg_unit_price", "unit_price"], 0
            ),
            "total_line_amount": QuoteEvaluationAgent._coalesce(
                quote, ["total_line_amount"], 0
            ),
            "category_match": bool(quote.get("category_match")),

            "buyer_id": QuoteEvaluationAgent._coalesce(
                quote, ["buyer_id"], None, strip_strings=True
            ),
            "country": QuoteEvaluationAgent._coalesce(
                quote, ["country"], None, strip_strings=True
            ),
            "region": QuoteEvaluationAgent._coalesce(
                quote, ["region"], None, strip_strings=True
            ),
            "po_id": QuoteEvaluationAgent._coalesce(
                quote, ["po_id"], None, strip_strings=True
            ),
            "tax_percent": QuoteEvaluationAgent._coalesce(
                quote, ["tax_percent"], 0
            ),
            "tax_amount": QuoteEvaluationAgent._coalesce(
                quote, ["tax_amount"], 0
            ),
            "total_amount_incl_tax": QuoteEvaluationAgent._coalesce(
                quote, ["total_amount_incl_tax"], 0
            ),
            "trigger_type": QuoteEvaluationAgent._coalesce(
                quote, ["trigger_type"], None, strip_strings=True
            ),
            "trigger_context_description": QuoteEvaluationAgent._coalesce(
                quote, ["trigger_context_description"], None, strip_strings=True
            ),
            "ai_flag_required": QuoteEvaluationAgent._coalesce(
                quote, ["ai_flag_required"], None, strip_strings=True
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
        if isinstance(obj, Decimal):
            try:
                return float(obj)
            except (TypeError, ValueError):
                return 0.0
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def _merge_quote_sources(
        self, primary: List[Dict], secondary: List[Dict]
    ) -> List[Dict]:
        """Combine quote collections while avoiding duplicate entries."""

        merged: List[Dict] = []
        seen: set[str] = set()

        for quote in primary + secondary:
            quote_id = quote.get("quote_id")
            key: Optional[str]
            if quote_id:
                key = f"id:{str(quote_id).strip().lower()}"
            else:
                supplier_key = self._quote_key(quote)
                cost_basis = quote.get("total_cost") or quote.get("total_amount")
                try:
                    cost_token = f"{float(cost_basis):.6f}"
                except (TypeError, ValueError):
                    cost_token = None
                if supplier_key and cost_token:
                    key = f"supplier:{supplier_key}:{cost_token}"
                elif supplier_key:
                    key = f"supplier:{supplier_key}"
                else:
                    key = None

            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(quote)

        return merged

    def _to_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

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

    def _expand_supplier_names(self, names: List[str]) -> List[str]:
        """Generate query variants for supplier names to improve lookups."""

        expanded: List[str] = []
        seen: Set[str] = set()
        for name in names:
            for variant in self._supplier_query_variants(name):
                key = variant.strip()
                if not key:
                    continue
                lowered = key.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                expanded.append(key)
        return expanded

    def _expand_supplier_ids(self, supplier_ids: List[str]) -> List[str]:
        """Return supplier id variants that cover common case permutations."""

        expanded: List[str] = []
        seen: Set[str] = set()
        for identifier in supplier_ids:
            if identifier is None:
                continue
            text = str(identifier).strip()
            if not text:
                continue
            variants = {text, text.lower(), text.upper()}
            for variant in variants:
                candidate = variant.strip()
                if not candidate:
                    continue
                lowered = candidate.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                expanded.append(candidate)
        return expanded

    @staticmethod
    def _supplier_query_variants(name: Any) -> List[str]:
        """Return expanded supplier name variants for database/vector filters."""

        if name is None:
            return []
        base = str(name).strip()
        if not base:
            return []

        variants: Set[str] = {base}
        ampersand = base.replace("&", " and ")
        variants.add(ampersand)
        normalised_space = re.sub(r"\s+", " ", base)
        if normalised_space:
            variants.add(normalised_space)
        punctuation_smoothed = re.sub(r"[\.,;:/\\\-]+", " ", ampersand)
        punctuation_smoothed = re.sub(r"\s+", " ", punctuation_smoothed).strip()
        if punctuation_smoothed:
            variants.add(punctuation_smoothed)

        final: Set[str] = set()
        for variant in variants:
            if not variant:
                continue
            final.add(variant)
            final.add(variant.lower())
            final.add(variant.upper())
        return [value for value in final if value]

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
    def _supplier_identifier_variants(value: Any) -> Set[str]:
        if value is None:
            return set()
        text = str(value).strip()
        if not text:
            return set()

        lowered = text.lower()
        variants: Set[str] = {lowered}
        collapsed = re.sub(r"[^a-z0-9]+", "", lowered)
        if collapsed:
            variants.add(collapsed)
        spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
        spaced = re.sub(r"\s+", " ", spaced).strip()
        if spaced:
            variants.add(spaced)
        if "&" in lowered:
            replaced = lowered.replace("&", "and")
            variants.add(replaced)
            replaced_spaced = re.sub(r"[^a-z0-9]+", " ", replaced)
            replaced_spaced = re.sub(r"\s+", " ", replaced_spaced).strip()
            if replaced_spaced:
                variants.add(replaced_spaced)
        return {variant for variant in variants if variant}

    @staticmethod
    def _normalize_supplier_identifier(value: Any) -> Optional[str]:
        variants = QuoteEvaluationAgent._supplier_identifier_variants(value)
        if not variants:
            return None
        return sorted(variants)[0]

    @staticmethod
    def _quote_identifier_values(quote: Dict) -> List[Any]:
        identifiers: List[Any] = []
        for key in (
            "supplier_id",
            "supplier_name",
            "name",
            "supplier",
            "vendor",
            "vendor_name",
            "supplier_code",
        ):
            if key in quote:
                identifiers.append(quote.get(key))
        metadata = quote.get("metadata")
        if isinstance(metadata, dict):
            for key in (
                "supplier_id",
                "supplier_name",
                "vendor_name",
                "supplier_code",
                "name",
            ):
                if key in metadata:
                    identifiers.append(metadata.get(key))
        extras = quote.get("identifiers")
        if isinstance(extras, (list, tuple, set)):
            identifiers.extend(extras)
        return identifiers

    def _quote_identifier_variants(self, quote: Dict) -> Set[str]:
        variants: Set[str] = set()
        for identifier in self._quote_identifier_values(quote):
            variants.update(self._supplier_identifier_variants(identifier))
        return variants

    def _filter_quotes_by_suppliers(
        self, quotes: List[Dict], suppliers: List[str]
    ) -> List[Dict]:
        """Restrict quotes to those whose supplier matches the provided list."""

        target_variants: Set[str] = set()
        for supplier in suppliers:
            target_variants.update(self._supplier_identifier_variants(supplier))
        if not target_variants:
            return quotes

        filtered: List[Dict] = []
        for quote in quotes:
            quote_variants = self._quote_identifier_variants(quote)
            if quote_variants & target_variants:
                filtered.append(quote)
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
        seen_suppliers: Set[str] = set()
        for supplier in supplier_order:
            if not isinstance(supplier, dict):
                continue

            candidate_variants: Set[str] = set()
            for key in ("supplier_id", "name", "supplier_name"):
                candidate_variants.update(
                    self._supplier_identifier_variants(supplier.get(key))
                )
            if not candidate_variants:
                continue

            supplier_quotes = []
            for quote in quotes:
                quote_variants = self._quote_identifier_variants(quote)
                if quote_variants & candidate_variants:
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

        if not prioritized:
            return quotes[:limit] if limit else quotes
        if limit:
            return prioritized[:limit]
        return prioritized

    def _quote_key(self, quote: Dict) -> Optional[str]:
        for key in ("supplier_id", "supplier_name", "name"):
            variants = sorted(self._supplier_identifier_variants(quote.get(key)))
            if variants:
                return variants[0]
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

