"""Agent that compares supplier quotes in a normalised structure."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class QuoteComparisonAgent(BaseAgent):
    """Aggregate quote data for candidate suppliers."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()

    def run(self, context: AgentContext) -> AgentOutput:
        supplier_ids = self._collect_supplier_ids(context.input_data)
        supplier_names = self._collect_supplier_names(context.input_data)
        weight_entry = self._prepare_weight_entry(
            context.input_data.get("weightings")
            or context.input_data.get("weights")
            or {}
        )

        passed_quotes = self._extract_passed_quotes(context.input_data)
        if passed_quotes:
            formatted, has_suppliers = self._build_from_passed_quotes(
                passed_quotes,
                supplier_ids,
                supplier_names,
                weight_entry,
            )
            if formatted is not None and has_suppliers:
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={"comparison": formatted},
                    pass_fields={"comparison": formatted},
                )

        quotes = self._read_table("proc.quote_agent")
        quote_lines = self._read_table("proc.quote_line_items_agent")
        supplier_master = self._read_table("proc.supplier")
        suppliers = (
            supplier_master[["supplier_id", "supplier_name"]]
            if not supplier_master.empty and "supplier_id" in supplier_master.columns
            else pd.DataFrame()
        )

        if quotes.empty or quote_lines.empty:
            result = [weight_entry]
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"comparison": result},
                pass_fields={"comparison": result},
            )

        quotes["quote_id"] = quotes["quote_id"].astype(str)
        quote_lines["quote_id"] = quote_lines["quote_id"].astype(str)

        if supplier_ids:
            quotes = quotes[quotes["supplier_id"].astype(str).isin(supplier_ids)]
            quote_lines = quote_lines[quote_lines["quote_id"].isin(quotes["quote_id"])]

        if quotes.empty:
            result = [weight_entry]
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"comparison": result},
                pass_fields={"comparison": result},
            )

        merged = quote_lines.merge(
            quotes, on="quote_id", how="left", suffixes=("_line", "")
        )
        merged["supplier_id"] = merged["supplier_id"].astype(str)

        numeric_cols = {
            "line_total": "line_total",
            "total_amount": "total_amount",
            "total_amount_incl_tax": "total_amount_incl_tax",
            "unit_price": "unit_price",
            "quantity": "quantity",
        }
        for original, alias in numeric_cols.items():
            if original in merged.columns:
                merged[alias] = pd.to_numeric(merged[original], errors="coerce")

        if "quote_date" in merged.columns:
            merged["quote_date"] = pd.to_datetime(merged["quote_date"], errors="coerce")
        if "validity_date" in merged.columns:
            merged["validity_date"] = pd.to_datetime(merged["validity_date"], errors="coerce")
        if "quote_date" in merged.columns and "validity_date" in merged.columns:
            merged["tenure_days"] = (merged["validity_date"] - merged["quote_date"]).dt.days

        aggregations = {
            "line_total": "sum",
            "total_amount": "sum",
            "total_amount_incl_tax": "sum",
            "unit_price": "mean",
            "quantity": "sum",
            "tenure_days": "mean",
            "quote_id": "nunique",
            "quote_file_s3_path": "first",
        }
        available_aggs = {
            col: func for col, func in aggregations.items() if col in merged.columns
        }
        summary = merged.groupby("supplier_id").agg(available_aggs).reset_index()

        if suppliers is not None and not suppliers.empty:
            suppliers["supplier_id"] = suppliers["supplier_id"].astype(str)
            summary = summary.merge(suppliers, on="supplier_id", how="left")

        results = [weight_entry]
        for _, row in summary.iterrows():
            supplier_id = row.get("supplier_id")
            supplier_name = row.get("supplier_name") or supplier_id
            path_value = row.get("quote_file_s3_path")
            entry = {
                "name": supplier_name,
                "supplier_id": supplier_id,
                "total_spend": float(row.get("line_total", 0.0) or 0.0),
                "total_cost": float(
                    row.get("total_amount_incl_tax")
                    if pd.notna(row.get("total_amount_incl_tax"))
                    else row.get("total_amount", 0.0) or 0.0
                ),
                "unit_price": float(row.get("unit_price", 0.0) or 0.0),
                "quote_file_s3_path": path_value if pd.notna(path_value) else None,
                "tenure": float(row.get("tenure_days", 0.0) or 0.0),
                "volume": float(row.get("quantity", 0.0) or 0.0),
            }
            results.append(entry)

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={"comparison": results},
            pass_fields={"comparison": results},
        )

    def _extract_passed_quotes(self, input_data: Dict) -> List[Dict]:
        for key in ("comparison", "quotes"):
            value = input_data.get(key)
            if isinstance(value, list) and value:
                return value
        return []

    def _collect_supplier_names(self, input_data: Dict) -> Set[str]:
        names: Set[str] = set()
        for key in ("supplier_names", "supplier_name"):
            raw = input_data.get(key)
            names.update(self._normalise_tokens(raw))

        ranking = input_data.get("ranking", [])
        if isinstance(ranking, list):
            for entry in ranking:
                if isinstance(entry, dict):
                    token = entry.get("name") or entry.get("supplier_name")
                    if token:
                        names.add(self._normalise_string(token))

        findings = input_data.get("findings", [])
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, dict):
                    token = finding.get("supplier_name") or finding.get("supplier")
                    if token:
                        names.add(self._normalise_string(token))

        return {name for name in names if name}

    def _normalise_tokens(self, value: Optional[Iterable]) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, (str, int, float, Decimal)):
            return {self._normalise_string(value)}
        tokens: Set[str] = set()
        for item in value:
            normalised = self._normalise_string(item)
            if normalised:
                tokens.add(normalised)
        return tokens

    @staticmethod
    def _normalise_string(value: Optional[object]) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text.lower()

    def _build_from_passed_quotes(
        self,
        quotes: Sequence[Dict],
        supplier_ids: Set[str],
        supplier_names: Set[str],
        weight_entry: Dict,
    ) -> Tuple[Optional[List[Dict]], bool]:
        supplier_id_tokens = {
            token
            for token in (self._normalise_string(v) for v in supplier_ids)
            if token
        }
        supplier_name_tokens = {name for name in supplier_names if name}

        weight_row = None
        supplier_rows: List[Dict] = []
        has_supplier_entries = False

        for entry in quotes:
            if not isinstance(entry, dict):
                continue

            name = str(entry.get("name", "")).strip()
            if name.lower() == "weighting":
                weight_row = self._merge_weight_entries(entry, weight_entry)
                continue

            has_supplier_entries = True
            if not self._entry_matches_suppliers(
                entry, supplier_id_tokens, supplier_name_tokens
            ):
                continue
            supplier_rows.append(self._format_passed_quote(entry))

        if not has_supplier_entries:
            return None, False

        if not supplier_rows:
            return None, True

        weight_row = weight_row or weight_entry
        results = [weight_row]
        results.extend(supplier_rows)
        return results, True

    def _entry_matches_suppliers(
        self,
        entry: Dict,
        supplier_ids: Set[str],
        supplier_names: Set[str],
    ) -> bool:
        if not supplier_ids and not supplier_names:
            return True

        candidates = set()
        for key in ("supplier_id", "quote_id"):
            candidate = entry.get(key)
            if candidate is not None:
                token = self._normalise_string(candidate)
                if token:
                    candidates.add(token)

        for key in ("name", "supplier_name"):
            candidate = entry.get(key)
            if isinstance(candidate, str):
                token = candidate.strip().lower()
                if token:
                    candidates.add(token)

        if supplier_ids and candidates & supplier_ids:
            return True
        if supplier_names and candidates & supplier_names:
            return True
        return False

    def _merge_weight_entries(self, source: Dict, fallback: Dict) -> Dict:
        merged = dict(fallback)
        numeric_keys = ("total_spend", "total_cost", "unit_price", "volume")
        for key in numeric_keys:
            value = source.get(key)
            if not self._is_null(value):
                merged[key] = self._to_float(value)

        if not self._is_null(source.get("tenure")):
            merged["tenure"] = source.get("tenure")

        path = source.get("quote_file_s3_path")
        if isinstance(path, str) and path.strip():
            merged["quote_file_s3_path"] = path.strip()
        elif source.get("quote_file_s3_path") is None:
            merged["quote_file_s3_path"] = None

        return merged

    def _format_passed_quote(self, entry: Dict) -> Dict:
        supplier_id = entry.get("supplier_id")
        supplier_identifier = self._clean_identifier(supplier_id)
        supplier_name = entry.get("name") or entry.get("supplier_name")
        if not supplier_name:
            fallback = supplier_identifier or self._clean_identifier(entry.get("quote_id"))
            supplier_name = f"Supplier {fallback}" if fallback else "Unknown supplier"
        total_cost = self._to_float(
            entry.get("total_cost")
            if not self._is_null(entry.get("total_cost"))
            else entry.get("total_amount")
        )

        quote_path = entry.get("quote_file_s3_path") or entry.get("s3_path")
        if isinstance(quote_path, str):
            quote_path = quote_path.strip() or None
        elif self._is_null(quote_path):
            quote_path = None

        tenure = entry.get("tenure") or entry.get("payment_terms")
        if isinstance(tenure, (int, float, Decimal)) and not self._is_null(tenure):
            tenure_value: Optional[float] = self._to_float(tenure)
        else:
            tenure_value = tenure if tenure not in ("", None) else None

        return {
            "name": supplier_name,
            "supplier_id": supplier_identifier,
            "total_spend": self._to_float(
                entry.get("total_spend")
                if not self._is_null(entry.get("total_spend"))
                else entry.get("total_amount")
            ),
            "total_cost": total_cost,
            "unit_price": self._to_float(
                entry.get("unit_price") if not self._is_null(entry.get("unit_price")) else entry.get("avg_unit_price")
            ),
            "quote_file_s3_path": quote_path,
            "tenure": tenure_value,
            "volume": self._to_float(entry.get("volume") or entry.get("line_items_count")),
        }

    def _clean_identifier(self, value: Optional[object]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _to_float(self, value: Optional[object]) -> float:
        if self._is_null(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                return 0.0

    @staticmethod
    def _is_null(value: Optional[object]) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _collect_supplier_ids(self, input_data: Dict) -> Set[str]:
        supplier_ids: Set[str] = set()
        for key in ("supplier_ids", "supplier_candidates"):
            values = input_data.get(key)
            if isinstance(values, (list, set, tuple)):
                supplier_ids.update(str(v).strip() for v in values if str(v).strip())
            elif values:
                supplier_ids.add(str(values).strip())

        ranking = input_data.get("ranking", [])
        if isinstance(ranking, list):
            for entry in ranking:
                if isinstance(entry, dict):
                    supplier = entry.get("supplier_id")
                    if supplier:
                        supplier_ids.add(str(supplier).strip())

        findings = input_data.get("findings", [])
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, dict):
                    supplier = finding.get("supplier_id")
                    if supplier:
                        supplier_ids.add(str(supplier).strip())
        return supplier_ids

    def _prepare_weight_entry(self, weights: Dict) -> Dict:
        default = {
            "name": "weighting",
            "total_spend": 0.0,
            "total_cost": 0.0,
            "unit_price": 0.0,
            "quote_file_s3_path": None,
            "tenure": None,
            "volume": 0.0,
        }
        if isinstance(weights, (int, float, Decimal)):
            result = default.copy()
            result["total_spend"] = float(weights)
            return result
        if isinstance(weights, str):
            try:
                value = float(weights)
            except (TypeError, ValueError):
                return default
            result = default.copy()
            result["total_spend"] = value
            return result
        if not isinstance(weights, dict):
            return default
        result = default.copy()
        for key in ("total_spend", "total_cost", "unit_price", "volume"):
            value = weights.get(key)
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                pass
        if weights.get("tenure") is not None:
            try:
                result["tenure"] = float(weights.get("tenure"))
            except (TypeError, ValueError):
                result["tenure"] = weights.get("tenure")
        result["quote_file_s3_path"] = weights.get("quote_file_s3_path")
        return result

    def _read_table(self, table: str) -> pd.DataFrame:
        sql = f"SELECT * FROM {table}"
        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        try:
            if callable(pandas_conn):
                with pandas_conn() as conn:
                    return pd.read_sql(sql, conn)
            with self.agent_nick.get_db_connection() as conn:
                return pd.read_sql(sql, conn)
        except Exception:
            logger.exception("QuoteComparisonAgent failed to read %s", table)
            return pd.DataFrame()
