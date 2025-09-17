"""Agent that compares supplier quotes in a normalised structure."""

from __future__ import annotations

import logging
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class QuoteComparisonAgent(BaseAgent):
    """Aggregate quote data for candidate suppliers."""

    DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
        "total_cost": 0.6,
        "tenure": 0.25,
        "volume": 0.15,
    }
    METRIC_CONFIG: Dict[str, Dict[str, str]] = {
        "total_cost": {"key": "total_cost_gbp", "direction": "lower", "label": "total cost"},
        "tenure": {"key": "tenure", "direction": "lower", "label": "lead time"},
        "volume": {"key": "volume", "direction": "higher", "label": "volume"},
    }
    EPSILON = 1e-9

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self._fx_to_gbp: Dict[str, float] = {"GBP": 1.0, "USD": 0.79}
        self._resolved_metric_weights: Dict[str, float] = {}

    def run(self, context: AgentContext) -> AgentOutput:
        supplier_ids = self._collect_supplier_ids(context.input_data)
        supplier_names = self._collect_supplier_names(context.input_data)
        weight_entry = self._prepare_weight_entry(
            context.input_data.get("weightings")
            or context.input_data.get("weights")
            or {}
        )
        metric_weights = (
            dict(self._resolved_metric_weights)
            if self._resolved_metric_weights
            else dict(self.DEFAULT_METRIC_WEIGHTS)
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
                finalised, recommended = self._finalise_results(formatted, metric_weights)
                recommended_summary = self._build_recommended_summary(recommended)
                payload = {"comparison": finalised}
                if recommended_summary:
                    payload["recommended_quote"] = recommended_summary
                return AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=payload,
                    pass_fields=payload,
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
            results, recommended = self._finalise_results([weight_entry], metric_weights)
            recommended_summary = self._build_recommended_summary(recommended)
            payload = {"comparison": results}
            if recommended_summary:
                payload["recommended_quote"] = recommended_summary
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=payload,
                pass_fields=payload,
            )

        quotes["quote_id"] = quotes["quote_id"].astype(str)
        quote_lines["quote_id"] = quote_lines["quote_id"].astype(str)

        if supplier_ids:
            quotes = quotes[quotes["supplier_id"].astype(str).isin(supplier_ids)]
            quote_lines = quote_lines[quote_lines["quote_id"].isin(quotes["quote_id"])]

        if quotes.empty:
            results, recommended = self._finalise_results([weight_entry], metric_weights)
            recommended_summary = self._build_recommended_summary(recommended)
            payload = {"comparison": results}
            if recommended_summary:
                payload["recommended_quote"] = recommended_summary
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=payload,
                pass_fields=payload,
            )

        merged = quote_lines.merge(
            quotes, on="quote_id", how="left", suffixes=("_line", "")
        )
        merged["supplier_id"] = merged["supplier_id"].astype(str)

        numeric_cols = {
            "line_total": "line_total",
            "total_amount": "total_amount",
            "total_amount_incl_tax": "total_amount_incl_tax",
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
            "quantity": "sum",
            "tenure_days": "mean",
            "quote_id": "nunique",
            "quote_file_s3_path": "first",
        }
        if "currency" in merged.columns:
            aggregations["currency"] = "first"
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
                "total_spend": self._to_float(row.get("line_total")),
                "total_cost": self._to_float(
                    row.get("total_amount_incl_tax")
                    if pd.notna(row.get("total_amount_incl_tax"))
                    else row.get("total_amount")
                ),
                "quote_file_s3_path": path_value if pd.notna(path_value) else None,
                "tenure": self._to_float(row.get("tenure_days")) if pd.notna(row.get("tenure_days")) else None,
                "volume": self._to_float(row.get("quantity")),
                "currency": row.get("currency") if pd.notna(row.get("currency")) else None,
            }
            results.append(entry)

        finalised_results, recommended = self._finalise_results(results, metric_weights)
        recommended_summary = self._build_recommended_summary(recommended)
        payload = {"comparison": finalised_results}
        if recommended_summary:
            payload["recommended_quote"] = recommended_summary

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=payload,
            pass_fields=payload,
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
        numeric_keys = ("total_spend", "total_cost", "volume")
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

        if source.get("weighting_factors"):
            merged["weighting_factors"] = dict(source["weighting_factors"])
        elif fallback.get("weighting_factors"):
            merged["weighting_factors"] = dict(fallback["weighting_factors"])

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

        tenure_value = self._parse_tenure(entry.get("tenure") or entry.get("payment_terms"))

        return {
            "name": supplier_name,
            "supplier_id": supplier_identifier,
            "total_spend": self._to_float(
                entry.get("total_spend")
                if not self._is_null(entry.get("total_spend"))
                else entry.get("total_amount")
            ),
            "total_cost": total_cost,
            "quote_file_s3_path": quote_path,
            "tenure": tenure_value,
            "volume": self._to_float(entry.get("volume") or entry.get("line_items_count")),
            "currency": entry.get("currency") or entry.get("currency_code"),
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
        entry = {
            "name": "weighting",
            "total_spend": 1.0,
            "total_cost": 0.0,
            "quote_file_s3_path": None,
            "tenure": 0.0,
            "volume": 0.0,
            "currency": "GBP",
        }
        metric_weights = self._extract_metric_weights(weights)
        entry["total_cost"] = metric_weights.get("total_cost", 0.0)
        entry["tenure"] = metric_weights.get("tenure", 0.0)
        entry["volume"] = metric_weights.get("volume", 0.0)
        entry["weighting_factors"] = metric_weights
        return entry

    def _extract_metric_weights(self, weights: Any) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        if isinstance(weights, (int, float, Decimal)):
            value = float(weights)
            if value > 0:
                raw["total_cost"] = value
        elif isinstance(weights, str):
            try:
                value = float(weights)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                raw["total_cost"] = value
        elif isinstance(weights, dict):
            for key in self.METRIC_CONFIG.keys():
                value = weights.get(key)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    raw[key] = numeric

        if not raw:
            raw = dict(self.DEFAULT_METRIC_WEIGHTS)

        total = sum(raw.values())
        if total <= 0:
            total = sum(self.DEFAULT_METRIC_WEIGHTS.values())
            raw = dict(self.DEFAULT_METRIC_WEIGHTS)

        normalised = {metric: value / total for metric, value in raw.items() if value > 0}
        self._resolved_metric_weights = normalised
        return normalised

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

    def _load_fx_rates(self) -> None:
        if self._fx_to_gbp:
            return

        rates: Dict[str, float] = {"GBP": 1.0, "USD": 0.79}
        for table in ("proc.currency_rates", "proc.fx_rates", "proc.indices"):
            df = self._read_table(table)
            if df.empty:
                continue
            currency_col = next(
                (col for col in ("currency", "from_currency", "code") if col in df.columns),
                None,
            )
            value_col = next(
                (col for col in ("rate_to_gbp", "gbp_rate", "value", "rate") if col in df.columns),
                None,
            )
            if not currency_col or not value_col:
                continue
            for _, row in df.iterrows():
                currency = row.get(currency_col)
                if not currency:
                    continue
                try:
                    rate = float(row.get(value_col))
                except (TypeError, ValueError):
                    continue
                if rate and rate > 0:
                    rates[str(currency).strip().upper()] = rate
        self._fx_to_gbp = rates

    def _rate_to_gbp(self, currency: Optional[str]) -> float:
        self._load_fx_rates()
        if not currency:
            return 1.0
        return self._fx_to_gbp.get(str(currency).strip().upper(), 1.0)

    def _augment_currency_fields(self, entry: Dict[str, Any]) -> None:
        if entry.get("name", "").lower() == "weighting":
            entry.setdefault("currency", "GBP")
            entry["total_cost_gbp"] = self._to_float(entry.get("total_cost"))
            entry["total_cost_usd"] = self._to_float(entry.get("total_cost"))
            entry["total_spend_gbp"] = self._to_float(entry.get("total_spend"))
            entry["total_spend_usd"] = self._to_float(entry.get("total_spend"))
            return

        currency_code = (
            str(entry.get("currency") or entry.get("currency_code") or "GBP")
            .strip()
            .upper()
        )
        rate = self._rate_to_gbp(currency_code)
        usd_rate = self._rate_to_gbp("USD")
        entry["currency"] = currency_code

        total_cost = self._to_float(entry.get("total_cost"))
        total_spend = self._to_float(entry.get("total_spend"))

        entry["total_cost_gbp"] = total_cost * rate
        entry["total_spend_gbp"] = total_spend * rate

        if usd_rate > 0:
            entry["total_cost_usd"] = entry["total_cost_gbp"] / usd_rate
            entry["total_spend_usd"] = entry["total_spend_gbp"] / usd_rate
        else:
            entry["total_cost_usd"] = total_cost
            entry["total_spend_usd"] = total_spend

    def _round_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._round_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._round_value(item) for item in value]
        if self._is_null(value):
            if isinstance(value, float) and math.isnan(value):
                return 0.0
            return None if value is not None else value
        if isinstance(value, Decimal):
            value = float(value)
        if isinstance(value, (int, float)):
            try:
                decimal_value = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                rounded = float(decimal_value)
            except (ArithmeticError, ValueError, TypeError):
                rounded = round(float(value), 2)
            if math.isclose(rounded, round(rounded)):
                return int(round(rounded))
            return rounded
        return value

    def _round_entry_values(self, entry: Dict[str, Any]) -> None:
        for key, value in list(entry.items()):
            entry[key] = self._round_value(value)

    def _calculate_weighting_scores(
        self, entries: List[Dict[str, Any]], weights: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        if not entries:
            return stats

        for metric, cfg in self.METRIC_CONFIG.items():
            values: List[float] = []
            for entry in entries:
                value = entry.get(cfg["key"])
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                values.append(numeric)
            if values:
                stats[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "direction": cfg["direction"],
                }

        for entry in entries:
            score_total = 0.0
            weight_total = 0.0
            breakdown: Dict[str, float] = {}
            for metric, cfg in self.METRIC_CONFIG.items():
                weight = weights.get(metric, 0.0)
                if weight <= 0:
                    continue
                stat = stats.get(metric)
                if not stat:
                    continue
                value = entry.get(cfg["key"])
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                min_val = stat["min"]
                max_val = stat["max"]
                if math.isclose(max_val, min_val, rel_tol=self.EPSILON, abs_tol=self.EPSILON):
                    normalised = 1.0
                else:
                    if stat["direction"] == "lower":
                        normalised = (max_val - numeric) / (max_val - min_val)
                    else:
                        normalised = (numeric - min_val) / (max_val - min_val)
                    normalised = max(0.0, min(1.0, normalised))
                breakdown[metric] = normalised
                score_total += normalised * weight
                weight_total += weight

            if weight_total > 0:
                entry["weighting_score"] = (score_total / weight_total) * 100.0
            else:
                entry["weighting_score"] = 0.0
            if breakdown:
                entry["weighting_breakdown"] = {
                    key: round(value * 100.0, 2) for key, value in breakdown.items()
                }

        return stats

    def _apply_recommendations(
        self,
        entries: List[Dict[str, Any]],
        stats: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        if not entries:
            return None

        best = max(
            entries,
            key=lambda e: (
                e.get("weighting_score", 0.0),
                -self._to_float(e.get("total_cost_gbp")),
            ),
        )
        best["recommendation"] = {
            "ticker": "RECOMMENDED",
            "justification": self._build_recommendation_reason(best, stats),
        }
        best_score = best.get("weighting_score", 0.0)
        best_cost = self._to_float(best.get("total_cost_gbp"))

        for entry in entries:
            if entry is best:
                continue
            reasons: List[str] = []
            score_diff = best_score - entry.get("weighting_score", 0.0)
            if score_diff > self.EPSILON:
                reasons.append(f"Score {score_diff:.1f} below best option")
            cost_diff = self._to_float(entry.get("total_cost_gbp")) - best_cost
            if cost_diff > self.EPSILON:
                reasons.append(
                    f"{self._format_currency(cost_diff, 'GBP')} higher total cost"
                )
            entry["recommendation"] = {
                "ticker": "ALTERNATIVE",
                "justification": ", ".join(reasons)
                if reasons
                else "Viable alternative with lower composite score",
            }

        return best

    def _build_recommendation_reason(
        self, entry: Dict[str, Any], stats: Dict[str, Dict[str, float]]
    ) -> str:
        reasons: List[str] = []
        for metric, cfg in self.METRIC_CONFIG.items():
            stat = stats.get(metric)
            if not stat:
                continue
            value = entry.get(cfg["key"])
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            target = stat["min"] if cfg["direction"] == "lower" else stat["max"]
            if math.isclose(numeric, target, rel_tol=self.EPSILON, abs_tol=self.EPSILON):
                if metric == "total_cost":
                    reasons.append(
                        f"Lowest {cfg['label']} ({self._format_currency(numeric, 'GBP')})"
                    )
                elif metric == "tenure":
                    reasons.append(f"Shortest lead time ({numeric:.0f} days)")
                else:
                    reasons.append(f"Highest {cfg['label']}")
        if not reasons:
            reasons.append(
                f"Highest composite score {entry.get('weighting_score', 0.0):.1f}"
            )
        reasons.append("Recommendation generated from weighted quote evaluation.")
        return "; ".join(reasons)

    def _finalise_results(
        self, rows: List[Dict[str, Any]], metric_weights: Dict[str, float]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not rows:
            return rows, None
        for entry in rows:
            self._augment_currency_fields(entry)
        supplier_entries = [entry for entry in rows if entry.get("name") != "weighting"]
        stats = self._calculate_weighting_scores(supplier_entries, metric_weights)
        recommended = self._apply_recommendations(supplier_entries, stats, metric_weights)
        for entry in rows:
            self._round_entry_values(entry)
        return rows, recommended

    def _build_recommended_summary(
        self, recommended_entry: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not recommended_entry:
            return None
        recommendation = recommended_entry.get("recommendation", {})
        summary = {
            "supplier_id": recommended_entry.get("supplier_id"),
            "name": recommended_entry.get("name"),
            "weighting_score": recommended_entry.get("weighting_score"),
            "total_cost_gbp": recommended_entry.get("total_cost_gbp"),
            "total_cost_usd": recommended_entry.get("total_cost_usd"),
            "ticker": recommendation.get("ticker"),
            "justification": recommendation.get("justification"),
        }
        return {key: self._round_value(value) for key, value in summary.items()}

    def _format_currency(self, amount: float, currency: Optional[str]) -> str:
        code = (currency or "GBP").upper()
        symbol = "£" if code == "GBP" else "$" if code == "USD" else ""
        if symbol:
            return f"{symbol}{amount:,.2f}"
        return f"{amount:,.2f} {code}"

    def _parse_tenure(self, value: Any) -> Optional[float]:
        if value is None or self._is_null(value):
            return None
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
