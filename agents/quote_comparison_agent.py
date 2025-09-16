"""Agent that compares supplier quotes in a normalised structure."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

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
        weight_entry = self._prepare_weight_entry(context.input_data.get("weightings") or context.input_data.get("weights") or {})

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

        merged = quote_lines.merge(quotes, on="quote_id", how="left", suffixes=("_line", ""))
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
        }
        available_aggs = {col: func for col, func in aggregations.items() if col in merged.columns}
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
