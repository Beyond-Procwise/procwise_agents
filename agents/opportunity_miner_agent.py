from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional,Any

import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """Dataclass representing a single opportunity finding."""

    opportunity_id: str
    detector_type: str
    supplier_id: Optional[str]
    category_id: Optional[str]
    item_id: Optional[str]
    financial_impact_gbp: float
    calculation_details: Dict
    source_records: List[str]
    detected_on: datetime
    weightage: float = 0.0
    candidate_suppliers: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["detected_on"] = self.detected_on.isoformat()
        return d


class OpportunityMinerAgent(BaseAgent):
    """Agent for identifying procurement anomalies and savings opportunities."""

    def __init__(self, agent_nick, min_financial_impact: float = 100.0) -> None:
        super().__init__(agent_nick)
        self.min_financial_impact = min_financial_impact

        # GPU configuration
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)
        logger.info("OpportunityMinerAgent using device: %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        """Entry point for the orchestration layer."""
        return self.process(context)

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------
    def process(self, context: AgentContext) -> AgentOutput:
        try:
            tables = self._ingest_data()
            tables = self._validate_data(tables)
            tables = self._normalise_currency(tables)
            tables = self._apply_index_adjustment(tables)

            findings: List[Finding] = []
            findings.extend(self._detect_unit_price_vs_benchmark(tables))
            findings.extend(self._detect_contract_value_overrun(tables))
            findings.extend(self._detect_po_invoice_discrepancy(tables))
            findings.extend(self._detect_early_payment_discount(tables))
            findings.extend(self._detect_demand_aggregation(tables))
            findings.extend(self._detect_logistics_cost_outliers(tables))
            findings.extend(self._detect_supplier_consolidation(tables))

            filtered = [f for f in findings if f.financial_impact_gbp >= self.min_financial_impact]
            self._load_supplier_risk_map()
            for f in filtered:
                f.candidate_suppliers = self._find_candidate_suppliers(f.item_id, f.supplier_id)

            # compute weightage
            total_impact = sum(f.financial_impact_gbp for f in filtered)
            if total_impact > 0:
                for f in filtered:
                    risk = float(self._supplier_risk_map.get(f.supplier_id, 0.0))
                    f.weightage = (f.financial_impact_gbp / total_impact) * (1.0 + risk)
            else:
                for f in filtered:
                    f.weightage = 0.0

            self._output_excel(filtered)
            self._output_feed(filtered)


            data = {
                "findings": [f.as_dict() for f in filtered],
                "opportunity_count": len(filtered),
                "total_savings": sum(f.financial_impact_gbp for f in filtered),
            }
            # pass candidate supplier IDs to downstream agents
            supplier_candidates = {
                s["supplier_id"]
                for f in filtered
                for s in f.candidate_suppliers
                if s.get("supplier_id")
            }
            data["supplier_candidates"] = list(supplier_candidates)

            return AgentOutput(status=AgentStatus.SUCCESS, data=data, confidence=1.0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("OpportunityMinerAgent error: %s", exc)
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    # ------------------------------------------------------------------
    # Data ingestion and preparation
    # ------------------------------------------------------------------
    # Mapping from internal identifiers to database tables.  Core
    # procurement data lives in the ``proc`` schema as indicated in the
    # requirements: ``proc.invoice_agent``, ``proc.purchase_order_agent``,
    # ``proc.contracts`` and ``proc.supplier``.
    TABLE_MAP = {
        "purchase_orders": "proc.purchase_order_agent",
        "purchase_order_lines": "proc.po_line_items_agent",
        "invoices": "proc.invoice_agent",
        "invoice_lines": "proc.invoice_line_items_agent",
        "contracts": "proc.contracts",
        # "price_benchmarks": "price_benchmarks",
        # "indices": "indices",
        # "shipments": "shipments",
        "supplier_master": "proc.supplier",
    }
    TABLES = list(TABLE_MAP.keys())

    def _ingest_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch required tables from the database.

        The agent now strictly relies on live data from the procurement schema
        so that findings always reflect the most recent state.  Unit tests may
        monkeypatch this method with ``_mock_data`` if needed.
        """

        dfs: Dict[str, pd.DataFrame] = {}
        with self.agent_nick.get_db_connection() as conn:
            for table, sql_name in self.TABLE_MAP.items():
                dfs[table] = pd.read_sql(f"SELECT * FROM {sql_name}", conn)
        return dfs

    def _mock_data(self) -> Dict[str, pd.DataFrame]:
        """Return a minimal in-memory dataset used for unit tests and demos."""

        # All values already in GBP for simplicity
        purchase_orders = pd.DataFrame(
            [
                {
                    "po_id": "PO1",
                    "supplier_id": "S1",
                    "currency": "GBP",
                    "total_amount": 100.0,
                    "payment_terms": "15",
                    "exchange_rate_to_usd": 1.3,
                    "converted_amount_usd": 130.0,
                    "contract_id": "CT1",
                }
            ]
        )

        invoices = pd.DataFrame(
            [
                {
                    "invoice_id": "INV1",
                    "po_id": "PO1",
                    "supplier_id": "S1",
                    "currency": "GBP",
                    "invoice_amount": 110.0,
                    "invoice_total_incl_tax": 110.0,
                    "payment_terms": "15",
                    "exchange_rate_to_usd": 1.3,
                    "converted_amount_usd": 143.0,
                }
            ]
        )

        purchase_order_lines = pd.DataFrame(
            [
                {
                    "po_line_id": "POL1",
                    "po_id": "PO1",
                    "item_id": "IT1",
                    "quantity": 10,
                    "unit_price": 10.0,
                    "line_total": 100.0,
                    "currency": "GBP",
                }
            ]
        )

        invoice_lines = pd.DataFrame(
            [
                {
                    "invoice_line_id": "INVL1",
                    "invoice_id": "INV1",
                    "po_id": "PO1",
                    "item_id": "IT1",
                    "quantity": 10,
                    "unit_price": 11.0,
                    "line_amount": 110.0,
                    "currency": "GBP",
                }
            ]
        )

        contracts = pd.DataFrame(
            [
                {
                    "contract_id": "CT1",
                    "contract_title": "Contract 1",
                    "supplier_id": "S1",
                    "currency": "GBP",
                    "total_contract_value": 100.0,
                    "spend_category": "CatA",
                },
                {
                    "contract_id": "CT2",
                    "contract_title": "Contract 2",
                    "supplier_id": "S2",
                    "currency": "GBP",
                    "total_contract_value": 200.0,
                    "spend_category": "CatA",
                },
            ]
        )

        indices = pd.DataFrame(
            [
                {"index_name": "FX_GBP", "value": 1.0, "effective_date": "2024-01-01", "currency": "GBP"}
            ]
        )

        shipments = pd.DataFrame(
            [
                {
                    "shipment_id": "SH1",
                    "po_id": "PO1",
                    "logistics_cost": 5.0,
                    "currency": "GBP",
                    "delivery_date": "2024-01-10",
                }
            ]
        )

        supplier_master = pd.DataFrame(
            [
                {"supplier_id": "S1", "supplier_name": "Supplier One"},
                {"supplier_id": "S2", "supplier_name": "Supplier Two"},
            ]
        )

        return {
            "purchase_orders": purchase_orders,
            "purchase_order_lines": purchase_order_lines,
            "invoices": invoices,
            "invoice_lines": invoice_lines,
            "contracts": contracts,
            "indices": indices,
            "shipments": shipments,
            "supplier_master": supplier_master,
        }

    def _validate_data(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Basic validation ensuring required columns exist and dropping nulls."""

        for name, df in list(tables.items()):
            if df.empty:
                continue
            # Drop rows that are entirely null
            tables[name] = df.dropna(how="all")
            logger.debug("Table %s columns: %s", name, list(tables[name].columns))
        return tables

    def _normalise_currency(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Convert all monetary values to GBP using simple FX mapping."""

        fx_rates = {"GBP": 1.0}
        indices = tables.get("indices", pd.DataFrame())
        if not indices.empty:
            for _, row in indices.iterrows():
                if row.get("currency") and row.get("value"):
                    fx_rates[row["currency"]] = float(row["value"])

        def convert(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
            if df.empty:
                return df
            currency_col = "currency" if "currency" in df.columns else (
                "default_currency" if "default_currency" in df.columns else None
            )
            if currency_col is None:
                return df
            rate_col = df[currency_col].map(lambda c: fx_rates.get(c, 1.0))
            for col in cols:
                if col in df.columns:
                    df[f"{col}_gbp"] = df[col] * rate_col
            return df

        tables["purchase_orders"] = convert(
            tables.get("purchase_orders", pd.DataFrame()), ["total_amount"]
        )
        tables["purchase_order_lines"] = convert(
            tables.get("purchase_order_lines", pd.DataFrame()),
            ["unit_price", "line_total", "tax_amount", "total_amount"],
        )
        tables["invoices"] = convert(
            tables.get("invoices", pd.DataFrame()), ["invoice_amount", "invoice_total_incl_tax"]
        )
        tables["invoice_lines"] = convert(
            tables.get("invoice_lines", pd.DataFrame()),
            ["unit_price", "line_amount", "tax_amount", "total_amount_incl_tax"],
        )
        tables["contracts"] = convert(tables.get("contracts", pd.DataFrame()), ["total_contract_value"])
        tables["shipments"] = convert(tables.get("shipments", pd.DataFrame()), ["logistics_cost"])
        return tables

    def _apply_index_adjustment(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Placeholder for index-based price adjustment."""

        contracts = tables.get("contracts", pd.DataFrame())
        if not contracts.empty:
            contracts["adjusted_price_gbp"] = contracts.get("agreed_price_gbp", contracts.get("agreed_price", 0.0))
            tables["contracts"] = contracts
        return tables

    # ------------------------------------------------------------------
    # Detector implementations
    # ------------------------------------------------------------------
    def _build_finding(
        self,
        detector: str,
        supplier_id: Optional[str],
        category_id: Optional[str],
        item_id: Optional[str],
        impact: float,
        details: Dict,
        sources: List[str],
    ) -> Finding:
        return Finding(
            opportunity_id=f"{detector}_{len(sources)}_{supplier_id or 'NA'}_{item_id or 'NA'}",
            detector_type=detector,
            supplier_id=supplier_id,
            category_id=category_id,
            item_id=item_id,
            financial_impact_gbp=float(impact),
            calculation_details=details,
            source_records=sources,
            detected_on=datetime.utcnow(),
        )

    def _detect_unit_price_vs_benchmark(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        po = tables.get("purchase_orders", pd.DataFrame())
        bm = tables.get("price_benchmarks", pd.DataFrame())
        required_lines = {"po_id", "item_id", "unit_price_gbp", "quantity"}
        required_po = {"po_id", "supplier_id"}
        required_bm = {"item_id", "benchmark_price_gbp"}
        if (
            po_lines.empty
            or po.empty
            or bm.empty
            or not required_lines.issubset(po_lines.columns)
            or not required_po.issubset(po.columns)
            or not required_bm.issubset(bm.columns)
        ):
            return findings
        merged = (
            po_lines.merge(po[list(required_po)], on="po_id", how="left")
            .merge(bm, on="item_id", suffixes=("", "_bm"))
        )
        merged["variance"] = merged["unit_price_gbp"] - merged["benchmark_price_gbp"]
        cond = merged["variance"] > 0
        for _, row in merged[cond].iterrows():
            savings = row["variance"] * row.get("quantity", 1)
            findings.append(
                self._build_finding(
                    "Unit Price vs Benchmark",
                    row.get("supplier_id"),
                    None,
                    row.get("item_id"),
                    savings,
                    {
                        "unit_price_gbp": row["unit_price_gbp"],
                        "benchmark_price_gbp": row["benchmark_price_gbp"],
                    },
                    [row.get("po_id")],
                )
            )
        return findings

    def _detect_contract_value_overrun(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables.get("purchase_orders", pd.DataFrame())
        inv = tables.get("invoices", pd.DataFrame())
        ct = tables.get("contracts", pd.DataFrame())
        required_po = {"po_id", "contract_id"}
        required_inv = {"po_id", "invoice_amount_gbp"}
        required_ct = {"contract_id", "total_contract_value_gbp", "supplier_id", "spend_category"}
        if (
            po.empty
            or inv.empty
            or ct.empty
            or not required_po.issubset(po.columns)
            or not required_inv.issubset(inv.columns)
            or not required_ct.issubset(ct.columns)
        ):
            return findings
        inv_contract = inv.merge(po[list(required_po)], on="po_id", how="left")
        inv_sum = inv_contract.groupby("contract_id")["invoice_amount_gbp"].sum().reset_index()
        merged = inv_sum.merge(ct, on="contract_id", how="inner")
        merged["variance"] = merged["invoice_amount_gbp"] - merged["total_contract_value_gbp"]
        cond = merged["variance"] > 0
        for _, row in merged[cond].iterrows():
            findings.append(
                self._build_finding(
                    "Contract Value Overrun",
                    row.get("supplier_id"),
                    row.get("spend_category"),
                    None,
                    row["variance"],
                    {
                        "invoice_total_gbp": row["invoice_amount_gbp"],
                        "contract_value_gbp": row["total_contract_value_gbp"],
                    },
                    [row.get("contract_id")],
                )
            )
        return findings

    def _detect_po_invoice_discrepancy(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables.get("purchase_orders", pd.DataFrame())
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        inv_lines = tables.get("invoice_lines", pd.DataFrame())
        required_po = {"po_id", "supplier_id"}
        required_po_lines = {"po_id", "line_total_gbp"}
        required_inv_lines = {"po_id", "line_amount_gbp"}
        if (
            po.empty
            or po_lines.empty
            or inv_lines.empty
            or not required_po.issubset(po.columns)
            or not required_po_lines.issubset(po_lines.columns)
            or not required_inv_lines.issubset(inv_lines.columns)
        ):
            return findings
        po_sum = po_lines.groupby("po_id")["line_total_gbp"].sum().reset_index(name="po_total_gbp")
        inv_sum = inv_lines.groupby("po_id")["line_amount_gbp"].sum().reset_index(name="inv_total_gbp")
        merged = po_sum.merge(inv_sum, on="po_id", how="outer").fillna(0.0)
        merged = merged.merge(po[list(required_po)], on="po_id", how="left")
        merged["amount_diff"] = merged["inv_total_gbp"] - merged["po_total_gbp"]
        cond = merged["amount_diff"] != 0
        for _, row in merged[cond].iterrows():
            impact = row["amount_diff"]
            findings.append(
                self._build_finding(
                    "PO↔Invoice Discrepancy",
                    row.get("supplier_id"),
                    None,
                    None,
                    impact,
                    {"amount_diff": row["amount_diff"]},
                    [row.get("po_id")],
                )
            )
        return findings

    def _detect_early_payment_discount(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        inv = tables.get("invoices", pd.DataFrame())
        required_inv = {"invoice_amount_gbp"}
        if inv.empty or not required_inv.issubset(inv.columns):
            return findings
        # Placeholder: assume a 2% discount could have been taken if payment_terms < 30
        for _, row in inv.iterrows():
            terms = pd.to_numeric(row.get("payment_terms"), errors="coerce")
            if pd.notna(terms) and 0 < terms <= 15:
                try:
                    terms = int(row.get("payment_terms", 0))
                except ValueError:
                    terms = 0
                if terms > 0 and terms <= 15:
                    discount = row["invoice_amount_gbp"] * 0.02
                    findings.append(
                        self._build_finding(
                            "Early Payment Discount Missed",
                            row.get("supplier_id"),
                            None,
                            None,
                            discount,
                            {"terms": float(terms)},
                            [row.get("invoice_id")],
                        )
                    )
        return findings

    def _detect_demand_aggregation(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables.get("purchase_orders", pd.DataFrame())
        required_po = {"supplier_id", "po_id", "total_amount_gbp"}
        if po.empty or not required_po.issubset(po.columns):
            return findings

        grouped = (
            po.groupby("supplier_id")
            .agg(total_spend_gbp=("total_amount_gbp", "sum"), po_ids=("po_id", list))
            .reset_index()
        )
        cond = grouped["total_spend_gbp"] < 500  # small spend could be aggregated
        for _, row in grouped[cond].iterrows():
            savings = row["total_spend_gbp"] * 0.05  # placeholder saving
            findings.append(
                self._build_finding(
                    "Demand Aggregation",
                    row.get("supplier_id"),
                    None,
                    None,
                    savings,
                    {"total_spend_gbp": row["total_spend_gbp"]},
                    list(row.get("po_ids", [])),
                )
            )
        return findings

    def _detect_logistics_cost_outliers(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        shipments = tables.get("shipments", pd.DataFrame())
        required_sh = {"logistics_cost_gbp"}
        if shipments.empty or not required_sh.issubset(shipments.columns):
            return findings
        avg = shipments["logistics_cost_gbp"].mean()
        cond = shipments["logistics_cost_gbp"] > avg * 1.5
        for _, row in shipments[cond].iterrows():
            impact = row["logistics_cost_gbp"] - avg
            findings.append(
                self._build_finding(
                    "Logistics Cost Outliers",
                    None,
                    None,
                    None,
                    impact,
                    {"average_cost": avg, "actual_cost": row["logistics_cost_gbp"]},
                    [row.get("shipment_id"), row.get("po_id")],
                )
            )
        return findings

    def _detect_supplier_consolidation(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        ct = tables.get("contracts", pd.DataFrame())
        required_ct = {"spend_category", "supplier_id", "contract_id"}
        if ct.empty or not required_ct.issubset(ct.columns):
            return findings

        supplier_counts = (
            ct.groupby("spend_category")
            .agg(
                supplier_count=("supplier_id", "nunique"),
                contract_ids=("contract_id", list),
            )
            .reset_index()
        )

        # Determine the top-spend supplier per category so that the resulting
        # opportunity record references a real supplier identifier instead of
        # ``null``.  ``total_contract_value`` is used as a proxy for spend.
        spend = (
            ct.groupby(["spend_category", "supplier_id"])["total_contract_value"]
            .sum()
            .reset_index()
        )

        cond = supplier_counts["supplier_count"] > 1
        for _, row in supplier_counts[cond].iterrows():
            cat = row.get("spend_category")
            savings = row["supplier_count"] * 10.0  # placeholder
            top_row = (
                spend[spend["spend_category"] == cat]
                .sort_values("total_contract_value", ascending=False)
                .head(1)
            )
            supplier_id = top_row["supplier_id"].iloc[0] if not top_row.empty else None
            findings.append(
                self._build_finding(
                    "Supplier Consolidation",
                    supplier_id,
                    cat,
                    None,
                    savings,
                    {"supplier_count": row["supplier_count"]},
                    list(row.get("contract_ids", [])),
                )
            )
        return findings

    def _load_supplier_risk_map(self) -> Dict[str, float]:
        """Load supplier risk scores from `proc.supplier` into a map: supplier_id -> risk_score."""
        self._supplier_risk_map = {}
        try:
            with self.agent_nick.get_db_connection() as conn:
                df = pd.read_sql(
                    "SELECT supplier_id, COALESCE(risk_score, 0.0) AS risk_score FROM proc.supplier",
                    conn,
                )
            if not df.empty:
                self._supplier_risk_map = dict(zip(df["supplier_id"], df["risk_score"]))
        except Exception:
            self._supplier_risk_map = {}
        return self._supplier_risk_map

    def _find_candidate_suppliers(self, item_id: Optional[str], current_supplier_id: Optional[str]) -> List[
        Dict[str, Any]]:
        """
        Query PO lines and invoice lines for other suppliers who sold the same item at a lower unit price.
        Returns a list of dicts: { "supplier_id": "...", "unit_price": 5400.0 }
        """
        if not item_id:
            return []

        sql = """
            SELECT supplier_id, COALESCE(unit_price_gbp, unit_price) AS unit_price
            FROM proc.po_line_items_agent
            WHERE item_id = %s AND supplier_id IS NOT NULL
            UNION ALL
            SELECT supplier_id, COALESCE(unit_price_gbp, unit_price) AS unit_price
            FROM proc.invoice_line_items_agent
            WHERE item_id = %s AND supplier_id IS NOT NULL
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                df = pd.read_sql(sql, conn, params=(item_id, item_id))
        except Exception:
            return []

        if df.empty:
            return []

        df = df.dropna(subset=["supplier_id", "unit_price"])
        if df.empty:
            return []

        # Determine the current supplier's unit price (min if multiple records)
        cur_unit = None
        if current_supplier_id is not None:
            cur_rows = df[df["supplier_id"] == current_supplier_id]["unit_price"]
            if not cur_rows.empty:
                cur_unit = cur_rows.min()

        # If we couldn't find a price for current supplier, set cur_unit high so we still return competitors
        if cur_unit is None:
            cur_unit = df["unit_price"].max() + 1.0

        candidates_df = df[(df["unit_price"] < cur_unit) & (df["supplier_id"] != current_supplier_id)]
        if candidates_df.empty:
            return []

        # For each supplier pick their best (lowest) unit price
        grouped = candidates_df.groupby("supplier_id", as_index=False)["unit_price"].min()
        return [{"supplier_id": r["supplier_id"], "unit_price": float(r["unit_price"])} for _, r in grouped.iterrows()]


    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _output_excel(self, findings: List[Finding]) -> None:
        if not findings:
            return
        df = pd.DataFrame([f.as_dict() for f in findings]).sort_values("financial_impact_gbp", ascending=False)
        with pd.ExcelWriter("opportunity_findings.xlsx") as writer:
            summary = df.groupby("detector_type")["financial_impact_gbp"].sum().reset_index()
            summary.to_excel(writer, sheet_name="summary", index=False)
            for detector, group in df.groupby("detector_type"):
                group.sort_values("financial_impact_gbp", ascending=False).to_excel(
                    writer, sheet_name=detector[:31], index=False
                )

    def _output_feed(self, findings: List[Finding]) -> None:
        path = "opportunity_findings.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([f.as_dict() for f in findings], f, ensure_ascii=False, indent=2)
        logger.info("Wrote %d findings to %s", len(findings), path)

