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
            logger.info(
                "OpportunityMinerAgent starting processing with input %s",
                context.input_data,
            )
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
            findings.extend(self._detect_supplier_price_difference(tables))

            filtered = [f for f in findings if f.financial_impact_gbp >= self.min_financial_impact]
            self._load_supplier_risk_map()
            for f in filtered:
                f.candidate_suppliers = self._find_candidate_suppliers(
                    f.item_id, f.supplier_id, f.source_records
                )

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
            logger.info(
                "OpportunityMinerAgent produced %d findings and %d candidate suppliers",
                len(filtered),
                len(supplier_candidates),
            )
            logger.debug("OpportunityMinerAgent findings: %s", data["findings"])
            logger.info("OpportunityMinerAgent finishing processing")

            return AgentOutput(
                status=AgentStatus.SUCCESS,
                data=data,
                pass_fields=data,
                confidence=1.0,
            )
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
        """Create a :class:`Finding` ensuring NaNs are normalised."""

        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            return str(value)

        supplier_id = _clean(supplier_id)
        category_id = _clean(category_id)
        item_id = _clean(item_id)
        if impact is None or (isinstance(impact, float) and pd.isna(impact)):
            impact = 0.0

        finding = Finding(
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

        logger.debug(
            "Built finding %s for supplier %s, category %s, item %s with impact %s",
            finding.opportunity_id,
            supplier_id,
            category_id,
            item_id,
            impact,
        )
        return finding

    def _detect_unit_price_vs_benchmark(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        po = tables.get("purchase_orders", pd.DataFrame())
        bm = tables.get("price_benchmarks", pd.DataFrame())

        unit_col = "unit_price_gbp" if "unit_price_gbp" in po_lines.columns else "unit_price"
        bm_price_col = (
            "benchmark_price_gbp" if "benchmark_price_gbp" in bm.columns else "benchmark_price"
        )

        required_lines = {"po_id", "item_id", unit_col, "quantity"}
        required_po = {"po_id", "supplier_id"}
        required_bm = {"item_id", bm_price_col}
        if (
            po_lines.empty
            or po.empty
            or bm.empty
            or not required_lines.issubset(po_lines.columns)
            or not required_po.issubset(po.columns)
            or not required_bm.issubset(bm.columns)
        ):
            return findings

        logger.debug(
            "_detect_unit_price_vs_benchmark using columns: unit_col=%s, bm_price_col=%s",
            unit_col,
            bm_price_col,
        )

        merged = (
            po_lines.merge(po[list(required_po)], on="po_id", how="left")
            .merge(bm, on="item_id", suffixes=("", "_bm"))
        )
        merged["variance"] = merged[unit_col] - merged[bm_price_col]
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
                    {"unit_price": row[unit_col], "benchmark_price": row[bm_price_col]},
                    [row.get("po_id")],
                )
            )
        return findings

    def _detect_contract_value_overrun(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables.get("purchase_orders", pd.DataFrame())
        inv = tables.get("invoices", pd.DataFrame())
        ct = tables.get("contracts", pd.DataFrame())
        invoice_col = (
            "invoice_amount_gbp" if "invoice_amount_gbp" in inv.columns else "invoice_amount"
        )
        tcv_col = (
            "total_contract_value_gbp"
            if "total_contract_value_gbp" in ct.columns
            else "total_contract_value"
        )
        category_col = "category_id" if "category_id" in ct.columns else "spend_category"

        required_po = {"po_id", "contract_id"}
        required_inv = {"po_id", invoice_col}
        required_ct = {"contract_id", tcv_col, "supplier_id", category_col}
        if (
            po.empty
            or inv.empty
            or ct.empty
            or not required_po.issubset(po.columns)
            or not required_inv.issubset(inv.columns)
            or not required_ct.issubset(ct.columns)
        ):
            return findings

        logger.debug(
            "_detect_contract_value_overrun using invoice_col=%s, contract_value_col=%s, category_col=%s",
            invoice_col,
            tcv_col,
            category_col,
        )

        inv_contract = inv.merge(po[list(required_po)], on="po_id", how="left")
        inv_sum = (
            inv_contract.groupby("contract_id")[invoice_col].sum().reset_index(name=invoice_col)
        )
        merged = inv_sum.merge(ct, on="contract_id", how="inner")
        merged["variance"] = merged[invoice_col] - merged[tcv_col]
        cond = merged["variance"] > 0
        for _, row in merged[cond].iterrows():
            findings.append(
                self._build_finding(
                    "Contract Value Overrun",
                    row.get("supplier_id"),
                    row.get(category_col),
                    None,
                    row["variance"],
                    {"invoice_total": row[invoice_col], "contract_value": row[tcv_col]},
                    [row.get("contract_id")],
                )
            )
        return findings

    def _detect_po_invoice_discrepancy(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables.get("purchase_orders", pd.DataFrame())
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        inv_lines = tables.get("invoice_lines", pd.DataFrame())

        category_col = (
            "category_id"
            if "category_id" in po.columns
            else ("spend_category" if "spend_category" in po.columns else None)
        )
        required_po = {"po_id", "supplier_id"}
        if category_col:
            required_po.add(category_col)

        line_total_col = (
            "line_total_gbp" if "line_total_gbp" in po_lines.columns else "line_total"
        )
        line_amount_col = (
            "line_amount_gbp" if "line_amount_gbp" in inv_lines.columns else "line_amount"
        )
        required_po_lines = {"po_id", "item_id", line_total_col}
        required_inv_lines = {"po_id", "item_id", line_amount_col}
        if (
            po.empty
            or po_lines.empty
            or inv_lines.empty
            or not required_po.issubset(po.columns)
            or not required_po_lines.issubset(po_lines.columns)
            or not required_inv_lines.issubset(inv_lines.columns)
        ):
            return findings

        logger.debug(
            "_detect_po_invoice_discrepancy using line_total_col=%s, line_amount_col=%s",
            line_total_col,
            line_amount_col,
        )

        # Aggregate totals at ``po_id`` + ``item_id`` granularity so that each
        # finding can reference the affected item and supplier.  When purchase
        # orders span multiple categories the ``category_id`` (or
        # ``spend_category``) is derived from the parent PO record.
        po_sum = (
            po_lines.groupby(["po_id", "item_id"])[line_total_col]
            .sum()
            .reset_index(name="po_total")
        )
        inv_sum = (
            inv_lines.groupby(["po_id", "item_id"])[line_amount_col]
            .sum()
            .reset_index(name="inv_total")
        )
        merged = (
            po_sum.merge(inv_sum, on=["po_id", "item_id"], how="outer")
            .fillna(0.0)
            .merge(po[list(required_po)], on="po_id", how="left")
        )
        merged["amount_diff"] = merged["inv_total"] - merged["po_total"]
        cond = merged["amount_diff"] != 0
        for _, row in merged[cond].iterrows():
            impact = row["amount_diff"]
            findings.append(
                self._build_finding(
                    "PO↔Invoice Discrepancy",
                    row.get("supplier_id"),
                    row.get(category_col) if category_col else None,
                    row.get("item_id"),
                    impact,
                    {
                        "amount_diff": row["amount_diff"],
                        "po_total": row["po_total"],
                        "inv_total": row["inv_total"],
                    },
                    [row.get("po_id")],
                )
            )
        return findings

    def _detect_early_payment_discount(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        inv = tables.get("invoices", pd.DataFrame())
        invoice_col = (
            "invoice_amount_gbp" if "invoice_amount_gbp" in inv.columns else "invoice_amount"
        )
        required_inv = {invoice_col}
        if inv.empty or not required_inv.issubset(inv.columns):
            return findings
        logger.debug(
            "_detect_early_payment_discount using invoice_col=%s", invoice_col
        )
        # Placeholder: assume a 2% discount could have been taken if payment_terms < 30
        for _, row in inv.iterrows():
            terms = pd.to_numeric(row.get("payment_terms"), errors="coerce")
            if pd.notna(terms) and 0 < terms <= 15:
                try:
                    terms = int(row.get("payment_terms", 0))
                except ValueError:
                    terms = 0
                if terms > 0 and terms <= 15:
                    discount = row[invoice_col] * 0.02
                    findings.append(
                        self._build_finding(
                            "Early Payment Discount Missed",
                            row.get("supplier_id"),
                            None,
                            None,
                            discount,
                            {"invoice_amount": row[invoice_col], "terms": float(terms)},
                            [row.get("invoice_id")],
                        )
                    )
        return findings

    def _detect_demand_aggregation(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables.get("purchase_orders", pd.DataFrame())
        total_col = (
            "total_amount_gbp" if "total_amount_gbp" in po.columns else "total_amount"
        )
        required_po = {"supplier_id", "po_id", total_col}
        if po.empty or not required_po.issubset(po.columns):
            return findings

        logger.debug(
            "_detect_demand_aggregation using total_col=%s", total_col
        )

        grouped = (
            po.groupby("supplier_id")
            .agg(total_spend=(total_col, "sum"), po_ids=("po_id", list))
            .reset_index()
        )
        cond = grouped["total_spend"] < 500  # small spend could be aggregated
        for _, row in grouped[cond].iterrows():
            savings = row["total_spend"] * 0.05  # placeholder saving
            findings.append(
                self._build_finding(
                    "Demand Aggregation",
                    row.get("supplier_id"),
                    None,
                    None,
                    savings,
                    {"total_spend": row["total_spend"]},
                    list(row.get("po_ids", [])),
                )
            )
        return findings

    def _detect_logistics_cost_outliers(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        shipments = tables.get("shipments", pd.DataFrame())
        cost_col = (
            "logistics_cost_gbp" if "logistics_cost_gbp" in shipments.columns else "logistics_cost"
        )
        required_sh = {cost_col}
        if shipments.empty or not required_sh.issubset(shipments.columns):
            return findings

        logger.debug(
            "_detect_logistics_cost_outliers using cost_col=%s", cost_col
        )

        avg = shipments[cost_col].mean()
        cond = shipments[cost_col] > avg * 1.5
        for _, row in shipments[cond].iterrows():
            impact = row[cost_col] - avg
            findings.append(
                self._build_finding(
                    "Logistics Cost Outliers",
                    None,
                    None,
                    None,
                    impact,
                    {"average_cost": avg, "actual_cost": row[cost_col]},
                    [row.get("shipment_id"), row.get("po_id")],
                )
            )
        return findings

    def _detect_supplier_consolidation(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        ct = tables.get("contracts", pd.DataFrame())
        cat_col = "category_id" if "category_id" in ct.columns else "spend_category"
        value_col = (
            "total_contract_value_gbp"
            if "total_contract_value_gbp" in ct.columns
            else "total_contract_value"
        )
        required_ct = {cat_col, "supplier_id", "contract_id", value_col}
        if ct.empty or not required_ct.issubset(ct.columns):
            return findings

        logger.debug(
            "_detect_supplier_consolidation using cat_col=%s, value_col=%s",
            cat_col,
            value_col,
        )

        supplier_counts = (
            ct.groupby(cat_col)
            .agg(
                supplier_count=("supplier_id", "nunique"),
                contract_ids=("contract_id", list),
            )
            .reset_index()
        )

        spend = (
            ct.groupby([cat_col, "supplier_id"])[value_col]
            .sum()
            .reset_index()
        )

        cond = supplier_counts["supplier_count"] > 1
        for _, row in supplier_counts[cond].iterrows():
            cat = row.get(cat_col)
            cat_spend = spend[spend[cat_col] == cat]
            total_spend = cat_spend[value_col].sum()
            top_row = cat_spend.sort_values(value_col, ascending=False).head(1)
            max_spend = top_row[value_col].iloc[0] if not top_row.empty else 0.0
            savings = total_spend - max_spend
            supplier_id = top_row["supplier_id"].iloc[0] if not top_row.empty else None
            findings.append(
                self._build_finding(
                    "Supplier Consolidation",
                    supplier_id,
                    cat,
                    None,
                    savings,
                    {"supplier_count": row["supplier_count"], "total_spend": total_spend},
                    list(row.get("contract_ids", [])),
                )
            )
        return findings

    def _detect_supplier_price_difference(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        unit_col = (
            "unit_price_gbp" if "unit_price_gbp" in po_lines.columns else "unit_price"
        )
        required = {"item_id", "supplier_id", unit_col, "quantity"}
        if po_lines.empty or not required.issubset(po_lines.columns):
            return findings

        for item_id, group in po_lines.groupby("item_id"):
            min_price = group[unit_col].min()
            for _, row in group.iterrows():
                price = row[unit_col]
                qty = row.get("quantity", 1)
                if price > min_price:
                    savings = (price - min_price) * qty
                    findings.append(
                        self._build_finding(
                            "Supplier Price Difference",
                            row.get("supplier_id"),
                            None,
                            item_id,
                            savings,
                            {"unit_price": price, "min_price": min_price},
                            [row.get("po_id")],
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

    def _find_candidate_suppliers(
        self,
        item_id: Optional[str],
        current_supplier_id: Optional[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return alternative suppliers for the given item."""

        logger.debug(
            "_find_candidate_suppliers initial item_id=%s current_supplier=%s", item_id, current_supplier_id
        )

        if not item_id and sources:
            try:
                with self.agent_nick.get_db_connection() as conn:
                    for src in sources:
                        df = pd.read_sql(
                            "SELECT item_id FROM proc.po_line_items_agent WHERE po_id = %s",
                            conn,
                            params=(src,),
                        )
                        if df.empty:
                            df = pd.read_sql(
                                "SELECT item_id FROM proc.invoice_line_items_agent WHERE invoice_id = %s",
                                conn,
                                params=(src,),
                            )
                        if not df.empty:
                            item_id = str(df["item_id"].dropna().iloc[0])
                            logger.debug(
                                "_find_candidate_suppliers inferred item_id %s from source %s",
                                item_id,
                                src,
                            )
                            break
            except Exception:
                logger.exception("Failed to infer item_id from sources %s", sources)

        if not item_id:
            logger.debug("_find_candidate_suppliers: no item_id available; skipping")
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
            logger.exception("_find_candidate_suppliers query failed for item %s", item_id)
            return []

        if df.empty:
            return []

        df = df.dropna(subset=["supplier_id", "unit_price"])
        if df.empty:
            return []

        cur_unit = None
        if current_supplier_id is not None:
            cur_rows = df[df["supplier_id"] == current_supplier_id]["unit_price"]
            if not cur_rows.empty:
                cur_unit = cur_rows.min()

        if cur_unit is None:
            cur_unit = df["unit_price"].max() + 1.0

        candidates_df = df[(df["unit_price"] < cur_unit) & (df["supplier_id"] != current_supplier_id)]
        if candidates_df.empty:
            return []

        grouped = candidates_df.groupby("supplier_id", as_index=False)["unit_price"].min()
        result = [
            {"supplier_id": r["supplier_id"], "unit_price": float(r["unit_price"])}
            for _, r in grouped.iterrows()
        ]
        logger.debug("_find_candidate_suppliers found %d candidates", len(result))
        return result


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

