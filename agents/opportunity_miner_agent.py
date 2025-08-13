"""Opportunity Miner Agent.

This agent implements a lightweight version of the functional
requirements supplied in the task description.  It ingests procurement
tables, performs basic validation and currency normalisation and then
runs a set of seven fixed detectors.  Each detector produces a list of
opportunities which are filtered using a global financial impact
threshold before being written to Excel and a JSON feed that mimics the
dashboard output.

The implementation is intentionally simplified: the goal is to provide a
clear, auditable pipeline that can be expanded upon.  The detectors use
very small placeholder calculations so the agent can operate without a
fully‑fledged dataset or database connection, making it suitable for the
MVP and unit tests.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

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

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["detected_on"] = self.detected_on.isoformat()
        return d


class OpportunityMinerAgent(BaseAgent):
    """Agent for identifying procurement anomalies and savings opportunities."""

    def __init__(self, agent_nick, min_financial_impact: float = 100.0) -> None:
        super().__init__(agent_nick)
        self.min_financial_impact = min_financial_impact

        # ------------------------------------------------------------------
        # GPU configuration
        # ------------------------------------------------------------------
        try:  # pragma: no cover - torch is optional for this repository
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - defensive
            self.device = "cpu"
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
            findings.extend(self._detect_contract_price_drift(tables))
            findings.extend(self._detect_po_invoice_discrepancy(tables))
            findings.extend(self._detect_early_payment_discount(tables))
            findings.extend(self._detect_demand_aggregation(tables))
            findings.extend(self._detect_logistics_cost_outliers(tables))
            findings.extend(self._detect_supplier_consolidation(tables))

            filtered = [f for f in findings if f.financial_impact_gbp >= self.min_financial_impact]

            self._output_excel(filtered)
            self._output_feed(filtered)

            data = {
                "findings": [f.as_dict() for f in filtered],
                "opportunity_count": len(filtered),
                "total_savings": sum(f.financial_impact_gbp for f in filtered),
            }

            return AgentOutput(status=AgentStatus.SUCCESS, data=data, confidence=1.0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("OpportunityMinerAgent error: %s", exc)
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))

    # ------------------------------------------------------------------
    # Data ingestion and preparation
    # ------------------------------------------------------------------
    TABLES = [
        "purchase_orders",
        "invoices",
        "contracts",
        "price_benchmarks",
        "indices",
        "shipments",
        "supplier_master",
    ]

    def _ingest_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch required tables from the database or fall back to mock data."""

        dfs: Dict[str, pd.DataFrame] = {}
        try:
            with self.agent_nick.get_db_connection() as conn:
                for table in self.TABLES:
                    dfs[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception as exc:  # pragma: no cover - database is optional for tests
            logger.warning("Using mock data for opportunity mining: %s", exc)
            dfs = self._mock_data()
        return dfs

    def _mock_data(self) -> Dict[str, pd.DataFrame]:
        """Return a minimal in-memory dataset used for unit tests and demos."""

        # All values already in GBP for simplicity
        purchase_orders = pd.DataFrame(
            [
                {
                    "po_id": "PO1",
                    "supplier_id": "S1",
                    "category_id": "C1",
                    "item_id": "I1",
                    "unit_price": 12.0,
                    "quantity": 10,
                    "currency": "GBP",
                    "order_date": "2024-01-01",
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
                    "item_id": "I1",
                    "unit_price": 13.0,
                    "quantity": 10,
                    "currency": "GBP",
                    "invoice_date": "2024-01-05",
                    "payment_terms": "30",
                }
            ]
        )

        contracts = pd.DataFrame(
            [
                {
                    "contract_id": "CT1",
                    "supplier_id": "S1",
                    "category_id": "C1",
                    "item_id": "I1",
                    "agreed_price": 11.0,
                    "currency": "GBP",
                    "start_date": "2023-01-01",
                    "end_date": "2025-01-01",
                    "index_link": None,
                }
            ]
        )

        price_benchmarks = pd.DataFrame(
            [
                {
                    "item_id": "I1",
                    "benchmark_price": 10.0,
                    "currency": "GBP",
                    "source": "internal",
                    "effective_date": "2024-01-01",
                }
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
                {"supplier_id": "S1", "supplier_name": "Supplier One", "region": "UK", "risk_rating": 1}
            ]
        )

        return {
            "purchase_orders": purchase_orders,
            "invoices": invoices,
            "contracts": contracts,
            "price_benchmarks": price_benchmarks,
            "indices": indices,
            "shipments": shipments,
            "supplier_master": supplier_master,
        }

    def _validate_data(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Basic validation ensuring required columns exist and dropping nulls."""

        required_columns = {
            "purchase_orders": ["po_id", "supplier_id", "category_id", "item_id", "unit_price", "quantity", "currency"],
            "invoices": ["invoice_id", "po_id", "supplier_id", "item_id", "unit_price", "quantity", "currency"],
            "contracts": ["contract_id", "supplier_id", "category_id", "item_id", "agreed_price", "currency"],
            "price_benchmarks": ["item_id", "benchmark_price", "currency"],
            "indices": ["index_name", "value", "currency"],
            "shipments": ["shipment_id", "po_id", "logistics_cost", "currency"],
            "supplier_master": ["supplier_id"],
        }

        for name, cols in required_columns.items():
            df = tables.get(name, pd.DataFrame())
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Table {name} missing columns: {missing}")
            tables[name] = df.dropna(subset=cols)

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
            rate_col = df["currency"].map(lambda c: fx_rates.get(c, 1.0))
            for col in cols:
                if col in df.columns:
                    df[f"{col}_gbp"] = df[col] * rate_col
            return df

        tables["purchase_orders"] = convert(tables["purchase_orders"], ["unit_price"])
        tables["invoices"] = convert(tables["invoices"], ["unit_price"])
        tables["contracts"] = convert(tables["contracts"], ["agreed_price"])
        tables["price_benchmarks"] = convert(tables["price_benchmarks"], ["benchmark_price"])
        tables["shipments"] = convert(tables["shipments"], ["logistics_cost"])
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
        po = tables["purchase_orders"]
        bm = tables["price_benchmarks"]
        if po.empty or bm.empty:
            return findings
        merged = po.merge(bm, on="item_id", suffixes=("_po", "_bm"))
        merged["variance"] = merged["unit_price_gbp"] - merged["benchmark_price_gbp"]
        cond = merged["variance"] > 0
        for _, row in merged[cond].iterrows():
            savings = row["variance"] * row.get("quantity", 1)
            findings.append(
                self._build_finding(
                    "Unit Price vs Benchmark",
                    row.get("supplier_id"),
                    row.get("category_id"),
                    row.get("item_id"),
                    savings,
                    {"unit_price_gbp": row["unit_price_gbp"], "benchmark_price_gbp": row["benchmark_price_gbp"]},
                    [row.get("po_id")],
                )
            )
        return findings

    def _detect_contract_price_drift(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        inv = tables["invoices"]
        ct = tables["contracts"]
        if inv.empty or ct.empty:
            return findings
        merged = inv.merge(ct, on=["supplier_id", "item_id", "contract_id"], how="inner")
        merged["variance"] = merged["unit_price_gbp"] - merged.get("adjusted_price_gbp", merged.get("agreed_price_gbp"))
        cond = merged["variance"] > 0
        for _, row in merged[cond].iterrows():
            savings = row["variance"] * row.get("quantity", 1)
            findings.append(
                self._build_finding(
                    "Contract Price Drift",
                    row.get("supplier_id"),
                    row.get("category_id"),
                    row.get("item_id"),
                    savings,
                    {"invoice_price_gbp": row["unit_price_gbp"], "contract_price_gbp": row["adjusted_price_gbp"]},
                    [row.get("invoice_id"), row.get("contract_id")],
                )
            )
        return findings

    def _detect_po_invoice_discrepancy(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables["purchase_orders"]
        inv = tables["invoices"]
        if po.empty or inv.empty:
            return findings
        merged = po.merge(inv, on="po_id", suffixes=("_po", "_inv"))
        merged["price_diff"] = merged["unit_price_gbp_inv"] - merged["unit_price_gbp_po"]
        merged["qty_diff"] = merged["quantity_inv"] - merged["quantity_po"]
        cond = (merged["price_diff"] != 0) | (merged["qty_diff"] != 0)
        for _, row in merged[cond].iterrows():
            impact = row["price_diff"] * row.get("quantity_inv", 1)
            findings.append(
                self._build_finding(
                    "PO↔Invoice Discrepancy",
                    row.get("supplier_id_po"),
                    row.get("category_id"),
                    row.get("item_id_po"),
                    impact,
                    {"price_diff": row["price_diff"], "qty_diff": row["qty_diff"]},
                    [row.get("po_id"), row.get("invoice_id")],
                )
            )
        return findings

    def _detect_early_payment_discount(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        inv = tables["invoices"]
        if inv.empty:
            return findings
        # Placeholder: assume a 2% discount could have been taken if payment_terms < 30
        for _, row in inv.iterrows():
            try:
                terms = int(row.get("payment_terms", 0))
            except ValueError:
                terms = 0
            if terms > 0 and terms <= 15:
                discount = row["unit_price_gbp"] * row.get("quantity", 1) * 0.02
                findings.append(
                    self._build_finding(
                        "Early Payment Discount Missed",
                        row.get("supplier_id"),
                        None,
                        row.get("item_id"),
                        discount,
                        {"terms": terms},
                        [row.get("invoice_id")],
                    )
                )
        return findings

    def _detect_demand_aggregation(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        po = tables["purchase_orders"]
        if po.empty:
            return findings
        grouped = po.groupby(["supplier_id", "item_id"]).agg({"quantity": "sum"}).reset_index()
        cond = grouped["quantity"] < 20  # small orders could be aggregated
        for _, row in grouped[cond].iterrows():
            savings = row["quantity"] * 0.5  # placeholder saving
            findings.append(
                self._build_finding(
                    "Demand Aggregation",
                    row.get("supplier_id"),
                    None,
                    row.get("item_id"),
                    savings,
                    {"total_quantity": row["quantity"]},
                    [],
                )
            )
        return findings

    def _detect_logistics_cost_outliers(self, tables: Dict[str, pd.DataFrame]) -> List[Finding]:
        findings: List[Finding] = []
        shipments = tables["shipments"]
        if shipments.empty:
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
        po = tables["purchase_orders"]
        if po.empty:
            return findings
        supplier_counts = po.groupby("category_id")["supplier_id"].nunique().reset_index(name="supplier_count")
        cond = supplier_counts["supplier_count"] > 1
        for _, row in supplier_counts[cond].iterrows():
            savings = row["supplier_count"] * 10.0  # placeholder
            findings.append(
                self._build_finding(
                    "Supplier Consolidation",
                    None,
                    row.get("category_id"),
                    None,
                    savings,
                    {"supplier_count": row["supplier_count"]},
                    [],
                )
            )
        return findings

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

