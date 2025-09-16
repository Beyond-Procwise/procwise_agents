from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

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
    supplier_name: Optional[str] = None
    weightage: float = 0.0
    candidate_suppliers: List[Dict[str, Any]] = field(default_factory=list)
    context_documents: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["detected_on"] = self.detected_on.isoformat()
        return d


class OpportunityMinerAgent(BaseAgent):
    """Agent for identifying procurement anomalies and savings opportunities."""

    def __init__(self, agent_nick, min_financial_impact: float = 100.0) -> None:
        super().__init__(agent_nick)
        self.min_financial_impact = min_financial_impact
        self._supplier_lookup: Dict[str, Optional[str]] = {}
        self._contract_supplier_map: Dict[str, str] = {}
        self._contract_metadata: Dict[str, Dict[str, Any]] = {}
        self._po_supplier_map: Dict[str, str] = {}
        self._po_contract_map: Dict[str, str] = {}
        self._invoice_supplier_map: Dict[str, str] = {}
        self._invoice_po_map: Dict[str, str] = {}
        self._supplier_risk_map: Dict[str, float] = {}
        self._event_log: List[Dict[str, Any]] = []
        self._escalations: List[Dict[str, Any]] = []
        self._column_cache: Dict[str, set[str]] = {}

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

    def _read_sql(self, query: str, params: Any = None) -> pd.DataFrame:
        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        if callable(pandas_conn):
            with pandas_conn() as conn:
                return pd.read_sql(query, conn, params=params)
        with self.agent_nick.get_db_connection() as conn:
            return pd.read_sql(query, conn, params=params)

    def _get_table_columns(self, schema: str, table: str) -> set[str]:
        """Return cached column names for ``schema.table``."""

        cache_key = f"{schema}.{table}"
        if cache_key in self._column_cache:
            return self._column_cache[cache_key]

        columns: set[str] = set()
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if callable(get_conn):
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_schema = %s AND table_name = %s
                            """,
                            (schema, table),
                        )
                        columns = {row[0] for row in cur.fetchall()}
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Column introspection failed for %s.%s", schema, table
                )
        self._column_cache[cache_key] = columns
        return columns

    def _price_expression(self, schema: str, table: str, alias: str) -> str:
        """Return a resilient SQL expression for unit prices."""

        query_engine = getattr(self.agent_nick, "query_engine", None)
        get_conn = getattr(self.agent_nick, "get_db_connection", None)

        if (
            callable(get_conn)
            and query_engine is not None
            and hasattr(query_engine, "_price_expression")
        ):
            try:
                with get_conn() as conn:
                    return query_engine._price_expression(  # type: ignore[attr-defined]
                        conn, schema, table, alias
                    )
            except Exception:
                logger.exception(
                    "Falling back after price column detection failed for %s.%s",
                    schema,
                    table,
                )

        columns = self._get_table_columns(schema, table)
        if "unit_price_gbp" in columns and "unit_price" in columns:
            return f"COALESCE({alias}.unit_price_gbp, {alias}.unit_price)"
        for candidate in (
            "unit_price_gbp",
            "unit_price",
            "price_gbp",
            "price",
            "net_price_gbp",
            "net_price",
        ):
            if candidate in columns:
                return f"{alias}.{candidate}"
        for column in columns:
            if "price" in column:
                return f"{alias}.{column}"
        logger.warning(
            "Unable to determine price column for %s.%s; defaulting to 0.0",
            schema,
            table,
        )
        return "0.0"

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------
    def process(self, context: AgentContext) -> AgentOutput:
        try:
            logger.info(
                "OpportunityMinerAgent starting processing with input %s",
                context.input_data,
            )
            qe = getattr(self.agent_nick, "query_engine", None)
            if qe and hasattr(qe, "train_procurement_context"):
                try:
                    qe.train_procurement_context()
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Failed to train procurement context")
            tables = self._ingest_data()
            tables = self._validate_data(tables)
            self._build_supplier_lookup(tables)
            tables = self._normalise_currency(tables)
            tables = self._apply_index_adjustment(tables)

            self._event_log = []
            self._escalations = []
            notifications: set[str] = set()

            min_impact_threshold = self._resolve_min_financial_impact(
                context.input_data
            )

            workflow_name = context.input_data.get("workflow")
            if not workflow_name:
                message = "Mandatory orchestrator field 'workflow' is missing."
                self._log_policy_event("unknown", None, "blocked", message, {})
                return self._blocked_output(message)

            policy_registry = self._get_policy_registry()
            alias_map = {"opportunity_mining": "contract_expiry_check"}
            lookup_name = workflow_name
            mapped = alias_map.get(workflow_name.lower()) if isinstance(workflow_name, str) else None
            if mapped:
                lookup_name = mapped
                context.input_data["workflow"] = lookup_name
            policy_cfg = policy_registry.get(lookup_name)
            if policy_cfg is None:
                message = f"Unsupported opportunity workflow '{workflow_name}'"
                self._log_policy_event(
                    "unknown", None, "blocked", message, {"workflow": workflow_name}
                )
                return self._blocked_output(message)

            missing_fields = self._missing_required_fields(
                context.input_data,
                policy_cfg.get("required_fields", []),
                policy_cfg.get("default_conditions", {}),
            )
            if missing_fields:
                message = (
                    "Missing mandatory orchestrator fields: "
                    + ", ".join(sorted(missing_fields))
                )
                self._log_policy_event(
                    policy_cfg["policy_id"],
                    None,
                    "blocked",
                    message,
                    {"missing_fields": sorted(missing_fields)},
                )
                return self._blocked_output(message, policy_id=policy_cfg["policy_id"])

            handler = policy_cfg["handler"]
            findings = handler(tables, context.input_data, notifications, policy_cfg)

            findings = self._enrich_findings(findings, tables)
            filtered = [
                f for f in findings if f.financial_impact_gbp >= min_impact_threshold
            ]
            self._load_supplier_risk_map()
            for f in filtered:
                f.candidate_suppliers = self._find_candidate_suppliers(
                    f.item_id, f.supplier_id, f.source_records
                )
            self._attach_vector_context(filtered)

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
                "min_financial_impact": min_impact_threshold,
                "policy_events": self._event_log,
                "escalations": self._escalations,
            }
            # pass candidate supplier IDs to downstream agents
            supplier_candidates = {
                s["supplier_id"]
                for f in filtered
                for s in f.candidate_suppliers
                if s.get("supplier_id")
            }
            data["supplier_candidates"] = list(supplier_candidates)
            data["notifications"] = sorted(notifications)
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
        "quotes": "proc.quote_agent",
        "quote_lines": "proc.quote_line_items_agent",
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
        monkeypatch this method to return deterministic fixtures.
        """

        dfs: Dict[str, pd.DataFrame] = {}
        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        if callable(pandas_conn):
            with pandas_conn() as conn:
                for table, sql_name in self.TABLE_MAP.items():
                    dfs[table] = pd.read_sql(f"SELECT * FROM {sql_name}", conn)
        else:
            with self.agent_nick.get_db_connection() as conn:
                for table, sql_name in self.TABLE_MAP.items():
                    dfs[table] = pd.read_sql(f"SELECT * FROM {sql_name}", conn)
        return dfs

    def _validate_data(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Basic validation ensuring required columns exist and dropping nulls."""

        for name, df in list(tables.items()):
            if df.empty:
                continue
            # Drop rows that are entirely null
            tables[name] = df.dropna(how="all")
            logger.debug("Table %s columns: %s", name, list(tables[name].columns))
        return tables

    def _build_supplier_lookup(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Build helper maps to resolve supplier metadata from ``proc.supplier``."""

        supplier_master = tables.get("supplier_master", pd.DataFrame())
        lookup: Dict[str, Optional[str]] = {}

        def _normalise(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            value = str(value).strip()
            return value or None

        if not supplier_master.empty and "supplier_id" in supplier_master.columns:
            df = supplier_master.dropna(subset=["supplier_id"]).copy()
            if not df.empty:
                df["supplier_id"] = df["supplier_id"].map(_normalise)
                if "supplier_name" in df.columns:
                    df["supplier_name"] = df["supplier_name"].map(_normalise)
                df = df.dropna(subset=["supplier_id"])
                for _, row in df.iterrows():
                    supplier_id = row["supplier_id"]
                    lookup[supplier_id] = row.get("supplier_name") or None

        self._supplier_lookup = lookup
        logger.debug("Loaded %d suppliers from master data", len(self._supplier_lookup))

        contracts = tables.get("contracts", pd.DataFrame())
        contract_map: Dict[str, str] = {}
        contract_meta: Dict[str, Dict[str, Any]] = {}
        if not contracts.empty and "contract_id" in contracts.columns:
            df = contracts.dropna(subset=["contract_id"]).copy()
            df["contract_id"] = df["contract_id"].map(_normalise)
            if "supplier_id" in df.columns:
                df["supplier_id"] = df["supplier_id"].map(_normalise)
            if "spend_category" in df.columns:
                df["spend_category"] = df["spend_category"].map(_normalise)
            if "category_id" in df.columns:
                df["category_id"] = df["category_id"].map(_normalise)
            df = df.dropna(subset=["contract_id"])
            for _, row in df.iterrows():
                contract_id = row.get("contract_id")
                if not contract_id:
                    continue
                supplier_id = row.get("supplier_id")
                spend_category = row.get("spend_category") or row.get("category_id")
                contract_meta[contract_id] = {
                    "supplier_id": supplier_id,
                    "spend_category": spend_category,
                    "contract_title": row.get("contract_title"),
                    "contract_type": row.get("contract_type"),
                    "total_contract_value": row.get("total_contract_value"),
                    "total_contract_value_gbp": row.get("total_contract_value_gbp"),
                    "contract_end_date": row.get("contract_end_date"),
                    "contract_start_date": row.get("contract_start_date"),
                }
                if supplier_id and (not lookup or supplier_id in lookup):
                    contract_map[contract_id] = supplier_id
        self._contract_supplier_map = contract_map
        self._contract_metadata = contract_meta
        logger.debug(
            "Mapped %d contracts to supplier IDs using validated supplier data",
            len(self._contract_supplier_map),
        )

        purchase_orders = tables.get("purchase_orders", pd.DataFrame())
        po_supplier_map: Dict[str, str] = {}
        po_contract_map: Dict[str, str] = {}
        if not purchase_orders.empty and "po_id" in purchase_orders.columns:
            df = purchase_orders.dropna(subset=["po_id"]).copy()
            df["po_id"] = df["po_id"].map(_normalise)
            if "supplier_id" in df.columns:
                df["supplier_id"] = df["supplier_id"].map(_normalise)
            if "contract_id" in df.columns:
                df["contract_id"] = df["contract_id"].map(_normalise)
            df = df.dropna(subset=["po_id"])
            for _, row in df.iterrows():
                po_id = row.get("po_id")
                if not po_id:
                    continue
                supplier_id = row.get("supplier_id")
                contract_id = row.get("contract_id")
                if contract_id:
                    po_contract_map[po_id] = contract_id
                candidate = supplier_id
                if not candidate and contract_id:
                    candidate = contract_map.get(contract_id)
                if candidate and (not lookup or candidate in lookup):
                    po_supplier_map[po_id] = candidate
        self._po_supplier_map = po_supplier_map
        self._po_contract_map = po_contract_map

        invoices = tables.get("invoices", pd.DataFrame())
        invoice_supplier_map: Dict[str, str] = {}
        invoice_po_map: Dict[str, str] = {}
        if not invoices.empty and "invoice_id" in invoices.columns:
            df = invoices.dropna(subset=["invoice_id"]).copy()
            df["invoice_id"] = df["invoice_id"].map(_normalise)
            if "supplier_id" in df.columns:
                df["supplier_id"] = df["supplier_id"].map(_normalise)
            if "po_id" in df.columns:
                df["po_id"] = df["po_id"].map(_normalise)
            df = df.dropna(subset=["invoice_id"])
            for _, row in df.iterrows():
                invoice_id = row.get("invoice_id")
                if not invoice_id:
                    continue
                po_id = row.get("po_id")
                supplier_id = row.get("supplier_id")
                if po_id:
                    invoice_po_map[invoice_id] = po_id
                candidate = supplier_id
                if not candidate and po_id and po_supplier_map.get(po_id):
                    candidate = po_supplier_map.get(po_id)
                if not candidate and po_id and po_contract_map.get(po_id):
                    candidate = contract_map.get(po_contract_map[po_id])
                if candidate and (not lookup or candidate in lookup):
                    invoice_supplier_map[invoice_id] = candidate
        self._invoice_supplier_map = invoice_supplier_map
        self._invoice_po_map = invoice_po_map

        logger.debug(
            "Derived supplier lookups: %d POs, %d invoices, %d contracts",
            len(self._po_supplier_map),
            len(self._invoice_supplier_map),
            len(self._contract_supplier_map),
        )

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
            [
                "unit_price",
                "line_total",
                "line_amount",
                "tax_amount",
                "total_amount",
                "total_amount_incl_tax",
            ],
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
    # Policy helpers
    # ------------------------------------------------------------------
    def _blocked_output(self, message: str, policy_id: Optional[str] = None) -> AgentOutput:
        data = {
            "findings": [],
            "opportunity_count": 0,
            "total_savings": 0.0,
            "supplier_candidates": [],
            "policy_events": self._event_log,
            "escalations": self._escalations,
            "notifications": [],
            "blocked_reason": message,
        }
        if policy_id:
            data["policy_id"] = policy_id
        logger.warning("OpportunityMinerAgent blocked execution: %s", message)
        return AgentOutput(
            status=AgentStatus.FAILED,
            data=data,
            pass_fields=data,
            error=message,
        )

    def _missing_required_fields(
        self,
        input_data: Dict[str, Any],
        required_fields: Iterable[str],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not required_fields:
            return []
        conditions = input_data.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
            input_data["conditions"] = conditions
        missing = []
        for field in required_fields:
            value = conditions.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                default_map = defaults or {}
                if field in default_map and default_map[field] is not None:
                    conditions[field] = default_map[field]
                else:
                    missing.append(field)
        return missing

    def _get_condition(self, input_data: Dict[str, Any], key: str, default: Any = None) -> Any:
        conditions = input_data.get("conditions")
        if isinstance(conditions, dict):
            return conditions.get(key, default)
        return default

    def _resolve_supplier_id(self, supplier_id: Optional[Any]) -> Optional[str]:
        if supplier_id is None:
            return None
        supplier = str(supplier_id).strip()
        if not supplier:
            return None
        if self._supplier_lookup and supplier not in self._supplier_lookup:
            logger.debug("Supplier %s not found in master data; skipping", supplier)
            return None
        return supplier

    def _log_policy_event(
        self,
        policy_id: str,
        supplier_id: Optional[str],
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "policy_id": policy_id,
            "supplier_id": supplier_id,
            "status": status,
            "message": message,
        }
        if details:
            event["details"] = details
        self._event_log.append(event)
        logger.info(
            "Policy %s event for supplier %s: %s", policy_id, supplier_id, message
        )

    def _record_escalation(
        self,
        policy_id: str,
        supplier_id: Optional[str],
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        escalation = {
            "policy_id": policy_id,
            "supplier_id": supplier_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if details:
            escalation["details"] = details
        self._escalations.append(escalation)
        self._log_policy_event(policy_id, supplier_id, "escalated", message, details)

    def _default_notifications(self, notifications: set[str]) -> None:
        notifications.update({"Negotiation", "Approvals", "CategoryManager"})

    def _resolve_min_financial_impact(self, input_data: Dict[str, Any]) -> float:
        """Determine the impact threshold for the current run.

        The agent exposes ``min_financial_impact`` as an instance-level default
        so that platform settings can cap noise.  Workflow callers may override
        the threshold per run via ``input_data['min_financial_impact']``.  Any
        malformed or negative value falls back to the configured default to
        avoid silently discarding all findings.
        """

        value = input_data.get("min_financial_impact")
        if value is None:
            return self.min_financial_impact
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid min_financial_impact '%s'; using default %s",
                value,
                self.min_financial_impact,
            )
            return self.min_financial_impact
        if threshold < 0:
            logger.warning(
                "Negative min_financial_impact %s provided; using default %s",
                threshold,
                self.min_financial_impact,
            )
            return self.min_financial_impact
        return threshold

    def _get_policy_registry(self) -> Dict[str, Dict[str, Any]]:
        return {
            "price_variance_check": {
                "policy_id": "oppfinderpolicy_001_price_benchmark_variance_detection",
                "detector": "Price Benchmark Variance",
                "required_fields": [
                    "supplier_id",
                    "item_id",
                    "actual_price",
                    "benchmark_price",
                ],
                "handler": self._policy_price_benchmark_variance,
            },
            "volume_consolidation_check": {
                "policy_id": "oppfinderpolicy_003_volume_consolidation",
                "detector": "Volume Consolidation",
                "required_fields": ["minimum_volume_gbp"],
                "handler": self._policy_volume_consolidation,
            },
            "contract_expiry_check": {
                "policy_id": "oppfinderpolicy_004_contract_expiry_opportunity",
                "detector": "Contract Expiry Opportunity",
                "required_fields": ["negotiation_window_days"],
                "handler": self._policy_contract_expiry,
                "default_conditions": {"negotiation_window_days": 90},
            },
            "supplier_risk_check": {
                "policy_id": "oppfinderpolicy_005_supplier_risk_alert",
                "detector": "Supplier Risk Alert",
                "required_fields": ["risk_threshold"],
                "handler": self._policy_supplier_risk,
            },
            "maverick_spend_check": {
                "policy_id": "oppfinderpolicy_006_maverick_spend_detection",
                "detector": "Maverick Spend Detection",
                "required_fields": ["minimum_value_gbp"],
                "handler": self._policy_maverick_spend,
            },
            "duplicate_supplier_check": {
                "policy_id": "oppfinderpolicy_007_duplicate_supplier",
                "detector": "Duplicate Supplier",
                "required_fields": ["minimum_overlap_gbp"],
                "handler": self._policy_duplicate_supplier,
            },
            "category_overspend_check": {
                "policy_id": "oppfinderpolicy_008_category_overspend",
                "detector": "Category Overspend",
                "required_fields": ["category_budgets"],
                "handler": self._policy_category_overspend,
            },
            "inflation_passthrough_check": {
                "policy_id": "oppfinderpolicy_009_inflation_pass-through",
                "detector": "Inflation Pass-Through",
                "required_fields": ["market_inflation_pct"],
                "handler": self._policy_inflation_passthrough,
            },
            "unused_contract_value_check": {
                "policy_id": "oppfinderpolicy_010_unused_contract_value",
                "detector": "Unused Contract Value",
                "required_fields": ["minimum_unused_value_gbp"],
                "handler": self._policy_unused_contract_value,
            },
            "supplier_performance_check": {
                "policy_id": "oppfinderpolicy_011_supplier_performance_deviation",
                "detector": "Supplier Performance Deviation",
                "required_fields": ["performance_records"],
                "handler": self._policy_supplier_performance,
            },
            "esg_opportunity_check": {
                "policy_id": "oppfinderpolicy_012_esg_opportunity",
                "detector": "ESG Opportunity",
                "required_fields": ["esg_scores"],
                "handler": self._policy_esg_opportunity,
            },
        }

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            result = pd.to_numeric(value, errors="coerce")
        except Exception:
            return default
        if pd.isna(result):
            return default
        return float(result)

    def _to_int(self, value: Any, default: int = 0) -> int:
        try:
            result = int(float(value))
            return result
        except Exception:
            return default

    def _to_date(self, value: Any) -> date:
        if value is None:
            return datetime.utcnow().date()
        try:
            ts = pd.to_datetime(value, errors="coerce")
        except Exception:
            return datetime.utcnow().date()
        if pd.isna(ts):
            return datetime.utcnow().date()
        return ts.date()

    def _po_ids_for_supplier_item(
        self,
        tables: Dict[str, pd.DataFrame],
        supplier_id: Optional[str],
        item_id: Optional[str],
    ) -> List[str]:
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if po_lines.empty or "po_id" not in po_lines.columns:
            return []
        df = po_lines.dropna(subset=["po_id"]).copy()
        if item_id is not None and "item_id" in df.columns:
            df = df[df["item_id"].astype(str) == str(item_id)]
        po_ids = df["po_id"].astype(str).unique().tolist()
        if supplier_id:
            po_ids = [po_id for po_id in po_ids if self._po_supplier_map.get(po_id) == supplier_id]
        return po_ids

    def _invoice_ids_for_po(
        self, tables: Dict[str, pd.DataFrame], po_ids: Iterable[str]
    ) -> List[str]:
        invoices = tables.get("invoices", pd.DataFrame())
        if invoices.empty or "po_id" not in invoices.columns or "invoice_id" not in invoices.columns:
            return []
        po_ids = {str(po_id) for po_id in po_ids}
        if not po_ids:
            return []
        df = invoices[invoices["po_id"].astype(str).isin(po_ids)]
        return df["invoice_id"].dropna().astype(str).unique().tolist()

    def _choose_first_column(
        self, df: pd.DataFrame, options: Iterable[str]
    ) -> Optional[str]:
        for option in options:
            if option in df.columns:
                return option
        return None

    def _category_for_po(self, po_id: Optional[str]) -> Optional[str]:
        if not po_id:
            return None
        contract_id = self._po_contract_map.get(po_id)
        if not contract_id:
            return None
        metadata = self._contract_metadata.get(contract_id, {})
        category = metadata.get("spend_category") or metadata.get("category_id")
        return category if category else None

    def _contract_value(self, contract_id: Optional[str]) -> float:
        if not contract_id:
            return 0.0
        contract = self._contract_metadata.get(contract_id, {})
        value = contract.get("total_contract_value_gbp") or contract.get("total_contract_value")
        if value is None:
            return 0.0
        return self._to_float(value)

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

    def _enrich_findings(
        self, findings: List[Finding], tables: Dict[str, pd.DataFrame]
    ) -> List[Finding]:
        """Fill in missing supplier/category/item IDs using live tables."""
        po = tables.get("purchase_orders", pd.DataFrame())
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        inv_lines = tables.get("invoice_lines", pd.DataFrame())
        contracts = tables.get("contracts", pd.DataFrame())
        quotes = tables.get("quotes", pd.DataFrame())
        quote_lines = tables.get("quote_lines", pd.DataFrame())
        supplier_lookup = getattr(self, "_supplier_lookup", {})
        contract_supplier_map = getattr(self, "_contract_supplier_map", {})
        contract_metadata = getattr(self, "_contract_metadata", {})
        po_supplier_map = getattr(self, "_po_supplier_map", {})
        po_contract_map = getattr(self, "_po_contract_map", {})
        invoice_supplier_map = getattr(self, "_invoice_supplier_map", {})
        invoice_po_map = getattr(self, "_invoice_po_map", {})


        def _lookup(df: pd.DataFrame, key_col: str, val_col: str, key: str) -> Optional[str]:
            try:
                if df.empty or key_col not in df.columns or val_col not in df.columns:
                    return None
                match = df[df[key_col].astype(str) == str(key)][val_col].dropna()
                return str(match.iloc[0]) if not match.empty else None
            except Exception:
                return None

        def _clean_supplier(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            supplier_id = str(value)
            if supplier_lookup and supplier_id not in supplier_lookup:
                logger.debug(
                    "Supplier ID %s not found in supplier master; ignoring", supplier_id
                )
                return None
            return supplier_id

        def _normalise(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            return str(value).strip()

        def _contract_category(contract_id: Optional[str]) -> Optional[str]:
            if not contract_id:
                return None
            meta = contract_metadata.get(contract_id)
            if not meta:
                return None
            return meta.get("spend_category") or meta.get("category_id")

        def _contract_supplier(contract_id: Optional[str]) -> Optional[str]:
            if not contract_id:
                return None
            supplier_id = contract_supplier_map.get(contract_id)
            if supplier_id:
                return supplier_id
            meta = contract_metadata.get(contract_id)
            if not meta:
                return None
            return meta.get("supplier_id")

        for f in findings:
            f.supplier_id = _clean_supplier(f.supplier_id)
            for src in f.source_records:
                src_key = _normalise(src)
                if src_key and not f.supplier_id:
                    candidate = (
                        contract_supplier_map.get(src_key)
                        or po_supplier_map.get(src_key)
                        or invoice_supplier_map.get(src_key)
                    )
                    if not candidate and src_key in po_contract_map:
                        candidate = _contract_supplier(po_contract_map.get(src_key))
                    if not candidate and src_key in invoice_po_map:
                        po_id = invoice_po_map.get(src_key)
                        candidate = (
                            po_supplier_map.get(po_id)
                            or _contract_supplier(po_contract_map.get(po_id))
                        )
                    if not candidate:
                        candidate = (
                            _lookup(po, "po_id", "supplier_id", src_key)
                            or _lookup(contracts, "contract_id", "supplier_id", src_key)
                            or _lookup(quotes, "quote_id", "supplier_id", src_key)
                        )
                    f.supplier_id = _clean_supplier(candidate)
                if src_key and not f.category_id:
                    candidate_cat = None
                    if src_key in contract_metadata:
                        candidate_cat = _contract_category(src_key)
                    if not candidate_cat and src_key in po_contract_map:
                        candidate_cat = _contract_category(po_contract_map.get(src_key))
                    if not candidate_cat and src_key in invoice_po_map:
                        po_id = invoice_po_map.get(src_key)
                        candidate_cat = _contract_category(po_contract_map.get(po_id))
                    if not candidate_cat:
                        candidate_cat = (
                            _lookup(po, "po_id", "spend_category", src_key)
                            or _lookup(po, "po_id", "category_id", src_key)
                            or _lookup(contracts, "contract_id", "spend_category", src_key)
                            or _lookup(contracts, "contract_id", "category_id", src_key)
                        )
                    if candidate_cat:
                        f.category_id = str(candidate_cat)

                if src_key and not f.item_id:
                    f.item_id = (
                        _lookup(po_lines, "po_id", "item_id", src_key)
                        or _lookup(inv_lines, "invoice_id", "item_id", src_key)
                        or _lookup(quote_lines, "quote_id", "item_id", src_key)
                        or f.item_id
                    )
            if f.supplier_id and supplier_lookup:
                f.supplier_name = supplier_lookup.get(f.supplier_id)
        return findings

    def _policy_price_benchmark_variance(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]

        supplier_id = self._resolve_supplier_id(self._get_condition(input_data, "supplier_id"))
        if not supplier_id:
            self._log_policy_event(
                policy_id, None, "blocked", "Supplier not recognised in master data", {}
            )
            return findings

        item_id = self._get_condition(input_data, "item_id")
        actual_price = self._to_float(self._get_condition(input_data, "actual_price"))
        benchmark_price = self._to_float(self._get_condition(input_data, "benchmark_price"))
        quantity = max(self._to_float(self._get_condition(input_data, "quantity", 1.0), 1.0), 1.0)
        threshold_pct = self._to_float(
            self._get_condition(input_data, "variance_threshold_pct", 0.0)
        )

        if benchmark_price <= 0:
            self._log_policy_event(
                policy_id,
                supplier_id,
                "blocked",
                "Benchmark price must be positive",
                {},
            )
            return findings

        variance_pct = (actual_price - benchmark_price) / benchmark_price
        if actual_price <= benchmark_price or variance_pct <= threshold_pct:
            self._log_policy_event(
                policy_id,
                supplier_id,
                "no_action",
                "Variance within threshold",
                {"variance_pct": variance_pct},
            )
            return findings

        impact = (actual_price - benchmark_price) * quantity
        po_ids = self._po_ids_for_supplier_item(tables, supplier_id, item_id)
        invoice_ids = self._invoice_ids_for_po(tables, po_ids)
        sources = po_ids + invoice_ids
        if not sources and item_id is not None:
            sources = [str(item_id)]
        category_id = self._category_for_po(po_ids[0]) if po_ids else None
        details = {
            "actual_price": actual_price,
            "benchmark_price": benchmark_price,
            "variance_pct": variance_pct,
            "quantity": quantity,
        }

        findings.append(
            self._build_finding(
                detector,
                supplier_id,
                category_id,
                item_id,
                impact,
                details,
                sources,
            )
        )
        self._record_escalation(
            policy_id, supplier_id, "Price variance threshold breached", details
        )
        self._default_notifications(notifications)
        return findings

    def _policy_volume_consolidation(

        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        minimum_volume = self._to_float(
            self._get_condition(input_data, "minimum_volume_gbp", 0.0)
        )

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if po_lines.empty:
            self._log_policy_event(
                policy_id, None, "no_action", "No purchase order line data available", {}
            )
            return findings

        value_col = self._choose_first_column(
            po_lines,
            [
                "line_amount_gbp",
                "total_amount_incl_tax_gbp",
                "line_amount",
                "total_amount_incl_tax",
            ],
        )
        price_col = self._choose_first_column(po_lines, ["unit_price_gbp", "unit_price"])
        qty_col = "quantity" if "quantity" in po_lines.columns else None
        if value_col is None:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to determine PO line value column",
                {},
            )
            return findings

        df = po_lines.dropna(subset=["po_id", "item_id"]).copy()
        if df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Purchase order lines missing item references",
                {},
            )
            return findings
        df["po_id"] = df["po_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df[value_col] = df[value_col].fillna(0.0)
        if price_col:
            df[price_col] = df[price_col].fillna(0.0)
        if qty_col:
            df[qty_col] = df[qty_col].fillna(0.0)
        df["supplier_id"] = df["po_id"].map(self._po_supplier_map)
        df = df.dropna(subset=["supplier_id"])
        df["supplier_id"] = df["supplier_id"].astype(str)

        if df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier-linked PO lines for consolidation",
                {},
            )
            return findings

        agg_map: Dict[str, str] = {value_col: "sum"}
        if price_col:
            agg_map[price_col] = "mean"
        if qty_col:
            agg_map[qty_col] = "sum"
        summary = (
            df.groupby(["item_id", "supplier_id"], as_index=False)
            .agg(agg_map)
            .rename(columns={value_col: "total_spend"})
        )
        if price_col:
            summary = summary.rename(columns={price_col: "avg_price"})
        else:
            summary["avg_price"] = summary["total_spend"]
        if qty_col:
            summary = summary.rename(columns={qty_col: "total_qty"})
        else:
            summary["total_qty"] = summary.apply(
                lambda r: r["total_spend"] / r["avg_price"]
                if r["avg_price"] not in (0, None)
                else 0.0,
                axis=1,
            )

        summary["supplier_id"] = summary["supplier_id"].map(self._resolve_supplier_id)
        summary = summary.dropna(subset=["supplier_id"])
        if summary.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No suppliers matched consolidation criteria",
                {},
            )
            return findings

        for item_id, item_df in summary.groupby("item_id"):
            total_spend = item_df["total_spend"].sum()
            if total_spend < minimum_volume or len(item_df) < 2:
                continue
            cheapest = item_df.sort_values("avg_price").iloc[0]
            for _, row in item_df.iterrows():
                supplier = row["supplier_id"]
                if supplier == cheapest["supplier_id"]:
                    continue
                qty = row.get("total_qty", 0.0)
                if qty <= 0:
                    qty = row["total_spend"] / cheapest["avg_price"] if cheapest["avg_price"] else row["total_spend"]
                savings_per_unit = row["avg_price"] - cheapest["avg_price"]
                potential_savings = max(0.0, savings_per_unit * qty)
                if potential_savings <= 0:
                    continue
                po_ids = self._po_ids_for_supplier_item(tables, supplier, item_id)
                category_id = self._category_for_po(po_ids[0]) if po_ids else None
                details = {
                    "current_supplier_price": row["avg_price"],
                    "best_supplier": cheapest["supplier_id"],
                    "best_price": cheapest["avg_price"],
                    "total_item_spend": total_spend,
                }
                findings.append(
                    self._build_finding(
                        detector,
                        supplier,
                        category_id,
                        item_id,
                        potential_savings,
                        details,
                        po_ids or [item_id],
                    )
                )
                self._log_policy_event(
                    policy_id,
                    supplier,
                    "detected",
                    "Volume consolidation opportunity identified",
                    details,
                )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No consolidation opportunities above threshold",
                {"minimum_volume_gbp": minimum_volume},
            )
        return findings

    def _contract_portfolio_fallback(
        self,
        tables: Dict[str, pd.DataFrame],
        detector: str,
        reference_date: date,
        ranking: Optional[Iterable[Dict[str, Any]]],
        limit: int = 3,
    ) -> List[Finding]:
        contracts = tables.get("contracts", pd.DataFrame())
        if contracts.empty or "supplier_id" not in contracts.columns or "contract_id" not in contracts.columns:
            return []

        df = contracts.dropna(subset=["supplier_id", "contract_id"]).copy()
        if df.empty:
            return []

        df["supplier_id"] = df["supplier_id"].astype(str).str.strip()
        df["contract_id"] = df["contract_id"].astype(str).str.strip()
        df = df[(df["supplier_id"] != "") & (df["contract_id"] != "")]
        if df.empty:
            return []

        if self._supplier_lookup:
            df = df[df["supplier_id"].isin(self._supplier_lookup)]
            if df.empty:
                return []

        value_col = self._choose_first_column(
            df,
            [
                "total_contract_value_gbp",
                "total_contract_value",
                "contract_value_gbp",
                "contract_value",
                "adjusted_price_gbp",
                "adjusted_price",
            ],
        )
        if value_col:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        else:
            value_col = "__contract_value__"
            df[value_col] = 0.0

        if "contract_end_date" in df.columns:
            end_dates = pd.to_datetime(df["contract_end_date"], errors="coerce").dt.date
            df["contract_end_date"] = end_dates
            if reference_date:
                mask = end_dates.isna() | (end_dates >= reference_date)
                if mask.any():
                    df = df[mask].copy()

        if df.empty:
            return []

        category_col = self._choose_first_column(df, ["category_id", "spend_category"])
        group = df.groupby("supplier_id", dropna=True)
        aggregated = group[value_col].sum().to_frame(name="total_value")
        aggregated["contract_count"] = group["contract_id"].nunique()
        aggregated["contract_ids"] = group["contract_id"].apply(
            lambda s: sorted({str(v) for v in s.dropna().astype(str)})
        )
        if "contract_end_date" in df.columns:
            def _next_expiry(series: pd.Series) -> Optional[date]:
                dates = [d for d in series if isinstance(d, date)]
                return min(dates) if dates else None

            aggregated["next_expiry"] = group["contract_end_date"].apply(_next_expiry)

        if category_col:
            aggregated["primary_category"] = group[category_col].apply(
                lambda s: s.dropna().astype(str).mode().iloc[0]
                if not s.dropna().empty
                else None
            )

        aggregated = aggregated.reset_index()

        supplier_master = tables.get("supplier_master", pd.DataFrame())
        if not supplier_master.empty and "supplier_id" in supplier_master.columns:
            supplier_info = supplier_master.dropna(subset=["supplier_id"]).copy()
            supplier_info["supplier_id"] = supplier_info["supplier_id"].astype(str).str.strip()
            info_cols = [
                col
                for col in [
                    "supplier_name",
                    "supplier_type",
                    "risk_score",
                    "is_preferred_supplier",
                    "registered_country",
                    "default_currency",
                ]
                if col in supplier_info.columns
            ]
            if info_cols:
                aggregated = aggregated.merge(
                    supplier_info[["supplier_id"] + info_cols],
                    on="supplier_id",
                    how="left",
                )

        ranking_df = pd.DataFrame(list(ranking or []))
        if not ranking_df.empty and "supplier_id" in ranking_df.columns:
            ranking_df = ranking_df.dropna(subset=["supplier_id"]).copy()
            ranking_df["supplier_id"] = ranking_df["supplier_id"].astype(str).str.strip()
            rename_map = {
                col: f"ranking_{col}"
                for col in ["final_score", "justification", "risk_score", "total_spend"]
                if col in ranking_df.columns
            }
            ranking_df = ranking_df.rename(columns=rename_map)
            ranking_cols = [col for col in ranking_df.columns if col != "supplier_id"]
            if ranking_cols:
                aggregated = aggregated.merge(
                    ranking_df[["supplier_id"] + ranking_cols].drop_duplicates("supplier_id"),
                    on="supplier_id",
                    how="left",
                )
            if "ranking_final_score" in aggregated.columns:
                aggregated["final_score_numeric"] = pd.to_numeric(
                    aggregated["ranking_final_score"], errors="coerce"
                )

        sort_cols: List[str] = []
        ascending: List[bool] = []
        if "final_score_numeric" in aggregated.columns:
            sort_cols.append("final_score_numeric")
            ascending.append(False)
        sort_cols.append("total_value")
        ascending.append(False)
        aggregated = aggregated.sort_values(sort_cols, ascending=ascending, na_position="last")

        findings: List[Finding] = []
        for _, row in aggregated.head(max(limit, 1)).iterrows():
            supplier_id = self._resolve_supplier_id(row.get("supplier_id"))
            if not supplier_id:
                continue
            impact = self._to_float(row.get("total_value"), 0.0)
            contract_ids = row.get("contract_ids")
            sources = (
                [str(src) for src in contract_ids if src]
                if isinstance(contract_ids, list)
                else []
            )
            details: Dict[str, Any] = {
                "analysis_type": "contract_portfolio_fallback",
                "contract_count": int(row.get("contract_count", 0) or 0),
            }
            if impact:
                details["total_contract_value_gbp"] = impact
            next_expiry = row.get("next_expiry")
            if isinstance(next_expiry, date):
                details["next_contract_end_date"] = str(next_expiry)
            supplier_name = row.get("supplier_name")
            if isinstance(supplier_name, str) and supplier_name:
                details["supplier_name"] = supplier_name
            ranking_score = row.get("ranking_final_score")
            if pd.notna(ranking_score):
                details["ranking_final_score"] = self._to_float(ranking_score)
            ranking_justification = row.get("ranking_justification")
            if isinstance(ranking_justification, str) and ranking_justification.strip():
                details["ranking_justification"] = ranking_justification.strip()
            ranking_spend = row.get("ranking_total_spend")
            if pd.notna(ranking_spend):
                details["ranking_total_spend"] = self._to_float(ranking_spend)
            ranking_risk = row.get("ranking_risk_score")
            if pd.notna(ranking_risk):
                details["ranking_risk_score"] = self._to_float(ranking_risk)
            risk_score = row.get("risk_score")
            if pd.notna(risk_score):
                details["risk_score"] = self._to_float(risk_score)
            if "is_preferred_supplier" in row:
                pref = row.get("is_preferred_supplier")
                if pd.isna(pref):
                    pref_value = None
                elif isinstance(pref, bool):
                    pref_value = pref
                else:
                    pref_value = str(pref).strip().lower() in {"true", "1", "y", "yes"}
                details["is_preferred_supplier"] = pref_value
            category_id = row.get("primary_category") if category_col else None
            finding = self._build_finding(
                f"{detector} - Portfolio",
                supplier_id,
                category_id,
                None,
                impact,
                details,
                sources,
            )
            if isinstance(supplier_name, str) and supplier_name:
                finding.supplier_name = supplier_name
            findings.append(finding)

        return findings

    def _policy_contract_expiry(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        window_days = self._to_int(self._get_condition(input_data, "negotiation_window_days"), 0)
        if window_days <= 0:
            self._log_policy_event(
                policy_id,
                None,
                "blocked",
                "Negotiation window must be greater than zero",
                {"negotiation_window_days": window_days},
            )
            return findings

        reference_date = self._to_date(self._get_condition(input_data, "reference_date"))
        contracts = tables.get("contracts", pd.DataFrame())
        if contracts.empty or "contract_id" not in contracts.columns:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No contract data available for expiry analysis",
                {},
            )
            return findings

        contract_value_col = self._choose_first_column(
            contracts, ["total_contract_value_gbp", "total_contract_value"]
        )
        category_col = "category_id" if "category_id" in contracts.columns else "spend_category"
        end_dates = pd.to_datetime(contracts.get("contract_end_date"), errors="coerce")
        for idx, row in contracts.iterrows():
            end_date = end_dates.iloc[idx]
            if pd.isna(end_date):
                continue
            end_date = end_date.date()
            days_to_expiry = (end_date - reference_date).days
            if days_to_expiry < 0 or days_to_expiry > window_days:
                continue
            supplier_id = self._resolve_supplier_id(row.get("supplier_id"))
            if not supplier_id:
                continue
            impact = self._to_float(row.get(contract_value_col, 0.0)) if contract_value_col else 0.0
            details = {
                "contract_title": row.get("contract_title"),
                "contract_end_date": str(end_date),
                "days_to_expiry": days_to_expiry,
            }
            category_id = row.get(category_col) if category_col in row else None
            sources = [row.get("contract_id")]
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    category_id,
                    row.get("contract_id"),
                    impact,
                    details,
                    [s for s in sources if s],
                )
            )
            self._record_escalation(
                policy_id,
                supplier_id,
                "Contract expiring within negotiation window",
                details,
            )

        if findings:
            self._default_notifications(notifications)
            return findings

        fallback_findings = self._contract_portfolio_fallback(
            tables,
            detector,
            reference_date,
            input_data.get("ranking"),
        )
        if fallback_findings:
            supplier_ids = [f.supplier_id for f in fallback_findings if f.supplier_id]
            self._log_policy_event(
                policy_id,
                None,
                "fallback",
                "No contracts expiring within negotiation window; surfaced top contract suppliers",
                {
                    "negotiation_window_days": window_days,
                    "supplier_candidates": supplier_ids,
                },
            )
            self._default_notifications(notifications)
            return fallback_findings

        self._log_policy_event(
            policy_id,
            None,
            "no_action",
            "No contracts expiring within negotiation window",
            {"negotiation_window_days": window_days},
        )
        return findings

    def _policy_supplier_risk(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        threshold = self._to_float(self._get_condition(input_data, "risk_threshold", 0.0))
        risk_weight = self._to_float(self._get_condition(input_data, "risk_weight", 1000.0))

        supplier_master = tables.get("supplier_master", pd.DataFrame())
        risk_col = "risk_score" if "risk_score" in supplier_master.columns else None
        if risk_col:
            risk_df = supplier_master[["supplier_id", risk_col]].dropna()
        else:
            if not self._supplier_risk_map:
                self._load_supplier_risk_map()
            risk_df = pd.DataFrame(
                {
                    "supplier_id": list(self._supplier_risk_map.keys()),
                    "risk_score": list(self._supplier_risk_map.values()),
                }
            )
            risk_col = "risk_score"

        if risk_df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier risk metrics available",
                {},
            )
            return findings

        for _, row in risk_df.iterrows():
            supplier_id = self._resolve_supplier_id(row.get("supplier_id"))
            if not supplier_id:
                continue
            risk_score = self._to_float(row.get(risk_col, 0.0))
            if risk_score < threshold:
                continue
            impact = risk_score * risk_weight
            details = {"risk_score": risk_score, "threshold": threshold}
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    None,
                    None,
                    impact,
                    details,
                    [supplier_id],
                )
            )
            self._record_escalation(
                policy_id, supplier_id, "Supplier risk threshold breached", details
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No suppliers breached the risk threshold",
                {"risk_threshold": threshold},
            )
        return findings

    def _policy_maverick_spend(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        minimum_value = self._to_float(self._get_condition(input_data, "minimum_value_gbp", 0.0))

        purchase_orders = tables.get("purchase_orders", pd.DataFrame())
        if purchase_orders.empty or "po_id" not in purchase_orders.columns:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No purchase order header data available",
                {},
            )
            return findings

        value_col = self._choose_first_column(
            purchase_orders, ["total_amount_gbp", "total_amount"]
        )
        if value_col is None:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to determine PO total value column",
                {},
            )
            return findings

        for _, row in purchase_orders.iterrows():
            po_id = row.get("po_id")
            contract_id = row.get("contract_id")
            if contract_id and contract_id in self._contract_supplier_map:
                continue
            supplier_id = self._resolve_supplier_id(row.get("supplier_id") or self._po_supplier_map.get(str(po_id)))
            if not supplier_id:
                continue
            value = self._to_float(row.get(value_col, 0.0))
            if value < minimum_value:
                continue
            details = {"po_total": value, "contract_id": contract_id}
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    None,
                    None,
                    value,
                    details,
                    [str(po_id)],
                )
            )
            self._record_escalation(
                policy_id,
                supplier_id,
                "PO issued without approved contract",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No maverick spend identified above threshold",
                {"minimum_value_gbp": minimum_value},
            )
        return findings

    def _policy_duplicate_supplier(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        minimum_overlap = self._to_float(
            self._get_condition(input_data, "minimum_overlap_gbp", 0.0)
        )

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if po_lines.empty or "item_id" not in po_lines.columns:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No PO line data available for duplicate supplier detection",
                {},
            )
            return findings

        value_col = self._choose_first_column(
            po_lines,
            [
                "line_amount_gbp",
                "total_amount_incl_tax_gbp",
                "line_amount",
                "total_amount_incl_tax",
            ],
        )
        if value_col is None:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to determine PO line value column",
                {},
            )
            return findings

        df = po_lines.dropna(subset=["item_id", "po_id"]).copy()
        df["po_id"] = df["po_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df[value_col] = df[value_col].fillna(0.0)
        df["supplier_id"] = df["po_id"].map(self._po_supplier_map)
        df = df.dropna(subset=["supplier_id"])
        df["supplier_id"] = df["supplier_id"].astype(str)

        if df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier-linked PO lines for duplicate detection",
                {},
            )
            return findings

        summary = (
            df.groupby(["item_id", "supplier_id"], as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "spend"})
        )

        for item_id, item_df in summary.groupby("item_id"):
            total_spend = item_df["spend"].sum()
            suppliers = [self._resolve_supplier_id(s) for s in item_df["supplier_id"].tolist()]
            suppliers = [s for s in suppliers if s]
            if len(suppliers) < 2 or total_spend < minimum_overlap:
                continue
            max_row = item_df.loc[item_df["spend"].idxmax()]
            supplier_id = self._resolve_supplier_id(max_row["supplier_id"])
            if not supplier_id:
                continue
            po_ids = self._po_ids_for_supplier_item(tables, supplier_id, item_id)
            category_id = self._category_for_po(po_ids[0]) if po_ids else None
            details = {
                "supplier_count": len(suppliers),
                "suppliers": suppliers,
                "total_spend": total_spend,
            }
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    category_id,
                    item_id,
                    total_spend,
                    details,
                    po_ids or [item_id],
                )
            )
            self._log_policy_event(
                policy_id,
                supplier_id,
                "detected",
                "Duplicate supplier identified for SKU/service",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No duplicate suppliers detected above threshold",
                {"minimum_overlap_gbp": minimum_overlap},
            )
        return findings

    def _policy_category_overspend(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        budgets = self._get_condition(input_data, "category_budgets", {})
        if not isinstance(budgets, dict) or not budgets:
            self._log_policy_event(
                policy_id,
                None,
                "blocked",
                "Category budgets must be provided as a dictionary",
                {},
            )
            return findings

        invoice_lines = tables.get("invoice_lines", pd.DataFrame())
        if invoice_lines.empty or "po_id" not in invoice_lines.columns:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No invoice line data for overspend analysis",
                {},
            )
            return findings

        amount_col = self._choose_first_column(
            invoice_lines,
            [
                "line_amount_gbp",
                "total_amount_incl_tax_gbp",
                "line_amount",
                "total_amount_incl_tax",
            ],
        )
        if amount_col is None:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to determine invoice line amount column",
                {},
            )
            return findings

        df = invoice_lines.dropna(subset=["po_id"]).copy()
        df["po_id"] = df["po_id"].astype(str)
        df[amount_col] = df[amount_col].fillna(0.0)
        df["category_id"] = df["po_id"].map(self._category_for_po)
        df["supplier_id"] = df["po_id"].map(self._po_supplier_map)
        df["contract_id"] = df["po_id"].map(self._po_contract_map)
        if df["supplier_id"].isna().any():
            df.loc[df["supplier_id"].isna(), "supplier_id"] = df.loc[
                df["supplier_id"].isna(), "contract_id"
            ].map(lambda cid: self._contract_supplier_map.get(cid) if cid else None)
        df = df.dropna(subset=["category_id", "supplier_id"])
        if df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to map invoice lines to categories",
                {},
            )
            return findings

        spend = (
            df.groupby("category_id", as_index=False)[amount_col]
            .sum()
            .rename(columns={amount_col: "actual_spend"})
        )
        supplier_spend = (
            df.groupby(["category_id", "supplier_id"], as_index=False)[amount_col]
            .sum()
            .rename(columns={amount_col: "supplier_spend"})
        )

        for _, row in spend.iterrows():
            category_id = row["category_id"]
            budget = self._to_float(budgets.get(category_id))
            actual = row["actual_spend"]
            if budget <= 0 or actual <= budget:
                continue
            supplier_rows = supplier_spend[supplier_spend["category_id"] == category_id]
            if supplier_rows.empty:
                continue
            top_supplier_row = supplier_rows.sort_values("supplier_spend", ascending=False).iloc[0]
            supplier_id = self._resolve_supplier_id(top_supplier_row["supplier_id"])
            if not supplier_id:
                continue
            impact = actual - budget
            invoice_ids = df[df["category_id"] == category_id].get("invoice_id")
            sources = invoice_ids.dropna().astype(str).unique().tolist() if invoice_ids is not None else []
            details = {
                "actual_spend": actual,
                "budget": budget,
                "supplier_breakdown": supplier_rows.set_index("supplier_id")[
                    "supplier_spend"
                ].to_dict(),
            }
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    category_id,
                    None,
                    impact,
                    details,
                    sources or [category_id],
                )
            )
            self._record_escalation(
                policy_id,
                supplier_id,
                "Category spend exceeded budget",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Category spend within allocated budgets",
                {},
            )
        return findings

    def _policy_inflation_passthrough(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        market_inflation = self._to_float(
            self._get_condition(input_data, "market_inflation_pct", 0.0)
        )
        tolerance = self._to_float(self._get_condition(input_data, "tolerance_pct", 0.0))
        allowed_increase = market_inflation + tolerance

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        invoice_lines = tables.get("invoice_lines", pd.DataFrame())
        if po_lines.empty or invoice_lines.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Insufficient PO or invoice data for inflation analysis",
                {},
            )
            return findings

        po_price_col = self._choose_first_column(po_lines, ["unit_price_gbp", "unit_price"])
        inv_price_col = self._choose_first_column(invoice_lines, ["unit_price_gbp", "unit_price"])
        qty_col = "quantity" if "quantity" in invoice_lines.columns else None
        if po_price_col is None or inv_price_col is None:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to determine unit price columns for inflation analysis",
                {},
            )
            return findings

        po_summary = po_lines.dropna(subset=["po_id", "item_id"]).copy()
        po_summary["po_id"] = po_summary["po_id"].astype(str)
        po_summary["item_id"] = po_summary["item_id"].astype(str)
        po_summary["supplier_id"] = po_summary["po_id"].map(self._po_supplier_map)
        po_summary = po_summary.dropna(subset=["supplier_id"])
        po_group = (
            po_summary.groupby(["supplier_id", "item_id"], as_index=False)[po_price_col]
            .mean()
            .rename(columns={po_price_col: "po_avg_price"})
        )

        inv_summary = invoice_lines.dropna(subset=["po_id", "item_id"]).copy()
        inv_summary["po_id"] = inv_summary["po_id"].astype(str)
        inv_summary["item_id"] = inv_summary["item_id"].astype(str)
        inv_summary["supplier_id"] = inv_summary["invoice_id"].map(self._invoice_supplier_map)
        inv_summary.loc[inv_summary["supplier_id"].isna(), "supplier_id"] = inv_summary[
            "po_id"
        ].map(self._po_supplier_map)
        inv_summary = inv_summary.dropna(subset=["supplier_id"])
        agg_map = {inv_price_col: "mean"}
        if qty_col and qty_col in inv_summary.columns:
            agg_map[qty_col] = "sum"
        inv_group = (
            inv_summary.groupby(["supplier_id", "item_id"], as_index=False)
            .agg(agg_map)
            .rename(columns={inv_price_col: "invoice_avg_price"})

        )
        if qty_col and qty_col in inv_group.columns:
            inv_group = inv_group.rename(columns={qty_col: "total_qty"})
        else:
            inv_group["total_qty"] = 0.0

        merged = inv_group.merge(po_group, on=["supplier_id", "item_id"], how="inner")
        if merged.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No overlapping PO and invoice data for inflation analysis",
                {},
            )
            return findings

        for _, row in merged.iterrows():
            supplier_id = self._resolve_supplier_id(row.get("supplier_id"))
            if not supplier_id:
                continue
            po_price = self._to_float(row.get("po_avg_price", 0.0))
            invoice_price = self._to_float(row.get("invoice_avg_price", 0.0))
            if po_price <= 0:
                continue
            increase_pct = (invoice_price - po_price) / po_price
            if increase_pct <= allowed_increase:
                continue
            qty = self._to_float(row.get("total_qty", 0.0))
            if qty <= 0:
                qty = 1.0
            impact = (invoice_price - po_price) * qty
            po_ids = self._po_ids_for_supplier_item(tables, supplier_id, row.get("item_id"))
            invoice_ids = inv_summary[
                (inv_summary["supplier_id"].astype(str) == supplier_id)
                & (inv_summary["item_id"].astype(str) == str(row.get("item_id")))
            ]["invoice_id"].dropna().astype(str).unique().tolist()
            details = {
                "po_price": po_price,
                "invoice_price": invoice_price,
                "increase_pct": increase_pct,
                "market_allowance_pct": allowed_increase,
            }
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    self._category_for_po(po_ids[0]) if po_ids else None,
                    row.get("item_id"),
                    impact,
                    details,
                    po_ids + invoice_ids,
                )
            )
            self._record_escalation(
                policy_id,
                supplier_id,
                "Inflation pass-through exceeds market allowance",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No inflation pass-through exceptions identified",
                {"market_inflation_pct": market_inflation, "tolerance_pct": tolerance},
            )
        return findings

    def _policy_unused_contract_value(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        minimum_unused = self._to_float(
            self._get_condition(input_data, "minimum_unused_value_gbp", 0.0)
        )

        contracts = tables.get("contracts", pd.DataFrame())
        invoice_lines = tables.get("invoice_lines", pd.DataFrame())
        if contracts.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No contract data available",
                {},
            )
            return findings

        amount_col = self._choose_first_column(
            invoice_lines,
            [
                "line_amount_gbp",
                "total_amount_incl_tax_gbp",
                "line_amount",
                "total_amount_incl_tax",
            ],
        )
        contract_value_col = self._choose_first_column(
            contracts, ["total_contract_value_gbp", "total_contract_value"]
        )

        if invoice_lines.empty or amount_col is None:
            spend_by_contract: Dict[str, float] = {cid: 0.0 for cid in contracts.get("contract_id", [])}
        else:
            df = invoice_lines.dropna(subset=["po_id"]).copy()
            df["po_id"] = df["po_id"].astype(str)
            df[amount_col] = df[amount_col].fillna(0.0)
            df["contract_id"] = df["po_id"].map(self._po_contract_map)
            df = df.dropna(subset=["contract_id"])
            spend_by_contract = (
                df.groupby("contract_id", as_index=False)[amount_col]
                .sum()
                .set_index("contract_id")[amount_col]
                .to_dict()
            )

        for _, row in contracts.iterrows():
            contract_id = row.get("contract_id")
            if not contract_id:
                continue
            contract_value = self._to_float(row.get(contract_value_col, 0.0))
            spend = spend_by_contract.get(str(contract_id), 0.0)
            unused = contract_value - spend
            if unused < minimum_unused:
                continue
            supplier_id = self._resolve_supplier_id(row.get("supplier_id"))
            if not supplier_id:
                continue
            invoice_ids: List[str] = []
            if not invoice_lines.empty and "invoice_id" in invoice_lines.columns:
                invoice_ids = (
                    invoice_lines[
                        invoice_lines["po_id"].astype(str).isin(
                            [po for po, cid in self._po_contract_map.items() if cid == contract_id]
                        )
                    ]["invoice_id"].dropna().astype(str).unique().tolist()
                )
            details = {
                "contract_value": contract_value,
                "actual_spend": spend,
            }
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    row.get("spend_category"),
                    None,
                    unused,
                    details,
                    invoice_ids or [contract_id],
                )
            )
            self._log_policy_event(
                policy_id,
                supplier_id,
                "detected",
                "Unused contract value identified",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "All contracts are sufficiently utilised",
                {},
            )
        return findings

    def _policy_supplier_performance(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        records = self._get_condition(input_data, "performance_records", [])
        if not isinstance(records, list) or not records:
            self._log_policy_event(
                policy_id,
                None,
                "blocked",
                "Performance records must be provided as a list",
                {},
            )
            return findings

        for record in records:
            if not isinstance(record, dict):
                continue
            supplier_id = self._resolve_supplier_id(record.get("supplier_id"))
            if not supplier_id:
                continue
            threshold = self._to_float(
                record.get("threshold")
                or record.get("sla_threshold")
                or record.get("target")
                or 0.0
            )
            score = self._to_float(record.get("score") or record.get("sla_score"))
            if threshold <= 0 or score >= threshold:
                continue
            impact = self._to_float(record.get("impact_gbp", 0.0))
            if impact == 0.0:
                penalty = self._to_float(record.get("penalty_gbp", 0.0))
                impact = max(0.0, threshold - score) * penalty
            details = {
                "metric": record.get("metric") or record.get("kpi"),
                "score": score,
                "threshold": threshold,
            }
            sources = record.get("evidence_ids") or []
            if isinstance(sources, str):
                sources = [sources]
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    record.get("category_id"),
                    record.get("item_id"),
                    impact,
                    details,
                    sources if isinstance(sources, list) else [str(sources)],
                )
            )
            self._record_escalation(
                policy_id,
                supplier_id,
                "Supplier performance deviation detected",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier performance deviations detected",
                {},
            )
        return findings

    def _policy_esg_opportunity(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = policy_cfg["policy_id"]
        detector = policy_cfg["detector"]
        esg_scores = self._get_condition(input_data, "esg_scores", [])
        if not isinstance(esg_scores, list) or not esg_scores:
            self._log_policy_event(
                policy_id,
                None,
                "blocked",
                "ESG scores must be provided as a list",
                {},
            )
            return findings

        incumbent_score = self._to_float(self._get_condition(input_data, "incumbent_score", 0.0))
        esg_threshold = self._to_float(self._get_condition(input_data, "esg_threshold", 0.0))
        savings_estimate = self._to_float(
            self._get_condition(input_data, "estimated_switch_savings_gbp", 0.0)
        )
        category_id = self._get_condition(input_data, "category_id")

        for score_entry in esg_scores:
            if not isinstance(score_entry, dict):
                continue
            supplier_id = self._resolve_supplier_id(score_entry.get("supplier_id"))
            if not supplier_id:
                continue
            score = self._to_float(score_entry.get("score"))
            if score < esg_threshold or score <= incumbent_score:
                continue
            details = {
                "esg_score": score,
                "incumbent_score": incumbent_score,
            }
            impact = savings_estimate
            sources = [supplier_id]
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    category_id,
                    score_entry.get("item_id"),
                    impact,
                    details,
                    sources,
                )
            )
            self._record_escalation(
                policy_id,
                supplier_id,
                "ESG opportunity identified with superior supplier",
                details,
            )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No ESG opportunities above threshold",
                {"esg_threshold": esg_threshold},
            )
        return findings


    def _load_supplier_risk_map(self) -> Dict[str, float]:
        """Load supplier risk scores from `proc.supplier` into a map: supplier_id -> risk_score."""
        self._supplier_risk_map = {}
        try:
            df = self._read_sql(
                "SELECT supplier_id, COALESCE(risk_score, 0.0) AS risk_score FROM proc.supplier"
            )
            if not df.empty:
                self._supplier_risk_map = dict(zip(df["supplier_id"], df["risk_score"]))
        except Exception:
            self._supplier_risk_map = {}
        return self._supplier_risk_map

    def _attach_vector_context(self, findings: List[Finding]) -> None:
        """Enrich findings with related documents from the vector store."""
        for f in findings:
            query_parts = [p for p in [f.supplier_id, f.category_id, f.item_id] if p]
            if not query_parts:
                continue
            hits = self.vector_search(" ".join(map(str, query_parts)), top_k=3)
            f.context_documents = [h.payload for h in hits]

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
                with self.agent_nick.pandas_connection() as conn:
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

        po_price_expr = self._price_expression(
            "proc", "po_line_items_agent", "li"
        )
        inv_price_expr = self._price_expression(
            "proc", "invoice_line_items_agent", "ili"
        )

        sql = f"""
            WITH po_suppliers AS (
                SELECT p.supplier_id,
                       {po_price_expr} AS unit_price
                FROM proc.po_line_items_agent li
                JOIN proc.purchase_order_agent p ON p.po_id = li.po_id
                WHERE li.item_id = %s AND p.supplier_id IS NOT NULL
            ), invoice_suppliers AS (
                SELECT ia.supplier_id,
                       {inv_price_expr} AS unit_price
                FROM proc.invoice_line_items_agent ili
                JOIN proc.invoice_agent ia ON ia.invoice_id = ili.invoice_id
                WHERE ili.item_id = %s AND ia.supplier_id IS NOT NULL
            )
            SELECT supplier_id, unit_price FROM po_suppliers
            UNION ALL
            SELECT supplier_id, unit_price FROM invoice_suppliers
        """
        try:
            df = self._read_sql(sql, params=(item_id, item_id))
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
        supplier_lookup = getattr(self, "_supplier_lookup", {})
        if supplier_lookup:
            filtered: List[Dict[str, Any]] = []
            for candidate in result:
                supplier_id = str(candidate.get("supplier_id"))
                if supplier_id not in supplier_lookup:
                    logger.debug(
                        "Candidate supplier %s missing from supplier master; skipping",
                        supplier_id,
                    )
                    continue
                candidate["supplier_id"] = supplier_id
                supplier_name = supplier_lookup.get(supplier_id)
                if supplier_name:
                    candidate["supplier_name"] = supplier_name
                filtered.append(candidate)
            result = filtered
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

