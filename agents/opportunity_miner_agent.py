from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.opportunity_service import load_opportunity_feedback
from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources

logger = logging.getLogger(__name__)


_CATALOG_MATCH_THRESHOLD = 0.45


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
    item_reference: Optional[str] = None
    policy_id: Optional[str] = None
    feedback_status: Optional[str] = None
    feedback_reason: Optional[str] = None
    feedback_updated_at: Optional[datetime] = None
    feedback_user: Optional[str] = None
    feedback_metadata: Optional[Dict[str, Any]] = None
    is_rejected: bool = False

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        detected = self.detected_on
        if isinstance(detected, datetime):
            d["detected_on"] = detected.isoformat()
        else:
            d["detected_on"] = str(detected)
        updated = self.feedback_updated_at
        if isinstance(updated, datetime):
            d["feedback_updated_at"] = updated.isoformat()
        elif updated is not None:
            d["feedback_updated_at"] = str(updated)
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

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def _instruction_sources_from_prompt(self, prompt: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(prompt, dict):
            return sources
        for field in ("prompt_config", "metadata", "prompts_desc", "template"):
            value = prompt.get(field)
            if value:
                sources.append(value)
        return sources

    def _instruction_sources_from_policy(self, policy: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(policy, dict):
            return sources
        for field in ("policy_details", "details", "policy_desc", "description"):
            value = policy.get(field)
            if value:
                sources.append(value)
        return sources

    def _apply_instruction_overrides(
        self, context: AgentContext, instructions: Dict[str, Any]
    ) -> None:
        if not isinstance(instructions, dict):
            return

        workflow_hint = instructions.get("workflow") or instructions.get("workflow_name")
        if workflow_hint:
            candidate = str(workflow_hint).strip()
            if candidate:
                context.input_data["workflow"] = candidate

        min_impact = (
            instructions.get("min_financial_impact")
            or instructions.get("minimum_financial_impact")
            or instructions.get("financial_impact_threshold")
        )
        threshold = self._coerce_float(min_impact)
        if threshold is not None and threshold >= 0:
            context.input_data["min_financial_impact"] = threshold

        conditions = context.input_data.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
            context.input_data["conditions"] = conditions

        alias_map = {
            "negotiation_window_days": {
                "negotiation_window_days",
                "negotiation_window",
                "window_days",
                "renewal_window_days",
            },
            "lookback_period_days": {
                "lookback_period_days",
                "lookback_period",
                "lookback_days",
            },
            "reference_date": {"reference_date", "analysis_date", "base_date"},
            "risk_threshold": {"risk_threshold", "risk_score_threshold"},
            "minimum_volume_gbp": {
                "minimum_volume_gbp",
                "min_volume_gbp",
                "volume_threshold",
            },
            "minimum_value_gbp": {
                "minimum_value_gbp",
                "minimum_spend_gbp",
                "min_value_gbp",
            },
            "minimum_unused_value_gbp": {
                "minimum_unused_value_gbp",
                "unused_value_threshold",
            },
            "minimum_overlap_gbp": {"minimum_overlap_gbp", "overlap_threshold"},
            "market_inflation_pct": {"market_inflation_pct", "inflation_pct"},
            "minimum_quantity_threshold": {
                "minimum_quantity_threshold",
                "quantity_threshold",
            },
            "minimum_spend_threshold": {
                "minimum_spend_threshold",
                "spend_threshold",
            },
            "supplier_id": {
                "supplier_id",
                "supplier",
                "supplier_code",
                "vendor_id",
            },
            "item_id": {
                "item_id",
                "item",
                "item_code",
                "material_id",
                "sku",
            },
            "actual_price": {
                "actual_price",
                "current_price",
                "price_paid",
                "unit_price",
                "current_unit_price",
            },
            "benchmark_price": {
                "benchmark_price",
                "benchmark_unit_price",
                "target_price",
                "reference_price",
            },
            "quantity": {
                "quantity",
                "volume",
                "units",
                "unit_count",
                "order_quantity",
            },
            "variance_threshold_pct": {
                "variance_threshold_pct",
                "variance_threshold",
                "threshold_pct",
                "price_variance_threshold",
                "variance_pct_threshold",
            },
        }

        def _assign_condition(key: str, raw: Any) -> None:
            if raw is None:
                return
            if isinstance(raw, dict):
                conditions[key] = raw
                return
            if isinstance(raw, (list, tuple, set)):
                values = [v for v in raw if v is not None]
                if values:
                    conditions[key] = list(values)
                return
            if isinstance(raw, bool):
                conditions[key] = raw
                return
            numeric = self._coerce_float(raw)
            if numeric is not None:
                if key.endswith("_days"):
                    conditions[key] = int(numeric)
                else:
                    conditions[key] = numeric
                return
            text = str(raw).strip()
            if text:
                conditions[key] = text

        for canonical, aliases in alias_map.items():
            for alias in aliases:
                if alias in instructions:
                    _assign_condition(canonical, instructions[alias])
                    break

        def _merge_condition_dict(payload: Any) -> None:
            if not isinstance(payload, dict):
                return
            for raw_key, raw_value in payload.items():
                if raw_key is None:
                    continue
                norm_key = self._normalise_policy_slug(raw_key) or str(raw_key).strip()
                if not norm_key:
                    continue
                _assign_condition(norm_key, raw_value)

        for block_key in (
            "conditions",
            "parameters",
            "policy_parameters",
            "default_conditions",
        ):
            _merge_condition_dict(instructions.get(block_key))

        rules_block = instructions.get("rules")
        if isinstance(rules_block, dict):
            _merge_condition_dict(rules_block)
            for nested_key in ("parameters", "conditions"):
                _merge_condition_dict(rules_block.get(nested_key))

        for canonical, aliases in alias_map.items():
            if canonical in conditions:
                continue
            for alias in aliases:
                if alias in context.input_data:
                    _assign_condition(canonical, context.input_data[alias])
                    break

    def _apply_instruction_settings(self, context: AgentContext) -> None:
        sources: List[Any] = []
        prompts = context.input_data.get("prompts") or []
        for prompt in prompts:
            sources.extend(self._instruction_sources_from_prompt(prompt))
        policies = context.input_data.get("policies") or []
        for policy in policies:
            sources.extend(self._instruction_sources_from_policy(policy))
        instructions = parse_instruction_sources(sources)
        self._apply_instruction_overrides(context, instructions)

    def _apply_policy_category_limits(
        self, per_policy: Dict[str, List[Finding]]
    ) -> Tuple[List[Finding], Dict[str, Dict[str, List[Finding]]]]:

        """Limit retained findings to at most two per category for each policy."""

        aggregated: List[Finding] = []
        seen_ids: set[str] = set()
        category_map: Dict[str, Dict[str, List[Finding]]] = {}

        for display, items in per_policy.items():
            if not isinstance(items, list):
                per_policy[display] = []
                category_map[display] = {}
                continue

            valid_items: List[Finding] = [
                item for item in items if isinstance(item, Finding)
            ]
            if not valid_items:
                per_policy[display] = []
                category_map[display] = {}
                continue

            sorted_items = sorted(
                valid_items,
                key=lambda f: f.financial_impact_gbp,
                reverse=True,
            )

            categories: Dict[str, List[Finding]] = {}
            for finding in sorted_items:
                cat_raw = finding.category_id
                category = str(cat_raw).strip() if cat_raw else "uncategorized"
                categories.setdefault(category, []).append(finding)

            limited_by_category: Dict[str, List[Finding]] = {}
            for category, cat_findings in categories.items():
                top = cat_findings[:2]
                if top:
                    limited_by_category[category] = top

            if not limited_by_category:
                limited_by_category["uncategorized"] = [sorted_items[0]]

            flattened: List[Finding] = []
            for bucket in limited_by_category.values():
                flattened.extend(bucket)

            per_policy[display] = flattened
            category_map[display] = limited_by_category

            for finding in flattened:
                key = finding.opportunity_id or f"{display}:{id(finding)}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                aggregated.append(finding)

        return aggregated, category_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        """Entry point for the orchestration layer."""
        return self.process(context)

    def _read_sql(self, query: str, params: Any = None) -> pd.DataFrame:
        """Read a SQL query using a SQLAlchemy engine when available."""

        def _fetch_with_cursor(connection) -> pd.DataFrame:
            with connection.cursor() as cursor:
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return pd.DataFrame(rows, columns=columns)

        engine_getter = getattr(self.agent_nick, "get_db_engine", None)
        engine = engine_getter() if callable(engine_getter) else None
        if engine is not None:
            with engine.connect() as conn:
                return pd.read_sql(query, conn, params=params)

        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        if callable(pandas_conn):
            with pandas_conn() as conn:
                if hasattr(conn, "cursor"):
                    return _fetch_with_cursor(conn)
                return pd.read_sql(query, conn, params=params)

        conn_getter = getattr(self.agent_nick, "get_db_connection", None)
        if callable(conn_getter):
            with closing(conn_getter()) as conn:
                return _fetch_with_cursor(conn)

        logger.debug("No database connection available for query; returning empty DataFrame")
        return pd.DataFrame()

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

            self._apply_instruction_settings(context)

            min_impact_threshold = self._resolve_min_financial_impact(
                context.input_data
            )
            explicit_min_override = any(
                key in context.input_data
                for key in (
                    "min_financial_impact",
                    "minimum_financial_impact",
                    "financial_impact_threshold",
                )
            )
            conditions_payload = context.input_data.get("conditions")
            if isinstance(conditions_payload, dict):
                explicit_min_override = explicit_min_override or any(
                    key in conditions_payload
                    for key in (
                        "min_financial_impact",
                        "minimum_financial_impact",
                        "financial_impact_threshold",
                    )
                )

            policy_registry, provided_policies = self._assemble_policy_registry(
                context.input_data
            )

            if not policy_registry:
                if provided_policies:
                    message = (
                        "No supported opportunity policies matched the provided configuration"
                    )
                    details = {
                        "policies": [
                            policy.get("policyId")
                            or policy.get("policyName")
                            or policy.get("policy_desc")
                            for policy in provided_policies
                        ]
                    }
                    self._log_policy_event("unknown", None, "blocked", message, details)
                    return self._blocked_output(message)
                message = "No opportunity policies available"
                self._log_policy_event("unknown", None, "blocked", message, {})
                return self._blocked_output(message)

            workflow_hint = context.input_data.get("workflow")
            workflow_name = (
                str(workflow_hint).strip() if isinstance(workflow_hint, str) and workflow_hint.strip() else None
            )
            if workflow_name:
                context.input_data["workflow"] = workflow_name

            if not workflow_name and not provided_policies:
                message = "Workflow name is required to execute opportunity mining"
                self._log_policy_event("unknown", None, "blocked", message, {})
                return self._blocked_output(message)

            requested_keys: List[str] = []
            if workflow_name:
                tokens = {workflow_name.lower()}
                slug = self._normalise_policy_slug(workflow_name)
                if slug:
                    tokens.add(slug)
                for key, cfg in policy_registry.items():
                    canonical = cfg.get("policy_slug", key)
                    aliases = {alias.lower() for alias in cfg.get("aliases", set())}
                    aliases.add(canonical.lower())
                    if tokens & aliases:
                        requested_keys = [canonical]
                        context.input_data["workflow"] = canonical
                        workflow_name = canonical
                        break

            if not requested_keys:
                requested_keys = list(policy_registry.keys())
                if requested_keys:
                    if workflow_name:
                        normalized = self._normalise_policy_slug(workflow_name)
                        for key in list(requested_keys):
                            cfg = policy_registry.get(key) or {}
                            aliases = {alias.lower() for alias in cfg.get("aliases", set())}
                            aliases.add(key.lower())
                            if normalized and normalized in aliases:
                                requested_keys = [key]
                                context.input_data["workflow"] = key
                                workflow_name = key
                                break
                    else:
                        context.input_data["workflow"] = requested_keys[0]
                        workflow_name = requested_keys[0]

            use_dynamic_registry = len(requested_keys) > 1

            base_conditions = (
                context.input_data.get("conditions").copy()
                if isinstance(context.input_data.get("conditions"), dict)
                else {}
            )

            policy_runs: Dict[str, Dict[str, Any]] = {}
            aggregated_findings: List[Finding] = []

            for key in requested_keys:
                policy_cfg = policy_registry.get(key)
                if not policy_cfg:
                    continue
                policy_input = dict(context.input_data)
                policy_input["conditions"] = dict(base_conditions)
                parameters = policy_cfg.get("parameters")
                if isinstance(parameters, dict):
                    for param_key, param_value in parameters.items():
                        if param_key not in policy_input["conditions"]:
                            policy_input["conditions"][param_key] = param_value


                if not isinstance(context.input_data.get("conditions"), dict):
                    context.input_data["conditions"] = {}
                if isinstance(context.input_data.get("conditions"), dict):
                    context.input_data["conditions"].update(policy_input["conditions"])

                handler = policy_cfg.get("handler")
                if handler is None:
                    self._log_policy_event(
                        policy_cfg.get("policy_id", "unknown"),
                        None,
                        "skipped",
                        "Policy handler missing; skipping execution",
                        {},
                    )
                    continue

                findings = handler(tables, policy_input, notifications, policy_cfg)
                aggregated_findings.extend(findings)
                display_name = (
                    policy_cfg.get("policy_name")
                    or policy_cfg.get("detector")
                    or key
                )
                policy_runs[key] = {
                    "policy_cfg": policy_cfg,
                    "display": display_name,
                    "findings": findings,
                }

            if not policy_runs and aggregated_findings:
                display = workflow_name or "opportunity_policy"
                policy_runs[display] = {
                    "policy_cfg": {
                        "policy_id": "unknown",
                        "policy_name": display,
                        "detector": display,
                    },
                    "display": display,
                    "findings": aggregated_findings,
                }

            per_policy_retained: Dict[str, List[Finding]] = {}

            if use_dynamic_registry:
                for key in requested_keys:
                    payload = policy_runs.get(key)
                    if not payload:
                        continue
                    policy_cfg = payload.get("policy_cfg", {})
                    display = payload.get("display", key)
                    findings = payload.get("findings", [])
                    if not findings:
                        per_policy_retained[display] = []
                        continue
                    valid = [
                        f
                        for f in findings
                        if f.financial_impact_gbp >= min_impact_threshold
                    ]
                    if valid:
                        per_policy_retained[display] = valid
                    elif not explicit_min_override:
                        best = max(
                            findings,
                            key=lambda f: f.financial_impact_gbp,
                        )
                        per_policy_retained[display] = [best]
                        self._log_policy_event(
                            policy_cfg.get("policy_id", "unknown"),
                            best.supplier_id,
                            "threshold_relaxed",
                            "Retained top opportunity despite financial impact below configured minimum",
                            {
                                "min_financial_impact": min_impact_threshold,
                                "opportunity_id": best.opportunity_id,
                            },
                        )
                    else:
                        per_policy_retained[display] = []
            else:
                key = requested_keys[0]
                payload = policy_runs.get(key)
                policy_cfg = payload.get("policy_cfg") if payload else policy_registry.get(key)
                display = (
                    payload.get("display")
                    if payload
                    else (
                        policy_cfg.get("policy_name")
                        if isinstance(policy_cfg, dict)
                        else key
                    )
                )
                findings = payload.get("findings", []) if payload else aggregated_findings
                if not findings:
                    per_policy_retained[display] = []
                else:
                    valid = [
                        f
                        for f in findings
                        if f.financial_impact_gbp >= min_impact_threshold
                    ]
                    if valid:
                        per_policy_retained[display] = valid
                    elif not explicit_min_override:
                        best = max(
                            findings,
                            key=lambda f: f.financial_impact_gbp,
                        )
                        per_policy_retained[display] = [best]
                        if isinstance(policy_cfg, dict):
                            self._log_policy_event(
                                policy_cfg.get("policy_id", "unknown"),
                                best.supplier_id,
                                "threshold_relaxed",
                                "Retained top opportunity despite financial impact below configured minimum",
                                {
                                    "min_financial_impact": min_impact_threshold,
                                    "opportunity_id": best.opportunity_id,
                                },
                            )
                    else:
                        per_policy_retained[display] = []

            filtered, per_policy_categories = self._apply_policy_category_limits(
                per_policy_retained
            )

            filtered = self._enrich_findings(filtered, tables)
            filtered = self._map_item_descriptions(filtered, tables)
            self._load_supplier_risk_map()
            for f in filtered:
                candidate_item = f.item_reference or f.item_id
                if not candidate_item and isinstance(f.calculation_details, dict):
                    candidate_item = (
                        f.calculation_details.get("item_reference")
                        or f.calculation_details.get("item_id")
                    )
                f.candidate_suppliers = self._find_candidate_suppliers(
                    candidate_item, f.supplier_id, f.source_records
                )
            self._attach_vector_context(filtered)
            self._apply_feedback_annotations(filtered)

            policy_opportunities: Dict[str, List[Dict[str, Any]]] = {}
            policy_suppliers: Dict[str, List[str]] = {}
            policy_category_opportunities: Dict[
                str, Dict[str, List[Dict[str, Any]]]
            ] = {}
            for key in requested_keys:
                payload = policy_runs.get(key)
                if not payload:
                    continue
                display = payload.get("display", key)
                retained = per_policy_retained.get(display, [])
                policy_opportunities[display] = [f.as_dict() for f in retained]
                policy_suppliers[display] = sorted(
                    {
                        str(f.supplier_id).strip()
                        for f in retained
                        if f.supplier_id and str(f.supplier_id).strip()
                    }
                )
                categories = per_policy_categories.get(display, {})
                policy_category_opportunities[display] = {
                    category: [f.as_dict() for f in findings]
                    for category, findings in categories.items()
                }

            for display, retained in per_policy_retained.items():
                policy_opportunities.setdefault(
                    display, [f.as_dict() for f in retained]
                )
                policy_suppliers.setdefault(
                    display,
                    sorted(
                        {
                            str(f.supplier_id).strip()
                            for f in retained
                            if f.supplier_id and str(f.supplier_id).strip()
                        }
                    ),
                )
                categories = per_policy_categories.get(display, {})
                policy_category_opportunities.setdefault(
                    display,
                    {
                        category: [f.as_dict() for f in findings]
                        for category, findings in categories.items()
                    },
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
                "min_financial_impact": min_impact_threshold,
                "policy_events": self._event_log,
                "escalations": self._escalations,
                "policy_opportunities": policy_opportunities,
                "policy_category_opportunities": policy_category_opportunities,
                "policy_suppliers": policy_suppliers,
            }
            # pass candidate supplier IDs to downstream agents
            supplier_candidates = {
                str(s["supplier_id"]).strip()
                for f in filtered
                for s in f.candidate_suppliers
                if s.get("supplier_id") and str(s["supplier_id"]).strip()
            }
            for f in filtered:
                if f.supplier_id and str(f.supplier_id).strip():
                    supplier_candidates.add(str(f.supplier_id).strip())
            for suppliers in policy_suppliers.values():
                for supplier in suppliers:
                    if supplier:
                        supplier_candidates.add(supplier)
            data["supplier_candidates"] = list(supplier_candidates)

            directory_map: Dict[str, Dict[str, Any]] = {}

            def _register_supplier(supplier_id: Optional[str], supplier_name: Optional[str]) -> None:
                if not supplier_id:
                    return
                sid = str(supplier_id).strip()
                if not sid:
                    return
                entry = directory_map.setdefault(sid, {"supplier_id": sid})
                if supplier_name and not entry.get("supplier_name"):
                    entry["supplier_name"] = str(supplier_name).strip()

            for f in filtered:
                _register_supplier(f.supplier_id, f.supplier_name)
                for candidate in f.candidate_suppliers:
                    _register_supplier(
                        candidate.get("supplier_id"), candidate.get("supplier_name")
                    )

            lookup = getattr(self, "_supplier_lookup", {})
            if lookup:
                for sid, entry in directory_map.items():
                    if entry.get("supplier_name"):
                        continue
                    supplier_name = lookup.get(sid)
                    if supplier_name:
                        entry["supplier_name"] = supplier_name

            if directory_map:
                data["supplier_directory"] = sorted(
                    directory_map.values(), key=lambda entry: entry["supplier_id"]
                )
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
        "product_mapping": "proc.cat_product_mapping",
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
        for table, sql_name in self.TABLE_MAP.items():
            try:
                dfs[table] = self._read_sql(f"SELECT * FROM {sql_name}")
            except Exception:
                logger.exception("Failed to ingest table %s (%s)", table, sql_name)
                dfs[table] = pd.DataFrame()
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
            rate_series = pd.to_numeric(rate_col, errors="coerce").fillna(1.0)
            for col in cols:
                if col in df.columns:
                    numeric_col = pd.to_numeric(df[col], errors="coerce")
                    df[f"{col}_gbp"] = numeric_col.fillna(0.0) * rate_series
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

        def _resolve_from(container: Any, field: str) -> Any:
            if not isinstance(container, dict):
                return None
            value = container.get(field)
            if value is None:
                return None
            if isinstance(value, str) and not value.strip():
                return None
            return value
        missing = []
        for field in required_fields:
            value = conditions.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                fallback = None
                for source in (
                    input_data,
                    input_data.get("parameters"),
                    input_data.get("defaults"),
                ):
                    fallback = _resolve_from(source, field)
                    if fallback is not None:
                        conditions[field] = fallback
                        break

                if fallback is not None:
                    continue

                default_map = defaults or {}
                if field in default_map and default_map[field] is not None:
                    conditions[field] = default_map[field]
                else:
                    missing.append(field)
        return missing

    @staticmethod
    def _normalise_policy_slug(name: Optional[Any]) -> Optional[str]:
        if not name:
            return None
        text = str(name).strip()
        if not text:
            return None
        text = re.sub(r"(?<!^)(?=[A-Z])", "_", text)
        text = re.sub(r"[^A-Za-z0-9]+", "_", text)
        slug = text.strip("_").lower()
        return slug or None

    def _coerce_policy_rules(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        details: Dict[str, Any] = {}

        def _merge_details(target: Dict[str, Any], candidate: Any) -> Dict[str, Any]:
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except Exception:  # pragma: no cover - defensive parsing
                    return target
            if isinstance(candidate, dict):
                if not target:
                    return dict(candidate)
                merged = dict(target)
                merged.update(candidate)
                return merged
            return target

        for field in ("details", "policy_details", "policy_desc", "description"):
            details = _merge_details(details, policy.get(field))

        rules = details.get("rules") if isinstance(details, dict) else {}
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:  # pragma: no cover - defensive
                rules = {}
        elif isinstance(rules, dict):
            rules = dict(rules)
        else:
            rules = {}

        if not rules and isinstance(details, dict):
            rules = dict(details)

        return rules

    def _resolve_dynamic_policy_handler(self, slug: str):
        mapping = {
            "volume_discount_opportunity": self._policy_volume_discount,
            "supplier_consolidation_opportunity": self._policy_supplier_consolidation,
        }
        return mapping.get(slug)

    def _decorate_policy_entry(
        self,
        key: str,
        config: Dict[str, Any],
        provided_policy: Optional[Dict[str, Any]] = None,
        slug_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = dict(config)
        policy_name = entry.get("policy_name") or entry.get("detector") or key
        entry["policy_name"] = policy_name
        policy_id = entry.get("policy_id") or entry.get("policyId") or key
        entry["policy_id"] = str(policy_id)
        key_slug = self._normalise_policy_slug(key)
        slug = key_slug or slug_hint or self._normalise_policy_slug(policy_name)
        if not slug:
            slug = str(key).strip().lower() or str(entry["policy_id"]).strip().lower()
        entry["policy_slug"] = slug

        alias_set: set[str] = set()

        def _add_alias(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (dict, list, tuple, set)):
                return
            text = str(value).strip()
            if not text:
                return
            alias_set.add(text.lower())
            slugged = self._normalise_policy_slug(text)
            if slugged:
                alias_set.add(slugged)

        _add_alias(slug)
        _add_alias(key)
        _add_alias(policy_name)
        _add_alias(entry["policy_id"])
        _add_alias(entry.get("detector"))

        existing_aliases = config.get("aliases")
        if isinstance(existing_aliases, (set, list, tuple)):
            for alias in existing_aliases:
                _add_alias(alias)

        if provided_policy:
            entry["source_policy"] = provided_policy
            for field in (
                "policyName",
                "policy_name",
                "policy_desc",
                "description",
                "policyId",
                "policy_id",
            ):
                _add_alias(provided_policy.get(field))

        entry["aliases"] = alias_set
        return entry

    def _collect_policy_tokens(self, policy: Dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for field in (
            "policyId",
            "policy_id",
            "id",
            "policyName",
            "policy_name",
            "policy_desc",
            "description",
            "detector",
        ):
            value = policy.get(field)
            if value is None or isinstance(value, (dict, list, tuple, set)):
                continue
            text = str(value).strip()
            if not text:
                continue
            tokens.add(text.lower())
            slug = self._normalise_policy_slug(text)
            if slug:
                tokens.add(slug)
        return tokens

    def _filter_registry_by_policies(
        self,
        registry: Dict[str, Dict[str, Any]],
        policies: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(policies, list) or not policies:
            return {
                entry["policy_slug"]: entry
                for key, cfg in registry.items()
                for entry in [self._decorate_policy_entry(key, cfg)]
            }

        filtered: Dict[str, Dict[str, Any]] = {}
        for policy in policies:
            if not isinstance(policy, dict):
                continue
            tokens = self._collect_policy_tokens(policy)
            slug_hint = self._normalise_policy_slug(
                policy.get("policyName")
                or policy.get("policy_desc")
                or policy.get("description")
            )
            matched_entry: Optional[Dict[str, Any]] = None
            for key, cfg in registry.items():
                probe = self._decorate_policy_entry(key, cfg, slug_hint=slug_hint)
                aliases = probe.get("aliases", set())
                if tokens & aliases:
                    matched_entry = self._decorate_policy_entry(
                        key, cfg, provided_policy=policy, slug_hint=slug_hint
                    )
                    break
            if not matched_entry:
                continue
            rules = self._coerce_policy_rules(policy)
            if rules:
                matched_entry["rules"] = rules
                parameters = rules.get("parameters")
                if isinstance(parameters, dict) and parameters:
                    existing_params = (
                        matched_entry.get("parameters")
                        if isinstance(matched_entry.get("parameters"), dict)
                        else {}
                    )
                    merged_params = {**existing_params, **parameters}
                    matched_entry["parameters"] = merged_params
                    if not matched_entry.get("required_fields"):
                        matched_entry["required_fields"] = list(merged_params.keys())
                default_conditions = rules.get("default_conditions")
                if isinstance(default_conditions, dict) and default_conditions:
                    existing_defaults = (
                        matched_entry.get("default_conditions")
                        if isinstance(matched_entry.get("default_conditions"), dict)
                        else {}
                    )
                    matched_entry["default_conditions"] = {
                        **existing_defaults,
                        **default_conditions,
                    }
            filtered[matched_entry["policy_slug"]] = matched_entry
        return filtered

    def _assemble_policy_registry(
        self, input_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        provided_policies = [
            policy
            for policy in input_data.get("policies") or []
            if isinstance(policy, dict)
        ]
        base_registry = self._filter_registry_by_policies(
            self._get_policy_registry(), provided_policies
        )
        dynamic_registry = self._build_dynamic_policy_registry(input_data)
        registry: Dict[str, Dict[str, Any]] = {}
        registry.update(base_registry)
        registry.update(dynamic_registry)
        if not provided_policies and not registry:
            registry = self._filter_registry_by_policies(
                self._get_policy_registry(), []
            )
        return registry, provided_policies

    def _build_dynamic_policy_registry(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        policies = input_data.get("policies")
        if not isinstance(policies, list):
            return {}
        registry: Dict[str, Dict[str, Any]] = {}
        for policy in policies:
            if not isinstance(policy, dict):
                continue
            name = (
                policy.get("policyName")
                or policy.get("policy_name")
                or policy.get("policy_desc")
                or policy.get("description")
            )
            slug = self._normalise_policy_slug(name)
            if not slug:
                continue
            handler = self._resolve_dynamic_policy_handler(slug)
            if handler is None:
                logger.debug("No dynamic handler registered for policy '%s'", name)
                continue
            rules = self._coerce_policy_rules(policy)
            parameters = rules.get("parameters") if isinstance(rules, dict) else {}
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except Exception:  # pragma: no cover - defensive
                    parameters = {}
            if not isinstance(parameters, dict):
                parameters = {}
            entry = {

                "policy_id": str(
                    policy.get("policyId") or policy.get("policy_id") or slug
                ),
                "detector": str(
                    policy.get("policyName")
                    or policy.get("description")
                    or name
                    or slug
                ),
                "policy_name": str(name) if name else slug,
                "handler": handler,
                "parameters": dict(parameters),
                "required_fields": list(parameters.keys()),
                "default_conditions": {},
                "source_policy": policy,
                "rules": rules,
            }
            decorated = self._decorate_policy_entry(
                slug, entry, provided_policy=policy, slug_hint=slug
            )
            registry[decorated["policy_slug"]] = decorated

        return registry

    def _get_condition(self, input_data: Dict[str, Any], key: str, default: Any = None) -> Any:
        def _resolve_from(container: Any) -> Any:
            if not isinstance(container, dict):
                return None
            value = container.get(key)
            if value is None:
                return None
            if isinstance(value, str) and not value.strip():
                return None
            return value

        conditions = input_data.get("conditions")
        if isinstance(conditions, dict):
            value = _resolve_from(conditions)
            if value is not None:
                return value

        for source in (
            input_data,
            input_data.get("parameters"),
            input_data.get("defaults"),
        ):
            value = _resolve_from(source)
            if value is not None:
                if isinstance(conditions, dict):
                    conditions.setdefault(key, value)
                return value

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
        """Return policy metadata keyed by canonical workflow slug."""

        def entry(
            slug: str,
            detector: str,
            handler,
            required: Iterable[str],
            default_conditions: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            aliases = {
                self._normalise_policy_slug(slug),
                self._normalise_policy_slug(detector),
            }
            return {
                "policy_slug": slug,
                "policy_id": slug,
                "detector": detector,
                "policy_name": detector,
                "required_fields": list(required),
                "handler": handler,
                "default_conditions": dict(default_conditions or {}),
                "aliases": {alias for alias in aliases if alias},
            }

        registry: Dict[str, Dict[str, Any]] = {
            "price_variance_check": entry(
                "price_variance_check",
                "Price Benchmark Variance",
                self._policy_price_benchmark_variance,
                ["supplier_id", "item_id", "actual_price", "benchmark_price"],
            ),
            "volume_consolidation_check": entry(
                "volume_consolidation_check",
                "Volume Consolidation",
                self._policy_volume_consolidation,
                ["minimum_volume_gbp"],
            ),
            "contract_expiry_check": entry(
                "contract_expiry_check",
                "Contract Expiry Opportunity",
                self._policy_contract_expiry,
                ["negotiation_window_days"],
                {"negotiation_window_days": 90},
            ),
            "supplier_risk_check": entry(
                "supplier_risk_check",
                "Supplier Risk Alert",
                self._policy_supplier_risk,
                ["risk_threshold"],
            ),
            "maverick_spend_check": entry(
                "maverick_spend_check",
                "Maverick Spend Detection",
                self._policy_maverick_spend,
                ["minimum_value_gbp"],
            ),
            "duplicate_supplier_check": entry(
                "duplicate_supplier_check",
                "Duplicate Supplier",
                self._policy_duplicate_supplier,
                ["minimum_overlap_gbp"],
            ),
            "category_overspend_check": entry(
                "category_overspend_check",
                "Category Overspend",
                self._policy_category_overspend,
                ["category_budgets"],
            ),
            "inflation_passthrough_check": entry(
                "inflation_passthrough_check",
                "Inflation Pass-Through",
                self._policy_inflation_passthrough,
                ["market_inflation_pct"],
            ),
            "unused_contract_value_check": entry(
                "unused_contract_value_check",
                "Unused Contract Value",
                self._policy_unused_contract_value,
                ["minimum_unused_value_gbp"],
            ),
            "supplier_performance_check": entry(
                "supplier_performance_check",
                "Supplier Performance Deviation",
                self._policy_supplier_performance,
                ["performance_records"],
            ),
            "esg_opportunity_check": entry(
                "esg_opportunity_check",
                "ESG Opportunity",
                self._policy_esg_opportunity,
                ["esg_scores"],
            ),
        }

        engine = getattr(self.agent_nick, "policy_engine", None)
        policies = []
        if engine is not None:
            if hasattr(engine, "iter_policies"):
                policies = list(engine.iter_policies())
            elif hasattr(engine, "list_policies"):
                policies = list(engine.list_policies())  # pragma: no cover - legacy

        def aliases_for(policy: Dict[str, Any]) -> set[str]:
            aliases: set[str] = set()
            for key in ("policyName", "policy_desc", "policyId", "slug"):
                alias = self._normalise_policy_slug(policy.get(key))
                if alias:
                    aliases.add(alias)
            for alias in policy.get("aliases", []):
                norm = self._normalise_policy_slug(alias)
                if norm:
                    aliases.add(norm)
            details = policy.get("details", {})
            if isinstance(details, dict):
                for key in (
                    "policy_name",
                    "identifier",
                    "policy_identifier",
                    "name",
                ):
                    alias = self._normalise_policy_slug(details.get(key))
                    if alias:
                        aliases.add(alias)
                rules = details.get("rules")
                if isinstance(rules, dict):
                    for key in ("policy_name", "name"):
                        alias = self._normalise_policy_slug(rules.get(key))
                        if alias:
                            aliases.add(alias)
            return aliases

        for policy in policies:
            policy_aliases = aliases_for(policy)
            if not policy_aliases:
                continue
            for slug, entry_data in registry.items():
                entry_aliases = entry_data.setdefault("aliases", set())
                base_aliases = {
                    self._normalise_policy_slug(slug),
                    self._normalise_policy_slug(entry_data.get("policy_id")),
                    self._normalise_policy_slug(entry_data.get("policy_name")),
                    self._normalise_policy_slug(entry_data.get("detector")),
                }
                entry_aliases.update(alias for alias in base_aliases if alias)
                if entry_aliases & policy_aliases:
                    entry_aliases.update(policy_aliases)
                    policy_id = policy.get("policyId")
                    if policy_id:
                        entry_data["policy_id"] = str(policy_id)
                    name = policy.get("policyName")
                    if name:
                        entry_data["policy_name"] = str(name)
                    desc = policy.get("policy_desc")
                    if desc:
                        entry_data["policy_desc"] = desc
                    entry_data["source_policy"] = policy
                    details = policy.get("details", {})
                    if isinstance(details, dict):
                        rules = details.get("rules")
                        if isinstance(rules, dict):
                            params = rules.get("parameters")
                            if isinstance(params, dict):
                                existing = (
                                    entry_data.get("parameters")
                                    if isinstance(entry_data.get("parameters"), dict)
                                    else {}
                                )
                                merged_params = {**existing, **params}
                                entry_data["parameters"] = merged_params
                                if not entry_data.get("required_fields"):
                                    entry_data["required_fields"] = list(merged_params.keys())
                            defaults = rules.get("default_conditions")
                            if isinstance(defaults, dict):
                                existing_defaults = (
                                    entry_data.get("default_conditions")
                                    if isinstance(entry_data.get("default_conditions"), dict)
                                    else {}
                                )
                                entry_data["default_conditions"] = {
                                    **existing_defaults,
                                    **defaults,
                                }
                    break

        return registry

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
        *,
        policy_id: Optional[str] = None,
        policy_name: Optional[str] = None,
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

        detector_slug = self._normalise_policy_slug(detector) or "detector"
        policy_slug = (
            self._normalise_policy_slug(policy_id)
            or self._normalise_policy_slug(policy_name)
            or detector_slug
        )

        def _slug_or_default(value: Optional[str], default: str) -> str:
            slug = self._normalise_policy_slug(value)
            return slug or default

        supplier_slug = _slug_or_default(supplier_id, "na")
        item_slug = _slug_or_default(item_id, "na")

        source_token = "none"
        if sources:
            normalised_sources = [
                str(src).strip()
                for src in sources
                if src is not None and str(src).strip()
            ]
            if normalised_sources:
                joined = "|".join(sorted(normalised_sources))
                source_token = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]

        opportunity_id = "_".join(
            token
            for token in [policy_slug, detector_slug, source_token, supplier_slug, item_slug]
            if token
        )

        policy_identifier = None
        if policy_id is not None:
            policy_identifier = str(policy_id).strip() or None

        finding = Finding(
            opportunity_id=opportunity_id,
            detector_type=detector,
            supplier_id=supplier_id,
            category_id=category_id,
            item_id=item_id,
            financial_impact_gbp=float(impact),
            calculation_details=details,
            source_records=sources,
            detected_on=datetime.utcnow(),
            item_reference=item_id,
            policy_id=policy_identifier,
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

    def _policy_volume_discount(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = str(policy_cfg.get("policy_id", "volume_discount_opportunity"))
        detector = (
            policy_cfg.get("policy_name")
            or policy_cfg.get("detector")
            or "VolumeDiscountOpportunity"
        )
        parameters = policy_cfg.get("parameters", {})
        min_qty = self._to_float(parameters.get("minimum_quantity_threshold"), 0.0)
        min_spend = self._to_float(parameters.get("minimum_spend_threshold"), 0.0)
        discount_rate = self._to_float(parameters.get("discount_rate"), 0.05)
        target_pct = parameters.get("target_discount_pct")
        if target_pct is not None:
            discount_rate = self._to_float(target_pct, 5.0) / 100.0
        if discount_rate > 1:
            discount_rate = discount_rate / 100.0
        if discount_rate <= 0:
            discount_rate = 0.05
        lookback_days = self._to_int(parameters.get("lookback_period_days"), 0)

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if po_lines.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No purchase order line data available for volume discount analysis",
                {},
            )
            return findings

        base_df = po_lines.dropna(subset=["po_id", "item_id"]).copy()
        if base_df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Purchase order lines missing item identifiers",
                {},
            )
            return findings

        base_df["po_id"] = base_df["po_id"].astype(str)
        base_df["item_id"] = base_df["item_id"].astype(str)

        value_col = self._choose_first_column(
            base_df,
            [
                "line_amount_gbp",
                "total_amount_incl_tax_gbp",
                "line_amount",
                "total_amount_incl_tax",
            ],
        )
        if value_col:
            base_df[value_col] = pd.to_numeric(base_df[value_col], errors="coerce").fillna(0.0)
        else:
            base_df["__line_value"] = (
                pd.to_numeric(base_df.get("unit_price"), errors="coerce").fillna(0.0)
                * pd.to_numeric(base_df.get("quantity"), errors="coerce").fillna(0.0)
            )
            value_col = "__line_value"

        qty_col = "quantity" if "quantity" in base_df.columns else None
        if qty_col:
            base_df[qty_col] = pd.to_numeric(base_df[qty_col], errors="coerce").fillna(0.0)
        else:
            base_df["__quantity"] = pd.to_numeric(
                base_df.get("line_quantity"), errors="coerce"
            ).fillna(0.0)
            if base_df["__quantity"].abs().sum() == 0:
                base_df["__quantity"] = base_df[value_col]
            qty_col = "__quantity"

        price_col = self._choose_first_column(
            base_df, ["unit_price_gbp", "unit_price", "price"]
        )
        if price_col:
            base_df[price_col] = pd.to_numeric(base_df[price_col], errors="coerce").fillna(0.0)

        filtered_df = base_df.copy()
        if lookback_days > 0:
            po_df = tables.get("purchase_orders", pd.DataFrame())
            if not po_df.empty and "po_id" in po_df.columns:
                date_col = next(
                    (col for col in ["order_date", "requested_date", "created_date"] if col in po_df.columns),
                    None,
                )
                if date_col:
                    po_subset = po_df.dropna(subset=["po_id"]).copy()
                    po_subset["po_id"] = po_subset["po_id"].astype(str)
                    po_subset[date_col] = pd.to_datetime(
                        po_subset[date_col], errors="coerce"
                    )
                    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
                    recent_ids = (
                        po_subset.loc[po_subset[date_col] >= cutoff, "po_id"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    if len(recent_ids) > 0:
                        filtered_df = filtered_df[
                            filtered_df["po_id"].isin(set(recent_ids))
                        ]
                    if filtered_df.empty:
                        filtered_df = base_df
                        self._log_policy_event(
                            policy_id,
                            None,
                            "fallback",
                            "No orders matched lookback window; using full dataset",
                            {"lookback_days": lookback_days},
                        )

        filtered_df["supplier_id"] = filtered_df["po_id"].map(self._po_supplier_map)
        filtered_df = filtered_df.dropna(subset=["supplier_id"])
        filtered_df["supplier_id"] = filtered_df["supplier_id"].astype(str)
        if filtered_df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier-linked purchase order lines for volume analysis",
                {},
            )
            return findings

        agg_map: Dict[str, str] = {value_col: "sum", qty_col: "sum"}
        if price_col:
            agg_map[price_col] = "mean"
        summary = (
            filtered_df.groupby(["supplier_id", "item_id"], as_index=False)
            .agg(agg_map)
            .rename(columns={value_col: "total_spend", qty_col: "total_qty"})
        )
        if price_col:
            summary = summary.rename(columns={price_col: "avg_price"})
        else:
            summary["avg_price"] = summary.apply(
                lambda row: row["total_spend"] / row["total_qty"]
                if row.get("total_qty", 0.0)
                else row["total_spend"],
                axis=1,
            )

        if summary.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier/item combinations available for analysis",
                {},
            )
            return findings

        matches = summary[
            (summary["total_qty"] >= min_qty)
            | (summary["total_spend"] >= min_spend)
        ]
        fallback_used = False
        if matches.empty:
            matches = summary.nlargest(1, "total_spend")
            fallback_used = not matches.empty

        if matches.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No opportunities matched volume discount thresholds",
                {
                    "minimum_quantity_threshold": min_qty,
                    "minimum_spend_threshold": min_spend,
                },
            )
            return findings

        for _, row in matches.iterrows():
            supplier_id = row["supplier_id"]
            item_id = row["item_id"]
            po_ids = self._po_ids_for_supplier_item(tables, supplier_id, item_id)
            category_id = self._category_for_po(po_ids[0]) if po_ids else None
            estimated_savings = max(0.0, float(row["total_spend"]) * float(discount_rate))
            if fallback_used and estimated_savings <= 0:
                estimated_savings = max(0.0, float(row["total_spend"]) * 0.05)
            details = {
                "total_quantity": float(row.get("total_qty", 0.0)),
                "total_spend": float(row.get("total_spend", 0.0)),
                "average_price": float(row.get("avg_price", 0.0)),
                "discount_rate": float(discount_rate),
                "estimated_savings": float(estimated_savings),
                "thresholds": {
                    "minimum_quantity_threshold": min_qty,
                    "minimum_spend_threshold": min_spend,
                },
            }
            if lookback_days > 0:
                details["lookback_days"] = lookback_days
            if fallback_used:
                details["threshold_relaxed"] = True
            sources = po_ids or [f"item:{item_id}"]
            finding = self._build_finding(
                detector,
                supplier_id,
                category_id,
                item_id,
                estimated_savings,
                details,
                sources,
                policy_id=policy_id,
                policy_name=policy_cfg.get("policy_name"),
            )
            findings.append(finding)
            self._log_policy_event(
                policy_id,
                supplier_id,
                "opportunity_identified",
                f"Volume discount opportunity identified for item {item_id}",
                {**details, "source_po_ids": po_ids},
            )

        if findings:
            self._default_notifications(notifications)
        return findings

    def _policy_supplier_consolidation(
        self,
        tables: Dict[str, pd.DataFrame],
        input_data: Dict[str, Any],
        notifications: set[str],
        policy_cfg: Dict[str, Any],
    ) -> List[Finding]:
        findings: List[Finding] = []
        policy_id = str(policy_cfg.get("policy_id", "supplier_consolidation_opportunity"))
        detector = (
            policy_cfg.get("policy_name")
            or policy_cfg.get("detector")
            or "SupplierConsolidationOpportunity"
        )
        parameters = policy_cfg.get("parameters", {})
        min_suppliers = max(2, self._to_int(parameters.get("minimum_supplier_count"), 2))
        savings_pct = self._to_float(parameters.get("consolidation_savings_pct"), 5.0)
        if savings_pct <= 0:
            savings_pct = 5.0
        savings_rate = savings_pct / 100.0
        min_spend = self._to_float(parameters.get("minimum_spend_threshold"), 0.0)

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if po_lines.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No purchase order line data available for consolidation analysis",
                {},
            )
            return findings

        df = po_lines.dropna(subset=["po_id"]).copy()
        if df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Purchase order lines missing identifiers",
                {},
            )
            return findings

        df["po_id"] = df["po_id"].astype(str)
        df["supplier_id"] = df["po_id"].map(self._po_supplier_map)
        df = df.dropna(subset=["supplier_id"])
        df["supplier_id"] = df["supplier_id"].astype(str)
        if df.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No supplier-linked purchase orders for consolidation",
                {},
            )
            return findings

        df["category"] = df["po_id"].map(self._category_for_po)
        if df.get("category") is None or df["category"].isna().all():
            if "item_description" in df.columns:
                df["category"] = df["item_description"].astype(str)
            else:
                df["category"] = "General"
        df["category"] = df["category"].fillna(
            df.get("item_description")
        ).fillna("General")
        df["category"] = df["category"].astype(str)

        value_col = self._choose_first_column(
            df,
            [
                "line_amount_gbp",
                "total_amount_incl_tax_gbp",
                "line_amount",
                "total_amount_incl_tax",
            ],
        )
        if value_col:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        else:
            df["__line_value"] = (
                pd.to_numeric(df.get("unit_price"), errors="coerce").fillna(0.0)
                * pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0.0)
            )
            value_col = "__line_value"

        summary = (
            df.groupby(["category", "supplier_id"], as_index=False)
            .agg({value_col: "sum"})
            .rename(columns={value_col: "total_spend"})
        )
        if summary.empty:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "Unable to aggregate supplier spend by category",
                {},
            )
            return findings

        category_groups = summary.groupby("category")
        for category, cat_df in category_groups:
            supplier_count = len(cat_df)
            if supplier_count < max(2, min_suppliers):
                continue
            total_spend = float(cat_df["total_spend"].sum())
            if total_spend < min_spend:
                continue
            ordered = cat_df.sort_values("total_spend", ascending=False)
            leader = ordered.iloc[0]
            remainder = ordered.iloc[1:]
            if remainder.empty:
                continue
            category_lines = df[df["category"] == category]
            for _, row in remainder.iterrows():
                supplier_id = row["supplier_id"]
                supplier_spend = float(row["total_spend"])
                savings = max(0.0, supplier_spend * savings_rate)
                po_ids = sorted(
                    {
                        str(po_id)
                        for po_id in category_lines.loc[
                            category_lines["supplier_id"] == supplier_id,
                            "po_id",
                        ]
                        .dropna()
                        .unique()
                    }
                )
                details = {
                    "category": category,
                    "current_supplier": supplier_id,
                    "preferred_supplier": leader["supplier_id"],
                    "category_spend": total_spend,
                    "reallocated_spend": supplier_spend,
                    "savings_rate": savings_rate,
                    "estimated_savings": savings,
                    "supplier_count": supplier_count,
                }
                finding = self._build_finding(
                    detector,
                    supplier_id,
                    category,
                    None,
                    savings,
                    details,
                    po_ids or [category],
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
                )
                findings.append(finding)
                self._log_policy_event(
                    policy_id,
                    supplier_id,
                    "opportunity_identified",
                    f"Supplier consolidation opportunity identified for category {category}",
                    {**details, "source_po_ids": po_ids},
                )

        if not findings:
            candidate_stats = (
                summary.groupby("category")
                .agg(supplier_count=("supplier_id", "nunique"), category_spend=("total_spend", "sum"))
                .sort_values(["supplier_count", "category_spend"], ascending=[False, False])
            )
            if not candidate_stats.empty and candidate_stats.iloc[0]["supplier_count"] >= 2:
                fallback_category = candidate_stats.index[0]
                cat_df = summary[summary["category"] == fallback_category].sort_values(
                    "total_spend", ascending=False
                )
                if len(cat_df) > 1:
                    leader = cat_df.iloc[0]
                    row = cat_df.iloc[1]
                    supplier_id = row["supplier_id"]
                    supplier_spend = float(row["total_spend"])
                    savings = max(0.0, supplier_spend * savings_rate)
                    category_lines = df[df["category"] == fallback_category]
                    po_ids = sorted(
                        {
                            str(po_id)
                            for po_id in category_lines.loc[
                                category_lines["supplier_id"] == supplier_id,
                                "po_id",
                            ]
                            .dropna()
                            .unique()
                        }
                    )
                    details = {
                        "category": fallback_category,
                        "current_supplier": supplier_id,
                        "preferred_supplier": leader["supplier_id"],
                        "category_spend": float(cat_df["total_spend"].sum()),
                        "reallocated_spend": supplier_spend,
                        "savings_rate": savings_rate,
                        "estimated_savings": savings,
                        "supplier_count": int(candidate_stats.iloc[0]["supplier_count"]),
                        "threshold_relaxed": True,
                    }
                    finding = self._build_finding(
                        detector,
                        supplier_id,
                        fallback_category,
                        None,
                        savings,
                        details,
                        po_ids or [fallback_category],
                        policy_id=policy_id,
                        policy_name=policy_cfg.get("policy_name"),
                    )
                    findings.append(finding)
                    self._log_policy_event(
                        policy_id,
                        supplier_id,
                        "threshold_relaxed",
                        "Surfaced consolidation opportunity after relaxing supplier count threshold",
                        {**details, "source_po_ids": po_ids},
                    )

        if findings:
            self._default_notifications(notifications)
        else:
            self._log_policy_event(
                policy_id,
                None,
                "no_action",
                "No consolidation opportunities identified",
                {"minimum_supplier_count": min_suppliers, "minimum_spend_threshold": min_spend},
            )
        return findings

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

    def _map_item_descriptions(
        self, findings: List[Finding], tables: Dict[str, pd.DataFrame]
    ) -> List[Finding]:
        """Replace ``item_id`` with the supplier specific item description.

        The user requirement mandates that the ``item_id`` exposed to
        downstream agents represents the supplier's product name as captured in
        ``proc.po_line_items_agent``.  Findings frequently surface purchase
        order identifiers or internal item codes which, while useful for audit
        trails, are not human friendly.  This helper cross references purchase
        order line items (and their parent purchase orders) to surface the
        description most representative of the supplier's product catalogue.
        The original identifier is retained in ``calculation_details`` under
        ``item_reference`` so that analysts can still trace the source record.
        """

        if not findings:
            return findings

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        purchase_orders = tables.get("purchase_orders", pd.DataFrame())
        if (
            po_lines.empty
            or "po_id" not in po_lines.columns
            or "item_description" not in po_lines.columns
        ):
            return findings
        if purchase_orders.empty or "po_id" not in purchase_orders.columns:
            return findings

        try:
            merged = po_lines.merge(
                purchase_orders[["po_id", "supplier_id"]],
                on="po_id",
                how="left",
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to join purchase order lines for description mapping")
            return findings

        if merged.empty:
            return findings

        def _clean(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            text = str(value).strip()
            return text or None

        merged["supplier_id"] = merged["supplier_id"].map(_clean)
        merged["po_id"] = merged["po_id"].map(_clean)
        if "item_id" in merged.columns:
            merged["item_id"] = merged["item_id"].map(_clean)
        merged["item_description"] = merged["item_description"].map(_clean)
        merged = merged.dropna(subset=["supplier_id", "item_description"])
        if merged.empty:
            return findings

        supplier_item_desc: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        supplier_po_desc: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        supplier_desc: dict[str, Counter[str]] = defaultdict(Counter)

        for row in merged.itertuples(index=False):
            supplier_id = getattr(row, "supplier_id", None)
            description = getattr(row, "item_description", None)
            if not supplier_id or not description:
                continue
            supplier_desc[supplier_id][description] += 1
            item_identifier = getattr(row, "item_id", None)
            if item_identifier:
                supplier_item_desc[(supplier_id, item_identifier)][description] += 1
            po_identifier = getattr(row, "po_id", None)
            if po_identifier:
                supplier_po_desc[(supplier_id, po_identifier)][description] += 1

        def _resolve(counter: Optional[Counter[str]]) -> Optional[str]:
            if not counter:
                return None
            most_common = counter.most_common(1)
            return most_common[0][0] if most_common else None

        description_matches: Dict[str, Dict[str, Any]] = {}
        catalog = tables.get("product_mapping", pd.DataFrame())
        if isinstance(catalog, pd.DataFrame) and not catalog.empty:
            if "product" in catalog.columns:
                catalog_df = catalog.dropna(subset=["product"]).copy()
                if not catalog_df.empty:
                    catalog_df["product"] = catalog_df["product"].astype(str).str.strip()
                    catalog_df = catalog_df[catalog_df["product"] != ""]
                    catalog_rows = []
                    for row in catalog_df.itertuples(index=False):
                        product_name = getattr(row, "product", None)
                        if not product_name:
                            continue
                        entry: Dict[str, Any] = {"product": str(product_name).strip()}
                        for level in range(1, 6):
                            field = f"category_level_{level}"
                            if hasattr(row, field):
                                entry[field] = getattr(row, field)
                        catalog_rows.append(entry)
                    if catalog_rows:
                        catalog_index = [
                            (entry["product"].lower(), entry)
                            for entry in catalog_rows
                        ]
                        unique_descriptions = {desc for counter in supplier_desc.values() for desc in counter}
                        for description in unique_descriptions:
                            key = description.lower()
                            best_score = 0.0
                            best_entry: Optional[Dict[str, Any]] = None
                            for product_key, entry in catalog_index:
                                score = SequenceMatcher(None, key, product_key).ratio()
                                if score > best_score:
                                    best_score = score
                                    best_entry = entry
                            if best_entry and best_score >= _CATALOG_MATCH_THRESHOLD:
                                description_matches[description] = {
                                    **best_entry,
                                    "match_score": best_score,
                                }

        for finding in findings:
            supplier_id = finding.supplier_id
            if not supplier_id:
                continue

            original_item = finding.item_id
            if original_item and not finding.item_reference:
                finding.item_reference = str(original_item)
            description = None
            if original_item:
                description = _resolve(
                    supplier_item_desc.get((supplier_id, str(original_item).strip()))
                )
            if not description:
                for src in finding.source_records:
                    src_key = _clean(src)
                    if not src_key:
                        continue
                    description = _resolve(
                        supplier_po_desc.get((supplier_id, src_key))
                    )
                    if description:
                        break
            if not description:
                description = _resolve(supplier_desc.get(supplier_id))

            if not description:
                continue

            if (
                original_item
                and original_item != description
                and isinstance(finding.calculation_details, dict)
            ):
                finding.calculation_details.setdefault("item_reference", original_item)
            if isinstance(finding.calculation_details, dict):
                finding.calculation_details.setdefault("item_description", description)
            catalog_entry = description_matches.get(description)
            if catalog_entry:
                product_name = str(catalog_entry.get("product", "")).strip()
                if product_name:
                    finding.item_id = product_name
                    if isinstance(finding.calculation_details, dict):
                        finding.calculation_details.setdefault("catalog_product", product_name)
                        category_details = {}
                        for level in range(1, 6):
                            field = f"category_level_{level}"
                            value = catalog_entry.get(field)
                            if value:
                                category_details[field] = str(value).strip()
                                if not finding.category_id:
                                    finding.category_id = str(value)
                        if category_details:
                            finding.calculation_details.setdefault(
                                "catalog_categories", category_details
                            )
                else:
                    finding.item_id = description
            else:
                finding.item_id = description

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
                policy_id=policy_id,
                policy_name=policy_cfg.get("policy_name"),
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
                        policy_id=policy_id,
                        policy_name=policy_cfg.get("policy_name"),
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
        policy_id: Optional[str] = None,
        policy_name: Optional[str] = None,
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
                policy_id=policy_id,
                policy_name=policy_name,
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
            policy_id=policy_id,
            policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                    policy_id=policy_id,
                    policy_name=policy_cfg.get("policy_name"),
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
                if "risk_score" in df.columns:
                    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
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

    def _apply_feedback_annotations(self, findings: List[Finding]) -> None:
        """Attach persisted feedback to each finding when available."""

        if not findings:
            return

        feedback_map = load_opportunity_feedback(self.agent_nick)
        if not feedback_map:
            return

        for finding in findings:
            info = feedback_map.get(finding.opportunity_id)
            if not info:
                continue

            finding.feedback_status = info.get("status")
            finding.feedback_reason = info.get("reason")
            finding.feedback_user = info.get("user_id")
            finding.feedback_metadata = info.get("metadata")
            updated = info.get("updated_on")
            if isinstance(updated, datetime):
                finding.feedback_updated_at = updated
            elif updated:
                try:
                    finding.feedback_updated_at = datetime.fromisoformat(str(updated))
                except Exception:  # pragma: no cover - defensive parsing
                    logger.debug(
                        "Unable to parse feedback timestamp for %s", finding.opportunity_id
                    )
            finding.is_rejected = (
                str(info.get("status") or "").strip().lower() == "rejected"
            )

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
            for src in sources:
                try:
                    df = self._read_sql(
                        "SELECT item_id FROM proc.po_line_items_agent WHERE po_id = %s",
                        params=(src,),
                    )
                    if df.empty:
                        df = self._read_sql(
                            "SELECT item_id FROM proc.invoice_line_items_agent WHERE invoice_id = %s",
                            params=(src,),
                        )
                    if df.empty:
                        continue
                    item_id = str(df["item_id"].dropna().iloc[0])
                    logger.debug(
                        "_find_candidate_suppliers inferred item_id %s from source %s",
                        item_id,
                        src,
                    )
                    break
                except Exception:
                    logger.exception("Failed to infer item_id from source %s", src)

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

