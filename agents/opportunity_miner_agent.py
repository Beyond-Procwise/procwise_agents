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
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from models.opportunity_priority_model import OpportunityPriorityModel
from services.data_flow_manager import DataFlowManager
from services.opportunity_service import load_opportunity_feedback
from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources, normalize_instruction_key

logger = logging.getLogger(__name__)


_CATALOG_MATCH_THRESHOLD = 0.45

_PURCHASE_LINE_VALUE_COLUMNS = [
    "line_amount_gbp",
    "total_amount_incl_tax_gbp",
    "line_total_gbp",
    "total_amount_gbp",
    "line_amount",
    "total_amount_incl_tax",
    "line_total",
    "total_amount",
]

_INVOICE_LINE_VALUE_COLUMNS = [
    "line_amount_gbp",
    "total_amount_incl_tax_gbp",
    "line_total_gbp",
    "total_amount_gbp",
    "line_amount",
    "total_amount_incl_tax",
    "line_total",
    "total_amount",
]


_CONDITION_KEY_ALIASES: Dict[str, set[str]] = {
    "supplier_id": {
        "supplier",
        "supplierid",
        "supplier_identifier",
        "supplier_code",
        "supplier_reference",
        "vendor",
        "vendor_id",
        "vendorid",
    },
    "supplier_name": {
        "suppliername",
        "vendor_name",
        "vendorname",
        "trading_name",
    },
    "contract_id": {
        "contract",
        "contractid",
        "contract_identifier",
        "contract_reference",
        "contract_number",
    },
    "po_id": {
        "po",
        "purchase_order_id",
        "purchaseorderid",
        "po_number",
        "purchase_order_number",
    },
    "invoice_id": {
        "invoice",
        "invoiceid",
        "invoice_number",
        "invoice_reference",
    },
    "item_id": {
        "item",
        "itemid",
        "item_code",
        "itemcode",
        "sku",
        "product_id",
        "productid",
        "material_id",
        "materialid",
        "item_reference",
    },
    "item_description": {
        "itemdesc",
        "item_description",
        "product_description",
        "productdesc",
        "service_description",
        "item_name",
        "product_name",
    },
    "actual_price": {
        "actualprice",
        "current_price",
        "currentprice",
        "unit_price",
        "unitprice",
        "price_paid",
        "paid_price",
        "current_unit_price",
    },
    "benchmark_price": {
        "benchmarkprice",
        "target_price",
        "targetprice",
        "reference_price",
        "referenceprice",
        "benchmark_unit_price",
    },
    "quantity": {
        "qty",
        "quantity_ordered",
        "ordered_quantity",
        "order_quantity",
        "volume",
    },
    "variance_threshold_pct": {
        "variance_threshold",
        "variance_thresholdpct",
        "variancepct",
        "threshold_pct",
        "thresholdpct",
    },
    "minimum_volume_gbp": {
        "min_volume_gbp",
        "minimum_volume",
        "minimum_spend_threshold",
        "minimum_spend_gbp",
        "volume_threshold",
    },
    "minimum_value_gbp": {
        "min_value_gbp",
        "minimum_value",
        "minimum_spend_gbp",
        "minimum_spend_threshold",
    },
    "minimum_overlap_gbp": {
        "min_overlap_gbp",
        "overlap_threshold_gbp",
    },
    "risk_threshold": {
        "risk_score_threshold",
        "minimum_risk_score",
    },
    "lookback_period_days": {
        "lookback_days",
        "lookback_period",
    },
    "negotiation_window_days": {
        "negotiation_window",
        "window_days",
        "renewal_window_days",
    },
    "reference_date": {
        "analysis_date",
        "base_date",
    },
    "market_inflation_pct": {
        "inflation_rate_pct",
        "inflation_pct",
        "market_inflation",
    },
    "minimum_unused_value_gbp": {
        "unused_value_threshold_gbp",
        "min_unused_value_gbp",
    },
}


_NUMERIC_CONDITION_KEYS: set[str] = {
    "actual_price",
    "benchmark_price",
    "quantity",
    "variance_threshold_pct",
    "minimum_volume_gbp",
    "minimum_value_gbp",
    "minimum_overlap_gbp",
    "risk_threshold",
    "market_inflation_pct",
    "minimum_unused_value_gbp",
}


_INTEGER_CONDITION_KEYS: set[str] = {
    "lookback_period_days",
    "negotiation_window_days",
}


def _build_condition_alias_maps() -> Tuple[Dict[str, set[str]], Dict[str, str]]:
    normalised: Dict[str, set[str]] = {}
    reverse: Dict[str, str] = {}

    for canonical, aliases in _CONDITION_KEY_ALIASES.items():
        values = {canonical, *aliases}
        normals: set[str] = set()
        for alias in values:
            key = normalize_instruction_key(alias)
            if key:
                normals.add(key)
        if not normals:
            continue
        normalised[canonical] = normals
        for key in normals:
            reverse.setdefault(key, canonical)

    return normalised, reverse


_NORMALISED_CONDITION_ALIASES, _CONDITION_ALIAS_LOOKUP = _build_condition_alias_maps()


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
    opportunity_ref_id: Optional[str] = None
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
    ml_priority_score: Optional[float] = None

    _PUBLIC_FIELDS: Tuple[str, ...] = (
        "opportunity_id",
        "opportunity_ref_id",
        "detector_type",
        "policy_id",
        "supplier_id",
        "supplier_name",
        "category_id",
        "item_id",
        "item_reference",
        "financial_impact_gbp",
        "calculation_details",
        "source_records",
        "detected_on",
        "weightage",
        "candidate_suppliers",
        "context_documents",
        "ml_priority_score",
        "feedback_status",
        "feedback_reason",
        "feedback_updated_at",
        "feedback_user",
        "feedback_metadata",
        "is_rejected",
    )

    _ALWAYS_INCLUDE: Tuple[str, ...] = (
        "calculation_details",
        "source_records",
        "candidate_suppliers",
        "context_documents",
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation limited to public fields."""

        serialised: Dict[str, Any] = {}
        for field_name in self._PUBLIC_FIELDS:
            value = getattr(self, field_name, None)

            if isinstance(value, datetime):
                serialised[field_name] = value.isoformat()
                continue

            if value is None:
                if field_name in self._ALWAYS_INCLUDE:
                    # Preserve structural fields even when empty.
                    serialised[field_name] = [] if field_name.endswith("records") else {}
                continue

            if field_name in self._ALWAYS_INCLUDE:
                serialised[field_name] = value
                continue

            if isinstance(value, (list, dict)):
                if not value and field_name not in self._ALWAYS_INCLUDE:
                    continue
                serialised[field_name] = value
                continue

            serialised[field_name] = value

        # Ensure empty defaults for always-included structured fields when the
        # dataclass attribute was present but falsey (e.g. ``[]`` or ``{}``).
        for field_name in self._ALWAYS_INCLUDE:
            if field_name not in serialised:
                serialised[field_name] = (
                    [] if field_name in {"source_records", "candidate_suppliers", "context_documents"} else {}
                )

        return serialised


class OpportunityMinerAgent(BaseAgent):
    """Agent for identifying procurement anomalies and savings opportunities."""

    def __init__(self, agent_nick, min_financial_impact: float = 100.0) -> None:
        super().__init__(agent_nick)
        self.min_financial_impact = min_financial_impact
        self._opportunity_sequence: int = 0
        self._supplier_lookup: Dict[str, Optional[str]] = {}
        self._supplier_alias_lookup: Dict[str, List[str]] = {}
        self._supplier_reference_lookup: Dict[str, Dict[str, Any]] = {}
        self._contract_supplier_map: Dict[str, str] = {}
        self._contract_metadata: Dict[str, Dict[str, Any]] = {}
        self._po_supplier_map: Dict[str, str] = {}
        self._po_contract_map: Dict[str, str] = {}
        self._invoice_supplier_map: Dict[str, str] = {}
        self._invoice_po_map: Dict[str, str] = {}
        self._item_supplier_map: Dict[str, Dict[str, float]] = {}
        self._item_supplier_frequency: Dict[str, Dict[str, int]] = {}
        self._item_description_supplier_map: Dict[str, Dict[str, float]] = {}
        self._item_description_supplier_frequency: Dict[str, Dict[str, int]] = {}
        self._supplier_risk_map: Dict[str, float] = {}
        self._event_log: List[Dict[str, Any]] = []
        self._escalations: List[Dict[str, Any]] = []
        self._column_cache: Dict[str, set[str]] = {}
        self._policy_engine_catalog: Optional[Dict[str, Dict[str, Any]]] = None
        self._data_profile: Dict[str, Any] = {}
        self._data_flow_manager: Optional[DataFlowManager] = None
        self._data_flow_snapshot: Dict[str, Any] = {}
        self._supplier_flow_index: Dict[str, Dict[str, Any]] = {}
        self._supplier_flow_name_index: Dict[str, Dict[str, Any]] = {}
        self._priority_model = OpportunityPriorityModel()

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

    @staticmethod
    def _normalise_supplier_token(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value)
        except Exception:
            return None
        text = text.strip().lower()
        return text or None

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

    def _resolve_canonical_condition_key(self, key: Any) -> Optional[str]:
        normalised = normalize_instruction_key(key)
        if not normalised:
            return None
        canonical = _CONDITION_ALIAS_LOOKUP.get(normalised)
        if canonical:
            return canonical
        for candidate, aliases in _NORMALISED_CONDITION_ALIASES.items():
            if normalised in aliases:
                return candidate
        return None

    def _is_condition_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return False
        except Exception:
            pass
        return True

    def _has_condition_value(self, conditions: Dict[str, Any], key: str) -> bool:
        if key not in conditions:
            return False
        existing = conditions.get(key)
        if existing is None:
            return False
        if isinstance(existing, str) and not existing.strip():
            return False
        try:
            if pd.isna(existing):  # type: ignore[arg-type]
                return False
        except Exception:
            pass
        return True

    def _coerce_condition_value(self, key: str, value: Any) -> Any:
        if key in _INTEGER_CONDITION_KEYS:
            try:
                numeric = int(float(value))
                return numeric
            except Exception:
                return value
        if key in _NUMERIC_CONDITION_KEYS:
            coerced = self._coerce_float(value)
            if coerced is not None:
                return coerced
        return value

    def _assign_condition(
        self,
        conditions: Dict[str, Any],
        key: str,
        value: Any,
        *,
        override: bool = False,
    ) -> None:
        if not self._is_condition_value(value):
            return
        if not override and self._has_condition_value(conditions, key):
            return
        conditions[key] = self._coerce_condition_value(key, value)

    def _merge_conditions_from_source(
        self,
        conditions: Dict[str, Any],
        source: Any,
        *,
        override: bool = False,
    ) -> None:
        if not isinstance(conditions, dict):
            return

        visited: set[int] = set()

        def _walk(payload: Any) -> None:
            if payload is None:
                return
            if isinstance(payload, dict):
                payload_id = id(payload)
                if payload_id in visited:
                    return
                visited.add(payload_id)
                if payload is conditions:
                    return
                for raw_key, raw_value in payload.items():
                    canonical = self._resolve_canonical_condition_key(raw_key)
                    if canonical and not isinstance(raw_value, (dict, list, tuple, set)):
                        self._assign_condition(
                            conditions, canonical, raw_value, override=override
                        )
                    else:
                        _walk(raw_value)
                return
            if isinstance(payload, (list, tuple, set)):
                for item in payload:
                    _walk(item)
                return
            if isinstance(payload, str):
                text = payload.strip()
                if not text:
                    return
                try:
                    parsed = json.loads(text)
                except Exception:
                    return
                _walk(parsed)

        _walk(source)

    def _find_column_for_key(self, df: pd.DataFrame, canonical_key: str) -> Optional[str]:
        if df.empty:
            return None
        aliases = _NORMALISED_CONDITION_ALIASES.get(canonical_key, set())
        if not aliases:
            aliases = {normalize_instruction_key(canonical_key)}
        for column in df.columns:
            if normalize_instruction_key(column) in aliases:
                return column
        return None

    def _resolve_condition_value(self, container: Any, key: str) -> Any:
        if not isinstance(container, dict):
            return None
        value = container.get(key)
        if self._is_condition_value(value):
            return value
        aliases = _NORMALISED_CONDITION_ALIASES.get(key, set())
        if not aliases:
            aliases = {normalize_instruction_key(key)}
        for candidate_key, candidate_value in container.items():
            if not self._is_condition_value(candidate_value):
                continue
            if normalize_instruction_key(candidate_key) in aliases:
                return candidate_value
        return None

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

        self._merge_conditions_from_source(conditions, instructions)


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

        conditions = context.input_data.get("conditions")
        if isinstance(conditions, dict):
            self._merge_conditions_from_source(conditions, context.input_data)

    def _ensure_instruction_payloads(self, context: AgentContext) -> None:
        if not isinstance(context.input_data, dict):
            return

        prompts_missing = not context.input_data.get("prompts")
        policies_missing = not context.input_data.get("policies")
        if not prompts_missing and not policies_missing:
            return

        agent_slug = "opportunity_miner"

        if prompts_missing:
            prompt_engine = getattr(self.agent_nick, "prompt_engine", None)
            if prompt_engine is None:
                try:
                    from orchestration.prompt_engine import PromptEngine

                    prompt_engine = PromptEngine(self.agent_nick)
                    setattr(self.agent_nick, "prompt_engine", prompt_engine)
                except Exception:  # pragma: no cover - defensive loading
                    logger.exception("Failed to initialise prompt engine for opportunity miner")
                    prompt_engine = None
            if prompt_engine is not None:
                try:
                    prompts = prompt_engine.prompts_for_agent(agent_slug)
                except Exception:  # pragma: no cover - defensive fetch
                    logger.exception("Failed to load prompts for opportunity miner")
                    prompts = []
                if prompts:
                    context.input_data["prompts"] = prompts

        if policies_missing:
            policy_engine = getattr(self.agent_nick, "policy_engine", None)
            if policy_engine is None:
                try:
                    from engines.policy_engine import PolicyEngine

                    policy_engine = PolicyEngine(self.agent_nick)
                    setattr(self.agent_nick, "policy_engine", policy_engine)
                except Exception:  # pragma: no cover - defensive loading
                    logger.exception("Failed to initialise policy engine for opportunity miner")
                    policy_engine = None
            if policy_engine is not None:
                try:
                    policies = [
                        dict(policy)
                        for policy in policy_engine.iter_policies()
                        if agent_slug in policy.get("policy_linked_agents", [])
                    ]
                except Exception:  # pragma: no cover - defensive fetch
                    logger.exception("Failed to load policies for opportunity miner")
                    policies = []
                if policies:
                    context.input_data["policies"] = policies

    def _finding_uniqueness_key(
        self, finding: Finding, policy_display: str
    ) -> Tuple[str, str]:
        """Return a stable key used to de-duplicate supplier opportunities."""

        def _candidate_key(*values: Any) -> Optional[str]:
            for value in values:
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    joined = "|".join(str(part).strip() for part in value if part is not None)
                    normalised = self._normalise_identifier(joined)
                else:
                    normalised = self._normalise_identifier(value)
                if normalised:
                    return normalised
            return None

        opportunity_key = _candidate_key(
            finding.opportunity_ref_id,
            finding.opportunity_id,
            (finding.detector_type, finding.policy_id, finding.item_id),
            (policy_display, finding.item_reference or finding.category_id),
        )
        if not opportunity_key:
            opportunity_key = self._normalise_identifier(policy_display) or "UNKNOWN"

        supplier_key = self._normalise_identifier(finding.supplier_id) or "_"
        return opportunity_key, supplier_key

    def _merge_duplicate_finding_metadata(
        self, primary: Finding, duplicate: Finding
    ) -> None:
        """Preserve contextual metadata when collapsing duplicate findings."""

        if duplicate.policy_id and not primary.policy_id:
            primary.policy_id = duplicate.policy_id

        primary_details: Dict[str, Any] = (
            dict(primary.calculation_details)
            if isinstance(primary.calculation_details, dict)
            else {}
        )
        duplicate_details: Dict[str, Any] = (
            dict(duplicate.calculation_details)
            if isinstance(duplicate.calculation_details, dict)
            else {}
        )

        related_policies: List[str] = []
        if primary.policy_id:
            related_policies.append(str(primary.policy_id))
        if duplicate.policy_id:
            related_policies.append(str(duplicate.policy_id))
        for payload in (primary_details.get("related_policies"), duplicate_details.get("related_policies")):
            if isinstance(payload, list):
                related_policies.extend(str(entry) for entry in payload if entry)
        if related_policies:
            primary_details["related_policies"] = sorted({p for p in related_policies if p})

        for key, value in duplicate_details.items():
            if key not in primary_details or primary_details[key] in (None, ""):
                primary_details[key] = value

        def _merge_list(field: str) -> None:
            primary_list = getattr(primary, field, None)
            duplicate_list = getattr(duplicate, field, None)
            if not isinstance(primary_list, list) or not isinstance(duplicate_list, list):
                return
            merged: List[Any] = []
            seen: set[str] = set()

            def _serialise(value: Any) -> str:
                try:
                    return json.dumps(value, sort_keys=True, default=str)
                except TypeError:
                    return str(value)

            for candidate in primary_list + duplicate_list:
                marker = _serialise(candidate)
                if marker in seen:
                    continue
                seen.add(marker)
                merged.append(candidate)
            setattr(primary, field, merged)

        _merge_list("source_records")
        _merge_list("candidate_suppliers")
        _merge_list("context_documents")

        primary.calculation_details = primary_details

    def _apply_policy_category_limits(
        self, per_policy: Dict[str, List[Finding]]
    ) -> Tuple[List[Finding], Dict[str, Dict[str, List[Finding]]]]:

        """Retain the top findings per policy while keeping category insights."""

        aggregated: List[Finding] = []
        seen_map: Dict[Tuple[str, str], int] = {}
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

            unique_sorted: List[Finding] = []
            seen_policy: set[Tuple[str, str]] = set()
            for finding in sorted_items:
                key = self._finding_uniqueness_key(finding, display)
                if key in seen_policy:
                    continue
                seen_policy.add(key)
                unique_sorted.append(finding)

            top_findings: List[Finding] = []
            seen_suppliers: set[str] = set()
            for finding in unique_sorted:
                supplier_key = self._normalise_identifier(finding.supplier_id) or "_"
                if supplier_key in seen_suppliers:
                    continue
                seen_suppliers.add(supplier_key)
                top_findings.append(finding)
                if len(top_findings) >= 2:
                    break

            if not top_findings and unique_sorted:
                top_findings = unique_sorted[:1]

            policy_categories: Dict[str, List[Finding]] = {"all": top_findings}

            for finding in unique_sorted:
                cat_raw = finding.category_id
                category = str(cat_raw).strip() if cat_raw else "uncategorized"
                policy_categories.setdefault(category, []).append(finding)

            per_policy[display] = top_findings
            category_map[display] = policy_categories

            for finding in top_findings:
                key = self._finding_uniqueness_key(finding, display)
                if key in seen_map:
                    existing_index = seen_map[key]
                    existing_finding = aggregated[existing_index]
                    preferred = existing_finding
                    comparison = finding
                    if finding.financial_impact_gbp > existing_finding.financial_impact_gbp:
                        aggregated[existing_index] = finding
                        preferred = finding
                        comparison = existing_finding
                    self._merge_duplicate_finding_metadata(preferred, comparison)
                    continue
                seen_map[key] = len(aggregated)
                aggregated.append(finding)

        return aggregated, category_map

    def _serialize_findings(
        self, findings: Iterable[Finding], *, limit: int = 2
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            [f for f in findings if isinstance(f, Finding)],
            key=lambda f: f.financial_impact_gbp,
            reverse=True,
        )
        if limit <= 0:
            return [f.as_dict() for f in ordered]
        return [f.as_dict() for f in ordered[:limit]]

    def _limit_opportunity_dicts(
        self, entries: Iterable[Dict[str, Any]], *, limit: int = 2
    ) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict):
                valid.append(entry)
        if not valid:
            return []

        def _impact(value: Dict[str, Any]) -> float:
            raw = value.get("financial_impact_gbp")
            if raw is None:
                raw = value.get("financialImpactGBP")
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        ordered = sorted(valid, key=_impact, reverse=True)
        if limit <= 0:
            return ordered
        return ordered[:limit]

    def _get_data_flow_manager(self) -> Optional[DataFlowManager]:
        if self._data_flow_manager is None:
            try:
                self._data_flow_manager = DataFlowManager(self.agent_nick)
            except Exception:  # pragma: no cover - defensive lazy init
                logger.exception("Failed to initialise DataFlowManager")
                self._data_flow_manager = None
        return self._data_flow_manager

    def _capture_data_flow(self, tables: Dict[str, pd.DataFrame]) -> None:
        self._data_flow_snapshot = {}
        self._supplier_flow_index = {}
        self._supplier_flow_name_index = {}
        if not isinstance(tables, dict) or not tables:
            return
        manager = self._get_data_flow_manager()
        if manager is None:
            return
        try:
            relations, graph = manager.build_data_flow_map(
                tables, table_name_map=self.TABLE_MAP
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to build data flow map")
            return
        self._data_flow_snapshot = {
            "relationships": relations,
            "graph": graph,
        }
        flows_raw = []
        if isinstance(graph, dict):
            flows_raw = graph.get("supplier_flows") or []
        flow_index: Dict[str, Dict[str, Any]] = {}
        flow_name_index: Dict[str, Dict[str, Any]] = {}
        for flow in flows_raw:
            if not isinstance(flow, dict):
                continue
            supplier_id = flow.get("supplier_id")
            if supplier_id is not None:
                key = str(supplier_id).strip()
                if key:
                    flow_index[key] = flow
            supplier_name = flow.get("supplier_name")
            token = self._normalise_supplier_token(supplier_name)
            if token:
                flow_name_index.setdefault(token, flow)
        self._supplier_flow_index = flow_index
        self._supplier_flow_name_index = flow_name_index
        try:
            manager.persist_knowledge_graph(relations, graph)
        except Exception:  # pragma: no cover - persistence best effort
            logger.exception("Failed to persist knowledge graph")

    def _summarise_top_opportunities(
        self,
        per_policy: Dict[str, List[Finding]],
        per_policy_categories: Optional[Dict[str, Dict[str, List[Finding]]]] = None,
        *,
        limit: int = 3,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        categories = per_policy_categories or {}

        for display, findings in per_policy.items():
            valid = [finding for finding in findings if isinstance(finding, Finding)]
            category_details = categories.get(display, {})
            category_breakdown: Dict[str, float] = {}
            for category, bucket in category_details.items():
                total = sum(
                    f.financial_impact_gbp
                    for f in bucket
                    if isinstance(f, Finding)
                )
                category_breakdown[str(category)] = float(total)

            if not valid:
                summary[display] = {
                    "total_opportunities": 0,
                    "total_financial_impact_gbp": 0.0,
                    "top_opportunities": [],
                    "categories": category_breakdown,
                }
                continue

            ordered = sorted(
                valid,
                key=lambda finding: finding.financial_impact_gbp,
                reverse=True,
            )
            total_impact = float(sum(f.financial_impact_gbp for f in valid))
            top_entries: List[Dict[str, Any]] = []
            for index, finding in enumerate(ordered[: max(limit, 1)]):
                top_entries.append(
                    {
                        "rank": index + 1,
                        "opportunity_id": finding.opportunity_id,
                        "opportunity_ref_id": finding.opportunity_ref_id,
                        "detector_type": finding.detector_type,
                        "supplier_id": finding.supplier_id,
                        "supplier_name": finding.supplier_name,
                        "category_id": finding.category_id,
                        "item_id": finding.item_id,
                        "financial_impact_gbp": float(finding.financial_impact_gbp),
                        "weightage": float(finding.weightage),
                        "policy_id": finding.policy_id,
                    }
                )

            summary[display] = {
                "total_opportunities": len(valid),
                "total_financial_impact_gbp": total_impact,
                "top_opportunities": top_entries,
                "categories": category_breakdown,
            }

        return summary

    def _summarise_supplier_link_issue(self) -> Optional[str]:
        snapshot = self._data_flow_snapshot or {}
        relationships = snapshot.get("relationships") or []
        for relation in relationships:
            source = relation.get("source_table")
            target = relation.get("target_table")
            if "proc.supplier" not in (source, target):
                continue
            status = relation.get("status")
            if status in {"linked", None}:
                continue
            source_label = relation.get("source_table_alias") or source
            target_label = relation.get("target_table_alias") or target
            status_label = {
                "missing_tables": "required tables are missing",
                "missing_data": "required tables are empty",
                "missing_column_source": "source column is unavailable",
                "missing_column_target": "target column is unavailable",
                "insufficient_values": "insufficient values to determine linkage",
                "no_overlap": "no shared identifiers between tables",
                "error": "relationship analysis failed",
            }.get(status, status.replace("_", " "))
            return f"{source_label} ↔ {target_label}: {status_label}"
        return None

    def _lookup_supplier_flow(
        self, supplier_id: Optional[Any], supplier_name: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        candidate_id = str(supplier_id).strip() if supplier_id is not None else ""
        if candidate_id and candidate_id in self._supplier_flow_index:
            return self._supplier_flow_index.get(candidate_id)

        tokens: List[str] = []
        if supplier_name:
            token = self._normalise_supplier_token(supplier_name)
            if token:
                tokens.append(token)

        if candidate_id and candidate_id in self._supplier_lookup:
            lookup_name = self._supplier_lookup.get(candidate_id)
            token = self._normalise_supplier_token(lookup_name)
            if token:
                tokens.append(token)
        if candidate_id and candidate_id in self._supplier_alias_lookup:
            aliases = self._supplier_alias_lookup.get(candidate_id) or []
            for alias in aliases:
                token = self._normalise_supplier_token(alias)
                if token:
                    tokens.append(token)

        for token in tokens:
            flow = self._supplier_flow_name_index.get(token)
            if flow:
                return flow
        return None

    def _supplier_flow_coverage(
        self, supplier_id: Optional[Any], supplier_name: Optional[Any] = None
    ) -> float:
        entry = self._lookup_supplier_flow(supplier_id, supplier_name)
        if not isinstance(entry, dict):
            return 0.0
        coverage = entry.get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            value = float(coverage)
            if value < 0:
                return 0.0
            if value > 1.0:
                return min(value, 1.0)
            return value
        total = 0
        hits = 0
        for key in ("contracts", "purchase_orders", "invoices", "quotes"):
            details = entry.get(key)
            if isinstance(details, dict):
                total += 1
                if details.get("count"):
                    hits += 1
        return hits / total if total else 0.0

    def _normalise_risk_score(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric < 0:
            return 0.0
        if numeric > 1.0:
            numeric = min(numeric / 100.0, 1.0)
        return min(numeric, 1.0)

    def _determine_supplier_gap_reason(
        self,
        display: str,
        retained: List[Finding],
        policy_payload: Optional[Dict[str, Any]],
        tables: Dict[str, pd.DataFrame],
        *,
        min_impact_threshold: float,
        explicit_min_override: bool,
    ) -> Optional[str]:
        if retained:
            if not any(f.supplier_id for f in retained):
                return (
                    "Policy findings do not include supplier identifiers after evaluating the "
                    "procurement data flow."
                )
            return None

        supplier_table = None
        for key in ("supplier_master", "proc.supplier"):
            df = tables.get(key)
            if df is not None:
                supplier_table = df
                break
        if supplier_table is not None and supplier_table.empty:
            return (
                "Supplier master dataset is empty; unable to highlight suppliers for this policy."
            )

        linkage_issue = self._summarise_supplier_link_issue()
        if linkage_issue:
            return f"Supplier linkage incomplete: {linkage_issue}"

        findings: List[Any] = []
        if isinstance(policy_payload, dict):
            payload_findings = policy_payload.get("findings", [])
            if isinstance(payload_findings, list):
                findings = payload_findings

        def _supplier_from(record: Any) -> Optional[str]:
            if isinstance(record, Finding):
                return record.supplier_id
            if isinstance(record, dict):
                value = record.get("supplier_id")
                if value:
                    text = str(value).strip()
                    return text or None
            return None

        suppliers_in_findings = [
            _supplier_from(item)
            for item in findings
            if _supplier_from(item) is not None
        ]

        if findings and not suppliers_in_findings:
            return (
                "Policy calculations produced opportunities without supplier identifiers."
            )

        if suppliers_in_findings and not retained:
            if not explicit_min_override and min_impact_threshold > 0:
                return (
                    "Available opportunities did not meet the configured minimum financial impact "
                    "threshold, so no suppliers were highlighted."
                )
            return "Policy findings were filtered out by the provided configuration."

        return "No qualifying transactions matched the policy criteria for the available data."


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
                df = pd.DataFrame(rows, columns=columns)
                return self._normalise_numeric_dataframe(df)

        engine_getter = getattr(self.agent_nick, "get_db_engine", None)
        engine = engine_getter() if callable(engine_getter) else None
        if engine is not None:
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
                return self._normalise_numeric_dataframe(df)

        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        if callable(pandas_conn):
            with pandas_conn() as conn:
                if hasattr(conn, "cursor"):
                    return _fetch_with_cursor(conn)
                df = pd.read_sql(query, conn, params=params)
                return self._normalise_numeric_dataframe(df)

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
            input_data = context.input_data or {}
            conditions = input_data.get("conditions")
            if not isinstance(conditions, dict):
                conditions = {}
                input_data["conditions"] = conditions

            resolved_supplier = self._resolve_supplier_id(
                input_data.get("supplier_id"), event=input_data
            )
            if resolved_supplier:
                input_data["supplier_id"] = resolved_supplier
                self._assign_condition(
                    conditions, "supplier_id", resolved_supplier, override=True
                )
            else:
                has_condition_values = any(
                    self._is_condition_value(value)
                    for key, value in conditions.items()
                    if key not in {"rfq_id"}
                )
                policies = input_data.get("policies")
                has_policies = isinstance(policies, (list, tuple)) and any(policies)
                if not has_condition_values and not has_policies:
                    logger.info(
                        "OpportunityMinerAgent skipping run due to missing supplier identifier"
                    )
                    return AgentOutput(
                        status=AgentStatus.SUCCESS,
                        data={"skipped": True, "reason": "missing_supplier_id"},
                    )
            input_keys = sorted(input_data.keys()) if isinstance(input_data, dict) else []
            preview = ", ".join(input_keys[:5]) if input_keys else "none"
            if len(input_keys) > 5:
                preview = f"{preview}, …"
            logger.info(
                "OpportunityMinerAgent starting processing with %d input field(s): %s",
                len(input_keys),
                preview,
            )
            self._opportunity_sequence = 0
            qe = getattr(self.agent_nick, "query_engine", None)
            workflow_completed = False
            self._data_profile = {}
            tables = self._ingest_data()
            tables = self._validate_data(tables)
            self._build_supplier_lookup(tables)
            tables = self._normalise_currency(tables)
            tables = self._apply_index_adjustment(tables)
            self._data_profile = self._build_data_profile(tables)
            self._capture_data_flow(tables)

            self._event_log = []
            self._escalations = []
            notifications: set[str] = set()
            self._policy_engine_catalog = None

            self._ensure_instruction_payloads(context)
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
            display_counts: Dict[str, int] = {}
            display_variants: Dict[str, List[str]] = {}

            for key in requested_keys:
                policy_cfg = policy_registry.get(key)
                if not policy_cfg:
                    continue
                policy_input = dict(context.input_data)
                policy_input["conditions"] = dict(base_conditions)
                self._merge_conditions_from_source(
                    policy_input["conditions"], policy_cfg.get("default_conditions")
                )
                self._merge_conditions_from_source(
                    policy_input["conditions"], policy_cfg.get("parameters")
                )
                self._merge_conditions_from_source(
                    policy_input["conditions"], policy_cfg.get("rules")
                )
                self._merge_conditions_from_source(
                    policy_input["conditions"], policy_cfg.get("source_policy")
                )


                required_fields = policy_cfg.get("required_fields") or []
                supplier_autodetect = bool(policy_cfg.get("supplier_autodetect"))
                needs_supplier = "supplier_id" in required_fields
                if needs_supplier:
                    conditions = policy_input.get("conditions", {})
                    supplier_metadata: Optional[Dict[str, Any]] = None
                    supplier_token = (
                        conditions.get("supplier_id") if isinstance(conditions, dict) else None
                    )
                    if not self._is_condition_value(supplier_token):
                        resolved_supplier, supplier_metadata = self._resolve_policy_supplier(policy_input)
                        conditions = policy_input.get("conditions", {})
                        supplier_token = (
                            conditions.get("supplier_id") if isinstance(conditions, dict) else None
                        )
                        if resolved_supplier:
                            supplier_metadata = None
                            supplier_token = resolved_supplier

                    if not self._is_condition_value(supplier_token) and not supplier_autodetect:
                        policy_id = policy_cfg.get("policy_id", "unknown")
                        message = (
                            supplier_metadata.get("message")
                            if supplier_metadata and isinstance(supplier_metadata, dict)
                            else None
                        )
                        if not message:
                            message = "Supplier identifier missing from policy conditions"
                        details: Dict[str, Any] = {}
                        if isinstance(supplier_metadata, dict):
                            details.update(supplier_metadata)
                        details.setdefault("missing_fields", ["supplier_id"])
                        self._log_policy_event(
                            policy_id,
                            None,
                            "blocked",
                            message,
                            details,
                        )
                        continue

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
                if isinstance(context.input_data.get("conditions"), dict):
                    context.input_data["conditions"].update(policy_input["conditions"])
                aggregated_findings.extend(findings)
                display_name = (
                    policy_cfg.get("policy_name")
                    or policy_cfg.get("detector")
                    or key
                )
                counter = display_counts.get(display_name, 0)
                unique_display = display_name
                if counter:
                    suffix_parts: List[str] = []
                    policy_id = policy_cfg.get("policy_id")
                    if policy_id:
                        suffix_parts.append(str(policy_id))
                    slug = policy_cfg.get("policy_slug") or key
                    if slug and str(slug) not in suffix_parts:
                        suffix_parts.append(str(slug))
                    suffix = " / ".join(part for part in suffix_parts if part)
                    if not suffix:
                        suffix = str(counter + 1)
                    unique_display = f"{display_name} ({suffix})"
                display_counts[display_name] = counter + 1
                display_variants.setdefault(display_name, []).append(unique_display)
                policy_runs[key] = {
                    "policy_cfg": policy_cfg,
                    "display": unique_display,
                    "findings": findings,
                }

            if not policy_runs and aggregated_findings:
                display = workflow_name or "opportunity_policy"
                counter = display_counts.get(display, 0)
                if counter:
                    display = f"{display} ({counter + 1})"
                display_counts[display] = counter + 1
                display_variants.setdefault(display, []).append(display)
                policy_runs[display] = {
                    "policy_cfg": {
                        "policy_id": "unknown",
                        "policy_name": display,
                        "detector": display,
                    },
                    "display": display,
                    "findings": aggregated_findings,
                }

            policy_runs_by_display: Dict[str, Dict[str, Any]] = {
                payload.get("display", key): payload
                for key, payload in policy_runs.items()
                if payload
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
                                "opportunity_ref_id": best.opportunity_ref_id,
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
                                    "opportunity_ref_id": best.opportunity_ref_id,
                                },
                            )
                    else:
                        per_policy_retained[display] = []

            filtered, per_policy_categories = self._apply_policy_category_limits(
                per_policy_retained
            )

            try:
                scores = self._priority_model.assign_scores(
                    [finding.financial_impact_gbp for finding in filtered]
                )
                for finding, score in zip(filtered, scores):
                    finding.ml_priority_score = float(score)
            except Exception:  # pragma: no cover - defensive scoring
                logger.exception("Priority model scoring failed")

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
            policy_metadata: Dict[str, Dict[str, Any]] = {}
            for key in requested_keys:
                payload = policy_runs.get(key)
                if not payload:
                    continue
                display = payload.get("display", key)
                retained = per_policy_retained.get(display, [])
                policy_opportunities[display] = self._serialize_findings(
                    retained, limit=2
                )
                policy_suppliers[display] = sorted(
                    {
                        str(f.supplier_id).strip()
                        for f in retained
                        if f.supplier_id and str(f.supplier_id).strip()
                    }
                )
                categories = per_policy_categories.get(display, {})
                policy_category_opportunities[display] = {
                    category: self._serialize_findings(findings, limit=2)
                    for category, findings in categories.items()
                }
                policy_cfg = payload.get("policy_cfg", {})
                if isinstance(policy_cfg, dict):
                    entry = {
                        "policy_id": policy_cfg.get("policy_id")
                        or policy_cfg.get("policyId"),
                        "policy_slug": policy_cfg.get("policy_slug"),
                        "detector": policy_cfg.get("detector"),
                    }
                    policy_metadata[display] = {
                        key: value for key, value in entry.items() if value is not None
                    }

            for display, retained in per_policy_retained.items():
                policy_opportunities.setdefault(
                    display,
                    self._serialize_findings(retained, limit=2),
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
                        category: self._serialize_findings(findings, limit=2)
                        for category, findings in categories.items()
                    },
                )
                policy_metadata.setdefault(display, {})

            for root_display, variants in display_variants.items():
                variant_suppliers = sorted(
                    {
                        supplier
                        for variant in variants
                        for supplier in policy_suppliers.get(variant, [])
                    }
                )
                if variant_suppliers and (
                    root_display not in policy_suppliers
                    or not policy_suppliers[root_display]
                ):
                    policy_suppliers[root_display] = variant_suppliers

                variant_opportunities = []
                for variant in variants:
                    variant_entries = policy_opportunities.get(variant, [])
                    if isinstance(variant_entries, list):
                        variant_opportunities.extend(variant_entries)
                limited_variant = self._limit_opportunity_dicts(
                    variant_opportunities, limit=2
                )
                if limited_variant and (
                    root_display not in policy_opportunities
                    or not policy_opportunities[root_display]
                ):
                    policy_opportunities[root_display] = limited_variant

                category_union: Dict[str, List[Dict[str, Any]]] = {}
                for variant in variants:
                    categories = policy_category_opportunities.get(variant, {})
                    for category, entries in categories.items():
                        if isinstance(entries, list):
                            category_union.setdefault(category, []).extend(entries)
                if category_union and (
                    root_display not in policy_category_opportunities
                    or not policy_category_opportunities[root_display]
                ):
                    policy_category_opportunities[root_display] = {
                        category: self._limit_opportunity_dicts(entries, limit=2)
                        for category, entries in category_union.items()
                    }

                for variant in variants:
                    if (
                        root_display not in policy_metadata
                        and variant in policy_metadata
                    ):
                        policy_metadata[root_display] = dict(policy_metadata[variant])

            policy_supplier_gaps: Dict[str, str] = {}
            for display, suppliers in policy_suppliers.items():
                if suppliers:
                    continue
                payload = policy_runs_by_display.get(display)
                retained = per_policy_retained.get(display, [])
                reason = self._determine_supplier_gap_reason(
                    display,
                    retained,
                    payload,
                    tables,
                    min_impact_threshold=min_impact_threshold,
                    explicit_min_override=explicit_min_override,
                )
                if reason:
                    policy_supplier_gaps[display] = reason

            # compute weightage with risk and data coverage awareness
            weight_factors: List[float] = []
            for f in filtered:
                supplier_id = f.supplier_id
                supplier_name = f.supplier_name or None
                sid_token = str(supplier_id).strip() if supplier_id is not None else ""
                if not supplier_name and sid_token and sid_token in self._supplier_lookup:
                    supplier_name = self._supplier_lookup.get(sid_token)
                risk_raw = self._supplier_risk_map.get(sid_token)
                risk_score = self._normalise_risk_score(risk_raw)
                coverage = self._supplier_flow_coverage(supplier_id, supplier_name)
                base_impact = max(f.financial_impact_gbp, 0.0)
                factor = base_impact * (1.0 + risk_score) * (1.0 + coverage)
                if factor <= 0.0 and base_impact > 0.0:
                    factor = base_impact
                weight_factors.append(factor)
                details = f.calculation_details if isinstance(f.calculation_details, dict) else {}
                details.setdefault("risk_score_normalised", risk_score)
                details.setdefault("flow_coverage", coverage)
                f.calculation_details = details

            total_factor = sum(weight_factors)
            for f, factor in zip(filtered, weight_factors):
                f.weightage = (factor / total_factor) if total_factor > 0 else 0.0

            policy_top_summary = self._summarise_top_opportunities(
                per_policy_retained, per_policy_categories, limit=2
            )

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
                "policy_supplier_gaps": policy_supplier_gaps,
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

            data["notifications"] = sorted(notifications)
            logger.info(
                "OpportunityMinerAgent produced %d findings and %d candidate suppliers",
                len(filtered),
                len(supplier_candidates),
            )
            if getattr(self.settings, "verbose_agent_debug", False):
                logger.debug("OpportunityMinerAgent findings: %s", data["findings"])
            logger.info("OpportunityMinerAgent finishing processing")
            workflow_completed = True

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
            # Drop rows that are entirely null and normalise numeric types.
            cleaned = df.dropna(how="all")
            cleaned = self._normalise_numeric_dataframe(cleaned)
            tables[name] = cleaned
            logger.debug("Table %s columns: %s", name, list(cleaned.columns))
        return tables

    def _normalise_numeric_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with Decimal values coerced to native floats."""

        if df.empty:
            return df

        def _convert_decimal(value: Any) -> Any:
            if isinstance(value, Decimal):
                return float(value)
            return value

        for column in df.columns:
            series = df[column]
            if not pd.api.types.is_object_dtype(series):
                continue
            if series.map(lambda value: isinstance(value, Decimal)).any():
                df[column] = series.apply(_convert_decimal)
                series = df[column]
            non_null = series.dropna()
            if non_null.empty:
                continue
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in non_null
            ):
                df[column] = pd.to_numeric(series, errors="coerce")
        return df

    def _normalise_numeric_value(self, value: Any) -> Any:
        """Coerce Decimal values inside nested structures to native floats."""

        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {k: self._normalise_numeric_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalise_numeric_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._normalise_numeric_value(v) for v in value)
        return value

    def _next_opportunity_id(self) -> str:
        """Return the next sequential opportunity identifier as a string."""

        self._opportunity_sequence += 1
        return str(self._opportunity_sequence)

    def _normalise_supplier_key(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        text = str(value).strip().lower()
        if not text:
            return None
        return re.sub(r"[^a-z0-9]", "", text)

    def _normalise_identifier(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            return None
        return text.upper()

    def _normalise_item_description(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            return None
        return text.lower()


    def _build_supplier_lookup(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Build helper maps to resolve supplier metadata from ``proc.supplier``."""

        supplier_master = tables.get("supplier_master", pd.DataFrame())
        lookup: Dict[str, Optional[str]] = {}
        alias_map: Dict[str, set[str]] = {}
        reference_lookup: Dict[str, Dict[str, Any]] = {}

        def _normalise_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except Exception:
                pass
            text = str(value).strip()
            return text or None

        def _normalise_id(value: Any) -> Optional[str]:
            return self._normalise_identifier(value)

        supplier_id_col = self._find_column_for_key(supplier_master, "supplier_id")
        if not supplier_master.empty and supplier_id_col:
            df = supplier_master.dropna(subset=[supplier_id_col]).copy()
            if not df.empty:
                df[supplier_id_col] = df[supplier_id_col].map(_normalise_id)
                name_col = self._find_column_for_key(df, "supplier_name")
                trading_col = "trading_name" if "trading_name" in df.columns else None
                if name_col and name_col in df.columns:
                    df[name_col] = df[name_col].map(_normalise_text)
                if trading_col and trading_col in df.columns:
                    df[trading_col] = df[trading_col].map(_normalise_text)
                df = df.dropna(subset=[supplier_id_col])
                for _, row in df.iterrows():
                    supplier_id = row.get(supplier_id_col)
                    if not supplier_id:
                        continue
                    lookup[supplier_id] = row.get(name_col) or row.get(trading_col) or None
                    aliases = {supplier_id}
                    if name_col and row.get(name_col):
                        aliases.add(row.get(name_col))
                    if trading_col and row.get(trading_col):
                        aliases.add(row.get(trading_col))
                    reference_lookup[str(supplier_id)] = {
                        "supplier_id": str(supplier_id),
                        "supplier_name": row.get(name_col) or row.get(trading_col) or None,
                        "aliases": sorted(
                            {
                                str(alias)
                                for alias in aliases
                                if alias is not None and str(alias).strip()
                            }
                        ),
                    }
                    for alias in list(aliases):
                        normalised_alias = self._normalise_supplier_key(alias)
                        if normalised_alias:
                            alias_map.setdefault(normalised_alias, set()).add(supplier_id)

        self._supplier_lookup = lookup
        self._supplier_alias_lookup = {
            key: sorted(values)
            for key, values in alias_map.items()
            if values
        }
        self._supplier_reference_lookup = reference_lookup
        logger.debug(
            "Loaded %d suppliers from master data with %d alias keys",
            len(self._supplier_lookup),
            len(self._supplier_alias_lookup),
        )

        contracts = tables.get("contracts", pd.DataFrame())
        contract_map: Dict[str, str] = {}
        contract_meta: Dict[str, Dict[str, Any]] = {}
        contract_id_col = "contract_id" if "contract_id" in contracts.columns else None
        if not contracts.empty and contract_id_col:
            df = contracts.dropna(subset=[contract_id_col]).copy()
            df[contract_id_col] = df[contract_id_col].map(_normalise_id)
            supplier_col = self._find_column_for_key(df, "supplier_id")
            if supplier_col and supplier_col in df.columns:
                df[supplier_col] = df[supplier_col].map(_normalise_id)
            spend_col = "spend_category" if "spend_category" in df.columns else None
            if spend_col and spend_col in df.columns:
                df[spend_col] = df[spend_col].map(_normalise_text)
            category_col = "category_id" if "category_id" in df.columns else None
            if category_col and category_col in df.columns:
                df[category_col] = df[category_col].map(_normalise_text)
            df = df.dropna(subset=[contract_id_col])
            for _, row in df.iterrows():
                contract_id = row.get(contract_id_col)
                if not contract_id:
                    continue
                supplier_id = row.get(supplier_col) if supplier_col else None
                supplier_id = _normalise_id(supplier_id)
                spend_category = None
                if spend_col:
                    spend_category = row.get(spend_col)
                if not spend_category and category_col:
                    spend_category = row.get(category_col)
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
            df["po_id"] = df["po_id"].map(_normalise_id)
            supplier_id_col = self._find_column_for_key(df, "supplier_id")
            supplier_name_col = self._find_column_for_key(df, "supplier_name")
            if supplier_id_col and supplier_id_col in df.columns:
                df[supplier_id_col] = df[supplier_id_col].map(_normalise_id)
            if "contract_id" in df.columns:
                df["contract_id"] = df["contract_id"].map(_normalise_id)
            df = df.dropna(subset=["po_id"])
            for _, row in df.iterrows():
                po_id = row.get("po_id")
                if not po_id:
                    continue
                supplier_candidate = None
                if supplier_id_col and supplier_id_col in df.columns:
                    supplier_candidate = row.get(supplier_id_col)
                    supplier_candidate = _normalise_id(supplier_candidate)
                if not supplier_candidate and supplier_name_col and supplier_name_col in df.columns:
                    supplier_candidate = row.get(supplier_name_col)
                supplier_id = self._resolve_supplier_id(supplier_candidate)
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
            df["invoice_id"] = df["invoice_id"].map(_normalise_id)
            supplier_id_col = self._find_column_for_key(df, "supplier_id")
            supplier_name_col = self._find_column_for_key(df, "supplier_name")
            if supplier_id_col and supplier_id_col in df.columns:
                df[supplier_id_col] = df[supplier_id_col].map(_normalise_id)
            if "po_id" in df.columns:
                df["po_id"] = df["po_id"].map(_normalise_id)
            df = df.dropna(subset=["invoice_id"])
            for _, row in df.iterrows():
                invoice_id = row.get("invoice_id")
                if not invoice_id:
                    continue
                po_id = row.get("po_id")
                supplier_candidate = None
                if supplier_id_col and supplier_id_col in df.columns:
                    supplier_candidate = row.get(supplier_id_col)
                    supplier_candidate = _normalise_id(supplier_candidate)
                if not supplier_candidate and supplier_name_col and supplier_name_col in df.columns:
                    supplier_candidate = row.get(supplier_name_col)
                supplier_id = self._resolve_supplier_id(supplier_candidate)
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

        item_spend_map: Dict[str, Dict[str, float]] = {}
        item_count_map: Dict[str, Dict[str, int]] = {}
        item_desc_spend_map: Dict[str, Dict[str, float]] = {}
        item_desc_count_map: Dict[str, Dict[str, int]] = {}

        def _record_item_reference(
            supplier_id: Optional[str],
            item_key: Optional[str],
            desc_key: Optional[str],
            amount: Any,
        ) -> None:
            if not supplier_id:
                return
            if not item_key and not desc_key:
                return
            amount_value = self._to_float(amount, 0.0)
            if amount_value <= 0:
                amount_value = 1.0
            if item_key:
                spend_bucket = item_spend_map.setdefault(item_key, {})
                spend_bucket[supplier_id] = spend_bucket.get(supplier_id, 0.0) + amount_value
                count_bucket = item_count_map.setdefault(item_key, {})
                count_bucket[supplier_id] = count_bucket.get(supplier_id, 0) + 1
            if desc_key:
                spend_bucket = item_desc_spend_map.setdefault(desc_key, {})
                spend_bucket[supplier_id] = spend_bucket.get(supplier_id, 0.0) + amount_value
                count_bucket = item_desc_count_map.setdefault(desc_key, {})
                count_bucket[supplier_id] = count_bucket.get(supplier_id, 0) + 1

        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if not po_lines.empty and "po_id" in po_lines.columns:
            df = po_lines.dropna(subset=["po_id"]).copy()
            df["po_id"] = df["po_id"].map(self._normalise_identifier)
            df = df.dropna(subset=["po_id"])
            df["supplier_id"] = df["po_id"].map(self._po_supplier_map)
            supplier_columns: List[str] = []
            for candidate in (
                self._find_column_for_key(df, "supplier_id"),
                self._find_column_for_key(df, "supplier_name"),
            ):
                if candidate and candidate in df.columns and candidate not in supplier_columns:
                    supplier_columns.append(candidate)
            fallback_suppliers = None
            for column in supplier_columns:
                resolved = df[column].map(self._resolve_supplier_id)
                if fallback_suppliers is None:
                    fallback_suppliers = resolved
                else:
                    fallback_suppliers = fallback_suppliers.combine_first(resolved)
            if fallback_suppliers is not None:
                mask = df["supplier_id"].isna()
                if mask.any():
                    if df["supplier_id"].dtype != object:
                        df["supplier_id"] = df["supplier_id"].astype("object")
                    df.loc[mask, "supplier_id"] = fallback_suppliers[mask]
            df["supplier_id"] = df["supplier_id"].map(self._resolve_supplier_id)
            df = df.dropna(subset=["supplier_id"])
            if not df.empty:
                value_col = self._choose_first_column(df, _PURCHASE_LINE_VALUE_COLUMNS)
                if value_col and value_col in df.columns:
                    df["amount_value"] = pd.to_numeric(
                        df[value_col], errors="coerce"
                    ).fillna(0.0)
                else:
                    df["amount_value"] = 1.0
                if "item_id" in df.columns:
                    df["item_key_norm"] = df["item_id"].map(self._normalise_identifier)
                else:
                    df["item_key_norm"] = None
                if "item_description" in df.columns:
                    df["desc_key_norm"] = df["item_description"].map(
                        self._normalise_item_description
                    )
                else:
                    df["desc_key_norm"] = None
                for row in df.itertuples(index=False):
                    _record_item_reference(
                        getattr(row, "supplier_id", None),
                        getattr(row, "item_key_norm", None),
                        getattr(row, "desc_key_norm", None),
                        getattr(row, "amount_value", 0.0),
                    )

        invoice_lines = tables.get("invoice_lines", pd.DataFrame())
        if not invoice_lines.empty and "invoice_id" in invoice_lines.columns:
            df = invoice_lines.dropna(subset=["invoice_id"]).copy()
            df["invoice_id"] = df["invoice_id"].map(self._normalise_identifier)
            df = df.dropna(subset=["invoice_id"])
            df["supplier_id"] = df["invoice_id"].map(self._invoice_supplier_map)
            if "po_id" in df.columns:
                df["po_id_norm"] = df["po_id"].map(self._normalise_identifier)
                df.loc[df["supplier_id"].isna(), "supplier_id"] = df.loc[
                    df["supplier_id"].isna(), "po_id_norm"
                ].map(self._po_supplier_map)
            supplier_columns: List[str] = []
            for candidate in (
                self._find_column_for_key(df, "supplier_id"),
                self._find_column_for_key(df, "supplier_name"),
            ):
                if candidate and candidate in df.columns and candidate not in supplier_columns:
                    supplier_columns.append(candidate)
            fallback_suppliers = None
            for column in supplier_columns:
                resolved = df[column].map(self._resolve_supplier_id)
                if fallback_suppliers is None:
                    fallback_suppliers = resolved
                else:
                    fallback_suppliers = fallback_suppliers.combine_first(resolved)
            if fallback_suppliers is not None:
                mask = df["supplier_id"].isna()
                if mask.any():
                    if df["supplier_id"].dtype != object:
                        df["supplier_id"] = df["supplier_id"].astype("object")
                    df.loc[mask, "supplier_id"] = fallback_suppliers[mask]
            df["supplier_id"] = df["supplier_id"].map(self._resolve_supplier_id)
            df = df.dropna(subset=["supplier_id"])
            if not df.empty:
                value_col = self._choose_first_column(df, _INVOICE_LINE_VALUE_COLUMNS)
                if value_col and value_col in df.columns:
                    df["amount_value"] = pd.to_numeric(
                        df[value_col], errors="coerce"
                    ).fillna(0.0)
                else:
                    df["amount_value"] = 1.0
                if "item_id" in df.columns:
                    df["item_key_norm"] = df["item_id"].map(self._normalise_identifier)
                else:
                    df["item_key_norm"] = None
                if "item_description" in df.columns:
                    df["desc_key_norm"] = df["item_description"].map(
                        self._normalise_item_description
                    )
                else:
                    df["desc_key_norm"] = None
                for row in df.itertuples(index=False):
                    _record_item_reference(
                        getattr(row, "supplier_id", None),
                        getattr(row, "item_key_norm", None),
                        getattr(row, "desc_key_norm", None),
                        getattr(row, "amount_value", 0.0),
                    )

        self._item_supplier_map = {key: dict(values) for key, values in item_spend_map.items()}
        self._item_supplier_frequency = {
            key: dict(values) for key, values in item_count_map.items()
        }
        self._item_description_supplier_map = {
            key: dict(values) for key, values in item_desc_spend_map.items()
        }
        self._item_description_supplier_frequency = {
            key: dict(values) for key, values in item_desc_count_map.items()
        }

        logger.debug(
            "Derived supplier lookups: %d POs, %d invoices, %d contracts",
            len(self._po_supplier_map),
            len(self._invoice_supplier_map),
            len(self._contract_supplier_map),
        )
        logger.debug(
            "Indexed supplier references for %d item IDs and %d item descriptions",
            len(self._item_supplier_map),
            len(self._item_description_supplier_map),
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

    def _build_data_profile(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "tables": {},
        }
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                continue
            profile["tables"][name] = {
                "row_count": int(df.shape[0]) if df is not None else 0,
                "column_count": int(df.shape[1]) if df is not None else 0,
                "columns": list(df.columns) if not df.empty else [],
            }
        return profile

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

            provided_identifier: Optional[str] = None
            raw_identifier = provided_policy.get("policyId") or provided_policy.get("policy_id")
            if raw_identifier not in (None, "", [], {}):
                provided_identifier = str(raw_identifier)
            if (not provided_identifier or provided_identifier.isdigit()) and provided_policy.get(
                "policyName"
            ):
                provided_identifier = str(provided_policy["policyName"])
            if (not provided_identifier or provided_identifier.isdigit()) and provided_policy.get(
                "policy_name"
            ):
                provided_identifier = str(provided_policy["policy_name"])
            if provided_identifier:
                entry["policy_id"] = provided_identifier

            provided_name = (
                provided_policy.get("policyName")
                or provided_policy.get("policy_name")
                or provided_policy.get("policy_desc")
                or provided_policy.get("description")
            )
            if provided_name:
                entry["policy_name"] = str(provided_name)
                _add_alias(provided_name)

            if provided_identifier:
                _add_alias(provided_identifier)

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
                # capture suffix portions to allow matching enriched policy slugs
                parts = slug.split("_")
                if parts and parts[0].startswith("oppfinderpolicy"):
                    parts = parts[2:]
                while len(parts) > 1:
                    candidate = "_".join(parts)
                    if len(candidate) >= 4:
                        tokens.add(candidate)
                    parts = parts[:-1]
        return tokens

    def _get_policy_engine_catalog(self) -> Dict[str, Dict[str, Any]]:
        if self._policy_engine_catalog is not None:
            return self._policy_engine_catalog

        catalog: Dict[str, Dict[str, Any]] = {}
        engine = getattr(self.agent_nick, "policy_engine", None)
        if engine is None:
            self._policy_engine_catalog = catalog
            return catalog

        try:
            if hasattr(engine, "iter_policies"):
                policies = list(engine.iter_policies())
            elif hasattr(engine, "list_policies"):
                policies = list(engine.list_policies())  # pragma: no cover - legacy
            else:
                policies = []
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load policies from policy engine")
            policies = []

        for policy in policies:
            if not isinstance(policy, dict):
                continue
            tokens = self._collect_policy_tokens(policy)
            slug = self._normalise_policy_slug(policy.get("slug"))
            if slug:
                tokens.add(slug)
            aliases = policy.get("aliases")
            if isinstance(aliases, (list, tuple, set)):
                for alias in aliases:
                    alias_slug = self._normalise_policy_slug(alias)
                    if alias_slug:
                        tokens.add(alias_slug)
            for token in tokens:
                if not token:
                    continue
                catalog.setdefault(str(token).lower(), policy)

        self._policy_engine_catalog = catalog
        return catalog

    def _enrich_provided_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(policy, dict):
            return policy

        enriched = dict(policy)
        try:
            existing_rules = self._coerce_policy_rules(policy)
        except Exception:  # pragma: no cover - defensive
            existing_rules = {}
        catalog = self._get_policy_engine_catalog()

        tokens = self._collect_policy_tokens(enriched)
        slug_hint = self._normalise_policy_slug(
            enriched.get("slug")
            or enriched.get("policyName")
            or enriched.get("policy_name")
            or enriched.get("policy_desc")
            or enriched.get("description")
        )
        if slug_hint:
            tokens.add(slug_hint)

        candidate = None
        for token in tokens:
            if token is None:
                continue
            probe = catalog.get(str(token).lower())
            if probe:
                candidate = probe
                break

        if candidate is None:
            return enriched

        def _assign(field: str, value: Any) -> None:
            if field not in enriched or enriched[field] in (None, "", [], {}):
                enriched[field] = value

        candidate_id = candidate.get("policyId") or candidate.get("policy_id")
        if candidate_id is not None:
            candidate_id = str(candidate_id)
            _assign("policyId", candidate_id)
            _assign("policy_id", candidate_id)

        for name in (
            candidate.get("policyName"),
            candidate.get("policy_name"),
            candidate.get("detector"),
        ):
            if name:
                _assign("policyName", name)
                _assign("policy_name", name)
                break

        description = candidate.get("policy_desc") or candidate.get("description")
        if description:
            _assign("policy_desc", description)
            _assign("description", description)

        details = candidate.get("details")
        if isinstance(details, dict):
            if not isinstance(enriched.get("details"), dict):
                enriched["details"] = dict(details)
            if not enriched.get("policy_details"):
                enriched["policy_details"] = dict(details)
            rules = details.get("rules")
            if isinstance(rules, dict) and not isinstance(enriched.get("rules"), dict):
                enriched["rules"] = dict(rules)

        slug = candidate.get("slug") or self._normalise_policy_slug(
            candidate.get("policy_name") or candidate.get("policyName")
        )
        if slug:
            _assign("slug", slug)

        alias_tokens: set[str] = set()
        existing_aliases = enriched.get("aliases")
        if isinstance(existing_aliases, (list, tuple, set)):
            for alias in existing_aliases:
                normalised = self._normalise_policy_slug(alias)
                if normalised:
                    alias_tokens.add(normalised)
        candidate_aliases = candidate.get("aliases")
        if isinstance(candidate_aliases, (list, tuple, set)):
            for alias in candidate_aliases:
                normalised = self._normalise_policy_slug(alias)
                if normalised:
                    alias_tokens.add(normalised)
        if slug:
            normalised_slug = self._normalise_policy_slug(slug)
            if normalised_slug:
                alias_tokens.add(normalised_slug)
        if alias_tokens:
            enriched["aliases"] = sorted(alias_tokens)

        def _merge_rules(target: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
            base: Dict[str, Any] = {}
            if isinstance(target, dict):
                base = dict(target)
            for key, value in overrides.items():
                if key == "parameters" and isinstance(value, dict):
                    existing = base.get("parameters")
                    if isinstance(existing, dict):
                        merged = dict(existing)
                        merged.update(value)
                        base["parameters"] = merged
                    else:
                        base["parameters"] = dict(value)
                elif key == "default_conditions" and isinstance(value, dict):
                    existing = base.get("default_conditions")
                    if isinstance(existing, dict):
                        merged = dict(existing)
                        merged.update(value)
                        base["default_conditions"] = merged
                    else:
                        base["default_conditions"] = dict(value)
                else:
                    base[key] = value
            return base

        if isinstance(existing_rules, dict) and existing_rules:
            overrides = existing_rules
            for container_key in ("policy_details", "details"):
                container = enriched.get(container_key)
                if isinstance(container, dict):
                    container["rules"] = _merge_rules(
                        container.get("rules"), overrides
                    )
            enriched["rules"] = _merge_rules(enriched.get("rules"), overrides)

        return enriched

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
            candidate_slug = slug_hint
            if candidate_slug:
                candidate_slug = candidate_slug.strip().lower()
            if candidate_slug:
                for key, cfg in registry.items():
                    base_probe = self._decorate_policy_entry(
                        key, cfg, slug_hint=slug_hint
                    )
                    entry_slug = str(base_probe.get("policy_slug") or "").strip().lower()
                    alias_tokens = {
                        alias
                        for alias in (base_probe.get("aliases") or set())
                        if alias
                    }
                    if entry_slug == candidate_slug or candidate_slug in alias_tokens:
                        matched_entry = self._decorate_policy_entry(
                            key, cfg, provided_policy=policy, slug_hint=slug_hint
                        )
                        break
            for key, cfg in registry.items():
                if matched_entry:
                    break
                probe = self._decorate_policy_entry(key, cfg, slug_hint=slug_hint)
                aliases = probe.get("aliases", set())
                alias_tokens = {alias for alias in aliases if alias}
                if tokens & alias_tokens:
                    matched = True
                else:
                    matched = False
                    for token in tokens:
                        if not token:
                            continue
                        for alias in alias_tokens:
                            if not alias:
                                continue
                            if alias in token or token in alias:
                                matched = True
                                break
                        if matched:
                            break
                if matched:
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
        provided_policies: List[Dict[str, Any]] = []
        policies_payload = input_data.get("policies")
        if isinstance(policies_payload, list):
            for policy in policies_payload:
                if isinstance(policy, dict):
                    provided_policies.append(self._enrich_provided_policy(policy))
        if provided_policies:
            input_data["policies"] = provided_policies
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
                "supplier_autodetect": bool(
                    getattr(handler, "supports_supplier_autodetect", False)
                ),
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
        conditions = input_data.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
            input_data["conditions"] = conditions

        value = self._resolve_condition_value(conditions, key)
        if self._is_condition_value(value):
            return value

        for source in (
            input_data,
            input_data.get("parameters"),
            input_data.get("defaults"),
        ):
            candidate = self._resolve_condition_value(source, key)
            if self._is_condition_value(candidate):
                self._assign_condition(conditions, key, candidate)
                return candidate

        return default

    def _resolve_supplier_id(
        self, supplier_id: Optional[Any], *, event: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if event is not None:
            resolved_from_event = self._resolve_supplier_from_event(event, supplier_id)
            if resolved_from_event:
                return resolved_from_event
        if supplier_id is None:
            return None
        raw_text = str(supplier_id).strip()
        if not raw_text:
            return None
        supplier = self._normalise_identifier(raw_text) or raw_text
        lookup = self._supplier_lookup
        if not lookup:
            return supplier
        if supplier in lookup:
            return supplier

        alias_key = self._normalise_supplier_key(raw_text)
        alias_lookup = self._supplier_alias_lookup

        def _choose_candidate(candidates: Iterable[str]) -> Optional[str]:
            chosen: Optional[str] = None
            best_score = 0.0
            raw_normalised = raw_text.lower()
            for candidate in candidates:
                if candidate is None:
                    continue
                candidate_texts = [candidate.lower()]
                display = lookup.get(candidate)
                if display:
                    candidate_texts.append(str(display).strip().lower())
                for text in candidate_texts:
                    if not text:
                        continue
                    score = SequenceMatcher(None, raw_normalised, text).ratio()
                    if score > best_score:
                        best_score = score
                        chosen = candidate
            if chosen is not None:
                return chosen
            candidate_list = [c for c in candidates if c is not None]
            return sorted(candidate_list)[0] if candidate_list else None

        candidates: Optional[Iterable[str]] = None
        if alias_key and alias_lookup:
            candidates = alias_lookup.get(alias_key)

        if not candidates and alias_key and alias_lookup:
            best_key = None
            best_ratio = 0.0
            for key, ids in alias_lookup.items():
                if not key or not ids:
                    continue
                ratio = SequenceMatcher(None, alias_key, key).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key
            if best_key and best_ratio >= 0.85:
                candidates = alias_lookup.get(best_key)

        if candidates:
            resolved = _choose_candidate(candidates)
            if resolved:
                return resolved

        logger.debug("Supplier %s not found in master data; skipping", supplier)
        return None

    def _resolve_supplier_from_event(
        self, payload: Dict[str, Any], supplier_value: Optional[Any] = None
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        conditions = payload.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
            payload["conditions"] = conditions

        def _commit(resolved: Optional[str]) -> Optional[str]:
            if not resolved:
                return None
            self._assign_condition(conditions, "supplier_id", resolved, override=True)
            payload["supplier_id"] = resolved
            return resolved

        if self._is_condition_value(supplier_value):
            direct = self._resolve_supplier_id(supplier_value)
            if direct:
                return _commit(direct)

        candidate = self._resolve_condition_value(conditions, "supplier_id")
        if self._is_condition_value(candidate):
            resolved = self._resolve_supplier_id(candidate)
            if resolved:
                return _commit(resolved)

        candidate = self._resolve_condition_value(payload, "supplier_id")
        if self._is_condition_value(candidate):
            resolved = self._resolve_supplier_id(candidate)
            if resolved:
                return _commit(resolved)

        rfq_candidate = self._resolve_condition_value(conditions, "rfq_id") or payload.get(
            "rfq_id"
        )
        rfq_id = self._normalise_identifier(rfq_candidate)
        if rfq_id:
            resolved = self._lookup_supplier_from_drafts(rfq_id)
            if resolved:
                return _commit(resolved)
            resolved = self._lookup_supplier_from_processed(rfq_id)
            if resolved:
                return _commit(resolved)

        for key in ("from_address", "supplier_email", "email", "contact_email"):
            candidate = self._resolve_condition_value(conditions, key)
            if not self._is_condition_value(candidate):
                candidate = payload.get(key)
            if self._is_condition_value(candidate):
                resolved = self._resolve_supplier_id(candidate)
                if resolved:
                    return _commit(resolved)

        policies = payload.get("policies")
        if isinstance(policies, (list, tuple)):
            for policy in policies:
                if not isinstance(policy, dict):
                    continue
                candidate = self._resolve_condition_value(policy, "supplier_id")
                if not self._is_condition_value(candidate):
                    candidate = self._resolve_condition_value(
                        policy.get("conditions"), "supplier_id"
                    )
                if self._is_condition_value(candidate):
                    resolved = self._resolve_supplier_id(candidate)
                    if resolved:
                        return _commit(resolved)

        return None

    def _lookup_supplier_from_drafts(self, rfq_id: str) -> Optional[str]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT supplier_id
                        FROM proc.draft_rfq_emails
                        WHERE rfq_id = %s
                        ORDER BY created_on DESC
                        LIMIT 1
                        """,
                        (rfq_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        resolved = self._resolve_supplier_id(row[0])
                        if resolved:
                            return resolved
        except Exception:
            logger.debug(
                "Supplier draft lookup failed for RFQ %s", rfq_id, exc_info=True
            )
        return None

    def _lookup_supplier_from_processed(self, rfq_id: str) -> Optional[str]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT key
                        FROM proc.processed_emails
                        WHERE rfq_id = %s
                        ORDER BY processed_at DESC
                        LIMIT 1
                        """,
                        (rfq_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    key = row[0]

                with conn.cursor() as cur:
                    try:
                        cur.execute(
                            """
                            SELECT supplier_id
                            FROM proc.email_thread_map
                            WHERE rfq_id = %s
                            ORDER BY updated_at DESC
                            LIMIT 1
                            """,
                            (rfq_id,),
                        )
                        thread_row = cur.fetchone()
                    except Exception:
                        conn.rollback()
                        thread_row = None

                    if thread_row and thread_row[0]:
                        resolved = self._resolve_supplier_id(thread_row[0])
                        if resolved:
                            return resolved

                    try:
                        cur.execute(
                            """
                            SELECT supplier_id, from_address
                            FROM proc.supplier_replies
                            WHERE rfq_id = %s AND (%s IS NULL OR s3_key = %s)
                            ORDER BY COALESCE(received_at, updated_at) DESC
                            LIMIT 1
                            """,
                            (rfq_id, key, key),
                        )
                        reply_row = cur.fetchone()
                    except Exception:
                        conn.rollback()
                        reply_row = None

                if reply_row:
                    supplier_candidate, from_address = reply_row
                    resolved = self._resolve_supplier_id(supplier_candidate)
                    if resolved:
                        return resolved
                    if from_address:
                        resolved = self._resolve_supplier_id(from_address)
                        if resolved:
                            return resolved
        except Exception:
            logger.debug(
                "Supplier lookup via processed emails failed for RFQ %s",
                rfq_id,
                exc_info=True,
            )
        return None

    def _select_top_supplier(
        self,
        spend_by_supplier: Optional[Dict[str, Any]],
        frequency_by_supplier: Optional[Dict[str, int]] = None,
    ) -> Optional[str]:
        if not spend_by_supplier:
            return None

        best_supplier: Optional[str] = None
        best_spend = float("-inf")
        best_count = -1

        for supplier, raw_amount in spend_by_supplier.items():
            resolved = self._resolve_supplier_id(supplier)
            if not resolved:
                continue
            amount = self._to_float(raw_amount, 0.0)
            count = int(frequency_by_supplier.get(supplier, 0)) if frequency_by_supplier else 0
            if (
                best_supplier is None
                or amount > best_spend
                or (amount == best_spend and count > best_count)
                or (
                    amount == best_spend
                    and count == best_count
                    and resolved < best_supplier
                )
            ):
                best_supplier = resolved
                best_spend = amount
                best_count = count

        return best_supplier

    def _resolve_policy_supplier(
        self, input_data: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        metadata: Dict[str, Any] = {"attempted_sources": {}}
        conditions = input_data.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
            input_data["conditions"] = conditions

        message: Optional[str] = None
        error: Optional[str] = None

        raw_supplier = self._get_condition(input_data, "supplier_id")
        if self._is_condition_value(raw_supplier):
            metadata["provided_identifier"] = str(raw_supplier).strip()
            resolved = self._resolve_supplier_id(raw_supplier)
            if resolved:
                metadata["resolved_source"] = "supplier_id"
                metadata["supplier_id"] = resolved
                self._assign_condition(conditions, "supplier_id", resolved, override=True)
                return resolved, metadata
            message = "Supplier not recognised in master data"
            error = "unmatched_supplier_identifier"

        raw_name = self._get_condition(input_data, "supplier_name")
        if self._is_condition_value(raw_name):
            metadata["provided_supplier_name"] = str(raw_name).strip()
            resolved = self._resolve_supplier_id(raw_name)
            if resolved:
                metadata["resolved_source"] = "supplier_name"
                metadata["supplier_id"] = resolved
                self._assign_condition(conditions, "supplier_id", resolved, override=True)
                return resolved, metadata
            if message is None:
                message = "Supplier not recognised in master data"
                error = "unmatched_supplier_identifier"

        contract_raw = self._get_condition(input_data, "contract_id")
        contract_id = self._normalise_identifier(contract_raw)
        if contract_id:
            metadata["attempted_sources"]["contract_id"] = contract_id
            contract_supplier_map = getattr(self, "_contract_supplier_map", {})
            supplier_candidate = contract_supplier_map.get(contract_id)
            if not supplier_candidate:
                contract_meta = getattr(self, "_contract_metadata", {})
                supplier_candidate = contract_meta.get(contract_id, {}).get("supplier_id")
            supplier_candidate = self._resolve_supplier_id(supplier_candidate)
            if supplier_candidate:
                metadata["resolved_source"] = "contract_id"
                metadata["contract_id"] = contract_id
                metadata["supplier_id"] = supplier_candidate
                self._assign_condition(conditions, "supplier_id", supplier_candidate, override=True)
                self._assign_condition(conditions, "contract_id", contract_id, override=True)
                return supplier_candidate, metadata

        po_raw = self._get_condition(input_data, "po_id")
        po_id = self._normalise_identifier(po_raw)
        if po_id:
            metadata["attempted_sources"]["po_id"] = po_id
            po_supplier_map = getattr(self, "_po_supplier_map", {})
            po_contract_map = getattr(self, "_po_contract_map", {})
            contract_supplier_map = getattr(self, "_contract_supplier_map", {})
            supplier_candidate = po_supplier_map.get(po_id)
            if not supplier_candidate:
                contract_id = po_contract_map.get(po_id)
                if contract_id:
                    metadata["attempted_sources"]["contract_id_from_po"] = contract_id
                    supplier_candidate = contract_supplier_map.get(contract_id)
                    if not supplier_candidate:
                        contract_meta = getattr(self, "_contract_metadata", {})
                        supplier_candidate = contract_meta.get(contract_id, {}).get("supplier_id")
            supplier_candidate = self._resolve_supplier_id(supplier_candidate)
            if supplier_candidate:
                metadata["resolved_source"] = "po_id"
                metadata["po_id"] = po_id
                metadata["supplier_id"] = supplier_candidate
                self._assign_condition(conditions, "supplier_id", supplier_candidate, override=True)
                self._assign_condition(conditions, "po_id", po_id, override=True)
                return supplier_candidate, metadata

        invoice_raw = self._get_condition(input_data, "invoice_id")
        invoice_id = self._normalise_identifier(invoice_raw)
        if invoice_id:
            metadata["attempted_sources"]["invoice_id"] = invoice_id
            invoice_supplier_map = getattr(self, "_invoice_supplier_map", {})
            invoice_po_map = getattr(self, "_invoice_po_map", {})
            po_supplier_map = getattr(self, "_po_supplier_map", {})
            po_contract_map = getattr(self, "_po_contract_map", {})
            contract_supplier_map = getattr(self, "_contract_supplier_map", {})
            supplier_candidate = invoice_supplier_map.get(invoice_id)
            if not supplier_candidate:
                po_id = invoice_po_map.get(invoice_id)
                if po_id:
                    metadata["attempted_sources"]["po_id_from_invoice"] = po_id
                    supplier_candidate = po_supplier_map.get(po_id)
                    if not supplier_candidate:
                        contract_id = po_contract_map.get(po_id)
                        if contract_id:
                            metadata["attempted_sources"]["contract_id_from_invoice"] = contract_id
                            supplier_candidate = contract_supplier_map.get(contract_id)
                            if not supplier_candidate:
                                contract_meta = getattr(self, "_contract_metadata", {})
                                supplier_candidate = contract_meta.get(contract_id, {}).get("supplier_id")
            supplier_candidate = self._resolve_supplier_id(supplier_candidate)
            if supplier_candidate:
                metadata["resolved_source"] = "invoice_id"
                metadata["invoice_id"] = invoice_id
                metadata["supplier_id"] = supplier_candidate
                self._assign_condition(conditions, "supplier_id", supplier_candidate, override=True)
                self._assign_condition(conditions, "invoice_id", invoice_id, override=True)
                return supplier_candidate, metadata

        item_candidate = self._get_condition(input_data, "item_id")
        if self._is_condition_value(item_candidate):
            item_value = str(item_candidate).strip()
            item_key = self._normalise_identifier(item_candidate)
            lookup = getattr(self, "_item_supplier_map", {}) or {}
            freq_lookup = getattr(self, "_item_supplier_frequency", {}) or {}
            spend_map = None
            count_map: Optional[Dict[str, int]] = None
            if item_key and item_key in lookup:
                spend_map = lookup.get(item_key)
                count_map = freq_lookup.get(item_key, {})
            elif item_value and item_value in lookup:
                spend_map = lookup.get(item_value)
                count_map = freq_lookup.get(item_value, {})
                item_key = item_value
            metadata["attempted_sources"]["item_id"] = item_key or item_value
            if spend_map:
                supplier_candidate = self._select_top_supplier(spend_map, count_map)
                if supplier_candidate:
                    metadata["resolved_source"] = "item_id"
                    metadata["supplier_id"] = supplier_candidate
                    if item_key:
                        metadata["item_id"] = item_key
                    if spend_map:
                        sorted_candidates = sorted(
                            spend_map.items(),
                            key=lambda kv: (
                                self._to_float(kv[1], 0.0),
                                (count_map or {}).get(kv[0], 0),
                                kv[0],
                            ),
                            reverse=True,
                        )
                        metadata["candidate_suppliers"] = [
                            candidate for candidate, _ in sorted_candidates[:5]
                        ]
                    self._assign_condition(
                        conditions, "supplier_id", supplier_candidate, override=True
                    )
                    return supplier_candidate, metadata

        description_candidate = self._get_condition(input_data, "item_description")
        if self._is_condition_value(description_candidate):
            desc_text = str(description_candidate).strip()
            desc_key = self._normalise_item_description(description_candidate)
            metadata["attempted_sources"]["item_description"] = desc_text
            desc_lookup = getattr(self, "_item_description_supplier_map", {}) or {}
            desc_freq_lookup = getattr(
                self, "_item_description_supplier_frequency", {}
            ) or {}
            spend_map = None
            count_map: Optional[Dict[str, int]] = None
            matched_key: Optional[str] = None
            if desc_key and desc_key in desc_lookup:
                spend_map = desc_lookup.get(desc_key)
                count_map = desc_freq_lookup.get(desc_key, {})
                matched_key = desc_key
            elif desc_lookup:
                comparison_key = desc_key or desc_text.lower()
                best_key = None
                best_score = 0.0
                for candidate_key in desc_lookup.keys():
                    if not candidate_key:
                        continue
                    score = SequenceMatcher(None, comparison_key, candidate_key).ratio()
                    if score > best_score:
                        best_score = score
                        best_key = candidate_key
                if best_key and best_score >= 0.8:
                    spend_map = desc_lookup.get(best_key)
                    count_map = desc_freq_lookup.get(best_key, {})
                    matched_key = best_key
                    metadata["item_description_match_score"] = best_score
            if spend_map:
                supplier_candidate = self._select_top_supplier(spend_map, count_map)
                if supplier_candidate:
                    metadata["resolved_source"] = "item_description"
                    metadata["supplier_id"] = supplier_candidate
                    if matched_key:
                        metadata["item_description"] = matched_key
                    sorted_candidates = sorted(
                        spend_map.items(),
                        key=lambda kv: (
                            self._to_float(kv[1], 0.0),
                            (count_map or {}).get(kv[0], 0),
                            kv[0],
                        ),
                        reverse=True,
                    )
                    metadata["candidate_suppliers"] = [
                        candidate for candidate, _ in sorted_candidates[:5]
                    ]
                    self._assign_condition(
                        conditions, "supplier_id", supplier_candidate, override=True
                    )
                    return supplier_candidate, metadata


        attempts = metadata.get("attempted_sources") or {}
        if not attempts:
            metadata.pop("attempted_sources", None)
        if message is None:
            if attempts:
                message = "Unable to resolve supplier from transactional context"
                error = "supplier_lookup_failed"
            else:
                message = "Supplier identifier missing from policy conditions"
                error = "missing_supplier_identifier"
        metadata["message"] = message
        if error:
            metadata["error"] = error
        return None, metadata

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
            *,
            supplier_autodetect: bool = False,
        ) -> Dict[str, Any]:
            aliases = {
                self._normalise_policy_slug(slug),
                self._normalise_policy_slug(detector),
            }
            autodetect_enabled = bool(
                supplier_autodetect
                or getattr(handler, "supports_supplier_autodetect", False)
            )
            return {
                "policy_slug": slug,
                "policy_id": slug,
                "detector": detector,
                "policy_name": detector,
                "required_fields": list(required),
                "handler": handler,
                "default_conditions": dict(default_conditions or {}),
                "supplier_autodetect": autodetect_enabled,
                "aliases": {alias for alias in aliases if alias},
            }

        registry: Dict[str, Dict[str, Any]] = {
            "price_variance_check": entry(
                "price_variance_check",
                "Price Benchmark Variance",
                self._policy_price_benchmark_variance,
                ["supplier_id", "item_id", "actual_price", "benchmark_price"],
                supplier_autodetect=True,
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
                supplier_autodetect=True,
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
        df["po_id"] = df["po_id"].map(self._normalise_identifier)
        df = df.dropna(subset=["po_id"])
        if item_id is not None and "item_id" in df.columns:
            target_item = str(item_id).strip()
            df = df[df["item_id"].astype(str).str.strip() == target_item]
        po_ids = df["po_id"].astype(str).unique().tolist()
        if supplier_id:
            resolved_supplier = self._resolve_supplier_id(supplier_id)
            po_ids = [
                po_id
                for po_id in po_ids
                if self._po_supplier_map.get(po_id) == resolved_supplier
            ]
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
        impact_value = self._to_float(impact)
        if pd.isna(impact_value):
            impact_value = 0.0

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
        normalised_sources: List[str] = []
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

        details = (
            self._normalise_numeric_value(details) if isinstance(details, dict) else {}
        )

        finding = Finding(
            opportunity_id=self._next_opportunity_id(),
            opportunity_ref_id=opportunity_id,
            detector_type=detector,
            supplier_id=supplier_id,
            category_id=category_id,
            item_id=item_id,
            financial_impact_gbp=impact_value,
            calculation_details=details,
            source_records=normalised_sources,
            detected_on=datetime.utcnow(),
            item_reference=item_id,
            policy_id=policy_identifier,
        )

        logger.debug(
            "Built finding %s (ref %s) for supplier %s, category %s, item %s with impact %s",
            finding.opportunity_id,
            finding.opportunity_ref_id,
            supplier_id,
            category_id,
            item_id,
            impact_value,
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
            _PURCHASE_LINE_VALUE_COLUMNS,
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
            _PURCHASE_LINE_VALUE_COLUMNS,
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

        supplier_id_col = self._find_column_for_key(purchase_orders, "supplier_id")
        supplier_name_col = self._find_column_for_key(purchase_orders, "supplier_name")
        supplier_col = None
        if supplier_id_col and supplier_id_col in purchase_orders.columns:
            supplier_col = supplier_id_col
        elif supplier_name_col and supplier_name_col in purchase_orders.columns:
            supplier_col = supplier_name_col
        if not supplier_col:
            return findings

        join_df = purchase_orders[["po_id", supplier_col]].copy()
        join_df = join_df.rename(columns={supplier_col: "supplier_reference"})

        try:
            merged = po_lines.merge(
                join_df,
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

        if "supplier_reference" not in merged.columns:
            return findings
        merged["supplier_reference"] = merged["supplier_reference"].map(_clean)
        merged["supplier_id"] = merged["supplier_reference"].map(self._resolve_supplier_id)
        unresolved_mask = merged["supplier_id"].isna() & merged["supplier_reference"].notna()
        if unresolved_mask.any():
            merged.loc[unresolved_mask, "supplier_id"] = merged.loc[
                unresolved_mask, "supplier_reference"
            ]
        merged["supplier_id"] = merged["supplier_id"].map(_clean)
        merged["po_id"] = merged["po_id"].map(_clean)
        if "item_id" in merged.columns:
            merged["item_id"] = merged["item_id"].map(_clean)
        merged["item_description"] = merged["item_description"].map(_clean)
        merged = merged.dropna(subset=["supplier_id", "item_description"])
        merged = merged.drop(columns=["supplier_reference"], errors="ignore")
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

    def _auto_detect_price_variance(
        self,
        tables: Dict[str, pd.DataFrame],
        policy_cfg: Dict[str, Any],
        threshold_pct: float,
        *,
        supplier_filter: Optional[str] = None,
        item_filter: Optional[Any] = None,
        quantity_override: Optional[float] = None,
    ) -> List[Finding]:
        po_lines = tables.get("purchase_order_lines", pd.DataFrame())
        if (
            po_lines.empty
            or "po_id" not in po_lines.columns
            or "item_id" not in po_lines.columns
        ):
            return []

        df = po_lines.dropna(subset=["po_id", "item_id"]).copy()
        if df.empty:
            return []

        df["po_id"] = df["po_id"].map(self._normalise_identifier)
        df["item_id"] = df["item_id"].astype(str).str.strip()
        df = df.dropna(subset=["po_id"])
        df = df[df["item_id"].astype(str).str.strip() != ""]
        df["supplier_id"] = df["po_id"].map(self._po_supplier_map)
        df["supplier_id"] = df["supplier_id"].map(self._resolve_supplier_id)
        df = df.dropna(subset=["supplier_id"])
        if df.empty:
            return []

        if supplier_filter:
            df = df[df["supplier_id"] == supplier_filter]
        if item_filter:
            norm_item = str(item_filter).strip()
            df = df[df["item_id"].str.casefold() == norm_item.casefold()]
        if df.empty:
            return []

        value_col = self._choose_first_column(df, _PURCHASE_LINE_VALUE_COLUMNS)
        price_col = self._choose_first_column(df, ["unit_price_gbp", "unit_price"])
        qty_col = "quantity" if "quantity" in df.columns else None

        if price_col:
            df["unit_price_calc"] = pd.to_numeric(
                df[price_col], errors="coerce"
            )
        else:
            df["unit_price_calc"] = pd.Series(
                [float("nan")] * len(df), index=df.index, dtype="float64"
            )
        if value_col:
            df["line_value_calc"] = pd.to_numeric(
                df[value_col], errors="coerce"
            )
        else:
            df["line_value_calc"] = pd.Series(
                [float("nan")] * len(df), index=df.index, dtype="float64"
            )
        if qty_col:
            df["quantity_calc"] = pd.to_numeric(df[qty_col], errors="coerce")
        else:
            df["quantity_calc"] = pd.Series(
                [float("nan")] * len(df), index=df.index, dtype="float64"
            )

        with_value = df["line_value_calc"].notna()
        with_quantity = df["quantity_calc"].notna()
        with_price = df["unit_price_calc"].notna()

        # derive missing quantities from value and price when possible
        fill_quantity = with_value & with_price & ~with_quantity
        if fill_quantity.any():
            denom = df.loc[fill_quantity, "unit_price_calc"].replace(0, pd.NA)
            df.loc[fill_quantity, "quantity_calc"] = (
                df.loc[fill_quantity, "line_value_calc"] / denom
            )

        # derive missing unit price from value and quantity
        with_value = df["line_value_calc"].notna()
        with_quantity = df["quantity_calc"].notna()
        fill_price = with_value & with_quantity & ~with_price
        if fill_price.any():
            denom = df.loc[fill_price, "quantity_calc"].replace(0, pd.NA)
            df.loc[fill_price, "unit_price_calc"] = (
                df.loc[fill_price, "line_value_calc"] / denom
            )

        # default quantity to 1 when still missing
        df.loc[df["quantity_calc"].isna(), "quantity_calc"] = 1.0

        df.loc[df["unit_price_calc"].isna(), "unit_price_calc"] = df.loc[
            df["unit_price_calc"].isna(), "line_value_calc"
        ]
        df.loc[df["line_value_calc"].isna(), "line_value_calc"] = (
            df.loc[df["line_value_calc"].isna(), "unit_price_calc"]
            * df.loc[df["line_value_calc"].isna(), "quantity_calc"]
        )

        df = df.replace([pd.NA, float("inf"), float("-inf")], float("nan"))
        df = df.dropna(subset=["unit_price_calc", "quantity_calc", "line_value_calc"])
        if df.empty:
            return []

        df = df[(df["unit_price_calc"] > 0) & (df["quantity_calc"] > 0)]
        if df.empty:
            return []

        grouped = (
            df.groupby(["item_id", "supplier_id"], as_index=False)
            .agg(
                total_qty=("quantity_calc", "sum"),
                avg_price=("unit_price_calc", "mean"),
                total_value=("line_value_calc", "sum"),
            )
            .dropna()
        )
        if grouped.empty:
            return []

        supplier_counts = (
            grouped.groupby("item_id")["supplier_id"].nunique().to_dict()
        )
        if not any(count >= 2 for count in supplier_counts.values()):
            return []

        benchmark_map = grouped.groupby("item_id")["avg_price"].min().to_dict()
        po_sources = (
            df.groupby(["item_id", "supplier_id"])["po_id"]
            .apply(
                lambda series: [
                    value
                    for value in dict.fromkeys(
                        [str(po).strip() for po in series if pd.notna(po)]
                    )
                    if value
                ]
            )
            .to_dict()
        )

        detector = policy_cfg.get("detector", "Price Benchmark Variance")
        policy_id = policy_cfg.get("policy_id")
        policy_name = policy_cfg.get("policy_name")

        findings: List[Finding] = []
        for row in grouped.itertuples(index=False):
            item_id = getattr(row, "item_id")
            supplier_id = getattr(row, "supplier_id")
            if supplier_counts.get(item_id, 0) < 2:
                continue
            benchmark_price = benchmark_map.get(item_id)
            if benchmark_price in (None, 0, 0.0):
                continue
            actual_price = float(getattr(row, "avg_price", 0.0) or 0.0)
            if actual_price <= benchmark_price:
                continue
            variance_pct = (actual_price - benchmark_price) / benchmark_price
            if variance_pct <= threshold_pct:
                continue
            total_qty = float(getattr(row, "total_qty", 0.0) or 0.0)
            if quantity_override and quantity_override > 0:
                total_qty = max(total_qty, float(quantity_override))
            if total_qty <= 0:
                total_qty = 1.0
            impact = (actual_price - benchmark_price) * total_qty
            po_ids = po_sources.get((item_id, supplier_id), [])
            invoice_ids = self._invoice_ids_for_po(tables, po_ids)
            sources = po_ids[:5] + invoice_ids[:5]
            if not sources and item_id is not None:
                sources = [str(item_id)]
            category_id = self._category_for_po(po_ids[0]) if po_ids else None
            details = {
                "actual_price": actual_price,
                "benchmark_price": benchmark_price,
                "variance_pct": variance_pct,
                "quantity": total_qty,
                "auto_detect": True,
            }
            finding = self._build_finding(
                detector,
                supplier_id,
                category_id,
                item_id,
                impact,
                details,
                sources,
                policy_id=policy_id,
                policy_name=policy_name,
            )
            findings.append(finding)

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

        threshold_pct = self._to_float(
            self._get_condition(input_data, "variance_threshold_pct", 0.0)
        )
        quantity_override = self._to_float(self._get_condition(input_data, "quantity"))

        supplier_filter = self._resolve_supplier_id(
            self._get_condition(input_data, "supplier_id")
        )
        item_filter = self._get_condition(input_data, "item_id")

        actual_price = self._to_float(self._get_condition(input_data, "actual_price"))
        benchmark_price = self._to_float(
            self._get_condition(input_data, "benchmark_price")
        )

        auto_mode = (
            supplier_filter is None
            or actual_price is None
            or benchmark_price is None
            or actual_price <= 0
            or benchmark_price <= 0
        )

        if auto_mode:
            auto_findings = self._auto_detect_price_variance(
                tables,
                policy_cfg,
                threshold_pct,
                supplier_filter=supplier_filter,
                item_filter=item_filter,
                quantity_override=quantity_override,
            )
            if auto_findings:
                for finding in auto_findings:
                    details = (
                        finding.calculation_details
                        if isinstance(finding.calculation_details, dict)
                        else {}
                    )
                    self._record_escalation(
                        policy_id,
                        finding.supplier_id,
                        "Price variance threshold breached",
                        details,
                    )
                self._default_notifications(notifications)
                return auto_findings

            message = "No suppliers exceeded the benchmark variance threshold"
            context: Dict[str, Any] = {
                "threshold_pct": threshold_pct,
                "auto_detect": True,
            }
            if supplier_filter:
                context["supplier_id"] = supplier_filter
            if item_filter:
                context["item_id"] = item_filter
            self._log_policy_event(policy_id, None, "no_action", message, context)
            return findings

        supplier_id = supplier_filter
        supplier_meta: Optional[Dict[str, Any]] = None
        if not supplier_id:
            supplier_id, supplier_meta = self._resolve_policy_supplier(input_data)
            if not supplier_id:
                message = supplier_meta.get("message") if supplier_meta else None
                if not message:
                    message = "Supplier identifier missing from policy conditions"
                self._log_policy_event(
                    policy_id,
                    None,
                    "blocked",
                    message,
                    supplier_meta if supplier_meta else {},
                )
                return findings

        item_id = item_filter or self._get_condition(input_data, "item_id")
        actual_price = actual_price or self._to_float(
            self._get_condition(input_data, "actual_price")
        )
        benchmark_price = benchmark_price or self._to_float(
            self._get_condition(input_data, "benchmark_price")
        )
        quantity = max(
            self._to_float(
                quantity_override if quantity_override is not None else 1.0, 1.0
            ),
            1.0,
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
            _PURCHASE_LINE_VALUE_COLUMNS,
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
        df["po_id"] = df["po_id"].map(self._normalise_identifier)
        df = df.dropna(subset=["po_id"])
        df["item_id"] = df["item_id"].astype(str).str.strip()
        df[value_col] = df[value_col].fillna(0.0)
        if price_col:
            df[price_col] = df[price_col].fillna(0.0)
        if qty_col:
            df[qty_col] = df[qty_col].fillna(0.0)
        supplier_col = self._find_column_for_key(df, "supplier_id")
        mapped_suppliers = df["po_id"].map(self._po_supplier_map)
        if supplier_col and supplier_col in df.columns:
            existing = df[supplier_col].apply(self._resolve_supplier_id)
            df["supplier_id"] = mapped_suppliers.where(~mapped_suppliers.isna(), existing)
        else:
            df["supplier_id"] = mapped_suppliers
        df["supplier_id"] = df["supplier_id"].map(self._resolve_supplier_id)
        df = df.dropna(subset=["supplier_id"])

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
            po_raw = row.get("po_id")
            po_id = self._normalise_identifier(po_raw)
            contract_raw = row.get("contract_id")
            contract_id = self._normalise_identifier(contract_raw)
            if contract_id and contract_id in self._contract_supplier_map:
                continue
            supplier_candidate = self._resolve_supplier_id(row.get("supplier_id"))
            if not supplier_candidate and po_id:
                supplier_candidate = self._po_supplier_map.get(po_id)
            if not supplier_candidate and contract_id:
                supplier_candidate = self._contract_supplier_map.get(contract_id)
            supplier_id = self._resolve_supplier_id(supplier_candidate)
            if not supplier_id:
                continue
            value = self._to_float(row.get(value_col, 0.0))
            if value < minimum_value:
                continue
            details = {"po_total": value}
            if contract_id or contract_raw:
                details["contract_id"] = contract_id or contract_raw
            findings.append(
                self._build_finding(
                    detector,
                    supplier_id,
                    None,
                    None,
                    value,
                    details,
                    [po_id or str(po_raw)],
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
            _PURCHASE_LINE_VALUE_COLUMNS,
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
            _INVOICE_LINE_VALUE_COLUMNS,
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
            _INVOICE_LINE_VALUE_COLUMNS,
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

    def _auto_supplier_performance_records(
        self,
        tables: Dict[str, pd.DataFrame],
        *,
        threshold: float,
        supplier_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        invoices = tables.get("invoices", pd.DataFrame())
        if invoices.empty or "invoice_id" not in invoices.columns:
            return []

        df = invoices.dropna(subset=["invoice_id"]).copy()
        if df.empty:
            return []

        df["invoice_id"] = df["invoice_id"].map(self._normalise_identifier)
        df = df.dropna(subset=["invoice_id"])
        if df.empty:
            return []

        supplier_id_col = self._find_column_for_key(df, "supplier_id")
        if supplier_id_col and supplier_id_col in df.columns:
            df[supplier_id_col] = df[supplier_id_col].map(self._resolve_supplier_id)
        supplier_name_col = self._find_column_for_key(df, "supplier_name")
        if supplier_name_col and supplier_name_col in df.columns:
            resolved_names = df[supplier_name_col].map(self._resolve_supplier_id)
            if supplier_id_col and supplier_id_col in df.columns:
                mask = df[supplier_id_col].isna()
                df.loc[mask, supplier_id_col] = resolved_names[mask]
            else:
                df["__supplier_name_resolved"] = resolved_names

        df["supplier_id_resolved"] = df.get(supplier_id_col) if supplier_id_col else None
        if "supplier_id_resolved" not in df.columns or df["supplier_id_resolved"].isna().all():
            df["supplier_id_resolved"] = df["invoice_id"].map(self._invoice_supplier_map)
        if df["supplier_id_resolved"].isna().any() and "po_id" in df.columns:
            po_norm = df["po_id"].map(self._normalise_identifier)
            fallback = po_norm.map(self._po_supplier_map)
            df.loc[df["supplier_id_resolved"].isna(), "supplier_id_resolved"] = fallback[
                df["supplier_id_resolved"].isna()
            ]
        if "__supplier_name_resolved" in df.columns:
            mask = df["supplier_id_resolved"].isna()
            df.loc[mask, "supplier_id_resolved"] = df.loc[
                mask, "__supplier_name_resolved"
            ]
        df["supplier_id_resolved"] = df["supplier_id_resolved"].map(
            self._resolve_supplier_id
        )
        df = df.dropna(subset=["supplier_id_resolved"])
        if df.empty:
            return []

        if supplier_filter:
            df = df[df["supplier_id_resolved"] == supplier_filter]
        if df.empty:
            return []

        due_candidates = ["due_date", "invoice_due_date"]
        paid_candidates = ["invoice_paid_date", "payment_date", "paid_date"]
        due_col = next((col for col in due_candidates if col in df.columns), None)
        paid_col = next((col for col in paid_candidates if col in df.columns), None)
        if not due_col:
            return []
        df["due_ts"] = pd.to_datetime(df[due_col], errors="coerce")
        if paid_col:
            df["paid_ts"] = pd.to_datetime(df[paid_col], errors="coerce")
        else:
            df["paid_ts"] = pd.to_datetime(df.get("invoice_paid_date"), errors="coerce")
        invoice_date_col = "invoice_date" if "invoice_date" in df.columns else None
        if invoice_date_col:
            invoice_ts = pd.to_datetime(df[invoice_date_col], errors="coerce")
            df.loc[df["paid_ts"].isna(), "paid_ts"] = invoice_ts[df["paid_ts"].isna()]

        df = df.dropna(subset=["due_ts"])
        if df.empty:
            return []

        df["days_late"] = (
            df["paid_ts"] - df["due_ts"]
        ).dt.days.replace({pd.NA: 0}).fillna(0).astype(float)
        df["days_late"] = df["days_late"].apply(lambda value: max(0.0, value))
        df["is_late"] = df["days_late"] > 0

        value_candidates = [
            "invoice_total_incl_tax",
            "invoice_amount",
            "total_amount",
            "converted_amount_usd",
        ]
        value_col = next((col for col in value_candidates if col in df.columns), None)
        if value_col:
            df["invoice_value"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        else:
            df["invoice_value"] = 0.0

        records: List[Dict[str, Any]] = []
        grouped = df.groupby("supplier_id_resolved")
        for supplier_id, group in grouped:
            total_invoices = int(group.shape[0])
            if total_invoices == 0:
                continue
            late_count = int(group["is_late"].sum())
            late_ratio = late_count / total_invoices if total_invoices else 0.0
            score = 1.0 - late_ratio
            if late_count == 0 or score >= threshold:
                continue
            late_days = group.loc[group["is_late"], "days_late"]
            avg_late_days = float(late_days.mean()) if not late_days.empty else 0.0
            total_value = float(group["invoice_value"].sum())
            late_value = float(group.loc[group["is_late"], "invoice_value"].sum())
            impact = late_value if late_value > 0 else total_value * late_ratio
            evidence = (
                group.sort_values("days_late", ascending=False)["invoice_id"]
                .astype(str)
                .head(5)
                .tolist()
            )
            record = {
                "supplier_id": supplier_id,
                "metric": "on_time_payment",
                "score": round(score, 4),
                "threshold": threshold,
                "impact_gbp": round(max(impact, 0.0), 2),
                "evidence_ids": evidence,
                "auto_detect": True,
                "avg_days_late": round(avg_late_days, 2),
                "late_ratio": round(late_ratio, 4),
            }
            records.append(record)

        return records

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
        supplier_filter = self._resolve_supplier_id(
            self._get_condition(input_data, "supplier_id")
        )
        threshold_raw = self._to_float(
            self._get_condition(input_data, "performance_threshold", 0.9)
        )
        threshold = threshold_raw if threshold_raw is not None else 0.9
        if threshold <= 0 or threshold >= 1:
            threshold = 0.9

        if not isinstance(records, list) or not records:
            auto_records = self._auto_supplier_performance_records(
                tables,
                threshold=threshold,
                supplier_filter=supplier_filter,
            )
            if auto_records:
                records = auto_records
            else:
                details = {"auto_detect": True, "threshold": threshold}
                if supplier_filter:
                    details["supplier_id"] = supplier_filter
                self._log_policy_event(
                    policy_id,
                    None,
                    "blocked",
                    "Performance records must be provided as a list",
                    details,
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
            if record.get("auto_detect"):
                details["auto_detect"] = True
            if record.get("avg_days_late") is not None:
                details.setdefault("avg_days_late", record.get("avg_days_late"))
            if record.get("late_ratio") is not None:
                details.setdefault("late_ratio", record.get("late_ratio"))
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
            info = None
            for key in (
                finding.opportunity_id,
                finding.opportunity_ref_id,
            ):
                if key and key in feedback_map:
                    info = feedback_map[key]
                    break
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
                        "Unable to parse feedback timestamp for %s",
                        finding.opportunity_ref_id or finding.opportunity_id,
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
                SELECT p.supplier_name AS supplier_reference,
                       {po_price_expr} AS unit_price
                FROM proc.po_line_items_agent li
                JOIN proc.purchase_order_agent p ON p.po_id = li.po_id
                WHERE li.item_id = %s
                  AND NULLIF(BTRIM(p.supplier_name), '') IS NOT NULL
            ), invoice_suppliers AS (
                SELECT ia.supplier_name AS supplier_reference,
                       {inv_price_expr} AS unit_price
                FROM proc.invoice_line_items_agent ili
                JOIN proc.invoice_agent ia ON ia.invoice_id = ili.invoice_id
                WHERE ili.item_id = %s
                  AND NULLIF(BTRIM(ia.supplier_name), '') IS NOT NULL
            )
            SELECT supplier_reference, unit_price FROM po_suppliers
            UNION ALL
            SELECT supplier_reference, unit_price FROM invoice_suppliers
        """
        try:
            df = self._read_sql(sql, params=(item_id, item_id))
        except Exception:
            logger.exception("_find_candidate_suppliers query failed for item %s", item_id)
            return []

        if df.empty:
            return []

        df = df.dropna(subset=["supplier_reference", "unit_price"])
        if df.empty:
            return []

        df = df.copy()
        df["supplier_reference"] = df["supplier_reference"].astype(str).str.strip()
        df = df[df["supplier_reference"] != ""]
        if df.empty:
            return []

        df["supplier_id"] = df["supplier_reference"].map(self._resolve_supplier_id)
        unresolved_mask = df["supplier_id"].isna()
        if unresolved_mask.any():
            df.loc[unresolved_mask, "supplier_id"] = df.loc[unresolved_mask, "supplier_reference"]
        df = df.drop(columns=["supplier_reference"])
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


OpportunityMinerAgent._policy_price_benchmark_variance.supports_supplier_autodetect = True
OpportunityMinerAgent._policy_supplier_performance.supports_supplier_autodetect = True

