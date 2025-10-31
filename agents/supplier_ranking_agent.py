from __future__ import annotations

import json
import logging
import multiprocessing
import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from utils.gpu import configure_gpu
from utils.instructions import parse_instruction_sources
from utils.db import read_sql_compat
from utils.reference_loader import load_reference_dataset
from services.supplier_relationship_service import SupplierRelationshipService
from .base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)

# Configure GPU and suppress pandas warnings
configure_gpu()
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pandas only supports SQLAlchemy connectable",
        category=UserWarning,
    )


def _parse_payment_terms_days(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None

    if any(keyword in s for keyword in ("immediate", "due on receipt", "upon receipt")):
        return 0.0

    match = re.search(r"(\d+)\s*(?:day|days|d)?", s)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _normalize_days_to_score(
    days: Optional[float], min_days: float = 0.0, max_days: float = 90.0
) -> Optional[float]:
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return None
    try:
        numeric = float(days)
    except (TypeError, ValueError):
        return None
    clamped = max(min_days, min(max_days, numeric))
    if max_days == min_days:
        return 100.0
    score = (1.0 - (clamped - min_days) / (max_days - min_days)) * 100
    return float(round(score, 2))


def ensure_payment_terms_score(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    if "payment_terms_days" not in df.columns or df["payment_terms_days"].isna().all():
        source_cols = [
            column
            for column in df.columns
            if column.lower() in {"payment_terms", "terms", "pay_terms"}
        ]
        if source_cols:
            src = source_cols[0]
            df["payment_terms_days"] = df[src].apply(_parse_payment_terms_days)
        else:
            df["payment_terms_days"] = np.nan

    for column in df.columns:
        if column.lower() in {
            "payment_terms_days",
            "payment_terms_in_days",
            "pt_days",
        }:
            df["payment_terms_days"] = pd.to_numeric(df[column], errors="coerce")
            break

    needs_score = "payment_terms_score" not in df.columns or df[
        "payment_terms_score"
    ].isna().all()
    if needs_score:
        df["payment_terms_score"] = df["payment_terms_days"].apply(
            _normalize_days_to_score
        )

    na_mask = df["payment_terms_score"].isna()
    if na_mask.any():
        logger.warning(
            "Imputing neutral payment_terms_score=50 for %d row(s) with unknown terms",
            int(na_mask.sum()),
        )
        df.loc[na_mask, "payment_terms_score"] = 50.0

    return df


class SupplierRankingAgent(BaseAgent):
    """Rank suppliers using procurement data, policies and contextual scores."""

    AGENTIC_PLAN_STEPS = (
        "Aggregate supplier performance, spend, and policy context relevant to the query.",
        "Normalise metrics, apply weightings, and compute composite supplier scores.",
        "Return ranked suppliers with rationale and hand-offs for downstream decisions.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.prompt_library: Dict[str, Any] = {}
        self.justification_template: Dict[str, Any] = {}
        self.policy_engine = agent_nick.policy_engine
        self.query_engine = agent_nick.query_engine
        self._device = configure_gpu()
        self._supplier_alias_map: Dict[str, str] = {}
        self._supplier_lookup: Dict[str, Optional[str]] = {}
        self._scoring_reference = load_reference_dataset("supplier_scoring_reference")
        try:
            self._relationship_service = SupplierRelationshipService(agent_nick)
        except Exception:
            logger.debug(
                "SupplierRankingAgent failed to initialise SupplierRelationshipService",
                exc_info=True,
            )
            self._relationship_service = None
        self._cache = {}
        self._max_workers = min(32, (multiprocessing.cpu_count() or 1) * 4)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize required database schema."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE SCHEMA IF NOT EXISTS proc;
                        
                        CREATE TABLE IF NOT EXISTS proc.procurement_flow (
                            supplier_id VARCHAR(255),
                            profile JSONB,
                            vector_embedding BYTEA,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (supplier_id)
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_proc_flow_supplier 
                        ON proc.procurement_flow(supplier_id);
                    """)
                conn.commit()
        except Exception:
            logger.exception("Failed to initialize schema")

    @lru_cache(maxsize=1000)
    def _fetch_supplier_profile(self, supplier_id: str) -> Dict[str, Any]:
        """Cache supplier profiles with vector handling."""
        cache_key = f"profile:{supplier_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT profile, vector_embedding 
                        FROM proc.procurement_flow 
                        WHERE supplier_id = %s 
                        LIMIT 1
                    """, (supplier_id,))
                    row = cur.fetchone()
                    if not row:
                        return {}
                    
                    profile = dict(row[0]) if row[0] else {}
                    if row[1] is not None:  # Handle vector embedding if present
                        try:
                            profile['vector_embedding'] = np.frombuffer(row[1], dtype=np.float32)
                        except Exception:
                            logger.warning(f"Failed to parse vector embedding for supplier {supplier_id}")
                    
                    self._cache[cache_key] = profile
                    return profile
        except Exception:
            logger.exception(f"Failed to fetch profile for supplier {supplier_id}")
            return {}

    def _batch_fetch_supplier_profiles(self, supplier_ids: Set[str], batch_size: int = 50) -> Dict[str, Dict[str, Any]]:
        """Fetch supplier profiles in batches with better error handling."""
        profiles = {}
        supplier_batches = [list(supplier_ids)[i:i + batch_size] for i in range(0, len(supplier_ids), batch_size)]
        
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'proc'
                            AND table_name = 'procurement_flow'
                        );
                    """)
                    table_exists = cur.fetchone()[0]
                    
                    if not table_exists:
                        logger.warning("proc.procurement_flow table does not exist - initializing schema")
                        self._init_schema()
                        return profiles

                    for batch in supplier_batches:
                        try:
                            cur.execute("""
                                SELECT supplier_id, profile, vector_embedding
                                FROM proc.procurement_flow 
                                WHERE supplier_id = ANY(%s)
                            """, (batch,))
                            
                            for row in cur.fetchall():
                                sid, profile_data, vector_data = row
                                if not profile_data:
                                    continue
                                    
                                profile = dict(profile_data)
                                if vector_data is not None:
                                    try:
                                        profile['vector_embedding'] = np.frombuffer(vector_data, dtype=np.float32)
                                    except Exception:
                                        logger.warning(f"Failed to parse vector embedding for supplier {sid}")
                                
                                cache_key = f"profile:{sid}"
                                self._cache[cache_key] = profile
                                profiles[sid] = profile
                                
                        except Exception:
                            logger.exception(f"Failed to fetch profiles for batch of {len(batch)} suppliers")
                            continue
                    
        except Exception:
            logger.exception("Database connection failed while fetching supplier profiles")
            
        return profiles

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

    def _collect_instruction_bundle(self, context: AgentContext) -> Dict[str, Any]:
        sources: List[Any] = []
        for policy in context.input_data.get("policies") or []:
            sources.extend(self._instruction_sources_from_policy(policy))
        for prompt in context.input_data.get("prompts") or []:
            sources.extend(self._instruction_sources_from_prompt(prompt))
        return parse_instruction_sources(sources)

    def _ingest_prompt_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("templates") and not self.prompt_library:
            self.prompt_library = dict(payload)
        if payload.get("prompt_template") and not self.justification_template:
            self.justification_template = dict(payload)

    def _load_prompt_from_db(self, prompt_id: int) -> None:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return
        try:
            with get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT prompts_desc FROM proc.prompt WHERE prompt_id = %s",
                        (prompt_id,),
                    )
                    row = cursor.fetchone()
            if not row:
                return
            raw = row[0]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="ignore")
            if isinstance(raw, str):
                text = raw.strip()
                if not text:
                    return
                try:
                    payload = json.loads(text)
                except Exception:
                    return
            elif isinstance(raw, dict):
                payload = raw
            else:
                return
            self._ingest_prompt_payload(payload)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load prompt %s from database", prompt_id)

    def _ensure_prompt_assets(self, context: AgentContext) -> None:
        if self.prompt_library and self.justification_template:
            return

        seen_ids: set[int] = set()
        for prompt in context.input_data.get("prompts") or []:
            if not isinstance(prompt, dict):
                continue
            self._ingest_prompt_payload(prompt)
            pid = prompt.get("promptId")
            try:
                if pid is not None:
                    seen_ids.add(int(pid))
            except (TypeError, ValueError):
                continue

        if (not self.prompt_library or not self.justification_template) and seen_ids:
            for pid in seen_ids:
                if self.prompt_library and self.justification_template:
                    break
                self._load_prompt_from_db(pid)

        if not self.prompt_library:
            self.prompt_library = {"templates": []}
        if not self.justification_template:
            self.justification_template = {
                "prompt_template": (
                    "Supplier {supplier_name} achieved a final score of {final_score:.2f}."
                    "\n{score_breakdown}"
                )
            }

    def _coerce_numeric_map(self, payload: Any) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric >= 0:
                    result[str(key).strip()] = numeric
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                if isinstance(item, dict):
                    result.update(self._coerce_numeric_map(item))
        elif isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return self._coerce_numeric_map(parsed)
        return result

    def _ensure_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
            return [token.strip() for token in re.split(r"[,;]", text) if token.strip()]
        return [str(value).strip()]

    def _payment_terms_to_days(self, series: Any) -> pd.Series:
        if series is None:
            return pd.Series(dtype="float64")
        if isinstance(series, pd.Series):
            values = series
        else:
            try:
                values = pd.Series(series)
            except Exception:
                return pd.Series(dtype="float64")
        reference_mapping = {}
        if isinstance(self._scoring_reference, dict):
            reference_mapping = self._scoring_reference.get("payment_terms_days", {}) or {}
        mapping = {
            str(key).lower(): float(value)
            for key, value in reference_mapping.items()
            if value is not None
        }
        days: List[Optional[float]] = []
        for value in values:
            if pd.isna(value):
                days.append(None)
                continue
            if isinstance(value, (int, float)):
                days.append(float(value))
                continue
            text = str(value).strip()
            if not text:
                days.append(None)
                continue
            lower = text.lower()
            normalized = re.sub(r"[^a-z0-9]", "", lower)
            mapped = mapping.get(normalized)
            if mapped is None:
                match = re.search(r"(\d+)", lower)
                if match:
                    mapped = float(match.group(1))
            days.append(mapped)
        return pd.Series(days, index=values.index, dtype="float64")

    def _coerce_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(policy)
        details = entry.get("details")
        if isinstance(details, str):
            try:
                entry["details"] = json.loads(details)
            except Exception:  # pragma: no cover - defensive
                entry["details"] = {}
        elif isinstance(details, dict):
            entry["details"] = dict(details)
        else:
            entry["details"] = {}
        if not entry["details"]:
            raw_details = entry.get("policy_details")
            if isinstance(raw_details, str):
                try:
                    entry["details"] = json.loads(raw_details)
                except Exception:  # pragma: no cover - defensive
                    entry["details"] = {}
            elif isinstance(raw_details, dict):
                entry["details"] = dict(raw_details)
        rules = entry["details"].get("rules") if isinstance(entry["details"], dict) else {}
        if isinstance(rules, str):
            try:
                entry["details"]["rules"] = json.loads(rules)
            except Exception:  # pragma: no cover - defensive
                entry["details"]["rules"] = {}
        elif isinstance(rules, dict):
            entry["details"]["rules"] = dict(rules)
        else:
            entry["details"]["rules"] = {}
        return entry

    def _resolve_policy_bundle(self, context: AgentContext) -> List[Dict[str, Any]]:
        raw = context.input_data.get("policies")
        bundle: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for policy in raw:
                if isinstance(policy, dict):
                    bundle.append(self._coerce_policy(policy))
        if bundle:
            return bundle
        return [self._coerce_policy(policy) for policy in self.policy_engine.supplier_policies]

    def _find_policy(
        self, policies: List[Dict[str, Any]], name: str
    ) -> Optional[Dict[str, Any]]:
        target = str(name).strip().lower()
        for policy in policies:
            policy_name = (
                policy.get("policyName")
                or policy.get("policy_name")
                or policy.get("name")
                or policy.get("description")
            )
            if policy_name and str(policy_name).strip().lower() == target:
                return policy
        return None

    def _extract_policy_rules(self, policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not policy:
            return {}
        details = policy.get("details")
        if not isinstance(details, dict):
            return {}
        rules = details.get("rules")
        return dict(rules) if isinstance(rules, dict) else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        logger.info("SupplierRankingAgent: Starting ranking...")

        supplier_data = context.input_data.get("supplier_data")
        if supplier_data is None:
            try:
                supplier_data = self.query_engine.fetch_supplier_data(
                    context.input_data
                )
            except Exception:
                logger.exception("Failed to fetch supplier data")
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="Failed to fetch supplier data",
                    ),
                )
        try:
            df = (
                supplier_data.copy()
                if isinstance(supplier_data, pd.DataFrame)
                else pd.DataFrame(supplier_data)
            )
        except Exception:
            logger.exception("Invalid supplier_data format")
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="Failed to parse supplier_data into DataFrame",
                ),
            )

        if "supplier_id" in df.columns:
            df["supplier_id"] = df["supplier_id"].astype(str).str.strip()


        if df.empty:
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="supplier_data is empty",
                ),
            )

        if "supplier_id" not in df.columns:
            if "supplier_name" in df.columns:
                df["supplier_id"] = df["supplier_name"].astype(str).str.strip()
                logger.warning(
                    "supplier_data missing 'supplier_id'; derived IDs from supplier_name"
                )
            else:
                logger.error("supplier_data missing 'supplier_id' column")
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="supplier_data missing 'supplier_id' column",
                    ),
                )

        self._ensure_prompt_assets(context)

        instructions = self._collect_instruction_bundle(context)

        policy_bundle = self._resolve_policy_bundle(context)

        raw_directory = context.input_data.get("supplier_directory")
        directory_entries: List[Dict[str, Any]] = []
        if isinstance(raw_directory, list):
            directory_entries = [entry for entry in raw_directory if isinstance(entry, dict)]
        if not directory_entries:
            directory_entries = self._build_directory_from_dataframe(df)
        directory_lookup: Dict[str, Dict[str, Any]] = {}
        directory_map: Dict[str, Optional[str]] = {}
        for entry in directory_entries:
            if not isinstance(entry, dict):
                continue
            supplier_id = entry.get("supplier_id")
            if supplier_id is None:
                continue
            sid = str(supplier_id).strip()
            if not sid:
                continue
            directory_lookup[sid] = entry
            supplier_name = entry.get("supplier_name")
            directory_map[sid] = (
                str(supplier_name).strip() if isinstance(supplier_name, str) and supplier_name.strip() else None
            )


        if directory_map and "supplier_id" in df.columns:
            mapped_names = df["supplier_id"].map(directory_map)
            if "supplier_name" in df.columns:
                df["supplier_name"] = mapped_names.combine_first(df["supplier_name"])
            else:
                df["supplier_name"] = mapped_names

        candidate_ids = context.input_data.get("supplier_candidates")
        candidate_set = self._normalise_id_set(candidate_ids)
        directory_ids = {sid for sid in directory_map.keys() if sid}
        if directory_ids:
            if candidate_set:
                candidate_set = {sid for sid in candidate_set if sid in directory_ids}
                if not candidate_set:
                    candidate_set = directory_ids
            else:
                candidate_set = directory_ids
        if candidate_set:
            df = df[df["supplier_id"].astype(str).str.strip().isin(candidate_set)].copy()
            df = self._ensure_candidate_rows(df, candidate_set, directory_lookup)
            if directory_map and "supplier_id" in df.columns:
                mapped_names = df["supplier_id"].map(directory_map)
                if "supplier_name" in df.columns:
                    df["supplier_name"] = mapped_names.combine_first(df["supplier_name"])
                else:
                    df["supplier_name"] = mapped_names

            if df.empty:
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="No matching suppliers found for candidates",
                    ),
                )

        if "supplier_name" in df.columns:
            df["supplier_name"] = df["supplier_name"].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )

        self._prime_supplier_aliases(df, directory_entries)

        if self._supplier_lookup and "supplier_name" in df.columns:
            canonical_names = df["supplier_id"].map(self._supplier_lookup.get)
            df["supplier_name"] = canonical_names.combine_first(df["supplier_name"])

        supplier_scope = set()
        if candidate_set:
            supplier_scope = {sid for sid in candidate_set if sid}
        if not supplier_scope and "supplier_id" in df.columns:
            supplier_scope = {
                sid
                for sid in (
                    self._coerce_supplier_id(value) for value in df["supplier_id"]
                )
                if sid
            }
        tables = self._load_procurement_tables(supplier_scope, df)
        df = self._merge_supplier_metrics(df, tables)
        profiles = self._build_supplier_profiles(tables, df["supplier_id"].astype(str))

        external_profiles = context.input_data.get("supplier_category_profiles")
        if isinstance(external_profiles, str):
            try:
                external_profiles = json.loads(external_profiles)
            except Exception:  # pragma: no cover - defensive
                external_profiles = {}
        if isinstance(external_profiles, dict):
            for supplier_id, extra in external_profiles.items():
                if not isinstance(extra, dict):
                    continue
                sid = str(supplier_id).strip()
                if not sid:
                    continue
                profile = profiles.setdefault(
                    sid,
                    {
                        "supplier_id": sid,
                        "po_ids": [],
                        "invoice_ids": [],
                        "items": [],
                        "categories": [],
                    },
                )
                primary_category = extra.get("primary_category")
                if primary_category and not profile.get("primary_category"):
                    profile["primary_category"] = primary_category
                categories_payload = extra.get("category_breakdown") or extra.get("categories")
                if isinstance(categories_payload, list):
                    profile["categories"] = categories_payload
                elif isinstance(categories_payload, dict):
                    profile["categories"] = [
                        {"category": key, "occurrences": value}
                        for key, value in categories_payload.items()
                    ]
                products = extra.get("products")
                if isinstance(products, list) and products:
                    profile["products"] = products
                sources = extra.get("sources")
                if isinstance(sources, list) and sources:
                    profile["sources"] = sources

        if profiles and "supplier_id" in df.columns:
            primary_category_map = {
                sid: profile.get("primary_category")
                for sid, profile in profiles.items()
                if isinstance(profile, dict)
            }
            df["primary_category"] = df["supplier_id"].map(primary_category_map)

        contexts_by_id, contexts_by_name = self._fetch_relationship_context(df)
        if contexts_by_id or contexts_by_name:
            df = self._merge_relationship_context(df, contexts_by_id, contexts_by_name)

        coverage_series = self._derive_metric_coverage(df)
        if not coverage_series.empty:
            df["flow_coverage"] = coverage_series
        else:
            if "flow_coverage" not in df.columns:
                df["flow_coverage"] = 0.0

        flow_payload = context.input_data.get("data_flow_snapshot")
        if isinstance(flow_payload, dict):
            flow_index, flow_name_index = self._build_flow_index(flow_payload)
        else:
            flow_index, flow_name_index = self._build_flow_index(context.input_data)
        alias_tokens_map = self._alias_tokens_by_supplier()
        if flow_index or flow_name_index:
            df = self._annotate_flow_coverage(
                df, flow_index, flow_name_index, alias_tokens_map
            )

        intent = context.input_data.get("intent", {})
        requested = intent.get("parameters", {}).get("criteria", [])
        criteria_override = (
            instructions.get("criteria")
            or instructions.get("metrics")
            or instructions.get("focus_metrics")
        )
        override_criteria = self._ensure_list(criteria_override)
        if override_criteria:
            requested = override_criteria
            intent.setdefault("parameters", {})["criteria"] = override_criteria

        weight_policy = self._find_policy(policy_bundle, "WeightAllocationPolicy")
        weight_rules = self._extract_policy_rules(weight_policy)
        default_weights = weight_rules.get("default_weights", {})
        override_weights_map: Dict[str, float] = {}
        for key in ("metric_weights", "weights", "weightings", "default_weights"):
            override_weights_map = self._coerce_numeric_map(instructions.get(key))
            if override_weights_map:
                break
        if override_weights_map:
            total_override = sum(override_weights_map.values())
            if total_override > 0:
                default_weights = {
                    metric: value / total_override
                    for metric, value in override_weights_map.items()
                    if value >= 0
                }
            else:
                default_weights = override_weights_map

        criteria = requested if requested else list(default_weights.keys())
        weights = {
            crit: default_weights.get(crit, 0.0)
            for crit in criteria
            if default_weights.get(crit, 0.0) > 0
        }

        if not weights:
            fallback_metrics = [
                metric
                for metric in ("price", "delivery", "risk", "payment_terms")
                if metric in df.columns or f"{metric}_score" in df.columns
            ]
            if fallback_metrics:
                equal_weight = 1.0 / len(fallback_metrics)
                weights = {metric: equal_weight for metric in fallback_metrics}

        df = self._prepare_scoring_columns(df, weights)
        df = ensure_payment_terms_score(df)
        scored_df = self._score_categorical_criteria(df, weights.keys(), policy_bundle)
        norm_policy = self._find_policy(policy_bundle, "NormalizationDirectionPolicy")
        direction_map = self._extract_policy_rules(norm_policy)
        scored_df = self._normalize_numeric_scores(scored_df, direction_map)

        normalised_weights = self._normalise_weight_map(scored_df, weights)
        if normalised_weights:
            weights = normalised_weights

        scored_df["final_score"] = 0.0
        for crit, weight in weights.items():
            score_col = f"{crit}_score"
            if score_col not in scored_df.columns:
                logger.warning("Criterion column missing: %s", score_col)
                continue
            scored_df["final_score"] += (
                pd.to_numeric(scored_df[score_col].fillna(0), errors="coerce")
                * float(weight)
            )

        scored_df = self._apply_flow_bonus(
            scored_df, flow_index, flow_name_index, alias_tokens_map
        )

        ranked_df = scored_df.sort_values(
            by="final_score", ascending=False
        ).reset_index(drop=True)

        top_n = intent.get("parameters", {}).get("top_n")
        if not top_n:
            query_text = context.input_data.get("query", "")
            match = re.search(r"top[-\s]*(\d+)", query_text, re.IGNORECASE)
            top_n = int(match.group(1)) if match else 3
        override_top_n = (
            instructions.get("top_n")
            or instructions.get("max_suppliers")
            or instructions.get("supplier_limit")
        )
        if override_top_n is not None:
            try:
                top_n = int(float(override_top_n))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        top_n = max(1, min(int(top_n), len(ranked_df)))

        top_df = ranked_df.head(top_n).copy()
        top_df["justification"] = top_df.apply(
            lambda row: self._generate_justification(row, weights.keys()), axis=1
        )

        ranking = [
            self._prepare_ranking_entry(row, profiles.get(str(row.get("supplier_id"))), weights)
            for _, row in top_df.iterrows()
        ]

        if candidate_set:
            ranking = [
                entry
                for entry in ranking
                if str(entry.get("supplier_id", "")).strip() in candidate_set
            ]

        total_rankings = len(ranking)
        for index, entry in enumerate(ranking, start=1):
            entry["rank_position"] = index
            entry["rank_count"] = total_rankings

        logger.info(
            "SupplierRankingAgent: Ranking complete with %d entries", len(ranking)
        )

        output_data = {
            "ranking": ranking,
            "supplier_profiles": profiles,
            "rank_count": total_rankings,
        }
        pass_fields = {
            "ranking": ranking,
            "supplier_profiles": profiles,
        }
        if total_rankings:
            pass_fields["rank_count"] = total_rankings
        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=output_data,
                pass_fields=pass_fields,
                next_agents=["EmailDraftingAgent"],
            ),
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _normalise_id_set(self, ids: Optional[Iterable]) -> set[str]:
        if not ids:
            return set()
        try:
            return {str(val).strip() for val in ids if str(val).strip()}
        except Exception:
            return {str(ids).strip()}

    def _normalise_supplier_token(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        if not text:
            return None
        return re.sub(r"\s+", " ", text.lower())

    def _coerce_supplier_id(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        return text or None

    def _build_directory_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        if "supplier_id" not in df.columns:
            return []

        entries: Dict[str, Dict[str, Any]] = {}
        names: pd.Series
        if "supplier_name" in df.columns:
            names = df["supplier_name"]
        else:
            names = pd.Series([None] * len(df), index=df.index, dtype="object")

        for supplier_id, supplier_name in zip(df["supplier_id"], names):
            sid = self._coerce_supplier_id(supplier_id)
            if not sid:
                continue
            entry = entries.setdefault(sid, {"supplier_id": sid})
            if isinstance(supplier_name, str) and supplier_name.strip():
                entry.setdefault("supplier_name", supplier_name.strip())
        return list(entries.values())

    def _prime_supplier_aliases(
        self, supplier_df: pd.DataFrame, directory_entries: Iterable[Dict[str, Any]]
    ) -> None:
        alias_map: Dict[str, str] = {}
        lookup: Dict[str, Optional[str]] = {}

        if isinstance(supplier_df, pd.DataFrame) and not supplier_df.empty:
            alias_fields = [
                col for col in ("supplier_name", "trading_name") if col in supplier_df.columns
            ]
            for row in supplier_df.itertuples(index=False):
                sid = self._coerce_supplier_id(getattr(row, "supplier_id", None))
                if not sid:
                    continue
                canonical_name = getattr(row, "supplier_name", None)
                if not isinstance(canonical_name, str) or not canonical_name.strip():
                    for field in alias_fields:
                        value = getattr(row, field, None)
                        if isinstance(value, str) and value.strip():
                            canonical_name = value
                            break
                if isinstance(canonical_name, str):
                    lookup.setdefault(sid, canonical_name.strip())
                else:
                    lookup.setdefault(sid, None)
                norm_id = self._normalise_supplier_token(sid)
                if norm_id:
                    alias_map.setdefault(norm_id, sid)
                for field in alias_fields:
                    value = getattr(row, field, None)
                    norm = self._normalise_supplier_token(value)
                    if norm and norm not in alias_map:
                        alias_map[norm] = sid

        for entry in directory_entries or []:
            if not isinstance(entry, dict):
                continue
            sid = self._coerce_supplier_id(entry.get("supplier_id"))
            if not sid:
                continue
            name = entry.get("supplier_name")
            if isinstance(name, str) and name.strip():
                lookup.setdefault(sid, name.strip())
            norm_id = self._normalise_supplier_token(sid)
            if norm_id:
                alias_map.setdefault(norm_id, sid)
            for candidate in (entry.get("supplier_name"), entry.get("trading_name")):
                norm = self._normalise_supplier_token(candidate)
                if norm and norm not in alias_map:
                    alias_map[norm] = sid

        self._supplier_alias_map = {key: val for key, val in alias_map.items() if key and val}
        self._supplier_lookup = {sid: lookup.get(sid) for sid in {val for val in alias_map.values()}}

    def _resolve_supplier_identifier(self, value: Any) -> Optional[str]:
        sid = self._coerce_supplier_id(value)
        norm = self._normalise_supplier_token(value)
        if norm and norm in self._supplier_alias_map:
            return self._supplier_alias_map[norm]
        if sid and (sid in self._supplier_lookup or sid in self._supplier_alias_map.values()):
            return sid
        return sid

    def _map_supplier_ids(self, df: pd.DataFrame, name_fields: Iterable[str]) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        if df.empty:
            return df

        out = df.copy()

        if "supplier_id" in out.columns:
            out["supplier_id"] = out["supplier_id"].apply(self._coerce_supplier_id)
        else:
            out["supplier_id"] = pd.Series([None] * len(out), index=out.index, dtype="object")

        missing_mask = out["supplier_id"].isna()
        for field in name_fields:
            if field not in out.columns:
                continue
            resolved = out.loc[missing_mask, field].apply(self._resolve_supplier_identifier)
            out.loc[missing_mask, "supplier_id"] = resolved
            missing_mask = out["supplier_id"].isna()
            if not missing_mask.any():
                break

        if "supplier_name" in out.columns:
            out["supplier_name"] = out["supplier_name"].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )
            if self._supplier_lookup:
                canonical = out["supplier_id"].map(self._supplier_lookup.get)
                out["supplier_name"] = canonical.combine_first(out["supplier_name"])

        return out

    def _ensure_candidate_rows(
        self,
        df: pd.DataFrame,
        candidate_ids: set[str],
        directory_lookup: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        if not candidate_ids:
            return df

        present_ids: set[str] = set()
        if "supplier_id" in df.columns and not df.empty:
            present_ids = set(df["supplier_id"].astype(str).str.strip())

        missing = [cid for cid in candidate_ids if cid not in present_ids]

        if not missing and not df.empty:
            return df

        fallback_rows: List[Dict[str, Any]] = []
        for cid in missing:
            entry = directory_lookup.get(cid, {})
            row: Dict[str, Any] = dict(entry) if isinstance(entry, dict) else {}
            row["supplier_id"] = cid
            supplier_name = row.get("supplier_name")
            if isinstance(supplier_name, str):
                row["supplier_name"] = supplier_name.strip()
            elif supplier_name is None:
                row["supplier_name"] = pd.NA
            fallback_rows.append(row)

        if not fallback_rows and df.empty:
            fallback_rows = [{"supplier_id": cid, "supplier_name": pd.NA} for cid in candidate_ids]

        if not fallback_rows:
            return df

        fallback_df = pd.DataFrame(fallback_rows)
        if "supplier_id" in fallback_df.columns:
            fallback_df["supplier_id"] = fallback_df["supplier_id"].astype(str).str.strip()

        if df.empty:
            return fallback_df

        combined = pd.concat([df, fallback_df], ignore_index=True, sort=False)
        if "supplier_id" in combined.columns:
            combined["supplier_id"] = combined["supplier_id"].astype(str).str.strip()
        return combined

    def _load_procurement_tables(
        self, supplier_ids: Iterable[str], supplier_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        suppliers = {str(s).strip() for s in supplier_ids if str(s).strip()}
        supplier_names: set[str] = set()
        if isinstance(supplier_df, pd.DataFrame) and not supplier_df.empty:
            if "supplier_name" in supplier_df.columns:
                supplier_names = {
                    str(name).strip().lower()
                    for name in supplier_df["supplier_name"].dropna()
                    if str(name).strip()
                }

        tables: Dict[str, pd.DataFrame] = {}

        try:
            po_df = self.query_engine.fetch_purchase_order_data(
                supplier_ids=suppliers or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.exception("Failed to load purchase orders")
            po_df = pd.DataFrame()
        tables["purchase_orders"] = self._map_supplier_ids(po_df, ("supplier_name",))

        try:
            invoice_df = self.query_engine.fetch_invoice_data(
                supplier_ids=suppliers or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.exception("Failed to load invoices")
            invoice_df = pd.DataFrame()
        tables["invoices"] = self._map_supplier_ids(invoice_df, ("supplier_name",))

        po_ids: List[str] = []
        if not po_df.empty and "po_id" in po_df.columns:
            po_ids = [
                str(value).strip()
                for value in po_df["po_id"].dropna()
                if str(value).strip()
            ]
        invoice_ids: List[str] = []
        if not invoice_df.empty and "invoice_id" in invoice_df.columns:
            invoice_ids = [
                str(value).strip()
                for value in invoice_df["invoice_id"].dropna()
                if str(value).strip()
            ]

        tables["po_lines"] = self._read_table(
            "proc.po_line_items_agent",
            "po_id = ANY(%s)" if po_ids else None,
            [po_ids] if po_ids else None,
            columns=(
                "po_id",
                "po_line_id",
                "line_number",
                "item_id",
                "item_description",
                "quantity",
                "unit_price",
                "line_total",
                "total_amount",
            ),
        )
        tables["invoice_lines"] = self._read_table(
            "proc.invoice_line_items_agent",
            "invoice_id = ANY(%s)" if invoice_ids else None,
            [invoice_ids] if invoice_ids else None,
            columns=(
                "invoice_id",
                "invoice_line_id",
                "po_id",
                "item_description",
                "total_amount_incl_tax",
            ),
        )
        try:
            flow = self.query_engine.fetch_procurement_flow(
                embed=False,
                supplier_ids=suppliers or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.exception("Failed to load procurement flow")
            flow = pd.DataFrame()
        flow = self._map_supplier_ids(flow, ("supplier_name",))
        if not flow.empty and suppliers and "supplier_id" in flow.columns:
            supplier_series = flow["supplier_id"].apply(self._coerce_supplier_id)
            mask = supplier_series.isin(suppliers)
            flow = flow[mask].copy()
            flow["supplier_id"] = supplier_series.loc[flow.index]
        tables["procurement_flow"] = flow
        return tables

    def _read_table(
        self,
        table: str,
        where: Optional[str] = None,
        params: Optional[Any] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        column_sql = ", ".join(columns) if columns else "*"
        sql = f"SELECT {column_sql} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        pandas_conn = getattr(self.agent_nick, "pandas_connection", None)
        try:
            if callable(pandas_conn):
                with pandas_conn() as conn:
                    return read_sql_compat(sql, conn, params=params)
            with self.agent_nick.get_db_connection() as conn:
                return read_sql_compat(sql, conn, params=params)
        except Exception:
            logger.exception("Failed to read table %s", table)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Metric enrichment
    # ------------------------------------------------------------------
    def _merge_supplier_metrics(
        self, df: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if df.empty:
            return df

        po_summary = self._summarise_purchase_orders(tables.get("purchase_orders", pd.DataFrame()))
        po_line_summary = self._summarise_po_lines(
            tables.get("po_lines", pd.DataFrame()),
            tables.get("purchase_orders", pd.DataFrame()),
        )
        invoice_summary = self._summarise_invoices(tables.get("invoices", pd.DataFrame()))
        invoice_line_summary = self._summarise_invoice_lines(
            tables.get("invoice_lines", pd.DataFrame()),
            tables.get("invoices", pd.DataFrame()),
        )

        result = df.copy()
        for summary in [po_summary, po_line_summary, invoice_summary, invoice_line_summary]:
            if summary.empty:
                continue
            result = result.merge(summary, on="supplier_id", how="left")

        for column in [
            "po_total_value",
            "po_line_spend",
            "invoice_total_value",
            "avg_unit_price",
            "total_volume",
            "avg_lead_time_days",
            "total_spend",
            "avg_payment_term_days",
            "avg_paid_days",
        ]:
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")

        spend_components = [
            result.get("po_total_value"),
            result.get("invoice_total_value"),
            result.get("po_line_spend"),
        ]
        numeric_components: List[pd.Series] = []
        for comp in spend_components:
            if isinstance(comp, pd.Series):
                numeric_components.append(pd.to_numeric(comp, errors="coerce").fillna(0.0))

        if "total_spend" in result.columns:
            total_spend_series = pd.to_numeric(
                result["total_spend"], errors="coerce"
            ).fillna(0.0)
        else:
            total_spend_series = pd.Series(0.0, index=result.index, dtype="float64")

        for comp_series in numeric_components:
            total_spend_series = total_spend_series.add(comp_series, fill_value=0.0)

        result["total_spend"] = total_spend_series

        if "avg_unit_price" not in result.columns:
            result["avg_unit_price"] = pd.NA
        if "avg_unit_price" in result.columns:
            missing_price = result["avg_unit_price"].isna()
            if "po_line_spend" in result.columns and "total_volume" in result.columns:
                # Fix deprecated use_inf_as_na
                calculated = (
                    result["po_line_spend"].fillna(0) / 
                    result["total_volume"].replace([np.inf, -np.inf], np.nan)
                ).fillna(0)
                # Ensure compatible dtype
                result.loc[missing_price, "avg_unit_price"] = calculated[missing_price].astype(float)

        if "payment_terms" not in result.columns and "po_payment_terms" in result.columns:
            result["payment_terms"] = result["po_payment_terms"]

        return result

    def _derive_metric_coverage(self, df: pd.DataFrame) -> pd.Series:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype="float64")

        def _signal(columns: Tuple[str, ...]) -> Optional[pd.Series]:
            series_list: List[pd.Series] = []
            for column in columns:
                if column not in df.columns:
                    continue
                series = pd.to_numeric(df[column], errors="coerce")
                if series.isna().all():
                    continue
                series_list.append(series.fillna(0.0))
            if not series_list:
                return None
            combined = sum(series_list)
            if isinstance(combined, pd.Series):
                return (combined.fillna(0.0) > 0).astype(float)
            return None

        signals: List[pd.Series] = []
        for cols in (
            ("po_total_value", "po_line_spend"),
            ("invoice_total_value", "invoice_count"),
            ("invoice_item_count", "invoice_count"),
            ("total_spend",),
        ):
            signal = _signal(cols)
            if signal is not None:
                signals.append(signal)

        if not signals:
            return pd.Series(0.0, index=df.index, dtype="float64")

        combined = sum(signals)
        if not isinstance(combined, pd.Series):
            return pd.Series(0.0, index=df.index, dtype="float64")

        return (combined / len(signals)).clip(lower=0.0, upper=1.0)

    def _summarise_purchase_orders(self, po_df: pd.DataFrame) -> pd.DataFrame:
        if po_df.empty:
            return pd.DataFrame()
        df = self._map_supplier_ids(po_df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)
        value_col = "total_amount_gbp" if "total_amount_gbp" in df.columns else "total_amount"
        if value_col in df.columns:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        else:
            df[value_col] = 0.0

        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        if "expected_delivery_date" in df.columns:
            df["expected_delivery_date"] = pd.to_datetime(
                df["expected_delivery_date"], errors="coerce"
            )
            df["lead_time_days"] = (
                df["expected_delivery_date"] - df.get("order_date")
            ).dt.days

        agg = df.groupby("supplier_id").agg(
            po_total_value=(value_col, "sum"),
            po_count=("po_id", "nunique"),
            po_payment_terms=("payment_terms", lambda s: self._mode_value(s)),
            last_order_date=("order_date", "max"),
            avg_lead_time_days=("lead_time_days", "mean"),
        )
        return agg.reset_index()

    def _summarise_po_lines(
        self, po_lines: pd.DataFrame, po_df: pd.DataFrame
    ) -> pd.DataFrame:
        if po_lines.empty or "po_id" not in po_lines.columns:
            return pd.DataFrame()
        if po_df.empty or "po_id" not in po_df.columns:
            return pd.DataFrame()
        po_lookup = self._map_supplier_ids(po_df, ("supplier_name",))
        join_cols = ["po_id"]
        extra_cols = [col for col in ("supplier_id", "supplier_name") if col in po_lookup.columns]
        if not extra_cols:
            return pd.DataFrame()
        join_df = po_lookup[join_cols + extra_cols].drop_duplicates("po_id")
        df = po_lines.merge(join_df, on="po_id", how="left")
        df = self._map_supplier_ids(df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)

        for col in ["unit_price", "line_total", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "line_total" not in df.columns:
            df["line_total"] = df.get("unit_price", 0.0) * df.get("quantity", 0.0)

        agg = df.groupby("supplier_id").agg(
            po_line_spend=("line_total", "sum"),
            avg_unit_price=("unit_price", "mean"),
            total_volume=("quantity", "sum"),
            catalog_items=("item_description", lambda s: self._top_values(s, limit=5)),
        )
        return agg.reset_index()

    def _summarise_invoices(self, invoice_df: pd.DataFrame) -> pd.DataFrame:
        if invoice_df.empty:
            return pd.DataFrame()
        df = self._map_supplier_ids(invoice_df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)
        total_col = "invoice_total_incl_tax" if "invoice_total_incl_tax" in df.columns else "invoice_amount"
        if total_col in df.columns:
            df[total_col] = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
        else:
            df[total_col] = 0.0
        if "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        if "due_date" in df.columns:
            df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
            df["term_days"] = (df["due_date"] - df.get("invoice_date")).dt.days
        if "invoice_paid_date" in df.columns:
            df["invoice_paid_date"] = pd.to_datetime(
                df["invoice_paid_date"], errors="coerce"
            )
            df["paid_days"] = (df["invoice_paid_date"] - df.get("invoice_date")).dt.days
        agg = df.groupby("supplier_id").agg(
            invoice_total_value=(total_col, "sum"),
            invoice_count=("invoice_id", "nunique"),
            last_invoice_date=("invoice_date", "max"),
            avg_payment_term_days=("term_days", "mean"),
            avg_paid_days=("paid_days", "mean"),
        )
        return agg.reset_index()

    def _summarise_invoice_lines(
        self, invoice_lines: pd.DataFrame, invoice_df: pd.DataFrame
    ) -> pd.DataFrame:
        if invoice_lines.empty or "invoice_id" not in invoice_lines.columns:
            return pd.DataFrame()
        if invoice_df.empty or "invoice_id" not in invoice_df.columns:
            return pd.DataFrame()
        invoice_lookup = self._map_supplier_ids(invoice_df, ("supplier_name",))
        join_cols = ["invoice_id"]
        extra_cols = [
            col for col in ("supplier_id", "supplier_name") if col in invoice_lookup.columns
        ]
        if not extra_cols:
            return pd.DataFrame()
        join_df = invoice_lookup[join_cols + extra_cols].drop_duplicates("invoice_id")
        df = invoice_lines.merge(join_df, on="invoice_id", how="left")
        df = self._map_supplier_ids(df, ("supplier_name",))
        if "supplier_id" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["supplier_id"]).copy()
        df["supplier_id"] = df["supplier_id"].astype(str)
        if "total_amount_incl_tax" in df.columns:
            df["total_amount_incl_tax"] = pd.to_numeric(
                df["total_amount_incl_tax"], errors="coerce"
            )
        agg = df.groupby("supplier_id").agg(
            invoice_item_count=("invoice_line_id", "nunique"),
        )
        return agg.reset_index()

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------
    def _build_supplier_profiles(
        self, tables: Dict[str, pd.DataFrame], supplier_ids: Iterable[str]
    ) -> Dict[str, Dict]:
        """Build supplier profiles with optimized batch processing."""
        suppliers = {str(s).strip() for s in supplier_ids if str(s).strip()}
        if not suppliers:
            return {}

        # First try batch fetch from cache/database
        profiles = self._batch_fetch_supplier_profiles(suppliers)
        
        # Process any missing suppliers
        flow = tables.get("procurement_flow", pd.DataFrame())
        if not flow.empty:
            missing_suppliers = suppliers - set(profiles.keys())
            if missing_suppliers:
                for supplier_id in missing_suppliers:
                    supplier_flow = flow[flow["supplier_id"] == supplier_id]
                    if not supplier_flow.empty:
                        profile = self._build_single_profile(supplier_flow)
                        if profile:
                            cache_key = f"profile:{supplier_id}"
                            self._cache[cache_key] = profile
                            profiles[supplier_id] = profile

        return profiles

    def _build_single_profile(self, supplier_flow: pd.DataFrame) -> Dict[str, Any]:
        """Build profile for a single supplier."""
        if supplier_flow.empty:
            return {}
            
        descriptions = [
            str(val).strip()
            for val in supplier_flow.get("item_description", pd.Series(dtype="object")).dropna()
            if str(val).strip()
        ]
        description_counter = Counter(descriptions)
        
        return {
            "supplier_id": supplier_flow["supplier_id"].iloc[0],
            "po_ids": sorted({str(val).strip() for val in supplier_flow.get("po_id", pd.Series(dtype="object")).dropna()}),
            "invoice_ids": sorted({str(val).strip() for val in supplier_flow.get("invoice_id", pd.Series(dtype="object")).dropna()}),
            "items": [val for val, _ in description_counter.most_common(5)],
            "primary_item": description_counter.most_common(1)[0][0] if description_counter else None,
            "categories": self._extract_categories(supplier_flow),
            "products": sorted({
                str(val).strip()
                for val in supplier_flow.get("product", pd.Series(dtype="object")).dropna()
                if str(val).strip()
            })
        }

    def _extract_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract category information efficiently."""
        categories = {}
        for level in range(1, 6):
            col = f"category_level_{level}"
            if col in df.columns:
                categories[col] = sorted({
                    str(val).strip()
                    for val in df[col].dropna()
                    if str(val).strip()
                })
        return categories

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _prepare_scoring_columns(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Prepare scoring columns with vector similarity if available."""
        result = df.copy()
        
        # Process regular scoring columns
        # ...existing code...

        # Add vector similarity scores if available
        if "vector_embedding" in result.columns:
            try:
                embeddings = np.stack(result["vector_embedding"].dropna())
                if len(embeddings) > 1:
                    similarities = np.dot(embeddings, embeddings.T)
                    np.fill_diagonal(similarities, 0)  # Exclude self-similarity
                    result["similarity_score"] = similarities.mean(axis=1)
                    result["final_score"] = result["final_score"] * (1 + result["similarity_score"] * 0.1)
            except Exception:
                logger.warning("Failed to compute vector similarities, continuing without them")

        return result

    def _score_categorical_criteria(
        self,
        df: pd.DataFrame,
        criteria: Iterable[str],
        policies: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        out = df.copy()
        policy = self._find_policy(policies, "CategoricalScoringPolicy")
        if not policy:
            return out
        rules = self._extract_policy_rules(policy)
        for crit in criteria:
            raw_col = crit
            score_col = f"{crit}_score"
            if raw_col in out.columns and crit in rules:
                mapping = rules[crit]
                out[score_col] = out[raw_col].map(mapping).fillna(mapping.get("default", 0))
        return out

    def _normalize_numeric_scores(self, df: pd.DataFrame, dirs: dict) -> pd.DataFrame:
        out = df.copy()
        for crit, direction in dirs.items():
            raw_col = crit
            score_col = f"{crit}_score"
            if raw_col not in df.columns:
                continue
            vals = pd.to_numeric(df[raw_col], errors="coerce")
            if vals.isna().all():
                out[score_col] = 0.0
                continue
            min_v, max_v = vals.min(), vals.max()
            if max_v - min_v == 0:
                out[score_col] = 10.0
            else:
                range_diff = float(max_v - min_v)
                if direction == "lower_is_better":
                    out[score_col] = 10 * (max_v - vals) / range_diff
                else:
                    out[score_col] = 10 * (vals - min_v) / range_diff
        return out

    def _normalise_weight_map(
        self, df: pd.DataFrame, weights: Dict[str, float]
    ) -> Dict[str, float]:
        available: Dict[str, float] = {}
        for crit, weight in weights.items():
            if weight <= 0:
                continue
            score_col = f"{crit}_score"
            raw_col = crit
            series = None
            if score_col in df.columns:
                series = pd.to_numeric(df[score_col], errors="coerce")
            elif raw_col in df.columns:
                series = pd.to_numeric(df[raw_col], errors="coerce")
            if series is None or series.isna().all():
                continue
            available[crit] = float(weight)
        if not available:
            return {}
        total = sum(available.values())
        if total <= 0:
            return {}
        return {crit: value / total for crit, value in available.items()}

    def _build_flow_index(
        self, payload: Any
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        flow_index: Dict[str, Dict[str, Any]] = {}
        name_index: Dict[str, Dict[str, Any]] = {}
        if not isinstance(payload, dict):
            return flow_index, name_index

        candidates: List[Any] = []
        graph = payload.get("graph")
        if isinstance(graph, dict):
            graph_flows = graph.get("supplier_flows") or []
            if isinstance(graph_flows, list):
                candidates.extend(graph_flows)
        direct_flows = payload.get("supplier_flows")
        if isinstance(direct_flows, list):
            candidates.extend(direct_flows)

        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            supplier_id = entry.get("supplier_id")
            if supplier_id is None:
                continue
            sid = str(supplier_id).strip()
            if not sid:
                continue
            flow_index[sid] = entry
            supplier_name = entry.get("supplier_name")
            token = self._normalise_supplier_token(supplier_name)
            if token:
                name_index.setdefault(token, entry)
        return flow_index, name_index

    def _alias_tokens_by_supplier(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for token, supplier_id in self._supplier_alias_map.items():
            if not supplier_id:
                continue
            mapping.setdefault(supplier_id, []).append(token)
        return mapping

    def _lookup_flow_entry(
        self,
        supplier_id: Optional[Any],
        supplier_name: Optional[Any],
        flow_index: Dict[str, Dict[str, Any]],
        flow_name_index: Dict[str, Dict[str, Any]],
        alias_tokens: Dict[str, List[str]],
    ) -> Optional[Dict[str, Any]]:
        sid_token = str(supplier_id).strip() if supplier_id is not None else ""
        if sid_token and sid_token in flow_index:
            return flow_index.get(sid_token)

        tokens: List[str] = []
        if supplier_name:
            token = self._normalise_supplier_token(supplier_name)
            if token:
                tokens.append(token)
        if sid_token and sid_token in self._supplier_lookup:
            lookup_name = self._supplier_lookup.get(sid_token)
            token = self._normalise_supplier_token(lookup_name)
            if token:
                tokens.append(token)
        for token in alias_tokens.get(sid_token, []):
            if token:
                tokens.append(token)

        for token in dict.fromkeys(tokens):
            flow = flow_name_index.get(token)
            if flow:
                return flow
        return None

    @staticmethod
    def _coverage_from_flow_entry(entry: Optional[Dict[str, Any]]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        coverage = entry.get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            value = float(coverage)
            if value < 0:
                return 0.0
            if value > 1.0:
                return 1.0
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

    def _annotate_flow_coverage(
        self,
        df: pd.DataFrame,
        flow_index: Dict[str, Dict[str, Any]],
        flow_name_index: Dict[str, Dict[str, Any]],
        alias_tokens: Dict[str, List[str]],
    ) -> pd.DataFrame:
        if df.empty or "supplier_id" not in df.columns:
            if "flow_coverage" not in df.columns:
                df["flow_coverage"] = 0.0
            return df
        annotated = df.copy()
        coverage_values: List[float] = []
        for _, row in annotated.iterrows():
            entry = self._lookup_flow_entry(
                row.get("supplier_id"),
                row.get("supplier_name"),
                flow_index,
                flow_name_index,
                alias_tokens,
            )
            coverage_values.append(self._coverage_from_flow_entry(entry))
        if "flow_coverage" in annotated.columns:
            baseline = (
                pd.to_numeric(annotated["flow_coverage"], errors="coerce")
                .fillna(0.0)
                .tolist()
            )
        else:
            baseline = [0.0] * len(annotated)
        combined = [max(base, cov) for base, cov in zip(baseline, coverage_values)]
        annotated["flow_coverage"] = combined
        return annotated

    def _apply_flow_bonus(
        self,
        df: pd.DataFrame,
        flow_index: Dict[str, Dict[str, Any]],
        flow_name_index: Dict[str, Dict[str, Any]],
        alias_tokens: Dict[str, List[str]],
    ) -> pd.DataFrame:
        if df.empty or "supplier_id" not in df.columns:
            if "flow_coverage" not in df.columns:
                df["flow_coverage"] = 0.0
            return df
        augmented = df.copy()
        coverage_values: List[float] = []
        for _, row in augmented.iterrows():
            entry = self._lookup_flow_entry(
                row.get("supplier_id"),
                row.get("supplier_name"),
                flow_index,
                flow_name_index,
                alias_tokens,
            )
            coverage = self._coverage_from_flow_entry(entry)
            coverage_values.append(coverage)
        augmented["flow_coverage"] = coverage_values
        if "final_score" in augmented.columns:
            augmented["final_score"] = augmented["final_score"] * (
                1.0 + augmented["flow_coverage"].fillna(0.0) * 0.1
            )
        return augmented

    def _generate_justification(self, row: pd.Series, criteria: Iterable[str]) -> str:
        if not self.justification_template:
            return "No justification template available."
        breakdown = []
        for crit in criteria:
            score_col = f"{crit}_score"
            if score_col in row:
                score_value = row.get(score_col)
                if isinstance(score_value, (int, float)):
                    breakdown.append(f"- {crit.replace('_', ' ').title()}: {score_value:.2f}")
                else:
                    breakdown.append(f"- {crit.replace('_', ' ').title()}: N/A")
        prompt = self.justification_template["prompt_template"].format(
            supplier_name=row.get("supplier_name", "Unknown"),
            final_score=row.get("final_score", 0.0),
            score_breakdown="\n".join(breakdown),
        )
        try:
            resp = self.call_ollama(prompt, model=self.settings.extraction_model)
            return resp.get("response", "").strip()
        except Exception:
            logger.exception("Justification generation failed")
            return "Justification generation failed."

    def _prepare_ranking_entry(
        self, row: pd.Series, profile: Optional[Dict], weights: Dict[str, float]
    ) -> Dict:
        entry = {
            "supplier_id": row.get("supplier_id"),
            "supplier_name": row.get("supplier_name"),
            "final_score": float(row.get("final_score", 0.0)),
            "price_score": row.get("price_score"),
            "delivery_score": row.get("delivery_score"),
            "risk_score": row.get("risk_score"),
            "payment_terms_score": row.get("payment_terms_score"),
            "payment_terms": row.get("payment_terms"),
            "avg_unit_price": row.get("avg_unit_price"),
            "total_spend": row.get("total_spend"),
            "po_count": row.get("po_count"),
            "invoice_count": row.get("invoice_count"),
            "lead_time_days": row.get("avg_lead_time_days"),
            "justification": row.get("justification"),
            "contact_name": row.get("contact_name_1"),
            "contact_email": row.get("contact_email_1"),
            "weights": dict(weights),
        }
        coverage = row.get("flow_coverage")
        if isinstance(coverage, (int, float)):
            entry["flow_coverage"] = float(coverage)
        relationship_cov = row.get("relationship_coverage")
        if isinstance(relationship_cov, (int, float)):
            entry["relationship_coverage"] = float(relationship_cov)
        relationship_summary = row.get("relationship_summary")
        if isinstance(relationship_summary, str) and relationship_summary.strip():
            entry["relationship_summary"] = relationship_summary.strip()
        relationship_statements = row.get("relationship_statements")
        if isinstance(relationship_statements, list) and relationship_statements:
            cleaned_statements = [
                statement.strip()
                for statement in relationship_statements
                if isinstance(statement, str) and statement.strip()
            ]
            if cleaned_statements:
                entry["relationship_statements"] = cleaned_statements
        if profile:
            entry.update(
                {
                    "po_ids": profile.get("po_ids", []),
                    "invoice_ids": profile.get("invoice_ids", []),
                    "primary_item": profile.get("primary_item"),
                    "items": profile.get("items", []),
                    "categories": profile.get("categories", {}),
                    "products": profile.get("products", []),
                }
            )
        return entry

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _mode_value(self, series: pd.Series) -> Optional[str]:
        try:
            cleaned = series.dropna().astype(str)
            if cleaned.empty:
                return None
            return cleaned.mode().iloc[0]
        except Exception:
            return None

    def _top_values(self, series: pd.Series, limit: int = 5) -> List[str]:
        cleaned = [
            str(val).strip()
            for val in series.dropna()
            if isinstance(val, str) and str(val).strip()
        ]
        counter = Counter(cleaned)
        return [val for val, _ in counter.most_common(limit)]

    def _fetch_relationship_context(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        service = getattr(self, "_relationship_service", None)
        if service is None or df.empty:
            return {}, {}

        contexts_by_id: Dict[str, Dict[str, Any]] = {}
        contexts_by_name: Dict[str, Dict[str, Any]] = {}
        seen: set[Tuple[str, str]] = set()

        for row in df.itertuples(index=False):
            sid = str(getattr(row, "supplier_id", "") or "").strip()
            name = getattr(row, "supplier_name", None)
            key = (sid, str(name or "").strip())
            if key in seen:
                continue
            seen.add(key)
            try:
                payloads = service.fetch_relationship(
                    supplier_id=sid or None,
                    supplier_name=name,
                    limit=1,
                )
            except Exception:
                logger.exception("Failed to load relationship context for supplier %s", sid or name)
                continue
            if not payloads:
                continue
            payload = payloads[0]
            if sid:
                contexts_by_id[sid] = payload
            token = self._normalise_supplier_token(name)
            if token:
                contexts_by_name[token] = payload

        return contexts_by_id, contexts_by_name

    def _merge_relationship_context(
        self,
        df: pd.DataFrame,
        contexts_by_id: Dict[str, Dict[str, Any]],
        contexts_by_name: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        if df.empty:
            return df

        augmented = df.copy()
        summaries: List[Optional[str]] = []
        statements_list: List[List[str]] = []
        coverage_values: List[Optional[float]] = []

        for _, row in augmented.iterrows():
            sid = str(row.get("supplier_id") or "").strip()
            name = row.get("supplier_name")
            payload = contexts_by_id.get(sid)
            if payload is None:
                token = self._normalise_supplier_token(name)
                if token:
                    payload = contexts_by_name.get(token)

            if isinstance(payload, dict):
                summary = payload.get("summary") or payload.get("content")
                summaries.append(summary.strip() if isinstance(summary, str) else None)
                rel_statements_raw = payload.get("relationship_statements")
                if isinstance(rel_statements_raw, list):
                    cleaned = [
                        statement.strip()
                        for statement in rel_statements_raw
                        if isinstance(statement, str) and statement.strip()
                    ]
                else:
                    cleaned = []
                statements_list.append(cleaned)
                coverage_raw = payload.get("coverage_ratio")
                try:
                    coverage_values.append(float(coverage_raw))
                except (TypeError, ValueError):
                    coverage_values.append(None)
            else:
                summaries.append(None)
                statements_list.append([])
                coverage_values.append(None)

        augmented["relationship_summary"] = summaries
        augmented["relationship_statements"] = statements_list

        if "flow_coverage" in augmented.columns:
            merged_coverage: List[Optional[float]] = []
            for base, extra in zip(augmented["flow_coverage"], coverage_values):
                if isinstance(extra, (int, float)):
                    base_val = float(base) if isinstance(base, (int, float)) else 0.0
                    merged_coverage.append(max(base_val, float(extra)))
                else:
                    merged_coverage.append(
                        float(base) if isinstance(base, (int, float)) else 0.0
                    )
            augmented["flow_coverage"] = merged_coverage
        else:
            augmented["flow_coverage"] = [
                float(value) if isinstance(value, (int, float)) else 0.0
                for value in coverage_values
            ]
        augmented["relationship_coverage"] = [
            float(value) if isinstance(value, (int, float)) else None
            for value in coverage_values
        ]

        return augmented

