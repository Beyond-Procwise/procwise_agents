import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
import re
from pathlib import Path
import os

import pandas as pd
import numpy as np
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)

# Ensure GPU is enabled where available
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class ProcessRoutingService:
    """Service to log agent processing state into proc.routing table."""

    DEFAULT_LLM_MODEL = "gpt-oss"

    STATUS_SUCCESS_TOKENS = {
        "completed",
        "complete",
        "success",
        "successful",
        "succeeded",
        "done",
        "ok",
        "okay",
        "pass",
        "passed",
        "finished",
        "resolved",
        "true",
        "yes",
    }
    STATUS_FAILURE_TOKENS = {
        "failed",
        "failure",
        "error",
        "errored",
        "exception",
        "timeout",
        "timed_out",
        "cancelled",
        "canceled",
        "aborted",
        "rejected",
        "denied",
        "false",
        "no",
    }

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self._agent_defaults_cache: Optional[Dict[str, Any]] = None
        self._agent_property_cache_by_id: Dict[str, Dict[str, Any]] = {}
        self._agent_type_cache_by_id: Dict[str, str] = {}
        self._prompt_id_catalog: set[int] = set()
        self._policy_id_catalog: set[int] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Best effort serialization for complex objects.

        Converts pandas objects, numpy types and dataclasses into plain
        Python data structures so they can be dumped to JSON. Falls back to
        ``str(obj)`` for unknown types.
        """
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.generic):
            return obj.item()
        if is_dataclass(obj):
            return asdict(obj)
        return str(obj)

    @classmethod
    def _safe_dumps(cls, data: Any) -> str:
        """JSON dump that tolerates DataFrames and other complex objects."""
        # First perform a best-effort sanitisation pass to replace NaN/NaT
        # values (which are invalid JSON tokens like ``NaN``) with ``null``
        # so PostgreSQL accepts the payload when inserting into ``json``/``jsonb``
        # columns. We also convert pandas structures and numpy scalars to
        # plain Python types.
        def _sanitize(obj: Any):
            # Handle pandas DataFrame / Series directly
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            # Numpy scalar -> native
            if isinstance(obj, np.generic):
                return _sanitize(obj.item())
            # Decimal values should be converted to native python numbers and
            # normalised when they represent non-finite values such as NaN or
            # infinity which PostgreSQL refuses inside JSON payloads.
            if isinstance(obj, Decimal):
                if not obj.is_finite():
                    return None
                return _sanitize(float(obj))
            # Basic python floats can still sneak in from math.nan or manual
            # calculations outside of pandas/numpy.
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            # Dataclasses -> dict
            if is_dataclass(obj):
                return _sanitize(asdict(obj))
            # None
            if obj is None:
                return None
            # Pandas/Numpy NA-like values -> None
            try:
                if pd.isna(obj):
                    return None
            except Exception:
                pass
            # Dicts and iterables
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_sanitize(v) for v in obj]
            # Fallback for other types: leave as-is and let json.dumps handle
            return obj

        sanitized = _sanitize(data)
        # Use our serializer for remaining complex objects (if any)
        return json.dumps(
            sanitized,
            default=cls._serialize,
            ensure_ascii=False,
            allow_nan=False,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_process_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure ``process_details`` conforms to the expected schema.

        The schema is:

        ``{"status": str, "agents": [{"agent": str, "dependencies": {"onSuccess": [],
        "onFailure": [], "onCompletion": []}, "status": str, "agent_ref_id": str}]}``

        Missing fields are populated with sensible defaults.
        """
        details = details.copy() if isinstance(details, dict) else {}
        # Default the overall workflow status to ``saved`` when missing or
        # empty so that callers can reason about progress using explicit
        # textual states rather than numeric placeholders.
        status = details.get("status")
        details["status"] = status if status else "saved"
        agents = []
        for agent in details.get("agents", []) or []:
            if not isinstance(agent, dict):
                continue
            deps = agent.get("dependencies") or {}
            deps.setdefault("onSuccess", [])
            deps.setdefault("onFailure", [])
            deps.setdefault("onCompletion", [])
            agent["dependencies"] = deps
            agent.setdefault("status", "saved")
            agent.setdefault("agent_ref_id", str(uuid.uuid4()))
            agents.append(agent)
        details["agents"] = agents
        return details

    @classmethod
    def classify_completion_status(
        cls, status: Any
    ) -> tuple[int, str, bool]:
        """Normalise a terminal workflow status into numeric and textual forms.

        The returned tuple is ``(numeric, label, recognised)`` where ``numeric`` is
        ``1`` for a completed flow and ``-1`` for a failed flow, ``label`` is the
        human-readable status string, and ``recognised`` indicates whether the
        input matched a known success/failure token rather than being coerced via
        a numeric fallback.
        """
        if isinstance(status, bool):
            return (1 if status else -1, "completed" if status else "failed", True)

        if isinstance(status, (np.integer, int)) and not isinstance(status, bool):
            numeric = int(status)
            return (1 if numeric > 0 else -1, "completed" if numeric > 0 else "failed", True)

        if isinstance(status, Decimal):
            if not status.is_finite():
                return (-1, "failed", False)
            numeric = float(status)
            return (1 if numeric > 0 else -1, "completed" if numeric > 0 else "failed", True)

        if isinstance(status, (float, np.floating)):
            if math.isnan(status):
                return (-1, "failed", False)
            return (1 if status > 0 else -1, "completed" if status > 0 else "failed", True)

        if isinstance(status, str):
            token = status.strip().lower()
            if not token:
                return (-1, "failed", False)
            if token in cls.STATUS_SUCCESS_TOKENS:
                return (1, "completed", True)
            if token in cls.STATUS_FAILURE_TOKENS:
                return (-1, "failed", True)
            try:
                numeric = float(token)
            except ValueError:
                return (-1, "failed", False)
            if math.isnan(numeric):
                return (-1, "failed", False)
            return (1 if numeric > 0 else -1, "completed" if numeric > 0 else "failed", True)

        return (-1, "failed", False)



    @staticmethod
    def _coerce_identifier_list(value: Any) -> List[int]:
        """Return a deduplicated list of integer identifiers."""

        if value is None:
            return []

        if isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]

        resolved: List[int] = []

        for item in items:
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                resolved.extend(ProcessRoutingService._coerce_identifier_list(item))
                continue
            if isinstance(item, dict):
                for key in ("promptId", "policyId", "id"):
                    if key in item:
                        resolved.extend(
                            ProcessRoutingService._coerce_identifier_list(item[key])
                        )
                        break
                continue
            if isinstance(item, str):
                token = item.strip()
                if not token:
                    continue
                if "," in token:
                    resolved.extend(
                        ProcessRoutingService._coerce_identifier_list(token.split(","))
                    )
                    continue
                try:
                    number = int(token)
                except ValueError:
                    continue
            else:
                try:
                    number = int(item)
                except (TypeError, ValueError):
                    continue
            resolved.append(number)

        deduped: List[int] = []
        seen: set[int] = set()
        for number in resolved:
            if number in seen:
                continue
            seen.add(number)
            deduped.append(number)
        return deduped

    def _resolve_agent_lookup_key(self, agent_ref_id: Any) -> Optional[str]:
        """Return the canonical agent identifier for ``agent_ref_id``."""

        if agent_ref_id is None:
            return None

        token = str(agent_ref_id).strip()
        if not token:
            return None

        caches = [
            getattr(self, "_agent_property_cache_by_id", {}) or {},
            getattr(self, "_agent_type_cache_by_id", {}) or {},
        ]

        def _match(candidate: str) -> Optional[str]:
            if not candidate:
                return None
            for cache in caches:
                if candidate in cache:
                    return candidate
            return None

        direct = _match(token)
        if direct:
            return direct

        if "_" in token:
            base = token.rsplit("_", 1)[0]
            base_match = _match(base)
            if base_match:
                return base_match

        return None

    @classmethod
    def _extract_llm_name(cls, value: Any) -> Optional[str]:
        """Best-effort extraction of an LLM name from arbitrary payloads."""

        if value is None:
            return None

        if isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            parts = token.split(":", 1)
            return parts[0].strip() or token

        if isinstance(value, dict):
            for key in ("llm", "name", "model", "model_name", "id", "value"):
                if key in value:
                    candidate = cls._extract_llm_name(value.get(key))
                    if candidate:
                        return candidate
            return None

        if isinstance(value, (list, tuple, set)):
            for item in value:
                candidate = cls._extract_llm_name(item)
                if candidate:
                    return candidate
            return None

        token = str(value).strip()
        if not token:
            return None
        parts = token.split(":", 1)
        return parts[0].strip() or token

    @classmethod
    def _normalise_agent_properties(
        cls, props: Optional[Dict[str, Any]], apply_default: bool = True
    ) -> Dict[str, Any]:
        """Clean agent property payloads coming from persisted workflows."""

        cleaned: Dict[str, Any] = {}
        if isinstance(props, dict):
            for key, value in props.items():
                if key == "memory":
                    continue
                cleaned[key] = value

        llm_candidate = cls._extract_llm_name(cleaned.get("llm"))
        if not llm_candidate:
            for alt_key in ("model", "model_name", "deployment", "engine"):
                llm_candidate = cls._extract_llm_name(cleaned.get(alt_key))
                if llm_candidate:
                    break
        if llm_candidate:
            cleaned["llm"] = llm_candidate
        elif apply_default:
            cleaned["llm"] = cls.DEFAULT_LLM_MODEL
        else:
            cleaned["llm"] = None

        cleaned["prompts"] = cls._coerce_identifier_list(cleaned.get("prompts"))
        cleaned["policies"] = cls._coerce_identifier_list(cleaned.get("policies"))

        return cleaned

    @staticmethod
    def _canonical_key(raw: str, agent_defs: Dict[str, str]) -> Optional[str]:
        """Normalize a raw agent identifier to a known registry key.

        Handles identifiers that include numeric suffixes or runtime prefixes
        (e.g. ``admin_quote_agent_000067``) by stripping those decorations and
        performing a best-effort match against the loaded agent definitions.
        ``None`` is returned when no suitable match can be found.
        """

        if not raw:
            return None

        key = re.sub(r"_[0-9]+(?:_[0-9]+)*$", "", raw)
        key = re.sub(r"^(?:admin|user|service)_", "", key)
        key = re.sub(r"_agent$", "", key)

        key_lower = key.lower()
        raw_lower = str(raw).lower()

        if key_lower in agent_defs:
            return key_lower
        if raw_lower in agent_defs:
            return raw_lower

        for slug, class_name in agent_defs.items():
            slug_lower = slug.lower()
            if slug_lower in {key_lower, raw_lower}:
                return slug
            if key_lower in slug_lower or slug_lower in key_lower:
                return slug
            class_lower = str(class_name or "").lower()
            if class_lower:
                if key_lower == class_lower or raw_lower == class_lower:
                    return slug
                if class_lower in key_lower or key_lower in class_lower:
                    return slug

        return None

    @classmethod
    def convert_agents_to_flow(cls, details: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy ``agents`` list into nested agent flow."""

        agents = details.get("agents") or []
        if not agents:
            return details

        # Maintain the original list order so that workflows run sequentially
        # when dependency metadata is missing or mis-specified.  Each agent is
        # indexed for quick lookups when explicit dependencies are provided.
        name_to_idx = {
            a.get("agent"): i
            for i, a in enumerate(agents)
            if isinstance(a, dict) and a.get("agent")
        }

        def _find_dependent(name: str, dep_key: str) -> Optional[int]:
            """Return the index of the agent that lists ``name`` as a dependency."""

            for i, agent in enumerate(agents):
                deps = (agent.get("dependencies") or {}).get(dep_key, [])
                if name in deps:
                    return i
            return None

        def _normalise_workflow_hint(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                candidate = value.strip()
            else:
                candidate = str(value).strip()
            return candidate or None

        default_workflow = _normalise_workflow_hint(details.get("workflow"))

        def build_from_index(
            idx: int,
            visited: Optional[set[int]] = None,
            inherited_workflow: Optional[str] = None,
        ) -> Dict[str, Any]:
            visited = visited or set()
            if idx in visited:
                return {}
            visited.add(idx)
            node = agents[idx]
            name = node.get("agent")
            raw_props = node.get(
                "agent_property", {"llm": None, "prompts": [], "policies": []}
            )
            props = dict(raw_props)

            llm_specified = False
            if isinstance(raw_props, dict):
                llm_specified = bool(cls._extract_llm_name(raw_props.get("llm")))

            workflow_hint = _normalise_workflow_hint(props.get("workflow"))
            if not workflow_hint:
                workflow_hint = _normalise_workflow_hint(node.get("workflow"))
            if not workflow_hint:
                workflow_hint = _normalise_workflow_hint(inherited_workflow)
            if not workflow_hint:
                workflow_hint = default_workflow
            if workflow_hint:
                props["workflow"] = workflow_hint
            else:
                props.pop("workflow", None)

            props = cls._normalise_agent_properties(props)

            flow = {
                "agent": name,
                "status": node.get("status", "saved"),
                "agent_type": str(node.get("agent_type", name or "")),
                "agent_property": props,
            }

            for identifier_key in ("agent_ref_id", "agent_id"):
                if node.get(identifier_key) is not None:
                    flow[identifier_key] = node.get(identifier_key)

            if not llm_specified:
                flow["_llm_from_default"] = True
            if workflow_hint:
                flow["workflow"] = workflow_hint
            deps = node.get("dependencies", {})

            # Determine next agents by finding nodes that depend on the current
            # one. Fall back to the current node's forward references for
            # backward compatibility, then to sequential ordering.
            nxt = _find_dependent(name, "onSuccess")
            if nxt is None and deps.get("onSuccess"):
                nxt = name_to_idx.get(deps["onSuccess"][0])
            inherited = workflow_hint or inherited_workflow
            if nxt is not None and nxt not in visited:
                flow["onSuccess"] = build_from_index(
                    nxt, visited.copy(), inherited
                )
            elif idx + 1 < len(agents) and (idx + 1) not in visited:
                flow["onSuccess"] = build_from_index(
                    idx + 1, visited.copy(), inherited
                )

            nxt = _find_dependent(name, "onFailure")
            if nxt is None and deps.get("onFailure"):
                nxt = name_to_idx.get(deps["onFailure"][0])
            if nxt is not None and nxt not in visited:
                flow["onFailure"] = build_from_index(
                    nxt, visited.copy(), inherited
                )

            nxt = _find_dependent(name, "onCompletion")
            if nxt is None and deps.get("onCompletion"):
                nxt = name_to_idx.get(deps["onCompletion"][0])
            if nxt is not None and nxt not in visited:
                flow["onCompletion"] = build_from_index(
                    nxt, visited.copy(), inherited
                )

            return flow

        # Always start from the first agent as defined in the list to ensure
        # determinism even when dependency links form a cycle or point
        # backwards.
        return build_from_index(0, inherited_workflow=default_workflow)

    # ------------------------------------------------------------------
    # Metadata enrichment
    # ------------------------------------------------------------------
    def _load_agent_links(self):
        """Fetch agent definitions and their linked prompts/policies."""

        agent_defs: Dict[str, str] = {}
        prompt_map: Dict[str, list[int]] = {}
        policy_map: Dict[str, list[int]] = {}
        default_props: Dict[str, Dict[str, Any]] = {}
        default_ts: Dict[str, datetime] = {}
        property_by_id: Dict[str, Dict[str, Any]] = {}
        type_by_id: Dict[str, str] = {}
        prompt_ids_catalog: set[int] = set()
        policy_ids_catalog: set[int] = set()

        # Load agent definitions from the bundled JSON file instead of the DB
        path = Path(__file__).resolve().parents[1] / "agent_definitions.json"
        with path.open() as f:
            data = json.load(f)
        for item in data:
            agent_class = item.get("agentType", "")
            if not agent_class:
                continue
            slug = re.sub(r"(?<!^)(?=[A-Z])", "_", agent_class).lower()
            if slug.endswith("_agent"):
                slug = slug[:-6]
            agent_defs[slug] = agent_class
            agent_defs[str(item.get("agentId"))] = agent_class

        def _coerce_timestamp(value: Any) -> datetime:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc)
            return datetime.min.replace(tzinfo=timezone.utc)

        def _parse_agent_property(payload: Any) -> Optional[Dict[str, Any]]:
            if payload is None:
                return None
            if isinstance(payload, dict):
                return dict(payload)
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode()
            if isinstance(payload, str):
                text = payload.strip()
                if not text:
                    return None
                try:
                    parsed = json.loads(text)
                except Exception:
                    logger.debug("Failed to parse agent_property JSON: %s", text)
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
            return None

        def _record_default(slug: Optional[str], props: Dict[str, Any], modified: Any, created: Any) -> None:
            if not slug or not isinstance(props, dict):
                return
            timestamp = _coerce_timestamp(modified or created)
            existing = default_ts.get(slug)
            if existing is not None and existing >= timestamp:
                return
            default_ts[slug] = timestamp
            default_props[slug] = props

        try:
            with self.agent_nick.get_db_connection() as conn:
                def _record(map_obj: Dict[str, list[int]], raw: str, pid: int):
                    slug = self._canonical_key(raw, agent_defs)
                    if slug:
                        map_obj.setdefault(slug, []).append(int(pid))
                    else:  # pragma: no cover - defensive logging
                        logger.warning(
                            "Agent type '%s' not found in definitions", raw
                        )

                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT prompt_id, prompt_linked_agents FROM proc.prompt"
                    )
                    for pid, linked in cursor.fetchall():
                        try:
                            prompt_ids_catalog.add(int(pid))
                        except Exception:
                            continue
                        for key in re.findall(r"[A-Za-z0-9_]+", str(linked or "")):
                            _record(prompt_map, key, pid)

                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT policy_id, policy_linked_agents FROM proc.policy"
                    )
                    for pid, linked in cursor.fetchall():
                        try:
                            policy_ids_catalog.add(int(pid))
                        except Exception:
                            continue
                        for key in re.findall(r"[A-Za-z0-9_]+", str(linked or "")):
                            _record(policy_map, key, pid)

                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT agent_id, agent_type, agent_name, agent_property, modified_time, created_time
                        FROM proc.agent
                        WHERE agent_property IS NOT NULL
                        """
                    )
                    for agent_id, agent_type, agent_name, props_payload, modified, created in cursor.fetchall():
                        parsed = _parse_agent_property(props_payload)
                        if not parsed:
                            continue

                        normalized = self._normalise_agent_properties(parsed)
                        slug: Optional[str] = None
                        canonical_type: Optional[str] = None
                        for candidate in (agent_type, agent_name):
                            if not candidate:
                                continue
                            slug = self._canonical_key(str(candidate), agent_defs)
                            if slug:
                                canonical_type = agent_defs.get(slug) or str(candidate)
                                break

                        if agent_id:
                            property_by_id[str(agent_id)] = dict(normalized)
                            if canonical_type:
                                type_by_id[str(agent_id)] = canonical_type
                            elif agent_type:
                                type_by_id[str(agent_id)] = str(agent_type)

                        if slug:
                            _record_default(slug, normalized, modified, created)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load agent linkage metadata")

        self._agent_defaults_cache = dict(default_props)
        self._agent_property_cache_by_id = property_by_id
        self._agent_type_cache_by_id = type_by_id
        self._prompt_id_catalog = prompt_ids_catalog
        self._policy_id_catalog = policy_ids_catalog
        return agent_defs, prompt_map, policy_map

    def _enrich_node(self, node, agent_defs, prompt_map, policy_map):
        """Recursively normalise agent types and attach prompt/policy IDs."""

        if not isinstance(node, dict):
            return
        agent_ref_id = node.get("agent_ref_id") or node.get("agent_id")
        db_props_payload: Dict[str, Any] = {}
        resolved_ref_key: Optional[str] = None
        db_agent_type: Optional[str] = None
        if agent_ref_id is not None:
            ref_key = str(agent_ref_id).strip()
            if ref_key:
                resolved_ref_key = self._resolve_agent_lookup_key(ref_key)
                lookup_key = resolved_ref_key or ref_key
                db_props_payload = dict(
                    getattr(self, "_agent_property_cache_by_id", {}).get(lookup_key) or {}
                )
                db_agent_type = getattr(self, "_agent_type_cache_by_id", {}).get(lookup_key)
                if db_agent_type and not node.get("agent_type"):
                    node["agent_type"] = db_agent_type
                if resolved_ref_key and node.get("agent_id") in (None, "", agent_ref_id):
                    node["agent_id"] = resolved_ref_key
        ref_for_log = resolved_ref_key or agent_ref_id

        raw_type = str(node.get("agent_type", "")).strip()
        base_key = self._canonical_key(raw_type, agent_defs)
        canonical_type: Optional[str] = None

        if base_key:
            canonical_type = agent_defs.get(base_key, raw_type)
        else:
            fallback_type: Optional[str] = db_agent_type
            if not fallback_type:
                lookup_key = self._resolve_agent_lookup_key(raw_type)
                if lookup_key:
                    fallback_type = getattr(self, "_agent_type_cache_by_id", {}).get(lookup_key)

            if fallback_type:
                fallback_slug = self._canonical_key(str(fallback_type), agent_defs)
                if fallback_slug:
                    base_key = fallback_slug
                    canonical_type = agent_defs.get(fallback_slug, str(fallback_type))
                else:
                    base_key = base_key or raw_type
                    canonical_type = str(fallback_type)
            else:
                base_key = raw_type

        if canonical_type:
            node["agent_type"] = canonical_type
        elif base_key:
            node["agent_type"] = agent_defs.get(base_key, raw_type)
        raw_props = node.get("agent_property", {"llm": None, "prompts": [], "policies": []})
        db_props = self._normalise_agent_properties(db_props_payload, apply_default=False)
        props = self._normalise_agent_properties(raw_props, apply_default=False)

        defaults_map = getattr(self, "_agent_defaults_cache", {}) or {}
        default_props = defaults_map.get(base_key) or {}
        merged_props: Dict[str, Any] = dict(default_props)

        llm_from_default = bool(node.pop("_llm_from_default", False))

        def _merge_into(
            target: Dict[str, Any],
            source: Dict[str, Any],
            skip_llm_override: bool = False,
        ) -> None:

            for key, value in source.items():
                if key in {"prompts", "policies"}:
                    continue
                if value is None:
                    continue
                if key == "llm" and skip_llm_override and target.get("llm") is not None:
                    continue
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    combined = dict(target[key])
                    combined.update(value)
                    target[key] = combined
                elif isinstance(value, dict):
                    target[key] = dict(value)
                else:
                    target[key] = value

        _merge_into(merged_props, db_props)
        _merge_into(merged_props, props, skip_llm_override=llm_from_default)

        for field in ("prompts", "policies"):
            combined_ids = set(self._coerce_identifier_list(default_props.get(field)))
            combined_ids.update(self._coerce_identifier_list(db_props.get(field)))
            combined_ids.update(self._coerce_identifier_list(props.get(field)))
            merged_props[field] = sorted(combined_ids)

        props = self._normalise_agent_properties(merged_props)
        props.pop("_llm_from_default", None)

        prompt_ids = set(props.get("prompts", []))
        prompt_ids.update(self._coerce_identifier_list(prompt_map.get(base_key)))
        props["prompts"] = sorted(prompt_ids)

        policy_ids = set(props.get("policies", []))
        policy_ids.update(self._coerce_identifier_list(policy_map.get(base_key)))
        props["policies"] = sorted(policy_ids)

        prompt_catalog = getattr(self, "_prompt_id_catalog", set()) or set()
        if prompt_catalog:
            invalid_prompts = [pid for pid in props.get("prompts", []) if pid not in prompt_catalog]
            if invalid_prompts:
                logger.warning(
                    "Discarding unknown prompt ids %s for agent %s", invalid_prompts, base_key or ref_for_log
                )
            props["prompts"] = [pid for pid in props.get("prompts", []) if pid in prompt_catalog]

        policy_catalog = getattr(self, "_policy_id_catalog", set()) or set()
        if policy_catalog:
            invalid_policies = [pid for pid in props.get("policies", []) if pid not in policy_catalog]
            if invalid_policies:
                logger.warning(
                    "Discarding unknown policy ids %s for agent %s", invalid_policies, base_key or ref_for_log
                )
            props["policies"] = [pid for pid in props.get("policies", []) if pid in policy_catalog]

        node["agent_property"] = props
        for branch in ["onSuccess", "onFailure", "onCompletion"]:
            if branch in node:
                self._enrich_node(node[branch], agent_defs, prompt_map, policy_map)

    def log_process(
        self,
        process_name: str,
        process_details: Dict[str, Any],
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        created_by: Optional[str] = None,
        process_status: Optional[int] = None,
    ) -> Optional[int]:
        """Insert a process routing record and return the new ``process_id``.

        The ``proc.routing`` table now serves as both the catalogue of
        processes and the repository for individual run details.  An initial
        ``process_status`` may be supplied to seed the run state, while
        subsequent status updates are handled via :meth:`update_process_status`
        or :meth:`log_run_detail`.
        """

        # ``process_status`` is mandatory in ``proc.routing``. Default to ``0``
        # (started) whenever a caller does not explicitly provide a value to
        # avoid ``NULL`` constraint violations.
        process_status = 0 if process_status is None else process_status

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO proc.routing
                            (process_name, process_details, created_by, user_id, user_name, process_status)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING process_id
                        """,
                        (
                            process_name,
                            self._safe_dumps(self.normalize_process_details(process_details)),
                            created_by or self.settings.script_user,
                            user_id,
                            user_name,
                            process_status,
                        ),
                    )
                    process_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(
                        "Logged process %s with id %s", process_name, process_id
                    )
                    return process_id
        except Exception:
            logger.exception("Failed to log process %s", process_name)
            return None

    def get_process_details(self, process_id: int, raw: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch the ``process_details`` blob for a given ``process_id``.

        When ``raw`` is ``True`` the stored JSON is returned without converting
        the ``agents`` array into a nested flow or enriching agent metadata.
        This is useful for direct mutations where the original structure must
        be preserved (e.g. updating individual agent statuses).
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT process_details FROM proc.routing WHERE process_id = %s",
                        (process_id,),
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        value = row[0]
                        if isinstance(value, (str, bytes, bytearray)):
                            value = json.loads(value)
                        details = self.normalize_process_details(value)
                        if not raw:
                            if "agent_type" not in details and "agents" in details:
                                details = self.convert_agents_to_flow(details)
                            agent_defs, prompt_map, policy_map = self._load_agent_links()
                            self._enrich_node(details, agent_defs, prompt_map, policy_map)
                        return details
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to fetch process %s", process_id)
        return None

    def update_process_details(
        self,
        process_id: int,
        process_details: Dict[str, Any],
        modified_by: Optional[str] = None,
    ) -> None:
        """Persist updated ``process_details`` for a process."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_details = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (
                            self._safe_dumps(self.normalize_process_details(process_details)),
                            modified_by or self.settings.script_user,
                            process_id,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Updated process %s details", process_id
                    )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to update details for process %s", process_id
            )

    def update_agent_status(
        self,
        process_id: int,
        agent_name: str,
        status: str,
        modified_by: Optional[str] = None,
    ) -> None:
        """Update the status for a specific agent within ``process_details``.

        The overall structure of the ``process_details`` column is preserved by
        normalising the payload and only mutating the targeted agent's
        ``status`` field.
        """
        details = self.get_process_details(process_id, raw=True)
        if not details or "agents" not in details:
            logger.warning(
                "No process_details found for process %s; skipping agent status update",
                process_id,
            )
            return

        agents = [a for a in details.get("agents", []) if isinstance(a, dict)]
        order = [a.get("agent") for a in agents]
        if agent_name not in order:
            logger.warning(
                "Agent %s not found in process %s; skipping", agent_name, process_id
            )
            return
        idx = order.index(agent_name)

        dependency_keys = ("onSuccess", "onFailure", "onCompletion")
        agent_lookup = {a.get("agent"): a for a in agents if a.get("agent")}

        target = agent_lookup.get(agent_name, {})
        target_deps = target.get("dependencies") or {}

        blocking_agents: set[str] = set()
        for key in dependency_keys:
            for dep_name in target_deps.get(key, []) or []:
                if dep_name:
                    blocking_agents.add(dep_name)

        if not blocking_agents:
            # If the workflow does not define any dependency metadata, retain
            # the previous sequential guard as a conservative fallback.
            has_dependency_metadata = any(
                (agent.get("dependencies") or {}).get(key)
                for agent in agents
                for key in dependency_keys
            )
            if not has_dependency_metadata:
                blocking_agents.update(
                    a.get("agent")
                    for a in agents[:idx]
                    if a.get("agent") and a.get("agent") != agent_name
                )

        for dep_name in blocking_agents:
            dependent = agent_lookup.get(dep_name)
            if dependent and dependent.get("status") == "saved":
                raise ValueError(
                    f"Cannot update {agent_name} before {dep_name} starts"
                )
        agents[idx]["status"] = status

        # Derive the overall workflow status based on individual agent states.
        statuses = [a.get("status") for a in agents]
        if any(s == "failed" for s in statuses):
            overall = "failed"
        elif statuses and all(s == "completed" for s in statuses):
            overall = "completed"
        elif any(s != "saved" for s in statuses):
            overall = "running"
        else:
            overall = "saved"

        details["agents"] = agents
        details["status"] = overall

        # Persist the updated details and synchronise the top-level
        # ``process_status`` flag when the workflow reaches a terminal state.
        if overall in ("completed", "failed"):
            numeric = 1 if overall == "completed" else -1
            self.update_process_status(
                process_id,
                numeric,
                modified_by,
                details,
            )
        else:
            self.update_process_details(process_id, details, modified_by)

    def update_process_status(
        self,
        process_id: int,
        status: Any,
        modified_by: Optional[str] = None,
        process_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update process status in ``proc.routing`` and keep ``process_details`` in sync.

        ``status`` may be supplied as the numeric flag, a textual token such as
        ``"completed"``/``"failed"`` or a percentage value emitted by the
        orchestrator. Values are normalised to the terminal flags accepted by
        ``proc.routing`` before persistence."""
        numeric_status, status_text, recognised = self.classify_completion_status(status)
        if not recognised:
            logger.warning(
                "Updated process %s to unrecognised status %s - coercing to %s",
                process_id,
                status,
                numeric_status,
            )
        # Ensure the ``process_details`` blob reflects the new status.
        details = process_details or self.get_process_details(process_id, raw=True) or {}
        details["status"] = status_text
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_status = %s,
                            process_details = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (
                            numeric_status,
                            self._safe_dumps(self.normalize_process_details(details)),
                            modified_by or self.settings.script_user,
                            process_id,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Updated process %s to status %s (process_status=%s)",
                        process_id,
                        status_text,
                        numeric_status,
                    )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to update status for process %s", process_id
            )


    def log_action(
        self,
        process_id: int,
        agent_type: str,
        action_desc: Any,
        process_output: Optional[Any] = None,
        status: str = "validated",
        action_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Insert or update an action record in ``proc.action``.

        If ``action_id`` is provided, the existing record is updated with the
        new ``process_output`` and ``status``. Otherwise a new record is
        inserted and its ``action_id`` returned.
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    if not action_id:
                        action_id = str(uuid.uuid4())
                        run_id = run_id or str(uuid.uuid4())
                        cursor.execute(
                            """
                            INSERT INTO proc.action (
                                action_id, process_id, run_id, agent_type,
                                process_output, status, action_desc, action_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                action_id,
                                str(process_id),
                                run_id,
                                agent_type,
                                self._safe_dumps(process_output) if process_output is not None else None,
                                status,
                                action_desc if isinstance(action_desc, str) else self._safe_dumps(action_desc),
                                datetime.now(timezone.utc),
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            UPDATE proc.action
                            SET process_output = %s,
                                status = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE action_id = %s
                            """,
                            (
                                self._safe_dumps(process_output) if process_output is not None else None,
                                status,
                                action_id,
                            ),
                        )
                    conn.commit()
                    logger.info(
                        "Logged action %s for process %s with status %s",
                        action_id,
                        process_id,
                        status,
                    )
                    return action_id
        except Exception:
            logger.exception(
                "Failed to log action for process %s", process_id
            )
            return None

    def log_run_detail(
            self,
            process_id: int,
            process_status: str,
            run_id: Optional[str] = None,
            process_details: Optional[Dict[str, Any]] = None,
            process_start_ts: Optional[datetime] = None,
            process_end_ts: Optional[datetime] = None,
            triggered_by: Optional[str] = None,
    ) -> Optional[str]:


        def _annotate_process_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            """Normalise ``details`` without altering workflow status.

            The per-agent statuses are managed via :meth:`update_agent_status`.
            When logging run details we simply ensure the payload adheres to the
            expected schema while preserving any existing top-level ``status``
            value so that the overall workflow state remains intact.
            """
            return self.normalize_process_details(details)

        run_id = run_id or str(uuid.uuid4())
        process_start_ts = process_start_ts or datetime.now(timezone.utc)
        process_end_ts = process_end_ts or datetime.now(timezone.utc)
        duration = (
            process_end_ts - process_start_ts
            if process_end_ts and process_start_ts
            else None
        )

        # Normalize textual input
        ps = str(process_status).lower() if process_status is not None else ""

        # Determine integer status:
        # - kickoff/ongoing -> 0
        # - success -> 1
        # - failure/unknown -> -1
        success_vals = ("1", "success", "completed", "done")
        kickoff_vals = ("saved", "validated")
        if ps in success_vals:
            status_int = 1
        elif ps in kickoff_vals:
            status_int = 0
        else:
            status_int = -1

        # Map textual status for raw payload with three distinct states
        if ps == "started":
            status_text = "started"
        elif ps in ("running", "in_progress", "validating"):
            status_text = "validated"
        elif ps in success_vals:
            status_text = "completed"
        else:
            status_text = "failed" if status_int < 0 else "completed"

        # Normalise process_details without mutating the existing workflow status
        annotated_details = _annotate_process_details(process_details)

        raw_payload = {
            "run_id": run_id,
            "process_start_ts": process_start_ts.isoformat(),
            "process_end_ts": process_end_ts.isoformat(),
            "duration": duration.total_seconds() if duration else None,
            "status": status_text,
            "triggered_by": triggered_by or self.settings.script_user,
        }

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_status = %s,
                            process_details = %s,
                            raw_data = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (
                            status_int,
                            self._safe_dumps(annotated_details) if annotated_details is not None else None,
                            self._safe_dumps(raw_payload),
                            triggered_by or self.settings.script_user,
                            process_id,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Recorded run %s for process %s with status %s",
                        run_id,
                        process_id,
                        status_int,
                    )
                    return run_id
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to log run for process %s", process_id
            )
            return None
