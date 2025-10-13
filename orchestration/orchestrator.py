import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
import os
from functools import lru_cache
from jinja2 import Template

try:  # Optional dependency for JSONPath mapping
    from jsonpath_ng import parse as jsonpath_parse
except Exception:  # pragma: no cover - library may be absent in tests
    jsonpath_parse = None

from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from engines.query_engine import QueryEngine
from services.process_routing_service import ProcessRoutingService
from services.backend_scheduler import BackendScheduler
from services.event_bus import get_event_bus, workflow_scope
from services.agent_manifest import AgentManifestService
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

# Ensure GPU is enabled when available
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


class Orchestrator:

    # Default workflow hints for agents that historically relied on implicit
    # orchestration metadata.  Older process definitions for the opportunity
    # miner never stored an explicit ``workflow`` value which meant the agent
    # started failing once the policy checks were tightened.  Providing a
    # canonical fallback keeps those legacy flows operational while newer
    # definitions can continue to supply their own workflow names.
    WORKFLOW_DEFAULTS: Dict[str, str] = {}

    WORKFLOW_ALIASES: Dict[str, str] = {}

    WORKFLOW_DEFAULT_CONDITIONS = {
        "contract_expiry_check": {"negotiation_window_days": 90},

    }

    # Token level aliases help resolve historical agent identifiers that were
    # generated dynamically (e.g. ``quotes_agent``) to their canonical
    # registry entries. This keeps fuzzy matching deterministic even when new
    # agents such as ``QuoteComparisonAgent`` introduce additional "quote"
    # slugs into the registry.
    AGENT_TOKEN_ALIASES = {
        "quotes": "quote_evaluation",
        "quote": "quote_evaluation",
        "comparison": "quote_comparison",
        "comparisons": "quote_comparison",
    }

    def __init__(self, agent_nick, *, training_endpoint=None):
        # Ensure GPU environment is initialised before any agent execution.
        # ``configure_gpu`` is idempotent so repeated calls are safe and allow
        # both the orchestrator and individual agents to run on the same
        # device. This provides a centralised location for enabling GPU usage
        # across the agentic framework.
        configure_gpu()

        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.agents = agent_nick.agents
        self.policy_engine = agent_nick.policy_engine
        self.query_engine = agent_nick.query_engine
        self.routing_engine = agent_nick.routing_engine
        self.routing_model = self.routing_engine.routing_model
        self.executor = ThreadPoolExecutor(max_workers=self.settings.max_workers)
        self.model_training_endpoint = training_endpoint
        self.backend_scheduler = BackendScheduler.ensure(
            agent_nick, training_endpoint=training_endpoint
        )
        self.event_bus = get_event_bus()
        self.manifest_service = AgentManifestService(agent_nick)
        self._prompt_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self._policy_cache: Optional[Dict[int, Dict[str, Any]]] = None

    def execute_ranking_flow(self, query: str) -> Dict:
        """Public wrapper for the supplier ranking workflow.

        The policy engine requires ranking criteria to be supplied for
        validation.  When only a free-text query is provided we attempt to
        derive the criteria by matching known policy terms within the query.
        If no explicit matches are found we fall back to the full set of
        default policy criteria so that the workflow still proceeds with
        reasonable defaults.
        """

        weight_policy = next(
            (p for p in self.policy_engine.supplier_policies if p.get("policyName") == "WeightAllocationPolicy"),
            {},
        )
        default_weights = (
            weight_policy.get("details", {}).get("rules", {}).get("default_weights", {})
        )
        lower_query = query.lower()
        criteria = [c for c in default_weights.keys() if c in lower_query] or list(
            default_weights.keys()
        )
        intent = {"parameters": {"criteria": criteria}}
        input_data = {"query": query, "criteria": criteria, "intent": intent}
        return self.execute_workflow("supplier_ranking", input_data)

    def execute_ranking_workflow(self, query: str) -> Dict:
        """Backward compatible alias for :meth:`execute_ranking_flow`."""
        return self.execute_ranking_flow(query)

    def execute_extraction_flow(
        self,
        s3_prefix: Optional[str] = None,
        s3_object_key: Optional[str] = None,
    ) -> Dict:
        """Public wrapper for the document extraction workflow."""
        return self.execute_workflow(
            "document_extraction",
            {"s3_prefix": s3_prefix, "s3_object_key": s3_object_key},
        )

    def execute_extraction_workflow(
        self,
        s3_prefix: Optional[str] = None,
        s3_object_key: Optional[str] = None,
    ) -> Dict:
        """Backward compatible alias for :meth:`execute_extraction_flow`."""
        return self.execute_extraction_flow(s3_prefix, s3_object_key)

    def execute_workflow(
        self, workflow_name: str, input_data: Dict, user_id: str = None
    ) -> Dict:
        """Execute a complete workflow"""
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting workflow {workflow_name} with ID {workflow_id}")

        context: Optional[AgentContext] = None
        enriched_input: Dict[str, Any] = {}

        try:
            # Create initial context
            enriched_input = {**(input_data or {})}
            self._ensure_workflow_metadata(enriched_input, workflow_name)
            manifest = self.manifest_service.build_manifest(workflow_name)
            if isinstance(enriched_input, dict):
                enriched_input.setdefault("agent_manifest", manifest)
                enriched_input.setdefault(
                    "policy_context", manifest.get("policies", [])
                )
                enriched_input.setdefault(
                    "knowledge_context", manifest.get("knowledge", {})
                )
            context = AgentContext(
                workflow_id=workflow_id,
                agent_id=workflow_name,
                user_id=user_id or self.settings.script_user,
                input_data=enriched_input,
                task_profile=manifest.get("task", {}),
                policy_context=manifest.get("policies", []),
                knowledge_base=manifest.get("knowledge", {}),
            )
            context.apply_manifest(manifest)

            # Validate against policies
            if not self._validate_workflow(workflow_name, context):
                return {
                    "status": "blocked",
                    "reason": "Policy validation failed",
                    "workflow_id": workflow_id,
                }

            # Execute workflow based on type
            if workflow_name == "document_extraction":
                result = self._execute_extraction_workflow(context)
            elif workflow_name == "supplier_ranking":
                result = self._execute_ranking_workflow(context)
            elif workflow_name == "quote_evaluation":
                result = self._execute_quote_workflow(context)
            elif workflow_name == "opportunity_mining":
                result = self._execute_opportunity_workflow(context)
            else:
                result = self._execute_generic_workflow(workflow_name, context)

            self._publish_workflow_complete(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                context=context,
                result=result,
                status="completed",
            )

            if self._workflow_completed_successfully(result):
                self._trigger_automatic_training(workflow_name)
            else:
                logger.info(
                    "Skipping automatic training for %s due to unsuccessful workflow outcome",
                    workflow_name,
                )

            return {
                "status": "completed",
                "workflow_id": workflow_id,
                "result": result,
                "execution_path": context.routing_history,
            }

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            if context is None:
                fallback_input: Dict[str, Any] = {}
                if isinstance(enriched_input, dict):
                    fallback_input = dict(enriched_input)
                elif isinstance(input_data, dict):
                    fallback_input = dict(input_data)
                context = AgentContext(
                    workflow_id=workflow_id,
                    agent_id=workflow_name,
                    user_id=user_id or self.settings.script_user,
                    input_data=fallback_input,
                )
            self._publish_workflow_complete(
                workflow_name=workflow_name,
                workflow_id=workflow_id,
                context=context,
                result={"error": str(e)},
                status="failed",
            )
            return {"status": "failed", "workflow_id": workflow_id, "error": str(e)}

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_agent_definitions() -> Dict[str, str]:
        """Return mapping of ``agent_type`` identifiers to agent class names.

        The JSON file is read once and cached for subsequent calls.  This
        prevents repeated disk I/O when resolving linked agents from prompts or
        policies, which can occur many times within a single workflow.
        """

        path = Path(__file__).resolve().parents[1] / "agent_definitions.json"
        with path.open() as f:
            data = json.load(f)

        defs: Dict[str, str] = {}
        for item in data:
            agent_class = item.get("agentType", "")
            if not agent_class:
                continue
            # Create lookups based on the ``agentType`` field rather than the
            # legacy numeric ``agentId``.  This matches the values supplied in
            # ``*_linked_agents`` columns where ``agent_type`` identifiers are
            # stored.  Numeric IDs are retained only for backward
            # compatibility.
            slug = Orchestrator._resolve_agent_name(agent_class)
            defs[slug] = agent_class
            defs[str(item.get("agentId"))] = agent_class

        return defs

    @staticmethod
    def _canonical_key(raw_key: str, agent_defs: Dict[str, str]) -> Optional[str]:
        """Return the registry slug for a potentially decorated agent key."""

        if not raw_key:
            return None

        # Strip runtime suffixes/prefixes and normalise for comparison
        key = re.sub(r"_[0-9]+(?:_[0-9]+)*$", "", raw_key)
        key = re.sub(r"^(?:admin|user|service)_", "", key)
        key = re.sub(r"_agent$", "", key)

        key_lower = key.lower()
        tokens_raw = [t for t in re.split(r"[_]+", key_lower) if t]

        for token in tokens_raw:
            alias = Orchestrator.AGENT_TOKEN_ALIASES.get(token)
            if alias and alias in agent_defs:
                return alias
            if token.endswith("s"):
                singular = token[:-1]
                alias = Orchestrator.AGENT_TOKEN_ALIASES.get(singular)
                if alias and alias in agent_defs:
                    return alias
        if key_lower in agent_defs:
            return key_lower

        raw_lower = raw_key.lower()
        if raw_lower in agent_defs:
            return raw_lower

        # Attempt to resolve CamelCase class names into registry slugs
        resolved = Orchestrator._resolve_agent_name(raw_key)
        if resolved in agent_defs:
            return resolved

        for slug in agent_defs:
            if slug in key_lower or key_lower in slug:
                return slug

        tokens: List[str] = []
        for t in tokens_raw:
            if len(t) <= 2 or t in {"agent", "test", "keerthi", "admin", "user", "service"}:
                continue
            if t.endswith("s"):
                t = t[:-1]
            tokens.append(t)

        for token in tokens:
            for slug in agent_defs:
                if token in slug or slug in token:
                    return slug

        # Fuzzy match individual tokens or the whole key to tolerate
        # dynamically generated identifiers (e.g.
        # ``keerthi_quotes_agent_test_0001``) and minor misspellings.
        try:  # ``difflib`` is part of the stdlib
            import difflib

            tokens = tokens_raw or re.split(r"[_]+", key_lower)
            candidates = list(agent_defs.keys())
            for token in tokens:
                match = difflib.get_close_matches(token, candidates, n=1, cutoff=0.8)
                if match:
                    return match[0]
            match = difflib.get_close_matches(key_lower, candidates, n=1, cutoff=0.6)
            if match:
                return match[0]
        except Exception:  # pragma: no cover - extremely defensive
            logger.exception("Fuzzy agent resolution failed for '%s'", raw_key)
        return None

    def _get_agent_details(self, agent_spec) -> List[Dict[str, Any]]:
        """Resolve agent metadata for identifiers from policy tables.

        The ``*_linked_agents`` columns expose agent *keys* (e.g.
        ``"supplier_ranking"``) in a PostgreSQL-style array string such as
        ``"{supplier_ranking,quote_evaluation}"``.  This helper extracts those
        keys, verifies them against the agent definitions registry and returns
        the canonical agent type for downstream use.
        """

        keys = [k for k in re.findall(r"[A-Za-z0-9_]+", str(agent_spec or ""))]
        if not keys:
            return []

        agent_defs = self._load_agent_definitions()
        details: List[Dict[str, Any]] = []
        for key in keys:
            slug = self._canonical_key(key, agent_defs)
            if slug:
                details.append({"agent_type": slug, "agent_name": agent_defs[slug]})
            else:  # pragma: no cover - defensive logging
                logger.warning("Agent type '%s' not found in definitions", key)
        return details

    def _load_prompts(self) -> Dict[int, Dict[str, Any]]:
        """Load prompt templates keyed by ``promptId`` from :class:`PromptEngine`."""

        if self._prompt_cache is not None:
            return dict(self._prompt_cache)

        prompt_engine = getattr(self.agent_nick, "prompt_engine", None)
        if prompt_engine is None:
            from orchestration.prompt_engine import PromptEngine  # local import to avoid cycles

            prompt_engine = PromptEngine(self.agent_nick)
            setattr(self.agent_nick, "prompt_engine", prompt_engine)

        try:
            catalog = prompt_engine.prompts_by_id()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load prompts from prompt engine")
            catalog = {}

        prompts: Dict[int, Dict[str, Any]] = {}
        for pid, prompt in catalog.items():
            entry = dict(prompt)
            linked = prompt.get("linked_agents") or prompt.get("agents")
            if isinstance(linked, (list, tuple, set)):
                linked_spec = "{" + ",".join(str(item) for item in linked) + "}"
            else:
                linked_spec = linked
            entry["agents"] = self._get_agent_details(linked_spec)
            prompts[int(pid)] = entry

        for pid, entry in self._load_prompt_fixtures().items():
            prompts.setdefault(pid, entry)

        if not prompts:
            logger.warning("Prompt catalog empty; falling back to bundled fixtures")

        self._prompt_cache = dict(prompts)
        return dict(prompts)

    def _load_prompt_fixtures(self) -> Dict[int, Dict[str, Any]]:
        """Load bundled prompt fixtures as a fallback when the database is empty."""

        fixtures: Dict[int, Dict[str, Any]] = {}
        base_dir = Path(__file__).resolve().parent.parent
        fallback_file = base_dir / "prompts" / "prompts.json"
        try:
            with fallback_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return {}
        except Exception:  # pragma: no cover - defensive parsing
            logger.exception("Failed to load prompt fixtures from %s", fallback_file)
            return {}

        templates = payload.get("templates")
        if not isinstance(templates, list):
            return {}

        for entry in templates:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("promptId") or entry.get("prompt_id")
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            prompt = dict(entry)
            linked = prompt.get("linked_agents") or prompt.get("agents")
            if isinstance(linked, (list, tuple, set)):
                linked_spec = "{" + ",".join(str(item) for item in linked) + "}"
            else:
                linked_spec = linked
            prompt["agents"] = self._get_agent_details(linked_spec)
            fixtures[pid_int] = prompt

        return fixtures

    def _load_policies(self) -> Dict[int, Dict[str, Any]]:
        """Aggregate policy definitions keyed by their ID.

        The ``policy_desc`` column in ``proc.policy`` contains plain text
        descriptions.  Like prompts, policies may reference agents via
        ``linked_agents``.  When database access fails a fallback to bundled
        JSON policy files is performed.
        """

        if self._policy_cache is not None:
            return dict(self._policy_cache)

        policies: Dict[int, Dict[str, Any]] = {}
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT policy_id, policy_name, policy_desc, policy_details, policy_linked_agents
                        FROM proc.policy
                        """
                    )
                    rows = cursor.fetchall()
                for pid, name, desc, details, linked in rows:
                    if not desc and not details:
                        continue
                    detail_payload: Any = {}
                    if isinstance(details, (bytes, bytearray)):
                        details = details.decode()
                    if isinstance(details, str):
                        candidate = details.strip()
                        if candidate:
                            try:
                                detail_payload = json.loads(candidate)
                            except Exception:
                                detail_payload = candidate
                        else:
                            detail_payload = {}
                    elif isinstance(details, dict):
                        detail_payload = details
                    elif details is not None:
                        detail_payload = details
                    desc_text = str(desc) if desc is not None else ""
                    name_text = str(name).strip() if name is not None else ""
                    detail_obj = detail_payload if detail_payload is not None else {}
                    value = {
                        "policyId": int(pid),
                        "policyName": name_text or None,
                        "policy_name": name_text or None,
                        "policy_desc": desc_text,
                        "description": desc_text,
                        "details": detail_obj,
                        "agents": self._get_agent_details(linked),
                    }
                    policies[int(pid)] = value
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load policies from DB")
            self._policy_cache = {}
            return {}

        self._policy_cache = dict(policies)
        return dict(policies)

    def _get_model_training_service(self):
        endpoint = getattr(self, "model_training_endpoint", None)
        if endpoint is None:
            return None

        settings = getattr(self.agent_nick, "settings", None)
        learning_enabled = bool(getattr(settings, "enable_learning", False))
        try:
            endpoint.configure_capture(learning_enabled)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to synchronise workflow capture via training endpoint")
        try:
            return endpoint.get_service()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to resolve model training service from endpoint")
            return None

    def _trigger_automatic_training(self, workflow_name: str) -> None:
        """Kick off background training using procurement datasets."""

        if getattr(self, "model_training_endpoint", None) is None:
            return

        service = self._get_model_training_service()
        if service is None:
            return
        try:
            service.dispatch_training_and_refresh(force=False, limit=1)
            logger.debug("Automatic training dispatch executed for %s", workflow_name)
        except Exception:  # pragma: no cover - defensive execution
            logger.exception("Automatic training dispatch failed for %s", workflow_name)

    def _workflow_completed_successfully(self, result: Any) -> bool:
        """Return ``True`` when the workflow outcome represents a success."""

        status_flag = None
        if isinstance(result, dict):
            status_flag = self._normalise_status_flag(result.get("status"))
            if status_flag is False:
                return False
            if self._has_error_payload(result.get("error")):
                return False
            if self._has_error_payload(result.get("errors")):
                return False
            ctx = result.get("ctx")
            if isinstance(ctx, dict) and self._has_error_payload(ctx.get("errors")):
                return False
        elif hasattr(result, "status"):
            status_flag = self._normalise_status_flag(getattr(result, "status"))
            if status_flag is False:
                return False
            if self._has_error_payload(getattr(result, "error", None)):
                return False

        return status_flag is not False

    @staticmethod
    def _has_error_payload(payload: Any) -> bool:
        """Return ``True`` when an error payload contains meaningful data."""

        if not payload:
            return False
        if isinstance(payload, dict):
            return any(Orchestrator._has_error_payload(value) for value in payload.values())
        if isinstance(payload, (list, tuple, set)):
            return any(Orchestrator._has_error_payload(value) for value in payload)
        return True

    @staticmethod
    def _normalise_status_flag(status: Any) -> Optional[bool]:
        """Interpret workflow status payloads into tri-state booleans."""

        if status is None:
            return None
        if isinstance(status, bool):
            return status
        if isinstance(status, (int, float)):
            return status > 0
        if isinstance(status, str):
            value = status.strip().lower()
            if not value:
                return None
            if value in {"completed", "success", "succeeded", "ok", "done"}:
                return True
            if value in {
                "failed",
                "error",
                "errored",
                "blocked",
                "incomplete",
                "partial",
                "pending",
                "running",
                "in_progress",
                "queued",
            }:
                return False
            return None
        return None

    @staticmethod
    def _resolve_agent_name(agent_type: str) -> str:
        """Convert class-like agent names to registry keys."""
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", agent_type).lower()
        if name.endswith("_agent"):
            name = name[:-6]
        return name

    @staticmethod
    def _coerce_workflow_hint(value: Any) -> Optional[str]:
        """Return a normalised workflow hint or ``None`` when unavailable."""

        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
        else:
            candidate = str(value).strip()
        return candidate or None

    @classmethod
    def _normalize_workflow_name(cls, workflow: Optional[str]) -> Optional[str]:
        """Map legacy workflow names onto the current registry."""

        if not workflow:
            return None
        candidate = str(workflow).strip()
        if not candidate:
            return None
        mapped = cls.WORKFLOW_ALIASES.get(candidate.lower())
        return mapped or candidate

    @classmethod
    def _workflow_has_defaults(cls, workflow: Optional[str]) -> bool:
        if not workflow:
            return False
        return workflow.lower() in cls.WORKFLOW_DEFAULT_CONDITIONS

    @classmethod
    def _apply_default_conditions(cls, input_data: Dict[str, Any], workflow: Optional[str]) -> None:
        """Ensure default condition payloads exist for known workflows."""

        if not workflow:
            return
        defaults = cls.WORKFLOW_DEFAULT_CONDITIONS.get(workflow.lower())
        if not defaults:
            return
        conditions = input_data.get("conditions")
        if not isinstance(conditions, dict):
            conditions = {}
            input_data["conditions"] = conditions
        for field, value in defaults.items():
            current = conditions.get(field)
            if current is None or (isinstance(current, str) and not current.strip()):
                conditions[field] = value


    def _default_workflow_for_agent(self, agent_key: Any) -> Optional[str]:
        """Return an implicit workflow for agents with legacy defaults."""

        if not agent_key:
            return None

        key = str(agent_key).strip()
        if not key:
            return None

        slug = self._resolve_agent_name(key)
        lower = key.lower()
        for candidate in (slug, lower):
            if candidate in self.WORKFLOW_DEFAULTS:
                return self.WORKFLOW_DEFAULTS[candidate]
        return None

    def _ensure_workflow_metadata(
        self, input_data: Dict[str, Any], *hints: Any, agent_key: Optional[str] = None
    ) -> None:
        """Ensure agent input contains a usable ``workflow`` field.

        Legacy flows and dynamic pass-through data occasionally provide the
        ``workflow`` field with a ``None`` value, causing agents such as the
        ``OpportunityMinerAgent`` to block execution.  This helper promotes the
        first non-empty hint to ``input_data['workflow']`` and removes the field
        entirely when no valid value can be resolved.
        """

        candidate = self._coerce_workflow_hint(input_data.get("workflow"))
        if not candidate:
            for hint in hints:
                candidate = self._coerce_workflow_hint(hint)
                if candidate:
                    break

        if not candidate:
            candidate = self._default_workflow_for_agent(agent_key)

        fallback_used = False
        alias_applied = False

        if candidate:
            normalised = self._normalize_workflow_name(candidate)
            if normalised and normalised != candidate:
                alias_applied = True
            candidate = normalised or candidate
        else:
            for hint in hints:
                candidate = self._coerce_workflow_hint(hint)
                if candidate:
                    normalised = self._normalize_workflow_name(candidate)
                    if normalised and normalised != candidate:
                        alias_applied = True
                    candidate = normalised or candidate
                    break

        if not candidate:
            candidate = self._default_workflow_for_agent(agent_key)
            if candidate:
                fallback_used = True
                normalised = self._normalize_workflow_name(candidate)
                if normalised and normalised != candidate:
                    alias_applied = True
                candidate = normalised or candidate


        if candidate:
            input_data["workflow"] = candidate
            if fallback_used or alias_applied or self._workflow_has_defaults(candidate):
                self._apply_default_conditions(input_data, candidate)
        else:
            input_data.pop("workflow", None)

    def execute_agent_flow(
        self,
        flow: Dict[str, Any],
        payload: Optional[Dict[str, Any]] = None,
        process_id: Optional[int] = None,
        prs: Any = None,
    ) -> Dict[str, Any]:
        """Execute a flow either in new JSON form or legacy tree structure."""

        if isinstance(flow, dict) and "entrypoint" in flow and "steps" in flow:
            return self._execute_json_flow(flow, payload or {}, process_id, prs)

        # Fallback to previous onSuccess/onFailure style graphs
        return self._execute_legacy_flow(flow, process_id, prs)

    # ------------------------------------------------------------------
    # New JSON flow executor
    # ------------------------------------------------------------------

    def _merge_pass_fields(self, target: Dict[str, Any], new_fields: Dict[str, Any]) -> None:
        """Merge ``new_fields`` into ``target`` without losing supplier data."""

        if not isinstance(target, dict) or not isinstance(new_fields, dict):
            return

        for key, value in new_fields.items():
            if self._merge_supplier_field(target, key, value):
                continue
            target[key] = value

    def _merge_supplier_field(self, target: Dict[str, Any], key: str, value: Any) -> bool:
        """Specialised merge for supplier-related payloads."""

        if key == "supplier_candidates" and isinstance(value, list):
            existing = target.get(key)
            combined: List[str] = []
            seen: set[str] = set()

            def _normalise(item: Any) -> Optional[str]:
                if item is None:
                    return None
                if isinstance(item, str):
                    candidate = item.strip()
                else:
                    candidate = str(item).strip()
                return candidate or None

            for sequence in (
                existing if isinstance(existing, list) else [],
                value,
            ):
                for item in sequence:
                    candidate = _normalise(item)
                    if not candidate or candidate in seen:
                        continue
                    combined.append(candidate)
                    seen.add(candidate)

            target[key] = combined
            return True

        if key == "supplier_directory" and isinstance(value, list):
            existing = target.get(key)
            merged: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []

            def _entry_key(entry: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
                if isinstance(entry, dict):
                    candidate = {k: v for k, v in entry.items() if v is not None}
                    sid = candidate.get("supplier_id")
                    if sid is not None:
                        token = str(sid).strip()
                        if token:
                            return token, candidate
                    try:
                        token = json.dumps(candidate, sort_keys=True, default=str)
                    except Exception:
                        token = repr(sorted(candidate.items()))
                    return token, candidate
                if entry is None:
                    return None, None
                token = str(entry).strip()
                if not token:
                    return None, None
                return token, {"value": entry}

            def _accumulate(source: Any) -> None:
                if not isinstance(source, list):
                    return
                for entry in source:
                    token, payload = _entry_key(entry)
                    if not token or not isinstance(payload, dict):
                        continue
                    current = merged.get(token, {})
                    updated = dict(current)
                    updated.update(payload)
                    merged[token] = updated
                    if token not in order:
                        order.append(token)

            _accumulate(existing if isinstance(existing, list) else [])
            _accumulate(value)
            target[key] = [merged[token] for token in order]
            return True

        if key == "policy_suppliers" and isinstance(value, dict):
            existing = target.get(key) if isinstance(target.get(key), dict) else {}
            combined: Dict[str, List[str]] = {}

            def _collect(source: Dict[str, Any]) -> None:
                for policy_key, suppliers in source.items():
                    policy_token = str(policy_key)
                    bucket = combined.setdefault(policy_token, [])
                    if isinstance(suppliers, (list, tuple, set)):
                        sequence = suppliers
                    else:
                        sequence = [suppliers]
                    for supplier in sequence:
                        if supplier is None:
                            continue
                        candidate = str(supplier).strip()
                        if candidate and candidate not in bucket:
                            bucket.append(candidate)

            if isinstance(existing, dict):
                _collect(existing)
            _collect(value)
            target[key] = combined
            return True

        return False

    def _execute_json_flow(
        self,
        flow: Dict[str, Any],
        payload: Dict[str, Any],
        process_id: Optional[int] = None,
        prs: Any = None,
    ) -> Dict[str, Any]:
        """Execute a flow defined with ``entrypoint`` and ``steps`` fields."""

        steps = flow.get("steps", {})
        entry = flow.get("entrypoint")
        defaults = flow.get("defaults", {})
        run_ctx: Dict[str, Any] = {"payload": payload, "errors": {}}
        agent_defs = self._load_agent_definitions()


        def _render(value: Any) -> Any:
            if isinstance(value, str):
                try:
                    rendered = Template(value).render(ctx=run_ctx, payload=payload)
                    if rendered.isdigit():
                        return int(rendered)
                    try:
                        return float(rendered)
                    except ValueError:
                        return rendered
                except Exception:
                    return value
            if isinstance(value, dict):
                return {k: _render(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_render(v) for v in value]
            return value

        def _extract(data: Dict[str, Any], expr: str) -> Any:
            if jsonpath_parse:
                try:
                    matches = jsonpath_parse(expr).find(data)
                    if matches:
                        return matches[0].value
                except Exception:  # pragma: no cover - invalid JSONPath
                    return None
            # Fallback: treat expression as dotted path
            cur = data
            for part in expr.lstrip("$").strip(".").split('.'):
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    return None
            return cur

        def _assign(target: Dict[str, Any], path: str, value: Any) -> None:
            keys = path.split('.')
            cur = target
            for key in keys[:-1]:
                cur = cur.setdefault(key, {})
            cur[keys[-1]] = value

        queue: List[str] = [entry]
        visited: set[str] = set()
        flow_status = 100

        while queue:
            step_name = queue.pop(0)
            step = steps.get(step_name)
            if not step or step_name in visited:
                continue
            visited.add(step_name)

            condition = step.get("condition")
            if condition:
                rendered = _render(condition)
                if str(rendered).lower() in ("", "0", "false", "none"):
                    next_steps = step.get("next", [])
                    if isinstance(next_steps, str):
                        next_steps = [next_steps]
                    queue.extend(next_steps)
                    continue

            agent_key_raw = step.get("agent")
            slug = self._canonical_key(agent_key_raw, agent_defs) or self._resolve_agent_name(agent_key_raw)
            agent = self.agents.get(slug) or self.agents.get(agent_key_raw)
            if not agent:
                logger.error("Agent %s not registered", agent_key_raw)
                run_ctx["errors"][step_name] = f"Agent '{agent_key_raw}' not registered"
                flow_status = 0
                continue
            agent_key = slug or agent_key_raw


            retries = int(step.get("retry", 0))
            timeout = step.get("timeout_seconds")
            on_error = step.get("on_error", "fail")

            # Merge the global payload with any default or step-specific input
            # so that agents receive the full context when an explicit ``input``
            # section is omitted.  Step definitions override defaults, which in
            # turn override the payload values.
            input_cfg = {
                **(payload or {}),
                **(defaults.get("input", {}) if isinstance(defaults, dict) else {}),
                **step.get("input", {}),
            }
            rendered_input = _render(input_cfg)

            combined_props: Dict[str, Any] = {}
            if isinstance(defaults, dict):
                default_props = defaults.get("agent_property") or defaults.get("properties")
                if isinstance(default_props, dict):
                    combined_props.update(default_props)
                for key in ("llm", "prompts", "policies", "workflow"):
                    if defaults.get(key) is not None:
                        combined_props.setdefault(key, defaults.get(key))

            step_props = step.get("agent_property") or step.get("properties")
            if isinstance(step_props, dict):
                combined_props.update(step_props)
            for key in ("llm", "prompts", "policies", "workflow"):
                if step.get(key) is not None:
                    combined_props[key] = step.get(key)

            normalised_props = ProcessRoutingService._normalise_agent_properties(
                combined_props, apply_default=False
            )

            agent_input = dict(rendered_input)
            llm_value = normalised_props.get("llm")
            if llm_value:
                agent_input["llm"] = llm_value
            for key, value in normalised_props.items():
                if key in {"llm", "prompts", "policies"}:
                    continue
                agent_input.setdefault(key, value)

            prompts_catalog = getattr(self, "_prompt_cache", None) or self._load_prompts()
            policies_catalog = getattr(self, "_policy_cache", None) or self._load_policies()

            def _merge_instruction_list(
                field: str, new_items: List[Dict[str, Any]], identifier: str
            ) -> None:
                if not new_items:
                    return
                existing = agent_input.get(field)
                if not isinstance(existing, list):
                    agent_input[field] = list(new_items)
                    return
                seen: set[Any] = set()
                for item in existing:
                    if isinstance(item, dict) and identifier in item:
                        seen.add(item[identifier])
                for item in new_items:
                    if not isinstance(item, dict):
                        continue
                    ident = item.get(identifier)
                    if ident in seen:
                        continue
                    existing.append(item)
                    if ident is not None:
                        seen.add(ident)

            prompt_objs: List[Dict[str, Any]] = []
            for pid in normalised_props.get("prompts", []):
                prompt = prompts_catalog.get(pid) if isinstance(prompts_catalog, dict) else None
                if prompt:
                    prompt_objs.append(prompt)
                else:
                    logger.warning(
                        "Unknown prompt id '%s' referenced in step %s", pid, step_name
                    )
            _merge_instruction_list("prompts", prompt_objs, "promptId")

            policy_objs: List[Dict[str, Any]] = []
            for pid in normalised_props.get("policies", []):
                policy = policies_catalog.get(pid) if isinstance(policies_catalog, dict) else None
                if policy:
                    policy_objs.append(policy)
                else:
                    logger.warning(
                        "Unknown policy id '%s' referenced in step %s", pid, step_name
                    )
            _merge_instruction_list("policies", policy_objs, "policyId")

            default_input = defaults.get("input", {}) if isinstance(defaults, dict) else {}
            self._ensure_workflow_metadata(
                agent_input,
                normalised_props.get("workflow"),
                step.get("workflow"),
                combined_props.get("workflow"),
                defaults.get("workflow") if isinstance(defaults, dict) else None,
                default_input.get("workflow") if isinstance(default_input, dict) else None,
                payload.get("workflow") if isinstance(payload, dict) else None,
                flow.get("workflow"),
                agent_key=agent_key,
            )

            # Ensure that database-backed defaults from ``proc.agent`` are
            # applied consistently so that agents receive their configured
            # prompts and policies even when the JSON flow omits them.  Without
            # this injection the orchestrator would pass only the static flow
            # payload which led to the agents operating on stale instructions.
            self._inject_agent_instructions(agent_key, agent_input)

            if not ProcessRoutingService._extract_llm_name(agent_input.get("llm")):
                agent_input["llm"] = ProcessRoutingService.DEFAULT_LLM_MODEL

            success = False
            attempt = 0
            result = None
            if prs and process_id is not None:
                # Mark the agent as ``validated`` to indicate it has started
                # execution. Downstream status transitions are handled once the
                # agent completes.
                prs.update_agent_status(process_id, step_name, "validated")
            logger.info(
                "Executing step %s with agent %s",
                step_name,
                agent_key,
            )
            while attempt <= retries and not success:
                attempt += 1
                manifest = self.manifest_service.build_manifest(agent_key)
                agent_input.setdefault("agent_manifest", manifest)
                agent_input.setdefault("policy_context", manifest.get("policies", []))
                agent_input.setdefault("knowledge_context", manifest.get("knowledge", {}))
                context = AgentContext(
                    workflow_id=str(uuid.uuid4()),
                    agent_id=agent_key,
                    user_id=self.settings.script_user,
                    input_data=agent_input,
                    task_profile=manifest.get("task", {}),
                    policy_context=manifest.get("policies", []),
                    knowledge_base=manifest.get("knowledge", {}),
                )
                context.apply_manifest(manifest)
                try:
                    if timeout:
                        fut = self.executor.submit(agent.execute, context)
                        result = fut.result(timeout=timeout)
                    else:
                        result = agent.execute(context)
                    success = result and result.status == AgentStatus.SUCCESS
                except Exception as exc:  # pragma: no cover - execution error
                    logger.exception("Agent %s execution failed", agent_key)
                    run_ctx["errors"][step_name] = str(exc)
                finally:
                    logger.info(
                        "Step %s attempt %s completed with status %s",
                        step_name,
                        attempt,
                        getattr(result, "status", None),
                    )
            if not success:
                flow_status = 0
                if on_error == "fail":
                    if prs and process_id is not None:
                        prs.update_agent_status(process_id, step_name, "failed")
                    return {"status": 0, "ctx": run_ctx}
            if prs and process_id is not None:
                prs.update_agent_status(
                    process_id,
                    step_name,
                    "completed" if success else "failed",
                )

            # Map outputs regardless of success; downstream may rely on partial data
            outputs = step.get("outputs", {})
            data = result.data if result and result.data else {}
            for key, expr in outputs.items():
                value = _extract(data, expr)
                _assign(run_ctx, key, value)
            if result and result.pass_fields:
                self._merge_pass_fields(run_ctx, result.pass_fields)
            if result and result.error:
                run_ctx["errors"][step_name] = result.error

            next_steps = step.get("next", [])
            if isinstance(next_steps, str):
                next_steps = [next_steps]
            queue.extend(next_steps)

        return {"status": flow_status, "ctx": run_ctx}

    def _execute_legacy_flow(
        self, flow: Dict[str, Any], process_id: Optional[int] = None, prs: Any = None
    ) -> Dict[str, Any]:
        """Execute legacy tree-based flows with ``onSuccess``/``onFailure`` links."""

        prompts = self._load_prompts()
        policies = self._load_policies()

        used_ids: set[str] = set()

        def _new_id() -> str:
            """Generate a unique workflow id for each node."""
            candidate = str(uuid.uuid4())
            while candidate in used_ids:
                candidate = str(uuid.uuid4())
            used_ids.add(candidate)
            return candidate

        def _run(node: Dict[str, Any], inherited: Optional[Dict[str, Any]] = None):
            """Recursively execute nodes while propagating pass fields."""

            required_fields = ["status", "agent_type", "agent_property"]
            for field in required_fields:
                if field not in node:
                    raise ValueError(f"Missing field '{field}' in agent node")

            props = node.get("agent_property", {})
            details = self._get_agent_details(node["agent_type"])
            if not details:
                raise ValueError(f"Unknown agent_type '{node['agent_type']}'")
            agent_class = details[0]["agent_name"]
            node["agent_type"] = agent_class
            agent_key = self._resolve_agent_name(agent_class)
            agent = self.agents.get(agent_key) or self.agents.get(agent_class)
            if not agent:
                raise ValueError(f"Agent '{agent_class}' not registered")

            prompt_objs = []
            for pid in props.get("prompts", []):
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid prompt id '{pid}' for agent '{agent_class}'")
                if pid_int not in prompts:
                    raise ValueError(f"Unknown prompt id '{pid}'")
                prompt_objs.append(prompts[pid_int])

            policy_objs = []
            for pid in props.get("policies", []):
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid policy id '{pid}' for agent '{agent_class}'")
                if pid_int not in policies:
                    raise ValueError(f"Unknown policy id '{pid}'")
                policy_objs.append(policies[pid_int])

            # Merge inherited pass fields with agent properties.  Any fields
            # produced by upstream agents become part of the child's input
            # data, enabling contextual chaining.
            input_data = {**(inherited or {})}
            if "llm" in props:
                input_data["llm"] = props["llm"]
            for key, value in props.items():
                if key in {"llm", "prompts", "policies"}:
                    continue
                input_data.setdefault(key, value)
            if prompt_objs:
                input_data["prompts"] = prompt_objs
            if policy_objs:
                input_data["policies"] = policy_objs
            self._ensure_workflow_metadata(
                input_data,
                props.get("workflow"),
                (inherited or {}).get("workflow") if isinstance(inherited, dict) else None,
                node.get("workflow"),
                agent_key=agent_key,
            )
            manifest = self.manifest_service.build_manifest(agent_key)
            input_data.setdefault("agent_manifest", manifest)
            input_data.setdefault("policy_context", manifest.get("policies", []))
            input_data.setdefault("knowledge_context", manifest.get("knowledge", {}))

            context = AgentContext(
                workflow_id=_new_id(),
                agent_id=agent_key,
                user_id=self.settings.script_user,
                input_data=input_data,
                task_profile=manifest.get("task", {}),
                policy_context=manifest.get("policies", []),
                knowledge_base=manifest.get("knowledge", {}),
            )
            context.apply_manifest(manifest)
            if prs and process_id is not None:
                # Mark the agent as validated when execution begins to mirror
                # the real-time status transitions expected by the workflow
                # tracking requirements.
                prs.update_agent_status(process_id, node.get("agent"), "validated")
            logger.info(
                "Executing agent %s with input %s", agent_key, input_data
            )
            result = agent.execute(context)
            logger.info(
                "Agent %s completed with status %s and output %s",
                agent_key,
                getattr(result, "status", None),
                getattr(result, "data", None),
            )
            node["status"] = (
                "completed"
                if result and result.status == AgentStatus.SUCCESS
                else "failed"
            )
            if prs and process_id is not None:
                prs.update_agent_status(
                    process_id,
                    node.get("agent"),
                    "completed" if result and result.status == AgentStatus.SUCCESS else "failed",
                )

            # Prepare fields for downstream nodes
            next_fields = dict(inherited or {})
            if result and result.pass_fields:
                self._merge_pass_fields(next_fields, result.pass_fields)

            if result and result.status == AgentStatus.SUCCESS and node.get("onSuccess"):
                _run(node["onSuccess"], next_fields)
            elif result and result.status == AgentStatus.FAILED and node.get("onFailure"):
                _run(node["onFailure"], next_fields)

            return node

        # Execute the first node in the flow.  Each node updates its own
        # ``status`` field and recursively processes child nodes based on the
        # success or failure of the current agent.  Previously the top-level
        # flow ``status`` was overwritten with the result of the *last* agent
        # executed.  This meant that if the first agent failed but a child
        # agent succeeded, the overall flow incorrectly appeared successful.
        #
        # The workflow semantics require that the status of the initial agent
        # determines the overall flow status.  Subsequent agents should update
        # only their own nodes.  We therefore execute the flow and return it
        # without replacing the root node's status based on downstream
        # results.
        _run(flow)

        def _contains_failure(node: Dict[str, Any]) -> bool:
            if node.get("status") == "failed":
                return True
            for key in ("onSuccess", "onFailure", "onCompletion"):
                child = node.get(key)
                if isinstance(child, dict) and _contains_failure(child):
                    return True
            return False

        flow["status"] = 0 if _contains_failure(flow) else 100
        return flow

    def _validate_workflow(self, workflow_name: str, context: AgentContext) -> bool:
        """Validate workflow against policies."""
        validation_result = self.policy_engine.validate_workflow(
            workflow_name, context.user_id, context.input_data
        )
        return validation_result.get("allowed", True)

    def _inject_agent_instructions(
        self, agent_key: str, input_data: Dict[str, Any]
    ) -> None:
        """Populate ``input_data`` with default prompts and policies for ``agent_key``."""

        if not isinstance(input_data, dict):
            return

        agent_defs = self._load_agent_definitions()
        canonical = self._canonical_key(agent_key, agent_defs) or self._resolve_agent_name(agent_key)
        canonical = canonical or agent_key

        default_config: Dict[str, Any] = {}
        prs = getattr(self.agent_nick, "process_routing_service", None)
        if prs is not None:
            defaults_map = getattr(prs, "_agent_defaults_cache", None)
            if defaults_map is None:
                try:  # pragma: no cover - cache priming
                    prs._load_agent_links()
                    defaults_map = getattr(prs, "_agent_defaults_cache", {})
                except Exception:
                    logger.exception("Failed to load agent defaults for %s", agent_key)
                    defaults_map = {}
            if isinstance(defaults_map, dict):
                default_config = dict(defaults_map.get(canonical, {}))

        default_llm = default_config.get("llm")
        if default_llm and not input_data.get("llm"):
            input_data["llm"] = default_llm

        for key, value in default_config.items():
            if key in {"llm", "prompts", "policies"}:
                continue
            if value is None:
                continue
            if isinstance(value, dict):
                existing = input_data.get(key)
                if isinstance(existing, dict):
                    merged = dict(value)
                    merged.update(existing)
                    input_data[key] = merged
                elif key not in input_data:
                    input_data[key] = dict(value)
            elif key not in input_data:
                input_data[key] = value

        prompts_catalog = self._load_prompts()
        policies_catalog = self._load_policies()

        default_prompt_ids = ProcessRoutingService._coerce_identifier_list(
            default_config.get("prompts")
        )
        default_policy_ids = ProcessRoutingService._coerce_identifier_list(
            default_config.get("policies")
        )

        def _merge(target_key: str, items: List[Dict[str, Any]], identifier: str) -> None:
            if not items:
                return
            existing = input_data.get(target_key)
            if not isinstance(existing, list):
                input_data[target_key] = list(items)
                return
            seen: set[Any] = set()
            for entry in existing:
                if isinstance(entry, dict) and identifier in entry:
                    seen.add(entry[identifier])
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                ident = entry.get(identifier)
                if ident in seen:
                    continue
                existing.append(entry)
                if ident is not None:
                    seen.add(ident)

        prompt_objs: List[Dict[str, Any]] = []
        for pid in default_prompt_ids:
            prompt = prompts_catalog.get(pid) if isinstance(prompts_catalog, dict) else None
            if prompt:
                prompt_objs.append(dict(prompt))
            else:  # pragma: no cover - defensive logging
                logger.warning("Default prompt id %s not found for agent %s", pid, canonical)

        if not prompt_objs and not input_data.get("prompts"):
            for prompt in (prompts_catalog.values() if isinstance(prompts_catalog, dict) else []):
                for meta in prompt.get("agents", []):
                    slug = meta.get("agent_type") or self._resolve_agent_name(meta.get("agent_name"))
                    if slug == canonical:
                        prompt_objs.append(dict(prompt))
                        break
        _merge("prompts", prompt_objs, "promptId")

        policy_objs: List[Dict[str, Any]] = []
        for pid in default_policy_ids:
            policy = policies_catalog.get(pid) if isinstance(policies_catalog, dict) else None
            if policy:
                policy_objs.append(dict(policy))
            else:  # pragma: no cover - defensive logging
                logger.warning("Default policy id %s not found for agent %s", pid, canonical)

        if not policy_objs and not input_data.get("policies"):
            for policy in (policies_catalog.values() if isinstance(policies_catalog, dict) else []):
                for meta in policy.get("agents", []):
                    slug = meta.get("agent_type") or self._resolve_agent_name(meta.get("agent_name"))
                    if slug == canonical:
                        policy_objs.append(dict(policy))
                        break
        _merge("policies", policy_objs, "policyId")

    def _execute_extraction_workflow(self, context: AgentContext) -> Dict:
        """Execute document extraction workflow"""
        # Step 1: Extract documents
        extraction_result = self._execute_agent("data_extraction", context)

        if extraction_result.status != AgentStatus.SUCCESS:
            return extraction_result.data

        # Step 2: Process next agents in parallel if enabled
        if self.settings.parallel_processing and extraction_result.next_agents:
            results = self._execute_parallel_agents(
                extraction_result.next_agents, context, extraction_result.pass_fields
            )
        else:
            results = self._execute_sequential_agents(
                extraction_result.next_agents, context, extraction_result.pass_fields
            )

        return {"extraction": extraction_result.data, "downstream_results": results}

    def _execute_ranking_workflow(self, context: AgentContext) -> Dict:
        """Execute supplier ranking workflow"""

        results: Dict[str, Any] = {}
        input_data = context.input_data

        should_run_opportunity = (
            "opportunity_miner" in self.agents
            and not input_data.get("skip_opportunity_step", False)
        )

        supplier_candidates: List[str] = []
        category_hint: Optional[str] = self._normalise_category(
            input_data.get("product_category")
        )


        if should_run_opportunity:
            opp_context = self._create_child_context(context, "opportunity_miner", {})
            opp_result = self._execute_agent("opportunity_miner", opp_context)
            results["opportunities"] = opp_result.data if opp_result else {}

            if not opp_result or opp_result.status != AgentStatus.SUCCESS:
                logger.info(
                    "OpportunityMinerAgent did not complete successfully; skipping ranking stage"
                )
                return results

            opportunity_payload: Dict[str, Any] = {}
            if opp_result:
                opportunity_payload = dict(opp_result.pass_fields or {})
                if not opportunity_payload:
                    opportunity_payload = dict(opp_result.data or {})

            derived_category = self._derive_product_category(opportunity_payload)
            if derived_category:
                category_hint = derived_category
                if isinstance(results.get("opportunities"), dict):
                    results["opportunities"].setdefault(
                        "product_category", derived_category
                    )


            supplier_directory = (
                (opp_result.pass_fields or {}).get("supplier_directory")
                or opp_result.data.get("supplier_directory")
                or []
            )
            if supplier_directory:
                input_data["supplier_directory"] = supplier_directory

            supplier_candidates_raw = (
                (opp_result.pass_fields or {}).get("supplier_candidates")
                or opp_result.data.get("supplier_candidates")
                or []
            )
            seen_candidates: set[str] = set()
            for candidate in supplier_candidates_raw:
                if candidate is None:
                    continue
                candidate_str = str(candidate).strip()
                if not candidate_str or candidate_str in seen_candidates:
                    continue
                seen_candidates.add(candidate_str)
                supplier_candidates.append(candidate_str)

            if supplier_candidates:
                input_data["supplier_candidates"] = supplier_candidates
            else:
                logger.info(
                    "OpportunityMinerAgent returned no supplier candidates; skipping ranking stage"
                )
                return results
        else:
            existing_candidates = input_data.get("supplier_candidates") or []
            seen_candidates: set[str] = set()
            for candidate in existing_candidates:
                if candidate is None:
                    continue
                candidate_str = str(candidate).strip()
                if not candidate_str or candidate_str in seen_candidates:
                    continue
                seen_candidates.add(candidate_str)
                supplier_candidates.append(candidate_str)
            if supplier_candidates:
                input_data["supplier_candidates"] = supplier_candidates

        if category_hint and not input_data.get("product_category"):
            input_data["product_category"] = category_hint


        # Get supplier data for ranking
        input_data["supplier_data"] = self.query_engine.fetch_supplier_data(
            input_data
        )

        ranking_result = self._execute_agent("supplier_ranking", context)
        if not ranking_result:
            return results

        results["ranking"] = ranking_result.data or {}

        if ranking_result.status != AgentStatus.SUCCESS:
            return results

        pass_fields: Dict[str, Any] = dict(ranking_result.pass_fields or {})
        if (
            "supplier_directory" not in pass_fields
            and "supplier_directory" in input_data
        ):
            pass_fields["supplier_directory"] = input_data["supplier_directory"]
        if supplier_candidates and "supplier_candidates" not in pass_fields:
            pass_fields["supplier_candidates"] = supplier_candidates

        if category_hint and "product_category" not in pass_fields:
            pass_fields["product_category"] = category_hint


        ranking_payload = pass_fields.get("ranking")
        if not ranking_payload:
            ranking_payload = results["ranking"].get("ranking") if isinstance(results["ranking"], dict) else None
            if ranking_payload:
                pass_fields["ranking"] = ranking_payload

        downstream_agents: List[str] = []
        seen_downstream: set[str] = set()

        def _normalise_agent(candidate: str) -> Optional[str]:
            if not candidate:
                return None
            if candidate in self.agents:
                return candidate
            resolved = self._resolve_agent_name(candidate)
            return resolved if resolved in self.agents else None

        if ranking_payload and "quote_evaluation" in self.agents:
            downstream_agents.append("quote_evaluation")
            seen_downstream.add("quote_evaluation")

        for agent in ranking_result.next_agents or []:
            normalised = _normalise_agent(agent)
            if not normalised or normalised == "opportunity_miner":
                continue
            if normalised not in seen_downstream:
                downstream_agents.append(normalised)
                seen_downstream.add(normalised)

        if should_run_opportunity and "email_drafting" in seen_downstream:
            downstream_agents = [
                agent_name for agent_name in downstream_agents if agent_name != "email_drafting"
            ]
            seen_downstream.discard("email_drafting")

        if downstream_agents:
            downstream_results: Dict[str, Any] = {}
            for agent_name in downstream_agents:
                child_ctx = self._create_child_context(context, agent_name, pass_fields)
                agent_result = self._execute_agent(agent_name, child_ctx)
                downstream_results[agent_name] = (
                    agent_result.data if agent_result else {}
                )
                if agent_result and agent_result.pass_fields:
                    self._merge_pass_fields(pass_fields, agent_result.pass_fields)
            results["downstream_results"] = downstream_results

        if should_run_opportunity or downstream_agents:
            return results

        return ranking_result.data

    def _execute_quote_workflow(self, context: AgentContext) -> Dict:
        """Execute quote evaluation workflow"""
        # Execute quote evaluation
        logger.info("Quote workflow starting")
        quote_result = self._execute_agent("quote_evaluation", context)

        results: Dict[str, Any] = dict(quote_result.data)

        neg_fields = quote_result.pass_fields or {}
        required = {"supplier", "current_offer", "target_price", "rfq_id"}
        if required.issubset(neg_fields.keys()):
            neg_context = self._create_child_context(context, "negotiation", neg_fields)
            neg_res = self._execute_agent("negotiation", neg_context)
            results["negotiation"] = neg_res.data if neg_res else {}
        else:
            logger.info(
                "NegotiationAgent skipped due to missing fields: %s",
                neg_fields.keys(),
            )

        return results

    def _execute_opportunity_workflow(self, context: AgentContext) -> Dict:
        """Execute opportunity to RFQ workflow"""
        opp_result = self._execute_agent("opportunity_miner", context)
        results: Dict[str, Any] = {"opportunities": opp_result.data if opp_result else {}}

        category_hint = self._normalise_category(
            context.input_data.get("product_category")
        )
        if opp_result:
            opportunity_payload: Dict[str, Any] = dict(opp_result.pass_fields or {})
            if not opportunity_payload:
                opportunity_payload = dict(opp_result.data or {})
            derived_category = self._derive_product_category(opportunity_payload)
            if derived_category:
                category_hint = derived_category
        if category_hint and isinstance(results.get("opportunities"), dict):
            results["opportunities"].setdefault("product_category", category_hint)


        candidates_raw = opp_result.data.get("supplier_candidates", []) if opp_result else []
        seen_candidates: set[str] = set()
        candidates: List[str] = []
        for candidate in candidates_raw:
            if candidate is None:
                continue
            candidate_str = str(candidate).strip()
            if not candidate_str or candidate_str in seen_candidates:
                continue
            seen_candidates.add(candidate_str)
            candidates.append(candidate_str)

        if candidates:
            pass_payload: Dict[str, Any] = dict(opp_result.pass_fields or {})
            pass_payload["supplier_candidates"] = candidates
            rank_ctx = self._create_child_context(
                context, "supplier_ranking", pass_payload
            )
            rank_ctx.input_data["supplier_data"] = self.query_engine.fetch_supplier_data(
                rank_ctx.input_data
            )
            rank_res = self._execute_agent("supplier_ranking", rank_ctx)
            results["ranking"] = rank_res.data if rank_res else {}

            ranking_payload = []
            if rank_res:
                ranking_payload = (
                    (rank_res.pass_fields or {}).get("ranking")
                    or (rank_res.data or {}).get("ranking")
                    or []
                )

            pass_fields: Dict[str, Any] = dict((rank_res.pass_fields or {}) if rank_res else {})
            if candidates and "supplier_candidates" not in pass_fields:
                pass_fields["supplier_candidates"] = candidates
            if ranking_payload and "ranking" not in pass_fields:
                pass_fields["ranking"] = ranking_payload
            if category_hint and "product_category" not in pass_fields:
                pass_fields["product_category"] = category_hint

            if ranking_payload and "quote_evaluation" in self.agents:
                quote_ctx = self._create_child_context(
                    context, "quote_evaluation", pass_fields
                )
                quote_res = self._execute_agent("quote_evaluation", quote_ctx)
                if quote_res:
                    results["quote_evaluation"] = quote_res.data or {}
                    if quote_res.pass_fields:
                        self._merge_pass_fields(pass_fields, quote_res.pass_fields)

                    if quote_res.next_agents:
                        downstream: Dict[str, Any] = {}
                        for agent_name in quote_res.next_agents:
                            normalised = self._resolve_agent_name(agent_name)
                            if normalised not in self.agents:
                                continue
                            if normalised in {"opportunity_miner", "supplier_ranking", "quote_evaluation"}:
                                continue
                            child_ctx = self._create_child_context(
                                context, normalised, pass_fields
                            )
                            agent_res = self._execute_agent(normalised, child_ctx)
                            downstream[normalised] = agent_res.data if agent_res else {}
                            if agent_res and agent_res.pass_fields:
                                self._merge_pass_fields(pass_fields, agent_res.pass_fields)
                        if downstream:
                            results.setdefault("downstream_results", {}).update(downstream)

            email_input = {
                "ranking": ranking_payload,
                "findings": opp_result.data.get("findings", []) if opp_result else [],
            }
            if "quote_evaluation" in results:
                email_input["quotes"] = results["quote_evaluation"].get("quotes")
            email_ctx = self._create_child_context(context, "email_drafting", email_input)
            email_res = self._execute_agent("email_drafting", email_ctx)
            results["email_drafts"] = email_res.data if email_res else {}
        else:
            logger.info("SupplierRankingAgent skipped due to empty candidate list")

        return results

    def _execute_generic_workflow(
        self, workflow_name: str, context: AgentContext
    ) -> Dict:
        """Execute generic workflow based on routing rules"""
        results: Dict[str, Any] = {}
        current_agents = [workflow_name]
        depth = 0
        max_depth = self.routing_model.get("global_settings", {}).get(
            "max_chain_depth", 10
        )
        pass_fields: Dict[str, Any] = {}

        while current_agents and depth < max_depth:
            next_agents: List[str] = []

            for agent_name in current_agents:
                if agent_name not in self.agents:
                    continue

                # Use a dedicated child context per agent so that routing
                # history and agent identifiers remain accurate. Shared
                # ``pass_fields`` are merged into the child's input data.
                child_context = self._create_child_context(
                    context, agent_name, pass_fields
                )
                result = self._execute_agent(agent_name, child_context)
                results[agent_name] = result.data

                if result.next_agents:
                    next_agents.extend(result.next_agents)

                # Merge fields to be passed to subsequent agents.
                if result.pass_fields:
                    self._merge_pass_fields(pass_fields, result.pass_fields)

            current_agents = next_agents
            depth += 1

        return results

    def _execute_agent(self, agent_name: str, context: AgentContext) -> Any:
        """Execute single agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return None

        self._inject_agent_instructions(agent_name, context.input_data)
        workflow_name = context.input_data.get("workflow")
        with workflow_scope(
            workflow_id=context.workflow_id,
            workflow_name=workflow_name,
            agent_name=agent_name,
            metadata={"agent_context": context.input_data},
        ):
            return agent.execute(context)

    def _publish_workflow_complete(
        self,
        *,
        workflow_name: str,
        workflow_id: str,
        context: AgentContext,
        result: Any,
        status: str,
    ) -> None:
        """Publish a workflow completion event to downstream services."""

        try:
            payload = {
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "status": status,
                "result": result,
                "input_data": context.input_data,
                "routing_history": list(context.routing_history),
                "user_id": context.user_id,
            }
            self.event_bus.publish("workflow.complete", payload)
        except Exception:  # pragma: no cover - defensive publication
            logger.exception("Failed to publish workflow.complete event for %s", workflow_id)

    def _execute_parallel_agents(
        self, agents: List[str], context: AgentContext, pass_fields: Dict
    ) -> Dict:
        """Execute agents in parallel"""
        results = {}
        futures = {}

        for agent_name in agents:
            if agent_name in self.agents:
                child_context = self._create_child_context(
                    context, agent_name, pass_fields
                )
                future = self.executor.submit(
                    self._execute_agent, agent_name, child_context
                )
                futures[future] = agent_name

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result(timeout=30)
                results[agent_name] = result.data if result else None
            except Exception as e:
                logger.error(f"Parallel execution failed for {agent_name}: {e}")
                results[agent_name] = {"error": str(e)}

        return results

    def _execute_sequential_agents(
        self, agents: List[str], context: AgentContext, pass_fields: Dict
    ) -> Dict:
        """Execute agents sequentially"""
        results = {}

        for agent_name in agents:
            if agent_name in self.agents:
                child_context = self._create_child_context(
                    context, agent_name, pass_fields
                )
                result = self._execute_agent(agent_name, child_context)
                results[agent_name] = result.data if result else None

                # Update pass fields for next agent
                if result and result.pass_fields:
                    self._merge_pass_fields(pass_fields, result.pass_fields)

        return results

    @staticmethod
    def _normalise_category(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value)
        except Exception:
            return None
        text = text.strip()
        return text or None

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _derive_product_category(self, opportunity_payload: Any) -> Optional[str]:
        if not isinstance(opportunity_payload, dict):
            return None

        for key in (
            "product_category",
            "category_id",
            "primary_category",
            "spend_category",
        ):
            direct = self._normalise_category(opportunity_payload.get(key))
            if direct:
                return direct

        conditions = opportunity_payload.get("conditions")
        if isinstance(conditions, dict):
            for key in ("product_category", "category_id"):
                direct = self._normalise_category(conditions.get(key))
                if direct:
                    return direct

        findings = opportunity_payload.get("findings")
        if isinstance(findings, list):
            category_totals: Dict[str, float] = {}
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                category = self._normalise_category(
                    finding.get("category_id")
                    or finding.get("spend_category")
                    or finding.get("category")
                    or finding.get("item_category")
                )
                if not category:
                    continue
                weight = finding.get("financial_impact_gbp")
                if weight in (None, ""):
                    for alt_key in (
                        "potential_savings",
                        "total_savings",
                        "estimated_savings",
                        "value",
                        "impact",
                    ):
                        weight = finding.get(alt_key)
                        if weight not in (None, ""):
                            break
                score = self._safe_float(weight)
                if score <= 0.0:
                    score = 1.0
                category_totals[category] = category_totals.get(category, 0.0) + score
            if category_totals:
                return max(category_totals.items(), key=lambda item: (item[1], item[0]))[0]

        return None

    def _create_child_context(
        self, parent_context: AgentContext, agent_name: str, pass_fields: Dict
    ) -> AgentContext:
        """Create child context for agent execution"""
        child_input = {**parent_context.input_data, **pass_fields}
        self._ensure_workflow_metadata(
            child_input,
            pass_fields.get("workflow") if isinstance(pass_fields, dict) else None,
            parent_context.input_data.get("workflow"),
            agent_key=agent_name,
        )
        self._inject_agent_instructions(agent_name, child_input)
        manifest = self.manifest_service.build_manifest(agent_name)
        child_input.setdefault("agent_manifest", manifest)
        child_input.setdefault("policy_context", manifest.get("policies", []))
        child_input.setdefault("knowledge_context", manifest.get("knowledge", {}))
        child_context = AgentContext(
            workflow_id=parent_context.workflow_id,
            agent_id=agent_name,
            user_id=parent_context.user_id,
            input_data=child_input,
            parent_agent=parent_context.agent_id,
            routing_history=parent_context.routing_history.copy(),
            task_profile=manifest.get("task", {}),
            policy_context=manifest.get("policies", []),
            knowledge_base=manifest.get("knowledge", {}),
        )
        child_context.apply_manifest(manifest)
        return child_context
