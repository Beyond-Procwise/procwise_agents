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

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings

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

    @classmethod
    def _normalise_agent_properties(
        cls, props: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Clean agent property payloads coming from persisted workflows."""

        cleaned: Dict[str, Any] = {}
        if isinstance(props, dict):
            for key, value in props.items():
                if key == "memory":
                    continue
                cleaned[key] = value

        llm_value = cleaned.get("llm")
        if isinstance(llm_value, str):
            base = llm_value.split(":", 1)[0].strip()
            cleaned["llm"] = base or llm_value.strip()
        elif llm_value is None:
            cleaned["llm"] = None
        else:
            cleaned["llm"] = str(llm_value)

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
        if key in agent_defs:
            return key
        if raw in agent_defs:
            return raw
        for slug in agent_defs:
            if slug in key or key in slug:
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
                        for key in re.findall(r"[A-Za-z0-9_]+", str(linked or "")):
                            _record(prompt_map, key, pid)

                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT policy_id, policy_linked_agents FROM proc.policy"
                    )
                    for pid, linked in cursor.fetchall():
                        for key in re.findall(r"[A-Za-z0-9_]+", str(linked or "")):
                            _record(policy_map, key, pid)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load agent linkage metadata")

        return agent_defs, prompt_map, policy_map

    def _enrich_node(self, node, agent_defs, prompt_map, policy_map):
        """Recursively normalise agent types and attach prompt/policy IDs."""

        if not isinstance(node, dict):
            return
        raw_type = str(node.get("agent_type", ""))
        base_key = self._canonical_key(raw_type, agent_defs)
        if base_key:
            node["agent_type"] = agent_defs.get(base_key, raw_type)
        else:
            base_key = raw_type
        raw_props = node.get("agent_property", {"llm": None, "prompts": [], "policies": []})
        props = self._normalise_agent_properties(raw_props)

        prompt_ids = set(props.get("prompts", []))
        prompt_ids.update(self._coerce_identifier_list(prompt_map.get(base_key)))
        props["prompts"] = sorted(prompt_ids)

        policy_ids = set(props.get("policies", []))
        policy_ids.update(self._coerce_identifier_list(policy_map.get(base_key)))
        props["policies"] = sorted(policy_ids)

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

        agents = details.get("agents", [])
        order = [a.get("agent") for a in agents]
        if agent_name not in order:
            logger.warning(
                "Agent %s not found in process %s; skipping", agent_name, process_id
            )
            return
        idx = order.index(agent_name)
        for prev in agents[:idx]:
            # Allow downstream agents to start once predecessors have at least
            # begun execution. Only agents still in the ``saved`` state are
            # considered incomplete for ordering purposes.
            if prev.get("status") == "saved":
                raise ValueError(
                    f"Cannot update {agent_name} before {prev.get('agent')} starts"
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
        status: int,
        modified_by: Optional[str] = None,
        process_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update process status in ``proc.routing`` and keep ``process_details`` in sync.

        Only ``1`` (success) and ``-1`` (failure) are valid. Any other value
        is coerced to ``-1`` if negative or ``1`` if positive."""
        if status not in (1, -1):
            coerced = 1 if status > 0 else -1
            logger.warning(
                "Updated process %s to invalid status %s - coercing to %s",
                process_id, status, coerced,
            )
            status = coerced
        # Ensure the ``process_details`` blob reflects the new status.
        details = process_details or self.get_process_details(process_id, raw=True) or {}
        details["status"] = "completed" if status == 1 else "failed"
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
                            status,
                            self._safe_dumps(self.normalize_process_details(details)),
                            modified_by or self.settings.script_user,
                            process_id,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Updated process %s to status %s", process_id, status
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
