import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import re
from pathlib import Path

import pandas as pd
import numpy as np
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


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
        return json.dumps(data, default=cls._serialize)

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
        details.setdefault("status", "")
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

    @staticmethod
    def convert_agents_to_flow(details: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy ``agents`` list into nested agent flow."""

        agents = details.get("agents") or []
        if not agents:
            return details

        agent_map = {
            a.get("agent"): a
            for a in agents
            if isinstance(a, dict) and a.get("agent")
        }

        referenced: set[str] = set()
        for a in agents:
            deps = a.get("dependencies", {})
            referenced.update(deps.get("onSuccess", []))
            referenced.update(deps.get("onFailure", []))
            referenced.update(deps.get("onCompletion", []))

        # The starting node is the one that is never referenced in any other
        # agent's dependency lists.  Dependencies denote downstream agents to
        # invoke upon completion of the current agent.
        roots = [name for name in agent_map if name not in referenced]
        root_name = roots[0] if roots else next(iter(agent_map), None)
        if not root_name:
            return details

        def build(name: str) -> Dict[str, Any]:
            node = agent_map.get(name, {})
            flow = {
                "agent": name,
                "status": node.get("status", "saved"),
                "agent_type": str(node.get("agent_type", node.get("agent", ""))),
                "agent_property": node.get(
                    "agent_property", {"llm": None, "prompts": [], "policies": []}
                ),
            }
            deps = node.get("dependencies", {})
            if deps.get("onSuccess"):
                flow["onSuccess"] = build(deps["onSuccess"][0])
            if deps.get("onFailure"):
                flow["onFailure"] = build(deps["onFailure"][0])
            if deps.get("onCompletion"):
                flow["onCompletion"] = build(deps["onCompletion"][0])
            return flow

        return build(root_name)

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
        props = node.setdefault("agent_property", {"llm": None, "prompts": [], "policies": []})
        props["prompts"] = prompt_map.get(base_key, [])
        props["policies"] = policy_map.get(base_key, [])
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
        details = self.get_process_details(process_id, raw=True) or {}
        agents = details.get("agents", [])
        total = len(agents)
        completed = 0
        for agent in agents:
            if agent.get("agent") == agent_name:
                agent["status"] = status
            if agent.get("status") in ("completed", "failed"):
                completed += 1
        progress = int((completed / total) * 100) if total else 0
        details["agents"] = agents
        details["status"] = progress
        self.update_process_details(process_id, details, modified_by)

    def update_process_status(self, process_id: int, status: int, modified_by: Optional[str] = None) -> None:
        """Update process status in ``proc.routing``.

        Only ``1`` (success) and ``-1`` (failure) are valid. Any other value
        is coerced to ``-1`` if negative or ``1`` if positive."""
        if status not in (1, -1):
            coerced = 1 if status > 0 else -1
            logger.warning(
                "Updated process %s to invalid status %s - coercing to %s",
                process_id, status, coerced,
            )
            status = coerced
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_status = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (status, modified_by or self.settings.script_user, process_id),
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
        status: str = "running",
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
                                datetime.utcnow(),
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


        def _annotate_process_details(details: Optional[Dict[str, Any]], status_progress: int) -> Dict[str, Any]:
            """Ensure `details` is a dict and annotate top-level progress.

            ``process_details`` describe entire workflows with nested agent
            dependencies.  Per-agent status updates are performed elsewhere, so
            this helper only updates the overall numeric ``status`` field to
            reflect the workflow's progress while preserving the existing
            structure.
            """
            details = self.normalize_process_details(details)
            details["status"] = status_progress
            return details

        run_id = run_id or str(uuid.uuid4())
        process_start_ts = process_start_ts or datetime.utcnow()
        process_end_ts = process_end_ts or datetime.utcnow()
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

        status_progress = 100 if ps in success_vals else 0

        # Annotate process_details per-agent with the derived numeric progress
        annotated_details = _annotate_process_details(process_details, status_progress)

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
