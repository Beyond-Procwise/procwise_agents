import logging
import uuid
from typing import Dict, List, Optional, Any
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
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

# Ensure GPU is enabled when available
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


class Orchestrator:
    """Main orchestrator for managing agent workflows"""

    def __init__(self, agent_nick):
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

    def execute_ranking_flow(self, query: str) -> Dict:
        """Public wrapper for the supplier ranking workflow."""
        return self.execute_workflow("supplier_ranking", {"query": query})

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

        try:
            # Create initial context
            context = AgentContext(
                workflow_id=workflow_id,
                agent_id=workflow_name,
                user_id=user_id or self.settings.script_user,
                input_data=input_data,
            )

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

            return {
                "status": "completed",
                "workflow_id": workflow_id,
                "result": result,
                "execution_path": context.routing_history,
            }

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
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
        """Load prompt templates keyed by ``promptId``.

        The ``prompts_desc`` field in ``proc.prompt`` stores free-form text
        rather than JSON.  Each row may also reference one or more linked
        agents via ``linked_agents`` which are resolved into agent metadata.
        A file-system fallback is used when the database is unavailable.
        """

        prompts: Dict[int, Dict[str, Any]] = {}
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT prompt_id, prompts_desc, prompt_linked_agents"
                        " FROM proc.prompt WHERE prompts_desc IS NOT NULL"
                    )
                    rows = cursor.fetchall()
                for pid, desc, linked in rows:
                    if not desc:
                        continue
                    value = {
                        "promptId": int(pid),
                        "template": str(desc),
                        "agents": self._get_agent_details(linked),
                    }
                    prompts[int(pid)] = value
            if prompts:
                return prompts
        except Exception:  # pragma: no cover - defensive fall back
            logger.exception("Failed to load prompts from DB, falling back to file")

        path = Path(__file__).resolve().parents[1] / "prompts" / "prompts.json"
        with path.open() as f:
            data = json.load(f)
        templates = data.get("templates", [])
        return {int(t["promptId"]): t for t in templates if "promptId" in t}

    def _load_policies(self) -> Dict[int, Dict[str, Any]]:
        """Aggregate policy definitions keyed by their ID.

        The ``policy_desc`` column in ``proc.policy`` contains plain text
        descriptions.  Like prompts, policies may reference agents via
        ``linked_agents``.  When database access fails a fallback to bundled
        JSON policy files is performed.
        """

        policies: Dict[int, Dict[str, Any]] = {}
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT policy_id, policy_desc, policy_linked_agents"
                        " FROM proc.policy WHERE policy_desc IS NOT NULL"
                    )
                    rows = cursor.fetchall()
                for pid, desc, linked in rows:
                    if not desc:
                        continue
                    value = {
                        "policyId": int(pid),
                        "description": str(desc),
                        "agents": self._get_agent_details(linked),
                    }
                    policies[int(pid)] = value
            if policies:
                return policies
        except Exception:  # pragma: no cover - defensive fall back
            logger.exception("Failed to load policies from DB, falling back to files")

        policy_dir = Path(__file__).resolve().parents[1] / "policies"
        idx = 1
        for file in sorted(policy_dir.glob("*.json")):
            with file.open() as f:
                items = json.load(f)
            for item in items:
                policies[idx] = item
                idx += 1
        return policies

    @staticmethod
    def _resolve_agent_name(agent_type: str) -> str:
        """Convert class-like agent names to registry keys."""
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", agent_type).lower()
        if name.endswith("_agent"):
            name = name[:-6]
        return name

    def execute_agent_flow(
        self, flow: Dict[str, Any], payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a flow either in new JSON form or legacy tree structure."""

        if isinstance(flow, dict) and "entrypoint" in flow and "steps" in flow:
            return self._execute_json_flow(flow, payload or {})

        # Fallback to previous onSuccess/onFailure style graphs
        return self._execute_legacy_flow(flow)

    # ------------------------------------------------------------------
    # New JSON flow executor
    # ------------------------------------------------------------------

    def _execute_json_flow(self, flow: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
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
        flow_status = "completed"

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
                flow_status = "failed"
                continue
            agent_key = slug or agent_key_raw

            retries = int(step.get("retry", 0))
            timeout = step.get("timeout_seconds")
            on_error = step.get("on_error", "fail")

            input_cfg = {**defaults.get("input", {}), **step.get("input", {})}
            rendered_input = _render(input_cfg)

            success = False
            attempt = 0
            result = None
            while attempt <= retries and not success:
                attempt += 1
                context = AgentContext(
                    workflow_id=str(uuid.uuid4()),
                    agent_id=agent_key,
                    user_id=self.settings.script_user,
                    input_data=rendered_input,
                )
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
                    success = False

            if not success:
                flow_status = "failed"
                if on_error == "fail":
                    return {"status": "failed", "ctx": run_ctx}

            # Map outputs regardless of success; downstream may rely on partial data
            outputs = step.get("outputs", {})
            data = result.data if result and result.data else {}
            for key, expr in outputs.items():
                value = _extract(data, expr)
                _assign(run_ctx, key, value)
            if result and result.pass_fields:
                for k, v in result.pass_fields.items():
                    run_ctx[k] = v
            if result and result.error:
                run_ctx["errors"][step_name] = result.error

            next_steps = step.get("next", [])
            if isinstance(next_steps, str):
                next_steps = [next_steps]
            queue.extend(next_steps)

        return {"status": flow_status, "ctx": run_ctx}

    def _execute_legacy_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
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
                if pid not in prompts:
                    raise ValueError(f"Unknown prompt id '{pid}'")
                prompt_objs.append(prompts[pid])

            policy_objs = []
            for pid in props.get("policies", []):
                if pid not in policies:
                    raise ValueError(f"Unknown policy id '{pid}'")
                policy_objs.append(policies[pid])

            # Merge inherited pass fields with agent properties.  Any fields
            # produced by upstream agents become part of the child's input
            # data, enabling contextual chaining.
            input_data = {**(inherited or {})}
            if "llm" in props:
                input_data["llm"] = props["llm"]
            if prompt_objs:
                input_data["prompts"] = prompt_objs
            if policy_objs:
                input_data["policies"] = policy_objs

            context = AgentContext(
                workflow_id=_new_id(),
                agent_id=agent_key,
                user_id=self.settings.script_user,
                input_data=input_data,
            )

            result = agent.execute(context)
            node["status"] = (
                "completed"
                if result and result.status == AgentStatus.SUCCESS
                else "failed"
            )

            # Prepare fields for downstream nodes
            next_fields = {**(inherited or {})}
            if result and result.pass_fields:
                next_fields.update(result.pass_fields)

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
        return flow

    def _validate_workflow(self, workflow_name: str, context: AgentContext) -> bool:
        """Validate workflow against policies"""
        if workflow_name == "supplier_ranking":
            return True
        validation_result = self.policy_engine.validate_workflow(
            workflow_name, context.user_id, context.input_data
        )
        return validation_result.get("allowed", True)

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
        # Get supplier data first
        context.input_data["supplier_data"] = self.query_engine.fetch_supplier_data(
            context.input_data
        )

        # Execute ranking
        ranking_result = self._execute_agent("supplier_ranking", context)

        # Process opportunities if found
        if "OpportunityMinerAgent" in ranking_result.next_agents:
            opp_context = self._create_child_context(
                context, "opportunity_miner", ranking_result.pass_fields
            )
            opp_result = self._execute_agent("opportunity_miner", opp_context)

            return {
                "ranking": ranking_result.data,
                "opportunities": opp_result.data if opp_result else None,
            }

        return ranking_result.data

    def _execute_quote_workflow(self, context: AgentContext) -> Dict:
        """Execute quote evaluation workflow"""
        # Execute quote evaluation
        quote_result = self._execute_agent("quote_evaluation", context)

        # Check if negotiation needed
        if quote_result.data.get("negotiation_required"):
            # Prepare negotiation
            neg_context = self._create_child_context(
                context, "negotiation", quote_result.pass_fields
            )
            # Would execute negotiation agent here

        return quote_result.data

    def _execute_opportunity_workflow(self, context: AgentContext) -> Dict:
        """Execute opportunity mining workflow"""
        # Execute opportunity mining
        opp_result = self._execute_agent("opportunity_miner", context)

        # Check for discrepancies
        if opp_result.data.get("total_savings_potential", 0) > 10000:
            disc_context = self._create_child_context(
                context, "discrepancy_detection", opp_result.pass_fields
            )
            disc_result = self._execute_agent("discrepancy_detection", disc_context)

            return {
                "opportunities": opp_result.data,
                "discrepancies": disc_result.data if disc_result else None,
            }

        return opp_result.data

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
                    pass_fields.update(result.pass_fields)

            current_agents = next_agents
            depth += 1

        return results

    def _execute_agent(self, agent_name: str, context: AgentContext) -> Any:
        """Execute single agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return None

        return agent.execute(context)

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
                    pass_fields.update(result.pass_fields)

        return results

    def _create_child_context(
        self, parent_context: AgentContext, agent_name: str, pass_fields: Dict
    ) -> AgentContext:
        """Create child context for agent execution"""
        return AgentContext(
            workflow_id=parent_context.workflow_id,
            agent_id=agent_name,
            user_id=parent_context.user_id,
            input_data={**parent_context.input_data, **pass_fields},
            parent_agent=parent_context.agent_id,
            routing_history=parent_context.routing_history.copy(),
        )
