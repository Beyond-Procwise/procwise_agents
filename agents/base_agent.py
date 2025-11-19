# ProcWise/agents/base_agent.py

import boto3
from botocore.config import Config
import json
import logging
import os
import importlib
import importlib.util
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping
from urllib.parse import quote_plus
import threading

import psycopg2
from http import HTTPStatus

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from services.lmstudio_client import (
    LMStudioClientError,
    get_lmstudio_client,
)

from config.settings import settings
from orchestration.prompt_engine import PromptEngine
from engines.policy_engine import PolicyEngine
from engines.query_engine import QueryEngine
from engines.routing_engine import RoutingEngine
from services.process_routing_service import ProcessRoutingService
from services.learning_repository import LearningRepository
from services.static_policy_loader import StaticPolicyLoader
from services.workflow_memory_service import WorkflowMemoryService
from utils.gpu import configure_gpu

try:  # Optional imports used for dataset persistence
    from models.context_trainer import ConversationDatasetWriter, TrainingConfig
except Exception:  # pragma: no cover - optional dependency failures handled gracefully
    ConversationDatasetWriter = None  # type: ignore[assignment]
    TrainingConfig = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _build_phi4_fallback_models() -> Tuple[str, ...]:
    """Return fallback model identifiers for the LM Studio deployment."""

    configured = getattr(settings, "rag_model", None)
    candidates: List[str] = []
    for name in (configured, "phi4:latest", "phi4"):
        if not name:
            continue
        if "phi4" not in name.lower():
            continue
        if name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates.append("phi4:latest")
    return tuple(candidates)


_LMSTUDIO_FALLBACK_MODELS: Tuple[str, ...] = _build_phi4_fallback_models()


def _slugify_agent_name(value: Any) -> str:
    """Return a lower-case slug for ``value`` suitable for registry lookups."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"(?<!^)(?=[A-Z][a-z0-9])", "_", text)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).lower().strip("_")
    return slug


_AGENT_MODEL_FIELD_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "rag_pipeline": ("rag_model", "extraction_model"),
    "rag_service": ("rag_model", "extraction_model"),
    "rag_agent": ("rag_model", "extraction_model"),
    "prompt_engine": ("rag_model", "extraction_model"),
    "data_extraction_agent": (
        "data_extraction_model",
        "document_extraction_model",
        "extraction_model",
    ),
    "supplier_ranking_agent": ("supplier_ranking_model", "extraction_model"),
    "supplier_interaction_agent": (
        "supplier_interaction_model",
        "extraction_model",
    ),
    "email_drafting_agent": (
        "email_compose_model",
        "negotiation_email_model",
        "extraction_model",
    ),
    "email_dispatch_agent": ("email_dispatch_model", "extraction_model"),
    "email_watcher_agent": ("email_watcher_model", "extraction_model"),
    "negotiation_agent": ("negotiation_email_model", "extraction_model"),
    "quote_evaluation_agent": ("quote_evaluation_model", "extraction_model"),
    "quote_comparison_agent": ("quote_comparison_model", "extraction_model"),
    "approvals_agent": ("approvals_model", "extraction_model"),
    "opportunity_miner_agent": ("opportunity_miner_model", "extraction_model"),
    "discrepancy_detection_agent": (
        "discrepancy_detection_model",
        "extraction_model",
    ),
}


class AgentStatus(str, Enum):
    """Execution status for an agent."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AgentContext:
    """Runtime information passed between agents."""

    workflow_id: str
    agent_id: str
    user_id: str
    input_data: Dict[str, Any]
    parent_agent: Optional[str] = None
    routing_history: List[str] = field(default_factory=list)
    task_profile: Dict[str, Any] = field(default_factory=dict)
    policy_context: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    process_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id is required for AgentContext")

        if isinstance(self.input_data, dict):
            self.input_data.setdefault("workflow_id", self.workflow_id)

        # track invocation time and update routing path
        self.timestamp = datetime.utcnow()
        self.routing_history.append(self.agent_id)

    def create_child_context(
        self,
        agent_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> "AgentContext":
        """Create a child context that preserves workflow metadata."""

        child_payload: Dict[str, Any] = {}
        if isinstance(self.input_data, dict):
            child_payload.update(self.input_data)
        if isinstance(input_data, dict):
            child_payload.update(input_data)

        child_payload["workflow_id"] = self.workflow_id

        return AgentContext(
            workflow_id=self.workflow_id,
            agent_id=agent_id,
            user_id=self.user_id,
            input_data=child_payload,
            parent_agent=self.agent_id,
            routing_history=self.routing_history.copy(),
            task_profile=dict(self.task_profile),
            policy_context=[dict(policy) for policy in self.policy_context],
            knowledge_base=dict(self.knowledge_base),
            process_id=self.process_id,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def apply_manifest(self, manifest: Optional[Dict[str, Any]]) -> None:
        """Populate task, policy, and knowledge slots from ``manifest``.

        ``manifest`` mirrors the structure returned by
        :class:`services.agent_manifest.AgentManifestService`.
        """

        if not manifest:
            return
        task_profile = manifest.get("task")
        if isinstance(task_profile, dict):
            self.task_profile = dict(task_profile)
        policies = manifest.get("policies")
        if isinstance(policies, list):
            self.policy_context = [dict(policy) for policy in policies]
        knowledge = manifest.get("knowledge")
        if isinstance(knowledge, dict):
            self.knowledge_base = dict(knowledge)

    def manifest(self) -> Dict[str, Any]:
        """Return a manifest-style payload summarising the context."""

        return {
            "task": dict(self.task_profile),
            "policies": [dict(policy) for policy in self.policy_context],
            "knowledge": dict(self.knowledge_base),
        }

    def create_child_context(
        self,
        agent_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> "AgentContext":
        """Create a child context inheriting workflow metadata.

        Child contexts inherit the workflow identifier, user, routing history,
        and manifest data unless explicitly overridden.  This helper keeps
        nested agent invocations consistent and avoids accidental workflow
        identifier drift when delegating tasks such as drafting negotiation
        emails.
        """

        child_input = dict(input_data) if isinstance(input_data, dict) else {}
        return AgentContext(
            workflow_id=overrides.get("workflow_id", self.workflow_id),
            agent_id=agent_id,
            user_id=overrides.get("user_id", self.user_id),
            input_data=child_input,
            parent_agent=overrides.get("parent_agent", self.agent_id),
            routing_history=list(self.routing_history),
            task_profile=overrides.get("task_profile", dict(self.task_profile)),
            policy_context=overrides.get(
                "policy_context", [dict(policy) for policy in self.policy_context]
            ),
            knowledge_base=overrides.get(
                "knowledge_base", dict(self.knowledge_base)
            ),
        )


@dataclass
class AgentOutput:
    """Standardised output returned by agents."""

    status: AgentStatus
    data: Dict[str, Any]
    next_agents: List[str] = field(default_factory=list)
    pass_fields: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    confidence: Optional[float] = None
    action_id: Optional[str] = None
    agentic_plan: Optional[str] = None
    context_snapshot: Optional[Dict[str, Any]] = None

class BaseAgent:
    DEFAULT_AGENTIC_PLAN_STEPS: Tuple[str, ...] = (
        "Review the incoming context and clarify the task objectives.",
        "Consult relevant knowledge bases, policies, and historical records.",
        "Synthesise findings into structured outputs and recommended next actions.",
    )

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        # Ensure GPU environment variables are set for all agents
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)
        logger.info(
            f"Initialized agent: {self.__class__.__name__} on device {self.device}"
        )
        prompt_engine = getattr(agent_nick, "prompt_engine", None)
        if prompt_engine is None:
            try:
                prompt_engine = PromptEngine(agent_nick)
            except Exception:  # pragma: no cover - defensive fallback for tests
                logger.debug("Falling back to in-memory prompt engine", exc_info=True)
                prompt_engine = PromptEngine(agent_nick=None, prompt_rows=[])
            setattr(agent_nick, "prompt_engine", prompt_engine)
        self.prompt_engine = prompt_engine
        self.learning_repository = getattr(agent_nick, "learning_repository", None)
        dataset_writer = getattr(agent_nick, "_context_dataset_writer", None)
        if dataset_writer is None and ConversationDatasetWriter and TrainingConfig:
            try:
                data_dir = getattr(
                    self.settings,
                    "context_training_data_dir",
                    TrainingConfig().data_dir if TrainingConfig else None,
                )
                if data_dir:
                    dataset_writer = ConversationDatasetWriter(data_dir)
            except Exception:
                logger.debug("Failed to initialise context dataset writer", exc_info=True)
                dataset_writer = None
            if dataset_writer is not None:
                setattr(agent_nick, "_context_dataset_writer", dataset_writer)
        self._context_dataset_writer = dataset_writer

    def run(self, *args, **kwargs):
        raise NotImplementedError("Each agent must implement its own 'run' method.")

    def execute(self, context: "AgentContext") -> "AgentOutput":
        """Execute the agent with process logging.

        This centralises writes to ``proc.routing`` and ``proc.action`` so that
        every agent invocation is captured in the database regardless of how it
        is triggered.
        """

        routing_service = getattr(self.agent_nick, "process_routing_service", None)
        if context.process_id and routing_service:
            if not routing_service.validate_workflow_id(
                context.process_id, context.workflow_id
            ):
                logger.error(
                    "%s execution blocked due to workflow mismatch",
                    self.__class__.__name__,
                )
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error=f"Workflow ID mismatch for process {context.process_id}",
                )

        logger.info("%s: starting", self.__class__.__name__)
        start_ts = datetime.utcnow()
        try:
            snapshot = self._prepare_context(context)
            result = self.run(context)
            if isinstance(result, AgentOutput):
                result = self._with_plan(context, result)
                result = self._with_context_snapshot(context, result, snapshot)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("%s execution failed", self.__class__.__name__)
            result = AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))
            snapshot = getattr(context, "_prepared_context_snapshot", None)
            result = self._with_plan(context, result)
            result = self._with_context_snapshot(context, result, snapshot)
        end_ts = datetime.utcnow()

        status = result.status.value
        # Persist the complete payloads so downstream consumers (including the
        # orchestrator, auditing dashboards, and the raw ``proc.routing``
        # records) have access to the full agent context and response. Earlier
        # implementations trimmed large strings/collections which resulted in
        # lossy data being stored. By passing the original structures here the
        # process logging tables retain the authoritative payload while the
        # storage layer remains responsible for any serialisation required.
        logged_input = context.input_data
        logged_output = self._prepare_logged_output(result.data)
        process_id: Optional[int] = None
        action_id: Optional[str] = None
        if routing_service:
            process_id = routing_service.log_process(
                process_name=self.__class__.__name__,
                process_details={"input": logged_input, "output": logged_output},
                user_id=context.user_id,
                user_name=self.agent_nick.settings.script_user,
                process_status=0,
                workflow_id=context.workflow_id,
            )
            if process_id is not None:
                context.process_id = process_id
                run_id = routing_service.log_run_detail(
                    process_id=process_id,
                    process_status=status,
                    process_details={"input": logged_input, "output": logged_output},
                    process_start_ts=start_ts,
                    process_end_ts=end_ts,
                    triggered_by=context.user_id,
                )
                action_id = routing_service.log_action(
                    process_id=process_id,
                    agent_type=self.__class__.__name__,
                    action_desc=logged_input,
                    process_output=logged_output,
                    status="completed"
                    if result.status == AgentStatus.SUCCESS
                    else "failed",
                    run_id=run_id,
                )
                result.action_id = action_id
                if isinstance(result.data, dict):
                    result.data["action_id"] = action_id
                    drafts = result.data.get("drafts")
                    if isinstance(drafts, list):
                        for draft in drafts:
                            if isinstance(draft, dict):
                                draft["action_id"] = action_id

        logger.info(
            "%s: completed with status %s", self.__class__.__name__, result.status.value
        )

        memory = getattr(self.agent_nick, "workflow_memory", None)
        if memory and getattr(memory, "enabled", False):
            input_summary = (
                WorkflowMemoryService.summarise_payload(logged_input)
                if isinstance(logged_input, dict)
                else {}
            )
            output_summary = (
                WorkflowMemoryService.summarise_payload(logged_output)
                if isinstance(logged_output, dict)
                else {}
            )
            try:
                memory.record_agent_execution(
                    context.workflow_id,
                    agent_name=self.__class__.__name__,
                    status=result.status.value,
                    summary={"input": input_summary, "output": output_summary},
                )
            except Exception:  # pragma: no cover - defensive logging only
                logger.debug(
                    "Workflow memory recording failed for %s", self.__class__.__name__, exc_info=True
                )

        self._persist_agentic_plan(context, result)
        try:
            self._record_context_example(context, result)
        except Exception:  # pragma: no cover - dataset capture should never fail hard
            logger.debug("Context dataset capture failed", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _prepare_logged_output(self, payload: Any) -> Any:
        """Return a copy of ``payload`` with heavy knowledge blobs removed."""

        return self._remove_knowledge_blocks(payload)

    @classmethod
    def _remove_knowledge_blocks(cls, value: Any) -> Any:
        """Recursively drop ``knowledge`` keys from nested payloads."""

        if isinstance(value, Mapping):
            cleaned: Dict[Any, Any] = {}
            for key, item in value.items():
                if key == "knowledge":
                    continue
                cleaned[key] = cls._remove_knowledge_blocks(item)
            return cleaned
        if isinstance(value, list):
            return [cls._remove_knowledge_blocks(item) for item in value]
        if isinstance(value, tuple):  # Serialisation converts tuples to lists later
            return [cls._remove_knowledge_blocks(item) for item in value]
        return value

    # ------------------------------------------------------------------
    # Context preparation helpers
    # ------------------------------------------------------------------
    def _conversation_memory_service(self):
        service = getattr(self.agent_nick, "conversation_memory", None)
        if service is not None:
            return service
        try:
            from services.conversation_memory import ConversationMemoryService
        except Exception:  # pragma: no cover - optional dependency missing
            logger.debug("Conversation memory service unavailable", exc_info=True)
            return None
        if not getattr(self.agent_nick, "qdrant_client", None):
            return None
        if not getattr(self.agent_nick, "embedding_model", None):
            return None
        try:
            service = ConversationMemoryService(self.agent_nick)
        except Exception:  # pragma: no cover - guard against runtime setup errors
            logger.debug("Failed to initialise conversation memory", exc_info=True)
            return None
        setattr(self.agent_nick, "conversation_memory", service)
        return service

    def _prepare_context(self, context: "AgentContext") -> Optional[Dict[str, Any]]:
        snapshot: Dict[str, Any] = {
            "workflow_id": getattr(context, "workflow_id", None),
            "agent_id": getattr(context, "agent_id", None),
            "user_id": getattr(context, "user_id", None),
        }

        conversation_entries = self._normalise_conversation_history(
            context.input_data.get("conversation_history")
        )
        if conversation_entries:
            snapshot["conversation_history"] = conversation_entries
            memory = self._conversation_memory_service()
            if memory is not None:
                try:
                    memory.ingest(snapshot.get("workflow_id"), conversation_entries)
                except Exception:
                    logger.debug("Failed to persist conversation history", exc_info=True)

        query_text = self._extract_query_text(context)
        memory = self._conversation_memory_service()
        if memory is not None and query_text:
            try:
                retrieved = memory.retrieve(snapshot.get("workflow_id"), query_text, limit=5)
            except Exception:
                logger.debug("Conversation retrieval failed", exc_info=True)
                retrieved = []
            if retrieved:
                snapshot["retrieved_memory"] = [
                    {
                        "content": item.content,
                        "score": item.score,
                        "metadata": item.metadata,
                    }
                    for item in retrieved
                ]

        procurement_context = self._collect_procurement_context(context, conversation_entries)
        if procurement_context:
            snapshot["procurement_context"] = procurement_context

        manifest = context.manifest()
        if manifest:
            snapshot["manifest"] = manifest

        # Remove heavy knowledge payloads before attaching the snapshot to the
        # runtime context or returning it to downstream consumers. This keeps
        # the live ``context_snapshot`` informative without persisting large
        # manifest knowledge bundles that are already available via
        # ``AgentContext.knowledge_base``.
        sanitized_snapshot = self._remove_knowledge_blocks(snapshot)

        if any(
            value
            for key, value in sanitized_snapshot.items()
            if key not in {"workflow_id", "agent_id", "user_id"}
        ):
            context.input_data = dict(context.input_data)
            context.input_data.setdefault("context_snapshot", sanitized_snapshot)
            context._prepared_context_snapshot = sanitized_snapshot  # type: ignore[attr-defined]
            return sanitized_snapshot
        context._prepared_context_snapshot = None  # type: ignore[attr-defined]
        return None

    def _with_context_snapshot(
        self,
        context: "AgentContext",
        output: AgentOutput,
        snapshot: Optional[Dict[str, Any]],
    ) -> AgentOutput:
        snapshot = snapshot or getattr(context, "_prepared_context_snapshot", None)
        if not snapshot:
            return output
        output.context_snapshot = snapshot
        if isinstance(output.data, dict):
            output.data.setdefault("context_snapshot", snapshot)
        return output

    def _normalise_conversation_history(
        self, history: Optional[Iterable[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        normalised: List[Dict[str, Any]] = []
        if not history:
            return normalised
        for entry in history:
            if not isinstance(entry, dict):
                continue
            clean: Dict[str, Any] = {}
            for key in (
                "message_id",
                "rfq_id",
                "supplier_id",
                "subject",
                "from_address",
                "message_body",
                "round",
                "negotiation_round",
                "workflow_id",
                "matched_via",
                "document_origin",
                "speaker",
                "summary",
                "negotiation_message",
            ):
                if key in entry:
                    clean[key] = entry[key]
            for sub in ("supplier_output", "negotiation_output"):
                payload = entry.get(sub)
                if isinstance(payload, dict):
                    clean[sub] = payload
            if clean:
                normalised.append(clean)
        return normalised

    def _extract_query_text(self, context: "AgentContext") -> str:
        candidates = []
        for key in (
            "prompt",
            "user_query",
            "user_input",
            "message_body",
            "question",
            "task_description",
        ):
            value = context.input_data.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        if not candidates:
            value = context.input_data.get("input")
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        combined = "\n".join(candidates)
        return combined.strip()

    def _collect_procurement_context(
        self,
        context: "AgentContext",
        conversation_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_engine = getattr(self.agent_nick, "query_engine", None)
        if query_engine is None:
            return []
        supplier_ids: List[str] = []
        supplier_names: List[str] = []
        for key in ("supplier_id", "supplier_ids"):
            value = context.input_data.get(key)
            if isinstance(value, str) and value.strip():
                supplier_ids.append(value.strip())
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                supplier_ids.extend(
                    str(item).strip() for item in value if str(item).strip()
                )
        supplier_name = context.input_data.get("supplier_name")
        if isinstance(supplier_name, str) and supplier_name.strip():
            supplier_names.append(supplier_name.strip())
        for entry in conversation_entries:
            candidate_id = entry.get("supplier_id")
            if isinstance(candidate_id, str) and candidate_id.strip():
                supplier_ids.append(candidate_id.strip())
        try:
            supplier_ids = list(dict.fromkeys(supplier_ids))
            supplier_names = list(dict.fromkeys(supplier_names))
        except Exception:
            supplier_ids = [sid for sid in supplier_ids if sid]
            supplier_names = [name for name in supplier_names if name]
        if not supplier_ids and not supplier_names:
            return []
        try:
            df = query_engine.fetch_procurement_flow(
                embed=True,
                supplier_ids=supplier_ids or None,
                supplier_names=supplier_names or None,
            )
        except Exception:
            logger.debug("Failed to fetch procurement context", exc_info=True)
            return []
        try:
            subset = df.head(10)
            important_cols = [
                col
                for col in (
                    "supplier_id",
                    "supplier_name",
                    "po_id",
                    "item_description",
                    "invoice_id",
                    "product",
                    "category_level_1",
                    "category_level_2",
                )
                if col in subset.columns
            ]
            if not important_cols:
                return []
            return subset[important_cols].to_dict("records")
        except Exception:
            logger.debug("Failed to normalise procurement context", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Agentic planning helpers
    # ------------------------------------------------------------------
    def _agentic_plan_steps(self, context: "AgentContext") -> List[str]:
        steps_source = getattr(self, "AGENTIC_PLAN_STEPS", None)
        steps: List[str]
        if callable(steps_source):
            generated = steps_source(context)
            if isinstance(generated, (list, tuple)):
                steps = [str(step) for step in generated if str(step).strip()]
            else:
                steps = []
        elif isinstance(steps_source, (list, tuple)):
            steps = [str(step) for step in steps_source if str(step).strip()]
        else:
            steps = []
        if not steps:
            steps = list(self.DEFAULT_AGENTIC_PLAN_STEPS)
        return steps

    def _format_agentic_plan(self, steps: List[str]) -> str:
        numbered: List[str] = []
        for idx, step in enumerate(steps, start=1):
            clean = str(step).strip()
            if clean:
                numbered.append(f"{idx}. {clean}")
        return "\n".join(numbered)

    def _build_agentic_plan(self, context: "AgentContext") -> str:
        steps = self._agentic_plan_steps(context)
        return self._format_agentic_plan(steps)

    def _with_plan(self, context: "AgentContext", output: AgentOutput) -> AgentOutput:
        try:
            plan = output.agentic_plan or self._build_agentic_plan(context)
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("Failed to build agentic plan", exc_info=True)
            plan = output.agentic_plan or self._format_agentic_plan(
                list(self.DEFAULT_AGENTIC_PLAN_STEPS)
            )
        output.agentic_plan = plan
        if isinstance(output.data, dict) and plan:
            output.data.setdefault("agentic_plan", plan)
        return output

    def _persist_agentic_plan(
        self, context: "AgentContext", output: AgentOutput
    ) -> None:
        """Store the agent's reasoning plan for auditing and replay."""

        plan = (output.agentic_plan or "").strip()
        if not plan:
            return

        get_connection = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_connection):
            logger.debug(
                "Skipping agentic plan persistence for %s because AgentNick lacks a DB connection",
                self.__class__.__name__,
            )
            return

        table_flag = "_agent_plan_table_ready"

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    if not getattr(self.agent_nick, table_flag, False):
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS proc.agent_plan (
                                plan_id BIGSERIAL PRIMARY KEY,
                                workflow_id text,
                                agent_id text,
                                agent_name text,
                                action_id text,
                                plan text NOT NULL,
                                created_at timestamp DEFAULT CURRENT_TIMESTAMP,
                                created_by text DEFAULT CURRENT_USER
                            )
                            """
                        )
                        setattr(self.agent_nick, table_flag, True)

                    cursor.execute(
                        """
                        INSERT INTO proc.agent_plan (
                            workflow_id,
                            agent_id,
                            agent_name,
                            action_id,
                            plan
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            getattr(context, "workflow_id", None),
                            getattr(context, "agent_id", None),
                            self.__class__.__name__,
                            getattr(output, "action_id", None),
                            plan,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception(
                "Failed to persist agentic plan for %s", self.__class__.__name__
            )

    # ------------------------------------------------------------------
    # Dataset capture helpers
    # ------------------------------------------------------------------
    def _record_context_example(
        self, context: "AgentContext", output: AgentOutput
    ) -> None:
        writer = getattr(self, "_context_dataset_writer", None)
        snapshot = output.context_snapshot or getattr(
            context, "_prepared_context_snapshot", None
        )
        if writer is None or snapshot is None:
            return
        try:
            context_text = self._format_training_context(snapshot)
            response_text = self._extract_response_text(output.data)
            metadata = {
                "workflow_id": snapshot.get("workflow_id"),
                "agent_id": snapshot.get("agent_id"),
                "agent_name": self.__class__.__name__,
            }
            writer.write_record(
                context_text=context_text,
                response_text=response_text,
                metadata=metadata,
            )
        except ValueError:
            logger.debug("Skipped empty training example for %s", self.__class__.__name__)

    def _format_training_context(self, snapshot: Dict[str, Any]) -> str:
        lines: List[str] = []
        conversation = snapshot.get("conversation_history") or []
        for entry in conversation:
            if not isinstance(entry, dict):
                continue
            speaker = (
                entry.get("speaker")
                or entry.get("from_address")
                or entry.get("document_origin")
                or "participant"
            )
            body = entry.get("message_body") or entry.get("negotiation_message") or entry.get("summary")
            if not isinstance(body, str):
                try:
                    body = json.dumps(body, ensure_ascii=False)
                except Exception:
                    body = str(body)
            body = (body or "").strip()
            if not body:
                continue
            round_info = entry.get("negotiation_round") or entry.get("round")
            origin = entry.get("document_origin")
            prefix_parts = [speaker]
            if origin:
                prefix_parts.append(f"origin={origin}")
            if round_info is not None:
                prefix_parts.append(f"round={round_info}")
            prefix = " ".join(prefix_parts)
            lines.append(f"{prefix}: {body}")
        retrieved = snapshot.get("retrieved_memory") or []
        if retrieved:
            lines.append("\nRetrieved context:")
            for item in retrieved[:5]:
                content = item.get("content") if isinstance(item, dict) else None
                if content:
                    lines.append(f"- {content}")
        procurement_records = snapshot.get("procurement_context") or []
        if procurement_records:
            lines.append("\nProcurement records:")
            for record in procurement_records[:5]:
                try:
                    lines.append(f"- {json.dumps(record, ensure_ascii=False)}")
                except Exception:
                    lines.append(f"- {record}")
        manifest = snapshot.get("manifest")
        if manifest:
            lines.append("\nManifest:")
            try:
                lines.append(json.dumps(manifest, ensure_ascii=False))
            except Exception:
                lines.append(str(manifest))
        text = "\n".join(lines).strip()
        return text or json.dumps(snapshot, ensure_ascii=False)

    def _extract_response_text(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return str(data)
        for key in (
            "message",
            "response",
            "summary",
            "draft",
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        drafts = data.get("drafts")
        if isinstance(drafts, list) and drafts:
            first = drafts[0]
            if isinstance(first, dict):
                body = first.get("body") or first.get("content") or first.get("message")
                if isinstance(body, str) and body.strip():
                    return body.strip()
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def call_lmstudio(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        format: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Wrapper around the LM Studio chat/completions API."""

        backend = getattr(self.settings, "llm_backend", "lmstudio") or "lmstudio"
        if backend.lower() == "langchain":
            try:
                return self._call_langchain_chat(
                    prompt=prompt,
                    model=model,
                    format=format,
                    messages=messages,
                    **kwargs,
                )
            except Exception as exc:
                logger.error(
                    "LangChain backend failed; falling back to LM Studio pipeline.",
                    exc_info=exc,
                )

        fallback_model = getattr(self.settings, "extraction_model", "gpt-oss")
        base_model = fallback_model
        resolver = getattr(self.agent_nick, "get_agent_model", None)
        if callable(resolver):
            try:
                resolved_model = resolver(
                    self.__class__.__name__, fallback=fallback_model
                )
            except Exception:  # pragma: no cover - defensive logging only
                logger.debug(
                    "Agent-specific model resolution failed for %s",
                    self.__class__.__name__,
                    exc_info=True,
                )
            else:
                if isinstance(resolved_model, str) and resolved_model.strip():
                    base_model = resolved_model.strip()

        options_from_kwargs = kwargs.pop("options", {}) or {}
        options = {**self.agent_nick.lmstudio_options(), **options_from_kwargs}
        response_format = {"type": "json_object"} if format == "json" else None

        candidate_names: List[str] = []
        for candidate in (model, base_model):
            if candidate and candidate not in candidate_names:
                candidate_names.append(candidate)
        for fallback_name in _LMSTUDIO_FALLBACK_MODELS:
            if fallback_name not in candidate_names:
                candidate_names.append(fallback_name)

        available_models = self._get_available_lmstudio_models()
        if available_models:
            filtered = [name for name in candidate_names if name in available_models]
            if filtered:
                candidate_names = filtered

        if not candidate_names:
            error_msg = "No available LM Studio models detected."
            logger.error(error_msg)
            return {"response": "", "error": error_msg}

        last_error: Optional[Exception] = None
        client = getattr(self.agent_nick, "lmstudio_client", None) or get_lmstudio_client()
        for model_to_use in candidate_names:
            try:
                if messages is not None:
                    return client.chat(
                        model=model_to_use,
                        messages=messages,
                        response_format=response_format,
                        options=options,
                    )
                return client.generate(
                    model=model_to_use,
                    prompt=prompt or "",
                    format=format,
                    options=options,
                )
            except LMStudioClientError as exc:
                logger.warning(
                    "LM Studio model '%s' unavailable or failed: %s",
                    model_to_use,
                    exc,
                )
                self._remove_cached_lmstudio_model(model_to_use)
                last_error = exc
                continue
            except Exception as exc:  # pragma: no cover - network / runtime issues
                last_error = exc
                break

        if last_error is not None:
            logger.error("LM Studio call failed", exc_info=last_error)
            return {"response": "", "error": str(last_error)}

        # Defensive fallback if the loop completes without returning (should not happen)
        logger.error("LM Studio call failed for unknown reasons")
        return {"response": "", "error": "Unknown LM Studio invocation failure"}

    def _call_langchain_chat(
        self,
        prompt: Optional[str],
        model: Optional[str],
        format: Optional[str],
        messages: Optional[List[Dict[str, str]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Invoke the configured LangChain chat model.

        The interface mirrors :meth:`call_lmstudio` so that agents can switch
        between the native LM Studio client and LangChain-powered execution
        using the ``LLM_BACKEND`` configuration flag.
        """

        chat_model = self._get_langchain_chat_model(model_override=model)

        allowed_kwarg_keys = {
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "response_format",
        }
        invocation_kwargs = {
            key: value for key, value in kwargs.items() if key in allowed_kwarg_keys
        }
        if format and "response_format" not in invocation_kwargs:
            if isinstance(format, str) and format.lower() == "json":
                invocation_kwargs["response_format"] = {"type": "json_object"}
            elif isinstance(format, (dict, list)):
                invocation_kwargs["response_format"] = format

        request_timeout = getattr(self.settings, "langchain_request_timeout", None)
        if request_timeout is not None:
            invocation_kwargs.setdefault("config", {})
            invocation_kwargs["config"].setdefault("configurable", {})
            invocation_kwargs["config"]["configurable"]["timeout"] = request_timeout

        langchain_messages = self._build_langchain_messages(messages, prompt)
        result = chat_model.invoke(langchain_messages, **invocation_kwargs)

        content: str
        additional: Dict[str, Any] = {}
        if hasattr(result, "content"):
            content = result.content  # type: ignore[assignment]
            to_dict = getattr(result, "to_dict", None)
            if callable(to_dict):
                additional = to_dict()
        elif isinstance(result, dict):
            content = str(
                result.get("content")
                or result.get("response")
                or result.get("message")
                or ""
            )
            additional = result
        else:
            content = str(result)

        response_payload: Dict[str, Any] = {"response": content}
        if additional:
            response_payload["langchain_raw"] = additional
        return response_payload

    def _get_langchain_chat_model(
        self,
        model_override: Optional[str] = None,
    ):
        cached_model = getattr(self.agent_nick, "_langchain_chat_model", None)
        cached_model_name = getattr(
            self.agent_nick, "_langchain_chat_model_name", None
        )
        target_model = model_override or getattr(self.settings, "langchain_model", None)
        if cached_model is not None and cached_model_name == target_model:
            return cached_model

        chat_model = self._initialise_langchain_chat_model(target_model)
        setattr(self.agent_nick, "_langchain_chat_model", chat_model)
        setattr(self.agent_nick, "_langchain_chat_model_name", target_model)
        return chat_model

    def _initialise_langchain_chat_model(self, model_name: Optional[str]):
        if importlib.util.find_spec("langchain") is None:
            raise RuntimeError(
                "LangChain backend requested but 'langchain' is not installed."
            )

        provider = (getattr(self.settings, "langchain_provider", "lmstudio") or "").lower()
        model_identifier = model_name or getattr(self.settings, "langchain_model", None)
        if not model_identifier:
            raise RuntimeError(
                "LANGCHAIN_MODEL must be configured when LLM_BACKEND=langchain"
            )

        base_url = getattr(self.settings, "langchain_api_base", None)
        api_key = getattr(self.settings, "langchain_api_key", None)

        if provider == "lmstudio" and not base_url:
            base_url = getattr(self.settings, "lmstudio_base_url", None)

        if provider in {"lmstudio", "openai", "azure-openai", "gpt"}:
            if importlib.util.find_spec("langchain_openai") is None:
                raise RuntimeError(
                    "LangChain OpenAI integration is not installed. Add 'langchain-openai'."
                )
            module = importlib.import_module("langchain_openai")
            chat_cls = getattr(module, "ChatOpenAI")
            init_kwargs = {"model": model_identifier}
            if api_key:
                init_kwargs["api_key"] = api_key
            if base_url:
                init_kwargs["base_url"] = base_url
            return chat_cls(**init_kwargs)

        module = importlib.import_module("langchain.chat_models")
        init_chat_model = getattr(module, "init_chat_model")
        config: Dict[str, Any] = {}
        if api_key:
            config["api_key"] = api_key
        if base_url:
            config["base_url"] = base_url
        return init_chat_model(provider, model=model_identifier, config=config or None)

    def _resolve_langchain_message_classes(self) -> Dict[str, Any]:
        cached = getattr(self.agent_nick, "_langchain_message_classes", None)
        if cached is not None:
            return cached

        module = importlib.import_module("langchain_core.messages")
        classes = {
            "system": getattr(module, "SystemMessage"),
            "human": getattr(module, "HumanMessage"),
            "ai": getattr(module, "AIMessage"),
        }
        tool_cls = getattr(module, "ToolMessage", None)
        if tool_cls is not None:
            classes["tool"] = tool_cls
        setattr(self.agent_nick, "_langchain_message_classes", classes)
        return classes

    def _build_langchain_messages(
        self,
        messages: Optional[List[Dict[str, Any]]],
        prompt: Optional[str],
    ) -> List[Any]:
        classes = self._resolve_langchain_message_classes()
        lc_messages: List[Any] = []

        if messages:
            for entry in messages:
                if not isinstance(entry, Mapping):
                    lc_messages.append(classes["human"](content=str(entry)))
                    continue
                role = str(entry.get("role", "user")).lower()
                content = entry.get("content", "")
                if isinstance(content, str):
                    text_content = content
                else:
                    try:
                        text_content = json.dumps(content, ensure_ascii=False)
                    except (TypeError, ValueError):
                        text_content = str(content)
                if role == "system":
                    lc_messages.append(classes["system"](content=text_content))
                elif role in {"assistant", "ai"}:
                    lc_messages.append(classes["ai"](content=text_content))
                elif role == "tool" and "tool" in classes:
                    tool_message = classes["tool"](
                        content=text_content,
                        tool_call_id=entry.get("tool_call_id"),
                    )
                    lc_messages.append(tool_message)
                else:
                    lc_messages.append(classes["human"](content=text_content))
        elif prompt:
            lc_messages.append(classes["human"](content=prompt))
        else:
            lc_messages.append(classes["human"](content=""))

        return lc_messages

    def _get_available_lmstudio_models(self, force_refresh: bool = False) -> List[str]:
        """Return the cached list of LM Studio models available on the host."""

        cache_attr = "_available_lmstudio_models"
        cached: Optional[List[str]] = getattr(self.agent_nick, cache_attr, None)
        if force_refresh or cached is None:
            names: List[str] = []
            try:
                client = getattr(self.agent_nick, "lmstudio_client", None) or get_lmstudio_client()
                names = client.list_models()
            except LMStudioClientError as exc:  # pragma: no cover - external dependency failure
                logger.warning("Failed to retrieve available LM Studio models: %s", exc)
            if not names:
                names = list(_LMSTUDIO_FALLBACK_MODELS)
            cached = names
            setattr(self.agent_nick, cache_attr, cached)
        if not cached:
            cached = list(_LMSTUDIO_FALLBACK_MODELS)
            setattr(self.agent_nick, cache_attr, cached)
        return list(cached)

    def _remove_cached_lmstudio_model(self, model_name: str) -> None:
        """Remove a model from the cached LM Studio model list."""

        cache: Optional[List[str]] = getattr(self.agent_nick, "_available_lmstudio_models", None)
        if not cache:
            return
        if model_name not in cache:
            return
        updated = [name for name in cache if name != model_name]
        setattr(self.agent_nick, "_available_lmstudio_models", updated)

    def vector_search(self, query: str, top_k: int = 5):
        """Search the vector database for similar content.

        Falls back gracefully when vector infrastructure is unavailable.
        """
        client = getattr(self.agent_nick, "qdrant_client", None)
        embedder = getattr(self.agent_nick, "embedding_model", None)
        collection = getattr(self.settings, "qdrant_collection_name", None)
        if not all([client, embedder, collection]) or not hasattr(client, "search"):
            return []
        try:
            vec = embedder.encode(query, normalize_embeddings=True).tolist()
            return client.search(
                collection_name=collection,
                query_vector=vec,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("vector search failed")
            return []

    @contextmanager
    def _borrow_s3_client(self):
        reserve = getattr(self.agent_nick, "reserve_s3_connection", None)
        if callable(reserve):
            with reserve() as client:
                yield client
                return
        client = getattr(self.agent_nick, "s3_client", None)
        if client is None:
            raise AttributeError("AgentNick must provide an s3_client")
        yield client

class AgentNick:
    def __init__(self):
        logger.info("AgentNick is waking up...")
        self.settings = settings
        logger.info("Initializing shared clients...")
        self.device = configure_gpu()
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        self._db_engine = None
        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key)
        self.embedding_model = SentenceTransformer(self.settings.embedding_model, device=self.device)
        self.lmstudio_client = get_lmstudio_client()
        self.learning_repository = LearningRepository(self)
        self.static_policy_loader: Optional[StaticPolicyLoader] = None
        s3_pool = max(4, int(getattr(self.settings, "s3_max_pool_connections", 64)))
        self._s3_pool_size = s3_pool
        self.s3_client = boto3.client(
            "s3",
            config=Config(
                max_pool_connections=s3_pool,
                retries={"max_attempts": 10, "mode": "standard"},
            ),
        )
        self._s3_semaphore: Optional[threading.BoundedSemaphore]
        try:
            self._s3_semaphore = threading.BoundedSemaphore(value=s3_pool)
        except Exception:  # pragma: no cover - threading edge cases
            logger.debug("Failed to allocate S3 semaphore", exc_info=True)
            self._s3_semaphore = None
        logger.info("Clients initialized.")

        self._initialise_static_policy_corpus()

        logger.info("Initializing core engines...")
        self.prompt_engine = PromptEngine(self)
        self.policy_engine = PolicyEngine(self)
        self.query_engine = QueryEngine(self)
        self.routing_engine = RoutingEngine(self)
        self.process_routing_service = ProcessRoutingService(self)
        self.workflow_memory = WorkflowMemoryService(self)
        self._agent_model_registry: Optional[Dict[str, str]] = None
        self._agent_model_fallback: Optional[str] = None
        self._build_agent_model_registry()
        logger.info("Engines initialized.")

        self.agents = {}
        self._initialize_qdrant_collection()
        logger.info("AgentNick is ready.")

    def _initialise_static_policy_corpus(self) -> None:
        """Ensure the static policy knowledge base is synchronised."""

        if not getattr(self.settings, "static_policy_auto_ingest", True):
            logger.info("Static policy auto-ingest disabled by configuration")
            return

        try:
            loader = StaticPolicyLoader(self)
        except Exception:  # pragma: no cover - defensive initialisation
            logger.exception("Failed to initialise static policy loader")
            return

        self.static_policy_loader = loader
        try:
            summary = loader.sync_static_policy()
        except Exception:  # pragma: no cover - unexpected ingestion failure
            logger.exception("Static policy ingestion encountered an unexpected error")
            return

        ingested = summary.get("ingested", 0)
        skipped = summary.get("skipped", 0)
        errors: Iterable = summary.get("errors", [])  # type: ignore[assignment]

        if ingested or skipped:
            logger.info(
                "Static policy sync complete: %s ingested, %s skipped", ingested, skipped
            )
        if errors:
            logger.warning("Static policy sync reported issues: %s", errors)

    @property
    def s3_pool_size(self) -> int:
        return getattr(self, "_s3_pool_size", 10)

    @contextmanager
    def reserve_s3_connection(self):
        semaphore = getattr(self, "_s3_semaphore", None)
        if semaphore is None:
            yield self.s3_client
            return
        acquired = False
        try:
            semaphore.acquire()
            acquired = True
            yield self.s3_client
        finally:
            if acquired:
                semaphore.release()

    def get_db_engine(self):
        """Return a cached SQLAlchemy engine when available.

        The procurement platform increasingly relies on pandas for analytic
        workloads. pandas issues warnings when supplied with bare DBAPI
        connections, so we lazily construct a SQLAlchemy engine and reuse it
        across calls.  If SQLAlchemy is unavailable or engine creation fails we
        fall back to the existing psycopg2 connection workflow.
        """

        if self._db_engine is False:
            return None
        if self._db_engine is not None:
            return self._db_engine

        try:  # Import lazily so tests without SQLAlchemy continue to run
            from sqlalchemy import create_engine
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("SQLAlchemy unavailable for engine creation: %s", exc)
            self._db_engine = False
            return None

        try:
            user = quote_plus(getattr(self.settings, "db_user", ""))
            password = quote_plus(getattr(self.settings, "db_password", ""))
            host = getattr(self.settings, "db_host", "localhost")
            port = getattr(self.settings, "db_port", 5432)
            dbname = getattr(self.settings, "db_name", "postgres")
            uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
            self._db_engine = create_engine(uri, pool_pre_ping=True)
        except Exception as exc:  # pragma: no cover - connection misconfig
            logger.warning("Failed to create SQLAlchemy engine: %s", exc)
            self._db_engine = False
            return None

        return self._db_engine

    @contextmanager
    def pandas_connection(self):
        """Yield an object suitable for :func:`pandas.read_sql` calls."""

        engine = self.get_db_engine()
        if engine is not None:
            with engine.connect() as connection:
                yield connection
            return

        conn = self.get_db_connection()
        try:
            yield conn
        finally:  # pragma: no cover - defensive cleanup
            try:
                conn.close()
            except Exception:
                logger.exception("Failed to close DB connection")

    def lmstudio_options(self) -> Dict[str, Any]:
        """Return default options for LM Studio requests."""
        return {"temperature": 0.2, "max_tokens": -1}

    def _build_agent_model_registry(self) -> Dict[str, str]:
        """Compile the agent → model preference map with overrides applied."""

        registry: Dict[str, str] = {}
        overrides = getattr(self.settings, "agent_model_overrides", {}) or {}
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                slug = _slugify_agent_name(key)
                if not slug:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                value = value.strip()
                if value:
                    registry[slug] = value
        for slug, fields in _AGENT_MODEL_FIELD_PREFERENCES.items():
            if slug in registry:
                continue
            for field_name in fields:
                candidate = getattr(self.settings, field_name, None)
                if isinstance(candidate, str) and candidate.strip():
                    registry[slug] = candidate.strip()
                    break
        fallback = getattr(self.settings, "extraction_model", None)
        if isinstance(fallback, str) and fallback.strip():
            self._agent_model_fallback = fallback.strip()
        self._agent_model_registry = registry
        return registry

    def refresh_agent_model_registry(self) -> None:
        """Force regeneration of the cached agent model registry."""

        self._agent_model_registry = None
        self._build_agent_model_registry()

    def get_agent_model(
        self,
        agent_identifier: Any,
        *,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """Return the preferred model for ``agent_identifier``."""

        slug = _slugify_agent_name(agent_identifier)
        registry = self._agent_model_registry
        if registry is None:
            registry = self._build_agent_model_registry()
        if slug:
            model_name = registry.get(slug)
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()
        if fallback is not None:
            return fallback
        if isinstance(self._agent_model_fallback, str) and self._agent_model_fallback:
            return self._agent_model_fallback
        fallback_model = getattr(self.settings, "extraction_model", None)
        if isinstance(fallback_model, str) and fallback_model.strip():
            self._agent_model_fallback = fallback_model.strip()
            return self._agent_model_fallback
        return fallback

    def get_db_connection(self):
        return psycopg2.connect(
            host=self.settings.db_host, dbname=self.settings.db_name,
            user=self.settings.db_user, password=self.settings.db_password,
            port=self.settings.db_port
        )

    def _initialize_qdrant_collection(self):
        collection_name = self.settings.qdrant_collection_name
        required_indexes = {
            "document_type": models.PayloadSchemaType.KEYWORD,
            "product_type": models.PayloadSchemaType.KEYWORD,
            "record_id": models.PayloadSchemaType.KEYWORD,
            "workflow_id": models.PayloadSchemaType.KEYWORD,
            "source_type": models.PayloadSchemaType.KEYWORD,
        }

        def ensure_indexes(schema: Dict[str, Any]) -> None:
            existing_indexes = set(schema.keys()) if schema else set()
            for field_name, field_schema in required_indexes.items():
                if field_name in existing_indexes:
                    continue
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                    logger.info("Successfully created index for '%s'.", field_name)
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    if status_code == HTTPStatus.CONFLICT:
                        logger.debug(
                            "Payload index '%s' already exists for collection '%s'.",
                            field_name,
                            collection_name,
                        )
                    else:
                        logger.warning(
                            "Failed to create payload index '%s': %s",
                            field_name,
                            exc,
                        )

        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=collection_name
            )
            logger.info(
                "Qdrant collection '%s' already exists.",
                collection_name,
            )
            ensure_indexes(getattr(collection_info, "payload_schema", {}) or {})
            return
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code != HTTPStatus.NOT_FOUND:
                logger.warning(
                    "Failed to fetch Qdrant collection '%s': %s",
                    collection_name,
                    exc,
                )
                return

        logger.info("Creating Qdrant collection '%s'...", collection_name)
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.settings.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == HTTPStatus.CONFLICT:
                logger.info(
                    "Qdrant collection '%s' already existed during creation race.",
                    collection_name,
                )
            else:
                logger.warning(
                    "Failed to create Qdrant collection '%s': %s",
                    collection_name,
                    exc,
                )
                return

        try:
            collection_info = self.qdrant_client.get_collection(
                collection_name=collection_name
            )
            ensure_indexes(getattr(collection_info, "payload_schema", {}) or {})
        except Exception:  # pragma: no cover - final guardrail
            logger.exception(
                "Failed to verify payload indexes for Qdrant collection '%s'",
                collection_name,
            )
