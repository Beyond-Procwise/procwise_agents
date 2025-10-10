# ProcWise/agents/base_agent.py

import boto3
from botocore.config import Config
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus
import threading

import psycopg2
from http import HTTPStatus

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import ollama
from ollama._types import ResponseError

from config.settings import settings
from orchestration.prompt_engine import PromptEngine
from engines.policy_engine import PolicyEngine
from engines.query_engine import QueryEngine
from engines.routing_engine import RoutingEngine
from services.process_routing_service import ProcessRoutingService
from services.learning_repository import LearningRepository
from utils.gpu import configure_gpu

try:  # Optional imports used for dataset persistence
    from models.context_trainer import ConversationDatasetWriter, TrainingConfig
except Exception:  # pragma: no cover - optional dependency failures handled gracefully
    ConversationDatasetWriter = None  # type: ignore[assignment]
    TrainingConfig = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_OLLAMA_FALLBACK_MODELS: Tuple[str, ...] = ("qwen3:30b", "mixtral:8x7b", "gemma3")


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

    def __post_init__(self) -> None:
        # track invocation time and update routing path
        self.timestamp = datetime.utcnow()
        self.routing_history.append(self.agent_id)

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
        logged_output = result.data
        process_id = self.agent_nick.process_routing_service.log_process(
            process_name=self.__class__.__name__,
            process_details={"input": logged_input, "output": logged_output},
            user_id=context.user_id,
            user_name=self.agent_nick.settings.script_user,
            process_status=0,
        )
        if process_id is not None:
            run_id = self.agent_nick.process_routing_service.log_run_detail(
                process_id=process_id,
                process_status=status,
                process_details={"input": logged_input, "output": logged_output},
                process_start_ts=start_ts,
                process_end_ts=end_ts,
                triggered_by=context.user_id,
            )
            action_id = self.agent_nick.process_routing_service.log_action(
                process_id=process_id,
                agent_type=self.__class__.__name__,
                action_desc=logged_input,
                process_output=logged_output,
                status="completed" if result.status == AgentStatus.SUCCESS else "failed",
                run_id=run_id,
            )
            result.action_id = action_id
            result.data["action_id"] = action_id
            drafts = result.data.get("drafts")
            if isinstance(drafts, list):
                for draft in drafts:
                    if isinstance(draft, dict):
                        draft["action_id"] = action_id
        logger.info(
            "%s: completed with status %s", self.__class__.__name__, result.status.value
        )

        self._persist_agentic_plan(context, result)
        try:
            self._record_context_example(context, result)
        except Exception:  # pragma: no cover - dataset capture should never fail hard
            logger.debug("Context dataset capture failed", exc_info=True)

        return result

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

        if any(value for key, value in snapshot.items() if key not in {"workflow_id", "agent_id", "user_id"}):
            context.input_data = dict(context.input_data)
            context.input_data.setdefault("context_snapshot", snapshot)
            context._prepared_context_snapshot = snapshot  # type: ignore[attr-defined]
            return snapshot
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
    def call_ollama(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        format: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Wrapper around :func:`ollama.generate`/``ollama.chat`` used by agents.

        ``messages`` enables use of the chat endpoint which streams tokens faster
        on GPU-enabled systems.  When ``messages`` is provided the ``prompt`` is
        ignored.
        """
        base_model = getattr(self.settings, "extraction_model", "gpt-oss")
        quantized = getattr(self.settings, "ollama_quantized_model", None)

        missing_models = getattr(self, "_missing_ollama_models", None)
        if missing_models is None:
            missing_models = set()
            self._missing_ollama_models = missing_models

        options_from_kwargs = kwargs.pop("options", {})
        base_kwargs = kwargs

        options = {**self.agent_nick.ollama_options(), **options_from_kwargs}
        def _is_gpt_oss(name: Optional[str]) -> bool:
            return bool(name and "gpt-oss" in name)
        if _is_gpt_oss(model) or _is_gpt_oss(base_model) or _is_gpt_oss(quantized):
            options.setdefault("reasoning_effort", "medium")
        optimized_defaults = {}
        gpu_layers = getattr(self.settings, "ollama_gpu_layers", None)
        if gpu_layers is not None:
            optimized_defaults["gpu_layers"] = int(gpu_layers)
        num_batch = getattr(self.settings, "ollama_num_batch", None)
        if num_batch is not None:
            optimized_defaults["num_batch"] = int(num_batch)
        context_window = getattr(self.settings, "ollama_context_window", None)
        if context_window:
            optimized_defaults["num_ctx"] = int(context_window)
        optimized_defaults.setdefault("num_thread", max(1, os.cpu_count() or 1))
        optimized_defaults.setdefault(
            "num_gpu", max(1, int(os.getenv("OLLAMA_NUM_GPU", "1")))
        )
        for key, value in optimized_defaults.items():
            options.setdefault(key, value)

        tokenizer = getattr(self.settings, "ollama_tokenizer", None)
        if tokenizer:
            options.setdefault("tokenizer", tokenizer)
        adapter = getattr(self.settings, "ollama_adapter", None)
        if adapter:
            options.setdefault("adapter", adapter)

        requested_candidates: List[Tuple[str, bool]] = []
        if (
            quantized
            and (model is None or model == base_model)
            and quantized not in missing_models
        ):
            requested_candidates.append((quantized, True))

        base_candidate = model or base_model
        if base_candidate:
            requested_candidates.append((base_candidate, False))

        deduped_candidates: List[Tuple[str, bool]] = []
        seen_names: set[str] = set()
        for name, is_quantized in requested_candidates:
            if name in seen_names:
                continue
            seen_names.add(name)
            deduped_candidates.append((name, is_quantized))
        requested_candidates = deduped_candidates

        fallback_candidates: List[Tuple[str, bool]] = []
        for fallback_name in _OLLAMA_FALLBACK_MODELS:
            if fallback_name in seen_names:
                continue
            seen_names.add(fallback_name)
            fallback_candidates.append((fallback_name, False))

        available_models = self._get_available_ollama_models()
        available_set = set(available_models)

        candidate_models: List[Tuple[str, bool]] = []
        if available_models:
            for name, is_quantized in requested_candidates:
                if name in available_set:
                    candidate_models.append((name, is_quantized))
        if not candidate_models:
            candidate_models.extend(fallback_candidates)
        if not candidate_models and available_models:
            fallback_model = next((m for m in available_models if m), None)
            if fallback_model:
                logger.warning(
                    "Requested Ollama models %s not available; falling back to '%s'.",
                    [name for name, _ in requested_candidates],
                    fallback_model,
                )
                candidate_models.append((fallback_model, False))

        if not candidate_models:
            error_msg = "No available Ollama models detected from 'ollama list'."
            logger.error(error_msg)
            return {"response": "", "error": error_msg}

        last_error: Optional[Exception] = None
        attempted_models: set[str] = set()
        idx = 0
        while idx < len(candidate_models):
            model_to_use, is_quantized = candidate_models[idx]
            idx += 1
            attempted_models.add(model_to_use)
            try:
                attempt_options = dict(options)
                if messages is not None:
                    return ollama.chat(
                        model=model_to_use,
                        messages=messages,
                        options=attempt_options,
                        stream=False,
                        **base_kwargs,
                    )
                return ollama.generate(
                    model=model_to_use,
                    prompt=prompt or "",
                    format=format,
                    stream=False,
                    options=attempt_options,
                    **base_kwargs,
                )
            except ResponseError as exc:
                message = exc.args[0] if exc.args else ""
                status_code = exc.args[1] if len(exc.args) > 1 else None
                is_not_found = "not found" in str(message).lower() or status_code == 404
                if is_quantized and is_not_found:
                    missing_models.add(model_to_use)
                    logger.warning(
                        "Quantized Ollama model '%s' unavailable (status: %s); falling back to '%s'.",
                        model_to_use,
                        status_code,
                        base_model,
                    )
                    self._remove_cached_ollama_model(model_to_use)
                    continue
                if is_not_found:
                    logger.warning(
                        "Ollama model '%s' unavailable (status: %s); refreshing available model list.",
                        model_to_use,
                        status_code,
                    )
                    self._remove_cached_ollama_model(model_to_use)
                    available_models = self._get_available_ollama_models(force_refresh=True)
                    fallback_model = next(
                        (m for m in available_models if m not in attempted_models),
                        None,
                    )
                    if fallback_model:
                        candidate_models.append((fallback_model, False))
                        continue
                last_error = exc
                break
            except Exception as exc:  # pragma: no cover - network / runtime issues
                last_error = exc
                break

        if last_error is not None:
            logger.error("Ollama call failed", exc_info=last_error)
            return {"response": "", "error": str(last_error)}

        # Defensive fallback if the loop completes without returning (should not happen)
        logger.error("Ollama call failed for unknown reasons")
        return {"response": "", "error": "Unknown Ollama invocation failure"}

    def _get_available_ollama_models(self, force_refresh: bool = False) -> List[str]:
        """Return the cached list of Ollama models available on the host."""

        cached: Optional[List[str]] = getattr(self.agent_nick, "_available_ollama_models", None)
        if force_refresh or cached is None:
            names: List[str] = []
            try:
                response = ollama.list()
                models = response.get("models", []) if isinstance(response, dict) else []
                for entry in models:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name") or entry.get("model")
                    if name:
                        names.append(name)
            except Exception as exc:  # pragma: no cover - external dependency failure
                logger.warning("Failed to retrieve available Ollama models: %s", exc)
            if not names:
                names = list(_OLLAMA_FALLBACK_MODELS)
            cached = names
            setattr(self.agent_nick, "_available_ollama_models", cached)
        if not cached:
            cached = list(_OLLAMA_FALLBACK_MODELS)
            setattr(self.agent_nick, "_available_ollama_models", cached)
        return list(cached)

    def _remove_cached_ollama_model(self, model_name: str) -> None:
        """Remove a model from the cached ``ollama list`` response."""

        cache: Optional[List[str]] = getattr(self.agent_nick, "_available_ollama_models", None)
        if not cache:
            return
        if model_name not in cache:
            return
        updated = [name for name in cache if name != model_name]
        setattr(self.agent_nick, "_available_ollama_models", updated)

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
        os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        self._db_engine = None
        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key)
        self.embedding_model = SentenceTransformer(self.settings.embedding_model, device=self.device)
        self.learning_repository = LearningRepository(self)
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

        logger.info("Initializing core engines...")
        self.prompt_engine = PromptEngine(self)
        self.policy_engine = PolicyEngine(self)
        self.query_engine = QueryEngine(self)
        self.routing_engine = RoutingEngine(self)
        self.process_routing_service = ProcessRoutingService(self)
        logger.info("Engines initialized.")

        self.agents = {}
        self._initialize_qdrant_collection()
        logger.info("AgentNick is ready.")

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

    def ollama_options(self) -> Dict[str, Any]:
        """Return default options for Ollama requests respecting GPU availability."""
        if self.device == "cuda":
            return {"num_gpu_layers": -1, "keep_alive": "10m"}
        return {"keep_alive": "10m"}

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
