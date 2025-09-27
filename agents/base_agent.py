# ProcWise/agents/base_agent.py

import boto3
from botocore.config import Config
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
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
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        # track invocation time and update routing path
        self.timestamp = datetime.utcnow()
        self.routing_history.append(self.agent_id)


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

class BaseAgent:
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
            result = self.run(context)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("%s execution failed", self.__class__.__name__)
            result = AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))
        end_ts = datetime.utcnow()

        status = result.status.value
        process_id = self.agent_nick.process_routing_service.log_process(
            process_name=self.__class__.__name__,
            process_details={"input": context.input_data, "output": result.data},
            user_id=context.user_id,
            user_name=self.agent_nick.settings.script_user,
            process_status=0,
        )
        if process_id is not None:
            run_id = self.agent_nick.process_routing_service.log_run_detail(
                process_id=process_id,
                process_status=status,
                process_details={"input": context.input_data, "output": result.data},
                process_start_ts=start_ts,
                process_end_ts=end_ts,
                triggered_by=context.user_id,
            )
            action_id = self.agent_nick.process_routing_service.log_action(
                process_id=process_id,
                agent_type=self.__class__.__name__,
                action_desc=context.input_data,
                process_output=result.data,
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
        return result

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

        candidate_models: List[tuple[str, bool]] = []
        if (
            quantized
            and (model is None or model == base_model)
            and quantized not in missing_models
        ):
            candidate_models.append((quantized, True))
        candidate_models.append((model or base_model, False))

        last_error: Optional[Exception] = None
        for model_to_use, is_quantized in candidate_models:
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
