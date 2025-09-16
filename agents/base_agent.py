# ProcWise/agents/base_agent.py

import boto3
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import psycopg2
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import ollama

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

    def run(self, *args, **kwargs):
        raise NotImplementedError("Each agent must implement its own 'run' method.")

    def execute(self, context: "AgentContext") -> "AgentOutput":
        """Execute the agent with process logging.

        This centralises writes to ``proc.routing`` and ``proc.action`` so that
        every agent invocation is captured in the database regardless of how it
        is triggered.
        """
        logger.info("%s: starting with input %s", self.__class__.__name__, context.input_data)
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
            result.data.setdefault("action_id", action_id)
            drafts = result.data.get("drafts")
            if isinstance(drafts, list):
                for draft in drafts:
                    draft.setdefault("action_id", action_id)
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
        model_to_use = model or getattr(self.settings, "extraction_model", "llama3")
        try:
            options = kwargs.pop("options", {})
            options = {**self.agent_nick.ollama_options(), **options}
            if messages is not None:
                return ollama.chat(
                    model=model_to_use,
                    messages=messages,
                    options=options,
                    stream=False,
                    **kwargs,
                )
            return ollama.generate(
                model=model_to_use,
                prompt=prompt or "",
                format=format,
                stream=False,
                options=options,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - network / runtime issues
            logger.exception("Ollama call failed")
            return {"response": "", "error": str(exc)}

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
        self.s3_client = boto3.client('s3')
        logger.info("Clients initialized.")

        logger.info("Initializing core engines...")
        self.prompt_engine = PromptEngine()
        self.policy_engine = PolicyEngine()
        self.query_engine = QueryEngine(self)
        self.routing_engine = RoutingEngine(self)
        self.process_routing_service = ProcessRoutingService(self)
        logger.info("Engines initialized.")

        self.agents = {}
        self._initialize_qdrant_collection()
        logger.info("AgentNick is ready.")

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
            yield engine
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
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Qdrant collection '{collection_name}' already exists.")
            existing_indexes = collection_info.payload_schema.keys()
            required_indexes = {
                "document_type": models.PayloadSchemaType.KEYWORD,
                "product_type": models.PayloadSchemaType.KEYWORD,
                "record_id": models.PayloadSchemaType.KEYWORD,
            }
            for field_name, field_schema in required_indexes.items():
                if field_name not in existing_indexes:
                    logger.warning(f"Index for '{field_name}' not found. Creating it now...")
                    self.qdrant_client.create_payload_index(collection_name=collection_name, field_name=field_name, field_schema=field_schema, wait=True)
                    logger.info(f"Successfully created index for '{field_name}'.")
        except Exception:
            logger.info(f"Creating Qdrant collection '{collection_name}'...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.settings.vector_size, distance=models.Distance.COSINE),
            )
            self.qdrant_client.create_payload_index(collection_name=collection_name, field_name="document_type", field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            self.qdrant_client.create_payload_index(collection_name=collection_name, field_name="product_type", field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            self.qdrant_client.create_payload_index(collection_name=collection_name, field_name="record_id", field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            logger.info("Collection and payload indexes created successfully.")
