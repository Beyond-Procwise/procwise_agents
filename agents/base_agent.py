# ProcWise/agents/base_agent.py

import boto3
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import psycopg2
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import ollama

from config.settings import settings
from orchestration.prompt_engine import PromptEngine
from engines.policy_engine import PolicyEngine
from engines.query_engine import QueryEngine
from engines.routing_engine import RoutingEngine

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

class BaseAgent:
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        logger.info(f"Initialized agent: {self.__class__.__name__}")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Each agent must implement its own 'run' method.")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def call_ollama(self, prompt: str, model: Optional[str] = None,
                    format: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Lightweight wrapper around :func:`ollama.generate` used by agents."""
        model_to_use = model or getattr(self.settings, 'extraction_model', 'llama3')
        try:
            return ollama.generate(
                model=model_to_use,
                prompt=prompt,
                format=format,
                stream=False,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - network / runtime issues
            logger.error("Ollama generation failed: %s", exc)
            return {"response": "", "error": str(exc)}

class AgentNick:
    def __init__(self):
        logger.info("AgentNick is waking up...")
        self.settings = settings
        logger.info("Initializing shared clients...")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key)
        self.embedding_model = SentenceTransformer(self.settings.embedding_model, device=self.device)
        self.s3_client = boto3.client('s3')
        logger.info("Clients initialized.")

        logger.info("Initializing core engines...")
        self.prompt_engine = PromptEngine()
        self.policy_engine = PolicyEngine()
        self.query_engine = QueryEngine(self)
        self.routing_engine = RoutingEngine(self)
        logger.info("Engines initialized.")

        self.agents = {}
        self._initialize_qdrant_collection()
        logger.info("AgentNick is ready.")

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
            required_indexes = {"document_type": models.PayloadSchemaType.KEYWORD, "product_type": models.PayloadSchemaType.KEYWORD}
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
            logger.info("Collection and payload indexes created successfully.")
