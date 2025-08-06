# ProcWise/agents/base_agent.py

import boto3
import psycopg2
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from config.settings import settings
# --- Re-add imports for all required engines ---
from orchestration.prompt_engine import PromptEngine
from engines.policy_engine import PolicyEngine
from engines.query_engine import QueryEngine

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        logger.info(f"Initialized agent: {self.__class__.__name__}")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Each agent must implement its own 'run' method.")


class AgentNick:
    def __init__(self):
        logger.info("AgentNick is waking up...")
        self.settings = settings

        logger.info("Initializing shared clients...")
        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key)
        self.embedding_model = SentenceTransformer(self.settings.embedding_model)
        self.s3_client = boto3.client('s3')
        logger.info("Clients initialized.")

        # --- THIS IS THE FIX: Ensure all engines are initialized ---
        logger.info("Initializing core engines...")
        self.prompt_engine = PromptEngine()
        self.policy_engine = PolicyEngine()
        self.query_engine = QueryEngine(self)
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
        """Ensures the Qdrant collection exists and has the correct payload indexes."""
        collection_name = self.settings.qdrant_collection_name
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Qdrant collection '{collection_name}' already exists.")
        except Exception:
            logger.info(f"Creating Qdrant collection '{collection_name}' with payload indexes...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.settings.vector_size, distance=models.Distance.COSINE),
            )
            self.qdrant_client.create_payload_index(collection_name=collection_name, field_name="document_type",
                                                    field_schema=models.PayloadSchemaType.KEYWORD)
            self.qdrant_client.create_payload_index(collection_name=collection_name, field_name="product_type",
                                                    field_schema=models.PayloadSchemaType.KEYWORD)
            logger.info("Collection and payload indexes for 'document_type' and 'product_type' created successfully.")
