# ProcWise/config/settings.py

import json
import os
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')

class Settings(BaseSettings):
    db_host: str = Field(..., env="DB_HOST")
    db_name: str = Field(..., env="DB_NAME")
    db_user: str = Field(..., env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")
    db_port: int = Field(..., env="DB_PORT")

    s3_bucket_name: str = Field(..., env="S3_BUCKET_NAME")
    s3_prefixes: List[str] = Field(..., env="S3_PREFIXES")
    s3_max_pool_connections: int = Field(
        default=64, env="S3_MAX_POOL_CONNECTIONS"
    )

    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="procwise_document_embeddings", env="QDRANT_COLLECTION_NAME"
    )
    uploaded_documents_collection_name: str = Field(
        default="uploaded_documents", env="UPLOADED_DOCUMENTS_COLLECTION_NAME"
    )
    knowledge_graph_collection_name: str = Field(
        default="procwise_knowledge_graph", env="KNOWLEDGE_GRAPH_COLLECTION_NAME"
    )
    learning_collection_name: str = Field(
        default="learning", env="LEARNING_COLLECTION_NAME"
    )
    static_policy_collection_name: str = Field(
        default="static_policy", env="STATIC_POLICY_COLLECTION_NAME"
    )
    static_policy_s3_prefix: str = Field(
        default="Static Policy/", env="STATIC_POLICY_S3_PREFIX"
    )
    static_policy_s3_bucket: Optional[str] = Field(
        default=None, env="STATIC_POLICY_S3_BUCKET"
    )
    static_policy_auto_ingest: bool = Field(
        default=True, env="STATIC_POLICY_AUTO_INGEST"
    )
    llamaparse_api_key: Optional[str] = Field(default=None, env="LLAMAPARSE_API_KEY")
    llamaparse_base_url: Optional[str] = Field(default=None, env="LLAMAPARSE_BASE_URL")
    context_training_data_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "resources", "training", "conversations"),
        env="CONTEXT_TRAINING_DATA_DIR",
    )
    context_model_output_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "models", "context-aware-model"),
        env="CONTEXT_MODEL_OUTPUT_DIR",
    )

    instruction_training_enabled: bool = Field(
        default=True, env="INSTRUCTION_TRAINING_ENABLED"
    )
    instruction_training_query: str = Field(
        default=(
            "SELECT example_id, query AS instruction, response AS output "
            "FROM proc.rag_training_examples "
            "WHERE sentiment = 'positive' AND confidence >= 0.7 "
            "AND COALESCE(used_in_training, FALSE) = FALSE"
        ),
        env="INSTRUCTION_TRAINING_QUERY",
    )
    instruction_training_dataset_path: str = Field(
        default=os.path.join(
            PROJECT_ROOT,
            "resources",
            "training",
            "instruction",
            "train.jsonl",
        ),
        env="INSTRUCTION_TRAINING_DATASET_PATH",
    )
    instruction_training_output_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "models", "instruction", "adapters"),
        env="INSTRUCTION_TRAINING_OUTPUT_DIR",
    )
    instruction_training_merged_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "models", "instruction", "merged"),
        env="INSTRUCTION_TRAINING_MERGED_DIR",
    )
    instruction_training_llama_cpp_dir: Optional[str] = Field(
        default=None, env="INSTRUCTION_TRAINING_LLAMA_CPP_DIR"
    )
    instruction_training_gguf_output: str = Field(
        default=os.path.join(
            PROJECT_ROOT,
            "models",
            "instruction",
            "gguf",
            "model-f16.gguf",
        ),
        env="INSTRUCTION_TRAINING_GGUF_OUTPUT",
    )
    instruction_training_quantized_output: Optional[str] = Field(
        default=os.path.join(
            PROJECT_ROOT,
            "models",
            "instruction",
            "gguf",
            "model-Q4_K_M.gguf",
        ),
        env="INSTRUCTION_TRAINING_QUANTIZED_OUTPUT",
    )
    instruction_training_quantize_preset: Optional[str] = Field(
        default="Q4_K_M", env="INSTRUCTION_TRAINING_QUANTIZE_PRESET"
    )
    instruction_training_base_model: str = Field(
        default="mistralai/Mistral-7B-v0.3", env="INSTRUCTION_TRAINING_BASE_MODEL"
    )
    instruction_training_system_prompt: Optional[str] = Field(
        default=None, env="INSTRUCTION_TRAINING_SYSTEM_PROMPT"
    )
    instruction_training_chat_template: Optional[str] = Field(
        default=None, env="INSTRUCTION_TRAINING_CHAT_TEMPLATE"
    )
    instruction_training_use_unsloth: bool = Field(
        default=False, env="INSTRUCTION_TRAINING_USE_UNSLOTH"
    )
    instruction_training_chunk_size: int = Field(
        default=1_000, env="INSTRUCTION_TRAINING_CHUNK_SIZE"
    )
    instruction_training_id_column: str = Field(
        default="example_id", env="INSTRUCTION_TRAINING_ID_COLUMN"
    )
    instruction_training_min_records: int = Field(
        default=25, env="INSTRUCTION_TRAINING_MIN_RECORDS"
    )
    instruction_training_safe_serialization: bool = Field(
        default=True, env="INSTRUCTION_TRAINING_SAFE_SERIALIZATION"
    )
    instruction_training_train_overrides: Dict[str, Any] = Field(
        default_factory=dict, env="INSTRUCTION_TRAINING_TRAIN_OVERRIDES"
    )
    instruction_training_merge_overrides: Dict[str, Any] = Field(
        default_factory=dict, env="INSTRUCTION_TRAINING_MERGE_OVERRIDES"
    )
    instruction_training_gguf_overrides: Dict[str, Any] = Field(
        default_factory=dict, env="INSTRUCTION_TRAINING_GGUF_OVERRIDES"
    )

    # Email settings
    ses_smtp_secret_name: str = Field(
        default="ses/smtp/credentials", env="SES_SMTP_SECRET_NAME"
    )
    ses_smtp_endpoint: str = Field(
        default="email-smtp.eu-west-1.amazonaws.com", env="SES_SMTP_ENDPOINT"
    )
    ses_smtp_port: int = Field(default=587, env="SES_SMTP_PORT")
    ses_default_sender: str = Field(..., env="SES_DEFAULT_SENDER")
    ses_inbound_bucket: Optional[str] = Field(default=None, env="SES_INBOUND_BUCKET")
    ses_inbound_prefix: str = Field(default="emails/", env="SES_INBOUND_PREFIX")
    ses_inbound_s3_uri: Optional[str] = Field(default=None, env="SES_INBOUND_S3_URI")
    ses_inbound_queue_url: Optional[str] = Field(default=None, env="SES_INBOUND_QUEUE_URL")
    ses_inbound_queue_wait_seconds: int = Field(
        default=2, env="SES_INBOUND_QUEUE_WAIT_SECONDS"
    )
    ses_inbound_queue_max_messages: int = Field(
        default=10, env="SES_INBOUND_QUEUE_MAX_MESSAGES"
    )

    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    workflow_redis_url: Optional[str] = Field(
        default=None, env="WORKFLOW_REDIS_URL"
    )
    langcache_api_key: Optional[str] = Field(
        default=None, env="LANGCACHE_API_KEY"
    )
    langcache_server_url: Optional[str] = Field(
        default=None, env="LANGCACHE_SERVER_URL"
    )
    langcache_cache_id: Optional[str] = Field(
        default=None, env="LANGCACHE_CACHE_ID"
    )
    langcache_embedding_ttl_seconds: int = Field(
        default=604800, env="LANGCACHE_EMBEDDING_TTL_SECONDS"
    )
    langcache_query_ttl_seconds: int = Field(
        default=86400, env="LANGCACHE_QUERY_TTL_SECONDS"
    )
    langcache_embedding_similarity_threshold: float = Field(
        default=0.985, env="LANGCACHE_EMBEDDING_SIMILARITY_THRESHOLD"
    )
    langcache_query_similarity_threshold: float = Field(
        default=0.93, env="LANGCACHE_QUERY_SIMILARITY_THRESHOLD"
    )
    redis_response_ttl_seconds: int = Field(
        default=86400, env="REDIS_RESPONSE_TTL_SECONDS"
    )
    chat_history_cache_ttl: float = Field(
        default=15.0, env="CHAT_HISTORY_CACHE_TTL"
    )
    chat_history_cache_max_entries: int = Field(
        default=256, env="CHAT_HISTORY_CACHE_MAX_ENTRIES"
    )

    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(default="neo4j", env="NEO4J_PASSWORD")

    lmstudio_base_url: str = Field(
        default="http://127.0.0.1:1234", env="LMSTUDIO_BASE_URL"
    )
    lmstudio_chat_model: str = Field(
        default="microsoft/phi-4-reasoning-plus", env="LMSTUDIO_CHAT_MODEL"
    )
    lmstudio_embedding_model: str = Field(
        default="nomic-embed-text", env="LMSTUDIO_EMBED_MODEL"
    )
    lmstudio_timeout: int = Field(default=120, env="LMSTUDIO_TIMEOUT")
    lmstudio_api_key: Optional[str] = Field(
        default=None, env="LMSTUDIO_API_KEY"
    )

    ses_inbound_role_arn: Optional[str] = Field(default=None, env="SES_INBOUND_ROLE_ARN")
    ses_region: Optional[str] = Field(default="eu-west-1", env="SES_REGION")
    ses_secret_role_arn: Optional[str] = Field(
        default=None, env="SES_SECRET_ROLE_ARN"
    )
    ses_smtp_iam_user: Optional[str] = Field(
        default=None, env="SES_SMTP_IAM_USER"
    )
    ses_smtp_propagation_attempts: int = Field(
        default=6, env="SES_SMTP_PROPAGATION_ATTEMPTS"
    )
    ses_smtp_propagation_wait_seconds: int = Field(
        default=30, env="SES_SMTP_PROPAGATION_WAIT_SECONDS"
    )
    email_response_poll_seconds: int = Field(
        default=60, env="EMAIL_RESPONSE_POLL_SECONDS"
    )
    email_inbound_initial_wait_seconds: int = Field(
        default=60, env="EMAIL_INBOUND_INITIAL_WAIT_SECONDS"
    )

    negotiation_multi_round_enabled: bool = Field(
        default=True, env="NEGOTIATION_MULTI_ROUND_ENABLED"
    )
    negotiation_max_rounds: int = Field(
        default=3, env="NEGOTIATION_MAX_ROUNDS"
    )
    negotiation_round_base_timeout: int = Field(
        default=900, env="NEGOTIATION_ROUND_BASE_TIMEOUT"
    )
    negotiation_per_supplier_timeout: int = Field(
        default=300, env="NEGOTIATION_PER_SUPPLIER_TIMEOUT"
    )
    negotiation_max_round_timeout: int = Field(
        default=3600, env="NEGOTIATION_MAX_ROUND_TIMEOUT"
    )

    # IMAP mailbox configuration
    imap_host: Optional[str] = Field(default=None, env="IMAP_HOST")
    imap_port: Optional[int] = Field(default=None, env="IMAP_PORT")
    imap_user: Optional[str] = Field(default=None, env="IMAP_USER")
    imap_username: Optional[str] = Field(default=None, env="IMAP_USERNAME")
    imap_domain: Optional[str] = Field(default=None, env="IMAP_DOMAIN")
    imap_login: Optional[str] = Field(default=None, env="IMAP_LOGIN")
    imap_password: Optional[str] = Field(default=None, env="IMAP_PASSWORD")
    imap_mailbox: str = Field(default="INBOX", env="IMAP_MAILBOX")
    imap_search_criteria: str = Field(default="ALL", env="IMAP_SEARCH_CRITERIA")

    extraction_model: str = "gpt-oss:20b"
    llm_backend: str = Field(default="lmstudio", env="LLM_BACKEND")
    langchain_provider: str = Field(default="lmstudio", env="LANGCHAIN_PROVIDER")
    langchain_model: Optional[str] = Field(
        default="qwen3:30b", env="LANGCHAIN_MODEL"
    )
    langchain_api_base: Optional[str] = Field(
        default=None, env="LANGCHAIN_API_BASE"
    )
    langchain_api_key: Optional[str] = Field(
        default=None, env="LANGCHAIN_API_KEY"
    )
    langchain_request_timeout: Optional[int] = Field(
        default=120, env="LANGCHAIN_REQUEST_TIMEOUT"
    )
    hitl_enabled: bool = Field(default=True, env="HITL_ENABLED")
    data_extraction_staging_schema: str = Field(
        default="proc_stage", env="DATA_EXTRACTION_STAGING_SCHEMA"
    )
    memory_store_uri: Optional[str] = Field(
        default=None, env="MEMORY_STORE_URI"
    )
    langgraph_tracing_enabled: bool = Field(
        default=False, env="LANGGRAPH_TRACING_ENABLED"
    )
    document_extraction_model: str = Field(
        default="llama3.2", env="DOCUMENT_EXTRACTION_MODEL"
    )
    rag_model: str = Field(default="qwen3:30b", env="RAG_LLM_MODEL")
    # ``BAAI/bge-large-en-v1.5`` provides state-of-the-art dense retrieval
    # performance for procurement terminology while remaining compatible with
    # Qdrant's HNSW indexes. The model outputs 1024 dimensional vectors which
    # improves clustering of nuanced supplier and category data.
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    vector_size: int = 1024
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large", env="RERANKER_MODEL"
    )
    rag_prefetch_limit: int = Field(
        default=48,
        env="RAG_PREFETCH_LIMIT",
        description="Upper bound for candidate documents fetched per query during RAG search.",
    )
    rag_reranker_batch_size: int = Field(
        default=32,
        env="RAG_RERANKER_BATCH_SIZE",
        description="Maximum number of document-query pairs scored per reranker batch.",
    )
    rag_reranker_max_chars: int = Field(
        default=1400,
        env="RAG_RERANKER_MAX_CHARS",
        description="Character limit applied to candidate passages before reranking to keep inference responsive.",
    )
    rag_reranker_cache_size: int = Field(
        default=384,
        env="RAG_RERANKER_CACHE_SIZE",
        description="Number of cross-encoder scores memoised to skip redundant reranking calls.",
    )
    rag_qdrant_search_ef: int = Field(
        default=160,
        env="RAG_QDRANT_SEARCH_EF",
        description="HNSW exploration factor for vector lookups; lower values speed up retrieval while trading slight recall.",
    )
    rag_qdrant_search_workers: int = Field(
        default=4,
        env="RAG_QDRANT_SEARCH_WORKERS",
        description="Maximum number of Qdrant collections queried in parallel during retrieval.",
    )
    procurement_knowledge_path: Optional[str] = Field(
        default=None, env="PROCUREMENT_KNOWLEDGE_PATH"
    )
    rag_chunk_chars: int = Field(default=1800, env="RAG_CHUNK_CHARS")
    rag_chunk_overlap: int = Field(default=350, env="RAG_CHUNK_OVERLAP")
    rag_chunk_min_tokens: int = Field(default=320, env="RAG_CHUNK_MIN_TOKENS")
    rag_chunk_max_tokens: int = Field(default=880, env="RAG_CHUNK_MAX_TOKENS")
    rag_chunk_overlap_ratio: float = Field(
        default=0.12, env="RAG_CHUNK_OVERLAP_RATIO"
    )
    agent_model_overrides: Dict[str, str] = Field(
        default_factory=dict, env="AGENT_MODEL_OVERRIDES"
    )
    stream_llm_responses: bool = Field(
        default=False, env="STREAM_LLM_RESPONSES"
    )

    script_user: str = "AgentNick"
    audit_columns: List[str] = ['created_date', 'created_by', 'last_modified_by', 'last_modified_date']

    # Worker and processing settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    parallel_processing: bool = Field(default=False, env="PARALLEL_PROCESSING")
    data_extraction_max_workers: int = Field(
        default=16, env="DATA_EXTRACTION_MAX_WORKERS"
    )
    force_ocr_vendors: List[str] = Field(
        default_factory=list, env="FORCE_OCR_VENDORS"
    )

    # Learning and optimization
    enable_learning: bool = Field(default=False, env="ENABLE_LEARNING")
    enable_training_scheduler: bool = Field(
        default=False, env="ENABLE_TRAINING_SCHEDULER"
    )
    verbose_agent_debug: bool = Field(
        default=False, env="VERBOSE_AGENT_DEBUG"
    )

    # Caching settings
    cache_enabled: bool = Field(default=False, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # Auditing settings
    audit_enabled: bool = Field(default=True, env="AUDIT_ENABLED")

    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = 'utf-8'
        extra = "ignore"

    @staticmethod
    def _parse_mapping(value: Any) -> Dict[str, Any]:
        if value in (None, "", {}):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("Value must be valid JSON mapping") from exc
            if not isinstance(parsed, dict):
                raise ValueError("JSON value must decode to an object")
            return parsed
        raise TypeError("Unsupported type; expected dict or JSON string")

    @field_validator("agent_model_overrides", mode="before")
    @classmethod
    def _coerce_agent_model_overrides(cls, value):
        """Normalise overrides supplied via environment variables."""

        parsed = cls._parse_mapping(value)
        return {
            str(key).strip(): str(val).strip()
            for key, val in parsed.items()
            if str(key).strip() and str(val).strip()
        }

    @field_validator(
        "instruction_training_train_overrides",
        "instruction_training_merge_overrides",
        "instruction_training_gguf_overrides",
        mode="before",
    )
    @classmethod
    def _coerce_instruction_overrides(cls, value):
        parsed = cls._parse_mapping(value)
        return parsed

try:
    settings = Settings()
except Exception as e:
    print(f"!!! FATAL ERROR: Could not load application settings from .env file: {e}")
    raise
