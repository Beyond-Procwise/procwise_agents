# ProcWise/config/settings.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional

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
    context_training_data_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "resources", "training", "conversations"),
        env="CONTEXT_TRAINING_DATA_DIR",
    )
    context_model_output_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "models", "context-aware-model"),
        env="CONTEXT_MODEL_OUTPUT_DIR",
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
    llm_backend: str = Field(default="ollama", env="LLM_BACKEND")
    langchain_provider: str = Field(default="ollama", env="LANGCHAIN_PROVIDER")
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
        default="gemma3:1b-it-qat", env="DOCUMENT_EXTRACTION_MODEL"
    )
    rag_model: str = Field(default="gemma3:1b-it-qat", env="RAG_LLM_MODEL")
    # ``BAAI/bge-large-en-v1.5`` provides state-of-the-art dense retrieval
    # performance for procurement terminology while remaining compatible with
    # Qdrant's HNSW indexes. The model outputs 1024 dimensional vectors which
    # improves clustering of nuanced supplier and category data.
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    vector_size: int = 1024
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large", env="RERANKER_MODEL"
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
    ollama_quantized_model: Optional[str] = Field(
        default=None, env="OLLAMA_QUANTIZED_MODEL"
    )
    ollama_gpu_layers: Optional[int] = Field(
        default=None, env="OLLAMA_GPU_LAYERS"
    )
    ollama_num_batch: Optional[int] = Field(default=256, env="OLLAMA_NUM_BATCH")
    ollama_context_window: int = Field(default=8192, env="OLLAMA_CONTEXT_WINDOW")
    ollama_tokenizer: Optional[str] = Field(
        default="llama3", env="OLLAMA_TOKENIZER"
    )
    ollama_adapter: Optional[str] = Field(
        default=None, env="OLLAMA_ADAPTER"
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

try:
    settings = Settings()
except Exception as e:
    print(f"!!! FATAL ERROR: Could not load application settings from .env file: {e}")
    raise
