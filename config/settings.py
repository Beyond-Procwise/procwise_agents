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

    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="ProcWise_document_embeddings", env="QDRANT_COLLECTION_NAME"
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
    ses_inbound_prefix: str = Field(default="ses/inbound/", env="SES_INBOUND_PREFIX")
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

    extraction_model: str = "llama3.2"
    # ``multi-qa-mpnet-base-dot-v1`` provides high-quality semantic
    # embeddings tailored for question/answer style retrieval.  Its
    # dimensionality (768) is smaller than ``all-roberta-large-v1`` which
    # reduces storage requirements while typically yielding better recall for
    # RAG workflows.
    embedding_model: str = "multi-qa-mpnet-base-dot-v1"
    vector_size: int = 768

    script_user: str = "AgentNick"
    audit_columns: List[str] = ['created_date', 'created_by', 'last_modified_by', 'last_modified_date']

    # Worker and processing settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    parallel_processing: bool = Field(default=False, env="PARALLEL_PROCESSING")

    # Learning and optimization
    enable_learning: bool = Field(default=False, env="ENABLE_LEARNING")

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
