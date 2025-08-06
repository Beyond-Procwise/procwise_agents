# ProcWise/config/settings.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

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
    # --- THIS IS THE SINGLE SOURCE OF TRUTH. A NEW NAME TO GUARANTEE A FRESH START. ---
    qdrant_collection_name: str = "procwise_collection_final_v7"

    extraction_model: str = "llama3.2"
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_size: int = 384

    script_user: str = "AgentNick"
    audit_columns: List[str] = ['created_date', 'created_by', 'last_modified_by', 'last_modified_date']

    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = 'utf-8'

try:
    settings = Settings()
except Exception as e:
    print(f"!!! FATAL ERROR: Could not load application settings from .env file: {e}")
    raise
