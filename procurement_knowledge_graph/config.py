from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

try:
    from config.settings import settings as global_settings  # type: ignore
except Exception:  # pragma: no cover - settings import best effort
    global_settings = None  # type: ignore


def _getenv(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        user = _getenv("POSTGRES_USER", "procwise")
        password = _getenv("POSTGRES_PASSWORD", "procwise")
        host = _getenv("POSTGRES_HOST", "localhost")
        port = _getenv("POSTGRES_PORT", "5432")
        database = _getenv("POSTGRES_DB", "procwise")
        return cls(dsn=f"postgresql://{user}:{password}@{host}:{port}/{database}")

    @classmethod
    def from_settings(cls, settings) -> "PostgresConfig":
        dsn = (
            f"postgresql://{settings.db_user}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        )
        return cls(dsn=dsn)


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        host = _getenv("NEO4J_HOST", "bolt://localhost:7687")
        username = _getenv("NEO4J_USERNAME", "neo4j")
        password = _getenv("NEO4J_PASSWORD", "neo4j")
        return cls(uri=host, username=username, password=password)

    @classmethod
    def from_settings(cls, settings) -> "Neo4jConfig":
        return cls(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
        )


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        host = _getenv("QDRANT_HOST", "localhost")
        port = int(_getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY")
        return cls(host=host, port=port, api_key=api_key)

    @classmethod
    def from_settings(cls, settings) -> "QdrantConfig":
        url = settings.qdrant_url
        parsed = urlparse(url)
        host = parsed.hostname or url
        port = parsed.port or 6333
        return cls(host=host, port=port, api_key=settings.qdrant_api_key)


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    generate_model: str
    embedding_model: str
    timeout: int

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        base_url = _getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        generate_model = _getenv("OLLAMA_GENERATE_MODEL", "llama3.2")
        embedding_model = _getenv("OLLAMA_EMBED_MODEL", "all-minilm")
        timeout = int(_getenv("OLLAMA_TIMEOUT", "120"))
        return cls(
            base_url=base_url,
            generate_model=generate_model,
            embedding_model=embedding_model,
            timeout=timeout,
        )

    @classmethod
    def from_settings(cls, settings) -> "OllamaConfig":
        return cls(
            base_url=settings.ollama_base_url,
            generate_model=settings.ollama_generate_model,
            embedding_model=settings.ollama_embedding_model,
            timeout=settings.ollama_timeout,
        )


@dataclass(frozen=True)
class AppConfig:
    postgres: PostgresConfig
    neo4j: Neo4jConfig
    qdrant: QdrantConfig
    ollama: OllamaConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            postgres=PostgresConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
            qdrant=QdrantConfig.from_env(),
            ollama=OllamaConfig.from_env(),
        )

    @classmethod
    def from_settings(cls, settings) -> "AppConfig":
        return cls(
            postgres=PostgresConfig.from_settings(settings),
            neo4j=Neo4jConfig.from_settings(settings),
            qdrant=QdrantConfig.from_settings(settings),
            ollama=OllamaConfig.from_settings(settings),
        )


def app_config_from_global_settings() -> AppConfig:
    if global_settings is None:
        raise RuntimeError(
            "Application settings are not initialised; cannot build knowledge graph configuration."
        )
    return AppConfig.from_settings(global_settings)


__all__ = [
    "AppConfig",
    "PostgresConfig",
    "Neo4jConfig",
    "QdrantConfig",
    "OllamaConfig",
    "app_config_from_global_settings",
]
