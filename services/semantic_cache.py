"""Thin wrapper around LangCache semantic caching for embeddings and queries."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:  # pragma: no cover - optional dependency guard
    from langcache import LangCache
except Exception:  # pragma: no cover - dependency is optional at runtime
    LangCache = None  # type: ignore

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """Provide a resilient facade over LangCache-backed semantic caching."""

    def __init__(self, settings, *, namespace: str) -> None:
        self._settings = settings
        self._namespace = namespace
        self._server_url: Optional[str] = None
        self._cache_id: Optional[str] = None
        self._api_key: Optional[str] = None
        self._embedding_ttl_ms: Optional[int] = None
        self._query_ttl_ms: Optional[int] = None
        self._embedding_threshold: float = 0.985
        self._query_threshold: float = 0.92
        self._enabled = False
        self._configure()

    # ------------------------------------------------------------------
    # Public embedding cache helpers
    # ------------------------------------------------------------------
    def get_embedding(self, prompt: str, model_name: str) -> Optional[List[float]]:
        if not self._enabled or not prompt.strip():
            return None

        attributes = {
            "type": "embedding",
            "namespace": self._namespace,
            "model": model_name,
        }
        entries = self._search(prompt, attributes, self._embedding_threshold)
        for entry in entries:
            try:
                payload = json.loads(entry.get("response", ""))
            except json.JSONDecodeError:
                continue
            vector = payload.get("vector")
            if isinstance(vector, list):
                try:
                    return [float(value) for value in vector]
                except (TypeError, ValueError):
                    continue
        return None

    def set_embedding(
        self, prompt: str, vector: Iterable[float], model_name: str
    ) -> None:
        if not self._enabled or not prompt.strip():
            return
        try:
            vector_list = [float(value) for value in vector]
        except (TypeError, ValueError):
            logger.debug("Unable to serialise embedding vector for cache storage")
            return

        payload = json.dumps(
            {
                "vector": vector_list,
                "model": model_name,
                "dimension": len(vector_list),
            }
        )
        attributes = {
            "type": "embedding",
            "namespace": self._namespace,
            "model": model_name,
        }
        self._set(prompt, payload, attributes, ttl_ms=self._embedding_ttl_ms)

    # ------------------------------------------------------------------
    # Public query cache helpers
    # ------------------------------------------------------------------
    def get_cached_queries(self, prompt: str) -> List[Dict[str, Any]]:
        if not self._enabled or not prompt.strip():
            return []
        attributes = {
            "type": "rag_query",
            "namespace": self._namespace,
        }
        return self._search(prompt, attributes, self._query_threshold)

    def set_query_results(self, prompt: str, results: List[Dict[str, Any]]) -> None:
        if not self._enabled or not prompt.strip():
            return
        try:
            payload = json.dumps({"hits": results})
        except (TypeError, ValueError):
            logger.debug("Failed to serialise query results for LangCache storage")
            return
        attributes = {
            "type": "rag_query",
            "namespace": self._namespace,
        }
        self._set(prompt, payload, attributes, ttl_ms=self._query_ttl_ms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _configure(self) -> None:
        settings = self._settings
        if settings is None:
            return

        self._server_url = getattr(settings, "langcache_server_url", None)
        self._cache_id = getattr(settings, "langcache_cache_id", None)
        self._api_key = getattr(settings, "langcache_api_key", None)
        embedding_ttl = getattr(settings, "langcache_embedding_ttl_seconds", None)
        query_ttl = getattr(settings, "langcache_query_ttl_seconds", None)
        embedding_threshold = getattr(
            settings, "langcache_embedding_similarity_threshold", None
        )
        query_threshold = getattr(
            settings, "langcache_query_similarity_threshold", None
        )

        if embedding_ttl is not None:
            try:
                self._embedding_ttl_ms = int(max(0, embedding_ttl)) * 1000
            except (TypeError, ValueError):
                self._embedding_ttl_ms = None
        if query_ttl is not None:
            try:
                self._query_ttl_ms = int(max(0, query_ttl)) * 1000
            except (TypeError, ValueError):
                self._query_ttl_ms = None
        if embedding_threshold is not None:
            try:
                self._embedding_threshold = float(embedding_threshold)
            except (TypeError, ValueError):
                pass
        if query_threshold is not None:
            try:
                self._query_threshold = float(query_threshold)
            except (TypeError, ValueError):
                pass

        self._enabled = bool(self._server_url and self._cache_id and LangCache is not None)

    @contextmanager
    def _client(self) -> Iterator[Optional[LangCache]]:  # type: ignore[type-arg]
        if not self._enabled or LangCache is None:
            yield None
            return
        try:
            with LangCache(
                server_url=self._server_url or "",
                cache_id=self._cache_id,
                api_key=self._api_key,
            ) as client:
                yield client
        except Exception:
            logger.debug("LangCache operation failed", exc_info=True)
            yield None

    def _search(
        self,
        prompt: str,
        attributes: Dict[str, str],
        similarity: Optional[float],
    ) -> List[Dict[str, Any]]:
        if not self._enabled:
            return []

        with self._client() as client:
            if client is None:
                return []
            try:
                response = client.search(
                    prompt=prompt,
                    similarity_threshold=similarity,
                    attributes=attributes,
                )
            except Exception:
                logger.debug("LangCache search failed", exc_info=True)
                return []

        data = []
        if response is None:
            return data

        entries = getattr(response, "data", None)
        if not isinstance(entries, list):
            return data

        for entry in entries:
            payload = {
                "id": getattr(entry, "id", None),
                "prompt": getattr(entry, "prompt", None),
                "response": getattr(entry, "response", None),
                "similarity": getattr(entry, "similarity", None),
                "attributes": getattr(entry, "attributes", {}),
            }
            data.append(payload)
        return data

    def _set(
        self,
        prompt: str,
        payload: str,
        attributes: Dict[str, str],
        *,
        ttl_ms: Optional[int],
    ) -> None:
        if not self._enabled:
            return

        with self._client() as client:
            if client is None:
                return
            try:
                client.set(
                    prompt=prompt,
                    response=payload,
                    attributes=attributes,
                    ttl_millis=ttl_ms,
                )
            except Exception:
                logger.debug("LangCache set operation failed", exc_info=True)

