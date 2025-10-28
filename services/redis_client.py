"""Utility helpers for working with Redis connections."""

from __future__ import annotations

import logging
import threading
from typing import Optional

try:  # pragma: no cover - optional dependency guard
    import redis
except Exception:  # pragma: no cover - handled gracefully at runtime
    redis = None

from config.settings import settings

logger = logging.getLogger(__name__)

_REDIS_LOCK = threading.Lock()
_REDIS_CLIENT: Optional["redis.Redis"] = None
_WORKFLOW_REDIS_CLIENT: Optional["redis.Redis"] = None


def get_redis_client() -> Optional["redis.Redis"]:
    """Return a cached Redis client for shared infrastructure use."""

    global _REDIS_CLIENT
    if redis is None:  # pragma: no cover - environment without redis installed
        return None

    with _REDIS_LOCK:
        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT

        url = getattr(settings, "redis_url", None)
        if not url:
            return None

        try:
            client = redis.from_url(url)
            client.ping()
        except Exception:  # pragma: no cover - connection issues
            logger.exception("Failed to initialise Redis client for url=%s", url)
            _REDIS_CLIENT = None
            return None

        _REDIS_CLIENT = client
        return _REDIS_CLIENT


def get_workflow_redis_client() -> Optional["redis.Redis"]:
    """Return the Redis client dedicated to workflow session coordination."""

    global _WORKFLOW_REDIS_CLIENT
    if redis is None:  # pragma: no cover - environment without redis installed
        return None

    workflow_url = getattr(settings, "workflow_redis_url", None)
    if not workflow_url:
        return get_redis_client()

    fallback_to_shared = False
    with _REDIS_LOCK:
        if _WORKFLOW_REDIS_CLIENT is not None:
            return _WORKFLOW_REDIS_CLIENT

        try:
            client = redis.from_url(workflow_url)
            client.ping()
        except Exception:  # pragma: no cover - connection issues
            logger.exception(
                "Failed to initialise workflow Redis client for url=%s",
                workflow_url,
            )
            _WORKFLOW_REDIS_CLIENT = None
            fallback_to_shared = True
        else:
            _WORKFLOW_REDIS_CLIENT = client
            return _WORKFLOW_REDIS_CLIENT

    if fallback_to_shared:
        return get_redis_client()

    return None


def reset_redis_client() -> None:
    """Reset the cached Redis client (primarily for tests)."""

    global _REDIS_CLIENT, _WORKFLOW_REDIS_CLIENT
    with _REDIS_LOCK:
        _REDIS_CLIENT = None
        _WORKFLOW_REDIS_CLIENT = None
