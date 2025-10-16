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


def get_redis_client() -> Optional["redis.Redis"]:
    """Return a cached Redis client if configuration permits."""

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


def reset_redis_client() -> None:
    """Reset the cached Redis client (primarily for tests)."""

    global _REDIS_CLIENT
    with _REDIS_LOCK:
        _REDIS_CLIENT = None
