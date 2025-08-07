import json
import logging
from typing import Any, Optional
import hashlib
import time

logger = logging.getLogger(__name__)


class CacheService:
    """Simple in-memory cache service (can be replaced with Redis)"""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.settings.cache_enabled:
            return None

        if key in self.cache:
            # Check if expired
            if self._is_expired(key):
                self.delete(key)
                return None

            logger.debug(f"Cache hit: {key}")
            return self.cache[key]

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        if not self.settings.cache_enabled:
            return

        self.cache[key] = value
        self.timestamps[key] = time.time()

        logger.debug(f"Cache set: {key}")

    def delete(self, key: str):
        """Delete from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            logger.debug(f"Cache deleted: {key}")

    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
        logger.debug("Cache cleared")

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True

        age = time.time() - self.timestamps[key]
        return age > self.settings.cache_ttl

    def generate_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()