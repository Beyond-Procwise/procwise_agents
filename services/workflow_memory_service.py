"""Redis-backed workflow memory coordination utilities."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from services.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class WorkflowMemoryService:
    """Persist lightweight workflow context and agent telemetry in Redis."""

    MAX_BYTES = 30 * 1024 * 1024  # 30 MB cap per workflow
    DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours

    def __init__(self, agent_nick: Any, *, redis_client=None) -> None:
        self.agent_nick = agent_nick
        self.redis = redis_client or get_redis_client()
        self.enabled = self.redis is not None

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _meta_key(workflow_id: str) -> str:
        return f"proc:workflow:{workflow_id}:meta"

    @staticmethod
    def _keys_key(workflow_id: str) -> str:
        return f"proc:workflow:{workflow_id}:keys"

    @staticmethod
    def _events_key(workflow_id: str) -> str:
        return f"proc:workflow:{workflow_id}:events"

    @staticmethod
    def _thread_key(workflow_id: str, unique_id: str) -> str:
        return f"proc:workflow:{workflow_id}:thread:{unique_id}"

    # ------------------------------------------------------------------
    # Workflow lifecycle
    # ------------------------------------------------------------------
    def start(self, workflow_id: Optional[str], workflow_name: Optional[str] = None) -> None:
        if not self.enabled or not workflow_id:
            return
        meta_key = self._meta_key(workflow_id)
        now = time.time()
        try:
            pipe = self.redis.pipeline()
            pipe.hset(meta_key, mapping={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name or "",
                "bytes": "0",
                "saturated": "0",
                "started_at": str(now),
            })
            pipe.expire(meta_key, self.DEFAULT_TTL_SECONDS)
            pipe.sadd(self._keys_key(workflow_id), meta_key)
            pipe.expire(self._keys_key(workflow_id), self.DEFAULT_TTL_SECONDS)
            pipe.execute()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to initialise workflow memory for %s", workflow_id, exc_info=True)

    def complete(self, workflow_id: Optional[str], *, success: bool = True) -> None:
        if not self.enabled or not workflow_id:
            return
        keys_key = self._keys_key(workflow_id)
        try:
            stored_keys = self.redis.smembers(keys_key) or []
        except Exception:  # pragma: no cover - defensive
            stored_keys = []
        delete_keys: List[str] = []
        for raw in stored_keys:
            try:
                key = raw.decode() if isinstance(raw, bytes) else str(raw)
            except Exception:
                continue
            if key:
                delete_keys.append(key)
        delete_keys.append(keys_key)
        if success:
            delete_keys.append(self._events_key(workflow_id))
        try:  # pragma: no cover - best effort cleanup
            if delete_keys:
                self.redis.delete(*delete_keys)
        except Exception:
            logger.debug("Failed to clear workflow memory keys for %s", workflow_id, exc_info=True)

    # ------------------------------------------------------------------
    # Core storage helpers
    # ------------------------------------------------------------------
    def _register_key(self, workflow_id: str, redis_key: str) -> None:
        try:
            self.redis.sadd(self._keys_key(workflow_id), redis_key)
            self.redis.expire(self._keys_key(workflow_id), self.DEFAULT_TTL_SECONDS)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to register workflow memory key %s", redis_key, exc_info=True)

    def _increment_bytes(self, workflow_id: str, delta: int) -> None:
        if delta <= 0:
            return
        meta_key = self._meta_key(workflow_id)
        try:
            self.redis.hincrby(meta_key, "bytes", delta)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to increment workflow memory usage for %s", workflow_id, exc_info=True)

    def _can_store(self, workflow_id: str, payload_size: int) -> bool:
        if payload_size <= 0:
            return True
        meta_key = self._meta_key(workflow_id)
        try:
            current_raw = self.redis.hget(meta_key, "bytes")
            current = int(current_raw) if current_raw else 0
        except Exception:  # pragma: no cover - defensive
            current = 0
        if current + payload_size <= self.MAX_BYTES:
            return True
        try:
            saturated = self.redis.hget(meta_key, "saturated")
            already_reported = saturated and saturated.decode() == "1" if isinstance(saturated, bytes) else saturated == "1"
            if not already_reported:
                self.redis.hset(meta_key, "saturated", "1")
                logger.warning(
                    "Workflow memory limit reached for %s; additional events will be skipped",
                    workflow_id,
                )
        except Exception:
            logger.debug("Failed to update saturation flag for workflow %s", workflow_id, exc_info=True)
        return False

    def _store_list_value(self, workflow_id: str, redis_key: str, value: Dict[str, Any]) -> None:
        if not self.enabled or not workflow_id:
            return
        try:
            serialised = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            logger.debug("Unable to serialise workflow memory value for %s", workflow_id, exc_info=True)
            return
        payload_size = len(serialised.encode("utf-8"))
        if not self._can_store(workflow_id, payload_size):
            return
        try:
            pipe = self.redis.pipeline()
            pipe.rpush(redis_key, serialised)
            pipe.expire(redis_key, self.DEFAULT_TTL_SECONDS)
            pipe.execute()
            self._register_key(workflow_id, redis_key)
            self._increment_bytes(workflow_id, payload_size)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to append workflow memory value for %s", workflow_id, exc_info=True)

    # ------------------------------------------------------------------
    # Event capture APIs
    # ------------------------------------------------------------------
    def record_agent_execution(
        self,
        workflow_id: Optional[str],
        *,
        agent_name: str,
        status: str,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not workflow_id or not agent_name:
            return
        payload = {
            "type": "agent_execution",
            "agent": agent_name,
            "status": status,
            "summary": summary or {},
            "timestamp": time.time(),
        }
        self._store_list_value(workflow_id, self._events_key(workflow_id), payload)

    def enqueue_learning_event(
        self,
        workflow_id: Optional[str],
        event: Dict[str, Any],
    ) -> None:
        if not self.enabled or not workflow_id or not isinstance(event, dict):
            return
        payload = {
            "type": "learning_event",
            "timestamp": time.time(),
            "event": event,
        }
        self._store_list_value(workflow_id, self._events_key(workflow_id), payload)

    def record_email_message(
        self,
        workflow_id: Optional[str],
        unique_id: Optional[str],
        message: Dict[str, Any],
    ) -> None:
        if not self.enabled or not workflow_id or not unique_id or not isinstance(message, dict):
            return
        payload = {
            "timestamp": time.time(),
            **message,
        }
        key = self._thread_key(workflow_id, unique_id)
        self._store_list_value(workflow_id, key, payload)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def drain_learning_events(self, workflow_id: Optional[str]) -> List[Dict[str, Any]]:
        if not self.enabled or not workflow_id:
            return []
        key = self._events_key(workflow_id)
        try:
            serialized = self.redis.lrange(key, 0, -1)
            events: List[Dict[str, Any]] = []
            for raw in serialized:
                try:
                    text = raw.decode() if isinstance(raw, bytes) else str(raw)
                    events.append(json.loads(text))
                except Exception:
                    continue
            if events:
                self.redis.delete(key)
            return events
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to drain learning events for %s", workflow_id, exc_info=True)
            return []

    def get_thread_messages(self, workflow_id: Optional[str], unique_id: Optional[str]) -> List[Dict[str, Any]]:
        if not self.enabled or not workflow_id or not unique_id:
            return []
        key = self._thread_key(workflow_id, unique_id)
        try:
            serialized = self.redis.lrange(key, 0, -1)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to load thread history for workflow=%s unique_id=%s", workflow_id, unique_id, exc_info=True)
            return []
        history: List[Dict[str, Any]] = []
        for raw in serialized:
            try:
                text = raw.decode() if isinstance(raw, bytes) else str(raw)
                entry = json.loads(text)
            except Exception:
                continue
            if isinstance(entry, dict):
                history.append(entry)
        return history

    # ------------------------------------------------------------------
    # Utility helpers for summarising payloads
    # ------------------------------------------------------------------
    @staticmethod
    def summarise_payload(payload: Any, *, max_keys: int = 12, max_length: int = 4000) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        summary: Dict[str, Any] = {}
        for idx, (key, value) in enumerate(payload.items()):
            if idx >= max_keys:
                break
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
            elif isinstance(value, (list, tuple)):
                summary[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)}]"
            elif value is None:
                summary[key] = None
            else:
                summary[key] = type(value).__name__
        encoded = json.dumps(summary, default=str, ensure_ascii=False)
        if len(encoded) > max_length:
            summary = {"truncated": True}
        return summary
