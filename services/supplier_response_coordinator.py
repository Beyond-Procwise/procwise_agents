"""Event-driven coordination utilities for supplier responses."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from config.settings import settings
from services.event_bus import get_event_bus
from services.redis_client import get_redis_client

logger = logging.getLogger(__name__)


@dataclass
class ResponseState:
    """Structured state returned by :class:`SupplierResponseCoordinator`."""

    workflow_id: str
    expected_unique_ids: List[str] = field(default_factory=list)
    collected_unique_ids: List[str] = field(default_factory=list)
    pending_unique_ids: List[str] = field(default_factory=list)
    expected_count: int = 0
    status: str = "pending"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "expected_unique_ids": list(self.expected_unique_ids),
            "collected_unique_ids": list(self.collected_unique_ids),
            "pending_unique_ids": list(self.pending_unique_ids),
            "expected_count": int(self.expected_count or 0),
            "collected_count": len(self.collected_unique_ids),
            "status": self.status,
        }

    @property
    def complete(self) -> bool:
        return self.status == "complete" and not self.pending_unique_ids


class BaseResponseCoordinator:
    """Abstract interface for coordinating supplier response completion."""

    def register_expected_responses(
        self, workflow_id: str, unique_ids: Sequence[str], expected_count: int
    ) -> ResponseState:
        raise NotImplementedError

    def record_response(self, workflow_id: str, unique_id: str) -> ResponseState:
        raise NotImplementedError

    def await_completion(self, workflow_id: str, timeout: Optional[float]) -> ResponseState:
        raise NotImplementedError

    def clear(self, workflow_id: str) -> None:
        pass


class _WorkflowEntry:
    def __init__(self) -> None:
        self.expected_unique_ids: List[str] = []
        self.expected_count: int = 0
        self.collected: Set[str] = set()
        self.event = threading.Event()

    def reset(self, expected_ids: Sequence[str], expected_count: int) -> None:
        self.expected_unique_ids = [uid for uid in expected_ids if uid]
        self.expected_count = max(expected_count, len(self.expected_unique_ids))
        self.collected.intersection_update(self.expected_unique_ids)
        if not self.expected_unique_ids:
            self.event.set()
        elif self.is_complete:
            self.event.set()
        else:
            self.event.clear()

    @property
    def is_complete(self) -> bool:
        if self.expected_unique_ids:
            return set(self.expected_unique_ids).issubset(self.collected)
        if self.expected_count > 0:
            return len(self.collected) >= self.expected_count
        return False

    def to_state(self, workflow_id: str, status: Optional[str] = None) -> ResponseState:
        pending = [uid for uid in self.expected_unique_ids if uid not in self.collected]
        collected: List[str]
        if self.expected_unique_ids:
            collected = [uid for uid in self.expected_unique_ids if uid in self.collected]
        else:
            collected = sorted(self.collected)
        state = ResponseState(
            workflow_id=workflow_id,
            expected_unique_ids=list(self.expected_unique_ids),
            collected_unique_ids=collected,
            pending_unique_ids=pending,
            expected_count=self.expected_count,
            status=status or ("complete" if not pending and self.expected_unique_ids else "pending"),
        )
        return state


class InMemoryResponseCoordinator(BaseResponseCoordinator):
    """Thread-safe in-memory coordinator used when Redis is unavailable."""

    def __init__(self) -> None:
        self._entries: Dict[str, _WorkflowEntry] = {}
        self._lock = threading.RLock()
        self._event_bus = get_event_bus()

    def register_expected_responses(
        self, workflow_id: str, unique_ids: Sequence[str], expected_count: int
    ) -> ResponseState:
        key = str(workflow_id)
        with self._lock:
            entry = self._entries.setdefault(key, _WorkflowEntry())
            entry.reset(unique_ids, expected_count)
            state = entry.to_state(key)
            if entry.is_complete:
                state.status = "complete"
        return state

    def record_response(self, workflow_id: str, unique_id: str) -> ResponseState:
        key = str(workflow_id)
        uid = str(unique_id)
        with self._lock:
            entry = self._entries.setdefault(key, _WorkflowEntry())
            entry.collected.add(uid)
            state = entry.to_state(key)
            if entry.is_complete:
                entry.event.set()
                state.status = "complete"
                try:
                    self._event_bus.publish(
                        f"workflow:{key}:complete", {"workflow_id": key, "status": "complete"}
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception("In-memory coordinator publish failed for workflow=%s", key)
        return state

    def await_completion(self, workflow_id: str, timeout: Optional[float]) -> ResponseState:
        key = str(workflow_id)
        with self._lock:
            entry = self._entries.setdefault(key, _WorkflowEntry())
            state = entry.to_state(key)
            if entry.is_complete:
                state.status = "complete"
                return state

        wait_timeout = None if timeout is None else max(0.0, float(timeout))
        completed = entry.event.wait(wait_timeout)
        with self._lock:
            state = entry.to_state(key, status="complete" if completed else "timeout")
        return state

    def clear(self, workflow_id: str) -> None:
        key = str(workflow_id)
        with self._lock:
            self._entries.pop(key, None)


class RedisResponseCoordinator(BaseResponseCoordinator):
    """Redis-backed implementation supporting distributed workers."""

    def __init__(self, redis_client, *, ttl_seconds: int) -> None:
        if redis_client is None:
            raise RuntimeError("Redis client unavailable")
        self.redis = redis_client
        self.ttl_seconds = max(60, int(ttl_seconds))

    @staticmethod
    def _responses_key(workflow_id: str) -> str:
        return f"workflow:{workflow_id}:responses"

    @staticmethod
    def _meta_key(workflow_id: str) -> str:
        return f"workflow:{workflow_id}:meta"

    @staticmethod
    def _completion_channel(workflow_id: str) -> str:
        return f"workflow:{workflow_id}:complete"

    def register_expected_responses(
        self, workflow_id: str, unique_ids: Sequence[str], expected_count: int
    ) -> ResponseState:
        key = str(workflow_id)
        expected = [str(uid) for uid in unique_ids if uid]
        count = max(int(expected_count or 0), len(expected))
        responses_key = self._responses_key(key)
        meta_key = self._meta_key(key)
        try:
            existing_raw = self.redis.hgetall(responses_key)
        except Exception:
            logger.exception("Failed to load existing response hash for workflow=%s", key)
            existing_raw = {}
        existing = {self._decode(k): self._decode(v) for k, v in existing_raw.items()}
        pipe = self.redis.pipeline()
        pipe.delete(responses_key)
        if expected:
            mapping = {
                uid: ("complete" if existing.get(uid) == "complete" else "pending")
                for uid in expected
            }
            pipe.hset(responses_key, mapping=mapping)
        pipe.expire(responses_key, self.ttl_seconds)
        meta_payload = json.dumps(
            {"expected_unique_ids": expected, "expected_count": count}
        )
        pipe.set(meta_key, meta_payload, ex=self.ttl_seconds)
        pipe.execute()
        state = self._collect_state(key)
        if not state.pending_unique_ids and expected:
            state.status = "complete"
        return state

    def record_response(self, workflow_id: str, unique_id: str) -> ResponseState:
        key = str(workflow_id)
        uid = str(unique_id)
        responses_key = self._responses_key(key)
        meta_key = self._meta_key(key)
        pipe = self.redis.pipeline()
        pipe.hset(responses_key, uid, "complete")
        pipe.expire(responses_key, self.ttl_seconds)
        pipe.execute()
        try:
            meta_exists = bool(self.redis.exists(meta_key))
        except Exception:
            logger.exception("Failed to verify Redis meta state for workflow=%s", key)
            meta_exists = False
        state = self._collect_state(key)
        if meta_exists and not state.pending_unique_ids and state.expected_unique_ids:
            self._publish_completion(key)
            state.status = "complete"
        return state

    def await_completion(self, workflow_id: str, timeout: Optional[float]) -> ResponseState:
        key = str(workflow_id)
        state = self._collect_state(key)
        if not state.pending_unique_ids and state.expected_unique_ids:
            state.status = "complete"
            return state

        deadline = None if timeout is None else time.time() + max(0.0, float(timeout))
        channel = self._completion_channel(key)
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)
        try:
            while True:
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    message = pubsub.get_message(timeout=min(1.0, max(0.01, remaining)))
                else:
                    message = pubsub.get_message(timeout=1.0)
                if not message:
                    state = self._collect_state(key)
                    if not state.pending_unique_ids and state.expected_unique_ids:
                        state.status = "complete"
                        return state
                    continue
                if message.get("type") != "message":
                    continue
                state = self._collect_state(key)
                if not state.pending_unique_ids and state.expected_unique_ids:
                    state.status = "complete"
                    return state
        finally:
            try:
                pubsub.close()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.exception("Failed to close Redis pubsub for workflow=%s", key)

        state = self._collect_state(key)
        if not state.pending_unique_ids and state.expected_unique_ids:
            state.status = "complete"
        else:
            state.status = "timeout"
        return state

    def clear(self, workflow_id: str) -> None:
        key = str(workflow_id)
        responses_key = self._responses_key(key)
        meta_key = self._meta_key(key)
        try:
            self.redis.delete(responses_key)
            self.redis.delete(meta_key)
        except Exception:  # pragma: no cover - defensive cleanup
            logger.exception("Failed to clear Redis workflow state for %s", key)

    def _collect_state(self, workflow_id: str) -> ResponseState:
        responses_key = self._responses_key(workflow_id)
        meta_key = self._meta_key(workflow_id)
        try:
            raw_status = self.redis.hgetall(responses_key)
        except Exception:
            logger.exception("Failed to read Redis response hash for workflow=%s", workflow_id)
            raw_status = {}
        status_map = {self._decode(k): self._decode(v) for k, v in raw_status.items()}

        expected_ids: List[str] = []
        expected_count = 0
        try:
            meta_raw = self.redis.get(meta_key)
        except Exception:
            logger.exception("Failed to read Redis response meta for workflow=%s", workflow_id)
            meta_raw = None
        if meta_raw:
            try:
                meta_payload = json.loads(self._decode(meta_raw))
            except Exception:
                meta_payload = {}
            expected_ids = [str(uid) for uid in meta_payload.get("expected_unique_ids", []) if uid]
            if meta_payload.get("expected_count") is not None:
                try:
                    expected_count = int(meta_payload.get("expected_count"))
                except Exception:
                    expected_count = len(expected_ids)
        if not expected_ids:
            expected_ids = sorted(status_map.keys())
        expected_count = max(expected_count, len(expected_ids))

        collected = [uid for uid in expected_ids if status_map.get(uid) == "complete"]
        pending = [uid for uid in expected_ids if status_map.get(uid) != "complete"]
        state = ResponseState(
            workflow_id=workflow_id,
            expected_unique_ids=expected_ids,
            collected_unique_ids=collected,
            pending_unique_ids=pending,
            expected_count=expected_count,
            status="pending",
        )
        return state

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _publish_completion(self, workflow_id: str) -> None:
        try:
            self.redis.publish(
                self._completion_channel(workflow_id),
                json.dumps({"workflow_id": workflow_id, "status": "complete"}),
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to publish Redis completion for workflow=%s", workflow_id)


_COORDINATOR_LOCK = threading.Lock()
_COORDINATOR: Optional[BaseResponseCoordinator] = None


def get_supplier_response_coordinator() -> BaseResponseCoordinator:
    """Return the configured supplier response coordinator instance."""

    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is not None:
            return _COORDINATOR

        redis_client = get_redis_client()
        if redis_client is not None:
            try:
                _COORDINATOR = RedisResponseCoordinator(
                    redis_client,
                    ttl_seconds=getattr(settings, "redis_response_ttl_seconds", 86400),
                )
                return _COORDINATOR
            except Exception:
                logger.exception("Falling back to in-memory coordinator after Redis failure")

        _COORDINATOR = InMemoryResponseCoordinator()
        return _COORDINATOR


def reset_coordinator() -> None:
    """Reset the cached coordinator (primarily for tests)."""

    global _COORDINATOR
    with _COORDINATOR_LOCK:
        _COORDINATOR = None


def notify_response_received(*, workflow_id: Optional[str], unique_id: Optional[str]) -> None:
    """Notify the coordinator that a supplier response has been stored."""

    workflow_key = str(workflow_id).strip() if workflow_id else None
    unique_key = str(unique_id).strip() if unique_id else None
    if not workflow_key or not unique_key:
        return

    try:
        coordinator = get_supplier_response_coordinator()
        coordinator.record_response(workflow_key, unique_key)
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "Failed to notify response coordinator for workflow=%s unique_id=%s",
            workflow_key,
            unique_key,
        )
