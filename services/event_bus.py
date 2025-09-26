"""Lightweight event bus and workflow context utilities for ProcWise."""

from __future__ import annotations

import logging
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EventBus:
    """Minimal synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Tuple[Callable[[Dict[str, Any]], None], bool]]] = {}
        self._lock = threading.RLock()

    def subscribe(
        self,
        event_name: str,
        callback: Callable[[Dict[str, Any]], None],
        *,
        once: bool = False,
    ) -> Callable[[Dict[str, Any]], None]:
        """Register ``callback`` to be invoked when ``event_name`` is published."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        key = str(event_name).strip()
        if not key:
            raise ValueError("event_name must be a non-empty string")
        with self._lock:
            self._listeners.setdefault(key, []).append((callback, bool(once)))
        return callback

    def unsubscribe(
        self, event_name: str, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Remove ``callback`` from ``event_name`` subscriptions if present."""

        key = str(event_name).strip()
        if not key:
            return
        with self._lock:
            listeners = self._listeners.get(key, [])
            self._listeners[key] = [entry for entry in listeners if entry[0] is not callback]
            if not self._listeners[key]:
                self._listeners.pop(key, None)

    def publish(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Invoke subscribers for ``event_name`` synchronously."""

        key = str(event_name).strip()
        if not key:
            return
        with self._lock:
            listeners = list(self._listeners.get(key, []))
        if not listeners:
            return
        payload = dict(payload or {})
        to_remove: List[Callable[[Dict[str, Any]], None]] = []
        for callback, once in listeners:
            try:
                callback(payload)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Event handler for %s failed", key)
            if once:
                to_remove.append(callback)
        if to_remove:
            with self._lock:
                current = self._listeners.get(key, [])
                self._listeners[key] = [entry for entry in current if entry[0] not in to_remove]
                if not self._listeners[key]:
                    self._listeners.pop(key, None)


_GLOBAL_EVENT_BUS: Optional[EventBus] = None
_EVENT_BUS_LOCK = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the singleton :class:`EventBus` instance."""

    global _GLOBAL_EVENT_BUS
    with _EVENT_BUS_LOCK:
        if _GLOBAL_EVENT_BUS is None:
            _GLOBAL_EVENT_BUS = EventBus()
    return _GLOBAL_EVENT_BUS


_thread_context = threading.local()


def get_current_workflow() -> Optional[Dict[str, Any]]:
    """Return metadata for the workflow currently executing on this thread."""

    return getattr(_thread_context, "workflow", None)


@contextmanager
def workflow_scope(
    *,
    workflow_id: str,
    workflow_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Temporarily set the workflow context for the current thread."""

    previous = get_current_workflow()
    _thread_context.workflow = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "agent_name": agent_name,
        "metadata": metadata or {},
    }
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_thread_context, "workflow"):
                delattr(_thread_context, "workflow")
        else:
            _thread_context.workflow = previous
