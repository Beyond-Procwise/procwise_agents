"""Background service that continuously polls IMAP for supplier responses."""

from __future__ import annotations

import imaplib
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from repositories import supplier_response_repo, workflow_email_tracking_repo
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
from services.email_watcher_v2 import EmailWatcherV2

try:  # pragma: no cover - settings import may fail in minimal environments
    from config.settings import settings as app_settings
except Exception:  # pragma: no cover - fallback when settings module unavailable
    app_settings = None

logger = logging.getLogger(__name__)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    if value in (None, ""):
        return default
    return value


def _setting(*attr_names: str) -> Optional[str]:
    if not app_settings:
        return None
    for name in attr_names:
        value = getattr(app_settings, name, None)
        if value not in (None, ""):
            return value
    return None


def _load_dispatch_rows(workflow_id: str) -> List[WorkflowDispatchRow]:
    try:
        workflow_email_tracking_repo.init_schema()
        return workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to load workflow dispatch records for %s", workflow_id)
        return []


def _serialise_response(row: Dict[str, Any], mailbox: Optional[str]) -> Dict[str, Any]:
    return {
        "workflow_id": row.get("workflow_id"),
        "unique_id": row.get("unique_id"),
        "supplier_id": row.get("supplier_id"),
        "body_text": row.get("response_text", ""),
        "subject": row.get("subject"),
        "from_addr": row.get("from_addr"),
        "message_id": row.get("message_id"),
        "received_at": row.get("received_time"),
        "mailbox": mailbox,
        "imap_uid": None,
        "price": row.get("price"),
        "lead_time": row.get("lead_time"),
    }


def _resolve_agent_dependency(
    agent_registry: Optional[Any],
    orchestrator: Optional[Any],
    preferred_keys: Sequence[str],
) -> Optional[Any]:
    """Best-effort resolution of shared agent instances.

    The email watcher operates in long-running background threads where
    instantiating new agent instances is expensive (pulling vector stores,
    DB connections, etc.).  This helper attempts to reuse agents that have
    already been registered with either the provided ``agent_registry`` or
    an ``orchestrator`` instance.  ``preferred_keys`` accepts both snake_case
    and CamelCase identifiers so legacy aliases resolve correctly.
    """

    if not preferred_keys:
        return None

    registries: List[Any] = []
    if agent_registry:
        registries.append(agent_registry)
    if orchestrator:
        orch_registry = getattr(orchestrator, "agents", None)
        if orch_registry:
            registries.append(orch_registry)
        agent_nick = getattr(orchestrator, "agent_nick", None)
        if agent_nick:
            nick_registry = getattr(agent_nick, "agents", None)
            if nick_registry:
                registries.append(nick_registry)

    for registry in registries:
        if registry is None:
            continue
        getter = getattr(registry, "get", None)
        for key in preferred_keys:
            candidate = None
            if callable(getter):
                try:
                    candidate = getter(key)
                except Exception:  # pragma: no cover - defensive
                    candidate = None
            if candidate is None and hasattr(registry, "__getitem__"):
                try:
                    candidate = registry[key]
                except Exception:  # pragma: no cover - registry miss or KeyError
                    candidate = None
            if candidate:
                return candidate

    fallback_sources: List[Any] = []
    if agent_registry:
        fallback_sources.append(agent_registry)
    if orchestrator:
        fallback_sources.append(orchestrator)
        agent_nick = getattr(orchestrator, "agent_nick", None)
        if agent_nick:
            fallback_sources.append(agent_nick)

    for source in fallback_sources:
        if source is None:
            continue
        for key in preferred_keys:
            candidate = getattr(source, key, None)
            if candidate:
                return candidate
            alt = getattr(source, f"{key}_agent", None)
            if alt:
                return alt

    return None


def run_email_watcher_for_workflow(
    *,
    workflow_id: str,
    run_id: Optional[str],
    wait_seconds_after_last_dispatch: int = 90,
    lookback_minutes: int = 240,
    mailbox_name: Optional[str] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[Any] = None,
    supplier_agent: Optional[Any] = None,
    negotiation_agent: Optional[Any] = None,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """Collect supplier responses for ``workflow_id`` using ``EmailWatcherV2``."""

    _ = run_id  # maintained for backwards compatibility
    _ = agent_registry
    _ = orchestrator
    _ = max_workers
    _ = lookback_minutes

    workflow_key = (workflow_id or "").strip()
    if not workflow_key:
        return {
            "status": "failed",
            "reason": "workflow_id is required",
            "workflow_id": workflow_id,
        }

    dispatch_rows = _load_dispatch_rows(workflow_key)
    if not dispatch_rows:
        return {
            "status": "skipped",
            "reason": f"No recorded dispatches for workflow {workflow_key}",
            "workflow_id": workflow_key,
            "expected": 0,
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
        }

    imap_host = _env("IMAP_HOST") or _setting("imap_host")
    imap_user = _env("IMAP_USER")
    imap_username = (
        _env("IMAP_USERNAME")
        or _setting("imap_username", "imap_user", "imap_login")
        or (imap_user.split("@")[0] if imap_user and "@" in imap_user else None)
    )
    imap_password = _env("IMAP_PASSWORD") or _setting("imap_password")
    imap_domain = _env("IMAP_DOMAIN") or _setting("imap_domain")
    imap_login = _env("IMAP_LOGIN") or _setting("imap_login")

    def _normalise_domain(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        domain = value.strip()
        if "@" in domain:
            domain = domain.split("@", 1)[-1]
        return domain or None

    def _pick_login() -> Optional[str]:
        if imap_login:
            return imap_login.strip()
        if imap_user:
            candidate = imap_user.strip()
            if candidate:
                return candidate
        username = (imap_username or "").strip()
        if not username:
            return None
        if "@" in username:
            return username
        domain = _normalise_domain(imap_domain)
        if domain:
            return f"{username}@{domain}"
        return username

    imap_login = _pick_login()
    try:
        imap_port = int(_env("IMAP_PORT")) if _env("IMAP_PORT") else None
    except Exception:
        imap_port = None
    imap_use_ssl_raw = (
        _env("IMAP_USE_SSL")
        or _env("IMAP_ENCRYPTION")
        or _setting("imap_use_ssl", "imap_encryption")
    )
    imap_use_ssl: Optional[bool]
    if imap_use_ssl_raw is None:
        imap_use_ssl = None
    else:
        raw_value = str(imap_use_ssl_raw).strip().lower()
        if raw_value in {"0", "false", "no", "none", "off"}:
            imap_use_ssl = False
        elif raw_value in {"ssl", "imaps", "true", "1", "yes", "on"}:
            imap_use_ssl = True
        else:
            imap_use_ssl = True
    mailbox = (
        mailbox_name
        or _env("IMAP_MAILBOX")
        or _setting("imap_mailbox")
        or "INBOX"
    )

    if not all([imap_host, imap_username, imap_password]):
        logger.warning(
            "IMAP credentials are not configured; skipping EmailWatcherV2 (host=%s user=%s)",
            imap_host,
            imap_username,
        )
        return {
            "status": "skipped",
            "reason": "IMAP credentials not configured",
            "workflow_id": workflow_key,
            "expected": len(dispatch_rows),
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
        }

    incomplete_dispatches = [
        row.unique_id
        for row in dispatch_rows
        if not row.message_id or row.dispatched_at is None
    ]
    if incomplete_dispatches:
        logger.debug(
            "Workflow %s has pending dispatch metadata; deferring watcher start", workflow_key
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": len(dispatch_rows),
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": sorted(incomplete_dispatches),
        }

    try:
        poll_interval = int(_env("IMAP_POLL_INTERVAL", "30"))
    except Exception:
        poll_interval = 30
    try:
        max_attempts = int(_env("IMAP_MAX_POLL_ATTEMPTS", "10"))
    except Exception:
        max_attempts = 10

    supplier_agent = supplier_agent or _resolve_agent_dependency(
        agent_registry, orchestrator, ("supplier_interaction", "SupplierInteractionAgent")
    )
    negotiation_agent = negotiation_agent or _resolve_agent_dependency(
        agent_registry, orchestrator, ("negotiation", "NegotiationAgent")
    )

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=max(0, int(wait_seconds_after_last_dispatch)),
        poll_interval_seconds=max(1, poll_interval),
        max_poll_attempts=max(1, max_attempts),
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        mailbox=mailbox,
        imap_host=imap_host,
        imap_username=imap_username,
        imap_password=imap_password,
        imap_port=imap_port,
        imap_use_ssl=imap_use_ssl,
        imap_login=imap_login,
    )
    try:
        result = watcher.wait_and_collect_responses(workflow_key)
    except imaplib.IMAP4.error as exc:
        logger.error(
            "IMAP authentication failed for host=%s user=%s: %s",
            imap_host,
            imap_login or imap_username,
            exc,
        )
        return {
            "status": "failed",
            "reason": "IMAP authentication failed",
            "workflow_id": workflow_key,
            "expected": len(dispatch_rows),
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
        }
    expected = result.get("dispatched_count", 0)
    responded = result.get("responded_count", 0)
    matched = result.get("matched_responses", {}) or {}
    matched_ids = sorted(matched.keys())

    pending_rows = supplier_response_repo.fetch_pending(workflow_id=workflow_key)
    responses = [_serialise_response(row, mailbox) for row in pending_rows]

    status = "processed" if result.get("complete") else "not_ready"

    response_payload = {
        "status": status,
        "workflow_id": workflow_key,
        "expected": expected,
        "found": responded,
        "rows": responses,
        "matched_unique_ids": matched_ids,
    }

    expected_ids = {row.unique_id for row in dispatch_rows}
    if status != "processed":
        missing = sorted(expected_ids - set(matched_ids))
        response_payload["reason"] = (
            "Not all responses received"
            if missing
            else "Responses still pending"
        )
        response_payload["missing_unique_ids"] = missing
        return response_payload

    if matched_ids:
        try:
            supplier_response_repo.delete_responses(
                workflow_id=workflow_key,
                unique_ids=matched_ids,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to remove supplier responses for workflow %s",
                workflow_key,
            )

    return response_payload


class EmailWatcherService:
    """Run ``EmailWatcherV2`` as a background polling service."""

    def __init__(
        self,
        *,
        poll_interval_seconds: Optional[int] = None,
        post_dispatch_interval_seconds: Optional[int] = None,
        dispatch_wait_seconds: Optional[int] = None,
        watcher_runner: Optional[Callable[..., Dict[str, object]]] = None,
        agent_registry: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        supplier_agent: Optional[Any] = None,
        negotiation_agent: Optional[Any] = None,
    ) -> None:
        if poll_interval_seconds is None:
            poll_interval_seconds = self._env_int("EMAIL_WATCHER_SERVICE_INTERVAL", fallback="90")
        if poll_interval_seconds <= 0:
            poll_interval_seconds = 90

        if post_dispatch_interval_seconds is None:
            post_dispatch_interval_seconds = self._env_int(
                "EMAIL_WATCHER_SERVICE_POST_DISPATCH_INTERVAL", fallback="30"
            )
        if post_dispatch_interval_seconds <= 0:
            post_dispatch_interval_seconds = 30

        if dispatch_wait_seconds is None:
            dispatch_wait_seconds = self._env_int("EMAIL_WATCHER_SERVICE_DISPATCH_WAIT", fallback="0")
        if dispatch_wait_seconds < 0:
            dispatch_wait_seconds = 0

        self._poll_interval = poll_interval_seconds
        self._post_dispatch_interval = post_dispatch_interval_seconds
        self._dispatch_wait = dispatch_wait_seconds
        self._runner: Callable[..., Dict[str, object]] = watcher_runner or run_email_watcher_for_workflow
        self._agent_registry = agent_registry
        self._orchestrator = orchestrator
        self._supplier_agent = supplier_agent
        self._negotiation_agent = negotiation_agent
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._forced_lock = threading.Lock()
        self._forced_workflows: Set[str] = set()

    @staticmethod
    def _env_int(name: str, *, fallback: str) -> int:
        try:
            return int(os.environ.get(name, fallback))
        except Exception:
            return int(fallback)

    def start(self) -> None:
        """Start the watcher loop if it is not already running."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="EmailWatcherService", daemon=True)
        self._thread.start()
        logger.info(
            "EmailWatcherService started (poll_interval=%ss post_dispatch_interval=%ss dispatch_wait=%ss)",
            self._poll_interval,
            self._post_dispatch_interval,
            self._dispatch_wait,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher loop to stop and wait for the thread."""

        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("EmailWatcherService stopped")

    def notify_workflow(self, workflow_id: str) -> None:
        """Wake the service to prioritise ``workflow_id`` in the next cycle."""

        workflow_key = (workflow_id or "").strip()
        if not workflow_key:
            return

        with self._forced_lock:
            self._forced_workflows.add(workflow_key)
        self._wake_event.set()

    def _consume_forced_workflows(self) -> List[str]:
        with self._forced_lock:
            items = list(self._forced_workflows)
            self._forced_workflows.clear()
        return items

    def _wait_for_next_cycle(self, seconds: float) -> None:
        if seconds <= 0:
            seconds = 0
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            awakened = self._wake_event.wait(timeout=remaining)
            if awakened:
                self._wake_event.clear()
                break

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            waiting_for_dispatch = False
            processed_workflow = False
            try:
                workflow_ids = workflow_email_tracking_repo.load_active_workflow_ids()
            except Exception:
                logger.exception("Failed to load workflows for email watcher service")
                workflow_ids = []

            forced = self._consume_forced_workflows()
            if forced:
                seen = set(workflow_ids)
                for workflow_id in forced:
                    if workflow_id not in seen:
                        workflow_ids.append(workflow_id)
                        seen.add(workflow_id)

            for workflow_id in workflow_ids:
                if self._stop_event.is_set():
                    break

                if not workflow_id:
                    continue

                try:
                    result = self._runner(
                        workflow_id=workflow_id,
                        run_id=None,
                        wait_seconds_after_last_dispatch=self._dispatch_wait,
                        agent_registry=self._agent_registry,
                        orchestrator=self._orchestrator,
                        supplier_agent=self._supplier_agent,
                        negotiation_agent=self._negotiation_agent,
                    )
                    status = str(result.get("status") or "").lower()
                    processed_workflow = True
                    if status == "failed":
                        logger.error(
                            "Email watcher service failed for workflow=%s: %s",
                            workflow_id,
                            result.get("reason") or result.get("error"),
                        )
                    elif status == "waiting_for_dispatch":
                        waiting_for_dispatch = True
                except Exception:
                    logger.exception("Email watcher service encountered an error for workflow %s", workflow_id)

            if waiting_for_dispatch or not processed_workflow:
                sleep_seconds = self._poll_interval
            else:
                sleep_seconds = self._post_dispatch_interval

            self._wait_for_next_cycle(sleep_seconds)
            if self._stop_event.is_set():
                break

        logger.debug("EmailWatcherService loop terminated")
