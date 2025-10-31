"""Background service that continuously polls IMAP for supplier responses."""

from __future__ import annotations

import imaplib
import logging
import os
import threading
import time
from dataclasses import MISSING
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from repositories import (
    draft_rfq_emails_repo,
    supplier_response_repo,
    workflow_email_tracking_repo,
    workflow_lifecycle_repo,
)
from repositories.workflow_email_tracking_repo import WorkflowDispatchRow
<<<<<<< HEAD
from services.email_watcher import EmailWatcher, EmailWatcherConfig
=======
from services.email_watcher_v2 import EmailWatcherV2
from services.watcher_utils import run_email_watcher_for_workflow
>>>>>>> f6b29da (updated changes)

try:  # pragma: no cover - settings import may fail in minimal environments
    from config.settings import settings as app_settings
except Exception:  # pragma: no cover - fallback when settings module unavailable
    app_settings = None

logger = logging.getLogger(__name__)


MAX_IMAP_AUTH_RETRIES = 3


def send_alert(alert_code: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Publish an alert event for operational visibility."""

    message = {"alert_code": alert_code}
    if payload:
        message.update(payload)
    try:
        bus = get_event_bus()
    except Exception:
        logger.exception("Failed to initialise event bus for alert %s", alert_code)
        return
    try:
        bus.publish("alerts", message)
    except Exception:
        logger.exception("Failed to publish alert %s", alert_code)


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


def _resolve_float(env_key: str, setting_keys: Sequence[str], default: float) -> float:
    raw_env = _env(env_key)
    if raw_env not in (None, ""):
        try:
            return float(raw_env)
        except Exception:
            logger.warning("Invalid %s value %s; falling back to %s", env_key, raw_env, default)
    for key in setting_keys:
        raw_setting = _setting(key)
        if raw_setting in (None, ""):
            continue
        try:
            return float(raw_setting)
        except Exception:
            logger.warning("Invalid setting %s=%s; falling back to %s", key, raw_setting, default)
    return default


def _resolve_optional_int(env_key: str, setting_keys: Sequence[str]) -> Optional[int]:
    raw_env = _env(env_key)
    if raw_env not in (None, ""):
        try:
            candidate = int(float(raw_env))
        except Exception:
            logger.warning("Invalid %s value %s; ignoring", env_key, raw_env)
        else:
            if candidate > 0:
                return candidate
    for key in setting_keys:
        raw_setting = _setting(key)
        if raw_setting in (None, ""):
            continue
        try:
            candidate = int(float(raw_setting))
        except Exception:
            logger.warning("Invalid setting %s=%s; ignoring", key, raw_setting)
            continue
        if candidate > 0:
            return candidate
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
        "rfq_id": row.get("rfq_id"),
        "body_text": row.get("response_text", ""),
        "body_html": row.get("response_body"),
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
    process_routing_service: Optional[Any] = None,
    max_workers: int = 8,
    workflow_memory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect supplier responses for ``workflow_id`` using the unified EmailWatcher."""

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

    expected_unique_ids, _, _ = draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
        workflow_id=workflow_key
    )
    expected_unique_ids = {
        (unique_id or "").strip()
        for unique_id in expected_unique_ids
        if unique_id and unique_id.strip()
    }

    rows_by_uid = {
        (row.unique_id or "").strip(): row
        for row in dispatch_rows
        if (row.unique_id or "").strip()
    }

    if not expected_unique_ids:
        fallback_unique_ids = set(rows_by_uid)
        if fallback_unique_ids:
            logger.debug(
                "Workflow %s has no drafting dispatch records; falling back to dispatch history",
                workflow_key,
            )
            expected_unique_ids = fallback_unique_ids
        else:
            logger.debug(
                "Workflow %s has no expected dispatch records from drafting or dispatch history; deferring watcher",
                workflow_key,
            )
            return {
                "status": "waiting_for_dispatch",
                "reason": "No expected dispatches recorded",
                "workflow_id": workflow_key,
                "expected": len(dispatch_rows),
                "found": 0,
                "rows": [],
                "matched_unique_ids": [],
                "pending_unique_ids": [],
                "missing_required_fields": {},
            }

    missing_expected_rows = sorted(
        uid for uid in expected_unique_ids if uid not in rows_by_uid
    )
    if missing_expected_rows:
        logger.debug(
            "Workflow %s awaiting dispatch rows for %s; deferring watcher",
            workflow_key,
            missing_expected_rows,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": len(expected_unique_ids),
            "found": len(expected_unique_ids) - len(missing_expected_rows),
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": missing_expected_rows,
            "missing_required_fields": {uid: ["dispatch_record"] for uid in missing_expected_rows},
        }

    expected_dispatch_total = len(expected_unique_ids)

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
    if not imap_username and imap_login:
        imap_username = imap_login.strip() or None
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

    if not imap_host or not imap_password or not (imap_username or imap_login):
        logger.warning(
            "IMAP credentials are not configured; skipping EmailWatcher (host=%s user=%s)",
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

    missing_required_fields: Dict[str, List[str]] = {}
    missing_message_ids: List[str] = []

    for unique_id in sorted(expected_unique_ids):
        row = rows_by_uid.get(unique_id)
        missing_fields: List[str] = []
        if not row:
            missing_fields.append("dispatch_record")
        else:
            if getattr(row, "dispatched_at", None) is None:
                missing_fields.append("dispatched_at")
            message_id = (row.message_id or "").strip()
            if not message_id:
                missing_message_ids.append(unique_id)
                missing_fields.append("message_id")

        if missing_fields:
            missing_required_fields[unique_id] = missing_fields

        supplier_email = (row.supplier_email or "").strip()
        if not supplier_email:
            missing_fields.append("supplier_email")

        subject_value = (row.subject or "").strip()
        if not subject_value:
            missing_fields.append("subject")

        if missing_fields:
            missing_required_fields[identifier] = missing_fields
            continue

    if missing_message_ids:
        logger.warning(
            "Workflow %s has dispatches missing message_id: %s",
            workflow_key,
            ", ".join(sorted(missing_message_ids)),
        )

    if missing_required_fields:
        logger.debug(
            "Workflow %s has pending dispatch metadata; deferring watcher start (missing=%s)",
            workflow_key,
            missing_required_fields,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": expected_dispatch_total,
            "found": 0,
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": sorted(missing_required_fields.keys()),
            "missing_required_fields": missing_required_fields,
        }

    completed_unique_ids = {
        unique_id
        for unique_id in expected_unique_ids
        if unique_id in rows_by_uid
        and (rows_by_uid[unique_id].message_id or "").strip()
        and getattr(rows_by_uid[unique_id], "dispatched_at", None) is not None
    }
    if expected_unique_ids and expected_unique_ids - completed_unique_ids:
        pending = sorted(expected_unique_ids - completed_unique_ids)
        logger.debug(
            "Workflow %s dispatches incomplete; waiting for %s to finish",
            workflow_key,
            pending,
        )
        return {
            "status": "waiting_for_dispatch",
            "reason": "Waiting for all dispatches to complete",
            "workflow_id": workflow_key,
            "expected": expected_dispatch_total,
            "found": len(completed_unique_ids),
            "rows": [],
            "matched_unique_ids": [],
            "pending_unique_ids": pending,
            "missing_required_fields": missing_required_fields,
        }

    try:
        poll_interval = int(_env("IMAP_POLL_INTERVAL", "30"))
    except Exception:
        poll_interval = 30
    try:
        max_attempts = int(_env("IMAP_MAX_POLL_ATTEMPTS", "10"))
    except Exception:
        max_attempts = 10

    poll_backoff_factor = _resolve_float(
        "IMAP_POLL_BACKOFF_FACTOR",
        ("imap_poll_backoff_factor", "poll_backoff_factor"),
        1.8,
    )
    poll_jitter_seconds = max(
        0.0,
        _resolve_float(
            "IMAP_POLL_JITTER_SECONDS",
            ("imap_poll_jitter_seconds", "poll_jitter_seconds"),
            2.0,
        ),
    )
    poll_max_interval_default = float(max(poll_interval * 6, poll_interval))
    poll_max_interval = _resolve_float(
        "IMAP_POLL_MAX_INTERVAL",
        ("imap_poll_max_interval", "poll_max_interval_seconds"),
        poll_max_interval_default,
    )
    poll_max_interval_seconds = max(int(poll_max_interval), poll_interval)
    poll_timeout_seconds = _resolve_optional_int(
        "IMAP_POLL_TIMEOUT_SECONDS",
        ("imap_poll_timeout_seconds", "poll_timeout_seconds"),
    )

    supplier_agent = supplier_agent or _resolve_agent_dependency(
        agent_registry, orchestrator, ("supplier_interaction", "SupplierInteractionAgent")
    )
    negotiation_agent = negotiation_agent or _resolve_agent_dependency(
        agent_registry, orchestrator, ("negotiation", "NegotiationAgent")
    )
    process_routing_service = process_routing_service or _resolve_agent_dependency(
        agent_registry,
        orchestrator,
        ("process_routing_service", "process_router", "ProcessRoutingService"),
    )
    if process_routing_service is None and supplier_agent is not None:
        process_routing_service = getattr(supplier_agent, "process_routing_service", None)

    response_grace_seconds = _resolve_optional_int(
        "EMAIL_WATCHER_RESPONSE_GRACE_SECONDS",
        ("email_response_grace_seconds", "response_grace_seconds"),
    )
    default_grace_field = EmailWatcherConfig.__dataclass_fields__["response_grace_seconds"]
    default_grace_seconds = default_grace_field.default
    if default_grace_seconds is MISSING:
        default_grace_seconds = 180

    config = EmailWatcherConfig(
        imap_host=imap_host,
        imap_username=imap_username,
        imap_password=imap_password,
        imap_port=imap_port or 993,
        imap_use_ssl=True if imap_use_ssl is None else imap_use_ssl,
        imap_login=imap_login,
<<<<<<< HEAD
        imap_mailbox=mailbox or "INBOX",
        dispatch_wait_seconds=max(0, int(wait_seconds_after_last_dispatch)),
        poll_interval_seconds=max(1, poll_interval),
        max_poll_attempts=max(1, max_attempts),
        poll_backoff_factor=max(1.0, poll_backoff_factor),
        poll_jitter_seconds=poll_jitter_seconds,
        poll_max_interval_seconds=poll_max_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        response_grace_seconds=response_grace_seconds
        if response_grace_seconds is not None
        else int(default_grace_seconds),
    )
    watcher = EmailWatcher(
        config=config,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
        process_router=process_routing_service,
        sleep=time.sleep,
    )
    logger.info(
        "watcher_started workflow=%s expected=%s status=active",
        workflow_key,
        expected_dispatch_total,
=======
        workflow_memory=workflow_memory,
>>>>>>> f6b29da (updated changes)
    )
    try:
        result = watcher.wait_for_responses(workflow_key)
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
    expected = result.get("dispatched_count", expected_dispatch_total)
    responded = result.get("responded_count", 0)
    matched = result.get("matched_responses", {}) or {}
    matched_ids = sorted(matched.keys())

    pending_rows = supplier_response_repo.fetch_pending(workflow_id=workflow_key)
    responses = [_serialise_response(row, mailbox) for row in pending_rows]

    watcher_status = result.get("status") or (
        "completed" if result.get("complete") else "pending"
    )
    status = "processed" if watcher_status == "completed" else watcher_status

    response_payload = {
        "status": status,
        "watcher_status": watcher_status,
        "workflow_id": workflow_key,
        "expected": expected,
        "found": responded,
        "rows": responses,
        "matched_unique_ids": matched_ids,
    }
    if workflow_status:
        response_payload["workflow_status"] = workflow_status
    if "expected_responses" in result:
        response_payload["expected_responses"] = result.get("expected_responses")
    if "elapsed_seconds" in result:
        response_payload["elapsed_seconds"] = result.get("elapsed_seconds")
    if "timeout_reached" in result:
        response_payload["timeout_reached"] = bool(result.get("timeout_reached"))
    if "pending_suppliers" in result:
        response_payload["pending_suppliers"] = result.get("pending_suppliers")
    if "pending_unique_ids" in result:
        response_payload["pending_unique_ids"] = result.get("pending_unique_ids")
    if "last_capture_ts" in result:
        response_payload["last_capture_ts"] = result.get("last_capture_ts")
    if "timeout_deadline" in result:
        response_payload["timeout_deadline"] = result.get("timeout_deadline")

    if result.get("pending_unique_ids"):
        response_payload["pending_unique_ids"] = list(result["pending_unique_ids"])
    if result.get("pending_suppliers"):
        response_payload["pending_suppliers"] = list(result["pending_suppliers"])
    if result.get("timeout_reason"):
        response_payload["timeout_reason"] = result["timeout_reason"]

    expected_ids = expected_unique_ids or {
        row.unique_id for row in dispatch_rows if row.unique_id
    }
    if status not in {"processed", "completed"}:
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
    """Run the unified :class:`EmailWatcher` as a background polling service."""

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
<<<<<<< HEAD
        process_routing_service: Optional[Any] = None,
=======
        workflow_memory: Optional[Any] = None,
>>>>>>> f6b29da (updated changes)
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
<<<<<<< HEAD
        self._process_router = process_routing_service
=======
        self._workflow_memory = workflow_memory
>>>>>>> f6b29da (updated changes)
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

        self._preempt_existing_workflows(workflow_key)
        if self._thread and self._thread.is_alive():
            self.stop()
            self.start()
        with self._forced_lock:
            self._forced_workflows = {workflow_key}
        self._wake_event.set()

    def _consume_forced_workflows(self) -> List[str]:
        with self._forced_lock:
            items = list(self._forced_workflows)
            self._forced_workflows.clear()
        return items

    def _preempt_existing_workflows(self, new_workflow_id: str) -> None:
        try:
            active_workflows = workflow_email_tracking_repo.load_active_workflow_ids()
        except Exception:
            logger.exception(
                "EmailWatcherService failed to enumerate active workflows before preemption"
            )
            return

        seen: Set[str] = set()
        for workflow_id in active_workflows:
            workflow_key = (workflow_id or "").strip()
            if not workflow_key or workflow_key == new_workflow_id:
                continue
            if workflow_key in seen:
                continue
            seen.add(workflow_key)
            try:
                workflow_email_tracking_repo.reset_workflow(workflow_key)
                workflow_lifecycle_repo.record_watcher_event(
                    workflow_key,
                    "watcher_stopped",
                    expected_responses=0,
                    received_responses=0,
                    metadata={
                        "stop_reason": "preempted_by_new_workflow",
                        "preempted_by": new_workflow_id,
                    },
                )
                logger.info(
                    "EmailWatcherService preempted workflow=%s in favour of workflow=%s",
                    workflow_key,
                    new_workflow_id,
                )
            except Exception:
                logger.exception(
                    "EmailWatcherService failed to preempt workflow=%s before starting workflow=%s",
                    workflow_key,
                    new_workflow_id,
                )
        if seen:
            with self._forced_lock:
                self._forced_workflows = {
                    wf for wf in self._forced_workflows if wf == new_workflow_id
                }

    def _should_skip_workflow(self, workflow_id: str) -> bool:
        """Return ``True`` when ``workflow_id`` should not be processed."""

        workflow_key = (workflow_id or "").strip()
        if not workflow_key:
            return False

        try:
            lifecycle = workflow_lifecycle_repo.get_lifecycle(workflow_key)
        except Exception:
            logger.exception(
                "EmailWatcherService failed to load lifecycle for workflow=%s",
                workflow_key,
            )
            return False

        if not lifecycle:
            return False

        negotiation_status = str(
            lifecycle.get("negotiation_status") or ""
        ).strip().lower()
        if negotiation_status not in {"completed", "finalized"}:
            return False

        watcher_status = str(lifecycle.get("watcher_status") or "").strip().lower()
        if watcher_status != "stopped":
            return False

        metadata = lifecycle.get("metadata") or {}
        stop_reason = str(metadata.get("stop_reason") or "").strip().lower()
        if stop_reason in {
            "negotiation_completed",
            "negotiation_completed_pending_responses",
        }:
            logger.debug(
                "EmailWatcherService suppressing workflow=%s due to completed negotiation",
                workflow_key,
            )
            return True

        return False

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

    def update_dependencies(
        self,
        *,
        agent_registry: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        supplier_agent: Optional[Any] = None,
        negotiation_agent: Optional[Any] = None,
    ) -> None:
        """Refresh shared dependency references used by the watcher."""

        if agent_registry is not None:
            self._agent_registry = agent_registry
        if orchestrator is not None:
            self._orchestrator = orchestrator
        if supplier_agent is not None:
            self._supplier_agent = supplier_agent
        if negotiation_agent is not None:
            self._negotiation_agent = negotiation_agent

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            waiting_for_dispatch = False
            processed_workflow = False
            forced = self._consume_forced_workflows()
            if forced:
                workflow_ids = list(dict.fromkeys(forced))
            else:
                workflow_ids = []

            for workflow_id in workflow_ids:
                if self._stop_event.is_set():
                    break

                if not workflow_id:
                    continue

                if self._should_skip_workflow(workflow_id):
                    processed_workflow = True
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
<<<<<<< HEAD
                        process_routing_service=self._process_router,
=======
                        workflow_memory=self._workflow_memory,
>>>>>>> f6b29da (updated changes)
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
