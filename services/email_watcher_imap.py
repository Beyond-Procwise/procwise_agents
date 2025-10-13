from __future__ import annotations

import logging
import os
import imaplib
from typing import Any, Dict, List, Optional

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


def run_email_watcher_for_workflow(
    *,
    workflow_id: str,
    run_id: Optional[str],
    wait_seconds_after_last_dispatch: int = 90,
    lookback_minutes: int = 240,
    mailbox_name: Optional[str] = None,
    agent_registry: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[Any] = None,
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
            # When a full email is provided as the "domain" value, keep only the domain portion.
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

    watcher = EmailWatcherV2(
        dispatch_wait_seconds=max(0, int(wait_seconds_after_last_dispatch)),
        poll_interval_seconds=max(1, poll_interval),
        max_poll_attempts=max(1, max_attempts),
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
