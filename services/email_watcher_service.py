"""Background service that continuously polls IMAP for supplier responses."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from repositories import workflow_email_tracking_repo
from services.email_watcher_imap import run_email_watcher_for_workflow

logger = logging.getLogger(__name__)


class EmailWatcherService:
    """Run ``EmailWatcherV2`` as a background polling service."""

    def __init__(
        self,
        *,
        poll_interval_seconds: Optional[int] = None,
        dispatch_wait_seconds: Optional[int] = None,
    ) -> None:
        if poll_interval_seconds is None:
            poll_interval_seconds = self._env_int("EMAIL_WATCHER_SERVICE_INTERVAL", fallback="30")
        if poll_interval_seconds <= 0:
            poll_interval_seconds = 30

        if dispatch_wait_seconds is None:
            dispatch_wait_seconds = self._env_int("EMAIL_WATCHER_SERVICE_DISPATCH_WAIT", fallback="0")
        if dispatch_wait_seconds < 0:
            dispatch_wait_seconds = 0

        self._poll_interval = poll_interval_seconds
        self._dispatch_wait = dispatch_wait_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        self._thread = threading.Thread(target=self._run_loop, name="EmailWatcherService", daemon=True)
        self._thread.start()
        logger.info(
            "EmailWatcherService started (poll_interval=%ss dispatch_wait=%ss)",
            self._poll_interval,
            self._dispatch_wait,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the watcher loop to stop and wait for the thread."""

        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        logger.info("EmailWatcherService stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                workflow_ids = workflow_email_tracking_repo.load_active_workflow_ids()
            except Exception:
                logger.exception("Failed to load workflows for email watcher service")
                workflow_ids = []

            for workflow_id in workflow_ids:
                if self._stop_event.is_set():
                    break

                if not workflow_id:
                    continue

                try:
                    result = run_email_watcher_for_workflow(
                        workflow_id=workflow_id,
                        run_id=None,
                        wait_seconds_after_last_dispatch=self._dispatch_wait,
                        agent_registry=None,
                        orchestrator=None,
                    )
                    status = str(result.get("status") or "").lower()
                    if status == "failed":
                        logger.error(
                            "Email watcher service failed for workflow=%s: %s",
                            workflow_id,
                            result.get("reason") or result.get("error"),
                        )
                except Exception:
                    logger.exception("Email watcher service encountered an error for workflow %s", workflow_id)

            sleep_seconds = self._poll_interval
            if sleep_seconds <= 0:
                sleep_seconds = 30

            if self._stop_event.wait(timeout=sleep_seconds):
                break

        logger.debug("EmailWatcherService loop terminated")
