"""Workflow helpers for coordinating supplier email interactions."""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from repositories import supplier_response_repo, workflow_email_tracking_repo

import logging

logger = logging.getLogger(__name__)


class SupplierResponseWorkflow:
    """Coordinate dispatch and response readiness for supplier workflows."""

    def __init__(
        self,
        *,
        workflow_repo=workflow_email_tracking_repo,
        response_repo=supplier_response_repo,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._workflow_repo = workflow_repo
        self._response_repo = response_repo
        self._sleep = sleep_fn

    @staticmethod
    def _normalise_ids(values: Iterable[Optional[str]]) -> List[str]:
        normalised: List[str] = []
        seen: Set[str] = set()
        for value in values:
            if value in (None, ""):
                continue
            try:
                candidate = str(value).strip()
            except Exception:
                continue
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalised.append(candidate)
        return normalised

    def await_dispatch_completion(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[Optional[str]],
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
    ) -> Dict[str, object]:
        """Wait until all tracked dispatches have recorded message IDs."""

        expected_ids = self._normalise_ids(unique_ids)
        expected_total = len(expected_ids)
        if not workflow_id or expected_total == 0:
            return {
                "workflow_id": workflow_id,
                "expected_dispatches": expected_total,
                "completed_dispatches": expected_total,
                "complete": True,
                "wait_seconds": 0.0,
                "unique_ids": expected_ids,
            }

        try:
            self._workflow_repo.init_schema()
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to initialise workflow email tracking schema before dispatch wait",
            )

        poll_seconds = max(0.0, float(poll_interval or 0))
        deadline = None
        if timeout is not None:
            try:
                deadline = time.monotonic() + max(0, int(timeout))
            except Exception:
                deadline = time.monotonic() + max(0, float(timeout or 0))
        start = time.monotonic()

        completed: Set[str] = set()
        while True:
            try:
                rows = self._workflow_repo.load_workflow_rows(workflow_id=workflow_id)
            except Exception:
                logger.exception(
                    "Failed to load dispatch metadata for workflow=%s", workflow_id
                )
                rows = []

            completed = {
                str(getattr(row, "unique_id", "")).strip()
                for row in rows
                if getattr(row, "dispatched_at", None) is not None
                and getattr(row, "message_id", None)
                and str(getattr(row, "unique_id", "")).strip()
            }
            completed &= set(expected_ids)

            if len(completed) >= expected_total:
                break

            if deadline is not None and time.monotonic() >= deadline:
                break

            if poll_seconds > 0:
                self._sleep(poll_seconds)
            else:
                self._sleep(0)

        elapsed = time.monotonic() - start
        return {
            "workflow_id": workflow_id,
            "expected_dispatches": expected_total,
            "completed_dispatches": len(completed),
            "complete": len(completed) >= expected_total,
            "wait_seconds": elapsed,
            "unique_ids": expected_ids,
        }

    def await_response_population(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[Optional[str]],
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
    ) -> Dict[str, object]:
        """Poll ``proc.supplier_response`` until all expected responses exist."""

        expected_ids = self._normalise_ids(unique_ids)
        expected_total = len(expected_ids)
        if not workflow_id or expected_total == 0:
            return {
                "workflow_id": workflow_id,
                "expected_responses": expected_total,
                "completed_responses": expected_total,
                "complete": True,
                "wait_seconds": 0.0,
                "unique_ids": expected_ids,
            }

        try:
            self._response_repo.init_schema()
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Failed to initialise supplier response schema before polling",
            )

        poll_seconds = max(0.0, float(poll_interval or 0))
        deadline = None
        if timeout is not None:
            try:
                deadline = time.monotonic() + max(0, int(timeout))
            except Exception:
                deadline = time.monotonic() + max(0, float(timeout or 0))
        start = time.monotonic()

        completed: Set[str] = set()
        while True:
            try:
                rows = self._response_repo.fetch_all(workflow_id=workflow_id)
            except Exception:
                logger.exception(
                    "Failed to fetch supplier responses for workflow=%s", workflow_id
                )
                rows = []

            completed = {
                str(row.get("unique_id", "")).strip()
                for row in rows
                if str(row.get("unique_id", "")).strip()
            }
            completed &= set(expected_ids)

            if len(completed) >= expected_total:
                break

            if deadline is not None and time.monotonic() >= deadline:
                break

            if poll_seconds > 0:
                self._sleep(poll_seconds)
            else:
                self._sleep(0)

        elapsed = time.monotonic() - start
        return {
            "workflow_id": workflow_id,
            "expected_responses": expected_total,
            "completed_responses": len(completed),
            "complete": len(completed) >= expected_total,
            "wait_seconds": elapsed,
            "unique_ids": expected_ids,
        }

    def ensure_ready(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[Optional[str]],
        dispatch_timeout: Optional[int] = None,
        dispatch_poll_interval: Optional[int] = None,
        response_timeout: Optional[int] = None,
        response_poll_interval: Optional[int] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Ensure dispatch and responses are ready for processing."""

        dispatch_info = self.await_dispatch_completion(
            workflow_id=workflow_id,
            unique_ids=unique_ids,
            timeout=dispatch_timeout,
            poll_interval=dispatch_poll_interval,
        )
        if not dispatch_info.get("complete", False):
            raise TimeoutError(
                f"Dispatch metadata incomplete for workflow {workflow_id}; "
                f"expected {dispatch_info['expected_dispatches']} dispatched emails, "
                f"found {dispatch_info['completed_dispatches']}"
            )

        response_info = self.await_response_population(
            workflow_id=workflow_id,
            unique_ids=unique_ids,
            timeout=response_timeout,
            poll_interval=response_poll_interval,
        )
        if not response_info.get("complete", False):
            raise TimeoutError(
                f"Supplier responses incomplete for workflow {workflow_id}; "
                f"expected {response_info['expected_responses']} responses, "
                f"found {response_info['completed_responses']}"
            )

        return {
            "dispatch": dispatch_info,
            "responses": response_info,
        }
