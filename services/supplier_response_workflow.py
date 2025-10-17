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
        self._workflow_initialised = False
        self._response_initialised = False

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
        wait_for_all: bool = True,
    ) -> Dict[str, object]:
        """Wait until tracked dispatches have recorded message IDs."""

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

        if not self._workflow_initialised:
            try:
                self._workflow_repo.init_schema()
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to initialise workflow email tracking schema before dispatch wait",
                )
            else:
                self._workflow_initialised = True

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

            if not wait_for_all:
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

    def await_first_dispatch(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[Optional[str]],
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
    ) -> Dict[str, object]:
        """Block until the first dispatch for the workflow is recorded."""

        expected_ids = self._normalise_ids(unique_ids)
        if not workflow_id or not expected_ids:
            return {
                "workflow_id": workflow_id,
                "activated": True,
                "first_unique_id": None,
                "wait_seconds": 0.0,
                "unique_ids": expected_ids,
            }

        if not self._workflow_initialised:
            try:
                self._workflow_repo.init_schema()
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to initialise workflow email tracking schema before activation wait",
                )
            else:
                self._workflow_initialised = True

        poll_seconds = max(0.0, float(poll_interval or 0))
        deadline = None
        if timeout is not None:
            try:
                deadline = time.monotonic() + max(0, int(timeout))
            except Exception:
                deadline = time.monotonic() + max(0, float(timeout or 0))
        start = time.monotonic()

        first_unique: Optional[str] = None
        activated = False

        while True:
            try:
                rows = self._workflow_repo.load_workflow_rows(workflow_id=workflow_id)
            except Exception:
                logger.exception(
                    "Failed to load dispatch metadata for workflow=%s", workflow_id
                )
                rows = []

            for row in rows:
                unique_id = str(getattr(row, "unique_id", "")).strip()
                if unique_id and unique_id in expected_ids:
                    if getattr(row, "dispatched_at", None) is not None and getattr(
                        row, "message_id", None
                    ):
                        first_unique = unique_id
                        activated = True
                        break

            if activated:
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
            "activated": activated,
            "first_unique_id": first_unique,
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
        wait_for_all: bool = True,
    ) -> Dict[str, object]:
        """Poll ``proc.supplier_response`` until expected responses exist."""

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

        if not self._response_initialised:
            try:
                self._response_repo.init_schema()
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Failed to initialise supplier response schema before polling",
                )
            else:
                self._response_initialised = True

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

            if not wait_for_all:
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

        activation_info = self.await_first_dispatch(
            workflow_id=workflow_id,
            unique_ids=unique_ids,
            timeout=dispatch_timeout,
            poll_interval=dispatch_poll_interval,
        )
        if not activation_info.get("activated", False):
            raise TimeoutError(
                f"Timed out waiting for initial dispatch for workflow {workflow_id}"
            )

        dispatch_info = self.await_dispatch_completion(
            workflow_id=workflow_id,
            unique_ids=unique_ids,
            timeout=dispatch_timeout,
            poll_interval=dispatch_poll_interval,
            wait_for_all=False,
        )

        response_info = self.await_response_population(
            workflow_id=workflow_id,
            unique_ids=unique_ids,
            timeout=response_timeout,
            poll_interval=response_poll_interval,
            wait_for_all=False,
        )

        return {
            "activation": activation_info,
            "dispatch": dispatch_info,
            "responses": response_info,
        }
