"""Workflow helpers for coordinating supplier email interactions."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

    @staticmethod
    def _coerce_timestamp(value: object) -> Optional[datetime]:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                try:
                    parsed = datetime.utcfromtimestamp(float(text))
                except Exception:
                    return None
                return parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        return None

    @classmethod
    def _normalise_dispatch_context(
        cls, context: Optional[Dict[object, object]]
    ) -> Tuple[Dict[str, Dict[str, object]], Optional[Dict[str, object]]]:
        """Coerce dispatch freshness requirements into a normalised mapping."""

        if not isinstance(context, dict):
            return {}, None

        mapping: Dict[str, Dict[str, object]] = {}
        default_entry: Optional[Dict[str, object]] = None

        for key, raw in context.items():
            if isinstance(raw, dict):
                raw_entry = dict(raw)
            else:
                raw_entry = {"min_dispatched_at": raw}

            min_dt = cls._coerce_timestamp(raw_entry.get("min_dispatched_at"))
            thread_idx_raw = raw_entry.get("thread_index")
            thread_idx: Optional[int]
            try:
                thread_idx = int(thread_idx_raw) if thread_idx_raw is not None else None
            except Exception:
                thread_idx = None

            if min_dt is None and thread_idx is None:
                continue

            entry: Dict[str, object] = {}
            if min_dt is not None:
                entry["min_dispatched_at"] = min_dt
            if thread_idx is not None:
                entry["thread_index"] = thread_idx

            if isinstance(key, str):
                key_text = key.strip()
            else:
                key_text = str(key).strip()

            if not key_text:
                continue

            lowered = key_text.lower()
            if lowered in {"*", "__all__", "default"}:
                default_entry = entry
                continue

            mapping[lowered] = entry

        return mapping, default_entry

    @staticmethod
    def _extract_thread_index(row: object) -> Optional[int]:
        headers = getattr(row, "thread_headers", None)
        if isinstance(headers, dict):
            candidates = (
                "X-Procwise-Thread-Index",
                "X-Thread-Index",
                "Thread-Index",
                "thread_index",
            )
            for key in candidates:
                value = headers.get(key)
                if value in (None, ""):
                    continue
                if isinstance(value, (list, tuple, set)):
                    for item in value:
                        try:
                            return int(str(item).strip())
                        except Exception:
                            continue
                else:
                    try:
                        return int(str(value).strip())
                    except Exception:
                        continue
        return None

    def await_dispatch_completion(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[Optional[str]],
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
        wait_for_all: bool = True,
        dispatch_context: Optional[Dict[object, object]] = None,
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
        context_map, default_context = self._normalise_dispatch_context(dispatch_context)
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
                if getattr(row, "message_id", None)
            }

            filtered: Set[str] = set()
            for candidate in completed:
                lowered = candidate.lower()
                requirement = context_map.get(lowered) or default_context
                if requirement:
                    min_dt = requirement.get("min_dispatched_at")
                    thread_requirement = requirement.get("thread_index")
                    row_candidates = [
                        row
                        for row in rows
                        if str(getattr(row, "unique_id", "")).strip() == candidate
                        and getattr(row, "message_id", None)
                    ]
                    row_valid = False
                    for row in row_candidates:
                        row_dt = self._coerce_timestamp(
                            getattr(row, "dispatched_at", None)
                        )
                        if min_dt and (row_dt is None or row_dt < min_dt):
                            continue

                        if thread_requirement is not None:
                            row_thread = self._extract_thread_index(row)
                            if row_thread is None and min_dt is None:
                                continue
                            if row_thread is not None and row_thread < thread_requirement:
                                continue
                        row_valid = True
                        break

                    if not row_valid:
                        continue

                filtered.add(candidate)

            completed = filtered
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
        dispatch_context: Optional[Dict[object, object]] = None,
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
            wait_for_all=True,
            dispatch_context=dispatch_context,
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
