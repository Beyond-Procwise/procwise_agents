import json
import logging
import re
import imaplib
import time
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from email import message_from_bytes
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_dispatch_chain_store import pending_dispatch_count
from services.supplier_response_workflow import SupplierResponseWorkflow
from repositories import (
    draft_rfq_emails_repo,
    supplier_response_repo,
    workflow_email_tracking_repo,
)
from repositories.supplier_response_repo import SupplierResponseRow
from utils.gpu import configure_gpu

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from agents.negotiation_agent import NegotiationAgent

logger = logging.getLogger(__name__)


class SupplierInteractionAgent(BaseAgent):
    """Monitor and parse supplier RFQ responses."""

    AGENTIC_PLAN_STEPS = (
        "Gather supplier communications from monitored mailboxes or direct payloads.",
        "Classify intent, extract negotiation signals, and align with RFQ session state.",
        "Return structured responses or trigger downstream negotiation actions.",
    )

    WORKFLOW_POLL_INTERVAL_SECONDS = 30
    RESPONSE_GATE_POLL_SECONDS = 30

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)
        self._email_watcher = None
        self._negotiation_agent = None
        self._response_coordinator = SupplierResponseWorkflow()
        self._dispatch_schema_ready = False
        self._response_schema_ready = False

    # Suppliers occasionally include upper-case letters or non-hex characters
    # in the terminal segment (``RFQ-20240101-ABCD123Z``).  The updated pattern
    # tolerates the broader ``[A-Za-z0-9]{8}`` space while remaining
    # case-insensitive so mailbox matching and downstream database checks stay
    # aligned regardless of the original casing.
    RFQ_PATTERN = re.compile(r"RFQ-\d{8}-[A-Za-z0-9]{8}", re.IGNORECASE)

    @staticmethod
    def _draft_tracking_context(draft: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        context: Dict[str, Optional[str]] = {
            "workflow_id": None,
            "run_id": None,
            "unique_id": None,
            "supplier_id": None,
        }
        if not isinstance(draft, dict):
            return context

        context["workflow_id"] = SupplierInteractionAgent._coerce_text(
            draft.get("workflow_id")
        )
        context["run_id"] = SupplierInteractionAgent._coerce_text(
            draft.get("dispatch_run_id") or draft.get("run_id")
        )
        context["unique_id"] = SupplierInteractionAgent._coerce_text(
            draft.get("unique_id")
        )
        context["supplier_id"] = SupplierInteractionAgent._coerce_text(
            draft.get("supplier_id")
        )

        metadata = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
        if metadata:
            context.setdefault("workflow_id", None)
            if not context["workflow_id"]:
                context["workflow_id"] = SupplierInteractionAgent._coerce_text(
                    metadata.get("workflow_id") or metadata.get("process_workflow_id")
                )
            if not context["run_id"]:
                context["run_id"] = SupplierInteractionAgent._coerce_text(
                    metadata.get("dispatch_run_id") or metadata.get("run_id")
                )
            if not context["unique_id"]:
                context["unique_id"] = SupplierInteractionAgent._coerce_text(
                    metadata.get("unique_id")
                )
            if not context["supplier_id"]:
                context["supplier_id"] = SupplierInteractionAgent._coerce_text(
                    metadata.get("supplier_id")
                )

        return context

    @staticmethod
    def _coerce_text(value: Optional[Any]) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    def _normalise_thread_references(self, references: Optional[Any]) -> List[str]:
        normalised: List[str] = []

        def _append(value: Optional[Any]) -> None:
            if value in (None, ""):
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _append(item)
                return
            try:
                text = str(value).strip()
            except Exception:
                return
            if not text:
                return
            matches = re.findall(r"<[^>]+>", text)
            tokens = matches if matches else re.split(r"[\s,]+", text)
            for token in tokens:
                cleaned = token.strip()
                if cleaned and cleaned not in normalised:
                    normalised.append(cleaned)

        _append(references)
        return normalised

    def _merge_thread_references(
        self, existing: List[str], additions: List[str]
    ) -> List[str]:
        for ref in additions:
            if ref and ref not in existing:
                existing.append(ref)
        return existing

    def _build_thread_headers_payload(
        self, message_id: Optional[Any], references: List[str]
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {}
        message_token = self._coerce_text(message_id)
        if message_token:
            payload["message_id"] = message_token
        refs = [ref for ref in references if isinstance(ref, str) and ref.strip()]
        if refs:
            payload["references"] = refs
        return payload or None

    @staticmethod
    def _response_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        body = row.get("body_text") or row.get("response_text") or ""
        subject = row.get("subject") or ""
        supplier_id = row.get("supplier_id")
        workflow_id = row.get("workflow_id")
        unique_id = row.get("unique_id")
        message_id = row.get("message_id")
        from_addr = row.get("from_addr")
        received_at = row.get("received_at") or row.get("received_time")
        response = {
            "workflow_id": workflow_id,
            "unique_id": unique_id,
            "supplier_id": supplier_id,
            "message": body,
            "subject": subject,
            "message_id": message_id,
            "mailbox": row.get("mailbox"),
            "imap_uid": row.get("imap_uid"),
            "received_at": received_at,
            "supplier_status": AgentStatus.SUCCESS.value,
            "supplier_output": {
                "response_text": body,
                "supplier_id": supplier_id,
                "workflow_id": workflow_id,
                "unique_id": unique_id,
                "price": row.get("price"),
                "lead_time": row.get("lead_time"),
                "received_time": received_at,
            },
        }
        headers = {
            "from": from_addr,
            "to": (row.get("to_addrs") or "").split(","),
            "subject": subject,
            "message_id": message_id,
            "workflow_id": workflow_id,
            "unique_id": unique_id,
            "supplier_id": supplier_id,
            "price": row.get("price"),
            "lead_time": row.get("lead_time"),
            "received_time": received_at,
        }
        response["email_headers"] = headers
        return response

    @staticmethod
    def _serialise_pending_row(row: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(row, dict):
            return {}
        return {
            "workflow_id": row.get("workflow_id"),
            "unique_id": row.get("unique_id"),
            "supplier_id": row.get("supplier_id"),
            "body_text": row.get("response_text") or "",
            "subject": row.get("subject"),
            "message_id": row.get("message_id"),
            "from_addr": row.get("from_addr"),
            "received_at": row.get("received_time"),
            "price": row.get("price"),
            "lead_time": row.get("lead_time"),
        }

    def _normalise_poll_interval(self, poll_interval: Optional[int]) -> float:
        default_interval = getattr(
            getattr(self.agent_nick, "settings", None),
            "email_response_poll_seconds",
            self.WORKFLOW_POLL_INTERVAL_SECONDS,
        )
        try:
            default_interval = float(default_interval)
        except Exception:
            default_interval = float(self.WORKFLOW_POLL_INTERVAL_SECONDS)
        if default_interval <= 0:
            default_interval = float(self.WORKFLOW_POLL_INTERVAL_SECONDS)

        if poll_interval is not None:
            try:
                interval = float(poll_interval)
            except Exception:
                interval = default_interval
            else:
                if interval <= 0:
                    interval = default_interval
        else:
            interval = default_interval

        if interval < 0:
            interval = 0.0
        return interval

    def _collect_workflow_unique_ids(
        self,
        *,
        workflow_id: Optional[str],
        seed_ids: Sequence[Optional[str]],
        timeout: Optional[int],
        poll_interval: Optional[int],
        expected_total: Optional[int] = None,
    ) -> Tuple[List[str], float]:
        start = time.monotonic()
        collected: List[str] = []
        seen: Set[str] = set()

        for value in seed_ids:
            text = self._coerce_text(value)
            if text and text not in seen:
                seen.add(text)
                collected.append(text)

        if not workflow_id:
            return collected, time.monotonic() - start

        interval = self._normalise_poll_interval(poll_interval)
        sleep_interval = interval if interval > 0 else 0.5

        try:
            total_timeout = float(timeout) if timeout is not None else 0.0
        except Exception:
            total_timeout = 0.0

        if total_timeout < 0:
            total_timeout = 0.0

        deadline = start + total_timeout if total_timeout > 0 else None

        target_total = 0
        if expected_total is not None:
            try:
                target_total = int(expected_total)
            except Exception:
                target_total = 0
        if target_total < 0:
            target_total = 0
        if target_total < len(collected):
            target_total = len(collected)

        while True:
            try:
                workflow_email_tracking_repo.init_schema()
                unique_rows = workflow_email_tracking_repo.load_workflow_unique_ids(
                    workflow_id=workflow_id
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to load workflow unique IDs for workflow=%s", workflow_id
                )
                unique_rows = []

            for value in unique_rows:
                text = self._coerce_text(value)
                if text and text not in seen:
                    seen.add(text)
                    collected.append(text)

            if target_total:
                if len(collected) >= target_total:
                    break
            elif collected:
                break

            if deadline is not None and time.monotonic() >= deadline:
                break

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(sleep_interval, remaining))
            else:
                time.sleep(min(sleep_interval, 1.0))

        return collected, time.monotonic() - start

    def _await_dispatch_ready(
        self,
        *,
        workflow_id: Optional[str],
        unique_ids: Sequence[Optional[str]],
        timeout: Optional[int],
        poll_interval: Optional[int],
        expected_total: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not workflow_id:
            return None

        interval = self._normalise_poll_interval(poll_interval)

        candidates, elapsed = self._collect_workflow_unique_ids(
            workflow_id=workflow_id,
            seed_ids=unique_ids,
            timeout=timeout,
            poll_interval=poll_interval,
            expected_total=expected_total,
        )

        coordinator = getattr(self, "_response_coordinator", None)
        if coordinator is None:
            coordinator = SupplierResponseWorkflow()
            self._response_coordinator = coordinator

        remaining_timeout: Optional[int]
        if timeout is None:
            remaining_timeout = None
        else:
            try:
                total_timeout = float(timeout)
            except Exception:
                total_timeout = 0.0
            if total_timeout <= 0:
                remaining_timeout = 0
            else:
                remaining = max(0.0, total_timeout - elapsed)
                remaining_timeout = int(remaining) if remaining > 0 else 0

        poll_value: Optional[int]
        if interval <= 0:
            poll_value = None
        else:
            poll_value = max(1, int(round(interval)))

        try:
            summary = coordinator.await_dispatch_completion(
                workflow_id=workflow_id,
                unique_ids=candidates,
                timeout=remaining_timeout,
                poll_interval=poll_value,
                wait_for_all=True,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to await dispatch completion for workflow=%s", workflow_id
            )
            return None

        if candidates and not summary.get("unique_ids"):
            summary["unique_ids"] = list(candidates)
        if candidates and not summary.get("expected_dispatches"):
            summary["expected_dispatches"] = len(candidates)

        if not summary.get("complete"):
            logger.warning(
                "Dispatch metadata incomplete for workflow=%s (completed=%s/%s)",
                workflow_id,
                summary.get("completed_dispatches"),
                summary.get("expected_dispatches"),
            )
        else:
            logger.debug(
                "Dispatch ready for workflow=%s covering %s unique IDs",
                workflow_id,
                summary.get("expected_dispatches"),
            )
        return summary

    def _prepare_response_expectations(
        self,
        metadata: Dict[str, Any],
        *,
        supplier_filter: Optional[Set[str]] = None,
        unique_filter: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        dispatch_rows = metadata.get("rows") or []
        dispatch_index: Dict[str, Dict[str, Optional[str]]] = {}
        for row in dispatch_rows:
            unique_value = self._coerce_text(getattr(row, "unique_id", None))
            supplier_value = self._coerce_text(getattr(row, "supplier_id", None))
            if unique_value:
                dispatch_index[unique_value] = {"supplier_id": supplier_value}

        normalised_unique_filter: Optional[Set[str]] = None
        if unique_filter:
            normalised_unique_filter = {
                self._coerce_text(value)
                for value in unique_filter
                if self._coerce_text(value)
            }

        normalised_supplier_filter: Optional[Set[str]] = None
        if supplier_filter:
            normalised_supplier_filter = {
                self._coerce_text(value)
                for value in supplier_filter
                if self._coerce_text(value)
            }

        expected_unique_ids: Set[str] = set(dispatch_index.keys())
        if normalised_supplier_filter:
            expected_unique_ids = {
                uid
                for uid in expected_unique_ids
                if dispatch_index.get(uid, {}).get("supplier_id") in normalised_supplier_filter
            }

        if normalised_unique_filter is not None:
            expected_unique_ids &= normalised_unique_filter
            if not expected_unique_ids and normalised_unique_filter:
                expected_unique_ids = set(normalised_unique_filter)

        expected_suppliers: Set[str] = {
            dispatch_index[uid]["supplier_id"]
            for uid in expected_unique_ids
            if dispatch_index.get(uid, {}).get("supplier_id")
        }
        if not expected_suppliers and normalised_supplier_filter:
            expected_suppliers = set(normalised_supplier_filter)

        return {
            "dispatch_rows": dispatch_rows,
            "dispatch_index": dispatch_index,
            "normalised_unique_filter": normalised_unique_filter,
            "normalised_supplier_filter": normalised_supplier_filter,
            "expected_unique_ids": expected_unique_ids,
            "expected_suppliers": expected_suppliers,
        }

    def _load_dispatch_metadata(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        try:
            workflow_email_tracking_repo.init_schema()
            rows = workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id)
        except Exception:
            logger.exception("Failed to load dispatch metadata for workflow=%s", workflow_id)
            return None

        if not rows:
            logger.info("No dispatch records stored for workflow=%s", workflow_id)
            return None

        pending = [
            row.unique_id
            for row in rows
            if row.dispatched_at is None or not row.message_id
        ]
        if pending:
            logger.debug(
                "Dispatch metadata incomplete for workflow=%s (pending unique_ids=%s)",
                workflow_id,
                pending,
            )
            return None

        last_dispatched = max(
            (row.dispatched_at for row in rows if row.dispatched_at),
            default=None,
        )
        if last_dispatched is None:
            logger.debug("Unable to determine dispatch timestamp for workflow=%s", workflow_id)
            return None

        return {
            "rows": rows,
            "last_dispatched_at": last_dispatched,
            "unique_ids": [row.unique_id for row in rows if row.unique_id],
        }

    def _await_supplier_response_rows(
        self,
        workflow_id: str,
        *,
        supplier_filter: Optional[Set[str]] = None,
        unique_filter: Optional[Set[str]] = None,
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
        dispatch_summary: Optional[Dict[str, Any]] = None,
        include_processed: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            total_timeout = float(timeout) if timeout is not None else 0.0
        except Exception:
            total_timeout = 0.0

        if total_timeout < 0:
            total_timeout = 0.0

        start = time.monotonic()
        deadline = start + total_timeout if total_timeout > 0 else None
        interval = self._normalise_poll_interval(poll_interval)

        watch_ids: Sequence[Optional[str]]
        if unique_filter:
            watch_ids = list(unique_filter)
        else:
            watch_ids = []

        dispatch_summary = dispatch_summary or self._await_dispatch_ready(
            workflow_id=workflow_id,
            unique_ids=watch_ids,
            timeout=timeout,
            poll_interval=poll_interval,
            expected_total=len([uid for uid in watch_ids if self._coerce_text(uid)])
            if watch_ids
            else None,
        )

        if not dispatch_summary or not dispatch_summary.get("complete"):
            logger.info(
                "Dispatch phase incomplete for workflow=%s; summary=%s",
                workflow_id,
                dispatch_summary,
            )
            return []

        metadata: Optional[Dict[str, Any]] = None
        while True:
            metadata = self._load_dispatch_metadata(workflow_id)
            if metadata is not None:
                break

            if deadline is not None and time.monotonic() >= deadline:
                break

            if interval > 0:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(interval, remaining, 1.0))
                else:
                    time.sleep(min(interval, 1.0))
            else:
                time.sleep(0)

        if metadata is None:
            logger.info(
                "Dispatch metadata unavailable after activation for workflow=%s",
                workflow_id,
            )
            return []

        dispatch_rows: List[Any] = list(metadata.get("rows") or [])
        dispatch_total = len(dispatch_rows)
        if dispatch_total <= 0:
            logger.info(
                "No dispatched emails recorded for workflow=%s; nothing to await", workflow_id
            )
            return []

        metadata_unique_ids = [
            self._coerce_text(value)
            for value in metadata.get("unique_ids", [])
            if self._coerce_text(value)
        ]
        expected_unique_ids: Set[str] = set(metadata_unique_ids)

        if unique_filter is not None:
            filtered_ids = {
                self._coerce_text(value)
                for value in unique_filter
                if self._coerce_text(value)
            }
            if filtered_ids:
                expected_unique_ids &= filtered_ids
                if not expected_unique_ids:
                    expected_unique_ids = filtered_ids

        if not expected_unique_ids:
            candidate_ids = dispatch_summary.get("unique_ids") or []
            expected_unique_ids = {
                self._coerce_text(value)
                for value in candidate_ids
                if self._coerce_text(value)
            }

        coordinator = getattr(self, "_response_coordinator", None)
        if coordinator is None:
            coordinator = SupplierResponseWorkflow()
            self._response_coordinator = coordinator

        remaining_timeout: Optional[int]
        if deadline is None:
            remaining_timeout = timeout if isinstance(timeout, int) else None
        else:
            remaining = max(0.0, deadline - time.monotonic())
            remaining_timeout = int(remaining) if remaining > 0 else 0

        poll_value: Optional[int]
        if interval <= 0:
            poll_value = None
        else:
            poll_value = max(1, int(round(interval)))

        try:
            response_summary = coordinator.await_response_population(
                workflow_id=workflow_id,
                unique_ids=list(expected_unique_ids),
                timeout=remaining_timeout,
                poll_interval=poll_value,
                wait_for_all=True,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to await supplier responses for workflow=%s", workflow_id
            )
            return []

        if not response_summary.get("complete"):
            logger.info(
                "Supplier responses incomplete for workflow=%s (received=%s/%s)",
                workflow_id,
                response_summary.get("completed_responses"),
                response_summary.get("expected_responses"),
            )
            return []

        attempts = 0
        max_attempts_without_deadline = 3 if interval > 0 else 1

        while True:
            rows = self._poll_supplier_response_rows(
                workflow_id,
                supplier_filter=supplier_filter,
                unique_filter=expected_unique_ids,
                metadata=metadata,
                include_processed=include_processed,
            )

            if rows:
                return self._process_responses_concurrently(rows)

            logger.debug(
                "Response poll returned no rows despite completion for workflow=%s",
                workflow_id,
            )

            if deadline is not None and time.monotonic() >= deadline:
                break

            if interval > 0:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(interval, remaining, 1.0))
                else:
                    attempts += 1
                    if attempts >= max_attempts_without_deadline:
                        break
                    time.sleep(min(interval, 1.0))
            else:
                break

        return []

    def _expected_dispatch_context(
        self, workflow_id: str
    ) -> Dict[str, Any]:
        """Resolve workflow dispatch expectations from draft storage."""

        expected_ids: Set[str] = set()
        supplier_index: Dict[str, Optional[str]] = {}
        last_dispatched_at: Optional[datetime] = None

        try:
            draft_ids, supplier_map, last_dt = (
                draft_rfq_emails_repo.expected_unique_ids_and_last_dispatch(
                    workflow_id=workflow_id
                )
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to load draft expectations for workflow=%s", workflow_id,
                exc_info=True,
            )
            draft_ids, supplier_map, last_dt = set(), {}, None

        for uid in draft_ids:
            text = self._coerce_text(uid)
            if text:
                expected_ids.add(text)

        for uid, supplier in supplier_map.items():
            key = self._coerce_text(uid)
            if not key:
                continue
            supplier_index[key] = self._coerce_text(supplier)

        if last_dt:
            last_dispatched_at = last_dt

        return {
            "expected_unique_ids": expected_ids,
            "supplier_index": supplier_index,
            "last_dispatched_at": last_dispatched_at,
            "expected_count": len(expected_ids),
        }

    def _await_dispatch_gate(
        self,
        workflow_id: str,
        *,
        timeout: Optional[int],
        poll_interval: Optional[int],
    ) -> Dict[str, Any]:
        """Ensure all expected supplier dispatches have been sent."""

        workflow_key = self._coerce_text(workflow_id)
        summary: Dict[str, Any] = {
            "complete": False,
            "expected_count": 0,
            "unique_ids": [],
            "rows": [],
        }

        if not workflow_key:
            return summary

        try:
            total_timeout = float(timeout) if timeout is not None else 0.0
        except Exception:
            total_timeout = 0.0

        if total_timeout < 0:
            total_timeout = 0.0

        interval = self._normalise_poll_interval(poll_interval)
        if interval <= 0:
            interval = 1.0

        start = time.monotonic()
        deadline = start + total_timeout if total_timeout > 0 else None

        while True:
            expectations = self._expected_dispatch_context(workflow_key)
            expected_ids: Set[str] = expectations.get("expected_unique_ids", set())

            try:
                workflow_email_tracking_repo.init_schema()
                rows = workflow_email_tracking_repo.load_workflow_rows(
                    workflow_id=workflow_key
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to load dispatch metadata while gating workflow=%s",
                    workflow_key,
                )
                rows = []

            dispatched_rows = [
                row
                for row in rows
                if getattr(row, "message_id", None)
                and getattr(row, "dispatched_at", None)
            ]
            dispatched_ids = {
                self._coerce_text(getattr(row, "unique_id", None))
                for row in dispatched_rows
                if self._coerce_text(getattr(row, "unique_id", None))
            }

            if not expected_ids and dispatched_ids:
                expected_ids = set(dispatched_ids)

            expected_total = max(len(expected_ids), expectations.get("expected_count", 0))

            summary.update(
                {
                    "expected_count": expected_total,
                    "unique_ids": sorted(expected_ids),
                    "rows": dispatched_rows,
                }
            )

            if expected_total == 0 and not dispatched_rows:
                logger.debug(
                    "No dispatch expectations recorded for workflow=%s", workflow_key
                )
                if deadline is not None and time.monotonic() >= deadline:
                    return summary
            elif expected_ids.issubset(dispatched_ids) and len(dispatched_ids) >= expected_total:
                summary["complete"] = True
                return summary

            if deadline is not None and time.monotonic() >= deadline:
                logger.info(
                    "Dispatch gate timed out for workflow=%s after %.1fs", workflow_key, time.monotonic() - start
                )
                return summary

            time.sleep(min(interval, 5.0))

    def _await_response_gate(
        self,
        workflow_id: str,
        *,
        expected_unique_ids: Sequence[str],
        expected_count: int,
        timeout: Optional[int],
    ) -> Dict[str, Any]:
        """Wait for all supplier responses corresponding to dispatches."""

        workflow_key = self._coerce_text(workflow_id)
        gate_summary: Dict[str, Any] = {
            "complete": False,
            "responses": [],
            "collected_count": 0,
        }

        if not workflow_key:
            return gate_summary

        expected_set: Set[str] = {
            self._coerce_text(uid)
            for uid in expected_unique_ids
            if self._coerce_text(uid)
        }

        if expected_count <= 0 and not expected_set:
            gate_summary["complete"] = True
            gate_summary["responses"] = []
            return gate_summary

        try:
            total_timeout = float(timeout) if timeout is not None else 0.0
        except Exception:
            total_timeout = 0.0

        if total_timeout < 0:
            total_timeout = 0.0

        start = time.monotonic()
        deadline = start + total_timeout if total_timeout > 0 else None

        interval = max(float(self.RESPONSE_GATE_POLL_SECONDS), 1.0)

        supplier_response_repo.init_schema()

        while True:
            pending_rows = supplier_response_repo.fetch_pending(
                workflow_id=workflow_key, include_processed=False
            )
            pending_ids = {
                self._coerce_text(row.get("unique_id"))
                for row in pending_rows
                if self._coerce_text(row.get("unique_id"))
            }

            if expected_count > 0:
                ready = expected_set.issubset(pending_ids) and len(pending_ids) >= expected_count
            else:
                ready = bool(pending_ids)

            if ready:
                responses = self._process_responses_concurrently(pending_rows)
                gate_summary.update(
                    {
                        "complete": True,
                        "responses": responses,
                        "collected_count": len(responses),
                    }
                )
                return gate_summary

            if deadline is not None and time.monotonic() >= deadline:
                logger.info(
                    "Response gate timed out for workflow=%s after %.1fs",
                    workflow_key,
                    time.monotonic() - start,
                )
                gate_summary["responses"] = self._process_responses_concurrently(
                    pending_rows
                )
                gate_summary["collected_count"] = len(gate_summary["responses"])
                return gate_summary

            if deadline is None:
                time.sleep(interval)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    gate_summary["responses"] = self._process_responses_concurrently(
                        pending_rows
                    )
                    gate_summary["collected_count"] = len(gate_summary["responses"])
                    return gate_summary
                time.sleep(min(interval, remaining))

    def _await_full_response_batch(
        self,
        workflow_id: str,
        *,
        timeout: Optional[int],
        poll_interval: Optional[int],
    ) -> Dict[str, Any]:
        workflow_key = self._coerce_text(workflow_id)
        result: Dict[str, Any] = {
            "ready": False,
            "expected_count": 0,
            "collected_count": 0,
            "responses": [],
            "unique_ids": [],
        }

        if not workflow_key:
            return result

        dispatch_state = self._await_dispatch_gate(
            workflow_key, timeout=timeout, poll_interval=poll_interval
        )

        result["expected_count"] = dispatch_state.get("expected_count", 0)
        result["unique_ids"] = list(dispatch_state.get("unique_ids", []))

        if not dispatch_state.get("complete"):
            logger.info(
                "Dispatch verification gate incomplete for workflow=%s",
                workflow_key,
            )
            return result

        response_state = self._await_response_gate(
            workflow_key,
            expected_unique_ids=result["unique_ids"],
            expected_count=result["expected_count"],
            timeout=timeout,
        )

        result["collected_count"] = response_state.get("collected_count", 0)

        if not response_state.get("complete"):
            logger.info(
                "Response aggregation gate incomplete for workflow=%s",
                workflow_key,
            )
            return result

        responses = response_state.get("responses", [])
        result.update(
            {
                "ready": True,
                "responses": responses,
                "collected_count": len(responses),
                "unique_ids": [
                    self._coerce_text(response.get("unique_id"))
                    for response in responses
                    if self._coerce_text(response.get("unique_id"))
                ]
                or result["unique_ids"],
            }
        )

        return result

    def _poll_supplier_response_rows(
        self,
        workflow_id: str,
        *,
        supplier_filter: Optional[Set[str]] = None,
        unique_filter: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_processed: bool = True,
    ) -> List[Dict[str, Any]]:
        if metadata is None:
            metadata = self._load_dispatch_metadata(workflow_id)
        if metadata is None:
            logger.debug(
                "Dispatch metadata unavailable for workflow=%s; skipping response poll",
                workflow_id,
            )
            return []

        try:
            supplier_response_repo.init_schema()
        except Exception:
            logger.exception("Failed to initialise supplier response schema for workflow=%s", workflow_id)
            return []

        expectations = self._prepare_response_expectations(
            metadata,
            supplier_filter=supplier_filter,
            unique_filter=unique_filter,
        )
        dispatch_rows = expectations["dispatch_rows"]
        dispatch_index: Dict[str, Dict[str, Optional[str]]] = expectations[
            "dispatch_index"
        ]
        normalised_unique_filter = expectations["normalised_unique_filter"]
        normalised_supplier_filter = expectations["normalised_supplier_filter"]
        expected_unique_ids: Set[str] = expectations["expected_unique_ids"]
        expected_suppliers: Set[str] = expectations["expected_suppliers"]

        pending_rows = supplier_response_repo.fetch_pending(
            workflow_id=workflow_id,
            include_processed=include_processed,
        )
        serialised = [
            self._serialise_pending_row(row)
            for row in pending_rows
            if isinstance(row, dict)
        ]

        if normalised_supplier_filter is not None:
            serialised = [
                row
                for row in serialised
                if self._coerce_text(row.get("supplier_id")) in normalised_supplier_filter
            ]

        if normalised_unique_filter is not None:
            serialised = [
                row
                for row in serialised
                if self._coerce_text(row.get("unique_id")) in normalised_unique_filter
            ]

        if not serialised:
            return []

        collected_by_unique: Dict[str, Dict[str, Any]] = {}
        collected_by_supplier: Dict[str, Dict[str, Any]] = {}
        other_rows: List[Dict[str, Any]] = []

        for row in serialised:
            unique_id = self._coerce_text(row.get("unique_id"))
            supplier_id = self._coerce_text(row.get("supplier_id"))
            if unique_id:
                if not expected_unique_ids or unique_id in expected_unique_ids:
                    collected_by_unique[unique_id] = row
                else:
                    other_rows.append(row)
            elif supplier_id:
                if not expected_suppliers or supplier_id in expected_suppliers:
                    collected_by_supplier[supplier_id] = row
                else:
                    other_rows.append(row)
            else:
                other_rows.append(row)

        if expected_unique_ids:
            if set(collected_by_unique.keys()) != expected_unique_ids:
                return []
        elif expected_suppliers:
            if set(collected_by_supplier.keys()) != expected_suppliers:
                return []

        ordered_results: List[Dict[str, Any]] = []
        if expected_unique_ids:
            ordered_unique_ids = [
                self._coerce_text(uid)
                for uid in metadata.get("unique_ids", [])
                if self._coerce_text(uid) in expected_unique_ids
            ]
            for uid in ordered_unique_ids:
                row = collected_by_unique.get(uid)
                if row and row not in ordered_results:
                    ordered_results.append(row)
            for uid in expected_unique_ids:
                row = collected_by_unique.get(uid)
                if row and row not in ordered_results:
                    ordered_results.append(row)
        elif expected_suppliers:
            ordered_suppliers = []
            for row in dispatch_rows:
                supplier_id = self._coerce_text(getattr(row, "supplier_id", None))
                if supplier_id and supplier_id in expected_suppliers:
                    ordered_suppliers.append(supplier_id)
            for supplier_id in ordered_suppliers:
                row = collected_by_supplier.get(supplier_id)
                if row and row not in ordered_results:
                    ordered_results.append(row)
            for supplier_id in expected_suppliers:
                row = collected_by_supplier.get(supplier_id)
                if row and row not in ordered_results:
                    ordered_results.append(row)

        if not ordered_results:
            ordered_results.extend(collected_by_unique.values())
            for row in collected_by_supplier.values():
                if row not in ordered_results:
                    ordered_results.append(row)

        for row in other_rows:
            if row not in ordered_results:
                ordered_results.append(row)

        return ordered_results

    def _process_responses_concurrently(
        self, rows: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        max_workers = os.cpu_count() or 1
        max_workers = max(1, min(len(rows), max_workers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._response_from_row, row) for row in rows]
            return [future.result() for future in futures]

    def _ensure_dispatch_tracking_schema(self) -> None:
        if self._dispatch_schema_ready:
            return
        try:
            workflow_email_tracking_repo.init_schema()
        except Exception:
            logger.exception(
                "Failed to initialise workflow email tracking schema before dispatch gate",
            )
        else:
            self._dispatch_schema_ready = True

    def _ensure_response_schema(self) -> None:
        if self._response_schema_ready:
            return
        try:
            supplier_response_repo.init_schema()
        except Exception:
            logger.exception(
                "Failed to initialise supplier response schema before response gate",
            )
        else:
            self._response_schema_ready = True

    @staticmethod
    def _serialise_dispatch_row(row: Any) -> Dict[str, Any]:
        payload = {
            "workflow_id": getattr(row, "workflow_id", None),
            "unique_id": getattr(row, "unique_id", None),
            "supplier_id": getattr(row, "supplier_id", None),
            "supplier_email": getattr(row, "supplier_email", None),
            "message_id": getattr(row, "message_id", None),
            "subject": getattr(row, "subject", None),
            "matched": getattr(row, "matched", None),
        }
        dispatched_at = getattr(row, "dispatched_at", None)
        responded_at = getattr(row, "responded_at", None)
        if isinstance(dispatched_at, datetime):
            payload["dispatched_at"] = dispatched_at.isoformat()
        else:
            payload["dispatched_at"] = dispatched_at
        if isinstance(responded_at, datetime):
            payload["responded_at"] = responded_at.isoformat()
        else:
            payload["responded_at"] = responded_at
        return payload

    @staticmethod
    def _normalise_unique_ids(raw: Optional[Any]) -> List[str]:
        if raw in (None, "", [], (), {}):
            return []
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, (set, tuple, list)):
            candidates = list(raw)
        else:
            candidates = [raw]

        normalised: List[str] = []
        for candidate in candidates:
            if candidate in (None, ""):
                continue
            try:
                text = str(candidate).strip()
            except Exception:
                continue
            if text:
                normalised.append(text)
        if not normalised:
            return []
        try:
            # Preserve order while deduplicating
            normalised = list(dict.fromkeys(normalised))
        except Exception:
            pass
        return normalised

    def _await_dispatch_gate(
        self,
        workflow_id: str,
        *,
        expected_count: Optional[int] = None,
        poll_interval: Optional[float] = None,
        timeout: Optional[int] = None,
        expected_unique_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        expected_ids = list(expected_unique_ids or [])
        normalised_ids = [self._coerce_text(uid) for uid in expected_ids if self._coerce_text(uid)]
        expected_ids = list(dict.fromkeys(normalised_ids)) if normalised_ids else []

        explicit_count = expected_count if isinstance(expected_count, int) else None
        count_hint = explicit_count if explicit_count is not None else 0
        if expected_ids:
            count_hint = max(len(expected_ids), count_hint)
        context_expectations: Optional[Dict[str, Any]] = None
        if not expected_ids:
            try:
                context_expectations = self._expected_dispatch_context(workflow_id)
            except Exception:  # pragma: no cover - defensive lookup
                logger.exception(
                    "Failed to load expected dispatch context for workflow=%s", workflow_id
                )
                context_expectations = None
            else:
                context_ids = context_expectations.get("expected_unique_ids", set()) if isinstance(context_expectations, dict) else set()
                if context_ids:
                    expected_ids = [self._coerce_text(uid) for uid in context_ids if self._coerce_text(uid)]
                    expected_ids = [uid for uid in expected_ids if uid]
                    if expected_ids:
                        count_hint = max(count_hint, len(expected_ids), context_expectations.get("expected_count", 0))

        if expected_ids and count_hint <= 0:
            count_hint = len(expected_ids)

        if explicit_count is not None and explicit_count <= 0 and not expected_ids:
            return {
                "workflow_id": workflow_id,
                "expected_dispatches": 0,
                "completed_dispatches": 0,
                "complete": True,
                "wait_seconds": 0.0,
                "unique_ids": [],
                "collected_unique_ids": [],
                "dispatch_records": [],
                "expected_unique_ids": [],
                "pending_unique_ids": [],
                "expected_count": 0,
                "completed_count": 0,
                "timed_out": False,
            }

        self._ensure_dispatch_tracking_schema()

        poll_seconds = max(0.0, float(poll_interval) if poll_interval is not None else self._normalise_poll_interval(None))
        deadline = None
        if timeout is not None and timeout > 0:
            try:
                deadline = time.monotonic() + float(timeout)
            except Exception:
                deadline = time.monotonic() + float(timeout or 0)
        start = time.monotonic()

        dispatch_map: Dict[str, Dict[str, Any]] = {}
        complete = False
        expected_set = {uid for uid in expected_ids if uid}

        while True:
            try:
                rows = workflow_email_tracking_repo.load_workflow_rows(
                    workflow_id=workflow_id
                )
            except Exception:
                logger.exception(
                    "Failed to load dispatch rows for workflow=%s", workflow_id
                )
                rows = []

            current: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                unique_id = self._coerce_text(getattr(row, "unique_id", None))
                if not unique_id:
                    continue
                if expected_set and unique_id not in expected_set:
                    continue
                dispatched_at = getattr(row, "dispatched_at", None)
                message_id = getattr(row, "message_id", None)
                if dispatched_at in (None, "") or message_id in (None, ""):
                    continue
                current[unique_id] = self._serialise_dispatch_row(row)

            dispatch_map = current
            if expected_set:
                if expected_set.issubset(dispatch_map.keys()) and (
                    not count_hint or len(dispatch_map) >= count_hint
                ):
                    complete = True
                    break
            else:
                if not expected_set and context_expectations:
                    context_ids = context_expectations.get("expected_unique_ids", set())
                    if context_ids:
                        expected_ids = [
                            self._coerce_text(uid)
                            for uid in context_ids
                            if self._coerce_text(uid)
                        ]
                        expected_ids = [uid for uid in expected_ids if uid]
                        expected_set = {uid for uid in expected_ids if uid}
                        if expected_set:
                            count_hint = max(count_hint, len(expected_set), context_expectations.get("expected_count", 0))
                            if expected_set.issubset(dispatch_map.keys()):
                                complete = True
                                break
                if not expected_set and dispatch_map:
                    inferred_ids = [uid for uid in dispatch_map.keys() if uid]
                    if inferred_ids:
                        expected_ids = inferred_ids
                        expected_set = set(inferred_ids)
                        count_hint = max(count_hint, len(expected_set))
                        if not count_hint or len(dispatch_map) >= count_hint:
                            complete = True
                            break
                if count_hint and len(dispatch_map) >= count_hint:
                    complete = True
                    break

            if deadline is not None and time.monotonic() >= deadline:
                break

            if poll_seconds > 0:
                time.sleep(poll_seconds)
            else:
                time.sleep(0)

        elapsed = time.monotonic() - start
        pending = list(expected_ids) if expected_set else []
        if expected_set and dispatch_map:
            pending = [uid for uid in expected_ids if uid not in dispatch_map]
        collected_ids = list(dispatch_map.keys())
        visible_unique_ids = list(expected_ids) if expected_ids else collected_ids
        return {
            "workflow_id": workflow_id,
            "expected_dispatches": count_hint,
            "expected_count": count_hint,
            "completed_dispatches": len(dispatch_map),
            "completed_count": len(dispatch_map),
            "complete": complete,
            "wait_seconds": elapsed,
            "unique_ids": visible_unique_ids,
            "collected_unique_ids": collected_ids,
            "dispatch_records": list(dispatch_map.values()),
            "expected_unique_ids": expected_ids,
            "pending_unique_ids": pending,
            "timed_out": bool(not complete and deadline is not None),
        }

    def _await_response_gate(
        self,
        workflow_id: str,
        *,
        expected_count: Optional[int] = None,
        poll_interval: Optional[float] = None,
        timeout: Optional[int] = None,
        expected_unique_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        expected_ids = list(expected_unique_ids or [])
        normalised_ids = [self._coerce_text(uid) for uid in expected_ids if self._coerce_text(uid)]
        expected_ids = list(dict.fromkeys(normalised_ids)) if normalised_ids else []

        explicit_count = expected_count if isinstance(expected_count, int) else None
        count_hint = explicit_count if explicit_count is not None else 0
        if expected_ids:
            count_hint = max(len(expected_ids), count_hint)

        if expected_ids and count_hint <= 0:
            count_hint = len(expected_ids)

        if explicit_count is not None and explicit_count <= 0 and not expected_ids:
            return {
                "workflow_id": workflow_id,
                "expected_responses": 0,
                "expected_count": 0,
                "collected_responses": 0,
                "completed_responses": 0,
                "collected_count": 0,
                "complete": True,
                "wait_seconds": 0.0,
                "responses": [],
                "unique_ids": [],
                "collected_unique_ids": [],
                "expected_unique_ids": [],
                "pending_unique_ids": [],
                "timed_out": False,
            }

        self._ensure_response_schema()

        base_interval = (
            float(poll_interval)
            if poll_interval is not None
            else self._normalise_poll_interval(None)
        )
        poll_seconds = max(0.0, base_interval) or float(self.RESPONSE_GATE_POLL_SECONDS)
        deadline = None
        if timeout is not None and timeout > 0:
            try:
                deadline = time.monotonic() + float(timeout)
            except Exception:
                deadline = time.monotonic() + float(timeout or 0)
        start = time.monotonic()

        response_map: Dict[str, Dict[str, Any]] = {}
        complete = False
        expected_set = {uid for uid in expected_ids if uid}

        while True:
            try:
                rows = supplier_response_repo.fetch_pending(
                    workflow_id=workflow_id, include_processed=False
                )
            except Exception:
                logger.exception(
                    "Failed to load pending supplier responses for workflow=%s", workflow_id
                )
                try:
                    rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
                except Exception:
                    logger.exception(
                        "Failed to load supplier responses for workflow=%s", workflow_id
                    )
                    rows = []

            current: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                unique_id = self._coerce_text(row.get("unique_id"))
                if not unique_id:
                    continue
                if expected_set and unique_id not in expected_set:
                    continue
                current.setdefault(unique_id, dict(row))

            response_map = current
            if expected_set:
                if expected_set.issubset(response_map.keys()):
                    complete = True
                    break
            else:
                if count_hint and len(response_map) >= count_hint:
                    complete = True
                    break

            if deadline is not None and time.monotonic() >= deadline:
                break

            if poll_seconds > 0:
                time.sleep(poll_seconds)
            else:
                time.sleep(0)

        elapsed = time.monotonic() - start
        pending = list(expected_ids) if expected_set else []
        if expected_set and response_map:
            pending = [uid for uid in expected_ids if uid not in response_map]
        collected_ids = list(response_map.keys())
        visible_unique_ids = list(expected_ids) if expected_ids else collected_ids
        return {
            "workflow_id": workflow_id,
            "expected_responses": count_hint,
            "expected_count": count_hint,
            "collected_responses": len(response_map),
            "completed_responses": len(response_map),
            "collected_count": len(response_map),
            "complete": complete,
            "wait_seconds": elapsed,
            "responses": list(response_map.values()),
            "unique_ids": visible_unique_ids,
            "collected_unique_ids": collected_ids,
            "expected_unique_ids": expected_ids,
            "pending_unique_ids": pending,
            "timed_out": bool(not complete and deadline is not None),
        }

    def _blocking_dispatch_gate(
        self,
        *,
        workflow_id: str,
        expected_count: int,
        poll_interval: Optional[float],
        expected_unique_ids: Sequence[str],
        timeout_seconds: Optional[float],
    ) -> Dict[str, Any]:
        self._ensure_dispatch_tracking_schema()
        expected_ids = [self._coerce_text(uid) for uid in expected_unique_ids if self._coerce_text(uid)]
        expected_set = {uid for uid in expected_ids if uid}
        required = max(expected_count, len(expected_set))
        start = time.monotonic()

        total_timeout = 0.0
        if timeout_seconds is not None:
            try:
                total_timeout = float(timeout_seconds)
            except Exception:
                total_timeout = 0.0
            if total_timeout < 0:
                total_timeout = 0.0

        deadline = start + total_timeout if total_timeout > 0 else None

        if required <= 0:
            return {
                "workflow_id": workflow_id,
                "expected_dispatches": required,
                "completed_dispatches": 0,
                "unique_ids": [],
                "dispatch_rows": [],
                "thread_headers": {},
                "complete": True,
                "wait_seconds": 0.0,
                "timed_out": False,
            }

        interval = poll_interval or self.WORKFLOW_POLL_INTERVAL_SECONDS
        interval = max(1.0, float(interval))

        last_logged = -1

        while True:
            rows = workflow_email_tracking_repo.load_workflow_rows(workflow_id=workflow_id)
            dispatch_rows = [row for row in rows if getattr(row, "dispatched_at", None)]

            ordered: Dict[str, Any] = {}
            thread_headers: Dict[str, Any] = {}
            for row in dispatch_rows:
                unique_id = self._coerce_text(getattr(row, "unique_id", None))
                if expected_set and unique_id not in expected_set:
                    continue
                if unique_id:
                    ordered.setdefault(unique_id, row)
                    headers = getattr(row, "thread_headers", None)
                    if headers:
                        thread_headers[unique_id] = headers

            completed = len(ordered)
            if completed >= required:
                serialised_rows = [self._serialise_dispatch_row(row) for row in ordered.values()]
                logger.info(
                    "SupplierInteractionAgent dispatch verification gate complete for workflow=%s dispatched=%s/%s",
                    workflow_id,
                    completed,
                    required,
                )
                return {
                    "workflow_id": workflow_id,
                    "expected_dispatches": required,
                    "completed_dispatches": completed,
                    "unique_ids": list(ordered.keys()),
                    "dispatch_rows": serialised_rows,
                    "thread_headers": thread_headers,
                    "complete": True,
                    "wait_seconds": time.monotonic() - start,
                    "timed_out": False,
                }

            if completed != last_logged:
                logger.info(
                    "Waiting for dispatch (%s/%s) workflow=%s",
                    completed,
                    required,
                    workflow_id,
                )
                last_logged = completed

            if deadline is not None and time.monotonic() >= deadline:
                logger.info(
                    "Dispatch verification gate timed out for workflow=%s after %.1fs (dispatched=%s/%s)",
                    workflow_id,
                    time.monotonic() - start,
                    completed,
                    required,
                )
                serialised_rows = [self._serialise_dispatch_row(row) for row in ordered.values()]
                return {
                    "workflow_id": workflow_id,
                    "expected_dispatches": required,
                    "completed_dispatches": completed,
                    "unique_ids": list(ordered.keys()),
                    "dispatch_rows": serialised_rows,
                    "thread_headers": thread_headers,
                    "complete": False,
                    "wait_seconds": time.monotonic() - start,
                    "timed_out": True,
                }

            time.sleep(interval)

    def _blocking_response_gate(
        self,
        *,
        workflow_id: str,
        expected_count: int,
        poll_interval: Optional[float],
        expected_unique_ids: Sequence[str],
        timeout_seconds: Optional[float],
    ) -> Dict[str, Any]:
        self._ensure_response_schema()
        expected_ids = [self._coerce_text(uid) for uid in expected_unique_ids if self._coerce_text(uid)]
        expected_set = {uid for uid in expected_ids if uid}
        required = max(expected_count, len(expected_set))
        start = time.monotonic()

        total_timeout = 0.0
        if timeout_seconds is not None:
            try:
                total_timeout = float(timeout_seconds)
            except Exception:
                total_timeout = 0.0
            if total_timeout < 0:
                total_timeout = 0.0

        deadline = start + total_timeout if total_timeout > 0 else None

        if required <= 0:
            return {
                "workflow_id": workflow_id,
                "expected_responses": required,
                "collected_responses": 0,
                "responses": [],
                "unique_ids": [],
                "complete": True,
                "wait_seconds": 0.0,
                "timed_out": False,
            }

        interval = poll_interval or self.RESPONSE_GATE_POLL_SECONDS
        interval = max(1.0, float(interval))
        last_logged = -1

        while True:
            rows = supplier_response_repo.fetch_all(workflow_id=workflow_id)
            ordered: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                unique_id = self._coerce_text(row.get("unique_id"))
                if expected_set and unique_id not in expected_set:
                    continue
                if not unique_id:
                    continue
                ordered.setdefault(unique_id, row)

            completed = len(ordered)
            if completed >= required:
                responses = [self._response_from_row(row) for row in ordered.values()]
                logger.info(
                    "SupplierInteractionAgent response aggregation gate complete for workflow=%s responses=%s/%s",
                    workflow_id,
                    completed,
                    required,
                )
                return {
                    "workflow_id": workflow_id,
                    "expected_responses": required,
                    "collected_responses": completed,
                    "responses": responses,
                    "unique_ids": list(ordered.keys()),
                    "complete": True,
                    "wait_seconds": time.monotonic() - start,
                    "timed_out": False,
                }

            if completed != last_logged:
                logger.info(
                    "Waiting for responses (%s/%s) workflow=%s",
                    completed,
                    required,
                    workflow_id,
                )
                last_logged = completed

            if deadline is not None and time.monotonic() >= deadline:
                logger.info(
                    "Response aggregation gate timed out for workflow=%s after %.1fs (responses=%s/%s)",
                    workflow_id,
                    time.monotonic() - start,
                    completed,
                    required,
                )
                responses = [self._response_from_row(row) for row in ordered.values()]
                return {
                    "workflow_id": workflow_id,
                    "expected_responses": required,
                    "collected_responses": completed,
                    "responses": responses,
                    "unique_ids": list(ordered.keys()),
                    "complete": False,
                    "wait_seconds": time.monotonic() - start,
                    "timed_out": True,
                }

            time.sleep(interval)

    def _run_stateful_gate(
        self,
        context: AgentContext,
        *,
        workflow_id: str,
        expected_count: int,
    ) -> AgentOutput:
        dispatch_poll = self._normalise_poll_interval(
            context.input_data.get("dispatch_poll_interval")
        )
        response_poll = self._normalise_poll_interval(
            context.input_data.get("response_poll_interval")
        )

        dispatch_timeout = self._coerce_optional_positive(
            context.input_data.get("dispatch_timeout")
            or context.input_data.get("dispatch_timeout_seconds")
        )
        response_timeout = self._coerce_optional_positive(
            context.input_data.get("response_timeout")
            or context.input_data.get("response_timeout_seconds")
        )

        expected_unique_ids = self._normalise_unique_ids(
            context.input_data.get("expected_unique_ids")
            or context.input_data.get("unique_ids")
        )

        if expected_unique_ids and expected_count <= 0:
            expected_count = len(expected_unique_ids)

        dispatch_summary = self._blocking_dispatch_gate(
            workflow_id=workflow_id,
            expected_count=expected_count,
            poll_interval=dispatch_poll,
            expected_unique_ids=expected_unique_ids,
            timeout_seconds=dispatch_timeout,
        )

        observed_unique_ids = dispatch_summary.get("unique_ids") or []
        if not dispatch_summary.get("complete"):
            payload = {
                "workflow_id": workflow_id,
                "expected_dispatch_count": expected_count,
                "expected_email_count": expected_count,
                "dispatch_verification": dispatch_summary,
                "response_aggregation": None,
                "supplier_responses": [],
                "supplier_responses_batch": [],
                "supplier_responses_count": 0,
                "batch_ready": False,
                "negotiation_batch": False,
                "all_responses_received": False,
                "expected_unique_ids": observed_unique_ids or expected_unique_ids,
                "timed_out": bool(dispatch_summary.get("timed_out")),
            }

            thread_headers = dispatch_summary.get("thread_headers")
            if thread_headers:
                payload["thread_headers"] = thread_headers

            error_msg = (
                "dispatch gate timed out" if dispatch_summary.get("timed_out") else "dispatch gate incomplete"
            )

            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data=payload,
                    pass_fields=payload,
                    next_agents=[],
                    error=error_msg,
                ),
            )

        response_summary = self._blocking_response_gate(
            workflow_id=workflow_id,
            expected_count=expected_count,
            poll_interval=response_poll,
            expected_unique_ids=observed_unique_ids or expected_unique_ids,
            timeout_seconds=response_timeout,
        )

        if not response_summary.get("complete"):
            responses = response_summary.get("responses") or []
            payload = {
                "workflow_id": workflow_id,
                "expected_dispatch_count": expected_count,
                "expected_email_count": expected_count,
                "dispatch_verification": dispatch_summary,
                "response_aggregation": response_summary,
                "supplier_responses": responses,
                "supplier_responses_batch": responses,
                "supplier_responses_count": len(responses),
                "batch_ready": False,
                "negotiation_batch": False,
                "all_responses_received": False,
                "expected_unique_ids": observed_unique_ids or expected_unique_ids,
                "timed_out": bool(response_summary.get("timed_out")),
            }

            thread_headers = dispatch_summary.get("thread_headers")
            if thread_headers:
                payload["thread_headers"] = thread_headers

            error_msg = (
                "response gate timed out"
                if response_summary.get("timed_out")
                else "response gate incomplete"
            )

            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data=payload,
                    pass_fields=payload,
                    next_agents=[],
                    error=error_msg,
                ),
            )

        responses = response_summary.get("responses", [])
        response_count = len(responses)

        payload = {
            "workflow_id": workflow_id,
            "expected_dispatch_count": expected_count,
            "expected_email_count": expected_count,
            "dispatch_verification": dispatch_summary,
            "response_aggregation": response_summary,
            "supplier_responses": responses,
            "supplier_responses_batch": responses,
            "supplier_responses_count": response_count,
            "batch_ready": True,
            "negotiation_batch": bool(responses),
            "all_responses_received": True,
            "expected_unique_ids": observed_unique_ids or expected_unique_ids,
        }

        thread_headers = dispatch_summary.get("thread_headers")
        if thread_headers:
            payload["thread_headers"] = thread_headers

        next_agents = ["NegotiationAgent"] if responses else []
        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=payload,
                pass_fields=payload,
                next_agents=next_agents,
            ),
        )

    def run(self, context: AgentContext) -> AgentOutput:
        """Process a single supplier email or poll the mailbox."""
        action = context.input_data.get("action")
        expected_dispatch_raw = context.input_data.get("expected_dispatch_count")
        if expected_dispatch_raw is None:
            expected_dispatch_raw = context.input_data.get("expected_email_count")
        gating_actions_blocklist = {"poll", "monitor", "business_monitor", "await_workflow_batch"}

        if expected_dispatch_raw is not None and action not in gating_actions_blocklist:
            workflow_key = self._coerce_text(
                context.input_data.get("workflow_id") or getattr(context, "workflow_id", None)
            )
            if not workflow_key:
                payload = {
                    "workflow_id": None,
                    "expected_dispatch_count": expected_dispatch_raw,
                    "expected_email_count": expected_dispatch_raw,
                    "dispatch_verification": None,
                    "response_aggregation": None,
                    "supplier_responses": [],
                    "supplier_responses_count": 0,
                    "batch_ready": False,
                    "negotiation_batch": False,
                    "all_responses_received": False,
                }
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data=payload,
                        pass_fields=payload,
                        error="workflow_id required for stateful supplier interaction",
                    ),
                )

            try:
                expected_email_count = max(0, int(expected_dispatch_raw))
            except Exception:
                logger.exception(
                    "Invalid expected_dispatch_count value '%s' for workflow=%s",
                    expected_dispatch_raw,
                    workflow_key,
                )
                expected_email_count = 0

            return self._run_stateful_gate(
                context,
                workflow_id=workflow_key,
                expected_count=expected_email_count,
            )

        if action == "poll":
            count = self.poll_mailbox()
            payload = {"polled": count}
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=payload,
                    pass_fields=payload,
                ),
            )
        if action == "monitor":
            interval = int(context.input_data.get("interval", 30))
            duration = int(context.input_data.get("duration", 300))
            count = self.monitor_inbox(interval=interval, duration=duration)
            payload = {"monitored": count}
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=payload,
                    pass_fields=payload,
                ),
            )
        if action == "business_monitor":
            start_hour = int(context.input_data.get("business_start_hour", 9))
            end_hour = int(context.input_data.get("business_end_hour", 17))
            interval = int(context.input_data.get("interval", 3600))
            max_cycles = context.input_data.get("max_cycles")
            try:
                cycle_limit = int(max_cycles) if max_cycles is not None else None
            except Exception:
                cycle_limit = None
            count = self.monitor_business_hours(
                start_hour=start_hour,
                end_hour=end_hour,
                interval=interval,
                max_cycles=cycle_limit,
            )
            payload = {"business_polls": count}
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=payload,
                    pass_fields=payload,
                ),
            )

        if action == "await_workflow_batch":
            workflow_key = self._coerce_text(
                context.input_data.get("workflow_id") or getattr(context, "workflow_id", None)
            )
            if not workflow_key:
                payload = {
                    "workflow_id": None,
                    "batch_ready": False,
                    "expected_responses": 0,
                    "collected_responses": 0,
                    "supplier_responses": [],
                    "unique_ids": [],
                }
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data=payload,
                        pass_fields=payload,
                        error="workflow_id required to await batch responses",
                    ),
                )

            poll_default = getattr(
                self.agent_nick.settings, "email_response_poll_seconds", self.WORKFLOW_POLL_INTERVAL_SECONDS
            )
            timeout_default = getattr(
                self.agent_nick.settings, "email_response_timeout_seconds", 900
            )

            poll_interval = self._coerce_int(
                context.input_data.get("response_poll_interval") or poll_default,
                default=self.WORKFLOW_POLL_INTERVAL_SECONDS,
            )
            timeout = self._coerce_int(
                context.input_data.get("response_timeout") or timeout_default,
                default=900,
            )

            summary = self._await_full_response_batch(
                workflow_key,
                timeout=timeout,
                poll_interval=poll_interval,
            )

            expected_total = summary.get("expected_count", 0)
            collected_total = summary.get("collected_count", 0)
            summary_ready = bool(summary.get("ready"))
            responses = summary.get("responses", []) if summary_ready else []
            batch_ready = summary_ready and (
                not expected_total or collected_total == expected_total
            )

            payload = {
                "workflow_id": workflow_key,
                "batch_ready": batch_ready,
                "expected_responses": expected_total,
                "collected_responses": collected_total,
                "supplier_responses": responses,
                "supplier_responses_count": len(responses),
                "unique_ids": summary.get("unique_ids", []),
                "batch_metadata": {
                    "expected": expected_total,
                    "collected": collected_total,
                    "ready": batch_ready,
                },
                "negotiation_batch": batch_ready,
                "supplier_responses_batch": responses,
            }

            next_agents = ["NegotiationAgent"] if batch_ready and responses else []

            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data=payload,
                    pass_fields=payload,
                    next_agents=next_agents,
                ),
            )

        input_data = dict(context.input_data)
        subject = str(input_data.get("subject") or "")
        message_text = input_data.get("message")
        from_address = input_data.get("from_address")
        message_id = self._coerce_text(input_data.get("message_id"))
        thread_headers_input = (
            input_data.get("thread_headers") if isinstance(input_data.get("thread_headers"), dict) else None
        )
        references: List[str] = []
        if thread_headers_input:
            message_candidate = self._coerce_text(thread_headers_input.get("message_id"))
            if message_candidate:
                message_id = message_candidate
            references = self._normalise_thread_references(
                thread_headers_input.get("references")
            )
        additional_refs = self._normalise_thread_references(
            input_data.get("references") or input_data.get("thread_references")
        )
        references = self._merge_thread_references(references, additional_refs)
        body = message_text if isinstance(message_text, str) else ""
        drafts: List[Dict[str, Any]] = [
            draft for draft in input_data.get("drafts", []) if isinstance(draft, dict)
        ]

        if not body:
            fallback_body = input_data.get("body")
            treat_body_as_message = bool(input_data.get("treat_body_as_message"))
            if isinstance(fallback_body, str) and (treat_body_as_message or not drafts):
                body = fallback_body

        supplier_id = input_data.get("supplier_id")
        rfq_id = input_data.get("rfq_id")
        workflow_id = self._coerce_text(
            input_data.get("workflow_id") or getattr(context, "workflow_id", None)
        )
        unique_id = self._coerce_text(input_data.get("unique_id"))
        draft_match = self._select_draft(drafts, supplier_id=supplier_id, rfq_id=rfq_id)
        if not supplier_id and draft_match:
            supplier_id = draft_match.get("supplier_id")
        if not rfq_id and draft_match:
            rfq_id = draft_match.get("rfq_id")

        if not supplier_id:
            candidates = input_data.get("supplier_candidates", [])
            supplier_id = candidates[0] if candidates else None

        draft_context = self._draft_tracking_context(draft_match)
        if not workflow_id:
            workflow_id = draft_context.get("workflow_id")
        if not unique_id:
            unique_id = draft_context.get("unique_id")
        if not supplier_id:
            supplier_id = draft_context.get("supplier_id") or supplier_id

        await_flag = input_data.get("await_response")
        should_wait = not body and (await_flag is True or (await_flag is None and bool(drafts)))

        precomputed: Optional[Dict[str, Any]] = None
        related_override: Optional[List[Any]] = None
        target_override: Optional[float] = None

        if should_wait:
            if not workflow_id:
                logger.error(
                    "Awaiting supplier response requires workflow_id; subject=%s",
                    subject,
                )
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="workflow_id required to await supplier response",
                    ),
                )

            timeout = self._coerce_int(input_data.get("response_timeout"), default=900)
            default_poll = getattr(
                self.agent_nick.settings, "email_response_poll_seconds", 60
            )
            poll_interval = self._coerce_int(
                input_data.get("response_poll_interval"), default=default_poll
            )
            poll_interval = max(1, poll_interval)
            batch_limit = self._coerce_int(input_data.get("response_batch_limit"), default=5)
            expected_sender = None
            if draft_match:
                expected_sender = (
                    draft_match.get("receiver")
                    or draft_match.get("recipient_email")
                    or draft_match.get("contact_email")
                )

            watch_candidates = [
                draft
                for draft in drafts
                if isinstance(draft, dict) and draft.get("rfq_id") and draft.get("supplier_id")
            ]

            await_all = bool(input_data.get("await_all_responses") and len(watch_candidates) > 1)
            parallel_results: List[Optional[Dict[str, Any]]] = []
            wait_result: Optional[Dict[str, Any]] = None

            if await_all:
                parallel_results = self.wait_for_multiple_responses(
                    watch_candidates,
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                )
                wait_result = self._select_parallel_response(
                    parallel_results,
                    rfq_id=rfq_id,
                    supplier_id=supplier_id,
                )
                for candidate in parallel_results:
                    if not candidate or candidate is wait_result:
                        continue
                    try:
                        self._persist_parallel_result(candidate)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception(
                            "Failed to persist parallel supplier response for rfq_id=%s supplier=%s",
                            candidate.get("rfq_id"),
                            candidate.get("supplier_id"),
                        )
            else:
                draft_action_id = None
                workflow_hint = None
                dispatch_run_id = None
                action_hint = None
                email_action_hint = None
                if watch_candidates:
                    primary_candidate = watch_candidates[0]
                    if isinstance(primary_candidate, dict):
                        context_hint = self._prepare_watch_context(primary_candidate)
                        if context_hint:
                            draft_action_id = context_hint.get("draft_action_id")
                            workflow_hint = context_hint.get("workflow_id")
                            dispatch_run_id = context_hint.get("dispatch_run_id")
                            action_hint = context_hint.get("action_id")
                            email_action_hint = context_hint.get("email_action_id")
                if workflow_hint is None:
                    workflow_hint = getattr(context, "workflow_id", None)
                if not workflow_hint:
                    workflow_hint = workflow_id
                wait_result = self.wait_for_response(
                    timeout=timeout,
                    poll_interval=poll_interval,
                    limit=batch_limit,
                    rfq_id=rfq_id,
                    supplier_id=supplier_id,
                    subject_hint=subject,
                    from_address=expected_sender,
                    draft_action_id=draft_action_id,
                    workflow_id=workflow_hint,
                    dispatch_run_id=dispatch_run_id,
                    action_id=action_hint,
                    email_action_id=email_action_hint,
                    unique_id=unique_id,
                )

            if not wait_result:
                if await_all and parallel_results and any(result is None for result in parallel_results):
                    logger.error(
                        "Supplier responses missing for workflow=%s (rfq_id=%s supplier=%s)",
                        workflow_id,
                        rfq_id,
                        supplier_id,
                    )
                logger.error(
                    "Supplier response not received for workflow %s (supplier=%s) before timeout=%ss",
                    workflow_id,
                    supplier_id,
                    timeout,
                )
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="supplier response not received",
                    ),
                )

            supplier_status = str(wait_result.get("supplier_status") or "").lower()
            if supplier_status and supplier_status != AgentStatus.SUCCESS.value:
                error_detail = wait_result.get("error") or "supplier response processing failed"
                logger.error(
                    "Supplier response for workflow=%s rfq=%s (supplier=%s) returned status=%s; error=%s",
                    wait_result.get("workflow_id") or workflow_id,
                    wait_result.get("rfq_id") or rfq_id,
                    wait_result.get("supplier_id") or supplier_id,
                    supplier_status,
                    error_detail,
                )
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error=error_detail,
                    ),
                )

            subject = str(wait_result.get("subject") or subject)
            supplier_id = wait_result.get("supplier_id") or supplier_id
            rfq_id = wait_result.get("rfq_id") or rfq_id
            workflow_id = self._coerce_text(wait_result.get("workflow_id")) or workflow_id
            unique_id = self._coerce_text(wait_result.get("unique_id")) or unique_id
            wait_message_id = self._coerce_text(wait_result.get("message_id"))
            if wait_message_id:
                message_id = wait_message_id

            wait_thread_headers = (
                wait_result.get("thread_headers")
                if isinstance(wait_result.get("thread_headers"), dict)
                else None
            )
            if wait_thread_headers:
                header_message = self._coerce_text(wait_thread_headers.get("message_id"))
                if header_message:
                    message_id = header_message
                references = self._merge_thread_references(
                    references,
                    self._normalise_thread_references(wait_thread_headers.get("references")),
                )

            references = self._merge_thread_references(
                references,
                self._normalise_thread_references(
                    wait_result.get("references") or wait_result.get("thread_references")
                ),
            )

            supplier_payload = wait_result.get("supplier_output")
            if isinstance(supplier_payload, dict):
                precomputed = {
                    "price": supplier_payload.get("price"),
                    "lead_time": supplier_payload.get("lead_time"),
                    "response_text": supplier_payload.get("response_text") or "",
                }
                related_override = supplier_payload.get("related_documents")
                body = precomputed.get("response_text") or ""
            else:
                body = str(
                    wait_result.get("message")
                    or wait_result.get("body")
                    or wait_result.get("supplier_message")
                    or ""
                )

            target_override = self._coerce_float(wait_result.get("target_price"))

        if not body:
            logger.error(
                "Supplier interaction failed because no message body was available (workflow=%s rfq_id=%s, supplier=%s)",
                workflow_id,
                rfq_id,
                supplier_id,
            )
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="message not provided",
                ),
            )

        if not rfq_id:
            rfq_id = self._extract_rfq_id(subject + " " + body)
        if not draft_match:
            draft_match = self._select_draft(drafts, supplier_id=supplier_id, rfq_id=rfq_id)
            if draft_match and not supplier_id:
                supplier_id = draft_match.get("supplier_id")
            if draft_match and not unique_id:
                unique_candidate = self._draft_tracking_context(draft_match).get("unique_id")
                if unique_candidate:
                    unique_id = unique_candidate
            if draft_match and not workflow_id:
                workflow_candidate = self._draft_tracking_context(draft_match).get("workflow_id")
                if workflow_candidate:
                    workflow_id = workflow_candidate

        parsed = (
            {
                "price": precomputed.get("price"),
                "lead_time": precomputed.get("lead_time"),
                "response_text": precomputed.get("response_text") or body,
            }
            if isinstance(precomputed, dict)
            else self._parse_response(
                body,
                subject=subject,
                rfq_id=rfq_id,
                supplier_id=supplier_id,
                drafts=drafts,
            )
        )

        target = self._coerce_float(input_data.get("target_price"))
        if target_override is not None:
            target = target_override

        if related_override is not None:
            related_docs = related_override
        else:
            context_hits = self.vector_search(parsed.get("response_text") or body, top_k=3)
            related_docs = [h.payload for h in context_hits]

        self._store_response(
            workflow_id,
            supplier_id,
            parsed.get("response_text") or body,
            parsed,
            unique_id=unique_id,
            rfq_id=rfq_id,
            message_id=message_id,
            from_address=from_address,
        )

        price = parsed.get("price")
        if price is not None and target is not None and price > target:
            next_agent = ["NegotiationAgent"]
        else:
            next_agent = ["QuoteEvaluationAgent"]

        payload = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            **parsed,
            "related_documents": related_docs,
        }
        thread_headers_payload = self._build_thread_headers_payload(message_id, references)
        if thread_headers_payload:
            payload["thread_headers"] = thread_headers_payload
        if target is not None:
            payload["target_price"] = target

        return self._with_plan(
            context,
            AgentOutput(
                status=AgentStatus.SUCCESS,
                data=payload,
                pass_fields=payload,
                next_agents=next_agent,
            ),
        )

    def poll_mailbox(self) -> int:
        """Poll inbox for RFQ responses. Returns number of messages processed."""
        host = getattr(self.agent_nick.settings, "imap_host", None)
        user = getattr(self.agent_nick.settings, "imap_user", None)
        password = getattr(self.agent_nick.settings, "imap_password", None)
        if not all([host, user, password]):
            logger.warning("IMAP settings not configured")
            return 0
        processed = 0
        try:
            with imaplib.IMAP4_SSL(host) as imap:
                imap.login(user, password)
                imap.select("INBOX")
                typ, data = imap.search(None, "ALL")
                for num in data[0].split():
                    typ, msg_data = imap.fetch(num, '(RFC822)')
                    if typ != 'OK':
                        continue
                    msg = message_from_bytes(msg_data[0][1])
                    subject = msg.get('subject', '')
                    body = msg.get_payload(decode=True) or b''
                    text = body.decode(errors='ignore')
                    rfq_id = self._extract_rfq_id(subject + " " + text)
                    if rfq_id:
                        parsed_payload = self._parse_response(
                            text,
                            subject=subject,
                            rfq_id=rfq_id,
                            supplier_id=None,
                        )
                        self._store_response(
                            None,
                            None,
                            text,
                            parsed_payload,
                            unique_id=None,
                            rfq_id=rfq_id,
                            message_id=msg.get('message-id'),
                            from_address=msg.get('from'),
                        )
                        processed += 1
        except Exception:  # pragma: no cover - best effort
            logger.exception("mailbox polling failed")
        return processed

    def monitor_inbox(self, interval: int = 30, duration: int = 300) -> int:
        """Continuously poll the inbox for a given duration.

        Args:
            interval: Seconds between polls.
            duration: Total time to monitor in seconds.
        Returns:
            Total number of messages processed.
        """
        end_time = time.time() + duration
        total = 0
        while time.time() < end_time:
            total += self.poll_mailbox()
            time.sleep(interval)
        return total

    def monitor_business_hours(
        self,
        start_hour: int = 9,
        end_hour: int = 17,
        interval: int = 3600,
        max_cycles: Optional[int] = None,
    ) -> int:
        """Monitor the mailbox once per hour during business hours."""

        processed = 0
        cycles = 0
        interval = max(1, interval)
        if end_hour <= start_hour:
            end_hour = start_hour + 8

        while True:
            now = datetime.now()
            current_hour = now.hour
            if start_hour <= current_hour < end_hour:
                processed += self.poll_mailbox()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(interval)
                continue

            if current_hour >= end_hour:
                break

            next_start = now.replace(
                hour=start_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
            if current_hour >= end_hour:
                next_start += timedelta(days=1)
            elif current_hour < start_hour:
                pass
            else:
                next_start = now + timedelta(seconds=interval)
            sleep_seconds = max(0, (next_start - now).total_seconds())
            time.sleep(min(interval, sleep_seconds))

        return processed


    def wait_for_response(
        self,
        *,
        watcher=None,
        timeout: int = 300,
        poll_interval: Optional[int] = None,
        limit: int = 1,
        rfq_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        subject_hint: Optional[str] = None,
        from_address: Optional[str] = None,
        max_attempts: Optional[int] = None,
        enable_negotiation: bool = True,
        draft_action_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        dispatch_run_id: Optional[str] = None,
        action_id: Optional[str] = None,
        email_action_id: Optional[str] = None,
        suppliers: Optional[Sequence[str]] = None,
        expected_replies: Optional[int] = None,
        unique_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Await supplier replies using the workflow-aware IMAP watcher."""

        unique_key = self._coerce_text(unique_id)
        workflow_key = self._coerce_text(workflow_id)

        canonical_workflow = None
        if unique_key:
            try:
                canonical_workflow = workflow_email_tracking_repo.lookup_workflow_for_unique(
                    unique_id=unique_key
                )
            except Exception:  # pragma: no cover - defensive alignment
                logger.exception(
                    "Failed to resolve canonical workflow for unique_id=%s", unique_key
                )

        if canonical_workflow:
            canonical_key = self._coerce_text(canonical_workflow)
            if canonical_key:
                if workflow_key and canonical_key != workflow_key:
                    logger.info(
                        "Realigning wait_for_response workflow_id from %s to %s using dispatch mapping",
                        workflow_key,
                        canonical_key,
                    )
                workflow_key = canonical_key

        if unique_key:
            existing_workflow = None
            try:
                existing_workflow = supplier_response_repo.lookup_workflow_for_unique(
                    unique_id=unique_key
                )
            except Exception:  # pragma: no cover - defensive alignment
                logger.exception(
                    "Failed to inspect stored workflow for unique_id=%s", unique_key
                )
            if existing_workflow:
                existing_key = self._coerce_text(existing_workflow)
                if existing_key:
                    if workflow_key and existing_key != workflow_key:
                        logger.info(
                            "Using previously stored workflow_id=%s for unique_id=%s instead of %s",
                            existing_key,
                            unique_key,
                            workflow_key,
                        )
                    workflow_key = existing_key

        if not workflow_key:
            logger.error(
                "wait_for_response requires workflow_id; supplier=%s",
                supplier_id,
            )
            return None

        target_supplier = self._coerce_text(supplier_id)
        supplier_filter: Optional[Set[str]] = None
        if target_supplier:
            supplier_filter = {target_supplier}

        unique_filter: Optional[Set[str]] = {unique_key} if unique_key else None

        dispatch_unique_ids: List[Optional[str]] = []
        if unique_filter:
            dispatch_unique_ids.extend(unique_filter)
        elif draft_match:
            dispatch_unique_ids.append(draft_context.get("unique_id"))
        elif watch_candidates:
            for candidate in watch_candidates:
                if isinstance(candidate, dict):
                    dispatch_unique_ids.append(candidate.get("unique_id"))

        expected_unique_total = len(
            [uid for uid in dispatch_unique_ids if self._coerce_text(uid)]
        )
        if expected_unique_total == 0 and unique_filter:
            expected_unique_total = len(unique_filter)
        if expected_unique_total == 0:
            expected_unique_total = 1

        dispatch_summary = self._await_dispatch_ready(
            workflow_id=workflow_key,
            unique_ids=dispatch_unique_ids,
            timeout=timeout,
            poll_interval=poll_interval,
            expected_total=expected_unique_total,
        )

        responses = self._await_supplier_response_rows(
            workflow_key,
            supplier_filter=supplier_filter,
            unique_filter=unique_filter,
            timeout=timeout,
            poll_interval=poll_interval,
            dispatch_summary=dispatch_summary,
        )

        if not responses:
            logger.info(
                "No supplier responses stored for workflow=%s",
                workflow_key,
            )
            return None

        return responses[0] if responses else None


    def wait_for_multiple_responses(
        self,
        drafts: List[Dict[str, Any]],
        *,
        timeout: int = 300,
        poll_interval: Optional[int] = None,
        limit: int = 1,
        enable_negotiation: bool = True,
    ) -> List[Optional[Dict]]:
        """Collect all supplier responses for the provided drafts."""

        if not drafts:
            return []

        contexts = [self._draft_tracking_context(draft) for draft in drafts]
        workflow_ids = {
            self._coerce_text(ctx.get("workflow_id"))
            for ctx in contexts
            if self._coerce_text(ctx.get("workflow_id"))
        }
        unique_ids: List[str] = []
        unique_id_set: Set[str] = set()
        for ctx in contexts:
            unique_value = self._coerce_text(ctx.get("unique_id"))
            if unique_value and unique_value not in unique_id_set:
                unique_id_set.add(unique_value)
                unique_ids.append(unique_value)

        workflow_key: Optional[str] = None
        canonical_candidates: Set[str] = set()
        for unique_value in unique_ids:
            try:
                candidate = workflow_email_tracking_repo.lookup_workflow_for_unique(
                    unique_id=unique_value
                )
            except Exception:  # pragma: no cover - defensive alignment
                logger.exception(
                    "Failed to resolve canonical workflow for unique_id=%s during aggregation",
                    unique_value,
                )
                continue
            coerced = self._coerce_text(candidate)
            if coerced:
                canonical_candidates.add(coerced)

        lowered_workflow_map: Dict[str, str] = {
            value.lower(): value for value in workflow_ids
        }

        if canonical_candidates:
            lowered_canonical = {value.lower(): value for value in canonical_candidates}
            overlap = lowered_canonical.keys() & lowered_workflow_map.keys()
            if overlap:
                choice_key = next(iter(overlap))
                canonical_choice = lowered_canonical[choice_key]
                if workflow_key and workflow_key != canonical_choice:
                    logger.info(
                        "Realigning aggregated workflow_id from %s to %s using dispatch mapping",
                        workflow_key,
                        canonical_choice,
                    )
                workflow_key = canonical_choice
            elif len(canonical_candidates) == 1:
                workflow_key = canonical_candidates.pop()
            elif workflow_ids:
                logger.error(
                    "Conflicting workflow_ids %s for unique_ids=%s",
                    sorted(canonical_candidates | workflow_ids),
                    sorted(unique_id_set),
                )
                return [None] * len(drafts)

        if workflow_key is None and workflow_ids:
            if len(lowered_workflow_map) == 1:
                workflow_key = next(iter(lowered_workflow_map.values()))
            else:
                logger.error(
                    "Cannot aggregate responses without a single workflow_id; contexts=%s",
                    contexts,
                )
                return [None] * len(drafts)

        if not workflow_key:
            logger.error(
                "Cannot aggregate responses without a workflow_id; contexts=%s unique_ids=%s",
                contexts,
                sorted(unique_id_set),
            )
            return [None] * len(drafts)

        unique_filter = set(unique_id_set) if unique_id_set else None

        batch_poll_interval = (
            poll_interval
            if poll_interval is not None
            else self.WORKFLOW_POLL_INTERVAL_SECONDS
        )

        expected_dispatch_total = max(len(drafts), len(unique_ids))

        dispatch_summary = self._await_dispatch_ready(
            workflow_id=workflow_key,
            unique_ids=list(unique_ids),
            timeout=timeout,
            poll_interval=batch_poll_interval,
            expected_total=expected_dispatch_total,
        )

        summary_unique_ids = [
            self._coerce_text(value)
            for value in (dispatch_summary or {}).get("unique_ids", [])
            if self._coerce_text(value)
        ]
        if summary_unique_ids:
            unique_ids = summary_unique_ids
            unique_id_set = set(summary_unique_ids)

        if unique_ids and len(unique_ids) >= len(drafts):
            unique_filter = set(unique_id_set)
        elif unique_filter is not None and len(unique_filter) < len(drafts):
            unique_filter = None

        responses = self._await_supplier_response_rows(
            workflow_key,
            unique_filter=unique_filter,
            timeout=timeout,
            poll_interval=batch_poll_interval,
            dispatch_summary=dispatch_summary,
        )

        if not responses:
            logger.info(
                "Aggregated watcher not ready for workflow=%s (no responses stored)",
                workflow_key,
            )
            return [None] * len(drafts)

        response_map = {
            row.get("unique_id"): row
            for row in responses
            if row.get("unique_id")
        }

        results: List[Optional[Dict]] = []
        for draft, ctx in zip(drafts, contexts):
            response = response_map.get(ctx.get("unique_id"))
            if response is None and ctx.get("supplier_id"):
                for row in responses:
                    if self._coerce_text(row.get("supplier_id")) == ctx.get("supplier_id"):
                        response = row
                        break
            results.append(response)

        return results

    def _prepare_watch_context(self, draft: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(draft, dict):
            return None

        rfq_id = draft.get("rfq_id")
        supplier_id = draft.get("supplier_id")
        if not rfq_id or not supplier_id:
            return None

        recipient_hint = draft.get("receiver") or draft.get("recipient_email")
        if recipient_hint is None:
            recipients_field = draft.get("recipients")
            if isinstance(recipients_field, (list, tuple)) and recipients_field:
                recipient_hint = recipients_field[0]
            elif isinstance(recipients_field, str):
                recipient_hint = recipients_field

        metadata = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else None
        tracking_context = self._draft_tracking_context(draft)
        unique_hint = tracking_context.get("unique_id")

        def _from_sources(keys: Tuple[str, ...]) -> Optional[str]:
            for container in (draft, metadata):
                if not isinstance(container, dict):
                    continue
                for key in keys:
                    value = container.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            return None

        action_id = _from_sources(("action_id",))
        email_action_id = _from_sources(("email_action_id",))
        draft_action_id = _from_sources(("draft_action_id",))
        if draft_action_id is None:
            draft_action_id = action_id or email_action_id

        workflow_candidates: List[str] = []
        for container in (draft, metadata):
            if not isinstance(container, dict):
                continue
            for key in (
                "workflow_id",
                "workflowId",
                "process_workflow_id",
                "workflow",
            ):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    workflow_candidates.append(value.strip())
        workflow_hint = workflow_candidates[0] if workflow_candidates else None

        if not workflow_hint:
            workflow_section = draft.get("workflow")
            if isinstance(workflow_section, dict):
                for key in ("workflow_id", "id", "workflowId"):
                    value = workflow_section.get(key)
                    if isinstance(value, str) and value.strip():
                        workflow_hint = value.strip()
                        break

        if not workflow_hint and isinstance(metadata, dict):
            context_meta = metadata.get("context")
            if isinstance(context_meta, dict):
                for key in ("workflow_id", "workflowId", "process_workflow_id"):
                    value = context_meta.get(key)
                    if isinstance(value, str) and value.strip():
                        workflow_hint = value.strip()
                        break

        dispatch_run_id = self._extract_dispatch_run_id(draft)

        subject_hint = draft.get("subject")
        subject_text = subject_hint if isinstance(subject_hint, str) else ""

        return {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "subject_hint": subject_text,
            "from_address": recipient_hint,
            "draft_action_id": draft_action_id,
            "action_id": action_id,
            "email_action_id": email_action_id,
            "workflow_id": workflow_hint,
            "dispatch_run_id": dispatch_run_id,
            "unique_id": unique_hint,
            "supplier_normalised": self._normalise_identifier(supplier_id),
            "rfq_normalised": self._normalise_identifier(rfq_id),
            "dispatch_normalised": self._normalise_identifier(dispatch_run_id),
            "subject_normalised": subject_text.strip().lower() if subject_text else "",
            "from_normalised": self._normalise_identifier(recipient_hint),
        }

    def _await_outstanding_dispatch_responses(
        self,
        drafts: List[Dict[str, Any]],
        results: List[Optional[Dict[str, Any]]],
        *,
        deadline: float,
        poll_interval: Optional[int],
        limit: int,
        enable_negotiation: bool,
    ) -> None:
        """Continue polling until dispatch-tracked replies have arrived."""

        if not drafts or not results:
            return

        poll_setting = getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)
        interval = poll_interval
        if interval is None:
            interval = poll_setting
        else:
            try:
                interval = max(1, int(interval))
            except Exception:
                interval = poll_setting

        while time.time() < deadline:
            outstanding = [idx for idx, value in enumerate(results) if value is None]
            if not outstanding:
                break

            awaiting = False
            for idx in outstanding:
                rfq_id = drafts[idx].get("rfq_id")
                if not rfq_id:
                    continue
                pending_total = self._pending_dispatch_count(rfq_id)
                if pending_total > 0:
                    awaiting = True
            if not awaiting:
                break

            remaining = deadline - time.time()
            if remaining <= 0:
                break

            per_attempt = max(1, min(interval, int(remaining)))

            for idx in list(outstanding):
                if results[idx] is not None or time.time() >= deadline:
                    continue
                context = self._prepare_watch_context(drafts[idx])
                if not context:
                    continue
                result = self.wait_for_response(
                    timeout=per_attempt,
                    poll_interval=poll_interval,
                    limit=limit,
                    rfq_id=context["rfq_id"],
                    supplier_id=context["supplier_id"],
                    subject_hint=context["subject_hint"],
                    from_address=context["from_address"],
                    enable_negotiation=enable_negotiation,
                    draft_action_id=context["draft_action_id"],
                    dispatch_run_id=context["dispatch_run_id"],
                    workflow_id=context["workflow_id"],
                    action_id=context["action_id"],
                    email_action_id=context["email_action_id"],
                    unique_id=context.get("unique_id"),
                )
                if result is not None:
                    results[idx] = result

            if any(results[idx] is None for idx in outstanding) and time.time() < deadline:
                sleep_window = min(interval, max(0, deadline - time.time()))
                if sleep_window > 0:
                    time.sleep(sleep_window)

    def _pending_dispatch_count(self, rfq_id: Optional[str]) -> int:
        if not rfq_id:
            return 0

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return 0

        try:
            with get_conn() as conn:
                return pending_dispatch_count(conn, rfq_id)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "Failed to fetch pending dispatch count for rfq_id=%s", rfq_id, exc_info=True
            )
            return 0

    def _get_negotiation_agent(self) -> Optional["NegotiationAgent"]:
        if self._negotiation_agent is not None:
            return self._negotiation_agent

        registry = getattr(self.agent_nick, "agents", None)
        candidate = None
        if isinstance(registry, dict):
            candidate = registry.get("negotiation") or registry.get("NegotiationAgent")

        if candidate is None:
            try:
                from agents.negotiation_agent import NegotiationAgent

                candidate = NegotiationAgent(self.agent_nick)
            except Exception:  # pragma: no cover - defensive fallback
                logger.debug("Negotiation agent initialisation failed for email watcher")
                candidate = None

        self._negotiation_agent = candidate
        return self._negotiation_agent

    def _select_parallel_response(
        self,
        results: List[Optional[Dict[str, Any]]],
        *,
        rfq_id: Optional[str],
        supplier_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        def _matches(target: Optional[str], candidate: Optional[str]) -> bool:
            return bool(
                target
                and candidate
                and self._normalise_identifier(target) == self._normalise_identifier(candidate)
            )

        for result in results:
            if not result:
                continue
            if supplier_id and _matches(supplier_id, result.get("supplier_id")):
                if not rfq_id or _matches(rfq_id, result.get("rfq_id")):
                    return result
        for result in results:
            if not result:
                continue
            if rfq_id and _matches(rfq_id, result.get("rfq_id")):
                return result
        for result in results:
            if not result:
                continue
            if supplier_id and _matches(supplier_id, result.get("supplier_id")):
                return result
        for result in results:
            if result:
                return result
        return None

    def _persist_parallel_result(self, result: Dict[str, Any]) -> None:
        rfq_id = result.get("rfq_id")
        workflow_id = result.get("workflow_id")
        unique_id = result.get("unique_id")
        supplier_id = result.get("supplier_id")
        if not workflow_id:
            return

        subject = result.get("subject")
        response_text = (
            result.get("supplier_message")
            or result.get("message")
            or result.get("body")
            or ""
        )
        response_text = str(response_text)

        supplier_payload = result.get("supplier_output")
        if isinstance(supplier_payload, dict):
            parsed = {
                "price": supplier_payload.get("price"),
                "lead_time": supplier_payload.get("lead_time"),
                "response_text": supplier_payload.get("response_text") or response_text,
            }
            message_body = parsed.get("response_text") or response_text
        else:
            message_body = response_text
            parsed = self._parse_response(
                message_body,
                subject=subject,
                rfq_id=rfq_id,
                supplier_id=supplier_id,
            )

        self._store_response(
            workflow_id,
            supplier_id,
            message_body,
            parsed,
            unique_id=unique_id,
            message_id=result.get("message_id"),
            rfq_id=rfq_id,
            from_address=result.get("from_address"),
        )

    def _resolve_supplier_id(
        self,
        rfq_id: Optional[str],
        supplier_id: Optional[str],
        *,
        message_id: Optional[str] = None,
        from_address: Optional[str] = None,
    ) -> Optional[str]:
        if supplier_id:
            try:
                candidate = str(supplier_id).strip()
            except Exception:
                candidate = None
            if candidate:
                return candidate
        if not rfq_id:
            return None

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None

        lookup_queries: List[Tuple[str, str, Tuple]] = []
        if message_id:
            lookup_queries.append(
                (
                    "email_thread_map",
                    """
                    SELECT supplier_id
                    FROM proc.email_thread_map
                    WHERE message_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (message_id,),
                )
            )

        lookup_queries.extend(
            [
                (
                    "negotiation_session_state",
                    """
                    SELECT supplier_id
                    FROM proc.negotiation_session_state
                    WHERE rfq_id = %s
                    ORDER BY updated_on DESC
                    LIMIT 1
                    """,
                    (rfq_id,),
                ),
                (
                    "negotiation_sessions",
                    """
                    SELECT supplier_id
                    FROM proc.negotiation_sessions
                    WHERE rfq_id = %s
                    ORDER BY round DESC, created_on DESC
                    LIMIT 1
                    """,
                    (rfq_id,),
                ),
            ]
        )

        if from_address:
            lookup_queries.append(
                (
                    "draft_rfq_emails_recipient",
                    """
                    SELECT supplier_id
                    FROM proc.draft_rfq_emails
                    WHERE rfq_id = %s
                      AND (
                          recipient_email = %s
                          OR LOWER(recipient_email) = LOWER(%s)
                      )
                    ORDER BY updated_on DESC, created_on DESC
                    LIMIT 1
                    """,
                    (rfq_id, from_address, from_address),
                )
            )

        lookup_queries.extend(
            [
                (
                    "draft_rfq_emails",
                    """
                    SELECT supplier_id
                    FROM proc.draft_rfq_emails
                    WHERE rfq_id = %s
                    ORDER BY updated_on DESC, created_on DESC
                    LIMIT 1
                    """,
                    (rfq_id,),
                ),
                (
                    "rfq_targets",
                    """
                    SELECT supplier_id
                    FROM proc.rfq_targets
                    WHERE rfq_id = %s
                    ORDER BY updated_on DESC
                    LIMIT 1
                    """,
                    (rfq_id,),
                ),
            ]
        )

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    for label, statement, params in lookup_queries:
                        try:
                            cur.execute(statement, params)
                        except Exception:
                            if hasattr(conn, "rollback"):
                                try:
                                    conn.rollback()
                                except Exception:  # pragma: no cover - defensive
                                    pass
                            logger.debug(
                                "Supplier lookup %s failed for rfq_id=%s", label, rfq_id, exc_info=True
                            )
                            continue
                        row = cur.fetchone()
                        if not row:
                            continue
                        value = row[0]
                        if value in (None, ""):
                            continue
                        resolved = str(value).strip()
                        if not resolved:
                            continue
                        logger.debug(
                            "Resolved supplier_id=%s for rfq_id=%s via %s lookup",
                            resolved,
                            rfq_id,
                            label,
                        )
                        return resolved
        except Exception:
            logger.exception("Failed to resolve supplier_id for rfq_id=%s", rfq_id)
        return None

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return parsed if parsed > 0 else default

    @staticmethod
    def _coerce_optional_positive(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_decimal(value: Any) -> Optional[Decimal]:
        if value in (None, ""):
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    @staticmethod
    def _normalise_identifier(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        lowered = text.lower()
        return lowered or None

    @staticmethod
    def _extract_dispatch_run_id(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        def _clean(value: Optional[Any]) -> Optional[str]:
            if value is None:
                return None
            try:
                text = str(value).strip()
            except Exception:
                return None
            return text or None

        for key in ("dispatch_run_id", "run_id"):
            candidate = _clean(payload.get(key))
            if candidate:
                return candidate

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
        if metadata:
            for key in ("dispatch_run_id", "run_id"):
                candidate = _clean(metadata.get(key))
                if candidate:
                    return candidate

        dispatch_meta = (
            payload.get("dispatch_metadata")
            if isinstance(payload.get("dispatch_metadata"), dict)
            else None
        )
        if dispatch_meta:
            for key in ("run_id", "dispatch_run_id", "dispatch_token"):
                candidate = _clean(dispatch_meta.get(key))
                if candidate:
                    return candidate

        return None

    @staticmethod
    def _build_workflow_filters(
        *,
        workflow_id: Optional[str],
        action_id: Optional[str] = None,
        draft_action_id: Optional[str] = None,
        email_action_id: Optional[str] = None,
    ) -> Dict[str, object]:
        filters: Dict[str, object] = {}

        def _add(key: str, value: Optional[str]) -> None:
            if value is None:
                return
            try:
                text = str(value).strip()
            except Exception:
                return
            if not text:
                return
            filters[key] = text

        _add("workflow_id", workflow_id)
        _add("action_id", action_id)
        _add("draft_action_id", draft_action_id)
        _add("email_action_id", email_action_id)

        return filters

    def _select_draft(
        self,
        drafts: List[Dict[str, Any]],
        *,
        supplier_id: Optional[Any] = None,
        rfq_id: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        if not drafts:
            return None

        target_supplier = self._normalise_identifier(supplier_id)
        target_rfq = self._normalise_identifier(rfq_id)

        def _matches(draft: Dict[str, Any]) -> bool:
            draft_supplier = self._normalise_identifier(draft.get("supplier_id"))
            draft_rfq = self._normalise_identifier(draft.get("rfq_id"))
            supplier_ok = not target_supplier or draft_supplier == target_supplier
            rfq_ok = not target_rfq or draft_rfq == target_rfq
            return supplier_ok and rfq_ok

        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            if _matches(draft):
                return draft

        if target_supplier:
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                if self._normalise_identifier(draft.get("supplier_id")) == target_supplier:
                    return draft

        if target_rfq:
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                if self._normalise_identifier(draft.get("rfq_id")) == target_rfq:
                    return draft

        for draft in drafts:
            if isinstance(draft, dict):
                return draft
        return None

    def _extract_rfq_id(self, text: str) -> Optional[str]:
        match = self.RFQ_PATTERN.search(text)
        return match.group(0) if match else None

    def _analyze_response_with_llm(
        self,
        *,
        subject: Optional[str],
        body: str,
        rfq_id: Optional[str],
        supplier_id: Optional[str],
        drafts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        prompt_payload = {
            "subject": subject or "",
            "rfq_id": rfq_id or "",
            "supplier_id": supplier_id or "",
            "message": body,
            "drafts": [
                {
                    "rfq_id": draft.get("rfq_id"),
                    "supplier_id": draft.get("supplier_id"),
                    "subject": draft.get("subject") or draft.get("title"),
                }
                for draft in (drafts or [])
                if isinstance(draft, dict)
            ],
        }

        try:
            model_name = getattr(
                self.settings,
                "supplier_interaction_model",
                getattr(self.settings, "extraction_model", None),
            )
            response = self.call_ollama(
                model=model_name,
                format="json",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyse supplier RFQ responses."
                            " Return strict JSON with keys:"
                            " price (number or null), lead_time_days"
                            " (number or null), summary (string)."
                            " Capture the supplier's intent and any"
                            " contextual signals in summary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt_payload, ensure_ascii=False),
                    },
                ],
                options={"temperature": 0.2},
            )
        except Exception:
            logger.debug("LLM response extraction failed", exc_info=True)
            return None

        content: Optional[str] = None
        if isinstance(response, dict):
            if isinstance(response.get("response"), str):
                content = response.get("response")
            elif isinstance(response.get("message"), dict):
                message_payload = response.get("message")
                if isinstance(message_payload, dict):
                    maybe_content = message_payload.get("content")
                    if isinstance(maybe_content, str):
                        content = maybe_content

        if not content:
            return None

        parsed: Optional[Dict[str, Any]]
        try:
            parsed = json.loads(content)
        except Exception:
            logger.debug("LLM response was not valid JSON: %s", content)
            return None

        return parsed if isinstance(parsed, dict) else None

    def _parse_response(
        self,
        text: str,
        *,
        subject: Optional[str] = None,
        rfq_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        drafts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        price_match = re.search(r"(\d+[.,]?\d*)", text)
        lead_match = re.search(r"(\d+)\s*days", text, re.IGNORECASE)
        price = float(price_match.group(1).replace(',', '')) if price_match else None
        lead_time = lead_match.group(1) if lead_match else None

        llm_payload = self._analyze_response_with_llm(
            subject=subject,
            body=text,
            rfq_id=rfq_id,
            supplier_id=supplier_id,
            drafts=drafts,
        )

        if isinstance(llm_payload, dict):
            llm_price = self._coerce_float(llm_payload.get("price"))
            if llm_price is None:
                llm_price = self._coerce_float(llm_payload.get("price_gbp"))
            if llm_price is not None:
                price = llm_price

            lead_candidate = llm_payload.get("lead_time_days") or llm_payload.get(
                "lead_time"
            )
            if lead_candidate is not None:
                if isinstance(lead_candidate, (int, float)):
                    lead_time = str(int(lead_candidate))
                else:
                    lead_text = str(lead_candidate)
                    lead_digits = re.search(r"(\d+)", lead_text)
                    if lead_digits:
                        lead_time = lead_digits.group(1)

            summary_text = llm_payload.get("summary") or llm_payload.get("context")
            if isinstance(summary_text, str) and summary_text.strip():
                summary = summary_text.strip()
            else:
                summary = None
        else:
            summary = None

        payload = {
            "price": price,
            "lead_time": lead_time,
            "response_text": text,
        }
        if summary:
            payload["context_summary"] = summary
        return payload

    def _store_response(
        self,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        text: str,
        parsed: Dict,
        *,
        unique_id: Optional[str] = None,
        rfq_id: Optional[str] = None,
        message_id: Optional[str] = None,
        from_address: Optional[str] = None,
        received_at: Optional[datetime] = None,
    ) -> None:
        unique_key = self._coerce_text(unique_id) or self._coerce_text(message_id)
        if not unique_key:
            logger.error(
                "Failed to store supplier response because unique_id could not be resolved (workflow=%s)",
                workflow_id,
            )
            return

        workflow_key = self._coerce_text(workflow_id)

        canonical_workflow = None
        try:
            canonical_workflow = workflow_email_tracking_repo.lookup_workflow_for_unique(
                unique_id=unique_key
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "Failed to resolve canonical workflow for unique_id=%s from dispatch tracking",
                unique_key,
            )

        if canonical_workflow:
            canonical_coerced = self._coerce_text(canonical_workflow)
            if canonical_coerced:
                if workflow_key and canonical_coerced != workflow_key:
                    logger.warning(
                        "Workflow mismatch detected for unique_id=%s; using dispatch workflow_id=%s instead of %s",
                        unique_key,
                        canonical_coerced,
                        workflow_key,
                    )
                workflow_key = canonical_coerced

        existing_workflow = None
        try:
            existing_workflow = supplier_response_repo.lookup_workflow_for_unique(unique_id=unique_key)
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "Failed to inspect stored workflow for unique_id=%s",
                unique_key,
            )

        if existing_workflow:
            existing_coerced = self._coerce_text(existing_workflow)
            if existing_coerced:
                if workflow_key and existing_coerced != workflow_key:
                    logger.warning(
                        "Workflow mismatch detected for unique_id=%s; aligning to stored workflow_id=%s instead of %s",
                        unique_key,
                        existing_coerced,
                        workflow_key,
                    )
                workflow_key = existing_coerced

        if not workflow_key:
            logger.debug(
                "Skipping supplier response persistence because workflow_id was not provided",
            )
            return

        unique_candidates: Set[str] = {unique_key}
        try:
            workflow_unique_ids = workflow_email_tracking_repo.load_workflow_unique_ids(
                workflow_id=workflow_key
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "Failed to load workflow dispatch identifiers for workflow=%s", workflow_key
            )
            workflow_unique_ids = []

        for candidate in workflow_unique_ids:
            coerced = self._coerce_text(candidate)
            if coerced:
                unique_candidates.add(coerced)

        try:
            supplier_response_repo.align_workflow_assignments(
                workflow_id=workflow_key, unique_ids=unique_candidates
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "Failed to realign supplier responses for workflow=%s", workflow_key
            )

        resolved_supplier = supplier_id
        if not resolved_supplier and rfq_id:
            resolved_supplier = self._resolve_supplier_id(
                rfq_id,
                supplier_id,
                message_id=message_id,
                from_address=from_address,
            )
        if not resolved_supplier:
            logger.error(
                "Failed to store supplier response because supplier_id could not be resolved (workflow=%s, rfq_id=%s)",
                workflow_key,
                rfq_id,
            )
            return

        supplier_email = None
        if isinstance(from_address, str) and from_address.strip():
            supplier_email = from_address.strip()

        price_value = self._coerce_decimal(parsed.get("price"))
        lead_value = None
        if parsed.get("lead_time") is not None:
            try:
                lead_value = int(str(parsed.get("lead_time")))
            except Exception:
                lead_value = None

        dispatch_row = None
        try:
            dispatch_row = workflow_email_tracking_repo.lookup_dispatch_row(
                workflow_id=workflow_key, unique_id=unique_key
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "Failed to load dispatch details for workflow=%s unique_id=%s",
                workflow_key,
                unique_key,
            )

        received_time = received_at or datetime.now(timezone.utc)
        response_time_value: Optional[Decimal] = None
        if dispatch_row and dispatch_row.dispatched_at and received_time:
            try:
                delta_seconds = (received_time - dispatch_row.dispatched_at).total_seconds()
                if delta_seconds >= 0:
                    response_time_value = Decimal(str(delta_seconds))
            except Exception:
                response_time_value = None

        original_message_id = dispatch_row.message_id if dispatch_row else None
        original_subject = dispatch_row.subject if dispatch_row else None

        try:
            supplier_response_repo.init_schema()
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to initialise supplier_response schema")
            return

        try:
            supplier_response_repo.insert_response(
                SupplierResponseRow(
                    workflow_id=workflow_key,
                    unique_id=unique_key,
                    supplier_id=resolved_supplier,
                    supplier_email=supplier_email,
                    response_text=text,
                    received_time=received_time,
                    response_time=response_time_value,
                    response_message_id=message_id,
                    response_subject=None,
                    response_from=from_address,
                    original_message_id=original_message_id,
                    original_subject=original_subject,
                    price=price_value,
                    lead_time=lead_value,
                    processed=False,
                )
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "Failed to persist supplier response for workflow=%s unique_id=%s",
                workflow_key,
                unique_key,
            )
