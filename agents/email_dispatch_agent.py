"""Agent responsible for dispatching approved supplier emails."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_dispatch_service import EmailDispatchService
from services.event_bus import get_event_bus
from services.supplier_response_coordinator import get_supplier_response_coordinator
from repositories import workflow_email_tracking_repo


logger = logging.getLogger(__name__)


class EmailDispatchAgent(BaseAgent):
    """Send supplier drafts via SES and register dispatch metadata."""

    AGENTIC_PLAN_STEPS = (
        "Load approved supplier drafts awaiting dispatch.",
        "Send each draft via the configured SMTP client and persist metadata.",
        "Signal downstream watchers once dispatch for the round completes.",
    )

    def __init__(self, agent_nick) -> None:
        super().__init__(agent_nick)
        self.dispatch_service = EmailDispatchService(agent_nick)
        self.event_bus = get_event_bus()
        self._response_coordinator = get_supplier_response_coordinator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_text(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            text = str(value)
        except Exception:
            return None
        trimmed = text.strip()
        return trimmed or None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    def _already_dispatched(
        self,
        workflow_id: Optional[str],
        unique_id: Optional[str],
        draft: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        message_id = self._coerce_text(draft.get("message_id"))
        dispatched = self._truthy(draft.get("sent_status"))
        if message_id and dispatched:
            return {
                "unique_id": unique_id,
                "supplier_id": self._coerce_text(draft.get("supplier_id")),
                "message_id": message_id,
                "dispatched_at": draft.get("dispatched_at") or draft.get("sent_on"),
                "thread_headers": draft.get("thread_headers"),
                "status": "skipped",
            }

        if not workflow_id or not unique_id:
            return None

        try:
            existing = workflow_email_tracking_repo.lookup_dispatch_row(
                workflow_id=workflow_id,
                unique_id=unique_id,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to lookup dispatch state for workflow=%s unique_id=%s",
                workflow_id,
                unique_id,
            )
            existing = None

        if existing and existing.message_id:
            return {
                "unique_id": unique_id,
                "supplier_id": existing.supplier_id,
                "message_id": existing.message_id,
                "dispatched_at": existing.dispatched_at.isoformat()
                if existing.dispatched_at
                else None,
                "thread_headers": existing.thread_headers,
                "status": "skipped",
            }

        return None

    def _build_dispatch_context(
        self,
        draft: Dict[str, Any],
        workflow_id: Optional[str],
        round_number: Optional[int],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if workflow_id:
            context["workflow_id"] = workflow_id
        unique_id = self._coerce_text(draft.get("unique_id"))
        if unique_id:
            context["unique_id"] = unique_id
        if round_number is not None:
            context["round"] = round_number
            context["round_number"] = round_number
        dispatch_key = draft.get("dispatch_key") or draft.get("action_id")
        if dispatch_key:
            context["dispatch_key"] = dispatch_key
        if draft.get("thread_headers"):
            context["thread_headers"] = draft.get("thread_headers")
        return context

    def _send_draft(
        self,
        draft: Dict[str, Any],
        workflow_id: Optional[str],
        round_number: Optional[int],
    ) -> Dict[str, Any]:
        unique_id = self._coerce_text(draft.get("unique_id"))
        supplier_id = self._coerce_text(draft.get("supplier_id"))
        if not unique_id and not self._coerce_text(draft.get("rfq_id")):
            failure = {
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "status": "failed",
                "error": "missing_unique_identifier",
            }
            logger.error(json.dumps({"event": "email_dispatch_failed", **failure}))
            return failure
        recipients = draft.get("recipients")
        sender = draft.get("sender")
        subject = draft.get("subject")
        body = draft.get("body")
        attachments = draft.get("attachments") if isinstance(draft.get("attachments"), list) else None

        dispatch_context = self._build_dispatch_context(draft, workflow_id, round_number)

        identifier = unique_id or self._coerce_text(draft.get("rfq_id")) or ""
        result = self.dispatch_service.send_draft(
            identifier,
            recipients=recipients,
            sender=sender,
            subject_override=subject,
            body_override=body,
            attachments=attachments,
            workflow_dispatch_context=dispatch_context,
        )

        dispatched_at_dt = datetime.now(timezone.utc)
        dispatched_at = dispatched_at_dt.isoformat()
        sent = bool(result.get("sent"))
        message_id = self._coerce_text(result.get("message_id"))
        updated_draft = result.get("draft") if isinstance(result.get("draft"), dict) else {}
        if updated_draft:
            draft.update(updated_draft)

        if sent:
            draft["sent_status"] = True
            draft.setdefault("dispatched_at", dispatched_at)
            draft.setdefault("message_id", message_id)
            if not draft.get("thread_headers") and updated_draft.get("thread_headers"):
                draft["thread_headers"] = updated_draft.get("thread_headers")

            record = {
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "message_id": message_id,
                "dispatched_at": draft.get("dispatched_at", dispatched_at),
                "thread_headers": draft.get("thread_headers"),
                "status": "sent",
                "round": dispatch_context.get("round")
                if isinstance(dispatch_context, dict)
                else round_number,
            }

            log_payload = {
                "event": "email_dispatched",
                "workflow_id": workflow_id,
                "unique_id": unique_id,
                "supplier_id": supplier_id,
                "message_id": message_id,
                "to": recipients,
                "from": sender,
                "subject": subject,
                "round": record.get("round"),
            }
            logger.info(json.dumps(log_payload))
            return record

        failure = {
            "unique_id": unique_id,
            "supplier_id": supplier_id,
            "message_id": message_id,
            "status": "failed",
            "error": result.get("error") or "dispatch_failed",
        }
        logger.error(json.dumps({"event": "email_dispatch_failed", **failure}))
        return failure

    # ------------------------------------------------------------------
    # Agent API
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        input_data = context.input_data or {}
        workflow_id = self._coerce_text(input_data.get("workflow_id") or context.workflow_id)
        round_number = self._coerce_int(
            input_data.get("round") or input_data.get("round_number")
        )

        drafts_payload = input_data.get("drafts")
        if drafts_payload is None:
            drafts: List[Dict[str, Any]] = []
        elif isinstance(drafts_payload, list):
            drafts = [draft for draft in drafts_payload if isinstance(draft, dict)]
        else:
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED,
                    data={
                        "workflow_id": workflow_id,
                        "error": "drafts must be provided as a list",
                    },
                    error="drafts must be provided as a list",
                ),
            )

        dispatch_records: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for draft in drafts:
            unique_id = self._coerce_text(draft.get("unique_id"))
            already = self._already_dispatched(workflow_id, unique_id, draft)
            if already:
                dispatch_records.append(already)
                continue

            record = self._send_draft(draft, workflow_id, round_number)
            dispatch_records.append(record)
            if record.get("status") != "sent":
                failures.append(record)

        expected_count = len(dispatch_records)
        dispatched_success = [
            record for record in dispatch_records if record.get("status") == "sent"
        ]

        payload: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "round": round_number,
            "dispatch_records": dispatch_records,
            "dispatched": dispatched_success,
            "failures": failures,
            "expected_dispatches": expected_count,
            "drafts": drafts,
        }

        if not failures and workflow_id:
            unique_ids = [
                record.get("unique_id")
                for record in dispatch_records
                if record.get("unique_id")
            ]
            event_payload = {
                "workflow_id": workflow_id,
                "round": round_number,
                "expected_count": expected_count,
                "unique_ids": unique_ids,
            }
            self.event_bus.publish("round_dispatch_complete", event_payload)
            logger.info(json.dumps({"event": "round_dispatch_complete", **event_payload}))
            try:
                self._response_coordinator.register_expected_responses(
                    workflow_id,
                    unique_ids,
                    expected_count,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Response coordinator registration failed for workflow=%s",
                    workflow_id,
                    exc_info=True,
                )

        summary_payload = {
            "event": "email_dispatch_summary",
            "workflow_id": workflow_id,
            "round": round_number,
            "expected_dispatches": expected_count,
            "sent": len(dispatched_success),
            "failed": len(failures),
        }
        logger.info(json.dumps(summary_payload))

        status = AgentStatus.FAILED if failures else AgentStatus.SUCCESS

        return self._with_plan(
            context,
            AgentOutput(
                status=status,
                data=payload,
                pass_fields={"drafts": drafts, "dispatch_records": dispatch_records},
            ),
        )

