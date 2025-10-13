import json
import logging
import re
import imaplib
import time
import os
from datetime import datetime, timedelta
from email import message_from_bytes
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_dispatch_chain_store import pending_dispatch_count
from repositories import supplier_response_repo, workflow_email_tracking_repo
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

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)
        self._email_watcher = None
        self._negotiation_agent = None

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

    @staticmethod
    def _response_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        body = row.get("body_text") or row.get("response_body") or ""
        subject = row.get("subject") or row.get("response_subject") or ""
        supplier_id = row.get("supplier_id")
        workflow_id = row.get("workflow_id")
        unique_id = row.get("unique_id")
        message_id = row.get("message_id") or row.get("response_message_id")
        from_addr = row.get("from_addr") or row.get("response_from")
        received_at = row.get("received_at") or row.get("response_date")
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
            "supplier_email": row.get("supplier_email"),
            "body_text": row.get("response_body") or "",
            "subject": row.get("response_subject"),
            "message_id": row.get("response_message_id"),
            "from_addr": row.get("response_from"),
            "received_at": row.get("response_date"),
            "mailbox": row.get("mailbox"),
            "imap_uid": row.get("imap_uid"),
            "match_confidence": row.get("match_confidence"),
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

    def _await_dispatch_metadata(
        self,
        workflow_id: str,
        *,
        poll_seconds: int,
        deadline: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Wait for dispatch tracking rows to be fully populated."""

        interval = poll_seconds if poll_seconds and poll_seconds > 0 else 5
        interval = max(1, interval)
        logged_wait = False

        while True:
            metadata = self._load_dispatch_metadata(workflow_id)
            if metadata is not None:
                return metadata

            now = time.time()
            if deadline is not None and now >= deadline:
                return None

            if not logged_wait:
                logger.info(
                    "Awaiting dispatch completion for workflow=%s before polling responses",
                    workflow_id,
                )
                logged_wait = True

            remaining = None if deadline is None else max(0.0, deadline - now)
            sleep_for = interval if remaining is None else min(interval, remaining)
            if sleep_for <= 0:
                return None
            time.sleep(sleep_for)

    def _poll_supplier_response_rows(
        self,
        workflow_id: str,
        *,
        poll_seconds: int,
        deadline: Optional[float],
        attempt_limit: Optional[int],
        supplier_filter: Optional[Set[str]] = None,
        unique_filter: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        metadata = self._await_dispatch_metadata(
            workflow_id,
            poll_seconds=poll_seconds,
            deadline=deadline,
        )
        if metadata is None:
            return []

        try:
            supplier_response_repo.init_schema()
        except Exception:
            logger.exception("Failed to initialise supplier response schema for workflow=%s", workflow_id)
            return []

        last_dispatched_at = metadata["last_dispatched_at"]
        now = time.time()
        wait_until = last_dispatched_at.timestamp() + 90.0
        if wait_until > now:
            wait_duration = wait_until - now
            if deadline is not None and now + wait_duration > deadline:
                wait_duration = max(0.0, deadline - now)
            if wait_duration > 0:
                logger.info(
                    "Waiting %.1fs after dispatch before polling supplier responses for workflow=%s",
                    wait_duration,
                    workflow_id,
                )
                time.sleep(wait_duration)

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

        expected_suppliers: Set[str] = {
            dispatch_index[uid]["supplier_id"]
            for uid in expected_unique_ids
            if dispatch_index.get(uid, {}).get("supplier_id")
        }
        if not expected_unique_ids and normalised_supplier_filter:
            expected_suppliers = set(normalised_supplier_filter)

        attempts = 0
        collected_by_unique: Dict[str, Dict[str, Any]] = {}
        collected_by_supplier: Dict[str, Dict[str, Any]] = {}
        other_rows: List[Dict[str, Any]] = []
        other_signatures: Set[Tuple[Tuple[str, Any], ...]] = set()
        collected: Optional[List[Dict[str, Any]]] = None

        while True:
            pending_rows = supplier_response_repo.fetch_pending(workflow_id=workflow_id)
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

            if serialised:
                for row in serialised:
                    unique_id = self._coerce_text(row.get("unique_id"))
                    supplier_id = self._coerce_text(row.get("supplier_id"))
                    if unique_id:
                        collected_by_unique[unique_id] = row
                    if supplier_id:
                        collected_by_supplier[supplier_id] = row
                    if not unique_id and not supplier_id:
                        signature = tuple(sorted(row.items()))
                        if signature not in other_signatures:
                            other_signatures.add(signature)
                            other_rows.append(row)

            unique_complete = bool(expected_unique_ids) and expected_unique_ids.issubset(
                collected_by_unique.keys()
            )
            supplier_complete = bool(expected_suppliers) and expected_suppliers.issubset(
                collected_by_supplier.keys()
            )

            if unique_complete or supplier_complete:
                break

            if not expected_unique_ids and not expected_suppliers and serialised:
                collected = serialised
                break

            attempts += 1
            if attempt_limit and attempts >= attempt_limit:
                break

            now = time.time()
            if deadline is not None and now >= deadline:
                break

            remaining = None if deadline is None else max(0.0, deadline - now)
            sleep_for = poll_seconds if remaining is None else min(poll_seconds, remaining)
            if sleep_for <= 0:
                break

            logger.debug(
                "Supplier responses pending for workflow=%s; sleeping %.1fs (attempt=%s)",
                workflow_id,
                sleep_for,
                attempts,
            )
            time.sleep(sleep_for)

        if collected is not None:
            return collected

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
            for uid, row in collected_by_unique.items():
                if uid not in ordered_unique_ids and row not in ordered_results:
                    ordered_results.append(row)
            for supplier_id in expected_suppliers:
                row = collected_by_supplier.get(supplier_id)
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
            for supplier_id, row in collected_by_supplier.items():
                if row not in ordered_results:
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

    def run(self, context: AgentContext) -> AgentOutput:
        """Process a single supplier email or poll the mailbox."""
        action = context.input_data.get("action")
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

        input_data = dict(context.input_data)
        subject = str(input_data.get("subject") or "")
        message_text = input_data.get("message")
        from_address = input_data.get("from_address")
        message_id = input_data.get("message_id")
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
        draft_match = self._select_draft(drafts, supplier_id=supplier_id, rfq_id=rfq_id)
        if not supplier_id and draft_match:
            supplier_id = draft_match.get("supplier_id")
        if not rfq_id and draft_match:
            rfq_id = draft_match.get("rfq_id")

        if not supplier_id:
            candidates = input_data.get("supplier_candidates", [])
            supplier_id = candidates[0] if candidates else None

        await_flag = input_data.get("await_response")
        should_wait = not body and (await_flag is True or (await_flag is None and bool(drafts)))

        precomputed: Optional[Dict[str, Any]] = None
        related_override: Optional[List[Any]] = None
        target_override: Optional[float] = None

        if should_wait:
            if not rfq_id:
                logger.error(
                    "Awaiting supplier response requires an RFQ identifier; subject=%s",
                    subject,
                )
                rfq_id = self._extract_rfq_id(subject)
            if not rfq_id:
                return self._with_plan(
                    context,
                    AgentOutput(
                        status=AgentStatus.FAILED,
                        data={},
                        error="rfq_id required to await supplier response",
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
                )

            if not wait_result:
                if await_all and parallel_results and any(result is None for result in parallel_results):
                    logger.error(
                        "Supplier responses missing for one or more RFQs while awaiting all responses; rfq_id=%s supplier=%s",
                        rfq_id,
                        supplier_id,
                    )
                logger.error(
                    "Supplier response not received for RFQ %s (supplier=%s) before timeout=%ss",
                    rfq_id,
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
                    "Supplier response for RFQ %s (supplier=%s) returned status=%s; error=%s",
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
            message_id = wait_result.get("message_id") or message_id

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
                "Supplier interaction failed because no message body was available (rfq_id=%s, supplier=%s)",
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
            rfq_id,
            supplier_id,
            parsed.get("response_text") or body,
            parsed,
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
                            rfq_id,
                            None,
                            text,
                            parsed_payload,
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
    ) -> Optional[Dict]:
        """Await supplier replies using the workflow-aware IMAP watcher."""

        workflow_key = self._coerce_text(workflow_id)
        if not workflow_key:
            logger.error(
                "wait_for_response requires workflow_id; received rfq_id=%s supplier=%s",
                rfq_id,
                supplier_id,
            )
            return None

        run_identifier = self._coerce_text(dispatch_run_id)
        if not run_identifier:
            run_identifier = self._coerce_text(draft_action_id) or self._coerce_text(action_id)

        poll_seconds = (
            self._coerce_int(poll_interval, default=0)
            if poll_interval is not None
            else getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)
        )
        if poll_seconds is None:
            poll_seconds = 60
        try:
            poll_seconds = int(poll_seconds)
        except Exception:
            poll_seconds = 60
        if poll_seconds <= 0:
            poll_seconds = 60

        try:
            timeout_seconds = int(timeout) if timeout is not None else 0
        except Exception:
            timeout_seconds = 0
        if timeout_seconds < 0:
            timeout_seconds = 0

        now = time.time()
        deadline = now + timeout_seconds if timeout_seconds else now
        attempt_limit: Optional[int] = None
        if max_attempts is not None:
            try:
                attempt_limit = max(1, int(max_attempts))
            except Exception:
                attempt_limit = None
        if attempt_limit is None:
            setting_attempts = getattr(
                getattr(self.agent_nick, "settings", None),
                "email_response_max_attempts",
                None,
            )
            if setting_attempts is not None:
                try:
                    attempt_limit = max(1, int(setting_attempts))
                except Exception:
                    attempt_limit = None
        target_supplier = self._coerce_text(supplier_id)
        supplier_filter: Optional[Set[str]] = None
        if target_supplier:
            supplier_filter = {target_supplier}

        responses = self._poll_supplier_response_rows(
            workflow_key,
            poll_seconds=poll_seconds,
            deadline=deadline,
            attempt_limit=attempt_limit,
            supplier_filter=supplier_filter,
        )

        if not responses:
            logger.info(
                "No supplier responses stored for workflow=%s run_id=%s",
                workflow_key,
                run_identifier,
            )
            return None

        return self._response_from_row(responses[0]) if responses else None


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
            ctx["workflow_id"] for ctx in contexts if ctx.get("workflow_id")
        }
        if len(workflow_ids) != 1:
            logger.error(
                "Cannot aggregate responses without a single workflow_id; contexts=%s",
                contexts,
            )
            return [None] * len(drafts)

        workflow_id = workflow_ids.pop()
        run_ids = {ctx.get("run_id") for ctx in contexts if ctx.get("run_id")}
        run_identifier = run_ids.pop() if len(run_ids) == 1 else None

        poll_seconds = (
            self._coerce_int(poll_interval, default=0)
            if poll_interval is not None
            else getattr(self.agent_nick.settings, "email_response_poll_seconds", 60)
        )
        if poll_seconds is None:
            poll_seconds = 60
        try:
            poll_seconds = int(poll_seconds)
        except Exception:
            poll_seconds = 60
        if poll_seconds <= 0:
            poll_seconds = 60

        try:
            timeout_seconds = int(timeout) if timeout is not None else 0
        except Exception:
            timeout_seconds = 0
        if timeout_seconds < 0:
            timeout_seconds = 0

        deadline = time.time() + timeout_seconds if timeout_seconds else time.time()

        responses = self._poll_supplier_response_rows(
            workflow_id,
            poll_seconds=poll_seconds,
            deadline=deadline,
            attempt_limit=None,
        )

        if not responses:
            logger.info(
                "Aggregated watcher not ready for workflow=%s (no responses stored)",
                workflow_id,
            )
            return [None] * len(drafts)

        response_map = {
            row.get("unique_id"): self._response_from_row(row)
            for row in responses
            if row.get("unique_id")
        }

        results: List[Optional[Dict]] = []
        for draft, ctx in zip(drafts, contexts):
            response = response_map.get(ctx.get("unique_id"))
            if response is None and ctx.get("supplier_id"):
                for row in responses:
                    if self._coerce_text(row.get("supplier_id")) == ctx.get("supplier_id"):
                        response = self._response_from_row(row)
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
        supplier_id = result.get("supplier_id")
        if not rfq_id:
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
            rfq_id,
            supplier_id,
            message_body,
            parsed,
            message_id=result.get("message_id"),
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
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
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
        rfq_id: Optional[str],
        supplier_id: Optional[str],
        text: str,
        parsed: Dict,
        *,
        message_id: Optional[str] = None,
        from_address: Optional[str] = None,
    ) -> None:
        if not rfq_id:
            return

        resolved_supplier = self._resolve_supplier_id(
            rfq_id,
            supplier_id,
            message_id=message_id,
            from_address=from_address,
        )
        if not resolved_supplier:
            logger.error(
                "failed to store supplier response because supplier_id could not be resolved (rfq_id=%s, message_id=%s)",
                rfq_id,
                message_id,
            )
            return

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.supplier_responses
                            (rfq_id, supplier_id, response_text, price, lead_time, submitted_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (rfq_id, supplier_id) DO UPDATE
                            SET response_text = EXCLUDED.response_text,
                                price = EXCLUDED.price,
                                lead_time = EXCLUDED.lead_time,
                                submitted_at = NOW()
                        """,
                        (
                            rfq_id,
                            resolved_supplier,
                            text,
                            parsed.get("price"),
                            parsed.get("lead_time"),
                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store supplier response")
