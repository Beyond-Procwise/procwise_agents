import json
import logging
import re
import imaplib
import threading
import time
import os
from datetime import datetime, timedelta
from email import message_from_bytes
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_dispatch_chain_store import pending_dispatch_count
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
                if watch_candidates:
                    primary_candidate = watch_candidates[0]
                    if isinstance(primary_candidate, dict):
                        for key in ("action_id", "draft_action_id", "email_action_id"):
                            candidate_action = primary_candidate.get(key)
                            if isinstance(candidate_action, str) and candidate_action.strip():
                                draft_action_id = candidate_action.strip()
                                break
                        workflow_hint = primary_candidate.get("workflow_id")
                        if not workflow_hint:
                            meta = primary_candidate.get("metadata")
                            if isinstance(meta, dict):
                                meta_workflow = meta.get("workflow_id") or meta.get("process_workflow_id")
                                if isinstance(meta_workflow, str) and meta_workflow.strip():
                                    workflow_hint = meta_workflow.strip()
                        dispatch_run_id = self._extract_dispatch_run_id(primary_candidate)
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
    ) -> Optional[Dict]:
        """Wait for an inbound supplier email and return the processed result.

        When ``enable_negotiation`` is ``True`` the underlying SES watcher is
        initialised with a negotiation agent reference so counter rounds can be
        triggered automatically as part of the polling loop.  NegotiationAgent
        itself disables the flag while awaiting responses to avoid recursive
        execution.
        """

        deadline = time.time() + max(timeout, 0)

        attempt_cap_value = max_attempts
        if attempt_cap_value is None:
            legacy_setting = getattr(
                self.agent_nick.settings, "email_response_max_attempts", None
            )
            if legacy_setting is not None:
                logger.debug(
                    "Ignoring legacy email_response_max_attempts=%s in favour of timeout-based S3 polling",
                    legacy_setting,
                )
            attempt_cap = None
        else:
            attempt_cap = self._coerce_int(attempt_cap_value, default=0)
            if attempt_cap <= 0:
                attempt_cap = None
        attempts_made = 0

        if (
            watcher is None
            and not getattr(self.agent_nick, "dispatch_service_started", False)
        ):
            logger.debug(
                "Email watcher initialisation deferred until dispatch service starts; waiting up to %ss",
                timeout,
            )
            while time.time() <= deadline:
                if getattr(self.agent_nick, "dispatch_service_started", False):
                    break
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0:
                    break
                time.sleep(min(1.0, remaining))
            if not getattr(self.agent_nick, "dispatch_service_started", False):
                logger.warning(
                    "Dispatch service did not start before timeout while waiting for supplier response (rfq_id=%s, supplier=%s)",
                    rfq_id,
                    supplier_id,
                )
                return None

        active_watcher = watcher
        if active_watcher is None:
            try:
                from services.email_watcher import SESEmailWatcher
            except Exception:  # pragma: no cover - optional import
                SESEmailWatcher = None

            if SESEmailWatcher is None:
                return None

            watcher_needs_refresh = False
            if self._email_watcher is None:
                watcher_needs_refresh = True
            else:
                current_flag = getattr(self._email_watcher, "enable_negotiation", True)
                if bool(current_flag) != bool(enable_negotiation):
                    watcher_needs_refresh = True

            if watcher_needs_refresh:
                negotiation_agent = None
                if enable_negotiation:
                    negotiation_agent = self._get_negotiation_agent()

                poll_setting = getattr(
                    self.agent_nick.settings, "email_response_poll_seconds", 60
                )
                if poll_interval is not None:
                    response_poll_seconds = self._coerce_int(
                        poll_interval, default=poll_setting
                    )
                else:
                    response_poll_seconds = poll_setting

                self._email_watcher = SESEmailWatcher(
                    self.agent_nick,
                    supplier_agent=self,
                    negotiation_agent=negotiation_agent,
                    enable_negotiation=enable_negotiation,
                    response_poll_seconds=response_poll_seconds,
                )
            active_watcher = self._email_watcher

        result: Optional[Dict] = None
        supplier_normalised = self._normalise_identifier(supplier_id)
        run_normalised = self._normalise_identifier(dispatch_run_id)
        subject_norm = str(subject_hint or "").strip().lower()
        sender_normalised = self._normalise_identifier(from_address)

        match_filters: Dict[str, object] = {}
        if supplier_id:
            match_filters["supplier_id"] = supplier_id
        if dispatch_run_id:
            match_filters["dispatch_run_id"] = dispatch_run_id
        if draft_action_id:
            match_filters["draft_action_id"] = draft_action_id

        while time.time() <= deadline:
            if attempt_cap is not None and attempts_made >= attempt_cap:
                logger.info(
                    "Reached maximum email poll attempts (%s) while waiting for RFQ %s",
                    attempt_cap,
                    rfq_id,
                )
                break
            attempts_made += 1
            try:
                if match_filters:
                    batch = active_watcher.poll_once(limit=limit, match_filters=match_filters)
                else:
                    batch = active_watcher.poll_once(limit=limit)
            except Exception:  # pragma: no cover - best effort
                logger.exception("wait_for_response poll failed")
                batch = []
            if batch:
                for candidate in batch:
                    if supplier_normalised and self._normalise_identifier(
                        candidate.get("supplier_id")
                    ) != supplier_normalised:
                        continue
                    if run_normalised:
                        candidate_run = self._normalise_identifier(
                            candidate.get("dispatch_run_id") or candidate.get("run_id")
                        )
                        if candidate_run != run_normalised:
                            continue
                    if subject_norm:
                        subj = str(candidate.get("subject") or "").lower()
                        if subject_norm not in subj:
                            continue
                    if sender_normalised:
                        sender = self._normalise_identifier(candidate.get("from_address"))
                        if sender and sender != sender_normalised:
                            continue
                    status_value = str(candidate.get("supplier_status") or "").lower()
                    payload_ready = candidate.get("supplier_output")
                    if status_value in {"processing", "pending"} and not payload_ready:
                        logger.debug(
                            "Supplier response still processing for RFQ %s (supplier=%s); status=%s",
                            candidate.get("rfq_id") or rfq_id,
                            candidate.get("supplier_id") or supplier_id,
                            status_value,
                        )
                        continue
                    if status_value == AgentStatus.FAILED.value:
                        error_detail = candidate.get("error") or "unknown_error"
                        logger.error(
                            "Supplier response poll detected failure for RFQ %s (supplier=%s): %s",
                            candidate.get("rfq_id") or rfq_id,
                            candidate.get("supplier_id") or supplier_id,
                            error_detail,
                        )
                        result = candidate
                        break
                    if not payload_ready:
                        logger.debug(
                            "Supplier response for RFQ %s matched filters but has no payload yet; continuing to wait",
                            candidate.get("rfq_id") or rfq_id,
                        )
                        continue
                    result = candidate
                    break
                if result is not None:
                    break
            interval_value = poll_interval or getattr(
                self.agent_nick.settings, "email_response_poll_seconds", 60
            )
            if interval_value <= 0:
                break
            time.sleep(
                min(max(1, interval_value), max(0, deadline - time.time()))
            )

        if result is None:
            if attempt_cap is not None and attempts_made >= attempt_cap:
                logger.warning(
                    "Stopped waiting for supplier response (rfq_id=%s, supplier=%s) after %s attempt(s)",
                    rfq_id,
                    supplier_id,
                    attempts_made,
                )
            else:
                logger.warning(
                    "Timed out waiting for supplier response (rfq_id=%s, supplier=%s) after %ss",
                    rfq_id,
                    supplier_id,
                    timeout,
                )

        return result

    def wait_for_multiple_responses(
        self,
        drafts: List[Dict[str, Any]],
        *,
        timeout: int = 300,
        poll_interval: Optional[int] = None,
        limit: int = 1,
        enable_negotiation: bool = True,
    ) -> List[Optional[Dict]]:
        """Spawn concurrent watchers for each draft and wait for supplier replies.

        Args:
            drafts: A list of draft payloads from ``EmailDraftingAgent``. Each
                payload should include at least an ``rfq_id`` and ``supplier_id``.
            timeout: Maximum seconds each watcher should wait for a reply.
            poll_interval: Optional override for the polling cadence in seconds.
            limit: Maximum number of messages to fetch per poll iteration.

        Returns:
            A list of supplier response payloads in the same order as ``drafts``.
            Entries are ``None`` when the response was not observed before the
            timeout or if the draft payload was missing required identifiers.
        """

        if not drafts:
            return []

        safe_interval = poll_interval
        if safe_interval is not None:
            try:
                safe_interval = max(1, int(safe_interval))
            except Exception:
                safe_interval = None

        deadline = time.time() + max(timeout, 0)

        results: List[Optional[Dict]] = [None] * len(drafts)
        errors: List[Optional[BaseException]] = [None] * len(drafts)
        lock = threading.Lock()
        threads: List[threading.Thread] = []

        try:
            from services.email_watcher import SESEmailWatcher
        except Exception:  # pragma: no cover - optional dependency
            logger.exception("Unable to initialise email watcher for parallel polling")
            return results

        negotiation_agent = self._get_negotiation_agent() if enable_negotiation else None

        def _watch_single(index: int, draft: Dict[str, Any]) -> None:
            rfq_id = draft.get("rfq_id")
            supplier_id = draft.get("supplier_id")
            if not rfq_id or not supplier_id:
                logger.warning(
                    "Skipping parallel watch for draft missing identifiers (index=%s)", index
                )
                return

            recipient_hint = draft.get("receiver") or draft.get("recipient_email")
            if recipient_hint is None:
                recipients_field = draft.get("recipients")
                if isinstance(recipients_field, (list, tuple)) and recipients_field:
                    recipient_hint = recipients_field[0]
                elif isinstance(recipients_field, str):
                    recipient_hint = recipients_field
            subject_hint = draft.get("subject")
            draft_action_id = None
            for key in ("action_id", "draft_action_id", "email_action_id"):
                candidate = draft.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    draft_action_id = candidate.strip()
                    break
            workflow_hint = draft.get("workflow_id")
            if not workflow_hint:
                meta = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else None
                if meta:
                    meta_workflow = meta.get("workflow_id") or meta.get("process_workflow_id")
                    if isinstance(meta_workflow, str) and meta_workflow.strip():
                        workflow_hint = meta_workflow.strip()
            dispatch_run_id = None
            for key in ("dispatch_run_id", "run_id"):
                candidate = draft.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    dispatch_run_id = candidate.strip()
                    break
            if dispatch_run_id is None and isinstance(draft.get("metadata"), dict):
                meta_run = draft["metadata"].get("dispatch_run_id") or draft["metadata"].get("run_id")
                if isinstance(meta_run, str) and meta_run.strip():
                    dispatch_run_id = meta_run.strip()

            poll_setting = getattr(
                self.agent_nick.settings, "email_response_poll_seconds", 60
            )
            if safe_interval is not None:
                response_interval = self._coerce_int(safe_interval, default=poll_setting)
            else:
                response_interval = poll_setting

            watcher_instance = SESEmailWatcher(
                self.agent_nick,
                supplier_agent=self,
                negotiation_agent=negotiation_agent,
                enable_negotiation=enable_negotiation,
                response_poll_seconds=response_interval,
            )

            try:
                result = self.wait_for_response(
                    watcher=watcher_instance,
                    timeout=timeout,
                    poll_interval=safe_interval,
                    limit=limit,
                    rfq_id=rfq_id,
                    supplier_id=supplier_id,
                    subject_hint=subject_hint,
                    from_address=recipient_hint,
                    enable_negotiation=enable_negotiation,
                    draft_action_id=draft_action_id,
                    workflow_id=workflow_hint,
                    dispatch_run_id=dispatch_run_id,

                )
                with lock:
                    results[index] = result
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "Parallel email watch failed for rfq_id=%s supplier=%s", rfq_id, supplier_id
                )
                with lock:
                    errors[index] = exc

        for idx, draft in enumerate(drafts):
            if not isinstance(draft, dict):
                logger.debug("Ignoring non-dict draft payload at index %s", idx)
                continue
            thread = threading.Thread(
                target=_watch_single,
                name=f"supplier-watch-{idx}",
                args=(idx, draft),
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=timeout + 1)

        for error in errors:
            if error is not None:
                raise error

        if any(result is None for result in results):
            self._await_outstanding_dispatch_responses(
                drafts,
                results,
                deadline=deadline,
                poll_interval=safe_interval,
                limit=limit,
                enable_negotiation=enable_negotiation,
            )

        return results

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
                draft = drafts[idx]
                rfq_id = draft.get("rfq_id")
                supplier_id = draft.get("supplier_id")
                recipient_hint = draft.get("receiver") or draft.get("recipient_email")
                if recipient_hint is None:
                    recipients_field = draft.get("recipients")
                    if isinstance(recipients_field, (list, tuple)) and recipients_field:
                        recipient_hint = recipients_field[0]
                    elif isinstance(recipients_field, str):
                        recipient_hint = recipients_field

                subject_hint = draft.get("subject")
                dispatch_run_id = self._extract_dispatch_run_id(draft)
                draft_action_id = None
                for key in ("action_id", "draft_action_id", "email_action_id"):
                    candidate_action = draft.get(key)
                    if isinstance(candidate_action, str) and candidate_action.strip():
                        draft_action_id = candidate_action.strip()
                        break

                result = self.wait_for_response(
                    timeout=per_attempt,
                    poll_interval=poll_interval,
                    limit=limit,
                    rfq_id=rfq_id,
                    supplier_id=supplier_id,
                    subject_hint=subject_hint,
                    from_address=recipient_hint,
                    enable_negotiation=enable_negotiation,
                    draft_action_id=draft_action_id,
                    dispatch_run_id=dispatch_run_id,
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
