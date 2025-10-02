import json
import logging
import re
import imaplib
import time
import os
from datetime import datetime, timedelta
from email import message_from_bytes
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class SupplierInteractionAgent(BaseAgent):
    """Monitor and parse supplier RFQ responses."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)
        self._email_watcher = None

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
            return AgentOutput(status=AgentStatus.SUCCESS, data=payload, pass_fields=payload)
        if action == "monitor":
            interval = int(context.input_data.get("interval", 30))
            duration = int(context.input_data.get("duration", 300))
            count = self.monitor_inbox(interval=interval, duration=duration)
            payload = {"monitored": count}
            return AgentOutput(status=AgentStatus.SUCCESS, data=payload, pass_fields=payload)
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
            return AgentOutput(status=AgentStatus.SUCCESS, data=payload, pass_fields=payload)

        input_data = dict(context.input_data)
        subject = str(input_data.get("subject") or "")
        message_text = input_data.get("message")
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
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="rfq_id required to await supplier response",
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

            wait_result = self.wait_for_response(
                timeout=timeout,
                poll_interval=poll_interval,
                limit=batch_limit,
                rfq_id=rfq_id,
                supplier_id=supplier_id,
                subject_hint=subject,
                from_address=expected_sender,
            )

            if not wait_result:
                logger.error(
                    "Supplier response not received for RFQ %s (supplier=%s) before timeout=%ss",
                    rfq_id,
                    supplier_id,
                    timeout,
                )
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error="supplier response not received",
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
                return AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error=error_detail,
                )

            subject = str(wait_result.get("subject") or subject)
            supplier_id = wait_result.get("supplier_id") or supplier_id
            rfq_id = wait_result.get("rfq_id") or rfq_id

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
            return AgentOutput(status=AgentStatus.FAILED, data={}, error="message not provided")

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

        self._store_response(rfq_id, supplier_id, parsed.get("response_text") or body, parsed)

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

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data=payload,
            pass_fields=payload,
            next_agents=next_agent,
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
                        self._store_response(
                            rfq_id,
                            None,
                            text,
                            self._parse_response(
                                text,
                                subject=subject,
                                rfq_id=rfq_id,
                                supplier_id=None,
                            ),
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
    ) -> Optional[Dict]:
        """Wait for an inbound supplier email and return the processed result."""

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
                from agents.negotiation_agent import NegotiationAgent
            except Exception:  # pragma: no cover - optional import
                SESEmailWatcher = None
                NegotiationAgent = None  # type: ignore

            if SESEmailWatcher is None:
                return None

            if self._email_watcher is None:
                registry = getattr(self.agent_nick, "agents", {})
                negotiation_agent = None
                if isinstance(registry, dict):
                    negotiation_agent = registry.get("negotiation") or registry.get(
                        "NegotiationAgent"
                    )

                self._email_watcher = SESEmailWatcher(
                    self.agent_nick,
                    supplier_agent=self,
                    negotiation_agent=negotiation_agent,
                    enable_negotiation=False,
                    response_poll_seconds=getattr(
                        self.agent_nick.settings, "email_response_poll_seconds", 60
                    ),
                )
            active_watcher = self._email_watcher

        result: Optional[Dict] = None
        rfq_normalised = self._normalise_identifier(rfq_id)
        supplier_normalised = self._normalise_identifier(supplier_id)
        subject_norm = str(subject_hint or "").strip().lower()
        sender_normalised = self._normalise_identifier(from_address)

        match_filters: Dict[str, object] = {}
        if rfq_id:
            match_filters["rfq_id"] = rfq_id
        if supplier_id:
            match_filters["supplier_id"] = supplier_id
        if subject_hint:
            match_filters["subject_contains"] = subject_hint
        if from_address:
            match_filters["from_address"] = from_address

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
                rfq_only_match: Optional[Dict[str, object]] = None
                for candidate in batch:
                    if rfq_normalised and self._normalise_identifier(
                        candidate.get("rfq_id")
                    ) != rfq_normalised:
                        continue
                    rfq_matches = bool(rfq_normalised)
                    if supplier_normalised and self._normalise_identifier(
                        candidate.get("supplier_id")
                    ) != supplier_normalised:
                        if rfq_matches and rfq_only_match is None:
                            rfq_only_match = candidate
                        continue
                    if subject_norm:
                        subj = str(candidate.get("subject") or "").lower()
                        if subject_norm not in subj:
                            if rfq_matches and rfq_only_match is None:
                                rfq_only_match = candidate
                            continue
                    if sender_normalised:
                        sender = self._normalise_identifier(candidate.get("from_address"))
                        if sender and sender != sender_normalised:
                            if rfq_matches and rfq_only_match is None:
                                rfq_only_match = candidate
                            continue
                    status_value = str(candidate.get("supplier_status") or "").lower()
                    payload_ready = candidate.get("supplier_output")
                    if status_value in {"processing", "pending"} and not payload_ready:
                        if rfq_matches and rfq_only_match is None:
                            rfq_only_match = candidate
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
                        if rfq_matches and rfq_only_match is None:
                            rfq_only_match = candidate
                        logger.debug(
                            "Supplier response for RFQ %s matched filters but has no payload yet; continuing to wait",
                            candidate.get("rfq_id") or rfq_id,
                        )
                        continue
                    result = candidate
                    break
                if result is None and rfq_only_match is not None:
                    fallback_status = str(rfq_only_match.get("supplier_status") or "").lower()
                    fallback_payload_ready = rfq_only_match.get("supplier_output")
                    if fallback_status in {"processing", "pending"} and not fallback_payload_ready:
                        rfq_only_match = None
                    else:
                        logger.info(
                            "Returning RFQ-matched response for %s despite secondary filter mismatch",
                            rfq_id,
                        )
                        result = rfq_only_match
                if result is None and not any(
                    [rfq_normalised, supplier_normalised, subject_norm, sender_normalised]
                ):
                    result = batch[-1]
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

    def _store_response(self, rfq_id: Optional[str], supplier_id: Optional[str], text: str, parsed: Dict) -> None:
        if not rfq_id:
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
                            supplier_id,
                            text,
                            parsed.get("price"),
                            parsed.get("lead_time"),
                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to store supplier response")
