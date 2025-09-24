import logging
import re
import imaplib
import time
import os
from datetime import datetime, timedelta
from email import message_from_bytes
from typing import Dict, Optional

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


class SupplierInteractionAgent(BaseAgent):
    """Monitor and parse supplier RFQ responses."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        os.environ.setdefault("PROCWISE_DEVICE", self.device)

    RFQ_PATTERN = re.compile(r"RFQ-\d{8}-[a-f0-9]{8}")

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

        subject = context.input_data.get("subject", "")
        body = context.input_data.get("message") or context.input_data.get("body", "")
        supplier_id = context.input_data.get("supplier_id")
        if not supplier_id:
            candidates = context.input_data.get("supplier_candidates", [])
            supplier_id = candidates[0] if candidates else None
        if not body:
            return AgentOutput(status=AgentStatus.FAILED, data={}, error="message not provided")

        # Retrieve related documents from the vector database for additional context
        context_hits = self.vector_search(body, top_k=3)
        related_docs = [h.payload for h in context_hits]

        rfq_id = self._extract_rfq_id(subject + " " + body)
        parsed = self._parse_response(body)
        self._store_response(rfq_id, supplier_id, body, parsed)

        target = context.input_data.get("target_price")
        next_agent = []
        if parsed.get("price") and target and parsed["price"] > float(target):
            next_agent = ["NegotiationAgent"]
        else:
            next_agent = ["QuoteEvaluationAgent"]

        payload = {"rfq_id": rfq_id, "supplier_id": supplier_id, **parsed, "related_documents": related_docs}
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
                        self._store_response(rfq_id, None, text, self._parse_response(text))
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
        poll_interval: int = 15,
        limit: int = 1,
    ) -> Optional[Dict]:
        """Wait for an inbound supplier email and return the processed result."""

        deadline = time.time() + max(timeout, 0)
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

            registry = getattr(self.agent_nick, "agents", {})
            negotiation_agent = None
            if isinstance(registry, dict):
                negotiation_agent = registry.get("negotiation") or registry.get(
                    "NegotiationAgent"
                )
            if negotiation_agent is None and NegotiationAgent is not None:
                negotiation_agent = NegotiationAgent(self.agent_nick)

            active_watcher = SESEmailWatcher(
                self.agent_nick,
                supplier_agent=self,
                negotiation_agent=negotiation_agent,
            )

        result: Optional[Dict] = None
        while time.time() <= deadline:
            try:
                batch = active_watcher.poll_once(limit=limit)
            except Exception:  # pragma: no cover - best effort
                logger.exception("wait_for_response poll failed")
                batch = []
            if batch:
                result = batch[-1]
                break
            if poll_interval <= 0:
                break
            time.sleep(min(poll_interval, max(0, deadline - time.time())))

        return result

    def _extract_rfq_id(self, text: str) -> Optional[str]:
        match = self.RFQ_PATTERN.search(text)
        return match.group(0) if match else None

    def _parse_response(self, text: str) -> Dict:
        price_match = re.search(r"(\d+[.,]?\d*)", text)
        lead_match = re.search(r"(\d+)\s*days", text, re.IGNORECASE)
        price = float(price_match.group(1).replace(',', '')) if price_match else None
        lead_time = lead_match.group(1) if lead_match else None
        return {"price": price, "lead_time": lead_time, "response_text": text}

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
