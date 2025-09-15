import logging
import re
import imaplib
import time
import os
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
            return AgentOutput(status=AgentStatus.SUCCESS, data={"polled": count})
        if action == "monitor":
            interval = int(context.input_data.get("interval", 30))
            duration = int(context.input_data.get("duration", 300))
            count = self.monitor_inbox(interval=interval, duration=duration)
            return AgentOutput(status=AgentStatus.SUCCESS, data={"monitored": count})

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

        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={"rfq_id": rfq_id, "supplier_id": supplier_id, **parsed, "related_documents": related_docs},
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
