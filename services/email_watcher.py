from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from typing import Callable, Dict, List, Optional, Protocol

import boto3

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from utils.gpu import configure_gpu


logger = logging.getLogger(__name__)

# Ensure GPU related environment flags are consistently applied even when the
# watcher is used standalone (e.g. in a scheduled job).
configure_gpu()


class EmailLoader(Protocol):
    """Callable protocol that returns a list of inbound email payloads."""

    def __call__(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        ...


class EmailWatcherState(Protocol):
    """Protocol describing the persistence mechanism for processed messages."""

    def __contains__(self, message_id: str) -> bool:  # pragma: no cover - protocol
        ...

    def add(self, message_id: str, metadata: Optional[Dict[str, object]] = None) -> None:  # pragma: no cover - protocol
        ...

    def get(self, message_id: str) -> Optional[Dict[str, object]]:  # pragma: no cover - protocol
        ...


@dataclass
class InMemoryEmailWatcherState:
    """Simple in-memory store of processed message identifiers."""

    _seen: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def __contains__(self, message_id: str) -> bool:
        return message_id in self._seen

    def add(self, message_id: str, metadata: Optional[Dict[str, object]] = None) -> None:
        self._seen[message_id] = metadata or {}

    def get(self, message_id: str) -> Optional[Dict[str, object]]:
        return self._seen.get(message_id)


def _strip_html(value: str) -> str:
    """Convert HTML content to a whitespace normalised string."""

    if not value:
        return ""
    # Basic tag removal keeps the implementation lightweight without
    # introducing heavy dependencies such as BeautifulSoup for tests.
    cleaned = re.sub(r"<\s*(script|style).*?>.*?<\s*/\s*\1\s*>", " ", value, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class SESEmailWatcher:
    """Poll and process inbound supplier RFQ responses."""

    def __init__(
        self,
        agent_nick,
        *,
        supplier_agent: Optional[SupplierInteractionAgent] = None,
        negotiation_agent: Optional[NegotiationAgent] = None,
        metadata_provider: Optional[Callable[[str], Dict[str, object]]] = None,
        message_loader: Optional[EmailLoader] = None,
        state_store: Optional[EmailWatcherState] = None,
        enable_negotiation: bool = True,
        response_poll_seconds: Optional[int] = None,
    ) -> None:
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings
        self.supplier_agent = supplier_agent or agent_nick.agents.get("supplier_interaction")
        if self.supplier_agent is None:
            self.supplier_agent = SupplierInteractionAgent(agent_nick)
        self.enable_negotiation = enable_negotiation
        if enable_negotiation:
            self.negotiation_agent = negotiation_agent or agent_nick.agents.get(
                "NegotiationAgent"
            )
            if self.negotiation_agent is None:
                self.negotiation_agent = NegotiationAgent(agent_nick)
        else:
            self.negotiation_agent = None
        self.metadata_provider = metadata_provider
        self.state_store = state_store or InMemoryEmailWatcherState()
        self._custom_loader = message_loader
        poll_interval = response_poll_seconds
        if poll_interval is None:
            poll_interval = getattr(
                self.settings, "email_response_poll_seconds", 60
            )
        try:
            poll_interval = int(poll_interval)
        except Exception:
            poll_interval = 60
        self.poll_interval_seconds = max(1, poll_interval)

        endpoint = getattr(self.settings, "ses_smtp_endpoint", "")
        self.region = getattr(self.settings, "ses_region", None) or self._parse_region(endpoint)
        self.bucket = getattr(self.settings, "ses_inbound_bucket", None) or getattr(
            self.settings, "s3_bucket_name", None
        )
        self.prefix = getattr(self.settings, "ses_inbound_prefix", "ses/inbound/")

        # Lazily created S3 client to avoid mandatory AWS credentials during unit tests.
        self._s3_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def poll_once(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        """Process a single batch of inbound emails.

        Parameters
        ----------
        limit:
            Optional cap on the number of emails retrieved from the source.

        Returns
        -------
        List[Dict[str, object]]
            Structured results for each processed email including negotiation
            details when triggered.
        """

        messages: List[Dict[str, object]] = []
        results: List[Dict[str, object]] = []

        try:
            loader = self._custom_loader or self._load_from_s3
            messages = loader(limit)
        except Exception:  # pragma: no cover - network/runtime
            logger.exception("Failed to load inbound SES messages")
            return results

        for message in messages:
            message_id = str(message.get("id") or uuid.uuid4())
            if self.state_store and message_id in self.state_store:
                continue

            try:
                processed, reason = self._process_message(message)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to process SES message %s", message_id)
                processed, reason = None, "processing_error"

            metadata: Dict[str, object]
            if processed:
                results.append(processed)
                metadata = {
                    "rfq_id": processed.get("rfq_id"),
                    "supplier_id": processed.get("supplier_id"),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "status": "processed",
                }
            else:
                metadata = {
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "status": "skipped",
                    "reason": reason or "unknown",
                }

            if self.state_store:
                self.state_store.add(message_id, metadata)

        return results

    def watch(
        self,
        *,
        interval: Optional[int] = None,
        limit: Optional[int] = None,
        stop_after: Optional[int] = None,
    ) -> int:
        """Continuously poll for messages until ``stop_after`` iterations."""

        iterations = 0
        processed_total = 0
        poll_delay = self.poll_interval_seconds if interval is None else max(interval, 1)
        while True:
            batch = self.poll_once(limit=limit)
            processed_total += len(batch)
            iterations += 1
            if stop_after is not None and iterations >= stop_after:
                break
            time.sleep(poll_delay)
        return processed_total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_message(self, message: Dict[str, object]) -> tuple[Optional[Dict[str, object]], Optional[str]]:
        subject = str(message.get("subject", ""))
        body = str(message.get("body", ""))
        from_address = str(message.get("from", ""))
        rfq_id = message.get("rfq_id")
        if not isinstance(rfq_id, str) or not rfq_id:
            rfq_id = self._extract_rfq_id(subject + " " + body)
        if not rfq_id:
            logger.debug("Skipping email without RFQ identifier: %s", subject)
            return None, "missing_rfq_id"

        metadata = self._load_metadata(rfq_id)
        supplier_id = metadata.get("supplier_id") or message.get("supplier_id")
        target_price = metadata.get("target_price") or message.get("target_price")
        negotiation_round = metadata.get("round") or message.get("round") or 1

        if target_price is not None:
            try:
                target_price = float(target_price)
            except (TypeError, ValueError):
                logger.warning("Invalid target price '%s' for RFQ %s", target_price, rfq_id)
                target_price = None

        context = AgentContext(
            workflow_id=str(uuid.uuid4()),
            agent_id="supplier_interaction",
            user_id=self.settings.script_user,
            input_data={
                "subject": subject,
                "message": body,
                "supplier_id": supplier_id,
                "rfq_id": rfq_id,
                "from_address": from_address,
                "target_price": target_price,
            },
        )

        interaction_output = self.supplier_agent.execute(context)
        negotiation_output: Optional[AgentOutput] = None
        triggered = False

        if (
            self.enable_negotiation
            and target_price is not None
            and interaction_output.status == AgentStatus.SUCCESS
            and "NegotiationAgent" in (interaction_output.next_agents or [])
            and self.negotiation_agent is not None
        ):
            current_offer = interaction_output.data.get("price")
            negotiation_context = AgentContext(
                workflow_id=context.workflow_id,
                agent_id="NegotiationAgent",
                user_id=context.user_id,
                input_data={
                    "supplier": supplier_id,
                    "current_offer": current_offer,
                    "target_price": target_price,
                    "rfq_id": rfq_id,
                    "round": negotiation_round,
                },
                parent_agent=context.agent_id,
                routing_history=list(context.routing_history),
            )
            negotiation_output = self.negotiation_agent.execute(negotiation_context)
            triggered = negotiation_output.status == AgentStatus.SUCCESS

        result = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "message_id": message.get("id"),
            "subject": subject,
            "from_address": from_address,
            "price": interaction_output.data.get("price"),
            "lead_time": interaction_output.data.get("lead_time"),
            "target_price": target_price,
            "negotiation_triggered": triggered,
            "supplier_status": interaction_output.status.value,
            "negotiation_status": negotiation_output.status.value if negotiation_output else None,
            "supplier_output": interaction_output.data,
            "negotiation_output": negotiation_output.data if negotiation_output else None,
        }
        return result, None

    def _load_metadata(self, rfq_id: str) -> Dict[str, object]:
        if self.metadata_provider is not None:
            try:
                data = self.metadata_provider(rfq_id) or {}
                return dict(data)
            except Exception:  # pragma: no cover - defensive
                logger.exception("metadata provider failed for %s", rfq_id)

        details: Dict[str, object] = {}
        try:
            with self.agent_nick.get_db_connection() as conn:  # pragma: no cover - network
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT supplier_id FROM proc.draft_rfq_emails WHERE rfq_id = %s ORDER BY created_on DESC LIMIT 1",
                        (rfq_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        details["supplier_id"] = row[0]

                    # Attempt to retrieve negotiation targets when the table exists.
                    try:
                        cur.execute(
                            "SELECT target_price, negotiation_round FROM proc.rfq_targets WHERE rfq_id = %s ORDER BY updated_on DESC LIMIT 1",
                            (rfq_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            if row[0] is not None:
                                details["target_price"] = float(row[0])
                            if len(row) > 1 and row[1] is not None:
                                details["round"] = int(row[1])
                    except Exception:
                        # Fallback to any historic negotiation sessions to estimate the round.
                        cur.execute(
                            "SELECT COALESCE(MAX(round), 0) + 1 FROM proc.negotiation_sessions WHERE rfq_id = %s",
                            (rfq_id,),
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            details.setdefault("round", int(row[0]))
        except Exception:
            logger.exception("Failed to load RFQ metadata for %s", rfq_id)

        return details

    def _load_from_s3(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        if not self.bucket:
            logger.warning("SES inbound bucket not configured; skipping poll")
            return []

        client = self._get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        messages: List[Dict[str, object]] = []
        for page in iterator:
            contents = page.get("Contents", [])
            contents.sort(key=lambda item: item.get("LastModified"))
            for obj in contents:
                key = obj.get("Key")
                if not key:
                    continue
                if self.state_store and key in self.state_store:
                    continue
                raw = self._download_object(client, key)
                if raw is None:
                    continue
                parsed = self._parse_email(raw)
                parsed["id"] = key
                messages.append(parsed)
                if limit is not None and len(messages) >= limit:
                    return messages
        return messages

    def _download_object(self, client, key: str) -> Optional[bytes]:
        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception:  # pragma: no cover - network/runtime
            logger.exception("Failed to download SES message %s", key)
            return None

    def _parse_email(self, raw_bytes: bytes) -> Dict[str, object]:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject = message.get("subject", "")
        from_address = message.get("from", "")
        body = self._extract_body(message)
        rfq_id = self._extract_rfq_id(f"{subject} {body}")
        return {
            "subject": subject,
            "from": from_address,
            "body": body,
            "rfq_id": rfq_id,
            "received_at": message.get("date"),
        }

    def _extract_body(self, message) -> str:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                if part.get_content_type() == "text/plain":
                    try:
                        return part.get_content()
                    except Exception:
                        payload = part.get_payload(decode=True) or b""
                        return payload.decode(errors="ignore")
            for part in message.walk():
                if part.get_content_type() == "text/html":
                    try:
                        html = part.get_content()
                    except Exception:
                        html = (part.get_payload(decode=True) or b"").decode(errors="ignore")
                    return _strip_html(html)
        try:
            return message.get_content()
        except Exception:
            payload = message.get_payload(decode=True) or b""
            return payload.decode(errors="ignore")

    def _extract_rfq_id(self, text: str) -> Optional[str]:
        pattern = getattr(self.supplier_agent, "RFQ_PATTERN", SupplierInteractionAgent.RFQ_PATTERN)
        if not pattern:
            return None
        match = pattern.search(text)
        return match.group(0) if match else None

    def _get_s3_client(self):
        if self._s3_client is None:
            client_kwargs = {}
            if self.region:
                client_kwargs["region_name"] = self.region
            self._s3_client = boto3.client("s3", **client_kwargs)
        return self._s3_client

    @staticmethod
    def _parse_region(endpoint: str) -> Optional[str]:
        match = re.search(r"email-smtp\.([a-z0-9-]+)\.amazonaws.com", endpoint)
        if match:
            return match.group(1)
        return None

