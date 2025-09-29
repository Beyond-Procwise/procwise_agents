from __future__ import annotations

import gzip
import logging
import re
import time
import uuid
from urllib.parse import unquote_plus, urlparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import boto3

try:  # pragma: no cover - optional dependency during tests
    from psycopg2 import errors as psycopg2_errors
except Exception:  # pragma: no cover - psycopg2 may be unavailable in tests
    psycopg2_errors = None  # type: ignore[assignment]

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

    def items(self) -> Iterable[Tuple[str, Dict[str, object]]]:  # pragma: no cover - protocol
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

    def items(self) -> Iterable[Tuple[str, Dict[str, object]]]:
        return list(self._seen.items())


@dataclass
class S3ObjectWatcher:
    """Remember previously observed keys for an S3 prefix."""

    limit: int = 512
    known: "OrderedDict[str, datetime]" = field(default_factory=OrderedDict)

    def is_new(self, key: str) -> bool:
        return bool(key) and key not in self.known

    def mark_known(self, key: str, last_modified: Optional[datetime]) -> None:
        if not key:
            return
        timestamp = last_modified if isinstance(last_modified, datetime) else datetime.now(timezone.utc)
        self.known[key] = timestamp
        self.known.move_to_end(key)
        while self.limit > 0 and len(self.known) > self.limit:
            self.known.popitem(last=False)


def _strip_html(value: str) -> str:
    """Convert HTML content to a whitespace normalised string."""

    if not value:
        return ""

    # Preserve hidden RFQ annotations injected as HTML comments so that
    # downstream RFQ pattern matching still works even when the supplier
    # replies without keeping the identifier in the visible text.
    comment_matches = re.findall(
        r"<!--\s*RFQ-ID\s*:\s*([A-Za-z0-9_-]+)\s*-->",
        value,
        flags=re.IGNORECASE,
    )

    # Basic tag removal keeps the implementation lightweight without
    # introducing heavy dependencies such as BeautifulSoup for tests.
    cleaned = re.sub(
        r"<\s*(script|style).*?>.*?<\s*/\s*\1\s*>",
        " ",
        value,
        flags=re.I | re.S,
    )
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    if comment_matches:
        # Prepend extracted identifiers to ensure they survive whitespace
        # normalisation and can be detected by the RFQ regex.
        return " ".join(comment_matches + [cleaned.strip()]).strip()

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

        self.mailbox_address = (
            getattr(self.settings, "supplier_mailbox", None)
            or getattr(self.settings, "imap_user", None)
            or "supplierconnect@procwise.co.uk"
        )

        endpoint = getattr(self.settings, "ses_smtp_endpoint", "")
        self.region = getattr(self.settings, "ses_region", None) or self._parse_region(endpoint)

        default_bucket = "procwisemvp"
        default_prefix = "emails/"

        uri_bucket: Optional[str] = None
        uri_prefix: Optional[str] = None
        inbound_s3_uri = getattr(self.settings, "ses_inbound_s3_uri", None)
        if inbound_s3_uri:
            try:
                uri_bucket, uri_prefix = self._parse_s3_uri(inbound_s3_uri)
            except ValueError:
                logger.warning(
                    "Invalid SES inbound S3 URI %r; falling back to default bucket and prefix",
                    inbound_s3_uri,
                )

        configured_bucket = getattr(self.settings, "ses_inbound_bucket", None)
        if configured_bucket and uri_bucket and configured_bucket != uri_bucket:
            logger.warning(
                "SES inbound bucket %s overrides bucket %s derived from S3 URI",
                configured_bucket,
                uri_bucket,
            )

        bucket_candidates = [
            configured_bucket,
            uri_bucket,
            getattr(self.settings, "s3_bucket_name", None),
            default_bucket,
        ]
        self.bucket = next((candidate for candidate in bucket_candidates if candidate), default_bucket)

        raw_prefix = getattr(self.settings, "ses_inbound_prefix", None)
        if raw_prefix is None:
            raw_prefix = uri_prefix or default_prefix
        else:
            text_prefix = str(raw_prefix).strip()
            if uri_prefix and text_prefix in {"", default_prefix}:
                raw_prefix = uri_prefix
            elif not text_prefix:
                raw_prefix = uri_prefix or default_prefix

        prefix_value = str(raw_prefix or default_prefix)
        self._prefixes = [self._ensure_trailing_slash(prefix_value)]

        # Ensure supporting negotiation tables exist so metadata lookups do
        # not fail on freshly provisioned databases.  The statements are safe
        # to run repeatedly thanks to ``IF NOT EXISTS`` guards.
        self._ensure_negotiation_tables()

        self._validate_inbound_configuration(prefix_value)

        # Prefer reusing the orchestrator's S3 client so we respect any
        # endpoint overrides or credential sources initialised during startup.
        shared_client = getattr(agent_nick, "s3_client", None)
        self._s3_client = shared_client if shared_client is not None else None
        self._s3_role_arn = (
            getattr(self.settings, "ses_inbound_role_arn", None)
            or getattr(self.settings, "ses_secret_role_arn", None)
        )
        if self._s3_role_arn:
            # When a dedicated role is configured we assume it lazily on the
            # first request to guarantee the mailbox bucket permissions are in
            # place even if a shared client exists without the required rights.
            self._s3_client = None

        self._assumed_credentials: Optional[Dict[str, str]] = None
        self._assumed_credentials_expiry: Optional[datetime] = None
        self._s3_watch_history_limit = self._coerce_int(
            getattr(self.settings, "email_s3_watch_history_limit", 512),
            default=512,
            minimum=10,
        )
        self._s3_prefix_watchers: Dict[str, S3ObjectWatcher] = {}
        self._processed_cache_limit = self._coerce_int(
            getattr(self.settings, "email_processed_cache_limit", 200),
            default=200,
            minimum=1,
        )
        self._processed_cache: OrderedDict[str, Dict[str, object]] = OrderedDict()
        self._match_poll_attempts = self._coerce_int(
            getattr(self.settings, "email_match_poll_attempts", 3),
            default=3,
            minimum=1,
        )

        logger.info(
            "Initialised email watcher for mailbox %s using S3 bucket %s and prefix %s (poll_interval=%ss)",
            self.mailbox_address,
            self.bucket,
            ", ".join(self._prefixes),
            self.poll_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def poll_once(
        self,
        limit: Optional[int] = None,
        *,
        match_filters: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
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

        results: List[Dict[str, object]] = []
        match_found = False
        attempt = 0
        max_attempts = self._match_poll_attempts if match_filters else 1

        logger.info(
            "Scanning S3 bucket %s/%s for new supplier responses (limit=%s)",
            self.bucket,
            ", ".join(self._prefixes) or "<root>",
            limit if limit is not None else "unbounded",
        )

        if match_filters:
            cached_matches = self._retrieve_cached_matches(match_filters, limit)
            if cached_matches:
                logger.info(
                    "Returning %d previously processed message(s) matching filters for mailbox %s",
                    len(cached_matches),
                    self.mailbox_address,
                )
                return cached_matches

        while attempt < max_attempts:
            attempt += 1
            messages: List[Dict[str, object]] = []
            try:
                if self._custom_loader is not None:
                    messages = self._custom_loader(limit)
                else:
                    messages = self._load_messages(limit, mark_seen=True)
            except Exception:  # pragma: no cover - network/runtime
                logger.exception("Failed to load inbound SES messages")
                if match_filters and attempt < max_attempts:
                    if self.poll_interval_seconds > 0:
                        time.sleep(self.poll_interval_seconds)
                    continue
                return results

            logger.info(
                "Retrieved %d new message(s) for mailbox %s on attempt %d/%d",
                len(messages),
                self.mailbox_address,
                attempt,
                max_attempts,
            )

            if not messages and match_filters and attempt < max_attempts:
                logger.debug(
                    "No messages returned for mailbox %s on attempt %d/%d; retrying",
                    self.mailbox_address,
                    attempt,
                    max_attempts,
                )
                if self.poll_interval_seconds > 0:
                    time.sleep(self.poll_interval_seconds)
                continue

            batch_match_found = False
            for message in messages:
                message_id = str(message.get("id") or uuid.uuid4())
                if self.state_store and message_id in self.state_store:
                    logger.debug(
                        "Skipping message %s for mailbox %s as it was already processed",
                        message_id,
                        self.mailbox_address,
                    )
                    continue

                try:
                    processed, reason = self._process_message(message)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Failed to process SES message %s", message_id)
                    processed, reason = None, "processing_error"

                metadata: Dict[str, object]
                message_match = False
                if processed:
                    logger.info(
                        "Processed message %s for RFQ %s from %s",
                        message_id,
                        processed.get("rfq_id"),
                        processed.get("from_address"),
                    )
                    processed_payload = self._record_processed_payload(message_id, processed)
                    metadata = {
                        "rfq_id": processed_payload.get("rfq_id"),
                        "supplier_id": processed_payload.get("supplier_id"),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "status": "processed",
                        "payload": processed_payload,
                    }
                    message_match = bool(
                        match_filters and self._matches_filters(processed_payload, match_filters)
                    )
                else:
                    logger.warning(
                        "Skipped message %s for mailbox %s: %s",
                        message_id,
                        self.mailbox_address,
                        reason or "unknown",
                    )
                    metadata = {
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "status": "skipped",
                        "reason": reason or "unknown",
                    }

                if self.state_store and reason != "processing_error":
                    self.state_store.add(message_id, metadata)

                if processed and (not match_filters or message_match):
                    results.append(processed_payload)

                if message_match:
                    match_found = True
                    batch_match_found = True
                    logger.debug(
                        "Stopping poll once after matching filters for message %s", message_id
                    )
                    break

            if batch_match_found or not match_filters:
                break

            if attempt < max_attempts:
                logger.debug(
                    "No messages matched filters for mailbox %s on attempt %d/%d; waiting %ss before retry",
                    self.mailbox_address,
                    attempt,
                    max_attempts,
                    self.poll_interval_seconds,
                )
                if self.poll_interval_seconds > 0:
                    time.sleep(self.poll_interval_seconds)

        if match_filters and not match_found:
            logger.info(
                "Exhausted %d attempt(s) polling mailbox %s without matching filters",
                attempt,
                self.mailbox_address,
            )

        return results

    def _record_processed_payload(
        self, message_id: str, payload: Dict[str, object]
    ) -> Dict[str, object]:
        """Persist processed payload snapshots for future filter matches."""

        snapshot = deepcopy(payload)
        if "message_id" not in snapshot:
            snapshot["message_id"] = message_id
        self._processed_cache[message_id] = snapshot
        while len(self._processed_cache) > self._processed_cache_limit:
            self._processed_cache.popitem(last=False)
        return deepcopy(snapshot)

    def _retrieve_cached_matches(
        self, filters: Dict[str, object], limit: Optional[int]
    ) -> List[Dict[str, object]]:
        if not filters:
            return []

        try:
            effective_limit = None if limit is None else max(int(limit), 0)
        except Exception:
            effective_limit = None if limit is None else 0

        if effective_limit == 0:
            return []

        matches: List[Dict[str, object]] = []
        seen_ids: Set[str] = set()

        def _consider(message_id: str, payload: Dict[str, object]) -> None:
            if not isinstance(payload, dict) or message_id in seen_ids:
                return
            if self._matches_filters(payload, filters):
                matches.append(deepcopy(payload))
                seen_ids.add(message_id)

        for message_id in reversed(list(self._processed_cache.keys())):
            payload = self._processed_cache.get(message_id)
            if payload:
                _consider(message_id, payload)
            if effective_limit is not None and len(matches) >= effective_limit:
                return matches[:effective_limit]

        for message_id, metadata in self._iter_state_metadata():
            payload = metadata.get("payload") if isinstance(metadata, dict) else None
            if isinstance(payload, dict):
                _consider(message_id, payload)
            if effective_limit is not None and len(matches) >= effective_limit:
                return matches[:effective_limit]

        if effective_limit is not None:
            return matches[:effective_limit]
        return matches

    def _iter_state_metadata(self) -> Iterable[Tuple[str, Dict[str, object]]]:
        if not self.state_store:
            return []
        try:
            items = self.state_store.items()
        except Exception:
            logger.debug("Email watcher state store does not expose items(); skipping cache lookup")
            return []
        return list(items)

    def watch(
        self,
        *,
        interval: Optional[int] = None,
        limit: Optional[int] = None,
        stop_after: Optional[int] = None,
        timeout_seconds: Optional[int] = 900,
    ) -> int:
        """Continuously poll for messages until ``stop_after`` iterations."""

        iterations = 0
        processed_total = 0
        poll_delay = self.poll_interval_seconds if interval is None else max(interval, 1)
        start_time = time.time()
        while True:
            batch = self.poll_once(limit=limit)
            processed_total += len(batch)
            iterations += 1
            logger.info(
                "Email watcher iteration %d processed %d message(s); total processed=%d",
                iterations,
                len(batch),
                processed_total,
            )
            if batch:
                logger.info(
                    "Email watcher exiting after processing %d new message(s) in iteration %d",
                    len(batch),
                    iterations,
                )
                break
            if stop_after is not None and iterations >= stop_after:
                break
            if timeout_seconds is not None and timeout_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.info(
                        "Email watcher reached timeout after %.1f seconds without new messages",
                        elapsed,
                    )
                    break
            time.sleep(poll_delay)
        return processed_total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def peek_recent_messages(self, limit: int = 3) -> List[Dict[str, object]]:
        """Return a non-destructive preview of the most recent inbound emails."""

        try:
            limit_int = max(0, int(limit))
        except Exception:
            limit_int = 3

        if limit_int == 0:
            return []

        if self._custom_loader is not None:
            messages = self._custom_loader(limit_int)
        else:
            messages = self._load_messages(limit_int, mark_seen=False)

        preview: List[Dict[str, object]] = []
        for message in messages[:limit_int]:
            body = str(message.get("body") or "")
            snippet = re.sub(r"\s+", " ", body).strip()
            preview.append(
                {
                    "id": message.get("id"),
                    "subject": message.get("subject"),
                    "from": message.get("from"),
                    "rfq_id": message.get("rfq_id"),
                    "received_at": message.get("received_at"),
                    "snippet": snippet[:160],
                }
            )
        return preview

    # ------------------------------------------------------------------
    # Database bootstrapping helpers
    # ------------------------------------------------------------------
    def _ensure_negotiation_tables(self) -> None:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.rfq_targets (
                            rfq_id TEXT NOT NULL,
                            supplier_id TEXT,
                            target_price NUMERIC(18, 2),
                            negotiation_round INTEGER NOT NULL DEFAULT 1,
                            notes TEXT,
                            created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            CONSTRAINT rfq_targets_pk PRIMARY KEY (rfq_id, negotiation_round)
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.negotiation_sessions (
                            rfq_id TEXT NOT NULL,
                            supplier_id TEXT NOT NULL,
                            round INTEGER NOT NULL,
                            counter_offer NUMERIC(18, 2),
                            created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            CONSTRAINT negotiation_sessions_pk PRIMARY KEY (rfq_id, supplier_id, round)
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS negotiation_sessions_rfq_supplier_idx
                            ON proc.negotiation_sessions (rfq_id, supplier_id)
                        """
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to ensure negotiation support tables exist")

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
        if interaction_output.status != AgentStatus.SUCCESS:
            error_detail = interaction_output.error
            if not error_detail and isinstance(interaction_output.data, dict):
                error_detail = interaction_output.data.get("error")
            logger.error(
                "Supplier interaction failed for RFQ %s via mailbox %s: %s",
                rfq_id,
                self.mailbox_address,
                error_detail or "unknown_error",
            )
            result = {
                "rfq_id": rfq_id,
                "supplier_id": supplier_id,
                "message_id": message.get("id"),
                "subject": subject,
                "from_address": from_address,
                "message_body": body,
                "target_price": target_price,
                "negotiation_triggered": False,
                "supplier_status": interaction_output.status.value,
                "negotiation_status": None,
                "supplier_output": interaction_output.data,
                "negotiation_output": None,
                "error": error_detail,
            }
            return result, "supplier_interaction_failed"
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
            logger.info(
                "Negotiation triggered for RFQ %s (round %s) via mailbox %s; status=%s",
                rfq_id,
                negotiation_round,
                self.mailbox_address,
                negotiation_output.status.value,
            )

        result = {
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "message_id": message.get("id"),
            "subject": subject,
            "from_address": from_address,
            "message_body": body,
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
                    except Exception as exc:
                        # Reset the transaction so that subsequent queries remain usable even
                        # when the rfq_targets table is absent (older schemas) or another
                        # transient issue occurs.
                        try:
                            conn.rollback()
                        except Exception:  # pragma: no cover - defensive
                            logger.debug("Rollback failed after rfq target lookup for %s", rfq_id)

                        if psycopg2_errors and isinstance(exc, psycopg2_errors.UndefinedTable):
                            logger.debug(
                                "RFQ targets table missing; skipping direct target metadata for %s",
                                rfq_id,
                            )
                        else:
                            logger.exception("Failed to load rfq_targets metadata for %s", rfq_id)

                        # Fallback to any historic negotiation sessions to estimate the round.
                        with conn.cursor() as fallback_cur:
                            fallback_cur.execute(
                                "SELECT COALESCE(MAX(round), 0) + 1 FROM proc.negotiation_sessions WHERE rfq_id = %s",
                                (rfq_id,),
                            )
                            row = fallback_cur.fetchone()
                            if row and row[0]:
                                details.setdefault("round", int(row[0]))
        except Exception:
            logger.exception("Failed to load RFQ metadata for %s", rfq_id)

        return details

    def _load_messages(self, limit: Optional[int], *, mark_seen: bool) -> List[Dict[str, object]]:
        del mark_seen  # unused in S3-only polling

        if limit is not None:
            try:
                effective_limit = max(int(limit), 0)
            except Exception:
                effective_limit = 0
            if effective_limit == 0:
                return []
        else:
            effective_limit = None

        if not self.bucket:
            logger.warning("S3 bucket not configured; unable to load inbound messages")
            return []

        return self._load_from_s3(effective_limit)

    @staticmethod
    def _matches_filters(payload: Dict[str, object], filters: Dict[str, object]) -> bool:
        if not filters:
            return False

        def _normalise(value: object) -> Optional[str]:
            if value is None:
                return None
            try:
                text = str(value).strip()
            except Exception:
                return None
            lowered = text.lower()
            return lowered or None

        payload_rfq = _normalise(payload.get("rfq_id"))
        payload_supplier = _normalise(payload.get("supplier_id"))
        payload_subject = _normalise(payload.get("subject")) or ""
        payload_sender = _normalise(payload.get("from_address"))
        payload_message = _normalise(payload.get("message_id")) or _normalise(payload.get("id"))

        for key, expected in filters.items():
            if expected in (None, ""):
                continue
            if key == "rfq_id":
                if payload_rfq != _normalise(expected):
                    return False
            elif key == "supplier_id":
                if payload_supplier != _normalise(expected):
                    return False
            elif key == "from_address":
                if payload_sender != _normalise(expected):
                    return False
            elif key == "subject_contains":
                needle = _normalise(expected)
                if needle and needle not in payload_subject:
                    return False
            elif key == "message_id":
                if payload_message != _normalise(expected):
                    return False
        return True

    def _load_from_s3(
        self,
        limit: Optional[int] = None,
        *,
        prefixes: Optional[Sequence[str]] = None,
        parser: Optional[Callable[[bytes], Dict[str, object]]] = None,
        newest_first: bool = False,
    ) -> List[Dict[str, object]]:
        if not self.bucket:
            logger.warning("SES inbound bucket not configured; skipping poll")
            return []

        client = self._get_s3_client()
        logger.debug(
            "Listing inbound emails from bucket=%s, prefixes=%s for mailbox %s",
            self.bucket,
            ", ".join(prefixes) if prefixes is not None else ", ".join(self._prefixes),
            self.mailbox_address,
        )
        paginator = client.get_paginator("list_objects_v2")

        collected: List[Tuple[Optional[datetime], Dict[str, object]]] = []
        seen_keys = set()

        active_prefixes = list(prefixes) if prefixes is not None else list(self._prefixes)
        parser_fn = parser or self._parse_email

        for prefix in active_prefixes:
            watcher = self._s3_prefix_watchers.get(prefix)
            if watcher is None:
                watcher = S3ObjectWatcher(limit=self._s3_watch_history_limit)
                self._s3_prefix_watchers[prefix] = watcher
            iterator = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            for page in iterator:
                contents = page.get("Contents", [])
                contents.sort(key=lambda item: item.get("LastModified"))
                for obj in contents:
                    key = obj.get("Key")
                    if not key or key in seen_keys:
                        continue
                    last_modified_raw = obj.get("LastModified")
                    last_modified = (
                        last_modified_raw
                        if isinstance(last_modified_raw, datetime)
                        else None
                    )
                    if self.state_store and key in self.state_store:
                        watcher.mark_known(key, last_modified)
                        continue
                    if not watcher.is_new(key):
                        continue
                    raw = self._download_object(client, key, bucket=self.bucket)
                    if raw is None:
                        continue
                    parsed = parser_fn(raw)
                    parsed["id"] = key
                    collected.append((last_modified, parsed))
                    seen_keys.add(key)
                    watcher.mark_known(key, last_modified)
                    logger.debug(
                        "Queued message %s for processing from mailbox %s",
                        key,
                        self.mailbox_address,
                    )
                    if limit is not None and len(collected) >= limit:
                        break
                if limit is not None and len(collected) >= limit:
                    break
            if limit is not None and len(collected) >= limit:
                break

        collected.sort(key=lambda item: item[0] or datetime.min, reverse=newest_first)
        messages = [payload for _, payload in collected]
        return messages

    def _download_object(
        self, client, key: str, *, bucket: Optional[str] = None
    ) -> Optional[bytes]:
        bucket_name = bucket or self.bucket
        if not bucket_name:
            logger.warning(
                "Attempted to download %s without an S3 bucket configured for mailbox %s",
                key,
                self.mailbox_address,
            )
            return None
        try:
            response = client.get_object(Bucket=bucket_name, Key=key)

            body = response["Body"].read()
            encoding = str(response.get("ContentEncoding", "") or "").lower()
            key_lower = key.lower()
            if encoding == "gzip" or key_lower.endswith(".gz"):
                try:
                    body = gzip.decompress(body)
                except OSError:
                    logger.warning(
                        "Failed to decompress gzip payload for %s; using raw bytes",
                        key,
                    )
            return body
        except Exception:  # pragma: no cover - network/runtime
            logger.exception(
                "Failed to download SES message %s from bucket %s",
                key,
                bucket_name,
            )
            return None

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

    def _extract_attachments(self, message) -> List[Dict[str, object]]:
        attachments: List[Dict[str, object]] = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            disposition = part.get_content_disposition()
            if disposition not in ("attachment", "inline"):
                continue
            filename = part.get_filename()
            try:
                payload = part.get_payload(decode=True) or b""
            except Exception:
                payload = b""
            attachments.append(
                {
                    "filename": filename,
                    "content_type": part.get_content_type(),
                    "content": payload,
                    "size": len(payload),
                    "disposition": disposition,
                }
            )
        return attachments

    def _extract_rfq_id(self, text: str) -> Optional[str]:
        pattern = getattr(self.supplier_agent, "RFQ_PATTERN", SupplierInteractionAgent.RFQ_PATTERN)
        if not pattern:
            return None
        match = pattern.search(text)
        return match.group(0) if match else None

    def _get_s3_client(self):
        if self._s3_client is not None:
            return self._s3_client

        shared_client = getattr(self.agent_nick, "s3_client", None)
        if shared_client is not None and self._s3_role_arn is None:
            self._s3_client = shared_client
            return self._s3_client

        client_kwargs = {}
        if self.region:
            client_kwargs["region_name"] = self.region

        if self._s3_role_arn:
            try:
                self._s3_client = self._s3_client_with_assumed_role(client_kwargs)
                return self._s3_client
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Unable to assume role %s for inbound S3 access; falling back to default credentials",
                    self._s3_role_arn,
                )

        self._s3_client = boto3.client("s3", **client_kwargs)
        return self._s3_client

    def _s3_client_with_assumed_role(self, client_kwargs):
        credentials = self._assume_role_credentials()
        return boto3.client("s3", **credentials, **client_kwargs)

    def _assume_role_credentials(self) -> Dict[str, str]:
        if not self._s3_role_arn:
            raise ValueError("No role ARN configured for assumption")

        now = datetime.now(timezone.utc)
        if (
            self._assumed_credentials
            and self._assumed_credentials_expiry
            and self._assumed_credentials_expiry - timedelta(minutes=5) > now
        ):
            return dict(self._assumed_credentials)


        sts_kwargs = {"region_name": self.region} if self.region else {}
        sts_client = boto3.client("sts", **sts_kwargs)
        session_name = f"ProcWiseEmailWatcher-{uuid.uuid4().hex[:8]}"
        response = sts_client.assume_role(
            RoleArn=self._s3_role_arn,
            RoleSessionName=session_name,
        )
        credentials = response.get("Credentials")
        if not credentials:
            raise ValueError("AssumeRole response missing credentials")

        access_key = credentials.get("AccessKeyId")
        secret_key = credentials.get("SecretAccessKey")
        token = credentials.get("SessionToken")
        expiration = credentials.get("Expiration")

        if not all([access_key, secret_key, token]):
            raise ValueError("Incomplete credentials returned from AssumeRole")

        expiry_dt: Optional[datetime] = None
        if isinstance(expiration, datetime):
            if expiration.tzinfo is None:
                expiry_dt = expiration.replace(tzinfo=timezone.utc)
            else:
                expiry_dt = expiration.astimezone(timezone.utc)

        if expiry_dt is None:
            expiry_dt = now + timedelta(hours=1)

        self._assumed_credentials = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "aws_session_token": token,
        }
        self._assumed_credentials_expiry = expiry_dt
        return dict(self._assumed_credentials)

    @staticmethod
    def _parse_region(endpoint: str) -> Optional[str]:
        match = re.search(r"email-smtp\.([a-z0-9-]+)\.amazonaws.com", endpoint)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _coerce_int(value, *, default: int, minimum: Optional[int] = None) -> int:
        try:
            coerced = int(value)
        except Exception:
            coerced = default
        if minimum is not None:
            coerced = max(coerced, minimum)
        return coerced

    def _validate_inbound_configuration(self, raw_prefix: object) -> None:
        """Emit logging about the active S3 polling configuration."""

        if not self.bucket:
            logger.warning(
                "Email watcher cannot poll for supplier replies because no S3 bucket is configured"
            )
            return

        prefix_summary = ", ".join(self._prefixes) or "<root>"
        logger.info(
            "Email watcher will poll s3://%s%s for mailbox %s",
            self.bucket,
            prefix_summary,
            self.mailbox_address,
        )

        if raw_prefix is None:
            logger.info(
                "No custom SES prefix supplied; defaulting to %s",
                prefix_summary,
            )

    @staticmethod
    def _ensure_trailing_slash(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return ""
        return text if text.endswith("/") else f"{text}/"

    @staticmethod
    def _parse_s3_uri(uri: str) -> Tuple[str, Optional[str]]:
        """Parse an S3 URI into bucket and prefix components."""

        if not uri:
            raise ValueError("S3 URI must be a non-empty string")

        parsed = urlparse(str(uri).strip())
        if parsed.scheme and parsed.scheme.lower() != "s3":
            raise ValueError(f"Unsupported scheme for S3 URI: {parsed.scheme}")

        bucket = parsed.netloc or parsed.path.split("/", 1)[0]
        if not bucket:
            raise ValueError("S3 URI must include a bucket name")

        prefix = parsed.path[1:] if parsed.path.startswith("/") else parsed.path
        if parsed.netloc and prefix.startswith(bucket):
            # Handle URIs like s3://bucket/bucket/... produced by naive joins.
            prefix = prefix[len(bucket) :].lstrip("/")

        prefix = prefix.strip()
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"

        return bucket, prefix or None

