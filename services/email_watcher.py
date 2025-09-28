from __future__ import annotations

import base64
import contextlib
import gzip
import imaplib
import json
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

        uri_bucket: Optional[str] = None
        uri_prefix: Optional[str] = None
        inbound_s3_uri = getattr(self.settings, "ses_inbound_s3_uri", None)
        if inbound_s3_uri:
            try:
                uri_bucket, uri_prefix = self._parse_s3_uri(inbound_s3_uri)
            except ValueError:
                logger.warning(
                    "Invalid SES inbound S3 URI %r; falling back to explicit bucket/prefix configuration",
                    inbound_s3_uri,
                )

        configured_bucket = getattr(self.settings, "ses_inbound_bucket", None)
        if configured_bucket and uri_bucket and configured_bucket != uri_bucket:
            logger.warning(
                "SES inbound bucket %s overrides bucket %s derived from S3 URI",
                configured_bucket,
                uri_bucket,
            )

        self.bucket = configured_bucket or uri_bucket or getattr(
            self.settings, "s3_bucket_name", None
        )

        raw_prefix = getattr(self.settings, "ses_inbound_prefix", None)
        if raw_prefix is None:
            raw_prefix = uri_prefix or "ses/inbound/"
        else:
            text_prefix = str(raw_prefix).strip()
            if uri_prefix and text_prefix in {"", "ses/inbound/"}:
                raw_prefix = uri_prefix
            elif not text_prefix:
                raw_prefix = uri_prefix or ""
        self._prefixes = self._build_prefixes(raw_prefix)
        processed_prefix_setting = getattr(self.settings, "ses_processed_prefix", "emails/")
        self._processed_prefixes = self._build_prefixes(processed_prefix_setting)

        # Ensure supporting negotiation tables exist so metadata lookups do
        # not fail on freshly provisioned databases.  The statements are safe
        # to run repeatedly thanks to ``IF NOT EXISTS`` guards.
        self._ensure_negotiation_tables()

        # Optional IMAP fallback configuration for mailboxes that are not
        # integrated with SES inbound rulesets (e.g. Microsoft 365 shared
        # mailboxes). All attributes default to ``None`` or sensible values so
        # existing deployments continue to rely on SES unless explicitly
        # configured otherwise.
        self._imap_host = getattr(self.settings, "imap_host", None)
        self._imap_user = getattr(self.settings, "imap_user", None)
        self._imap_password = getattr(self.settings, "imap_password", None)
        self._imap_port = self._coerce_int(
            getattr(self.settings, "imap_port", 993), default=993, minimum=1
        )
        self._imap_folder = getattr(self.settings, "imap_folder", "INBOX") or "INBOX"
        self._imap_search = getattr(self.settings, "imap_search_criteria", "UNSEEN") or "UNSEEN"
        self._imap_use_ssl = bool(getattr(self.settings, "imap_use_ssl", True))
        self._imap_mark_seen = bool(getattr(self.settings, "imap_mark_seen", True))
        self._imap_batch_size = self._coerce_int(
            getattr(self.settings, "imap_batch_size", 25), default=25, minimum=1
        )

        self._queue_url = getattr(self.settings, "ses_inbound_queue_url", None)
        queue_flag = getattr(self.settings, "ses_inbound_queue_enabled", None)
        self._queue_disabled_reason: Optional[str] = None
        if queue_flag is None:
            # Default to S3-based polling even when an SQS queue URL is configured.
            # Production infrastructure currently relies on processed email snapshots
            # written to S3 and no queue is provisioned, so attempting to contact SQS
            # only yields noisy errors.  Operators can explicitly opt back in by
            # setting ``ses_inbound_queue_enabled=True`` once the queue is restored.
            self._queue_enabled = False
            if self._queue_url:
                self._queue_disabled_reason = (
                    "temporarily disabled while inbound processing relies on processed S3 emails"
                )
        else:
            self._queue_enabled = bool(queue_flag)
            if self._queue_url and not self._queue_enabled:
                self._queue_disabled_reason = (
                    "disabled via configuration; relying on processed S3 emails"
                )
        if self._queue_url and not self._queue_enabled:
            logger.info(
                "SES inbound queue %s configured for mailbox %s but polling is disabled (%s)",
                self._queue_url,
                self.mailbox_address,
                self._queue_disabled_reason or "no reason provided",
            )
        self._queue_wait_seconds = self._coerce_int(
            getattr(self.settings, "ses_inbound_queue_wait_seconds", 2), default=2, minimum=0
        )
        self._queue_max_messages = self._coerce_int(
            getattr(self.settings, "ses_inbound_queue_max_messages", 10),
            default=10,
            minimum=1,
        )

        self._validate_inbound_configuration(raw_prefix)

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

        self._sqs_client = None
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
            "Initialised email watcher for mailbox %s (bucket=%s, prefixes=%s, processed_prefixes=%s, queue=%s, queue_enabled=%s, poll_interval=%ss)",
            self.mailbox_address,
            self.bucket,
            ", ".join(self._prefixes),
            ", ".join(self._processed_prefixes),
            self._queue_url,
            self._queue_enabled,
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
            "Polling inbound mailbox %s with limit=%s",
            self.mailbox_address,
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

            rfq_filter = match_filters.get("rfq_id") if isinstance(match_filters, dict) else None
            if rfq_filter:
                attempt_cap = max(1, min(max_attempts, 3))
                processed_results = self._poll_recent_processed_match(
                    limit=limit,
                    match_filters=match_filters,
                    attempt_cap=attempt_cap,
                )
                if processed_results:
                    return processed_results
                logger.info(
                    "Processed mailbox %s %d time(s) without matching RFQ %s",
                    self.mailbox_address,
                    attempt_cap,
                    rfq_filter,
                )
                return []

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
                    self._acknowledge_queue_message(message, success=True)
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
                    results.append(processed_payload)
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

                if message_match:
                    match_found = True
                    batch_match_found = True
                    logger.debug(
                        "Stopping poll once after matching filters for message %s", message_id
                    )
                    self._acknowledge_queue_message(message, success=True)
                    break

                ack_success = bool(processed) or (reason is not None and reason != "processing_error")
                self._acknowledge_queue_message(message, success=ack_success)

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

    def _poll_recent_processed_match(
        self,
        *,
        limit: Optional[int],
        match_filters: Dict[str, object],
        attempt_cap: int,
    ) -> List[Dict[str, object]]:
        use_loader = self._custom_loader is not None
        if not use_loader and (not self.bucket or not self._processed_prefixes):
            return []

        try:
            effective_limit = max(1, int(limit) if limit is not None else 1)
        except Exception:
            effective_limit = 1

        attempts = 0
        checked: Set[str] = set()
        results: List[Dict[str, object]] = []
        expected_rfq = str(match_filters.get("rfq_id") or "").strip().lower()

        retry_delay = min(float(self.poll_interval_seconds), 1.0)
        while attempts < attempt_cap:
            attempts += 1
            try:
                if use_loader:
                    messages = self._custom_loader(None)  # type: ignore[arg-type]
                else:
                    messages = self._load_processed_messages(None)
            except Exception:
                logger.exception("Failed to load processed SES messages")
                messages = []

            if not messages:
                if attempts < attempt_cap and retry_delay > 0:
                    time.sleep(retry_delay)
                continue

            matched = False
            for raw_message in messages:
                message_id = str(
                    raw_message.get("message_id")
                    or raw_message.get("id")
                    or uuid.uuid4()
                )
                if message_id in checked:
                    continue
                checked.add(message_id)

                payload = dict(raw_message)
                payload.setdefault("message_id", message_id)
                candidate_rfq = payload.get("rfq_id")
                if not isinstance(candidate_rfq, str) or not candidate_rfq.strip():
                    subject = str(payload.get("subject") or "")
                    body = str(payload.get("body") or "")
                    candidate_rfq = self._extract_rfq_id(f"{subject} {body}") or ""
                candidate_norm = str(candidate_rfq or "").strip().lower()
                if not candidate_norm or candidate_norm != expected_rfq:
                    logger.debug(
                        "Processed email %s did not match RFQ %s", message_id, expected_rfq or "<none>"
                    )
                    continue

                already_processed = isinstance(payload.get("supplier_output"), dict) or (
                    payload.get("negotiation_output") is not None
                )

                processed_payload: Optional[Dict[str, object]] = None
                reason: Optional[str] = None
                if already_processed:
                    processed_payload = payload
                else:
                    try:
                        processed_payload, reason = self._process_message(payload)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("Failed to process SES message %s", message_id)
                        processed_payload, reason = None, "processing_error"

                if not processed_payload:
                    logger.debug(
                        "Skipping processed email %s due to %s", message_id, reason or "unknown"
                    )
                    continue

                recorded = self._record_processed_payload(message_id, processed_payload)
                if self.state_store:
                    metadata = {
                        "rfq_id": recorded.get("rfq_id"),
                        "supplier_id": recorded.get("supplier_id"),
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "status": "processed",
                        "payload": recorded,
                    }
                    self.state_store.add(message_id, metadata)
                results.append(recorded)
                matched = True
                break

            if matched:
                break

            if attempts < attempt_cap and retry_delay > 0:
                time.sleep(retry_delay)

        return results

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
        messages: List[Dict[str, object]] = []

        remaining = None if limit is None else max(limit - len(messages), 0)
        if remaining == 0:
            return messages[: limit or None]

        if self.bucket and self._processed_prefixes:
            processed_messages = self._load_processed_messages(
                remaining if remaining else None
            )
            messages.extend(processed_messages)

        remaining = None if limit is None else max(limit - len(messages), 0)
        if remaining == 0:
            return messages[: limit or None]

        if self._queue_enabled and self._queue_url and mark_seen:
            queue_messages = self._load_from_queue(
                remaining if remaining else None, mark_seen=mark_seen
            )
            messages.extend(queue_messages)

        remaining = None if limit is None else max(limit - len(messages), 0)
        if remaining == 0:
            return messages[: limit or None]

        if self.bucket:
            s3_messages = self._load_from_s3(remaining if remaining else None)

            messages.extend(s3_messages)

        remaining = None if limit is None else max(limit - len(messages), 0)
        if remaining == 0:
            return messages[: limit or None]

        if self._imap_configured():
            imap_messages = self._load_from_imap(
                remaining if remaining is not None and remaining > 0 else None,
                mark_seen=mark_seen,
            )
            messages.extend(imap_messages)

        if limit is not None and len(messages) > limit:
            return messages[:limit]

        return messages

    def _load_processed_messages(self, limit: Optional[int]) -> List[Dict[str, object]]:
        if not self.bucket or not self._processed_prefixes:
            return []

        messages = self._load_from_s3(
            limit,
            prefixes=self._processed_prefixes,
            parser=self._parse_processed_payload,
            newest_first=True,
        )
        return messages

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

    def _load_from_queue(
        self, limit: Optional[int], *, mark_seen: bool
    ) -> List[Dict[str, object]]:
        if not self._queue_enabled or not self._queue_url:
            return []

        client = self._get_sqs_client()
        if client is None:
            return []

        s3_client = None
        messages: List[Dict[str, object]] = []

        while True:
            if limit is not None and len(messages) >= limit:
                break

            max_batch = self._queue_max_messages
            if limit is not None:
                max_batch = max(1, min(self._queue_max_messages, limit - len(messages)))

            try:
                response = client.receive_message(
                    QueueUrl=self._queue_url,
                    MaxNumberOfMessages=max_batch,
                    WaitTimeSeconds=self._queue_wait_seconds,
                    MessageAttributeNames=["All"],
                )
            except Exception:
                logger.exception(
                    "Failed to receive messages from queue %s for mailbox %s",
                    self._queue_url,
                    self.mailbox_address,
                )
                self._queue_enabled = False
                if not self._queue_disabled_reason:
                    self._queue_disabled_reason = "receive failures"
                logger.info(
                    "Disabling SES inbound queue polling for mailbox %s after receive failures", 
                    self.mailbox_address,
                )
                break

            sqs_messages = response.get("Messages", [])
            if not sqs_messages:
                break

            for sqs_message in sqs_messages:
                if limit is not None and len(messages) >= limit:
                    break

                payloads = self._extract_queue_objects(sqs_message)
                if not payloads:
                    logger.debug(
                        "Queue message %s did not contain any S3 payload references",
                        sqs_message.get("MessageId"),
                    )
                    continue

                parsed_messages: List[Dict[str, object]] = []
                had_failures = False

                for bucket_name, key in payloads:
                    bucket_name = bucket_name or self.bucket
                    if not key:
                        had_failures = True
                        continue

                    store_key = key
                    if self.state_store and store_key in self.state_store:
                        # Include a lightweight stub so the ack logic can
                        # progress even when the email was already processed.
                        parsed_messages.append({"id": store_key})
                        continue

                    if s3_client is None:
                        s3_client = self._get_s3_client()

                    raw = self._download_object(s3_client, store_key, bucket=bucket_name)
                    if raw is None:
                        had_failures = True
                        continue

                    try:
                        parsed = self._parse_email(raw)
                    except Exception:
                        had_failures = True
                        logger.exception(
                            "Failed to parse inbound email from %s for mailbox %s",
                            store_key,
                            self.mailbox_address,
                        )
                        continue

                    parsed["id"] = store_key
                    parsed_messages.append(parsed)

                if not parsed_messages:
                    # Nothing to process from this queue entry; leave it for a
                    # future attempt (visibility timeout will re-expose it).
                    continue

                ack_context = {
                    "receipt_handle": sqs_message.get("ReceiptHandle"),
                    "queue_url": self._queue_url,
                    "pending": len(parsed_messages),
                    "failed": had_failures,
                }

                for parsed in parsed_messages:
                    parsed["_queue_ack"] = ack_context
                    messages.append(parsed)

            if limit is None:
                # Allow another long poll to run in the same cycle to drain the
                # queue quickly when burst traffic arrives.
                continue

            if len(messages) >= limit:
                break

        return messages

    def _extract_queue_objects(
        self, sqs_message: Dict[str, object]
    ) -> List[Tuple[Optional[str], str]]:
        objects: List[Tuple[Optional[str], str]] = []
        seen: Set[Tuple[Optional[str], str]] = set()

        body = sqs_message.get("Body") if isinstance(sqs_message, dict) else None
        for bucket, key in self._parse_sqs_payload(body):
            if not key:
                continue
            decoded_key = unquote_plus(key)
            bucket_name = bucket or self.bucket
            identifier = (bucket_name, decoded_key)
            if identifier in seen:
                continue
            seen.add(identifier)
            objects.append((bucket_name, decoded_key))

        if objects:
            return objects

        attributes = sqs_message.get("MessageAttributes") if isinstance(sqs_message, dict) else None
        if isinstance(attributes, dict):
            bucket_attr = attributes.get("bucket") or attributes.get("Bucket")
            key_attr = attributes.get("key") or attributes.get("Key")
            bucket_name = None
            if isinstance(bucket_attr, dict):
                bucket_name = bucket_attr.get("StringValue")
            key_value = None
            if isinstance(key_attr, dict):
                key_value = key_attr.get("StringValue")
            if key_value:
                objects.append((bucket_name or self.bucket, unquote_plus(key_value)))

        return objects

    def _parse_sqs_payload(
        self, payload: Optional[object]
    ) -> Iterable[Tuple[Optional[str], str]]:
        if payload is None:
            return []

        stack: List[object] = [payload]
        results: List[Tuple[Optional[str], str]] = []

        while stack:
            current = stack.pop()
            if current is None:
                continue
            if isinstance(current, str):
                text = current.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except ValueError:
                    results.append((None, text))
                    continue
                stack.append(parsed)
                continue

            if isinstance(current, dict):
                records = current.get("Records")
                if isinstance(records, list):
                    stack.extend(records)

                message = current.get("Message")
                if message is not None:
                    stack.append(message)

                receipt = current.get("receipt")
                if isinstance(receipt, dict):
                    action = receipt.get("action")
                    if isinstance(action, dict):
                        bucket_name = (
                            action.get("bucketName")
                            or action.get("bucket")
                            or action.get("s3BucketName")
                        )
                        object_key = (
                            action.get("objectKey")
                            or action.get("key")
                            or action.get("s3ObjectKey")
                        )
                        if object_key:
                            results.append((bucket_name, object_key))

                action = current.get("action")
                if isinstance(action, dict):
                    bucket_name = (
                        action.get("bucketName")
                        or action.get("bucket")
                        or action.get("s3BucketName")
                    )
                    object_key = (
                        action.get("objectKey")
                        or action.get("key")
                        or action.get("s3ObjectKey")
                    )
                    if object_key:
                        results.append((bucket_name, object_key))

                s3_section = current.get("s3")
                if isinstance(s3_section, dict):
                    bucket = s3_section.get("bucket")
                    bucket_name = None
                    if isinstance(bucket, dict):
                        bucket_name = bucket.get("name")
                    elif isinstance(bucket, str):
                        bucket_name = bucket
                    obj = s3_section.get("object")
                    key_value = obj.get("key") if isinstance(obj, dict) else None
                    if key_value:
                        results.append((bucket_name, key_value))

                bucket_name = (
                    current.get("bucketName")
                    or current.get("bucket")
                    or current.get("s3BucketName")
                )
                object_key = (
                    current.get("objectKey")
                    or current.get("key")
                    or current.get("s3ObjectKey")
                )
                if object_key:
                    results.append((bucket_name, object_key))

                for value in current.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                continue

            if isinstance(current, list):
                stack.extend(current)

        return results


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

    def _load_from_imap(
        self,
        limit: Optional[int] = None,
        *,
        mark_seen: bool = False,
    ) -> List[Dict[str, object]]:
        if not self._imap_configured():
            logger.debug(
                "IMAP mailbox not configured; skipping fallback for %s",
                self.mailbox_address,
            )
            return []

        mailbox = self._imap_folder
        search_criteria = self._imap_search
        fetch_cap = limit if limit is not None else self._imap_batch_size

        messages: List[Dict[str, object]] = []
        try:
            if self._imap_use_ssl:
                client = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
            else:
                client = imaplib.IMAP4(self._imap_host, self._imap_port)
        except Exception:
            logger.exception(
                "Failed to connect to IMAP server %s for mailbox %s",
                self._imap_host,
                self.mailbox_address,
            )
            return []

        try:
            client.login(self._imap_user, self._imap_password)
        except Exception:
            logger.exception(
                "Failed to authenticate to IMAP mailbox %s", self.mailbox_address
            )
            with contextlib.suppress(Exception):
                client.logout()
            return []

        try:
            status, _ = client.select(mailbox)
            if status != "OK":
                logger.warning(
                    "IMAP select failed for mailbox %s (folder %s): %s",
                    self.mailbox_address,
                    mailbox,
                    status,
                )
                return []
            status, data = client.search(None, search_criteria)
            if status != "OK":
                logger.warning(
                    "IMAP search with criteria %s failed for mailbox %s: %s",
                    search_criteria,
                    self.mailbox_address,
                    status,
                )
                return []

            raw_ids = data[0].split() if data else []
            if not raw_ids and search_criteria.upper() != "ALL":
                logger.debug(
                    "IMAP search for %s returned no results with criteria %s; retrying with ALL",
                    self.mailbox_address,
                    search_criteria,
                )
                status, data = client.search(None, "ALL")
                if status == "OK":
                    raw_ids = data[0].split() if data else []

            if not raw_ids:
                return []

            # Fetch newest messages first.
            raw_ids = list(reversed(raw_ids))
            if fetch_cap is not None:
                raw_ids = raw_ids[: max(fetch_cap, 0)]

            for msg_num in raw_ids:
                if limit is not None and len(messages) >= limit:
                    break
                try:
                    status, parts = client.fetch(msg_num, "(RFC822 UID)")
                except Exception:
                    logger.exception(
                        "Failed to fetch IMAP message %s for mailbox %s",
                        msg_num,
                        self.mailbox_address,
                    )
                    continue
                if status != "OK" or not parts:
                    continue

                raw_bytes = None
                uid = None
                for part in parts:
                    if not isinstance(part, tuple):
                        continue
                    header = part[0]
                    if isinstance(header, bytes):
                        header = header.decode(errors="ignore")
                    match = re.search(r"UID (\d+)", str(header))
                    if match:
                        uid = match.group(1)
                    raw_bytes = part[1]
                    break

                if raw_bytes is None:
                    continue

                parsed = self._parse_email(raw_bytes)
                identifier = uid or parsed.get("message_id") or msg_num.decode(errors="ignore").strip()
                if not identifier:
                    identifier = str(uuid.uuid4())
                store_key = f"imap:{identifier}"
                if self.state_store and store_key in self.state_store:
                    continue

                parsed["id"] = store_key
                messages.append(parsed)

                if mark_seen and self._imap_mark_seen:
                    try:
                        client.store(msg_num, "+FLAGS", "(\\Seen)")
                    except Exception:
                        logger.debug("Failed to mark IMAP message %s as seen", msg_num, exc_info=True)

        finally:
            with contextlib.suppress(Exception):
                client.logout()

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

    def _acknowledge_queue_message(self, message: Dict[str, object], *, success: bool) -> None:
        if not self._queue_enabled:
            return

        ack_context = message.get("_queue_ack")
        if not isinstance(ack_context, dict):
            return

        pending = ack_context.get("pending")
        if isinstance(pending, int):
            pending = max(0, pending - 1)
        else:
            pending = 0
        ack_context["pending"] = pending

        if not success:
            ack_context["failed"] = True

        if pending > 0:
            return

        if ack_context.get("failed"):
            return

        receipt = ack_context.get("receipt_handle")
        queue_url = ack_context.get("queue_url") or self._queue_url
        if not receipt or not queue_url:
            return

        try:
            client = self._get_sqs_client()
            if client is None:
                return
            client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            ack_context["receipt_handle"] = None
        except Exception:
            logger.exception(
                "Failed to delete queue message for mailbox %s (queue=%s)",
                self.mailbox_address,
                queue_url,
            )

    def _parse_processed_payload(self, raw_bytes: bytes) -> Dict[str, object]:
        """Parse processed email snapshots stored as JSON in S3."""

        text: Optional[str] = None
        try:
            text = raw_bytes.decode("utf-8")
        except Exception:
            text = None

        if text is not None:
            with contextlib.suppress(json.JSONDecodeError):
                payload = json.loads(text)
                if isinstance(payload, dict):
                    if isinstance(payload.get("payload"), dict):
                        payload = payload["payload"]
                    return self._normalise_processed_json(payload)

        return self._parse_email(raw_bytes)

    def _normalise_processed_json(self, payload: Dict[str, object]) -> Dict[str, object]:
        def _first_string(
            keys: Sequence[str], *, source: Optional[Dict[str, object]] = None
        ) -> str:
            target = payload if source is None else source
            for key in keys:
                value = target.get(key) if isinstance(target, dict) else None
                if value is None:
                    continue
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        return text
                elif isinstance(value, (int, float)):
                    return str(value)
            return ""

        def _extract_text(value: object) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for candidate in ("text", "body", "content", "value"):
                    inner = value.get(candidate)
                    if isinstance(inner, str):
                        return inner
            return ""

        subject = _first_string(["subject", "Subject", "email_subject"])
        from_address = _first_string(
            ["from", "from_address", "fromAddress", "sender", "sender_email", "fromEmail"]
        )

        text_body = _extract_text(
            payload.get("body")
            or payload.get("text_body")
            or payload.get("text")
            or payload.get("message")
        )
        html_body = _extract_text(payload.get("html_body") or payload.get("html"))
        if not text_body and html_body:
            text_body = _strip_html(html_body)
        body = text_body or html_body or ""

        rfq_id = _first_string(["rfq_id", "rfqId", "rfq"])
        if not rfq_id:
            rfq_id = self._extract_rfq_id(" ".join(filter(None, [subject, body])))

        received_at = payload.get("received_at") or payload.get("receivedAt")
        message_id = _first_string(["message_id", "messageId", "id"])

        recipients_value = payload.get("recipients") or payload.get("to") or payload.get("To")
        if isinstance(recipients_value, str):
            recipients = [recipients_value]
        elif isinstance(recipients_value, Sequence):
            recipients = [
                str(item)
                for item in recipients_value
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
        else:
            recipients = []

        attachments: List[Dict[str, object]] = []
        attachment_candidates = payload.get("attachments") or payload.get("Attachments")
        if isinstance(attachment_candidates, list):
            for item in attachment_candidates:
                if not isinstance(item, dict):
                    continue
                filename = _first_string(
                    ["filename", "name", "file_name"], source=item
                )
                content_value = item.get("content") or item.get("data") or item.get("body")
                content_bytes: bytes = b""
                if isinstance(content_value, bytes):
                    content_bytes = content_value
                elif isinstance(content_value, str):
                    stripped = content_value.strip()
                    if stripped:
                        with contextlib.suppress(Exception):
                            decoded = base64.b64decode(stripped, validate=True)
                            content_bytes = decoded
                        if not content_bytes:
                            content_bytes = stripped.encode()
                else:
                    with contextlib.suppress(Exception):
                        content_bytes = bytes(content_value)
                    if not isinstance(content_bytes, bytes):
                        content_bytes = b""
                disposition = _first_string(
                    ["disposition", "content_disposition"], source=item
                ) or "attachment"
                content_type = _first_string(
                    ["content_type", "mime_type", "type"], source=item
                )
                attachments.append(
                    {
                        "filename": filename or None,
                        "content_type": content_type or None,
                        "content": content_bytes,
                        "size": len(content_bytes),
                        "disposition": disposition or "attachment",
                    }
                )

        return {
            "subject": subject,
            "from": from_address,
            "body": body,
            "rfq_id": rfq_id,
            "received_at": received_at,
            "message_id": message_id,
            "recipients": recipients,
            "attachments": attachments,
        }

    def _parse_email(self, raw_bytes: bytes) -> Dict[str, object]:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject = message.get("subject", "")
        from_address = message.get("from", "")
        recipients = message.get_all("to", [])
        body = self._extract_body(message)
        rfq_id = self._extract_rfq_id(f"{subject} {body}")
        attachments = self._extract_attachments(message)
        return {
            "subject": subject,
            "from": from_address,
            "body": body,
            "rfq_id": rfq_id,
            "received_at": message.get("date"),
            "message_id": message.get("message-id"),
            "recipients": recipients,
            "attachments": attachments,
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

    def _get_sqs_client(self):
        if not self._queue_enabled or not self._queue_url:
            return None
        if self._sqs_client is not None:
            return self._sqs_client

        client_kwargs: Dict[str, object] = {}
        if self.region:
            client_kwargs["region_name"] = self.region

        if self._s3_role_arn:
            try:
                credentials = self._assume_role_credentials()
                client_kwargs.update(credentials)
            except Exception:
                logger.exception(
                    "Unable to assume role %s for SQS access; using default credentials",
                    self._s3_role_arn,
                )

        try:
            self._sqs_client = boto3.client("sqs", **client_kwargs)
        except Exception:
            logger.exception(
                "Failed to create SQS client for mailbox %s (queue=%s)",
                self.mailbox_address,
                self._queue_url,
            )
            self._sqs_client = None
        return self._sqs_client


    @staticmethod
    def _parse_region(endpoint: str) -> Optional[str]:
        match = re.search(r"email-smtp\.([a-z0-9-]+)\.amazonaws.com", endpoint)
        if match:
            return match.group(1)
        return None

    def _imap_configured(self) -> bool:
        return bool(self._imap_host and self._imap_user and self._imap_password)

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
        """Log actionable guidance for common SES inbound misconfigurations."""

        if not self.bucket:
            logger.warning(
                "SES inbound bucket is not configured; configure 'ses_inbound_bucket' "
                "or 's3_bucket_name' so the watcher can download supplier replies"
            )
            if self._queue_url:
                logger.warning(
                    "SES inbound queue %s is configured without a bucket; ensure "
                    "the queue payload includes bucket overrides or align the "
                    "bucket configuration",
                    self._queue_url,
                )
            return

        if not self._queue_url:
            logger.warning(
                "SES inbound queue URL is not configured for bucket %s. Configure "
                "'ses_inbound_queue_url' and S3→SQS notifications so new emails "
                "arrive immediately instead of relying on eventual S3 polling",
                self.bucket,
            )
        elif not self._queue_enabled:
            if self._queue_disabled_reason:
                logger.info(
                    "SES inbound queue %s polling disabled for mailbox %s: %s",
                    self._queue_url,
                    self.mailbox_address,
                    self._queue_disabled_reason,
                )
            else:
                logger.info(
                    "SES inbound queue %s polling disabled for mailbox %s", self._queue_url, self.mailbox_address
                )
        else:
            logger.debug(
                "SES inbound queue %s is configured for bucket %s", self._queue_url, self.bucket
            )

        if raw_prefix is None:
            return

        expected_prefixes = ", ".join(self._prefixes)
        logger.info(
            "SES inbound prefixes normalised from %r: %s", raw_prefix, expected_prefixes or "<root>"
        )
        logger.info(
            "SES processed email prefixes: %s",
            ", ".join(self._processed_prefixes) or "<none>",
        )

    def _build_prefixes(self, raw_prefix: object) -> List[str]:
        base_candidates = self._normalise_prefix_input(raw_prefix)
        base_normalised: List[str] = []
        include_empty = False

        for candidate in base_candidates:
            if not candidate:
                include_empty = True
                continue
            base_normalised.append(self._ensure_trailing_slash(candidate))

        if not base_normalised and include_empty:
            base_normalised.append("")

        mailbox_variants: List[str] = []
        mailbox = (self.mailbox_address or "").strip()
        if mailbox:
            mailbox_variants.append(mailbox)
            mailbox_variants.append(mailbox.replace("@", "/"))

        derived: List[str] = []
        bases = base_normalised or [""]
        for base in bases:
            derived.append(self._ensure_trailing_slash(base))
            for variant in mailbox_variants:
                combined = self._join_prefix(base, variant)
                if combined:
                    derived.append(combined)

        for variant in mailbox_variants:
            derived.append(self._ensure_trailing_slash(variant))

        cleaned = [item if item is not None else "" for item in derived]
        unique = self._unique_preserve(cleaned)
        return unique or [""]

    @staticmethod
    def _normalise_prefix_input(value: object) -> List[str]:
        if value is None:
            return [""]
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return parts if parts else [""]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            results: List[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    results.append(text)
            return results or [""]
        text = str(value).strip()
        return [text] if text else [""]

    @staticmethod
    def _ensure_trailing_slash(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return ""
        return text if text.endswith("/") else f"{text}/"

    @staticmethod
    def _join_prefix(prefix: str, suffix: str) -> str:
        base = SESEmailWatcher._ensure_trailing_slash(prefix)
        clean_suffix = (suffix or "").strip().strip("/")
        if not clean_suffix:
            return base
        if base:
            return f"{base}{clean_suffix}/"
        return f"{clean_suffix}/"

    @staticmethod
    def _unique_preserve(items: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

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

