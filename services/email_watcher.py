from __future__ import annotations

import gzip
import imaplib
import logging
import mimetypes
import re
import time
import unicodedata
import uuid
from urllib.parse import unquote_plus, urlparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

try:  # pragma: no cover - optional dependency during tests
    from psycopg2 import errors as psycopg2_errors
except Exception:  # pragma: no cover - psycopg2 may be unavailable in tests
    psycopg2_errors = None  # type: ignore[assignment]

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from utils.gpu import configure_gpu


logger = logging.getLogger(__name__)

_DASH_CLASS = r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]"
_ZW_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF\u2060\u200E\u200F]")
_WS_RE = re.compile(r"\s+")


def _norm(value: str) -> str:
    """Normalise textual content for RFQ comparisons."""

    if not value:
        return ""

    text = unicodedata.normalize("NFKC", value)
    text = _ZW_RE.sub("", text)
    text = re.sub(_DASH_CLASS, "-", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _canon_id(value: str) -> str:
    return _norm(value).upper()


def _rfq_match_key(value: object) -> Optional[str]:
    """Return the normalised tail used for RFQ comparisons."""

    if value in (None, ""):
        return None

    try:
        canonical = _canon_id(str(value))
    except Exception:
        return None

    if not canonical:
        return None

    tail_source = re.sub(r"[^A-Z0-9]", "", canonical)
    if not tail_source:
        return None

    tail = tail_source[-8:]
    return tail.lower() if tail else None


RFQ_ID_RE = re.compile(r"\bRFQ-\d{8}-[A-Za-z0-9]{8}\b", re.IGNORECASE)

# Ensure GPU related environment flags are consistently applied even when the
# watcher is used standalone (e.g. in a scheduled job).
configure_gpu()


def _decode_mime_header(value: object) -> str:
    """Decode MIME encoded headers into a display-friendly string."""

    if value in (None, ""):
        return ""
    try:
        text = str(value)
    except Exception:
        return ""

    try:
        header = make_header(decode_header(text))
        return str(header).strip()
    except Exception:
        return text.strip()


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
    primed: bool = False

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
        r"<!--\s*(?:RFQ-ID\s*:\s*|PROCWISE:RFQ_ID=)([A-Za-z0-9_-]+)\s*-->",
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
        self.mailbox_address = (
            getattr(self.settings, "supplier_mailbox", None)
            or getattr(self.settings, "imap_user", None)
            or "supplierconnect@procwise.co.uk"
        )
        poll_interval = response_poll_seconds
        if poll_interval is None:
            poll_interval = getattr(
                self.settings, "email_response_poll_seconds", 60
            )
        try:
            poll_interval = int(poll_interval)
        except Exception:
            poll_interval = 60

        self._poll_window_minutes = self._coerce_int(
            getattr(self.settings, "email_watch_window_minutes", 10),
            default=10,
            minimum=1,
        )
        sleep_default = self._coerce_int(
            getattr(self.settings, "email_match_poll_sleep_seconds", 60),
            default=60,
            minimum=1,
        )
        self._match_poll_sleep_seconds = max(1, sleep_default)

        self.poll_interval_seconds = max(1, poll_interval)
        if self.poll_interval_seconds < self._match_poll_sleep_seconds:
            self.poll_interval_seconds = self._match_poll_sleep_seconds

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
        configured_prefix = getattr(self.settings, "ses_inbound_prefix", None)
        self.bucket, prefix_value = self._resolve_bucket_and_prefix(
            bucket_candidates,
            configured_prefix=configured_prefix,
            uri_prefix=uri_prefix,
            default_bucket=default_bucket,
            default_prefix=default_prefix,
        )

        self._prefixes = [self._ensure_trailing_slash(prefix_value)]

        # Ensure supporting negotiation tables exist so metadata lookups do
        # not fail on freshly provisioned databases.  The statements are safe
        # to run repeatedly thanks to ``IF NOT EXISTS`` guards.
        self._ensure_negotiation_tables()
        self._ensure_processed_registry_table()
        self._ensure_draft_rfq_email_constraints()
        self._ensure_email_watcher_watermarks_table()

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
        self._completed_targets: Set[str] = set()
        self._rfq_run_counts: Dict[str, int] = {}
        self._rfq_index_hits: Dict[str, str] = {}
        self._rfq_tail_cache: Dict[str, List[Dict[str, object]]] = {}
        self._workflow_rfq_index: Dict[str, Set[str]] = defaultdict(set)
        self._rfq_workflow_index: Dict[str, Set[str]] = defaultdict(set)
        self._last_watermark_ts: Optional[datetime] = None
        self._last_watermark_key: str = ""
        self._load_watermark()
        self._match_poll_attempts = self._coerce_int(
            getattr(self.settings, "email_match_poll_attempts", 3),
            default=3,
            minimum=0,
        )
        self._match_poll_timeout_seconds = max(
            0,
            self._coerce_int(
                getattr(self.settings, "email_match_poll_timeout_seconds", 300),
                default=300,
                minimum=0,
            ),
        )
        self._imap_fallback_attempts = self._coerce_int(
            getattr(self.settings, "email_s3_imap_fallback_attempts", 3),
            default=3,
            minimum=1,
        )
        self._consecutive_empty_s3_batches = 0
        self._require_new_s3_objects = False
        self._dispatch_wait_seconds = max(
            0,
            self._coerce_int(
                getattr(self.settings, "email_inbound_initial_wait_seconds", 60),
                default=60,
                minimum=0,
            ),
        )
        self._last_dispatch_notified_at: Optional[float] = None
        self._last_dispatch_wait_acknowledged: Optional[float] = None

        logger.info(
            "Initialised email watcher for mailbox %s using %s (poll_interval=%ss)",
            self.mailbox_address,
            self._format_bucket_prefixes(),
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
        """Process a single batch of inbound emails."""

        filters: Dict[str, object] = dict(match_filters) if match_filters else {}
        results: List[Dict[str, object]] = []
        target_rfq_normalised: Optional[str] = None
        if filters:
            target_rfq_normalised = self._normalise_rfq_value(filters.get("rfq_id"))
        run_count = 0
        if target_rfq_normalised:
            run_count = self._rfq_run_counts.get(target_rfq_normalised, 0) + 1
            self._rfq_run_counts[target_rfq_normalised] = run_count
            if run_count > 1 and target_rfq_normalised in self._completed_targets:
                logger.info(
                    "RFQ %s requested again for mailbox %s (run %s); resetting completion state",
                    target_rfq_normalised,
                    self.mailbox_address,
                    run_count,
                )
                self._completed_targets.discard(target_rfq_normalised)
            hit = self._lookup_rfq_hit_pg(target_rfq_normalised)
            if hit:
                cached_key = self._rfq_index_hits.get(target_rfq_normalised)
                if cached_key and cached_key == hit["key"]:
                    logger.info(
                        "RFQ %s index entry %s already surfaced on a prior run (run %s); continuing with S3 scan",
                        target_rfq_normalised,
                        cached_key,
                        run_count or 1,
                    )
                else:
                    self._rfq_index_hits[target_rfq_normalised] = hit["key"]
                    results.append(
                        {
                            "rfq_id": hit["rfq_id"],
                            "supplier_id": None,
                            "message_id": None,
                            "subject": None,
                            "from_address": None,
                            "message_body": None,
                            "target_price": None,
                            "negotiation_triggered": False,
                            "supplier_status": None,
                            "negotiation_status": None,
                            "supplier_output": None,
                            "negotiation_output": None,
                            "canonical_s3_key": hit["key"],
                            "matched_via": "index",
                        }
                    )
                    logger.info(
                        "RFQ %s matched via index at %s — skipping S3 scan for mailbox %s",
                        target_rfq_normalised,
                        hit["processed_at"],
                        self.mailbox_address,
                    )
                    return results

        logger.info(
            "Scanning %s for new supplier responses (limit=%s)",
            self._format_bucket_prefixes(),
            limit if limit is not None else "unbounded",
        )

        previous_new_flag = self._require_new_s3_objects
        previous_watermark = (self._last_watermark_ts, self._last_watermark_key)
        self._require_new_s3_objects = bool(filters)

        try:
            prefixes = self._derive_prefixes_for_filters(filters)
            effective_limit = self._coerce_limit(limit)

            match_found = False

            if target_rfq_normalised and target_rfq_normalised in self._completed_targets:
                logger.info(
                    "RFQ %s already processed for mailbox %s; skipping poll",
                    target_rfq_normalised,
                    self.mailbox_address,
                )
                return []

            self._respect_post_dispatch_wait()

            attempts = 0
            poll_deadline: Optional[float] = None
            if target_rfq_normalised:
                unlimited_attempts = self._match_poll_attempts <= 0
                max_attempts = self._match_poll_attempts if self._match_poll_attempts > 0 else 0
                if self._match_poll_timeout_seconds > 0:
                    poll_deadline = time.time() + self._match_poll_timeout_seconds
            else:
                unlimited_attempts = False
                max_attempts = 1
                poll_deadline = None
            total_candidates = 0
            total_processed = 0


            while True:
                attempts += 1
                try:
                    loader_empty = False
                    if self._custom_loader is not None:
                        messages = self._custom_loader(limit)
                        candidate_batch = list(messages)
                        logger.info(
                            "Retrieved %d candidate message(s) from loader for mailbox %s",
                            len(candidate_batch),
                            self.mailbox_address,
                        )
                        loader_empty = len(candidate_batch) == 0
                        for message in candidate_batch:
                            last_modified_hint = message.get("_last_modified")
                            if isinstance(last_modified_hint, datetime):
                                self._update_watermark(last_modified_hint, str(message.get("id") or ""))
                            total_candidates += 1
                            matched, should_stop, rfq_matched, was_processed = self._process_candidate_message(
                                message,
                                match_filters=filters,
                                target_rfq_normalised=target_rfq_normalised,
                                results=results,
                                effective_limit=effective_limit,
                            )
                            if was_processed:
                                total_processed += 1
                            if matched or rfq_matched:
                                match_found = True
                            if should_stop:
                                if not match_found:
                                    match_found = True
                                break

                    if self._custom_loader is None or loader_empty:
                        processed_batch: List[Dict[str, object]] = []

                        def _on_message(
                            parsed: Dict[str, object],
                            last_modified: Optional[datetime] = None,
                        ) -> bool:
                            nonlocal total_processed, total_candidates, match_found
                            total_candidates += 1
                            processed_batch.append(parsed)
                            matched, should_stop, rfq_matched, was_processed = self._process_candidate_message(
                                parsed,
                                match_filters=filters,
                                target_rfq_normalised=target_rfq_normalised,
                                results=results,
                                effective_limit=effective_limit,
                            )
                            if last_modified is not None:
                                self._update_watermark(last_modified, str(parsed.get("id") or ""))
                            if was_processed:
                                total_processed += 1
                            # Treat RFQ match as authoritative regardless of additional filters
                            if matched or rfq_matched:
                                match_found = True
                            return should_stop


                        messages = self._load_messages(
                            limit,
                            mark_seen=True,
                            prefixes=prefixes,
                            on_message=_on_message,
                        )
                        batch_count = len(processed_batch) if filters else len(messages)
                        logger.info(
                            "Retrieved %d candidate message(s) from S3 for mailbox %s",
                            batch_count,
                            self.mailbox_address,
                        )
                        if not filters:
                            for message in messages:
                                last_modified_hint = message.get("_last_modified")
                                if isinstance(last_modified_hint, datetime):
                                    self._update_watermark(last_modified_hint, str(message.get("id") or ""))
                                total_candidates += 1
                                matched, should_stop, rfq_matched, was_processed = self._process_candidate_message(
                                    message,
                                    match_filters=filters,
                                    target_rfq_normalised=target_rfq_normalised,
                                    results=results,
                                    effective_limit=effective_limit,
                                )
                                if was_processed:
                                    total_processed += 1
                                # RFQ match is authoritative even when filters are present
                                if matched or rfq_matched:
                                    match_found = True
                                if should_stop:
                                    break
                except Exception:  # pragma: no cover - network/runtime
                    logger.exception("Failed to load inbound SES messages")
                    break

                if match_found:
                    break

                if not target_rfq_normalised:
                    break

                if not unlimited_attempts and max_attempts > 0 and attempts >= max_attempts:
                    logger.debug(
                        "RFQ %s not found in mailbox %s on attempt %d/%s; reached polling limit",
                        target_rfq_normalised,
                        self.mailbox_address,
                        attempts,
                        max_attempts,
                    )
                    break

                if poll_deadline is not None and time.time() >= poll_deadline:
                    logger.debug(
                        "RFQ %s not found in mailbox %s within %.1fs; reached polling deadline",
                        target_rfq_normalised,
                        self.mailbox_address,
                        self._match_poll_timeout_seconds,
                    )
                    break

                logger.debug(
                    "RFQ %s not found in mailbox %s on attempt %d/%s; waiting %.1fs before retrying",
                    target_rfq_normalised,
                    self.mailbox_address,
                    attempts,
                    max_attempts if max_attempts > 0 else "inf",
                    self.poll_interval_seconds,
                )
                if self.poll_interval_seconds >= 0:
                    sleep_for = self.poll_interval_seconds
                else:
                    sleep_for = self._match_poll_sleep_seconds
                if poll_deadline is not None:
                    remaining = poll_deadline - time.time()
                    if remaining <= 0:
                        break
                    sleep_for = min(sleep_for, max(0.0, remaining))
                if sleep_for > 0:
                    time.sleep(sleep_for)

            if target_rfq_normalised and match_found:
                self._completed_targets.add(target_rfq_normalised)
                if results:
                    match_key = results[-1].get("message_id")
                else:
                    match_key = None
                logger.info(
                    "RFQ %s matched at %s — stopping watcher for mailbox %s",
                    target_rfq_normalised,
                    match_key or "<unknown>",
                    self.mailbox_address,
                )
                logger.info(
                    "Scan summary for mailbox %s (candidates=%d, processed=%d, matched=True). Watermark=%s",
                    self.mailbox_address,
                    total_candidates,
                    total_processed,
                    self._format_watermark(),
                )
            else:
                logger.info(
                    "Completed scan for mailbox %s (candidates=%d, processed=%d, matched=%s). Watermark=%s",
                    self.mailbox_address,
                    total_candidates,
                    total_processed,
                    match_found,
                    self._format_watermark(),
                )

            return results
        finally:
            if (
                self._last_watermark_ts,
                self._last_watermark_key,
            ) != previous_watermark:
                self._store_watermark()
            self._require_new_s3_objects = previous_new_flag

    def _process_candidate_message(
        self,
        message: Dict[str, object],
        *,
        match_filters: Dict[str, object],
        target_rfq_normalised: Optional[str],
        results: List[Dict[str, object]],
        effective_limit: Optional[int],
    ) -> Tuple[bool, bool, bool, bool]:
        if not isinstance(message, dict):
            return False, False, False, False

        bucket_hint = message.get("_bucket") or message.get("bucket")
        if bucket_hint:
            message["_bucket"] = bucket_hint

        s3_key = message.get("id") if isinstance(message.get("id"), str) else message.get("s3_key")
        if (not message.get("body") or not message.get("subject")) and s3_key:
            bucket_for_fetch = bucket_hint or getattr(self, "bucket", None)
            if bucket_for_fetch:
                try:
                    client = self._get_s3_client()
                    download = self._download_object(client, s3_key, bucket=bucket_for_fetch)
                except Exception:
                    logger.exception(
                        "Failed to hydrate SES message %s from bucket %s",
                        s3_key,
                        bucket_for_fetch,
                    )
                    download = None
                if download is not None:
                    raw_bytes, size_bytes = download
                    parsed_payload = self._invoke_parser(
                        self._parse_inbound_object, raw_bytes, s3_key
                    )
                    parsed_payload.setdefault("id", s3_key)
                    parsed_payload.setdefault("s3_key", s3_key)
                    parsed_payload.setdefault("_bucket", bucket_for_fetch)
                    if size_bytes is not None:
                        parsed_payload["_content_length"] = size_bytes
                    for key, value in parsed_payload.items():
                        if key not in message or not message.get(key):
                            message[key] = value
                    if size_bytes is not None and not message.get("_content_length"):
                        message["_content_length"] = size_bytes
                    if not bucket_hint:
                        bucket_hint = bucket_for_fetch
                        message["_bucket"] = bucket_hint

        message_id = str(message.get("id") or uuid.uuid4())
        bucket = bucket_hint or getattr(self, "bucket", None)
        s3_key = message.get("id") if isinstance(message.get("id"), str) else message.get("s3_key")
        etag = message.get("_s3_etag") or message.get("s3_etag")
        size_hint = (
            message.get("_content_length")
            or message.get("size_bytes")
            or message.get("content_length")
        )
        size_bytes: Optional[int]
        size_bytes = None
        if isinstance(size_hint, (int, float)):
            size_bytes = int(size_hint)
        elif isinstance(size_hint, str):
            try:
                size_bytes = int(float(size_hint))
            except (TypeError, ValueError):
                size_bytes = None

        if self.state_store and message_id in self.state_store:
            logger.debug(
                "Skipping previously processed message %s for mailbox %s",
                message_id,
                self.mailbox_address,
            )
            return False, False, False, False

        if self._is_processed_in_registry(bucket, s3_key, etag):
            logger.debug(
                "Skipping S3 object %s (etag=%s) for mailbox %s; already processed",
                s3_key,
                etag,
                self.mailbox_address,
            )
            return False, False, False, False

        try:
            processed, reason = self._process_message(message)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to process SES message %s", message_id)
            processed, reason = None, "processing_error"

        metadata: Dict[str, object]
        processed_payload: Optional[Dict[str, object]] = None
        message_match = False
        rfq_match = False
        was_processed = False

        if processed:
            processed_payload = self._record_processed_payload(message_id, processed)
            original_rfq = processed_payload.get("rfq_id") if processed_payload else None
            rfq_extracted = (
                self._normalise_rfq_value(original_rfq) if original_rfq else None
            )
            if target_rfq_normalised:
                rfq_match = rfq_extracted == target_rfq_normalised
            else:
                rfq_match = bool(rfq_extracted)

            if match_filters:
                self._apply_filter_defaults(processed_payload, match_filters)
                message_match = self._matches_filters(processed_payload, match_filters)
                # Treat RFQ match as authoritative even if other filters fail
                if target_rfq_normalised and rfq_match:
                    message_match = True
            was_processed = True

            canonical_key = self._ensure_s3_mapping(s3_key, processed_payload.get("rfq_id"))
            if canonical_key:
                processed_payload["canonical_s3_key"] = canonical_key
                processed["canonical_s3_key"] = canonical_key
                if target_rfq_normalised:
                    self._rfq_index_hits[target_rfq_normalised] = canonical_key

            log_s3_key = canonical_key or s3_key
            if log_s3_key and not processed_payload.get("s3_key"):
                processed_payload["s3_key"] = log_s3_key
            if etag and not processed_payload.get("etag"):
                processed_payload["etag"] = etag
            if size_bytes is not None and not processed_payload.get("size_bytes"):
                processed_payload["size_bytes"] = size_bytes

            canonical_rfq_full = None
            try:
                if processed_payload.get("rfq_id"):
                    canonical_rfq_full = _canon_id(str(processed_payload.get("rfq_id")))
            except Exception:
                canonical_rfq_full = None
            header_candidate = message.get("rfq_id_header")
            match_source = "unknown"
            header_tail = (
                self._normalise_rfq_value(header_candidate)
                if isinstance(header_candidate, str)
                else None
            )
            if canonical_rfq_full and header_tail and header_tail == rfq_extracted:
                match_source = "header"
            else:
                subject_candidate = (
                    processed_payload.get("subject") or message.get("subject")
                )
                body_candidate = (
                    processed_payload.get("message_body") or message.get("body")
                )
                if (
                    match_source == "unknown"
                    and canonical_rfq_full
                    and subject_candidate
                    and canonical_rfq_full in _norm(str(subject_candidate)).upper()
                ):
                    match_source = "subject"
                if (
                    match_source == "unknown"
                    and canonical_rfq_full
                    and body_candidate
                    and canonical_rfq_full in _norm(str(body_candidate)).upper()
                ):
                    match_source = "body"
            if match_source == "unknown" and canonical_rfq_full:
                match_source = "body"

            processed_payload["matched_via"] = match_source
            processed_payload.setdefault("mailbox", self.mailbox_address)

            logger.info(
                "rfq_email_processed mailbox=%s rfq_id=%s supplier_id=%s s3_key=%s etag=%s size_bytes=%s price=%s lead_time=%s matched_via=%s",
                self.mailbox_address,
                processed_payload.get("rfq_id"),
                processed_payload.get("supplier_id"),
                log_s3_key,
                etag or "",
                size_bytes if size_bytes is not None else "unknown",
                processed_payload.get("price"),
                processed_payload.get("lead_time"),
                match_source,
            )

            metadata = {
                "rfq_id": processed_payload.get("rfq_id") if processed_payload else None,
                "supplier_id": processed_payload.get("supplier_id") if processed_payload else None,
                "workflow_id": processed_payload.get("workflow_id") if processed_payload else None,
                "related_rfq_ids": processed_payload.get("related_rfq_ids") if processed_payload else None,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "processed",
                "payload": processed_payload,
            }
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

        if was_processed and processed_payload:
            rfq_id = processed_payload.get("rfq_id")
            self._record_processed_in_registry(bucket, s3_key, etag, rfq_id)

        matched = bool(message_match)
        should_stop = False

        include_payload = False
        if processed_payload:
            if not match_filters or matched or (target_rfq_normalised and rfq_match):
                include_payload = True

        if include_payload and effective_limit != 0:
            results.append(processed_payload)
            if (
                effective_limit is not None
                and effective_limit != 0
                and len(results) >= effective_limit
            ):
                should_stop = True

        # Stop conditions: either a filter match or an RFQ match should stop polling
        if matched:
            should_stop = True
            logger.debug(
                "Stopping poll once after matching filters for message %s",
                message_id,
            )
        elif rfq_match:
            should_stop = True
            logger.debug("Stopping poll after RFQ match for message %s", message_id)

        return matched, should_stop, bool(rfq_match), was_processed


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
            messages = self._load_messages(limit_int, mark_seen=False, prefixes=None)

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
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.negotiation_session_state (
                            rfq_id TEXT NOT NULL,
                            supplier_id TEXT NOT NULL,
                            supplier_reply_count INTEGER NOT NULL DEFAULT 0,
                            current_round INTEGER NOT NULL DEFAULT 1,
                            status TEXT NOT NULL DEFAULT 'ACTIVE',
                            awaiting_response BOOLEAN NOT NULL DEFAULT FALSE,
                            last_supplier_msg_id TEXT,
                            last_agent_msg_id TEXT,
                            last_email_sent_at TIMESTAMPTZ,
                            updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            CONSTRAINT negotiation_session_state_pk PRIMARY KEY (rfq_id, supplier_id)
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS negotiation_session_state_status_idx
                            ON proc.negotiation_session_state (status)
                        """
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to ensure negotiation support tables exist")

    def _load_negotiation_state(
        self, rfq_id: Optional[str], supplier_id: Optional[str]
    ) -> tuple[bool, Optional[str]]:
        if not rfq_id or not supplier_id:
            return False, None
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return False, None
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT awaiting_response, last_supplier_msg_id
                          FROM proc.negotiation_session_state
                         WHERE rfq_id = %s AND supplier_id = %s
                        """,
                        (rfq_id, supplier_id),
                    )
                    row = cur.fetchone()
                    if not row:
                        return False, None
                    awaiting = bool(row[0])
                    last_msg = row[1] if len(row) > 1 else None
                    if isinstance(last_msg, memoryview):
                        last_msg = last_msg.tobytes()
                    if isinstance(last_msg, (bytes, bytearray)):
                        try:
                            last_msg = last_msg.decode("utf-8", errors="ignore")
                        except Exception:
                            last_msg = str(last_msg)
                    return awaiting, last_msg
        except Exception:
            logger.exception(
                "Failed to load negotiation state for rfq %s supplier %s",
                rfq_id,
                supplier_id,
            )
        return False, None

    def _ensure_processed_registry_table(self) -> None:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.processed_emails (
                            bucket TEXT NOT NULL,
                            key TEXT NOT NULL,
                            etag TEXT NOT NULL DEFAULT '',
                            rfq_id TEXT,
                            processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE proc.processed_emails
                            ALTER COLUMN etag SET DEFAULT ''
                        """
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS processed_emails_bucket_key_etag_uidx
                            ON proc.processed_emails (bucket, key, etag)
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS processed_emails_rfq_ts_idx
                            ON proc.processed_emails (rfq_id, processed_at DESC)
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS processed_emails_rfq_key_idx
                            ON proc.processed_emails (rfq_id, key)
                        """
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to ensure processed email registry table exists")

    def _ensure_draft_rfq_email_constraints(self) -> None:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        try:
            with get_conn() as conn:
                success = True
                with conn.cursor() as cur:
                    try:
                        cur.execute(
                            """
                            ALTER TABLE proc.draft_rfq_emails
                                ADD CONSTRAINT draft_rfq_emails_rfq_id_uk UNIQUE (rfq_id)
                            """
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        success = False
                        if psycopg2_errors and isinstance(
                            exc, getattr(psycopg2_errors, "DuplicateObject", ())
                        ):
                            conn.rollback()
                        elif psycopg2_errors and isinstance(
                            exc, getattr(psycopg2_errors, "UniqueViolation", ())
                        ):
                            logger.warning(
                                "Duplicate RFQ identifiers detected while enforcing uniqueness"
                            )
                            conn.rollback()
                        else:
                            conn.rollback()
                            raise
                if success:
                    conn.commit()
        except Exception:
            logger.exception(
                "Failed to ensure unique constraint on proc.draft_rfq_emails"
            )

    def _ensure_email_watcher_watermarks_table(self) -> None:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.email_watcher_watermarks (
                            mailbox TEXT NOT NULL,
                            prefix  TEXT NOT NULL,
                            ts      TIMESTAMPTZ NOT NULL,
                            key     TEXT NOT NULL,
                            PRIMARY KEY (mailbox, prefix)
                        )
                        """
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to ensure email watcher watermark table exists")

    def _is_processed_in_registry(
        self,
        bucket: Optional[str],
        key: Optional[str],
        etag: Optional[str],
    ) -> bool:
        if not bucket or not key:
            return False

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return False

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 1
                        FROM proc.processed_emails
                        WHERE bucket = %s AND key = %s AND etag = COALESCE(%s, '')
                        LIMIT 1
                        """,
                        (bucket, key, etag or ""),
                    )
                    row = cur.fetchone()
                    return bool(row)
        except Exception:
            logger.exception("Failed to check processed email registry for %s", key)
        return False

    def _lookup_rfq_hit_pg(self, rfq_id_norm: str) -> Optional[Dict[str, str]]:
        tail = self._normalise_rfq_value(rfq_id_norm)
        if not tail:
            return None
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT rfq_id, key, processed_at
                        FROM proc.processed_emails
                        WHERE RIGHT(REGEXP_REPLACE(UPPER(rfq_id), '[^A-Z0-9]', '', 'g'), 8) = %s
                        ORDER BY processed_at DESC
                        LIMIT 1
                        """,
                        (tail.upper(),),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return {
                        "rfq_id": row[0],
                        "key": row[1],
                        "processed_at": row[2].isoformat(),
                    }
        except Exception:
            logger.exception("RFQ index lookup failed for %s", rfq_id_norm)
            return None

    def _load_watermark(self) -> None:
        if not self._prefixes:
            return

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        prefix = self._prefixes[0]
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT ts, key
                        FROM proc.email_watcher_watermarks
                        WHERE mailbox = %s AND prefix = %s
                        """,
                        (self.mailbox_address, prefix),
                    )
                    row = cur.fetchone()
                    if not row:
                        return
                    ts_value, key_value = row
                    self._last_watermark_ts = ts_value
                    self._last_watermark_key = str(key_value or "")
        except Exception:
            logger.exception(
                "Failed to load email watcher watermark for mailbox %s",
                self.mailbox_address,
            )

    def _store_watermark(self) -> None:
        if self._last_watermark_ts is None or not self._prefixes:
            return

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        prefix = self._prefixes[0]
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.email_watcher_watermarks (mailbox, prefix, ts, key)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (mailbox, prefix)
                        DO UPDATE SET ts = EXCLUDED.ts, key = EXCLUDED.key
                        """,
                        (
                            self.mailbox_address,
                            prefix,
                            self._last_watermark_ts,
                            self._last_watermark_key,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception(
                "Failed to persist email watcher watermark for mailbox %s",
                self.mailbox_address,
            )

    def _record_processed_in_registry(
        self,
        bucket: Optional[str],
        key: Optional[str],
        etag: Optional[str],
        rfq_id: Optional[str],
    ) -> None:
        if not bucket or not key:
            return

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.processed_emails (bucket, key, etag, rfq_id, processed_at)
                        VALUES (%s, %s, COALESCE(%s, ''), %s, NOW())
                        ON CONFLICT (bucket, key, etag)
                        DO UPDATE SET rfq_id = EXCLUDED.rfq_id, processed_at = NOW()
                        """,
                        (bucket, key, etag or "", rfq_id),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to record processed email %s in registry", key)

    def _process_message(self, message: Dict[str, object]) -> tuple[Optional[Dict[str, object]], Optional[str]]:
        subject = str(message.get("subject", ""))
        body = str(message.get("body", ""))
        subject_normalised = _norm(subject)
        body_normalised = _norm(body)
        from_address = str(message.get("from", ""))
        rfq_candidates: List[str] = []
        tail_supplier_map: Dict[str, Optional[str]] = {}
        raw_rfq = message.get("rfq_id")
        if isinstance(raw_rfq, str) and raw_rfq.strip():
            rfq_candidates.append(raw_rfq)
        elif isinstance(raw_rfq, (list, tuple, set)):
            for entry in raw_rfq:
                if isinstance(entry, str) and entry.strip():
                    rfq_candidates.append(entry)

        extracted_candidates = self._extract_rfq_ids(
            f"{subject_normalised} {body_normalised}",
            raw_subject=subject,
            normalised_subject=subject_normalised,
        )
        rfq_candidates.extend(extracted_candidates)

        metadata_candidates = self._rfq_candidates_from_metadata(message)
        for candidate, supplier_hint in metadata_candidates:
            if candidate not in rfq_candidates:
                rfq_candidates.append(candidate)
            canonical_hint = self._canonical_rfq(candidate)
            if canonical_hint and supplier_hint and canonical_hint not in tail_supplier_map:
                tail_supplier_map[canonical_hint] = supplier_hint

        canonical_map: Dict[str, str] = {}
        ordered_canonicals: List[str] = []
        for candidate in rfq_candidates:
            canonical = self._canonical_rfq(candidate)
            if not canonical or canonical in canonical_map:
                continue
            canonical_map[canonical] = str(candidate)
            ordered_canonicals.append(canonical)

        if not ordered_canonicals:
            search_text = " ".join(
                value
                for value in (subject_normalised, body_normalised)
                if isinstance(value, str)
            )
            for canonical, resolved_value, supplier_hint in self._resolve_rfq_from_tail_tokens(
                search_text, message
            ):
                if canonical in canonical_map:
                    continue
                canonical_map[canonical] = resolved_value
                ordered_canonicals.append(canonical)
                if supplier_hint and canonical not in tail_supplier_map:
                    tail_supplier_map[canonical] = supplier_hint

        if not ordered_canonicals:
            logger.debug("Skipping email without RFQ identifier: %s", subject)
            return None, "missing_rfq_id"

        primary_canonical = ordered_canonicals[0]
        rfq_id = canonical_map[primary_canonical]
        additional_canonicals = ordered_canonicals[1:]
        additional_rfqs = [canonical_map[c] for c in additional_canonicals]

        if not additional_rfqs and len(rfq_candidates) > 1:
            fallback_values: List[str] = []
            fallback_canonicals: List[str] = []
            for candidate in rfq_candidates:
                canonical = self._canonical_rfq(candidate)
                if not canonical or canonical == primary_canonical:
                    continue
                if canonical in fallback_canonicals:
                    continue
                fallback_canonicals.append(canonical)
                fallback_values.append(canonical_map.get(canonical, str(candidate)))
            if fallback_values:
                additional_rfqs = fallback_values
                additional_canonicals = fallback_canonicals

        metadata = self._load_metadata(rfq_id)
        canonical_rfq_id = metadata.get("canonical_rfq_id")
        if canonical_rfq_id:
            rfq_id = str(canonical_rfq_id)
        supplier_id = metadata.get("supplier_id") or message.get("supplier_id")
        if not supplier_id:
            supplier_id = tail_supplier_map.get(primary_canonical)
        target_price = metadata.get("target_price") or message.get("target_price")
        negotiation_round = metadata.get("round") or message.get("round") or 1

        if target_price is not None:
            try:
                target_price = float(target_price)
            except (TypeError, ValueError):
                logger.warning("Invalid target price '%s' for RFQ %s", target_price, rfq_id)
                target_price = None

        processed: Dict[str, object] = {"message_id": message.get("id")}

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

        self._register_workflow_mapping(
            context.workflow_id, [primary_canonical, *additional_canonicals]
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
            processed.update(
                {
                    "rfq_id": rfq_id,
                    "supplier_id": supplier_id,
                    "subject": subject,
                    "from_address": from_address,
                    "message_body": body,
                    "target_price": target_price,
                    "workflow_id": context.workflow_id,
                    "related_rfq_ids": additional_rfqs,
                    "negotiation_triggered": False,
                    "supplier_status": interaction_output.status.value,
                    "negotiation_status": None,
                    "supplier_output": interaction_output.data,
                    "negotiation_output": None,
                    "error": error_detail,
                }
            )
            return processed, "supplier_interaction_failed"
        negotiation_output: Optional[AgentOutput] = None
        triggered = False

        if (
            self.enable_negotiation
            and target_price is not None
            and interaction_output.status == AgentStatus.SUCCESS
            and "NegotiationAgent" in (interaction_output.next_agents or [])
            and self.negotiation_agent is not None
        ):
            awaiting_state, last_supplier_msg = self._load_negotiation_state(rfq_id, supplier_id)
            message_token = processed.get("message_id")
            message_key = str(message_token) if message_token else None
            last_key = str(last_supplier_msg) if last_supplier_msg else None
            if awaiting_state and (message_key is None or message_key == last_key):
                logger.info(
                    "Negotiation paused for RFQ %s – awaiting supplier response before counter.",
                    rfq_id,
                )
            else:
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
                        "message_id": message_token,
                        "from_address": from_address,
                        "supplier_message": interaction_output.data.get("response_text"),
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

        processed.update(
            {
                "rfq_id": rfq_id,
                "supplier_id": supplier_id,
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
                "workflow_id": context.workflow_id,
                "related_rfq_ids": additional_rfqs,
            }
        )
        self._remember_rfq_tail(rfq_id, supplier_id)
        for extra in additional_rfqs:
            self._remember_rfq_tail(extra, supplier_id)
        return processed, None

    def _load_metadata(self, rfq_id: str) -> Dict[str, object]:
        if self.metadata_provider is not None:
            try:
                data = self.metadata_provider(rfq_id) or {}
                return dict(data)
            except Exception:  # pragma: no cover - defensive
                logger.exception("metadata provider failed for %s", rfq_id)

        details: Dict[str, object] = {}
        resolved_record = self._resolve_rfq_record(rfq_id)
        canonical_rfq_id: Optional[str] = None
        if resolved_record:
            canonical_rfq_id = resolved_record.get("rfq_id")
            if resolved_record.get("supplier_id"):
                details["supplier_id"] = resolved_record["supplier_id"]

        lookup_rfq_id = canonical_rfq_id or rfq_id
        if canonical_rfq_id:
            details["canonical_rfq_id"] = canonical_rfq_id

        self._remember_rfq_tail(lookup_rfq_id, details.get("supplier_id"))
        try:
            with self.agent_nick.get_db_connection() as conn:  # pragma: no cover - network
                with conn.cursor() as cur:
                    if "supplier_id" not in details:
                        cur.execute(
                            "SELECT supplier_id FROM proc.draft_rfq_emails WHERE rfq_id = %s ORDER BY created_on DESC LIMIT 1",
                            (lookup_rfq_id,),
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            details["supplier_id"] = row[0]

                    # Attempt to retrieve negotiation targets when the table exists.
                    try:
                        cur.execute(
                            "SELECT target_price, negotiation_round FROM proc.rfq_targets WHERE rfq_id = %s ORDER BY updated_on DESC LIMIT 1",
                            (lookup_rfq_id,),
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
                                (lookup_rfq_id,),
                            )
                            row = fallback_cur.fetchone()
                            if row and row[0]:
                                details.setdefault("round", int(row[0]))
        except Exception:
            logger.exception("Failed to load RFQ metadata for %s", rfq_id)

        return details

    def _get_rfq_candidates_for_tail(self, tail: str) -> List[Dict[str, object]]:
        tail_key = (tail or "").strip().lower()
        if not tail_key:
            return []

        if tail_key in self._rfq_tail_cache:
            return list(self._rfq_tail_cache.get(tail_key, []))

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            self._rfq_tail_cache[tail_key] = []
            return []

        rows: List[Tuple] = []
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT rfq_id, supplier_id
                        FROM proc.draft_rfq_emails
                        WHERE RIGHT(REGEXP_REPLACE(LOWER(rfq_id), '[^a-z0-9]', '', 'g'), 8) = %s
                        ORDER BY updated_on DESC, created_on DESC
                        """,
                        (tail_key,),
                    )
                    fetch_all = getattr(cur, "fetchall", None)
                    if callable(fetch_all):
                        rows = fetch_all()
                    else:  # pragma: no cover - defensive
                        row = cur.fetchone()
                        rows = [row] if row else []
        except Exception:
            logger.exception("Failed to load RFQ records for tail %s", tail_key)
            rows = []

        entries: List[Dict[str, object]] = []
        for row in rows:
            if not row:
                continue
            rfq_value = row[0]
            supplier_value = row[1] if len(row) > 1 else None
            entries.append({"rfq_id": rfq_value, "supplier_id": supplier_value})

        self._rfq_tail_cache[tail_key] = list(entries)
        return entries

    def _resolve_rfq_record(self, rfq_id: str) -> Optional[Dict[str, object]]:
        tail = self._normalise_rfq_value(rfq_id)
        if not tail:
            return None

        candidates = self._get_rfq_candidates_for_tail(tail)
        if not candidates:
            return None

        canonical = _canon_id(rfq_id)
        for entry in candidates:
            stored_id = entry.get("rfq_id")
            if stored_id and _canon_id(str(stored_id)) == canonical:
                return entry

        return candidates[0]

    def _remember_rfq_tail(self, rfq_id: Optional[str], supplier_id: Optional[str]) -> None:
        if not rfq_id:
            return

        tail = self._normalise_rfq_value(rfq_id)
        if not tail:
            return

        entry = {"rfq_id": rfq_id, "supplier_id": supplier_id}
        cached = self._rfq_tail_cache.get(tail)
        if not cached:
            self._rfq_tail_cache[tail] = [entry]
            return

        canonical = _canon_id(rfq_id)
        for idx, existing in enumerate(list(cached)):
            stored_id = existing.get("rfq_id")
            if stored_id and _canon_id(str(stored_id)) == canonical:
                cached[idx] = entry
                break
        else:
            cached.insert(0, entry)
            if len(cached) > 10:
                del cached[10:]

    def _load_messages(
        self,
        limit: Optional[int],
        *,
        mark_seen: bool,
        prefixes: Optional[Sequence[str]] = None,
        on_message: Optional[Callable[[Dict[str, object], Optional[datetime]], bool]] = None,
    ) -> List[Dict[str, object]]:

        if limit is not None:
            try:
                effective_limit = max(int(limit), 0)
            except Exception:
                effective_limit = 0
            if effective_limit == 0:
                return []
        else:
            effective_limit = None

        messages: List[Dict[str, object]] = []
        attempted_s3 = False

        if self.bucket:
            attempted_s3 = True
            messages = self._load_from_s3(
                effective_limit,
                prefixes=prefixes,
                on_message=on_message,
            )
        else:
            logger.warning("S3 bucket not configured; skipping direct poll for mailbox %s", self.mailbox_address)

        if messages:
            self._consecutive_empty_s3_batches = 0
            return messages

        if attempted_s3:
            self._consecutive_empty_s3_batches += 1
        else:
            self._consecutive_empty_s3_batches = 0

        should_fallback_to_imap = not attempted_s3 or (
            self._imap_fallback_attempts > 0
            and self._consecutive_empty_s3_batches >= self._imap_fallback_attempts
        )

        if should_fallback_to_imap and self._imap_configured():
            if attempted_s3 and self._consecutive_empty_s3_batches == self._imap_fallback_attempts:
                logger.info(
                    "No new S3 messages after %d poll(s); falling back to IMAP for mailbox %s",
                    self._imap_fallback_attempts,
                    self.mailbox_address,
                )
            imap_messages = self._load_from_imap(
                effective_limit,
                mark_seen=bool(mark_seen),
                on_message=on_message,
            )
            if imap_messages:
                self._consecutive_empty_s3_batches = 0
                return imap_messages

        return messages

    def _imap_configured(self) -> bool:
        host, user, password, _, _ = self._imap_settings()
        return bool(host and user and password)

    def _imap_settings(self) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        str,
        str,
    ]:
        host = getattr(self.settings, "imap_host", None)
        user = getattr(self.settings, "imap_user", None)
        password = getattr(self.settings, "imap_password", None)
        mailbox = getattr(self.settings, "imap_mailbox", "INBOX") or "INBOX"
        search_criteria = getattr(self.settings, "imap_search_criteria", "ALL") or "ALL"
        return host, user, password, mailbox, search_criteria

    def _load_from_imap(
        self,
        limit: Optional[int],
        *,
        mark_seen: bool,
        on_message: Optional[Callable[[Dict[str, object], Optional[datetime]], bool]] = None,
    ) -> List[Dict[str, object]]:
        host, user, password, mailbox, search_criteria = self._imap_settings()
        if not host or not user or not password:
            return []

        collected: List[Tuple[Optional[datetime], Dict[str, object]]] = []
        seen_ids: Set[str] = set()

        try:
            with imaplib.IMAP4_SSL(host) as client:
                client.login(user, password)
                status, _ = client.select(mailbox)
                if status != "OK":
                    logger.warning(
                        "IMAP fallback could not select mailbox %s for %s", mailbox, self.mailbox_address
                    )
                    client.logout()
                    return []

                status, data = client.search(None, search_criteria)
                if status != "OK" or not data:
                    client.logout()
                    return []

                raw_ids = data[0].split()
                if not raw_ids:
                    client.logout()
                    return []

                if limit is not None and limit > 0:
                    raw_ids = raw_ids[-limit:]

                fetch_command = "(RFC822)" if mark_seen else "(BODY.PEEK[])"

                for raw_id in reversed(raw_ids):
                    message_id_str = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    if message_id_str in seen_ids:
                        continue

                    status, payload = client.fetch(raw_id, fetch_command)
                    if status != "OK" or not payload:
                        continue

                    raw_bytes: Optional[bytes] = None
                    for part in payload:
                        if isinstance(part, tuple) and len(part) >= 2:
                            raw_bytes = part[1]
                            break
                    if raw_bytes is None:
                        continue

                    parsed = self._parse_inbound_object(
                        raw_bytes, key=f"imap/{message_id_str}"
                    )
                    parsed.setdefault("id", f"imap/{message_id_str}")
                    parsed.setdefault("s3_key", f"imap/{message_id_str}")
                    parsed.setdefault(
                        "_bucket", f"imap::{self.mailbox_address or 'unknown'}"
                    )
                    if not parsed.get("message_id"):
                        parsed["message_id"] = parsed.get("id")
                    parsed.setdefault("_source", "imap")
                    timestamp = self._coerce_imap_timestamp(parsed.get("received_at"))
                    if timestamp is None:
                        timestamp = datetime.now(timezone.utc)
                    collected.append((timestamp, parsed))
                    seen_ids.add(message_id_str)

                    if mark_seen:
                        try:
                            client.store(raw_id, "+FLAGS", "(\\Seen)")
                        except Exception:
                            logger.debug(
                                "Failed to set IMAP seen flag for %s", message_id_str, exc_info=True
                            )

                    should_stop = False
                    if on_message is not None:
                        try:
                            should_stop = bool(on_message(parsed, timestamp))
                        except Exception:
                            logger.exception(
                                "on_message callback failed for IMAP message %s", message_id_str
                            )

                    if limit is not None and len(collected) >= limit:
                        break
                    if should_stop:
                        break

                client.logout()
        except Exception:
            logger.exception(
                "IMAP fallback polling failed for mailbox %s", self.mailbox_address
            )
            return []

        if not collected:
            return []

        collected.sort(
            key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return [payload for _, payload in collected]

    @staticmethod
    def _coerce_imap_timestamp(value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, str):
            try:
                parsed = parsedate_to_datetime(value)
            except Exception:
                parsed = None
            if parsed is not None and parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        return None

    @staticmethod
    def _coerce_limit(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except Exception:
            return 0
        return parsed if parsed >= 0 else 0

    @staticmethod
    def _normalise_filter_value(value: object) -> Optional[str]:
        if value is None:
            return None
        try:
            text = _norm(str(value))
        except Exception:
            return None
        lowered = text.lower()
        return lowered or None

    @staticmethod
    def _normalise_rfq_value(value: object) -> Optional[str]:
        return _rfq_match_key(value)

    @staticmethod
    def _canonical_rfq(value: object) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            canonical = _canon_id(str(value))
        except Exception:
            return None
        return canonical or None

    def _register_workflow_mapping(
        self, workflow_id: Optional[str], rfq_canonicals: Iterable[str]
    ) -> None:
        if not rfq_canonicals:
            return

        workflow_key = None
        if workflow_id:
            workflow_key = self._normalise_filter_value(workflow_id)
        for canonical in rfq_canonicals:
            if not canonical:
                continue
            self._rfq_workflow_index.setdefault(canonical, set())
            if workflow_key:
                mapping = self._workflow_rfq_index.setdefault(workflow_key, set())
                mapping.add(canonical)
                self._rfq_workflow_index[canonical].add(workflow_key)

    @staticmethod
    def _normalise_email(value: object) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            _, address = parseaddr(text)
        except Exception:
            address = None
        if address:
            address = address.strip()
        else:
            # Fallback for strings like "Name <email>" when parseaddr fails
            match = re.search(r"<\s*([^>]+)\s*>", text)
            address = match.group(1).strip() if match else None
        if not address and "@" in text:
            address = text
        return address.lower() if address else None

    def _matches_filters(self, payload: Dict[str, object], filters: Dict[str, object]) -> bool:
        if not filters:
            return False

        payload_rfq_tail = self._normalise_rfq_value(payload.get("rfq_id"))
        payload_rfq_full = self._normalise_filter_value(payload.get("rfq_id"))
        payload_supplier = self._normalise_filter_value(payload.get("supplier_id"))
        payload_subject = self._normalise_filter_value(payload.get("subject")) or ""
        payload_sender = self._normalise_filter_value(payload.get("from_address"))
        payload_sender_email = self._normalise_email(payload.get("from_address"))
        payload_message = self._normalise_filter_value(payload.get("message_id")) or self._normalise_filter_value(payload.get("id"))
        payload_workflow = self._normalise_filter_value(payload.get("workflow_id"))

        payload_rfq_canonicals: Set[str] = set()
        primary_canonical = self._canonical_rfq(payload.get("rfq_id"))
        if primary_canonical:
            payload_rfq_canonicals.add(primary_canonical)
        related = payload.get("related_rfq_ids")
        if isinstance(related, list):
            for entry in related:
                canonical = self._canonical_rfq(entry)
                if canonical:
                    payload_rfq_canonicals.add(canonical)

        def _like(actual: Optional[str], expected_like: object) -> bool:
            needle = self._normalise_filter_value(expected_like)
            if not needle:
                return True
            if actual is None:
                return False

            pattern = re.escape(needle)
            # Support SQL-style and glob-style wildcards for convenience.
            pattern = (
                pattern.replace("%", ".*")
                .replace(r"\%", ".*")
                .replace("_", ".")
                .replace(r"\_", ".")
                .replace(r"\*", ".*")
            )
            regex = re.compile(f"^{pattern}$")
            if regex.fullmatch(actual):
                return True

            # Allow bare substrings (without wildcards) to behave like ``LIKE %needle%``
            if needle and "%" not in needle and "_" not in needle and "*" not in needle:
                return needle in actual

            return False

        for key, expected in filters.items():
            if expected in (None, ""):
                continue
            if key == "rfq_id":
                if payload_rfq_tail != self._normalise_rfq_value(expected):
                    return False
            elif key == "rfq_id_like":
                if not _like(payload_rfq_full, expected):
                    return False
            elif key == "supplier_id":
                if payload_supplier != self._normalise_filter_value(expected):
                    return False
            elif key == "supplier_id_like":
                if not _like(payload_supplier, expected):
                    return False
            elif key == "from_address":
                expected_normalised = self._normalise_filter_value(expected)
                expected_email = self._normalise_email(expected)
                candidates = [payload_sender, payload_sender_email]
                expectations = [expected_normalised, expected_email]
                if not any(
                    actual and expected_val and actual == expected_val
                    for actual in candidates
                    for expected_val in expectations
                ):
                    return False
            elif key == "workflow_id":
                expected_workflow = self._normalise_filter_value(expected)
                if not expected_workflow:
                    continue
                if payload_workflow == expected_workflow:
                    continue
                workflow_map = self._workflow_rfq_index.get(expected_workflow, set())
                if payload_rfq_canonicals and workflow_map:
                    if any(candidate in workflow_map for candidate in payload_rfq_canonicals):
                        continue
                if payload_rfq_canonicals:
                    if any(
                        expected_workflow in self._rfq_workflow_index.get(candidate, set())
                        for candidate in payload_rfq_canonicals
                    ):
                        continue
                return False
            elif key == "from_address_like":
                if not any(
                    _like(candidate, expected)
                    for candidate in (payload_sender, payload_sender_email)
                    if candidate
                ):
                    return False
            elif key == "subject_contains":
                needle = self._normalise_filter_value(expected)
                if needle and needle not in payload_subject:
                    return False
            elif key == "subject_like":
                if not _like(payload_subject, expected):
                    return False
            elif key == "message_id":
                if payload_message != self._normalise_filter_value(expected):
                    return False
            elif key == "message_id_like":
                if not _like(payload_message, expected):
                    return False
        return True

    @staticmethod
    def _apply_filter_defaults(
        payload: Dict[str, object], filters: Dict[str, object]
    ) -> None:
        if not isinstance(payload, dict) or not filters:
            return

        def _should_fill(key: str) -> bool:
            value = payload.get(key)
            if value is None:
                return True
            try:
                return str(value).strip() == ""
            except Exception:
                return False

        for field in ("rfq_id", "supplier_id", "from_address", "workflow_id"):
            if field in filters and _should_fill(field):
                candidate = filters.get(field)
                if candidate not in (None, ""):
                    payload[field] = candidate

        if "message_id" in filters and _should_fill("message_id"):
            candidate = filters.get("message_id")
            if candidate not in (None, ""):
                payload["message_id"] = candidate

    def _format_bucket_prefixes(
        self, prefixes: Optional[Sequence[str]] = None
    ) -> str:
        bucket = self.bucket or "<unset>"
        active_prefixes = list(prefixes) if prefixes is not None else list(self._prefixes)
        if not active_prefixes:
            active_prefixes = [""]

        entries: List[str] = []
        for raw_prefix in active_prefixes:
            prefix = (raw_prefix or "").lstrip("/")
            if prefix:
                uri = f"s3://{bucket}/{prefix}"
            else:
                uri = f"s3://{bucket}/"
            if not uri.endswith("/"):
                uri = f"{uri}/"
            entries.append(uri)

        return ", ".join(entries)

    def _derive_prefixes_for_filters(
        self, filters: Optional[Dict[str, object]]
    ) -> Optional[List[str]]:
        if not filters or not self._prefixes:
            return None

        rfq_value = filters.get("rfq_id") or filters.get("RFQ_ID")
        if not rfq_value:
            return None

        try:
            rfq_text = str(rfq_value).strip()
        except Exception:
            return None

        if not rfq_text:
            return None

        base_prefix = self._prefixes[0] if self._prefixes else ""
        base = base_prefix.rstrip("/")
        candidate = self._ensure_trailing_slash(f"{base}/{rfq_text.upper()}/ingest") if base else None
        if not candidate:
            return None

        prefixes = [candidate]
        if candidate not in self._prefixes:
            prefixes.append(base_prefix)
        return prefixes

    def record_dispatch_timestamp(self, dispatched_at: Optional[float] = None) -> None:
        """Record the moment an outbound email dispatch completed."""

        timestamp: Optional[float]
        if dispatched_at is None:
            timestamp = time.time()
        else:
            try:
                timestamp = float(dispatched_at)
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid dispatch timestamp %r", dispatched_at)
                return

        if timestamp is None:
            return

        self._last_dispatch_notified_at = timestamp
        try:
            setattr(self.agent_nick, "email_dispatch_last_sent_at", timestamp)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Unable to persist dispatch timestamp on agent context")

    def _respect_post_dispatch_wait(self) -> None:
        if self._dispatch_wait_seconds <= 0:
            return

        candidate_time: Optional[float] = self._last_dispatch_notified_at
        agent_time = getattr(self.agent_nick, "email_dispatch_last_sent_at", None)
        if isinstance(agent_time, (int, float)):
            agent_value = float(agent_time)
            candidate_time = agent_value if candidate_time is None else max(candidate_time, agent_value)

        if candidate_time is None:
            return

        if (
            self._last_dispatch_wait_acknowledged is not None
            and candidate_time <= self._last_dispatch_wait_acknowledged
        ):
            return

        now = time.time()
        elapsed = now - candidate_time
        remaining = self._dispatch_wait_seconds - elapsed
        if remaining > 0:
            pause_seconds = min(remaining, 5.0)
            logger.debug(
                "Short pause %.1fs after email dispatch (remaining %.1fs; target=%s, mailbox=%s)",
                pause_seconds,
                max(remaining - pause_seconds, 0.0),
                self._format_bucket_prefixes(),
                self.mailbox_address,
            )
            time.sleep(pause_seconds)

        self._last_dispatch_wait_acknowledged = candidate_time

    def _scan_recent_s3_objects(
        self,
        client,
        prefixes: Sequence[str],
        *,
        watermark_ts: Optional[datetime],
        watermark_key: str,
        enforce_window: bool = True,
    ) -> List[Tuple[str, str, Optional[datetime], Optional[str]]]:
        object_refs: List[Tuple[str, str, Optional[datetime], Optional[str]]] = []
        newest_seen: Optional[datetime] = None
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=self._poll_window_minutes)
        paginator = client.get_paginator("list_objects_v2")

        for prefix in prefixes:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                for obj in contents:
                    key = obj.get("Key")
                    if not key:
                        continue
                    if key.endswith("/") or "AMAZON_SES_SETUP_NOTIFICATION" in key:
                        continue
                    if "/ingest/" in key:
                        continue

                    last_modified_raw = obj.get("LastModified")
                    if isinstance(last_modified_raw, datetime):
                        if last_modified_raw.tzinfo is None:
                            last_modified = last_modified_raw.replace(tzinfo=timezone.utc)
                        else:
                            last_modified = last_modified_raw.astimezone(timezone.utc)
                    else:
                        last_modified = None

                    if last_modified is not None:
                        if newest_seen is None or last_modified > newest_seen:
                            newest_seen = last_modified
                        if enforce_window and last_modified < window_start:
                            continue
                    else:
                        # Unable to determine age; skip to avoid processing stale data.
                        if enforce_window:
                            continue

                    if not self._is_newer_than_watermark(
                        last_modified, key, watermark_ts, watermark_key
                    ):
                        continue

                    etag_value = obj.get("ETag") if isinstance(obj.get("ETag"), str) else None
                    object_refs.append((prefix, key, last_modified, etag_value))

        if not object_refs:
            if enforce_window and (newest_seen is None or newest_seen < window_start):
                newest_text = newest_seen.isoformat() if newest_seen else "<none>"
                sleep_hint = (
                    self.poll_interval_seconds
                    if self.poll_interval_seconds >= 0
                    else self._match_poll_sleep_seconds
                )
                logger.info(
                    "Newest key older than %dm window (newest=%s). Sleeping %ss.",
                    self._poll_window_minutes,
                    newest_text,
                    sleep_hint,
                )
            return []

        tz_aware_min = datetime.min.replace(tzinfo=timezone.utc)
        object_refs.sort(
            key=lambda item: ((item[2] or tz_aware_min), item[1]),
            reverse=True,
        )

        self._log_s3_preview(client, object_refs[: min(3, len(object_refs))])
        return object_refs

    def _log_s3_preview(
        self,
        client,
        preview_refs: Sequence[Tuple[str, str, Optional[datetime], Optional[str]]],
    ) -> None:
        if not preview_refs:
            return

        bucket_name = self.bucket
        if not bucket_name:
            return

        for _, key, _, _ in preview_refs:
            try:
                response = client.get_object(
                    Bucket=bucket_name,
                    Key=key,
                    Range="bytes=0-4095",
                )
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
                if error_code in ("InvalidRange", "NotImplemented"):
                    response = client.get_object(Bucket=bucket_name, Key=key)
                else:
                    logger.debug("Preview fetch failed for %s: %s", key, exc)
                    continue
            except Exception:
                logger.exception("Preview fetch failed for %s", key)
                continue

            try:
                blob = response["Body"].read()
            except Exception:
                logger.debug("Unable to read preview body for %s", key)
                continue

            headers = self._parse_email_headers(blob)
            subject = headers.get("subject") or ""
            recipient = headers.get("to") or ""
            logger.debug(
                "%s | Subject: %s | To: %s",
                key,
                subject or "<unknown>",
                recipient or "<unknown>",
            )

    @staticmethod
    def _parse_email_headers(raw_bytes: bytes) -> Dict[str, str]:
        if not raw_bytes:
            return {}

        try:
            parser = BytesParser(policy=policy.default)
            message = parser.parsebytes(raw_bytes, headersonly=True)
        except Exception:
            return {}

        return {
            "subject": _decode_mime_header(message.get("Subject")),
            "to": _decode_mime_header(message.get("To")),
        }

    def _load_from_s3(
        self,
        limit: Optional[int] = None,
        *,
        prefixes: Optional[Sequence[str]] = None,
        parser: Optional[Callable[[bytes], Dict[str, object]]] = None,
        newest_first: bool = True,
        on_message: Optional[Callable[[Dict[str, object], Optional[datetime]], bool]] = None,
    ) -> List[Dict[str, object]]:
        if not self.bucket:
            logger.warning("SES inbound bucket not configured; skipping poll")
            return []

        client = self._get_s3_client()
        logger.debug(
            "Listing inbound emails from %s for mailbox %s",
            self._format_bucket_prefixes(prefixes),
            self.mailbox_address,
        )
        collected: List[Tuple[Optional[datetime], Dict[str, object]]] = []
        seen_keys: Set[str] = set()

        active_prefixes = list(prefixes) if prefixes is not None else list(self._prefixes)
        parser_fn = parser or self._parse_inbound_object

        bypass_known_filters = bool(self._require_new_s3_objects)

        watermark_ts = self._last_watermark_ts
        watermark_key = self._last_watermark_key

        for prefix in active_prefixes:
            if prefix not in self._s3_prefix_watchers:
                self._s3_prefix_watchers[prefix] = S3ObjectWatcher(
                    limit=self._s3_watch_history_limit
                )

        enforce_recent_window = False
        if watermark_ts is None:
            enforce_recent_window = all(
                self._s3_prefix_watchers[prefix].primed
                or bool(self._s3_prefix_watchers[prefix].known)
                for prefix in active_prefixes
            )

        object_refs = self._scan_recent_s3_objects(
            client,
            active_prefixes,
            watermark_ts=watermark_ts,
            watermark_key=watermark_key,
            enforce_window=enforce_recent_window,
        )

        if not object_refs:
            return []

        processed_prefixes: Dict[str, bool] = {prefix: False for prefix in active_prefixes}

        for prefix, key, last_modified, etag in object_refs:
            if key in seen_keys:
                continue

            watcher = self._s3_prefix_watchers[prefix]

            skip_known_checks = not bypass_known_filters or watcher.primed

            if skip_known_checks:
                if self.state_store and key in self.state_store:
                    watcher.mark_known(key, last_modified)
                    continue
                if not watcher.is_new(key):
                    continue

            download = self._download_object(client, key, bucket=self.bucket)
            if download is None:
                continue
            raw, size_bytes = download

            parsed = self._invoke_parser(parser_fn, raw, key)
            parsed["id"] = key
            parsed["s3_key"] = key
            parsed["_s3_etag"] = etag
            parsed["_last_modified"] = last_modified
            if size_bytes is not None:
                parsed["_content_length"] = size_bytes
            collected.append((last_modified, parsed))
            seen_keys.add(key)
            watcher.mark_known(key, last_modified)
            processed_prefixes[prefix] = True
            logger.debug(
                "Queued message %s for processing from mailbox %s",
                key,
                self.mailbox_address,
            )

            should_stop = False
            if on_message is not None:
                try:
                    should_stop = bool(on_message(parsed, last_modified))
                except Exception:
                    logger.exception("on_message callback failed for %s", key)

            if limit is not None and len(collected) >= limit:
                break

            if should_stop:
                break

        if bypass_known_filters:
            for prefix, processed in processed_prefixes.items():
                if processed:
                    watcher = self._s3_prefix_watchers.get(prefix)
                    if watcher is not None and not watcher.primed:
                        watcher.primed = True

        if limit is not None and len(collected) >= limit:
            logger.debug(
                "Reached processing limit (%s) for mailbox %s",
                limit,
                self.mailbox_address,
            )

        collected.sort(key=lambda item: item[0] or datetime.min, reverse=newest_first)
        messages = [payload for _, payload in collected]
        return messages

    def _is_newer_than_watermark(
        self,
        last_modified: Optional[datetime],
        key: str,
        watermark_ts: Optional[datetime],
        watermark_key: str,
    ) -> bool:
        if last_modified is None:
            return watermark_ts is None
        candidate = last_modified
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)
        if watermark_ts is None:
            return True
        reference = watermark_ts
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        if candidate > reference:
            return True
        if candidate == reference and key > watermark_key:
            return True
        return False

    def _update_watermark(self, last_modified: datetime, key: str) -> None:
        if last_modified.tzinfo is None:
            candidate = last_modified.replace(tzinfo=timezone.utc)
        else:
            candidate = last_modified.astimezone(timezone.utc)

        if self._last_watermark_ts is None:
            self._last_watermark_ts = candidate
            self._last_watermark_key = key
            return

        reference = self._last_watermark_ts
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)

        if candidate > reference:
            self._last_watermark_ts = candidate
            self._last_watermark_key = key
        elif candidate == reference and key > self._last_watermark_key:
            self._last_watermark_ts = candidate
            self._last_watermark_key = key

    def _format_watermark(self) -> str:
        if self._last_watermark_ts is None:
            return "<unset>"
        timestamp = self._last_watermark_ts
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return f"{timestamp.isoformat()}::{self._last_watermark_key or '<none>'}"

    @staticmethod
    def _invoke_parser(
        parser: Callable[..., Dict[str, object]], raw: bytes, key: Optional[str]
    ) -> Dict[str, object]:
        try:
            return parser(raw, key=key)  # type: ignore[misc]
        except TypeError:
            return parser(raw)

    def _download_object(
        self, client, key: str, *, bucket: Optional[str] = None
    ) -> Optional[Tuple[bytes, Optional[int]]]:
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
            size_value = response.get("ContentLength")
            try:
                size_bytes: Optional[int] = int(size_value) if size_value is not None else None
            except (TypeError, ValueError):
                size_bytes = None
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
            return body, size_bytes
        except Exception:  # pragma: no cover - network/runtime
            logger.exception(
                "Failed to download SES message %s from bucket %s",
                key,
                bucket_name,
            )
            return None

    def _parse_inbound_object(
        self, raw_bytes: bytes, *, key: Optional[str] = None
    ) -> Dict[str, object]:
        """Parse an inbound S3 object into an email-like payload.

        The watcher historically only consumed raw ``.eml`` payloads written by
        SES.  Some suppliers now upload structured responses (for example PDFs
        or spreadsheets) directly into the monitored prefix.  In those cases we
        still want to kick off downstream processing even though the payload
        does not look like an RFC5322 message.  This helper first attempts to
        parse the object as an email and falls back to a generic file wrapper
        when no meaningful headers or body are present.
        """

        try:
            parsed_email = self._parse_email(raw_bytes)
        except Exception:
            logger.debug(
                "Treating S3 object %s as binary attachment after email parse failure",
                key or "<unknown>",
            )
            return self._build_file_payload(raw_bytes, key=key)

        if self._email_payload_has_content(parsed_email):
            return parsed_email

        logger.debug(
            "Treating S3 object %s as binary attachment due to empty email payload",
            key or "<unknown>",
        )
        return self._build_file_payload(raw_bytes, key=key, base_payload=parsed_email)

    def _parse_email(self, raw_bytes: bytes) -> Dict[str, object]:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        raw_subject = _decode_mime_header(message.get("subject", ""))
        normalised_subject = _norm(raw_subject)
        from_address = message.get("from", "")
        recipients = message.get_all("to", [])
        body = self._extract_body(message)
        normalised_body = _norm(body)
        header_rfq = message.get("X-Procwise-RFQ-ID") or message.get("X-Procwise-RFQ-Id")
        header_rfq = str(header_rfq).strip() if header_rfq else ""
        rfq_id = header_rfq or self._extract_rfq_id(
            f"{normalised_subject} {normalised_body}",
            raw_subject=raw_subject,
            normalised_subject=normalised_subject,
        )
        attachments = self._extract_attachments(message)
        return {
            "subject": raw_subject,
            "from": from_address,
            "body": body,
            "rfq_id": rfq_id,
            "rfq_id_header": header_rfq or None,
            "received_at": message.get("date"),
            "message_id": message.get("message-id"),
            "recipients": recipients,
            "attachments": attachments,
        }

    @staticmethod
    def _email_payload_has_content(payload: Dict[str, object]) -> bool:
        subject = str(payload.get("subject") or "").strip()
        attachments = payload.get("attachments") or []
        if isinstance(attachments, list) and attachments:
            return True

        from_hint = str(payload.get("from") or "").strip()
        recipients = payload.get("recipients") or []
        message_id = str(payload.get("message_id") or "").strip()

        if subject or from_hint or message_id:
            return True
        if isinstance(recipients, list) and recipients:
            return True

        return False

    def _build_file_payload(
        self,
        raw_bytes: bytes,
        *,
        key: Optional[str] = None,
        base_payload: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        filename = None
        if key:
            filename = key.rsplit("/", 1)[-1]
        if not filename:
            filename = "inbound-object"

        preview_bytes = raw_bytes[:4096]
        try:
            preview_text = preview_bytes.decode("utf-8", errors="ignore")
        except Exception:
            preview_text = ""

        rfq_sources: List[str] = []
        if key:
            rfq_sources.append(key)
        if base_payload:
            for candidate in (base_payload.get("subject"), base_payload.get("body")):
                if isinstance(candidate, str):
                    rfq_sources.append(candidate)
        if preview_text:
            rfq_sources.append(preview_text)

        rfq_id: Optional[str] = None
        for source in rfq_sources:
            rfq_id = self._extract_rfq_id(source)
            if rfq_id:
                break

        guessed_type, _ = mimetypes.guess_type(filename)
        attachment = {
            "filename": filename,
            "content_type": guessed_type or "application/octet-stream",
            "content": raw_bytes,
            "size": len(raw_bytes),
            "disposition": "attachment",
        }

        subject = filename
        from_hint = ""
        received_at = None
        message_id = None
        recipients: List[str] = []

        if base_payload:
            subject = base_payload.get("subject") or subject
            from_hint = str(base_payload.get("from") or "")
            received_at = base_payload.get("received_at")
            message_id = base_payload.get("message_id")
            base_recipients = base_payload.get("recipients")
            if isinstance(base_recipients, list):
                recipients = list(base_recipients)

        return {
            "subject": subject,
            "from": from_hint,
            "body": preview_text,
            "rfq_id": rfq_id,
            "received_at": received_at,
            "message_id": message_id,
            "recipients": recipients,
            "attachments": [attachment],
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

    def _extract_rfq_ids(
        self,
        text: str,
        *,
        raw_subject: Optional[str] = None,
        normalised_subject: Optional[str] = None,
    ) -> List[str]:
        if not text:
            if raw_subject is not None:
                logger.debug(
                    "RFQ match debug | subj_raw='%s' | subj_norm='%s' | ids=%s",
                    raw_subject[:200],
                    (normalised_subject or _norm(raw_subject))[:200],
                    [],
                )
            return []

        pattern = getattr(self.supplier_agent, "RFQ_PATTERN", None)
        if not pattern:
            pattern = RFQ_ID_RE

        if not hasattr(pattern, "findall"):
            try:
                pattern = re.compile(str(pattern))
            except re.error:
                logger.debug("Invalid RFQ pattern provided; skipping match")
                return None

        normalised_text = _norm(text)

        try:
            raw_matches = pattern.findall(normalised_text)
        except Exception:
            match = pattern.search(normalised_text)
            raw_matches = [match.group(0)] if match else []

        candidates: List[str] = []
        seen: Set[str] = set()

        for raw_match in raw_matches:
            if isinstance(raw_match, tuple):
                candidate = next((part for part in raw_match if part), "")
            else:
                candidate = raw_match
            if not candidate:
                continue
            canonical = _canon_id(candidate)
            if canonical in seen:
                continue
            seen.add(canonical)
            candidates.append(candidate)

        if not candidates:
            loose_matches = re.findall(
                r"RFQ(?:ID)?[:#\s-]*([0-9]{4,8})[-_\s]*([A-Za-z0-9]{4,})",
                normalised_text,
                flags=re.IGNORECASE,
            )
            for date_part, suffix in loose_matches:
                cleaned_suffix = re.sub(r"[^A-Za-z0-9]", "", suffix)
                if not cleaned_suffix:
                    continue
                trimmed_suffix = cleaned_suffix[:8]
                trimmed_date = re.sub(r"[^0-9]", "", date_part)[-8:]
                if len(trimmed_date) != 8 or len(trimmed_suffix) < 4:
                    continue
                candidate = f"RFQ-{trimmed_date}-{trimmed_suffix.upper()}"
                canonical = _canon_id(candidate)
                if canonical in seen:
                    continue
                seen.add(canonical)
                candidates.append(candidate)

        if raw_subject is not None:
            subject_for_log = normalised_subject or _norm(raw_subject)
            logger.debug(
                "RFQ match debug | subj_raw='%s' | subj_norm='%s' | ids=%s",
                raw_subject[:200],
                subject_for_log[:200] if subject_for_log else "",
                candidates,
            )

        return candidates

    def _extract_rfq_id(
        self,
        text: str,
        *,
        raw_subject: Optional[str] = None,
        normalised_subject: Optional[str] = None,
    ) -> Optional[str]:
        matches = self._extract_rfq_ids(
            text,
            raw_subject=raw_subject,
            normalised_subject=normalised_subject,
        )
        return matches[0] if matches else None

    def _rfq_candidates_from_metadata(
        self, message: Dict[str, object]
    ) -> List[Tuple[str, Optional[str]]]:
        candidates: List[Tuple[str, Optional[str]]] = []
        supplier_hint = message.get("supplier_id")
        keys = (
            "s3_key",
            "id",
            "message_id",
            "object_key",
            "receipt_handle",
        )
        for key in keys:
            value = message.get(key)
            if not isinstance(value, str) or not value:
                continue
            for match in RFQ_ID_RE.findall(value):
                if isinstance(match, tuple):
                    rfq_value = next((part for part in match if part), "")
                else:
                    rfq_value = match
                if not rfq_value:
                    continue
                candidates.append((rfq_value, supplier_hint if supplier_hint else None))
        return candidates

    def _resolve_rfq_from_tail_tokens(
        self, text: str, message: Dict[str, object]
    ) -> List[Tuple[str, str, Optional[str]]]:
        segments: List[str] = []
        if isinstance(text, str) and text.strip():
            segments.append(text)
        for key in ("s3_key", "id", "message_id"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                segments.append(value)
        combined = " ".join(segments)
        if not combined:
            return []

        tails = re.findall(r"\b([A-Za-z0-9]{8})\b", combined)
        if not tails:
            return []

        resolved: List[Tuple[str, str, Optional[str]]] = []
        seen_tails: Set[str] = set()
        for tail in tails:
            tail_key = tail.lower()
            if tail_key in seen_tails:
                continue
            seen_tails.add(tail_key)
            lookup_records = self._get_rfq_candidates_for_tail(tail_key)
            if not lookup_records:
                continue
            for record in lookup_records:
                rfq_value = record.get("rfq_id")
                if not rfq_value:
                    continue
                canonical = self._canonical_rfq(rfq_value)
                if not canonical:
                    continue
                if any(existing[0] == canonical for existing in resolved):
                    continue
                supplier_hint = record.get("supplier_id") if isinstance(record, dict) else None
                resolved.append((canonical, str(rfq_value), supplier_hint))
        return resolved

    def _ensure_s3_mapping(self, s3_key: Optional[str], rfq_id: Optional[object]) -> Optional[str]:
        if not self.bucket or not s3_key or rfq_id is None:
            return None

        rfq_text = str(rfq_id).strip()
        if not rfq_text:
            return None

        try:
            client = self._get_s3_client()
        except Exception:
            logger.debug("Unable to obtain S3 client for canonical mapping", exc_info=True)
            return None

        if client is None:
            return None

        self._tag_s3_object(client, self.bucket, s3_key, rfq_text)
        return self._copy_to_canonical(client, self.bucket, s3_key, rfq_text)

    def _tag_s3_object(self, client, bucket: str, key: str, rfq_id: str) -> None:
        try:
            client.put_object_tagging(
                Bucket=bucket,
                Key=key,
                Tagging={"TagSet": [{"Key": "rfq-id", "Value": rfq_id}]},
            )
        except ClientError as exc:
            logger.debug("Tagging failed for %s: %s", key, exc)
        except Exception:
            logger.debug("Tagging failed for %s due to unexpected error", key, exc_info=True)

    def _copy_to_canonical(self, client, bucket: str, key: str, rfq_id: str) -> Optional[str]:
        if not key:
            return None

        base_name = key.split("/")[-1]
        prefix_root = self._prefixes[0] if self._prefixes else "emails/"
        normalised_root = prefix_root.rstrip("/")
        if normalised_root:
            canonical_key = f"{normalised_root}/{rfq_id}/ingest/{base_name}"
        else:
            canonical_key = f"{rfq_id}/ingest/{base_name}"

        if canonical_key == key:
            return canonical_key

        try:
            client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": key},
                Key=canonical_key,
                TaggingDirective="REPLACE",
                Tagging=f"rfq-id={rfq_id}",
            )
        except ClientError as exc:
            logger.debug("Copy to canonical failed for %s: %s", key, exc)
            return None
        except Exception:
            logger.debug("Copy to canonical failed for %s due to unexpected error", key, exc_info=True)
            return None

        return canonical_key

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

        max_pool = self._coerce_int(
            getattr(self.settings, "s3_max_pool_connections", 50),
            default=50,
            minimum=1,
        )
        client_kwargs["config"] = Config(
            max_pool_connections=max_pool,
            retries={"max_attempts": 10, "mode": "adaptive"},
            read_timeout=20,
            connect_timeout=5,
        )

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

        prefix_summary = self._format_bucket_prefixes()
        logger.info(
            "Email watcher will poll %s for mailbox %s",
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

    def _resolve_bucket_and_prefix(
        self,
        bucket_candidates: Sequence[Optional[str]],
        *,
        configured_prefix: Optional[str],
        uri_prefix: Optional[str],
        default_bucket: str,
        default_prefix: str,
    ) -> Tuple[str, str]:
        default_prefix_normalised = self._ensure_trailing_slash(default_prefix)
        prefix_candidate: Optional[str]
        explicit_prefix_supplied = False

        if configured_prefix is None:
            prefix_candidate = uri_prefix or default_prefix_normalised
        else:
            trimmed = str(configured_prefix).strip()
            if not trimmed:
                prefix_candidate = uri_prefix or default_prefix_normalised
            else:
                normalised_configured = self._ensure_trailing_slash(trimmed)
                if normalised_configured.lower() != default_prefix_normalised.lower():
                    prefix_candidate = normalised_configured
                    explicit_prefix_supplied = True
                else:
                    prefix_candidate = uri_prefix or normalised_configured

        resolved_bucket: Optional[str] = None
        resolved_prefix = str(prefix_candidate or default_prefix_normalised)

        for candidate in bucket_candidates:
            bucket_text = str(candidate).strip() if candidate else ""
            if not bucket_text:
                continue
            bucket_value, derived_prefix = self._split_bucket_candidate(
                bucket_text, default_bucket=default_bucket
            )
            if bucket_value and not resolved_bucket:
                resolved_bucket = bucket_value
            if derived_prefix and not explicit_prefix_supplied:
                resolved_prefix = derived_prefix
            if resolved_bucket:
                break

        if not resolved_bucket:
            resolved_bucket = default_bucket

        resolved_prefix = self._ensure_trailing_slash(resolved_prefix or default_prefix)
        return resolved_bucket, resolved_prefix

    @staticmethod
    def _split_bucket_candidate(
        value: str,
        *,
        default_bucket: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        text = value.strip()
        if not text:
            return None, None
        if text.lower().startswith("s3://"):
            bucket, prefix = SESEmailWatcher._parse_s3_uri(text)
            return bucket, prefix or None
        if "/" in text:
            bucket, suffix = text.split("/", 1)
            return bucket or None, (suffix or None)
        lower_default = default_bucket.lower()
        if lower_default and text.lower().startswith(lower_default) and text.lower() != lower_default:
            suffix = text[len(default_bucket) :].lstrip("-_./")
            if suffix:
                if not suffix.endswith("/"):
                    suffix = f"{suffix}/"
                return default_bucket, suffix
        return text, None

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

