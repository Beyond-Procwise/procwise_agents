from __future__ import annotations

import gzip
import hashlib
import imaplib
import logging
import mimetypes
import re
import time
import unicodedata
import uuid
import json
import os
from concurrent.futures import ThreadPoolExecutor, Future
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

try:  # pragma: no cover - optional dependency in constrained environments
    from agents.quote_comparison_agent import QuoteComparisonAgent
except Exception:  # pragma: no cover - quote comparison may be unavailable during tests
    QuoteComparisonAgent = None  # type: ignore[assignment]
from services.email_dispatch_chain_store import (
    find_best_chain_match,
    mark_response as mark_dispatch_response,
)
from services.email_dispatch_service import find_recent_sent_for_supplier
from services.email_thread_store import (
    ensure_thread_table,
    lookup_thread_metadata,
    sanitise_thread_table_name,
)
from utils.email_markers import (
    extract_marker_token,
    extract_rfq_id,
    extract_run_id,
    extract_supplier_id,
    split_hidden_marker,
)
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
_SIMILARITY_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
_IMAP_UID_RE = re.compile(r"imap/(\d+)")


def _generate_unique_id(
    uid: str,
    msgid: str,
    date_hdr: str,
    sender: str,
    subject: str,
) -> str:
    """Generate a stable composite identifier for an IMAP message."""

    base = f"{uid}|{msgid or ''}|{date_hdr or ''}|{sender or ''}|{subject or ''}"
    unique_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{uid}-{unique_hash}"


def _extract_imap_uid(key: Optional[str]) -> Optional[int]:
    if not key:
        return None
    match = _IMAP_UID_RE.search(str(key))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


@dataclass
class MatchResult:
    rfq_id: str
    supplier_id: Optional[str]
    dispatch_message_id: Optional[str]
    matched_via: str
    confidence: float


def _extract_similarity_tokens(*values: object, limit: int = 64) -> Set[str]:
    """Return a set of normalised tokens for fuzzy comparisons."""

    tokens: Set[str] = set()
    if limit <= 0:
        limit = 0

    for value in values:
        if value in (None, ""):
            continue
        try:
            text = _norm(str(value))
        except Exception:
            continue
        if not text:
            continue
        lowered = text.lower()
        for match in _SIMILARITY_TOKEN_RE.findall(lowered):
            if not match:
                continue
            if limit and len(tokens) >= limit:
                break
            tokens.add(match)
        if limit and len(tokens) >= limit:
            break
    return tokens


def _extract_email_domain(value: object) -> Optional[str]:
    """Extract a lowercase domain from an email-like value."""

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
    candidate = address.strip() if address else text
    if "@" not in candidate:
        return None
    domain = candidate.split("@")[-1].strip().lower()
    return domain or None

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

    _MATCH_PRIORITIES: Dict[str, int] = {
        "header": 0,
        "thread_map": 1,
        "body": 4,
        "subject": 3,
        "dispatch_chain": 5,
        "fallback": 6,
    }

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
        quote_comparison_agent: Optional["QuoteComparisonAgent"] = None,
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
        if quote_comparison_agent is not None:
            self.quote_comparison_agent = quote_comparison_agent
        else:
            candidate = None
            if isinstance(getattr(agent_nick, "agents", None), dict):
                candidate = agent_nick.agents.get("quote_comparison")
            self.quote_comparison_agent = candidate
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
        self._imap_warning_logged = False

        # Ensure supporting negotiation tables exist so metadata lookups do
        # not fail on freshly provisioned databases.  The statements are safe
        # to run repeatedly thanks to ``IF NOT EXISTS`` guards.
        self._ensure_negotiation_tables()
        self._ensure_processed_registry_table()
        self._ensure_draft_rfq_email_constraints()
        self._ensure_email_watcher_watermarks_table()

        self._validate_inbound_configuration(prefix_value)

        self._thread_table_name = sanitise_thread_table_name(
            getattr(self.settings, "email_thread_table", None),
            logger=logger,
        )
        self._thread_table_ready = False

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
        self._workflow_run_index: Dict[str, str] = {}
        self._workflow_expected_counts: Dict[str, int] = {}
        self._workflow_processed_counts: Dict[str, int] = {}
        self._workflow_negotiation_jobs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        self._action_payload_cache: Dict[str, Dict[str, object]] = {}
        self._last_watermark_ts: Optional[datetime] = None
        self._last_watermark_key: str = ""
        self._last_uid: Optional[int] = None
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
        self._consecutive_empty_imap_batches = 0
        self._require_new_s3_objects = False
        self._dispatch_wait_seconds = max(
            0,
            self._coerce_int(
                getattr(self.settings, "email_inbound_initial_wait_seconds", 60),
                default=60,
                minimum=0,
            ),
        )
        self._post_dispatch_settle_seconds = max(
            0,
            self._coerce_int(
                getattr(
                    self.settings,
                    "email_inbound_post_dispatch_delay_seconds",
                    60,
                ),
                default=60,
                minimum=0,
            ),
        )
        self._last_dispatch_notified_at: Optional[float] = None
        self._last_dispatch_wait_acknowledged: Optional[float] = None
        self._dispatch_expectations: Dict[str, "_DispatchExpectation"] = {}
        self._completed_dispatch_actions: Set[str] = set()
        self._workflow_dispatch_actions: Dict[str, str] = {}
        self._workflow_context_buffer: Dict[str, List[Dict[str, object]]] = {}
        self._last_candidate_source: str = "none"
        self._negotiation_parallel_workers = max(
            1,
            self._coerce_int(
                getattr(self.settings, "negotiation_parallel_workers", 4),
                default=4,
                minimum=1,
            ),
        )

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
        expected_replies: Optional[int] = None,
        poll_interval: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Process a single batch of inbound emails.

        When ``expected_replies`` is provided the scan stops early once that many
        matching messages have been processed.  The ``poll_interval`` argument is
        accepted for API compatibility with callers coordinating wait intervals
        but is not used directly by this method.
        """

        filters: Dict[str, object] = dict(match_filters) if match_filters else {}
        results: List[Dict[str, object]] = []
        expected_cap = None
        if expected_replies is not None:
            expected_cap = self._coerce_int_value(expected_replies)
            if expected_cap is not None and expected_cap <= 0:
                expected_cap = None
        processed_matches = 0
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
                        "RFQ %s matched via index at %s — skipping mailbox poll for %s",
                        target_rfq_normalised,
                        hit["processed_at"],
                        self.mailbox_address,
                    )
                    return results

        logger.info(
            "Polling IMAP mailbox %s for new supplier responses (limit=%s)",
            self.mailbox_address or "<unknown>",
            limit if limit is not None else "unbounded",
        )

        previous_new_flag = self._require_new_s3_objects
        previous_watermark = (self._last_watermark_ts, self._last_watermark_key)
        self._require_new_s3_objects = bool(filters)

        try:
            prefixes = self._derive_prefixes_for_filters(filters)
            effective_limit = self._coerce_limit(limit)
            if expected_cap is not None:
                if effective_limit is None or expected_cap > effective_limit:
                    effective_limit = expected_cap

            loader_limit = limit
            if filters and self._filters_should_expand_limit(filters):
                loader_limit = None
            if expected_cap is not None:
                loader_limit = None

            match_found = False
            limit_exhausted = False
            stop_requested = False
            latest_candidate_ts: Optional[datetime] = None
            latest_candidate_key: str = ""

            def _record_watermark_candidate(
                candidate_ts: Optional[datetime],
                candidate_key: Optional[str],
            ) -> None:
                nonlocal latest_candidate_ts, latest_candidate_key
                if candidate_ts is None:
                    return
                key = str(candidate_key or "")
                timestamp = candidate_ts
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if latest_candidate_ts is None:
                    latest_candidate_ts = timestamp
                    latest_candidate_key = key
                    return
                reference = latest_candidate_ts
                if reference.tzinfo is None:
                    reference = reference.replace(tzinfo=timezone.utc)
                if timestamp > reference or (
                    timestamp == reference and key > latest_candidate_key
                ):
                    latest_candidate_ts = timestamp
                    latest_candidate_key = key

            if target_rfq_normalised and target_rfq_normalised in self._completed_targets:
                logger.info(
                    "RFQ %s already processed for mailbox %s; skipping poll",
                    target_rfq_normalised,
                    self.mailbox_address,
                )
                return []

            dispatch_expectation, dispatch_completed = self._respect_post_dispatch_wait(
                filters
            )
            if dispatch_expectation is not None and self._custom_loader is None:
                try:
                    self._acknowledge_recent_dispatch(dispatch_expectation, dispatch_completed)
                except Exception:
                    logger.exception(
                        "Failed to reconcile dispatched emails for action=%s",
                        dispatch_expectation.action_id,
                    )

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
                        messages = self._custom_loader(loader_limit)
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
                                _record_watermark_candidate(
                                    last_modified_hint,
                                    str(
                                        message.get("s3_key")
                                        or message.get("id")
                                        or ""
                                    ),
                                )
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
                            if matched:
                                match_found = True
                                if expected_cap is not None:
                                    processed_matches += 1
                                    if processed_matches >= expected_cap:
                                        stop_requested = True
                                        break
                            elif not filters and rfq_matched:
                                match_found = True
                            if (
                                effective_limit is not None
                                and effective_limit != 0
                                and len(results) >= effective_limit
                            ):
                                limit_exhausted = True
                            if should_stop and expected_cap is None:
                                stop_requested = True
                                break

                    if self._custom_loader is None or loader_empty:
                        processed_batch: List[Dict[str, object]] = []

                        def _on_message(
                            parsed: Dict[str, object],
                            last_modified: Optional[datetime] = None,
                        ) -> bool:
                            nonlocal total_processed, total_candidates, match_found, processed_matches, stop_requested
                            total_candidates += 1
                            if filters:
                                processed_batch.append(parsed)
                            matched, should_stop, rfq_matched, was_processed = self._process_candidate_message(
                                parsed,
                                match_filters=filters,
                                target_rfq_normalised=target_rfq_normalised,
                                results=results,
                                effective_limit=effective_limit,
                            )
                            if last_modified is not None:
                                _record_watermark_candidate(
                                    last_modified,
                                    str(
                                        parsed.get("s3_key")
                                        or parsed.get("id")
                                        or ""
                                    ),
                                )
                            if was_processed:
                                total_processed += 1
                            if matched:
                                match_found = True
                                if expected_cap is not None:
                                    processed_matches += 1
                            if (
                                effective_limit is not None
                                and effective_limit != 0
                                and len(results) >= effective_limit
                            ):
                                limit_exhausted = True
                            if not filters and rfq_matched:
                                match_found = True
                            if expected_cap is not None and matched and processed_matches >= expected_cap:
                                stop_requested = True
                                return True
                            if should_stop and expected_cap is None:
                                stop_requested = True
                                return True
                            return False

                        messages = self._load_messages(
                            loader_limit,
                            mark_seen=True,
                            prefixes=prefixes,
                            since=self._last_watermark_ts,
                            on_message=_on_message,
                        )
                        batch_count = len(processed_batch) if filters else len(messages)
                        if self._last_candidate_source == "imap":
                            source_label = "IMAP"
                        elif self._last_candidate_source == "s3":
                            source_label = "S3"
                        else:
                            source_label = "mail source"
                        logger.info(
                            "Retrieved %d candidate message(s) from %s for mailbox %s",
                            batch_count,
                            source_label,
                            self.mailbox_address,
                        )
                        if not filters:
                            for message in messages:
                                last_modified_hint = message.get("_last_modified")
                                if isinstance(last_modified_hint, datetime):
                                    _record_watermark_candidate(
                                        last_modified_hint,
                                        str(
                                            message.get("s3_key")
                                            or message.get("id")
                                            or ""
                                        ),
                                    )
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
                                if matched:
                                    match_found = True
                                    if expected_cap is not None:
                                        processed_matches += 1
                                        if processed_matches >= expected_cap:
                                            stop_requested = True
                                            break
                                elif not filters and rfq_matched:
                                    match_found = True
                                if (
                                    effective_limit is not None
                                    and effective_limit != 0
                                    and len(results) >= effective_limit
                                ):
                                    limit_exhausted = True
                                if should_stop and expected_cap is None:
                                    stop_requested = True
                                    break
                except Exception:  # pragma: no cover - network/runtime
                    logger.exception("Failed to load inbound SES messages")
                    break

                if expected_cap is None and target_rfq_normalised and match_found:
                    stop_requested = True

                if limit_exhausted or stop_requested:
                    break

                if expected_cap is None and match_found and self._should_stop_after_match(filters):
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
                if latest_candidate_ts is not None:
                    self._update_watermark(latest_candidate_ts, latest_candidate_key)
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
                if latest_candidate_ts is not None:
                    self._update_watermark(latest_candidate_ts, latest_candidate_key)
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

    def _should_stop_after_match(self, match_filters: Dict[str, object]) -> bool:
        if not match_filters:
            return False

        active_keys = {
            key
            for key, value in match_filters.items()
            if value not in (None, "", [], {}, ())
        }
        if not active_keys:
            return False

        non_stop_keys = {"supplier_id", "rfq_id"}
        if active_keys & non_stop_keys:
            # When matching on supplier/RFQ identifiers we want to process the
            # full batch of responses instead of short-circuiting on the first
            # hit.
            return False

        workflow_aggregate_keys = {
            "workflow_id",
            "action_id",
            "draft_action_id",
            "email_action_id",
        }

        if active_keys <= workflow_aggregate_keys and active_keys:
            # Allow processing of every response associated with the workflow/action.
            return False

        return True

    @staticmethod
    def _filters_should_expand_limit(match_filters: Dict[str, object]) -> bool:
        if not match_filters:
            return False

        active_keys = {
            key
            for key, value in match_filters.items()
            if value not in (None, "", [], {}, ())
        }
        if not active_keys:
            return False

        expand_keys = {"supplier_id", "rfq_id"}
        if active_keys & expand_keys:
            return True

        workflow_keys = {
            "workflow_id",
            "action_id",
            "draft_action_id",
            "email_action_id",
        }
        return bool(active_keys <= workflow_keys and active_keys)

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

        message_identifier = message.get("message_id") or message.get("message-id")
        if isinstance(message_identifier, bytes):
            try:
                message_identifier = message_identifier.decode()
            except Exception:
                message_identifier = str(message_identifier)
        if isinstance(message_identifier, str):
            message_identifier = message_identifier.strip()
        if not message_identifier:
            message_identifier = message.get("id")
        if isinstance(message_identifier, bytes):
            try:
                message_identifier = message_identifier.decode()
            except Exception:
                message_identifier = str(message_identifier)
        if not isinstance(message_identifier, str) or not message_identifier:
            message_identifier = str(uuid.uuid4())
        message_id = message_identifier
        message["message_id"] = message_id
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

        if self._is_processed_in_registry(
            bucket,
            s3_key,
            etag,
            message_id=message_id,
            mailbox=self.mailbox_address,
        ):
            logger.debug(
                "Skipping S3 object %s (etag=%s) for mailbox %s; already processed",
                s3_key,
                etag,
                self.mailbox_address,
            )
            return False, False, False, False

        execute_supplier_now = not match_filters

        try:
            processed, reason = self._process_message(
                message, execute_supplier=execute_supplier_now
            )
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
            thread_headers = message.get("thread_headers")
            if thread_headers:
                processed_payload.setdefault("thread_headers", thread_headers)
                if isinstance(processed, dict):
                    processed.setdefault("thread_headers", thread_headers)
            dispatch_record = (
                message.get("_dispatch_record")
                if isinstance(message.get("_dispatch_record"), dict)
                else None
            )
            self._hydrate_payload_from_marker(processed_payload, message)
            if isinstance(processed, dict):
                self._hydrate_payload_from_marker(processed, message)
            if dispatch_record:
                record_rfq = dispatch_record.get("rfq_id")
                record_supplier = dispatch_record.get("supplier_id")
                if record_rfq and processed_payload.get("rfq_id") is None:
                    processed_payload["rfq_id"] = record_rfq
                    if isinstance(processed, dict) and processed.get("rfq_id") is None:
                        processed["rfq_id"] = record_rfq
                if record_supplier and processed_payload.get("supplier_id") is None:
                    processed_payload["supplier_id"] = record_supplier
                    if isinstance(processed, dict) and processed.get("supplier_id") is None:
                        processed["supplier_id"] = record_supplier
                if processed_payload.get("dispatch_run_id") in (None, ""):
                    record_run = dispatch_record.get("run_id") or dispatch_record.get("dispatch_run_id")
                    if record_run:
                        processed_payload["dispatch_run_id"] = record_run
                        if isinstance(processed, dict) and not processed.get("dispatch_run_id"):
                            processed["dispatch_run_id"] = record_run
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
                if not message_match:
                    message_match = self._fallback_match_via_recent_drafts(
                        processed_payload,
                        match_filters,
                    )
                if (
                    message_match
                    and not execute_supplier_now
                    and reason != "processing_error"
                ):
                    try:
                        processed, reason = self._process_message(
                            message, execute_supplier=True
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception(
                            "Failed to finalise supplier processing for %s", message_id
                        )
                        processed, reason = None, "processing_error"
                        message_match = False
                    else:
                        if processed:
                            processed_payload = self._record_processed_payload(
                                message_id, processed
                            )
                            thread_headers = message.get("thread_headers")
                            if thread_headers:
                                processed_payload.setdefault("thread_headers", thread_headers)
                                if isinstance(processed, dict):
                                    processed.setdefault("thread_headers", thread_headers)
                            if dispatch_record:
                                record_rfq = dispatch_record.get("rfq_id")
                                record_supplier = dispatch_record.get("supplier_id")
                                if record_rfq and not processed_payload.get("rfq_id"):
                                    processed_payload["rfq_id"] = record_rfq
                                    if isinstance(processed, dict) and not processed.get("rfq_id"):
                                        processed["rfq_id"] = record_rfq
                                if record_supplier and not processed_payload.get("supplier_id"):
                                    processed_payload["supplier_id"] = record_supplier
                                    if isinstance(processed, dict) and not processed.get("supplier_id"):
                                        processed["supplier_id"] = record_supplier
                                if processed_payload.get("dispatch_run_id") in (None, ""):
                                    record_run = dispatch_record.get("run_id") or dispatch_record.get(
                                        "dispatch_run_id"
                                    )
                                    if record_run:
                                        processed_payload["dispatch_run_id"] = record_run
                                        if isinstance(processed, dict) and not processed.get(
                                            "dispatch_run_id"
                                        ):
                                            processed["dispatch_run_id"] = record_run
                            self._apply_filter_defaults(
                                processed_payload, match_filters
                            )
                            message_match = self._matches_filters(
                                processed_payload, match_filters
                            )
                            if not message_match:
                                message_match = self._fallback_match_via_recent_drafts(
                                    processed_payload,
                                    match_filters,
                                )
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

            match_source = processed_payload.get("matched_via") or processed.get("matched_via")
            match_score = processed_payload.get("match_score")
            if match_score is None and isinstance(processed, dict):
                match_score = processed.get("match_score")

            if not match_source or match_source == "unknown":
                header_candidate = message.get("rfq_id_header")
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
                        canonical_rfq_full
                        and subject_candidate
                        and canonical_rfq_full in _norm(str(subject_candidate)).upper()
                    ):
                        match_source = "subject"
                    elif (
                        canonical_rfq_full
                        and body_candidate
                        and canonical_rfq_full in _norm(str(body_candidate)).upper()
                    ):
                        match_source = "body"
                if (not match_source or match_source == "unknown") and dispatch_record:
                    record_source = dispatch_record.get("match_source")
                    if isinstance(record_source, str) and record_source:
                        match_source = record_source
                    else:
                        match_source = "dispatch"
                if not match_source or match_source == "unknown":
                    match_source = "body" if canonical_rfq_full else "unknown"

            processed_payload["matched_via"] = match_source or "unknown"
            if isinstance(processed, dict):
                processed["matched_via"] = match_source or "unknown"
            if match_score is not None:
                processed_payload["match_score"] = match_score
                if isinstance(processed, dict):
                    processed["match_score"] = match_score
            processed_payload.setdefault("mailbox", self.mailbox_address)

            logger.info(
                "rfq_email_processed mailbox=%s rfq_id=%s supplier_id=%s s3_key=%s etag=%s size_bytes=%s price=%s lead_time=%s matched_via=%s score=%s",
                self.mailbox_address,
                processed_payload.get("rfq_id"),
                processed_payload.get("supplier_id"),
                log_s3_key,
                etag or "",
                size_bytes if size_bytes is not None else "unknown",
                processed_payload.get("price"),
                processed_payload.get("lead_time"),
                processed_payload.get("matched_via"),
                processed_payload.get("match_score"),
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
            run_identifier = processed_payload.get("dispatch_run_id")
            if run_identifier:
                metadata["dispatch_run_id"] = run_identifier
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
            self._record_processed_in_registry(
                bucket,
                s3_key,
                etag,
                rfq_id,
                message_id=message_id,
                mailbox=self.mailbox_address,
            )

        if processed_payload:
            logger.info(
                "Processed inbound email %s → RFQ %s, Supplier %s",
                message_id,
                processed_payload.get("rfq_id") or "<unknown>",
                processed_payload.get("supplier_id") or "<unknown>",
            )
        else:
            logger.info(
                "Ignored inbound email %s for mailbox %s (status=%s)",
                message_id,
                self.mailbox_address,
                metadata.get("status") if isinstance(metadata, dict) else "unknown",
            )

        matched = bool(message_match)
        should_stop = False

        include_payload = False
        if processed_payload:
            if match_filters:
                include_payload = message_match
            elif target_rfq_normalised and rfq_match:
                include_payload = True
            elif not match_filters:
                include_payload = True

        if include_payload and effective_limit != 0:
            can_append = True
            if (
                effective_limit is not None
                and effective_limit != 0
                and len(results) >= effective_limit
            ):
                can_append = False
            if can_append:
                results.append(processed_payload)

        # Stop conditions: either a filter match or an RFQ match should stop polling
        if matched and self._should_stop_after_match(match_filters):
            should_stop = True
            logger.debug(
                "Stopping poll once after matching filters for message %s",
                message_id,
            )

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
            effective_limit = None
            if limit is not None:
                try:
                    remaining = int(limit) - processed_total
                except Exception:
                    remaining = limit  # type: ignore[assignment]
                if remaining <= 0:
                    logger.info(
                        "Email watcher reached requested limit (%s) after %d iteration(s)",
                        limit,
                        iterations,
                    )
                    break
                effective_limit = max(1, remaining)
            batch = self.poll_once(limit=effective_limit)
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
                if limit is None or processed_total >= (limit or 0):
                    break
                logger.debug(
                    "Processed batch below requested limit (%s); continuing polling for remaining %d message(s)",
                    limit,
                    max(0, (limit or 0) - processed_total),
                )
                continue
            if processed_total > 0:
                logger.info(
                    "Email watcher processed %d message(s) but no further matches were found; exiting",
                    processed_total,
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
            messages = self._load_messages(
                limit_int, mark_seen=False, prefixes=None, since=None
            )

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
                            base_subject TEXT,
                            initial_body TEXT,
                            updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            CONSTRAINT negotiation_session_state_pk PRIMARY KEY (rfq_id, supplier_id)
                        )
                        """
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS base_subject TEXT"
                    )
                    cur.execute(
                        "ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS initial_body TEXT"
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
                            processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            mailbox TEXT,
                            message_id TEXT
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
                        ALTER TABLE proc.processed_emails
                            ADD COLUMN IF NOT EXISTS mailbox TEXT
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE proc.processed_emails
                            ADD COLUMN IF NOT EXISTS message_id TEXT
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
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS processed_emails_message_id_idx
                            ON proc.processed_emails (message_id)
                        """
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS processed_emails_mailbox_message_id_uidx
                            ON proc.processed_emails (COALESCE(mailbox, ''), message_id)
                            WHERE message_id IS NOT NULL AND message_id <> ''
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
        *,
        message_id: Optional[str] = None,
        mailbox: Optional[str] = None,
    ) -> bool:
        bucket_checked = False
        if bucket and key:
            bucket_checked = True

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return False

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    if bucket_checked:
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
                        if row:
                            return True

                    if message_id:
                        mailbox_filter = mailbox or None
                        cur.execute(
                            """
                            SELECT 1
                            FROM proc.processed_emails
                            WHERE message_id = %s
                              AND (%s IS NULL OR mailbox IS NULL OR mailbox = %s)
                            LIMIT 1
                            """,
                            (message_id, mailbox_filter, mailbox_filter),
                        )
                        row = cur.fetchone()
                        if row:
                            return True
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
                    self._last_uid = _extract_imap_uid(self._last_watermark_key)
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
        *,
        message_id: Optional[str] = None,
        mailbox: Optional[str] = None,
    ) -> None:
        entries: List[Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]] = []

        if bucket and key:
            entries.append((bucket, key, etag or "", rfq_id, mailbox, message_id))
        elif message_id:
            mailbox_label = str(mailbox or "unknown").strip() or "unknown"
            sentinel_bucket = f"imap::{mailbox_label}"
            sentinel_key = message_id
            entries.append((sentinel_bucket, sentinel_key, "", rfq_id, mailbox, message_id))

        if not entries:
            return

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    for entry_bucket, entry_key, entry_etag, entry_rfq, entry_mailbox, entry_message_id in entries:
                        cur.execute(
                            """
                            INSERT INTO proc.processed_emails (
                                bucket,
                                key,
                                etag,
                                rfq_id,
                                processed_at,
                                mailbox,
                                message_id
                            )
                            VALUES (%s, %s, %s, %s, NOW(), %s, %s)
                            ON CONFLICT (bucket, key, etag)
                            DO UPDATE SET
                                rfq_id = EXCLUDED.rfq_id,
                                mailbox = COALESCE(EXCLUDED.mailbox, proc.processed_emails.mailbox),
                                message_id = COALESCE(EXCLUDED.message_id, proc.processed_emails.message_id),
                                processed_at = NOW()
                            """,
                            (
                                entry_bucket,
                                entry_key,
                                entry_etag,
                                entry_rfq,
                                entry_mailbox,
                                entry_message_id,
                            ),
                        )
                conn.commit()
        except Exception:
            logger.exception(
                "Failed to record processed email %s in registry",
                key or message_id,
            )

    def _process_message_fields_only(self, message: Dict[str, object]) -> Dict[str, object]:
        processed: Dict[str, object] = {}

        subject = str(message.get("subject") or "")
        body = str(message.get("body") or "")
        processed["subject"] = subject
        processed["message_body"] = body

        supplier_hint = message.get("supplier_id")
        if supplier_hint not in (None, ""):
            try:
                processed["supplier_id"] = str(supplier_hint)
            except Exception:
                processed["supplier_id"] = str(supplier_hint)

        comment, remainder = split_hidden_marker(body)
        analysis_body = remainder or body

        hidden_supplier = extract_supplier_id(comment)
        if hidden_supplier not in (None, ""):
            try:
                supplier_value = str(hidden_supplier).strip()
            except Exception:
                supplier_value = str(hidden_supplier)
            if supplier_value:
                processed["supplier_id"] = supplier_value
                message.setdefault("supplier_id", supplier_value)

        rfq_candidate: Optional[str] = None
        matched_via: Optional[str] = None
        hidden_marker = False

        hidden_rfq = extract_rfq_id(comment)
        if hidden_rfq:
            rfq_candidate = hidden_rfq
            matched_via = "body"
            hidden_marker = True

        if not rfq_candidate:
            field_rfq = message.get("rfq_id")
            if isinstance(field_rfq, str) and field_rfq.strip():
                rfq_candidate = field_rfq.strip()
                matched_via = message.get("matched_via") or "body"

        if not rfq_candidate and subject:
            subject_norm = _norm(subject)
            subject_match = self._extract_rfq_id(
                subject,
                raw_subject=subject,
                normalised_subject=subject_norm,
            )
            if subject_match:
                rfq_candidate = subject_match
                matched_via = "subject"

        if not rfq_candidate and analysis_body:
            body_match = self._extract_rfq_id(analysis_body)
            if body_match:
                rfq_candidate = body_match
                matched_via = "body"

        if not rfq_candidate:
            metadata_candidates = self._rfq_candidates_from_metadata(message)
            if metadata_candidates:
                rfq_candidate, supplier_from_meta = metadata_candidates[0]
                if supplier_from_meta not in (None, ""):
                    try:
                        processed.setdefault("supplier_id", str(supplier_from_meta))
                    except Exception:
                        processed.setdefault("supplier_id", str(supplier_from_meta))
                matched_via = "body"

        if rfq_candidate:
            processed["rfq_id"] = rfq_candidate
            processed["matched_via"] = matched_via or "body"
            if hidden_marker:
                processed["_hidden_marker_match"] = True

        return processed

    def match_inbound_email(self, message: Dict[str, object]) -> Optional[MatchResult]:
        header_value = message.get("rfq_id_header")
        header_tail = self._normalise_rfq_value(header_value)
        if header_tail:
            canonical = self._canonical_rfq(header_value) or _canon_id(header_tail)
            if canonical:
                supplier_hint = message.get("supplier_id")
                supplier_value = (
                    str(supplier_hint).strip()
                    if supplier_hint not in (None, "")
                    else None
                )
                return MatchResult(
                    rfq_id=canonical,
                    supplier_id=supplier_value,
                    dispatch_message_id=None,
                    matched_via="header",
                    confidence=1.0,
                )

        thread_identifiers = self._collect_thread_identifiers(message)
        if thread_identifiers:
            message.setdefault("_match_thread_ids", thread_identifiers)

        thread_match: Optional[Tuple[str, Optional[str]]] = None
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if thread_identifiers and callable(get_conn) and self._thread_table_name:
            try:
                with get_conn() as conn:
                    if self._ensure_thread_table_ready(conn):
                        thread_match = lookup_thread_metadata(
                            conn,
                            self._thread_table_name,
                            thread_identifiers,
                            logger=logger,
                        )
            except Exception:
                logger.exception(
                    "Failed to resolve RFQ from thread headers for mailbox %s",
                    self.mailbox_address,
                )
                thread_match = None
        if thread_match:
            rfq_from_threads, supplier_from_thread = thread_match
            canonical = self._canonical_rfq(rfq_from_threads) or _canon_id(rfq_from_threads)
            if canonical:
                supplier_hint = (
                    message.get("supplier_id")
                    or supplier_from_thread
                )
                supplier_value = (
                    str(supplier_hint).strip()
                    if supplier_hint not in (None, "")
                    else None
                )
                return MatchResult(
                    rfq_id=canonical,
                    supplier_id=supplier_value,
                    dispatch_message_id=None,
                    matched_via="thread_map",
                    confidence=0.95,
                )

        processed = self._process_message_fields_only(message)
        rfq_value = processed.get("rfq_id")
        rfq_norm = self._normalise_rfq_value(rfq_value)
        if rfq_norm and rfq_value:
            canonical = self._canonical_rfq(rfq_value) or _canon_id(rfq_value)
            if canonical:
                supplier_hint = processed.get("supplier_id") or message.get("supplier_id")
                supplier_value = (
                    str(supplier_hint).strip()
                    if supplier_hint not in (None, "")
                    else None
                )
                via = processed.get("matched_via") or "body"
                hidden_marker = bool(processed.get("_hidden_marker_match"))
                score = 0.7
                if via == "subject":
                    score = 0.8
                elif via == "body":
                    score = 0.9 if hidden_marker else 0.7
                return MatchResult(
                    rfq_id=canonical,
                    supplier_id=supplier_value,
                    dispatch_message_id=None,
                    matched_via=via,
                    confidence=score,
                )

        chain_hit: Optional[Dict[str, object]] = None
        if callable(get_conn):
            raw_lookback = getattr(self.settings, "dispatch_chain_lookback_days", 5)
            try:
                lookback_days = int(raw_lookback)
            except Exception:
                lookback_days = 5
            if lookback_days <= 0:
                lookback_days = 1
            try:
                with get_conn() as conn:
                    chain_hit = find_best_chain_match(
                        conn,
                        message,
                        mailbox=self.mailbox_address,
                        lookback_days=lookback_days,
                        thread_identifiers=message.get("_match_thread_ids") or thread_identifiers,
                        recipients_hint=True,
                    )
            except Exception:
                logger.debug("Failed to resolve RFQ via dispatch chain", exc_info=True)
                chain_hit = None
        if chain_hit and chain_hit.get("rfq_id"):
            canonical = self._canonical_rfq(chain_hit.get("rfq_id")) or _canon_id(
                chain_hit.get("rfq_id")
            )
            if canonical:
                metadata_doc = chain_hit.get("dispatch_metadata")
                dispatch_record = {
                    "rfq_id": chain_hit.get("rfq_id"),
                    "supplier_id": chain_hit.get("supplier_id"),
                    "match_source": chain_hit.get("matched_via") or "dispatch_chain",
                }
                if chain_hit.get("message_id"):
                    dispatch_record["message_id"] = chain_hit.get("message_id")
                if chain_hit.get("thread_index") is not None:
                    dispatch_record["thread_index"] = chain_hit.get("thread_index")
                if chain_hit.get("created_at") is not None:
                    dispatch_record["created_at"] = chain_hit.get("created_at")
                if isinstance(metadata_doc, dict):
                    dispatch_record["dispatch_metadata"] = metadata_doc
                    run_identifier = metadata_doc.get("run_id") or metadata_doc.get("dispatch_token")
                    if run_identifier:
                        dispatch_record["run_id"] = run_identifier
                message["_dispatch_record"] = dispatch_record
                supplier_hint = chain_hit.get("supplier_id") or message.get("supplier_id")
                supplier_value = (
                    str(supplier_hint).strip()
                    if supplier_hint not in (None, "")
                    else None
                )
                matched_label = chain_hit.get("matched_via") or "dispatch_chain"
                confidence = 0.6 if matched_label == "time_window" else 0.75
                dispatch_message_id = chain_hit.get("message_id")
                return MatchResult(
                    rfq_id=canonical,
                    supplier_id=supplier_value,
                    dispatch_message_id=dispatch_message_id,
                    matched_via="dispatch_chain",
                    confidence=confidence,
                )

        fallback_hit: Optional[Dict[str, object]] = None
        if callable(get_conn):
            try:
                with get_conn() as conn:
                    fallback_hit = find_recent_sent_for_supplier(
                        conn,
                        message,
                        mailbox=self.mailbox_address,
                        window_minutes=getattr(
                            self.settings,
                            "recent_supplier_fallback_minutes",
                            10,
                        ),
                    )
            except Exception:
                logger.debug(
                    "Failed to resolve recent supplier dispatch fallback", exc_info=True
                )
                fallback_hit = None
        if fallback_hit and fallback_hit.get("rfq_id"):
            canonical = self._canonical_rfq(fallback_hit.get("rfq_id")) or _canon_id(
                fallback_hit.get("rfq_id")
            )
            if canonical:
                metadata_doc = fallback_hit.get("dispatch_metadata")
                dispatch_record = {
                    "rfq_id": fallback_hit.get("rfq_id"),
                    "supplier_id": fallback_hit.get("supplier_id"),
                    "match_source": "fallback",
                }
                if fallback_hit.get("message_id"):
                    dispatch_record["message_id"] = fallback_hit.get("message_id")
                if fallback_hit.get("thread_index") is not None:
                    dispatch_record["thread_index"] = fallback_hit.get("thread_index")
                if fallback_hit.get("created_at") is not None:
                    dispatch_record["created_at"] = fallback_hit.get("created_at")
                if isinstance(metadata_doc, dict):
                    dispatch_record["dispatch_metadata"] = metadata_doc
                    run_identifier = metadata_doc.get("run_id") or metadata_doc.get("dispatch_token")
                    if run_identifier:
                        dispatch_record["run_id"] = run_identifier
                message["_dispatch_record"] = dispatch_record
                supplier_hint = fallback_hit.get("supplier_id") or message.get("supplier_id")
                supplier_value = (
                    str(supplier_hint).strip()
                    if supplier_hint not in (None, "")
                    else None
                )
                dispatch_message_id = fallback_hit.get("message_id")
                return MatchResult(
                    rfq_id=canonical,
                    supplier_id=supplier_value,
                    dispatch_message_id=dispatch_message_id,
                    matched_via="fallback",
                    confidence=0.5,
                )

        return None

    def _process_message(
        self,
        message: Dict[str, object],
        *,
        execute_supplier: bool = True,
    ) -> tuple[Optional[Dict[str, object]], Optional[str]]:
        subject = str(message.get("subject", ""))
        body = str(message.get("body", ""))
        comment, body_remainder = split_hidden_marker(body)
        analysis_body = body_remainder or body
        subject_normalised = _norm(subject)
        body_normalised = _norm(analysis_body)
        from_address = str(message.get("from", ""))

        hidden_supplier = extract_supplier_id(comment)
        if hidden_supplier not in (None, "") and not message.get("supplier_id"):
            try:
                supplier_value = str(hidden_supplier).strip()
            except Exception:
                supplier_value = str(hidden_supplier)
            if supplier_value:
                message["supplier_id"] = supplier_value

        match_hint = self.match_inbound_email(message)

        tail_supplier_map: Dict[str, Optional[str]] = {}
        canonical_map: Dict[str, str] = {}
        canonical_order: Dict[str, int] = {}
        ordered_canonicals: List[str] = []
        candidates: Dict[str, SESEmailWatcher._MatchCandidate] = {}

        def _register_candidate(
            rfq_value: Optional[str],
            *,
            supplier_hint: Optional[object] = None,
            matched_via: str,
            score: float,
            priority: Optional[int] = None,
            dispatch_record: Optional[Dict[str, object]] = None,
            thread_index: Optional[int] = None,
            created_at: Optional[datetime] = None,
        ) -> None:
            if not rfq_value:
                return
            canonical = self._canonical_rfq(rfq_value)
            if not canonical:
                return
            order_index = canonical_order.get(canonical)
            if order_index is None:
                order_index = len(canonical_order)
                canonical_order[canonical] = order_index
                canonical_map[canonical] = str(rfq_value)
                ordered_canonicals.append(canonical)
            supplier_value: Optional[str] = None
            if supplier_hint not in (None, ""):
                supplier_value = str(supplier_hint)
                tail_supplier_map.setdefault(canonical, supplier_value)
            candidate = SESEmailWatcher._MatchCandidate(
                rfq_id=str(rfq_value),
                supplier_id=supplier_value,
                matched_via=matched_via,
                score=score,
                priority=priority if priority is not None else self._priority_for(matched_via),
                order=order_index,
                dispatch_record=dict(dispatch_record) if isinstance(dispatch_record, dict) else dispatch_record,
                thread_index=thread_index,
                created_at=created_at if isinstance(created_at, datetime) else None,
            )
            existing = candidates.get(canonical)
            if existing is None or self._prefer_candidate(candidate, existing):
                if existing is not None:
                    if not candidate.supplier_id and existing.supplier_id:
                        candidate.supplier_id = existing.supplier_id
                    if candidate.dispatch_record is None and existing.dispatch_record is not None:
                        candidate.dispatch_record = existing.dispatch_record
                    if candidate.thread_index is None and existing.thread_index is not None:
                        candidate.thread_index = existing.thread_index
                    if candidate.created_at is None and existing.created_at is not None:
                        candidate.created_at = existing.created_at
                candidates[canonical] = candidate
            else:
                if supplier_value and not existing.supplier_id:
                    existing.supplier_id = supplier_value
                if dispatch_record and existing.dispatch_record is None:
                    existing.dispatch_record = dict(dispatch_record)
                if thread_index is not None and (existing.thread_index or 0) < thread_index:
                    existing.thread_index = thread_index
                if created_at and existing.created_at is None and isinstance(created_at, datetime):
                    existing.created_at = created_at

        def _best_score() -> float:
            if not candidates:
                return -1.0
            return max(candidate.score for candidate in candidates.values())

        if match_hint:
            dispatch_hint = (
                message.get("_dispatch_record")
                if isinstance(message.get("_dispatch_record"), dict)
                else None
            )
            supplier_hint_value = match_hint.supplier_id or message.get("supplier_id")
            _register_candidate(
                match_hint.rfq_id,
                supplier_hint=supplier_hint_value,
                matched_via=match_hint.matched_via,
                score=match_hint.confidence,
                dispatch_record=dispatch_hint,
            )

        raw_rfq = message.get("rfq_id")
        supplier_hint = message.get("supplier_id")
        if isinstance(raw_rfq, str) and raw_rfq.strip():
            _register_candidate(raw_rfq, supplier_hint=supplier_hint, matched_via="body", score=0.7)
        elif isinstance(raw_rfq, (list, tuple, set)):
            for entry in raw_rfq:
                if isinstance(entry, str) and entry.strip():
                    _register_candidate(entry, supplier_hint=supplier_hint, matched_via="body", score=0.7)

        header_candidate = message.get("rfq_id_header")
        if isinstance(header_candidate, str) and header_candidate.strip():
            _register_candidate(header_candidate, supplier_hint=supplier_hint, matched_via="header", score=1.0)

        thread_identifiers = message.get("_match_thread_ids") or self._collect_thread_identifiers(
            message
        )
        if _best_score() < 0.95:
            thread_match = self._resolve_rfq_from_thread_headers(message)
            if thread_match:
                rfq_value, supplier_from_thread = thread_match
                _register_candidate(
                    rfq_value,
                    supplier_hint=supplier_from_thread or supplier_hint,
                    matched_via="thread_map",
                    score=0.95,
                )

        hidden_rfq = extract_rfq_id(comment)
        if hidden_rfq:
            _register_candidate(hidden_rfq, supplier_hint=supplier_hint, matched_via="body", score=0.9)

        subject_matches = self._extract_rfq_ids(
            subject,
            raw_subject=subject,
            normalised_subject=subject_normalised,
        )
        for match in subject_matches:
            _register_candidate(match, supplier_hint=supplier_hint, matched_via="subject", score=0.8)

        body_matches = self._extract_rfq_ids(analysis_body)
        for match in body_matches:
            _register_candidate(match, supplier_hint=supplier_hint, matched_via="body", score=0.7)

        metadata_candidates = self._rfq_candidates_from_metadata(message)
        for candidate, supplier_from_meta in metadata_candidates:
            _register_candidate(
                candidate,
                supplier_hint=supplier_from_meta or supplier_hint,
                matched_via="body",
                score=0.7,
            )

        search_text = " ".join(
            value
            for value in (subject_normalised, body_normalised)
            if isinstance(value, str)
        )
        for canonical, resolved_value, supplier_from_tail in self._resolve_rfq_from_tail_tokens(
            search_text, message
        ):
            _register_candidate(
                resolved_value,
                supplier_hint=supplier_from_tail or supplier_hint,
                matched_via="body",
                score=0.7,
            )

        if _best_score() < 0.75:
            for match in self._resolve_rfq_via_dispatch_chain(message, thread_identifiers):
                rfq_value = match.get("rfq_id")
                if not rfq_value:
                    continue
                metadata_doc = self._safe_parse_json(match.get("dispatch_metadata"))
                dispatch_record = {
                    "rfq_id": rfq_value,
                    "supplier_id": match.get("supplier_id"),
                    "match_source": "dispatch_chain",
                }
                if match.get("message_id"):
                    dispatch_record["message_id"] = match.get("message_id")
                if match.get("thread_index") is not None:
                    dispatch_record["thread_index"] = match.get("thread_index")
                created_at = match.get("created_at") if isinstance(match.get("created_at"), datetime) else None
                if created_at is not None:
                    dispatch_record["created_at"] = created_at
                if metadata_doc:
                    dispatch_record["dispatch_metadata"] = metadata_doc
                    run_identifier = metadata_doc.get("run_id") or metadata_doc.get("dispatch_token")
                    if run_identifier:
                        dispatch_record["run_id"] = run_identifier
                _register_candidate(
                    rfq_value,
                    supplier_hint=match.get("supplier_id") or supplier_hint,
                    matched_via="dispatch_chain",
                    score=float(match.get("score", 0.6)),
                    dispatch_record=dispatch_record,
                    thread_index=match.get("thread_index"),
                    created_at=created_at,
                )

        if not ordered_canonicals:
            dispatch_resolution = self._resolve_rfq_from_recent_dispatch(message)
            if dispatch_resolution:
                dispatch_rfq, dispatch_supplier, dispatch_record = dispatch_resolution
                if dispatch_record:
                    dispatch_record = dict(dispatch_record)
                    dispatch_record.setdefault("match_source", "fallback")
                _register_candidate(
                    dispatch_rfq,
                    supplier_hint=dispatch_supplier or supplier_hint,
                    matched_via="fallback",
                    score=0.5,
                    dispatch_record=dispatch_record,
                )

        if not ordered_canonicals or not candidates:
            logger.debug("Skipping email without RFQ identifier: %s", subject)
            return None, "missing_rfq_id"

        best_canonical = None
        best_candidate: Optional[SESEmailWatcher._MatchCandidate] = None
        for canonical, candidate in candidates.items():
            if best_candidate is None:
                best_candidate = candidate
                best_canonical = canonical
                continue
            if self._prefer_candidate(candidate, best_candidate):
                best_candidate = candidate
                best_canonical = canonical

        if best_candidate is None or best_canonical is None:
            logger.debug("Unable to determine RFQ candidate for message: %s", subject)
            return None, "missing_rfq_id"

        primary_canonical = best_canonical
        rfq_id = canonical_map[primary_canonical]

        if best_candidate.dispatch_record:
            message["_dispatch_record"] = best_candidate.dispatch_record
        if best_candidate.supplier_id and primary_canonical not in tail_supplier_map:
            tail_supplier_map[primary_canonical] = best_candidate.supplier_id
        additional_canonicals = ordered_canonicals[1:]
        additional_rfqs = [canonical_map[c] for c in additional_canonicals]

        metadata = self._load_metadata(rfq_id)
        existing_dispatch_record = (
            message.get("_dispatch_record")
            if isinstance(message.get("_dispatch_record"), dict)
            else None
        )
        canonical_rfq_id = metadata.get("canonical_rfq_id")
        if canonical_rfq_id:
            rfq_id = str(canonical_rfq_id)
        dispatch_record = metadata.get("dispatch_record")
        if isinstance(dispatch_record, dict) and existing_dispatch_record:
            for key in ("match_source", "matched_tokens", "matched_domains"):
                if key in existing_dispatch_record and key not in dispatch_record:
                    dispatch_record[key] = existing_dispatch_record[key]
        supplier_id = metadata.get("supplier_id") or message.get("supplier_id")
        if dispatch_record:
            message["_dispatch_record"] = dispatch_record
            record_supplier = dispatch_record.get("supplier_id")
            if record_supplier:
                supplier_id = record_supplier
                tail_supplier_map.setdefault(primary_canonical, record_supplier)
        if not supplier_id:
            supplier_id = tail_supplier_map.get(primary_canonical)
        target_price = metadata.get("target_price") or message.get("target_price")
        negotiation_round = metadata.get("round") or message.get("round") or 1

        workflow_id = metadata.get("workflow_id")
        dispatch_payload = metadata.get("dispatch_payload")
        action_payload = metadata.get("action_payload")
        if not workflow_id and dispatch_payload:
            workflow_id = self._extract_workflow_id_from_payload(dispatch_payload)
        if not workflow_id:
            if action_payload is None:
                action_id = metadata.get("action_id")
                if not action_id and dispatch_payload:
                    action_id = self._extract_action_id_from_payload(dispatch_payload)
                    if action_id:
                        metadata["action_id"] = action_id
                if action_id:
                    action_payload = self._load_action_output(action_id)
                    if action_payload:
                        metadata["action_payload"] = action_payload
            if action_payload:
                workflow_id = self._extract_workflow_id_from_payload(action_payload)
        if not workflow_id:
            workflow_id = str(uuid.uuid4())
        metadata.setdefault("workflow_id", workflow_id)

        if target_price is not None:
            try:
                target_price = float(target_price)
            except (TypeError, ValueError):
                logger.warning("Invalid target price '%s' for RFQ %s", target_price, rfq_id)
                target_price = None

        message_identifier = message.get("message_id") or message.get("id")
        processed: Dict[str, object] = {"message_id": message_identifier}
        dispatch_run_id = metadata.get("dispatch_run_id") or message.get("dispatch_run_id")
        if dispatch_run_id:
            processed["dispatch_run_id"] = dispatch_run_id
            self._remember_workflow_run(dispatch_run_id, workflow_id)
        if message.get("s3_key"):
            processed["s3_key"] = message.get("s3_key")
        if message.get("_bucket"):
            processed["_bucket"] = message.get("_bucket")

        processed.update(
            {
                "rfq_id": rfq_id,
                "supplier_id": supplier_id,
                "subject": subject,
                "from_address": from_address,
                "message_body": body,
                "target_price": target_price,
                "workflow_id": workflow_id,
                "related_rfq_ids": additional_rfqs,
                "round": negotiation_round,
            }
        )
        processed["matched_via"] = best_candidate.matched_via
        processed["match_score"] = best_candidate.score

        if not execute_supplier:
            return processed, None

        try:
            get_conn = getattr(self.agent_nick, "get_db_connection", None)
            if callable(get_conn) and rfq_id:
                with get_conn() as conn:
                    headers = message.get("headers") if isinstance(message.get("headers"), dict) else {}
                    in_reply_to_candidate = None
                    references_candidate: Optional[Iterable[str] | str] = None
                    if headers:
                        in_reply_to_candidate = headers.get("In-Reply-To") or headers.get("in-reply-to")
                        references_candidate = headers.get("References") or headers.get("references")
                    in_reply_to_value = message.get("in_reply_to") or in_reply_to_candidate
                    references_value = message.get("references") or references_candidate
                    mark_dispatch_response(
                        conn,
                        rfq_id=rfq_id,
                        in_reply_to=in_reply_to_value,
                        references=references_value,
                        response_message_id=message_identifier,
                        response_metadata={
                            "subject": subject,
                            "from_address": from_address,
                        },
                    )
                    conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.debug(
                "Failed to reconcile dispatch chain for message %s", message_identifier, exc_info=True
            )

        thread_headers = self._build_thread_headers_for_reply(message)
        if thread_headers:
            message.setdefault("thread_headers", thread_headers)

        context = AgentContext(
            workflow_id=workflow_id,
            agent_id="supplier_interaction",
            user_id=self.settings.script_user,
            input_data={
                "subject": subject,
                "message": body,
                "supplier_id": supplier_id,
                "rfq_id": rfq_id,
                "from_address": from_address,
                "target_price": target_price,
                "message_id": message_identifier,
                "s3_key": message.get("s3_key"),
            },
        )

        if thread_headers:
            context.input_data["thread_headers"] = thread_headers

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
        negotiation_job: Optional[Dict[str, object]] = None

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
                thread_headers = message.get("thread_headers")
                if thread_headers:
                    negotiation_context.input_data["thread_headers"] = thread_headers
                negotiation_job = {
                    "context": negotiation_context,
                    "rfq_id": rfq_id,
                    "round": negotiation_round,
                    "supplier_id": supplier_id,
                }

        processed.update(
            {
                "price": interaction_output.data.get("price"),
                "lead_time": interaction_output.data.get("lead_time"),
                "negotiation_triggered": triggered,
                "supplier_status": interaction_output.status.value,
                "negotiation_status": negotiation_output.status.value if negotiation_output else None,
                "supplier_output": interaction_output.data,
                "negotiation_output": negotiation_output.data if negotiation_output else None,
                "workflow_id": context.workflow_id,
                "related_rfq_ids": additional_rfqs,
            }
        )
        processed.setdefault("negotiation_triggered", False)
        processed.setdefault("negotiation_status", None)
        processed.setdefault("negotiation_output", None)

        triggered, negotiation_output = self._register_processed_response(
            context.workflow_id,
            metadata,
            processed,
            negotiation_job,
        )
        if negotiation_output is not None:
            processed["negotiation_status"] = negotiation_output.status.value
            processed["negotiation_output"] = negotiation_output.data
        if triggered:
            processed["negotiation_triggered"] = True

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
                    cur.execute(
                        """
                        SELECT supplier_id, supplier_name, rfq_id, sent_on, sent, created_on, updated_on, payload
                        FROM proc.draft_rfq_emails
                        WHERE rfq_id = %s
                        ORDER BY updated_on DESC, created_on DESC
                        LIMIT 1
                        """,
                        (lookup_rfq_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        (
                            draft_supplier_id,
                            draft_supplier_name,
                            draft_rfq_id,
                            draft_sent_on,
                            draft_sent_flag,
                            draft_created_on,
                            draft_updated_on,
                            draft_payload,
                        ) = row
                        dispatch_record = {
                            "supplier_id": draft_supplier_id,
                            "supplier_name": draft_supplier_name,
                            "rfq_id": draft_rfq_id,
                            "sent_on": draft_sent_on,
                            "sent": bool(draft_sent_flag),
                            "created_on": draft_created_on,
                            "updated_on": draft_updated_on,
                        }
                        dispatch_payload = self._safe_parse_json(draft_payload)
                        if dispatch_payload:
                            dispatch_record["payload"] = dispatch_payload
                            details["dispatch_payload"] = dispatch_payload
                            run_identifier = (
                                dispatch_payload.get("dispatch_run_id")
                                or dispatch_payload.get("run_id")
                            )
                            if not run_identifier:
                                meta_block = dispatch_payload.get("dispatch_metadata")
                                if isinstance(meta_block, dict):
                                    run_identifier = (
                                        meta_block.get("run_id")
                                        or meta_block.get("dispatch_run_id")
                                        or meta_block.get("dispatch_token")
                                    )
                            if run_identifier:
                                details["dispatch_run_id"] = run_identifier
                                dispatch_record["run_id"] = run_identifier
                            action_identifier = self._extract_action_id_from_payload(
                                dispatch_payload
                            )
                            if action_identifier and not details.get("action_id"):
                                details["action_id"] = action_identifier
                            workflow_hint = self._extract_workflow_id_from_payload(
                                dispatch_payload
                            )
                            if workflow_hint and not details.get("workflow_id"):
                                details["workflow_id"] = workflow_hint
                            expected_hint = self._coerce_int_value(
                                dispatch_payload.get("expected_supplier_count")
                            )
                            if expected_hint is not None:
                                details.setdefault("expected_supplier_count", expected_hint)
                        details["dispatch_record"] = dispatch_record
                        if draft_supplier_name and not details.get("supplier_name"):
                            details["supplier_name"] = draft_supplier_name
                        if draft_supplier_id:
                            if details.get("supplier_id") not in (draft_supplier_id, None):
                                logger.debug(
                                    "Overriding supplier %s with dispatch supplier %s for RFQ %s",
                                    details.get("supplier_id"),
                                    draft_supplier_id,
                                    lookup_rfq_id,
                                )
                            details["supplier_id"] = draft_supplier_id
                        elif "supplier_id" not in details:
                            details["supplier_id"] = None

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

    @staticmethod
    def _safe_parse_json(value: object) -> Optional[Dict[str, object]]:
        if value in (None, ""):
            return None
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                logger.debug("Failed to parse JSON payload", exc_info=True)
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _coerce_int_value(value: object) -> Optional[int]:
        if isinstance(value, bool):
            return int(value) if value else None
        if isinstance(value, (int, float)):
            candidate = int(value)
            return candidate if candidate >= 0 else None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = float(text)
            except Exception:
                return None
            candidate = int(parsed)
            return candidate if candidate >= 0 else None
        return None

    @staticmethod
    def _extract_action_id_from_payload(
        payload: Optional[Dict[str, object]]
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in ("action_id", "draft_action_id", "email_action_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("action_id", "draft_action_id", "email_action_id"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _extract_workflow_id_from_payload(
        payload: Optional[Dict[str, object]]
    ) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        def _collect_candidates(container: Dict[str, object]) -> List[Optional[str]]:
            candidates: List[Optional[str]] = []
            keys = ("workflow_id", "workflowId", "workflow")
            for key in keys:
                candidates.append(container.get(key))
            return candidates

        candidates = _collect_candidates(payload)

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(_collect_candidates(metadata))

        for field in ("input", "input_data", "context", "source_data", "process_details"):
            section = payload.get(field)
            if isinstance(section, dict):
                candidates.extend(_collect_candidates(section))

        drafts = payload.get("drafts")
        if isinstance(drafts, list):
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                candidates.extend(_collect_candidates(draft))
                draft_meta = draft.get("metadata")
                if isinstance(draft_meta, dict):
                    candidates.extend(_collect_candidates(draft_meta))

        dispatch_record = payload.get("dispatch_record")
        if isinstance(dispatch_record, dict):
            candidates.extend(_collect_candidates(dispatch_record))

        for candidate in candidates:
            if isinstance(candidate, str):
                text = candidate.strip()
                if text:
                    return text
        return None

    def _load_action_output(self, action_id: Optional[str]) -> Dict[str, object]:
        if not action_id:
            return {}
        cached = self._action_payload_cache.get(action_id)
        if cached is not None:
            return cached

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return {}

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT process_output FROM proc.action WHERE action_id = %s",
                        (action_id,),
                    )
                    row = cur.fetchone()
        except Exception:
            logger.debug(
                "Failed to load action payload for %s", action_id, exc_info=True
            )
            payload = {}
        else:
            payload = self._safe_parse_json(row[0]) if row and row[0] else {}
            if payload is None:
                payload = {}

        self._action_payload_cache[action_id] = payload
        return payload

    def _derive_expected_supplier_count(
        self,
        metadata: Optional[Dict[str, object]],
        action_payload: Optional[Dict[str, object]] = None,
    ) -> Optional[int]:
        sources: List[Dict[str, object]] = []
        if isinstance(metadata, dict):
            sources.append(metadata)
            dispatch_payload = metadata.get("dispatch_payload")
            if isinstance(dispatch_payload, dict):
                sources.append(dispatch_payload)
                meta_payload = dispatch_payload.get("metadata")
                if isinstance(meta_payload, dict):
                    sources.append(meta_payload)
            dispatch_record = metadata.get("dispatch_record")
            if isinstance(dispatch_record, dict):
                record_payload = dispatch_record.get("payload")
                if isinstance(record_payload, dict):
                    sources.append(record_payload)
        if isinstance(action_payload, dict):
            sources.append(action_payload)
            action_meta = action_payload.get("metadata")
            if isinstance(action_meta, dict):
                sources.append(action_meta)

        for source in sources:
            for key in (
                "expected_supplier_count",
                "expected_suppliers",
                "total_suppliers",
                "supplier_total",
                "supplier_count",
            ):
                candidate = self._coerce_int_value(source.get(key))
                if candidate is not None and candidate > 0:
                    return candidate

        for source in sources:
            drafts = source.get("drafts")
            if isinstance(drafts, list) and drafts:
                suppliers: Set[str] = set()
                for draft in drafts:
                    if not isinstance(draft, dict):
                        continue
                    supplier_id = draft.get("supplier_id")
                    if supplier_id:
                        suppliers.add(str(supplier_id))
                if suppliers:
                    return len(suppliers)
                valid_drafts = [draft for draft in drafts if isinstance(draft, dict)]
                if valid_drafts:
                    return len(valid_drafts)

        return None

    def _ensure_workflow_expectations(
        self,
        workflow_id: Optional[str],
        metadata: Dict[str, object],
        *,
        group_key: Optional[str] = None,
    ) -> Optional[int]:
        workflow_key = self._normalise_workflow_key(workflow_id)
        tracking_key = group_key or workflow_key
        if not tracking_key:
            return None
        action_payload = metadata.get("action_payload")
        if not action_payload:
            action_id = metadata.get("action_id")
            if not action_id:
                dispatch_payload = metadata.get("dispatch_payload")
                action_id = self._extract_action_id_from_payload(dispatch_payload)
                if action_id:
                    metadata["action_id"] = action_id
            if action_id:
                action_payload = self._load_action_output(action_id)
                if action_payload:
                    metadata["action_payload"] = action_payload
                    workflow_hint = self._extract_workflow_id_from_payload(action_payload)
                    if workflow_hint and not metadata.get("workflow_id"):
                        metadata["workflow_id"] = workflow_hint

        existing_expected = self._workflow_expected_counts.get(tracking_key)
        existing_processed = self._workflow_processed_counts.get(tracking_key)
        expected = self._derive_expected_supplier_count(metadata, action_payload)
        if expected is not None:
            if existing_expected != expected:
                self._workflow_expected_counts[tracking_key] = expected
                if existing_processed is None:
                    self._workflow_processed_counts.pop(tracking_key, None)
                else:
                    adjusted = min(existing_processed, expected)
                    if adjusted > 0:
                        self._workflow_processed_counts[tracking_key] = adjusted
                    else:
                        self._workflow_processed_counts.pop(tracking_key, None)
            return expected

        return existing_expected

    def _normalise_workflow_key(self, workflow_id: Optional[str]) -> Optional[str]:
        return self._normalise_filter_value(workflow_id) if workflow_id else None

    def _remember_workflow_run(
        self, run_id: Optional[object], workflow_id: Optional[str]
    ) -> None:
        if run_id in (None, "") or workflow_id in (None, ""):
            return

        run_key = self._normalise_filter_value(run_id)
        workflow_key = self._normalise_filter_value(workflow_id)
        if not run_key or not workflow_key:
            return

        existing = self._workflow_run_index.get(run_key)
        if existing and existing != workflow_key:
            logger.debug(
                "Updating workflow mapping for run_id=%s from %s to %s",
                run_key,
                existing,
                workflow_key,
            )
        self._workflow_run_index[run_key] = workflow_key

    def _normalise_group_key(
        self,
        run_id: Optional[object],
        workflow_id: Optional[str],
    ) -> Optional[str]:
        run_candidate = self._normalise_filter_value(run_id) if run_id else None
        workflow_candidate = self._normalise_filter_value(workflow_id) if workflow_id else None
        if run_candidate and workflow_candidate:
            self._workflow_run_index[run_candidate] = workflow_candidate
            return f"wf::{workflow_candidate}"
        if run_candidate:
            mapped = self._workflow_run_index.get(run_candidate)
            if mapped:
                return f"wf::{mapped}"
            return f"run::{run_candidate}"
        if workflow_candidate:
            return f"wf::{workflow_candidate}"
        return None

    def _flush_negotiation_jobs(
        self, workflow_key: str, current_processed: Dict[str, object]
    ) -> Tuple[bool, Optional[AgentOutput]]:
        jobs = self._workflow_negotiation_jobs.pop(workflow_key, [])
        history_entries = self._workflow_context_buffer.pop(workflow_key, [])
        self._workflow_processed_counts.pop(workflow_key, None)
        self._workflow_expected_counts.pop(workflow_key, None)

        conversation_payload: List[Dict[str, object]] = []
        if history_entries:
            conversation_payload = [
                self._build_conversation_entry(entry)
                for entry in history_entries
                if isinstance(entry, dict)
            ]

        result: Tuple[bool, Optional[AgentOutput]] = (
            current_processed.get("negotiation_triggered", False),
            None,
        )

        if not jobs:
            if conversation_payload:
                for entry in history_entries:
                    if isinstance(entry, dict):
                        entry.setdefault("conversation_history", conversation_payload)
                self._dispatch_quote_comparison(history_entries)
            return result

        def _apply_result(
            processed: Optional[Dict[str, object]],
            triggered: bool,
            output: Optional[AgentOutput],
        ) -> None:
            if not isinstance(processed, dict):
                return
            processed["negotiation_triggered"] = triggered
            processed["negotiation_status"] = output.status.value if output else None
            processed["negotiation_output"] = output.data if output else None
            if output and isinstance(output.data, dict):
                processed["negotiation_round"] = output.data.get("round")
            if conversation_payload:
                processed.setdefault("conversation_history", conversation_payload)
            self._update_processed_snapshot(processed)
            nonlocal result
            if processed is current_processed:
                result = (triggered, output)

        worker_count = min(self._negotiation_parallel_workers, max(len(jobs), 1))
        futures: List[Tuple[Optional[Dict[str, object]], Optional[Future]]]
        futures = []

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for entry in jobs:
                if not isinstance(entry, dict):
                    futures.append((None, None))
                    continue
                job = entry.get("job")
                processed = entry.get("processed")
                if isinstance(processed, dict) and conversation_payload:
                    processed.setdefault("conversation_history", conversation_payload)
                context = job.get("context") if isinstance(job, dict) else None
                if isinstance(context, AgentContext) and conversation_payload:
                    context.input_data = dict(context.input_data)
                    context.input_data.setdefault("conversation_history", conversation_payload)
                if not job:
                    futures.append((processed if isinstance(processed, dict) else None, None))
                    continue
                future = executor.submit(self._execute_negotiation_job, job)
                futures.append((processed if isinstance(processed, dict) else None, future))

            for processed, future in futures:
                if future is None:
                    _apply_result(processed, False, None)
                    continue
                try:
                    triggered, output = future.result()
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("Negotiation job execution failed")
                    triggered, output = False, None
                _apply_result(processed, triggered, output)

        if conversation_payload:
            for entry in history_entries:
                if isinstance(entry, dict):
                    entry.setdefault("conversation_history", conversation_payload)
            self._dispatch_quote_comparison(history_entries)

        return result

    def _build_conversation_entry(self, processed: Dict[str, object]) -> Dict[str, object]:
        entry: Dict[str, object] = {
            "message_id": processed.get("message_id"),
            "rfq_id": processed.get("rfq_id"),
            "supplier_id": processed.get("supplier_id"),
            "subject": processed.get("subject"),
            "from_address": processed.get("from_address"),
            "message_body": processed.get("message_body"),
            "round": processed.get("round"),
            "negotiation_round": processed.get("negotiation_round"),
            "workflow_id": processed.get("workflow_id"),
            "matched_via": processed.get("matched_via"),
        }
        supplier_output = processed.get("supplier_output")
        if isinstance(supplier_output, dict):
            entry["supplier_output"] = deepcopy(supplier_output)
            if "price" in supplier_output and entry.get("price") is None:
                entry["price"] = supplier_output.get("price")
            if "lead_time" in supplier_output and entry.get("lead_time") is None:
                entry["lead_time"] = supplier_output.get("lead_time")
        negotiation_output = processed.get("negotiation_output")
        if isinstance(negotiation_output, dict):
            entry["negotiation_output"] = deepcopy(negotiation_output)
            entry.setdefault("negotiation_message", negotiation_output.get("message"))
            entry.setdefault("counter_proposals", negotiation_output.get("counter_proposals"))
            if negotiation_output.get("round") and entry.get("negotiation_round") is None:
                entry["negotiation_round"] = negotiation_output.get("round")
        return entry

    def _ensure_quote_comparison_agent(self) -> Optional["QuoteComparisonAgent"]:
        agent = getattr(self, "quote_comparison_agent", None)
        if agent is not None:
            return agent
        if QuoteComparisonAgent is None:
            return None
        try:
            agent = QuoteComparisonAgent(self.agent_nick)
        except Exception:  # pragma: no cover - defensive initialisation
            logger.exception("Failed to initialise QuoteComparisonAgent")
            agent = None
        self.quote_comparison_agent = agent
        return agent

    def _build_quote_payload(self, processed: Dict[str, object]) -> Optional[Dict[str, object]]:
        supplier_id = processed.get("supplier_id")
        supplier_name = (
            processed.get("supplier_name")
            or processed.get("supplier_display_name")
            or supplier_id
        )
        supplier_output = processed.get("supplier_output")
        negotiation_output = processed.get("negotiation_output")

        price: Optional[float] = None
        if isinstance(negotiation_output, dict):
            proposals = negotiation_output.get("counter_proposals")
            if isinstance(proposals, list):
                for proposal in proposals:
                    if isinstance(proposal, dict) and proposal.get("price") is not None:
                        try:
                            price = float(proposal.get("price"))
                            break
                        except Exception:
                            continue
            if price is None:
                counter_price = negotiation_output.get("counter_price")
                try:
                    price = float(counter_price) if counter_price is not None else None
                except Exception:
                    price = None
        if price is None and isinstance(supplier_output, dict):
            try:
                raw_price = supplier_output.get("price")
                price = float(raw_price) if raw_price is not None else None
            except Exception:
                price = None

        lead_time = None
        if isinstance(negotiation_output, dict):
            lead_time = negotiation_output.get("lead_time_request")
        if lead_time is None and isinstance(supplier_output, dict):
            lead_time = supplier_output.get("lead_time")

        currency = None
        if isinstance(negotiation_output, dict):
            currency = negotiation_output.get("currency")
        if currency is None and isinstance(supplier_output, dict):
            currency = supplier_output.get("currency") or supplier_output.get("currency_code")

        if supplier_id is None and supplier_name is None and price is None:
            return None

        payload: Dict[str, object] = {
            "name": supplier_name or "Unknown supplier",
            "supplier_id": supplier_id,
            "total_cost": price,
            "tenure": lead_time,
            "currency": currency,
            "quote_file_s3_path": processed.get("s3_key"),
        }

        negotiation_round = processed.get("negotiation_round") or processed.get("round")
        if negotiation_round is not None:
            payload["round"] = negotiation_round
            try:
                payload["closing_round"] = int(negotiation_round) >= 3
            except Exception:
                payload["closing_round"] = False

        return payload

    def _dispatch_quote_comparison(
        self, history_entries: Sequence[Dict[str, object]]
    ) -> None:
        if not history_entries:
            return

        quote_agent = self._ensure_quote_comparison_agent()
        if quote_agent is None:
            return

        quotes: List[Dict[str, object]] = []
        supplier_ids: Set[str] = set()

        for entry in history_entries:
            if not isinstance(entry, dict):
                continue
            payload = self._build_quote_payload(entry)
            if not payload:
                continue
            quotes.append(payload)
            supplier_token = payload.get("supplier_id")
            if supplier_token:
                supplier_ids.add(str(supplier_token))

        if not quotes:
            return

        workflow_id = None
        for entry in history_entries:
            candidate = entry.get("workflow_id") if isinstance(entry, dict) else None
            if candidate:
                workflow_id = str(candidate)
                break
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        context = AgentContext(
            workflow_id=workflow_id,
            agent_id="quote_comparison",
            user_id=self.settings.script_user,
            input_data={
                "quotes": quotes,
                "supplier_ids": list(supplier_ids),
            },
        )

        try:
            quote_agent.execute(context)
        except Exception:  # pragma: no cover - analysis should not break watcher
            logger.exception(
                "Quote comparison analysis failed for workflow %s", workflow_id
            )
    def _execute_negotiation_job(
        self, job: Dict[str, object]
    ) -> Tuple[bool, Optional[AgentOutput]]:
        negotiation_agent = self.negotiation_agent
        context = job.get("context")
        if negotiation_agent is None or not isinstance(context, AgentContext):
            return False, None

        negotiation_output = negotiation_agent.execute(context)
        triggered = negotiation_output.status == AgentStatus.SUCCESS
        logger.info(
            "Negotiation triggered for RFQ %s (round %s) via mailbox %s; status=%s",
            job.get("rfq_id"),
            job.get("round"),
            self.mailbox_address,
            negotiation_output.status.value,
        )
        return triggered, negotiation_output

    def _register_processed_response(
        self,
        workflow_id: Optional[str],
        metadata: Dict[str, object],
        processed: Dict[str, object],
        negotiation_job: Optional[Dict[str, object]],
    ) -> Tuple[bool, Optional[AgentOutput]]:
        run_identifier = metadata.get("dispatch_run_id") or processed.get("dispatch_run_id")
        workflow_key = self._normalise_workflow_key(workflow_id)
        self._remember_workflow_run(run_identifier, workflow_id)
        group_key = self._normalise_group_key(run_identifier, workflow_id)
        tracking_key = group_key or workflow_key
        default_result: Tuple[bool, Optional[AgentOutput]] = (
            processed.get("negotiation_triggered", False),
            None,
        )

        if tracking_key:
            history = self._workflow_context_buffer.setdefault(tracking_key, [])
            history.append(processed)
            expected = self._ensure_workflow_expectations(
                workflow_id,
                metadata,
                group_key=group_key,
            )
            if negotiation_job:
                queue = self._workflow_negotiation_jobs.setdefault(tracking_key, [])
                queue.append({"job": negotiation_job, "processed": processed})
            count = self._workflow_processed_counts.get(tracking_key, 0) + 1
            self._workflow_processed_counts[tracking_key] = count
            if expected is None or count >= expected:
                return self._flush_negotiation_jobs(tracking_key, processed)
            return default_result

        if negotiation_job:
            history_payload = [self._build_conversation_entry(processed)]
            context = negotiation_job.get("context")
            if isinstance(context, AgentContext):
                context.input_data = dict(context.input_data)
                context.input_data.setdefault("conversation_history", history_payload)
            processed["conversation_history"] = history_payload
            triggered, output = self._execute_negotiation_job(negotiation_job)
            processed["negotiation_triggered"] = triggered
            processed["negotiation_status"] = (
                output.status.value if output else None
            )
            processed["negotiation_output"] = output.data if output else None
            if output and isinstance(output.data, dict):
                processed["negotiation_round"] = output.data.get("round")
            self._update_processed_snapshot(processed)
            self._dispatch_quote_comparison([processed])
            return triggered, output

        return default_result

    def _update_processed_snapshot(self, processed: Dict[str, object]) -> None:
        message_id = processed.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            return
        if message_id in self._processed_cache:
            self._processed_cache[message_id] = deepcopy(processed)

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

    def _resolve_rfq_from_recent_dispatch(
        self, message: Dict[str, object]
    ) -> Optional[Tuple[str, Optional[str], Optional[Dict[str, object]]]]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None

        supplier_candidates: List[str] = []
        supplier_hint = self._normalise_filter_value(message.get("supplier_id"))
        if supplier_hint:
            supplier_candidates.append(supplier_hint)

        email_candidates: List[str] = []
        for key in ("from", "reply_to", "supplier_email"):
            candidate = self._normalise_email(message.get(key))
            if candidate and candidate not in email_candidates:
                email_candidates.append(candidate)

        headers = message.get("headers")
        if isinstance(headers, dict):
            for header_key in (
                "From",
                "from",
                "Reply-To",
                "reply-to",
                "Return-Path",
                "return-path",
            ):
                candidate = self._normalise_email(headers.get(header_key))
                if candidate and candidate not in email_candidates:
                    email_candidates.append(candidate)

        where_clauses: List[str] = []
        params: List[object] = []

        if supplier_candidates:
            where_clauses.append("supplier_id = ANY(%s)")
            params.append(supplier_candidates)

        if email_candidates:
            where_clauses.append("LOWER(COALESCE(recipient_email, '')) = ANY(%s)")
            params.append(email_candidates)

        if where_clauses:
            query = """
                SELECT rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on
                FROM proc.draft_rfq_emails
                WHERE sent = TRUE AND ({})
                ORDER BY updated_on DESC, created_on DESC
                LIMIT 1
            """.format(" OR ".join(where_clauses))

            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, tuple(params))
                        row = cur.fetchone()
            except Exception:
                logger.debug("Failed to resolve RFQ from dispatch history", exc_info=True)
                row = None

            if row:
                (
                    rfq_value,
                    supplier_value,
                    supplier_name,
                    sent_on,
                    sent_flag,
                    created_on,
                    updated_on,
                ) = row

                if rfq_value:
                    dispatch_record = {
                        "rfq_id": rfq_value,
                        "supplier_id": supplier_value,
                        "supplier_name": supplier_name,
                        "sent_on": sent_on,
                        "sent": bool(sent_flag),
                        "created_on": created_on,
                        "updated_on": updated_on,
                        "match_source": "dispatch",
                    }
                    return str(rfq_value), supplier_value, dispatch_record

        return self._resolve_recent_dispatch_by_similarity(message, get_conn)

    def _resolve_recent_dispatch_by_similarity(
        self,
        message: Dict[str, object],
        get_conn: Callable[[], object],
    ) -> Optional[Tuple[str, Optional[str], Optional[Dict[str, object]]]]:
        subject = message.get("subject")
        body = message.get("body")

        body_segment: Optional[str]
        if isinstance(body, str):
            body_segment = body[:2000]
        else:
            body_segment = None

        inbound_tokens = _extract_similarity_tokens(subject, body_segment)
        if not inbound_tokens:
            return None

        inbound_domains: List[str] = []
        for key in ("from", "reply_to", "supplier_email"):
            domain = _extract_email_domain(message.get(key))
            if domain and domain not in inbound_domains:
                inbound_domains.append(domain)

        headers = message.get("headers")
        if isinstance(headers, dict):
            for header_key in ("From", "from", "Reply-To", "reply-to"):
                domain = _extract_email_domain(headers.get(header_key))
                if domain and domain not in inbound_domains:
                    inbound_domains.append(domain)

        query = """
            SELECT rfq_id, supplier_id, supplier_name, sent_on, sent, created_on, updated_on, payload
            FROM proc.draft_rfq_emails
            WHERE sent = TRUE
              AND updated_on >= NOW() - INTERVAL '7 days'
            ORDER BY updated_on DESC, created_on DESC
            LIMIT %s
        """

        rows: List[Tuple] = []
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (25,))
                    fetched = cur.fetchall()
                    if fetched:
                        rows = list(fetched)
        except Exception:
            logger.debug("Failed to load recent RFQ dispatches for similarity", exc_info=True)
            return None

        if not rows:
            return None

        best_row: Optional[Tuple] = None
        best_score = -1.0
        best_overlap: Set[str] = set()
        best_domains: List[str] = []

        min_overlap = 1 if len(inbound_tokens) <= 4 else 2

        for row in rows:
            if not row or not row[0]:
                continue

            candidate_subject = row[7] if len(row) > 7 else None
            candidate_body = row[8] if len(row) > 8 else None
            if isinstance(candidate_body, str):
                candidate_body = candidate_body[:2000]

            candidate_tokens = _extract_similarity_tokens(
                candidate_subject, candidate_body
            )
            if not candidate_tokens:
                continue

            overlap = inbound_tokens & candidate_tokens

            recipient_domain = _extract_email_domain(row[9] if len(row) > 9 else None)
            sender_domain = _extract_email_domain(row[10] if len(row) > 10 else None)
            candidate_domains = [d for d in (recipient_domain, sender_domain) if d]
            domain_match = False
            if inbound_domains and candidate_domains:
                domain_match = any(domain in inbound_domains for domain in candidate_domains)

            if len(overlap) < min_overlap and not domain_match:
                continue

            token_score = len(overlap) / max(len(inbound_tokens), 1)
            domain_bonus = 0.3 if domain_match else 0.0
            score = token_score + domain_bonus

            if score < 0.2:
                continue

            updated_on = row[6] if len(row) > 6 else None

            if score > best_score:
                best_row = row
                best_score = score
                best_overlap = set(overlap)
                best_domains = list(candidate_domains)
            elif score == best_score and best_row is not None:
                current_updated = best_row[6] if len(best_row) > 6 else None
                if isinstance(updated_on, datetime) and isinstance(current_updated, datetime):
                    if updated_on > current_updated:
                        best_row = row
                        best_overlap = set(overlap)
                        best_domains = list(candidate_domains)
                elif updated_on and not current_updated:
                    best_row = row
                    best_overlap = set(overlap)
                    best_domains = list(candidate_domains)

        if not best_row:
            return None

        (
            rfq_value,
            supplier_value,
            supplier_name,
            sent_on,
            sent_flag,
            created_on,
            updated_on,
            payload_json,
        ) = row

        if not rfq_value:
            return None

        dispatch_record = {
            "rfq_id": rfq_value,
            "supplier_id": supplier_value,
            "supplier_name": supplier_name,
            "sent_on": sent_on,
            "sent": bool(sent_flag),
            "created_on": created_on,
            "updated_on": updated_on,
            "match_source": "dispatch_similarity",
            "matched_tokens": sorted(best_overlap),
        }
        payload_doc = self._safe_parse_json(payload_json)
        if payload_doc:
            dispatch_record["payload"] = payload_doc

        return str(rfq_value), supplier_value, dispatch_record

    def _load_messages(
        self,
        limit: Optional[int],
        *,
        mark_seen: bool,
        prefixes: Optional[Sequence[str]] = None,
        since: Optional[datetime] = None,
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
        self._last_candidate_source = "none"

        imap_configured = self._imap_configured()

        if not imap_configured:
            if not self._imap_warning_logged:
                logger.error(
                    "IMAP mailbox credentials (imap_host, imap_user, imap_password) are not configured for %s; inbound polling is disabled",
                    self.mailbox_address or "<unknown>",
                )
                self._imap_warning_logged = True
            # Treat a missing configuration the same as an empty IMAP batch so we
            # can fall back to S3 once the retry threshold is hit.
            self._consecutive_empty_imap_batches = max(
                self._consecutive_empty_imap_batches,
                self._imap_fallback_attempts,
            )
        else:
            if self._imap_warning_logged:
                logger.info(
                    "IMAP mailbox credentials configured for %s; resuming inbound polling",
                    self.mailbox_address or "<unknown>",
                )
                self._imap_warning_logged = False

            messages = self._load_from_imap(
                effective_limit,
                mark_seen=bool(mark_seen),
                since=since,
                on_message=on_message,
            )

            self._last_candidate_source = "imap"

            if messages:
                self._consecutive_empty_imap_batches = 0
                return messages

            self._consecutive_empty_imap_batches += 1

        fallback_due = (
            self._imap_fallback_attempts > 0
            and self._consecutive_empty_imap_batches >= self._imap_fallback_attempts
        )

        if fallback_due and self.bucket:
            logger.info(
                "No IMAP messages after %d poll(s); falling back to S3 for mailbox %s",
                self._consecutive_empty_imap_batches,
                self.mailbox_address,
            )
            messages = self._load_from_s3(
                effective_limit,
                prefixes=prefixes,
                on_message=on_message,
            )
            self._last_candidate_source = "s3"
            self._consecutive_empty_imap_batches = 0
            return messages

        if fallback_due and not self.bucket:
            logger.debug(
                "IMAP fallback to S3 skipped for mailbox %s because no bucket is configured",
                self.mailbox_address,
            )
            self._consecutive_empty_imap_batches = 0

        if not imap_configured:
            self._last_candidate_source = "imap"

        return messages

    def _imap_configured(self) -> bool:
        host, user, password, _, _, _ = self._imap_settings()
        return bool(host and user and password)

    def _imap_settings(
        self,
    ) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        str,
        str,
        Optional[int],
    ]:
        host = getattr(self.settings, "imap_host", None) or os.getenv("IMAP_HOST")

        login_override = getattr(self.settings, "imap_login", None) or os.getenv("IMAP_LOGIN")
        raw_user = getattr(self.settings, "imap_user", None) or os.getenv("IMAP_USER")
        username_hint = getattr(self.settings, "imap_username", None) or os.getenv("IMAP_USERNAME")
        domain_hint = getattr(self.settings, "imap_domain", None) or os.getenv("IMAP_DOMAIN")

        user = login_override or raw_user or username_hint
        domain_str = str(domain_hint) if domain_hint else None
        if not user and domain_str:
            user = domain_str
        elif domain_str:
            user_str = str(user) if user is not None else ""
            if "@" in domain_str:
                if user_str and "@" not in user_str:
                    suffix = domain_str.split("@", 1)[-1]
                    user = f"{user_str}@{suffix}" if suffix else user_str
                elif not user_str:
                    user = domain_str
            elif user_str and "@" not in user_str:
                user = f"{user_str}@{domain_str}" if domain_str else user_str

        password = (
            getattr(self.settings, "imap_password", None)
            or os.getenv("IMAP_PASSWORD")
        )
        mailbox = getattr(self.settings, "imap_mailbox", "INBOX") or "INBOX"
        search_criteria = getattr(self.settings, "imap_search_criteria", "ALL") or "ALL"
        port_value = getattr(self.settings, "imap_port", None)
        if port_value in (None, ""):
            port_value = os.getenv("IMAP_PORT")
        if port_value in (None, "") and host:
            port_value = "993"
        port = self._coerce_int_value(port_value) if port_value not in (None, "") else None
        return host, user, password, mailbox, search_criteria, port

    @staticmethod
    def _compose_imap_search_criteria(
        base: Optional[str], since: Optional[datetime]
    ) -> str:
        criteria = (base or "").strip()
        if not criteria or criteria.upper() == "ALL":
            criteria = "UNSEEN"

        since_clause = ""
        if since is not None:
            reference = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            since_clause = reference.strftime("SINCE %d-%b-%Y")

        if since_clause:
            if criteria:
                return f"({criteria}) {since_clause}"
            return since_clause

        return criteria or "ALL"

    def _load_from_imap(
        self,
        limit: Optional[int],
        *,
        mark_seen: bool,
        since: Optional[datetime] = None,
        on_message: Optional[Callable[[Dict[str, object], Optional[datetime]], bool]] = None,
    ) -> List[Dict[str, object]]:
        host, user, password, mailbox, search_criteria, port = self._imap_settings()
        if not host or not user or not password:
            return []

        collected: List[Tuple[Optional[datetime], Dict[str, object]]] = []
        seen_ids: Set[str] = set()

        try:
            if port:
                client_manager = imaplib.IMAP4_SSL(host, port)
            else:
                client_manager = imaplib.IMAP4_SSL(host)

            with client_manager as client:
                client.login(user, password)
                status, _ = client.select(mailbox)
                if status != "OK":
                    logger.warning(
                        "IMAP fallback could not select mailbox %s for %s", mailbox, self.mailbox_address
                    )
                    client.logout()
                    return []

                final_criteria = self._compose_imap_search_criteria(search_criteria, since)
                start_uid: Optional[int] = None
                if self._last_uid is not None:
                    try:
                        start_uid = max(int(self._last_uid) + 1, 1)
                    except Exception:
                        start_uid = None

                search_terms: List[str] = []
                if final_criteria:
                    search_terms.append(final_criteria)
                if not search_terms:
                    search_terms.append("UNSEEN")
                if start_uid is not None:
                    search_terms.extend(["UID", f"{start_uid}:*"])

                status, data = client.uid("search", None, *search_terms)
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

                processed_uids: Set[int] = set()

                for raw_id in reversed(raw_ids):
                    message_id_str = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    if message_id_str in seen_ids:
                        continue

                    status, payload = client.uid("fetch", message_id_str, fetch_command)
                    if status != "OK" or not payload:
                        continue

                    raw_bytes: Optional[bytes] = None
                    for part in payload:
                        if isinstance(part, tuple) and len(part) >= 2:
                            raw_bytes = part[1]
                            break
                    if raw_bytes is None:
                        continue

                    parsed = self._parse_inbound_object(raw_bytes, key=f"imap/{message_id_str}")
                    uid_str = message_id_str
                    msg_id_header = str(parsed.get("message_id") or "")
                    date_header = str(parsed.get("received_at") or "")
                    sender = str(parsed.get("from") or "")
                    subject = str(parsed.get("subject") or "")
                    unique_id = _generate_unique_id(
                        uid_str,
                        msg_id_header,
                        date_header,
                        sender,
                        subject,
                    )
                    parsed["id"] = unique_id
                    parsed.setdefault("s3_key", f"imap/{uid_str}")
                    parsed.setdefault(
                        "_bucket", f"imap::{self.mailbox_address or 'unknown'}"
                    )
                    parsed.setdefault("mailbox", mailbox)
                    if msg_id_header:
                        parsed["message_id"] = msg_id_header
                    else:
                        parsed["message_id"] = unique_id
                    parsed.setdefault("_imap_uid", uid_str)
                    parsed.setdefault("_source", "imap")
                    timestamp = self._coerce_imap_timestamp(parsed.get("received_at"))
                    if timestamp is None:
                        timestamp = datetime.now(timezone.utc)
                    collected.append((timestamp, parsed))
                    seen_ids.add(message_id_str)
                    try:
                        processed_uids.add(int(uid_str))
                    except Exception:
                        pass

                    if mark_seen:
                        try:
                            client.uid("store", message_id_str, "+FLAGS", "(\\Seen)")
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

                if processed_uids:
                    latest_uid = max(processed_uids)
                    previous_uid = self._last_uid or 0
                    if latest_uid > previous_uid:
                        self._last_uid = latest_uid
                        logger.debug(
                            "Updated last seen IMAP UID to %s for mailbox %s",
                            self._last_uid,
                            self.mailbox_address,
                        )
        except Exception:
            logger.error(
                "IMAP fallback polling failed for mailbox %s",
                self.mailbox_address or "<unknown>",
                exc_info=True,
            )
            return []

        if not collected:
            return []

        collected.sort(
            key=lambda item: item[0] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        messages = [payload for _, payload in collected]
        if messages:
            ids = [
                str(message.get("message_id") or message.get("id") or "<no-id>")
                for message in messages
            ]
            times = [
                str(message.get("received_at") or message.get("_last_modified") or "?")
                for message in messages
            ]
            logger.debug(
                "IMAP poll retrieved %d message(s) for mailbox %s: ids=%s times=%s",
                len(messages),
                self.mailbox_address,
                ids,
                times,
            )
        return messages

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

        payload_supplier = self._normalise_filter_value(payload.get("supplier_id"))
        payload_run_id = self._normalise_filter_value(
            payload.get("dispatch_run_id")
            or payload.get("run_id")
            or payload.get("dispatch_token")
        )

        supplier_filter = self._normalise_filter_value(filters.get("supplier_id"))
        run_filter = self._normalise_filter_value(
            filters.get("dispatch_run_id") or filters.get("run_id")
        )
        supplier_like_filter = self._normalise_filter_value(
            filters.get("supplier_id_like")
        )

        if supplier_filter or run_filter:
            if supplier_filter:
                if not payload_supplier or payload_supplier != supplier_filter:
                    return False
            if run_filter:
                if not payload_run_id or payload_run_id != run_filter:
                    return False
            return True

        if supplier_like_filter and payload_supplier:
            if supplier_like_filter not in payload_supplier:
                return False

        payload_subject = self._normalise_filter_value(payload.get("subject")) or ""
        payload_sender = self._normalise_filter_value(payload.get("from_address"))
        payload_sender_email = self._normalise_email(payload.get("from_address"))
        payload_message = self._normalise_filter_value(payload.get("message_id")) or self._normalise_filter_value(payload.get("id"))
        payload_workflow = self._normalise_filter_value(payload.get("workflow_id"))

        def _like(actual: Optional[str], expected_like: object) -> bool:
            needle = self._normalise_filter_value(expected_like)
            if not needle:
                return True
            if actual is None:
                return False

            pattern = re.escape(needle)
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

            if needle and "%" not in needle and "_" not in needle and "*" not in needle:
                return needle in actual

            return False

        for key, expected in filters.items():
            if expected in (None, ""):
                continue
            if key == "from_address":
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
                if payload_workflow != expected_workflow:
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

        return bool(filters)

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

        for field in ("from_address", "workflow_id"):
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

    @dataclass
    class _DispatchExpectation:
        action_id: Optional[str]
        workflow_id: Optional[str]
        draft_ids: Tuple[int, ...]
        draft_count: int
        supplier_count: int
        run_identifiers: Tuple[str, ...] = ()

    @dataclass
    class _MatchCandidate:
        rfq_id: str
        supplier_id: Optional[str]
        matched_via: str
        score: float
        priority: int
        order: int
        dispatch_record: Optional[Dict[str, object]] = None
        thread_index: Optional[int] = None
        created_at: Optional[datetime] = None

    @dataclass
    class _DraftSnapshot:
        id: int
        rfq_id: Optional[str]
        subject: str
        body: str
        dispatch_token: Optional[str]
        run_id: Optional[str] = None
        recipients: Tuple[str, ...] = ()
        supplier_id: Optional[str] = None
        workflow_id: Optional[str] = None
        subject_norm: str = field(init=False)
        body_norm: str = field(init=False)
        supplier_norm: Optional[str] = field(init=False, default=None)
        matched_via: str = field(default="unknown")

        def __post_init__(self) -> None:
            comment, remainder = split_hidden_marker(self.body or "")
            cleaned_body = remainder or self.body or ""
            token = extract_marker_token(comment)
            run_identifier = extract_run_id(comment)
            if self.dispatch_token is None and token:
                object.__setattr__(self, "dispatch_token", token)
            if self.run_id is None:
                candidate_run = run_identifier or token or self.dispatch_token
                if candidate_run:
                    object.__setattr__(self, "run_id", candidate_run)
            object.__setattr__(self, "subject_norm", _norm(self.subject or ""))
            object.__setattr__(self, "body_norm", _norm(cleaned_body))
            supplier_norm: Optional[str] = None
            if self.supplier_id:
                try:
                    supplier_norm = _norm(str(self.supplier_id)).lower() or None
                except Exception:
                    supplier_norm = None
            object.__setattr__(self, "supplier_norm", supplier_norm)

        def normalised_recipients(self) -> Set[str]:
            recipients = set()
            for item in self.recipients:
                if not isinstance(item, str):
                    continue
                cleaned = item.strip().lower()
                if cleaned:
                    recipients.add(cleaned)
            return recipients

    @staticmethod
    def _coerce_identifier(value: object) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    @staticmethod
    def _coerce_process_output(value: object) -> Optional[Dict[str, object]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, (bytes, bytearray)):
            try:
                value = value.decode("utf-8")
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                logger.debug("Unable to decode process_output JSON", exc_info=True)
                return None
        return None

    @staticmethod
    def _coerce_bool(value: object) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value in (0, 1):
                return bool(value)
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "sent", "success", "completed", "done", "ok"}:
                return True
            if lowered in {
                "false",
                "0",
                "no",
                "n",
                "failed",
                "error",
                "cancelled",
                "canceled",
                "pending",
                "queued",
                "unsent",
                "draft",
            }:
                return False
        return None

    def _should_skip_dispatch_entry(self, entry: Dict[str, object]) -> bool:
        failure_statuses = {
            "failed",
            "error",
            "errored",
            "exception",
            "timeout",
            "timed_out",
            "cancelled",
            "canceled",
            "aborted",
            "rejected",
        }
        unsent_statuses = {
            "pending",
            "queued",
            "draft",
            "created",
            "initialising",
            "initializing",
            "starting",
            "scheduled",
            "preparing",
        }

        def _iter_sources(candidate: Dict[str, object]) -> Iterable[Dict[str, object]]:
            yield candidate
            metadata = candidate.get("metadata")
            if isinstance(metadata, dict):
                yield metadata

        for source in _iter_sources(entry):
            for key in ("error", "exception", "failure", "failure_reason", "error_message"):
                if source.get(key) not in (None, "", False, 0):
                    return True

            status_value = source.get("status") or source.get("state")
            if isinstance(status_value, str):
                lowered_status = status_value.strip().lower()
                if lowered_status in failure_statuses:
                    return True
                if lowered_status in unsent_statuses:
                    return True

            for flag_key in ("sent_status", "sent", "success", "completed"):
                flag_value = self._coerce_bool(source.get(flag_key))
                if flag_value is False:
                    return True

        return False

    def _dispatch_entry_run_identifier(self, entry: Dict[str, object]) -> Optional[str]:
        """Return the first run identifier token embedded in a dispatch entry."""

        if not isinstance(entry, dict):
            return None

        visited: Set[int] = set()
        stack: List[Dict[str, object]] = [entry]
        while stack:
            candidate = stack.pop()
            obj_id = id(candidate)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            for key in ("dispatch_run_id", "run_id", "dispatch_token"):
                value = self._coerce_identifier(candidate.get(key))
                if value:
                    return value

            metadata = candidate.get("metadata")
            if isinstance(metadata, dict):
                stack.append(metadata)

            for nested_key in ("draft", "dispatch"):
                nested = candidate.get(nested_key)
                if isinstance(nested, dict):
                    stack.append(nested)

        return None

    def _dispatch_entry_unique_key(self, entry: Dict[str, object]) -> Optional[Tuple[str, ...]]:
        """Return identifying tokens describing a dispatch entry for deduplication."""

        if not isinstance(entry, dict):
            return None

        recipients: Set[str] = set()
        supplier_value: Optional[str] = None
        rfq_value: Optional[str] = None
        identifier_value: Optional[str] = None
        message_identifier: Optional[str] = None

        def _collect(obj: object) -> None:
            nonlocal supplier_value, rfq_value, identifier_value, message_identifier
            if not isinstance(obj, dict):
                return

            candidate_supplier = self._coerce_identifier(obj.get("supplier_id"))
            if candidate_supplier and not supplier_value:
                supplier_value = candidate_supplier

            candidate_rfq = self._coerce_identifier(obj.get("rfq_id"))
            if candidate_rfq and not rfq_value:
                rfq_value = candidate_rfq

            candidate_identifier = obj.get("draft_record_id") or obj.get("id")
            candidate_identifier = self._coerce_identifier(candidate_identifier)
            if candidate_identifier and not identifier_value:
                identifier_value = candidate_identifier

            candidate_message = (
                obj.get("dispatch_message_id")
                or obj.get("message_id")
                or obj.get("message-id")
            )
            candidate_message = self._coerce_identifier(candidate_message)
            if candidate_message and not message_identifier:
                message_identifier = candidate_message

            recips = obj.get("recipients")
            if isinstance(recips, (list, tuple, set)):
                for item in recips:
                    token = self._coerce_identifier(item)
                    if token:
                        token_norm = token.strip().lower()
                        if token_norm:
                            recipients.add(token_norm)
            elif isinstance(recips, str):
                token_norm = recips.strip().lower()
                if token_norm:
                    recipients.add(token_norm)

            metadata = obj.get("metadata")
            if isinstance(metadata, dict):
                _collect(metadata)

            for key in ("draft", "dispatch"):
                nested = obj.get(key)
                if isinstance(nested, dict):
                    _collect(nested)

            for group_key in ("drafts", "dispatches"):
                group = obj.get(group_key)
                if isinstance(group, dict):
                    _collect(group)
                elif isinstance(group, (list, tuple, set)):
                    for item in group:
                        _collect(item)

        _collect(entry)

        run_identifier = self._dispatch_entry_run_identifier(entry)
        run_key = (
            self._normalise_filter_value(run_identifier)
            or (_norm(str(run_identifier)).lower() if run_identifier else None)
        )
        supplier_key = (
            self._normalise_filter_value(supplier_value)
            or (_norm(supplier_value).lower() if supplier_value else None)
        )
        rfq_key = (
            self._normalise_filter_value(rfq_value)
            or (_norm(rfq_value).lower() if rfq_value else None)
        )
        message_key = (
            self._normalise_filter_value(message_identifier)
            or (_norm(message_identifier).lower() if message_identifier else None)
        )
        identifier_key = identifier_value.strip().lower() if identifier_value else None
        recipients_key = ",".join(sorted(recipients)) if recipients else None

        strong_tokens: List[str] = []
        fallback_tokens: List[str] = []

        def _add(token: Optional[str], *, strong: bool = True) -> None:
            if not token:
                return
            if strong:
                strong_tokens.append(token)
            else:
                fallback_tokens.append(token)

        _add(f"msg:{message_key}" if message_key else None)
        _add(f"id:{identifier_key}" if identifier_key else None)

        if run_key and supplier_key and recipients_key:
            _add(f"run:{run_key}|supplier:{supplier_key}|rcpt:{recipients_key}")
        if run_key and rfq_key and supplier_key:
            _add(f"run:{run_key}|rfq:{rfq_key}|supplier:{supplier_key}")
        if run_key and rfq_key:
            _add(f"run:{run_key}|rfq:{rfq_key}")
        if run_key and supplier_key:
            _add(f"run:{run_key}|supplier:{supplier_key}")
        if rfq_key and supplier_key:
            _add(f"rfq:{rfq_key}|supplier:{supplier_key}")
        if recipients_key and supplier_key:
            _add(f"supplier:{supplier_key}|rcpt:{recipients_key}")

        _add(f"rfq:{rfq_key}" if rfq_key else None, strong=False)
        _add(f"run:{run_key}" if run_key else None, strong=False)
        _add(f"rcpt:{recipients_key}" if recipients_key else None, strong=False)
        _add(f"supplier:{supplier_key}" if supplier_key else None, strong=False)

        tokens: List[str] = strong_tokens or fallback_tokens
        if not tokens:
            return None

        deduped = tuple(dict.fromkeys(tokens))
        return deduped if deduped else None

    def _extract_dispatch_entries(self, payload: object) -> List[Dict[str, object]]:
        """Return potential dispatch records from an action payload."""

        entries: List[Dict[str, object]] = []
        seen: Set[str] = set()
        visited: Set[int] = set()

        def _run_identifier(entry: Dict[str, object]) -> Optional[str]:
            candidate = self._dispatch_entry_run_identifier(entry)
            if not candidate:
                return None
            try:
                return candidate.strip().lower() or None
            except Exception:
                return None

        def _key(entry: Dict[str, object]) -> Optional[str]:
            run_identifier = _run_identifier(entry)
            identifier = entry.get("draft_record_id") or entry.get("id")

            recipients = entry.get("recipients")
            recipient_key: str = ""
            has_recipients = False
            if isinstance(recipients, (list, tuple, set)):
                cleaned = []
                for item in recipients:
                    if item in (None, ""):
                        continue
                    try:
                        cleaned.append(str(item).strip().lower())
                    except Exception:
                        continue
                if cleaned:
                    has_recipients = True
                    recipient_key = ",".join(sorted(cleaned))
            elif recipients not in (None, ""):
                try:
                    recipient_key = str(recipients).strip().lower()
                except Exception:
                    recipient_key = ""
                else:
                    if recipient_key:
                        has_recipients = True

            supplier = self._coerce_identifier(entry.get("supplier_id")) or ""
            rfq = self._coerce_identifier(entry.get("rfq_id")) or ""

            key_parts: List[str] = []
            if run_identifier:
                key_parts.append(f"run:{run_identifier}")
            if identifier not in (None, ""):
                key_parts.append(f"id:{identifier}")
            if supplier:
                key_parts.append(f"supplier:{supplier.lower()}")
            if rfq:
                key_parts.append(f"rfq:{rfq.lower()}")
            if has_recipients:
                key_parts.append(f"rcpt:{recipient_key}")

            if not key_parts and has_recipients:
                key_parts.append(f"rcpt:{recipient_key}")

            if key_parts:
                return "|".join(key_parts)

            return None

        def _add(entry: Optional[Dict[str, object]]) -> None:
            if not isinstance(entry, dict):
                return
            if id(entry) in visited:
                return
            visited.add(id(entry))

            def _inherit_field(source: Dict[str, object], field: str) -> None:
                if entry.get(field) in (None, ""):
                    candidate = source.get(field)
                    if candidate not in (None, ""):
                        entry[field] = candidate

            for container_key in ("draft", "dispatch"):
                candidate = entry.get(container_key)
                if isinstance(candidate, dict):
                    _inherit_field(candidate, "supplier_id")
                    _inherit_field(candidate, "recipients")
                    nested = candidate.get("draft") if isinstance(candidate.get("draft"), dict) else None
                    if isinstance(nested, dict):
                        _inherit_field(nested, "supplier_id")
                        _inherit_field(nested, "recipients")

            if "draft" in entry and not entry.get("dispatches") and not entry.get("drafts"):
                # Pure metadata container; rely on nested draft entries instead
                return

            def _has_signal(value: object) -> bool:
                if value in (None, "", False, 0):
                    return False
                if isinstance(value, (list, tuple, set, dict)):
                    return bool(value)
                return True

            has_recipient_or_id = any(
                entry.get(key) not in (None, "")
                for key in ("recipients", "draft_record_id", "id", "message_id", "dispatch_message_id")
            )
            if entry.get("supplier_id") not in (None, "") and not has_recipient_or_id:
                return

            if not any(
                _has_signal(entry.get(key))
                for key in (
                    "recipients",
                    "supplier_id",
                    "rfq_id",
                    "draft_record_id",
                    "id",
                    "message_id",
                    "dispatch_message_id",
                )
            ):
                return

            if self._should_skip_dispatch_entry(entry):
                return
            key = _key(entry)
            if key is None or key in seen:
                return
            seen.add(key)
            entries.append(entry)

        def _walk(obj: object) -> None:
            if isinstance(obj, dict):
                _add(obj)
                for candidate_key in ("draft", "dispatch"):
                    candidate = obj.get(candidate_key)
                    if isinstance(candidate, dict):
                        _walk(candidate)
                for group_key in ("drafts", "dispatches"):
                    group = obj.get(group_key)
                    if isinstance(group, dict):
                        _walk(group)
                    elif isinstance(group, (list, tuple, set)):
                        for item in group:
                            _walk(item)
                for value in obj.values():
                    if isinstance(value, dict):
                        _walk(value)
                    elif isinstance(value, (list, tuple, set)):
                        for item in value:
                            _walk(item)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _walk(item)

        _walk(payload)
        return entries

    def _build_dispatch_expectation(
        self,
        action_id: Optional[str],
        workflow_id: Optional[str],
        drafts: Iterable[object],
    ) -> Optional["_DispatchExpectation"]:
        draft_ids: List[int] = []
        supplier_keys: Set[str] = set()
        draft_count = 0
        run_identifiers: Set[str] = set()

        unique_dispatch_tokens: Set[str] = set()

        for entry in drafts:
            if not isinstance(entry, dict):
                continue
            if self._should_skip_dispatch_entry(entry):
                continue
            tokens = self._dispatch_entry_unique_key(entry)
            if tokens:
                if any(token in unique_dispatch_tokens for token in tokens):
                    continue
                unique_dispatch_tokens.update(tokens)
            draft_count += 1
            candidate_id = entry.get("draft_record_id") or entry.get("id")
            try:
                if candidate_id is not None:
                    draft_ids.append(int(candidate_id))
            except Exception:
                logger.debug("Ignoring non-numeric draft identifier: %r", candidate_id)
            supplier = entry.get("supplier_id")
            supplier_key = self._coerce_identifier(supplier)
            if supplier_key:
                supplier_keys.add(supplier_key.lower())
            run_identifier = self._dispatch_entry_run_identifier(entry)
            if run_identifier:
                normalised_run = self._normalise_filter_value(run_identifier)
                if normalised_run:
                    run_identifiers.add(normalised_run)
                else:
                    run_identifiers.add(run_identifier)

        if draft_count == 0:
            return None

        expectation = self._DispatchExpectation(
            action_id=action_id,
            workflow_id=workflow_id,
            draft_ids=tuple(sorted(set(draft_ids))),
            draft_count=draft_count,
            supplier_count=len(supplier_keys),
            run_identifiers=tuple(sorted(run_identifiers)),
        )

        if workflow_id:
            for run_value in expectation.run_identifiers:
                self._remember_workflow_run(run_value, workflow_id)

        return expectation

    def _load_dispatch_expectation(
        self,
        action_id: Optional[str],
        workflow_id: Optional[str],
    ) -> Optional["_DispatchExpectation"]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None

        rows: List[Tuple[object, object]] = []
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    if action_id:
                        cur.execute(
                            """
                            SELECT action_id, process_output
                            FROM proc.action
                            WHERE action_id = %s
                            """,
                            (action_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            rows.append(row)
                    else:
                        cur.execute(
                            """
                            SELECT action_id, process_output
                            FROM proc.action
                            WHERE agent_type = %s OR agent_type = %s
                            ORDER BY action_date DESC
                            LIMIT 20
                            """,
                            ("EmailDraftingAgent", "email_dispatch"),
                        )
                        rows.extend(cur.fetchall() or [])
        except Exception:
            logger.exception(
                "Failed to load dispatch expectation for action=%s workflow=%s",
                action_id,
                workflow_id,
            )
            return None

        workflow_key = self._normalise_filter_value(workflow_id)
        for row_action_id, payload in rows:
            output = self._coerce_process_output(payload)
            if not isinstance(output, dict):
                continue
            candidate_workflow = self._coerce_identifier(output.get("workflow_id"))
            if workflow_key and candidate_workflow:
                if self._normalise_filter_value(candidate_workflow) != workflow_key:
                    continue
            elif workflow_key and not action_id:
                # Without an explicit action identifier prefer matching workflow metadata
                continue

            dispatch_entries = self._extract_dispatch_entries(output)
            if dispatch_entries:
                drafts = dispatch_entries
            else:
                drafts = (
                    output.get("drafts") if isinstance(output.get("drafts"), list) else []
                )
            expectation = self._build_dispatch_expectation(
                self._coerce_identifier(row_action_id),
                candidate_workflow or workflow_id,
                drafts,
            )
            if expectation is not None:
                return expectation

        return None

    def _resolve_dispatch_expectation(
        self, filters: Optional[Dict[str, object]]
    ) -> Optional["_DispatchExpectation"]:
        if not filters:
            return None

        workflow_candidate = self._coerce_identifier(filters.get("workflow_id"))
        run_candidate = self._coerce_identifier(
            filters.get("dispatch_run_id") or filters.get("run_id")
        )
        workflow_key = self._normalise_filter_value(workflow_candidate) if workflow_candidate else None
        group_key = self._normalise_group_key(run_candidate, workflow_candidate)
        action_candidate: Optional[str] = None
        for key in ("action_id", "draft_action_id", "email_action_id"):
            candidate = self._coerce_identifier(filters.get(key)) if filters else None
            if candidate:
                action_candidate = candidate
                break

        mapping_key = group_key or workflow_key
        if mapping_key and action_candidate:
            self._workflow_dispatch_actions[mapping_key] = action_candidate
        elif mapping_key and mapping_key in self._workflow_dispatch_actions:
            action_candidate = self._workflow_dispatch_actions[mapping_key]

        if not action_candidate and not (workflow_candidate or run_candidate):
            return None

        if action_candidate and action_candidate in self._completed_dispatch_actions:
            return None

        if action_candidate and action_candidate in self._dispatch_expectations:
            return self._dispatch_expectations[action_candidate]

        expectation = self._load_dispatch_expectation(action_candidate, workflow_candidate)
        if expectation and expectation.action_id:
            self._dispatch_expectations[expectation.action_id] = expectation
            if expectation.workflow_id:
                mapped_key = self._normalise_filter_value(expectation.workflow_id)
                if mapped_key:
                    self._workflow_dispatch_actions[mapped_key] = expectation.action_id
        return expectation

    def _count_sent_drafts(self, expectation: "_DispatchExpectation") -> Optional[int]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return None
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    sent_counts: List[int] = []
                    needs_fallback = False

                    if expectation.draft_ids:
                        cur.execute(
                            """
                            SELECT COUNT(*)
                            FROM proc.draft_rfq_emails
                            WHERE sent = TRUE AND id = ANY(%s)
                            """,
                            (list(expectation.draft_ids),),
                        )
                        row = cur.fetchone()
                        id_count = int(row[0]) if row else 0
                        sent_counts.append(id_count)
                        if expectation.action_id and (
                            len(expectation.draft_ids) < expectation.draft_count
                            or id_count < expectation.draft_count
                        ):
                            needs_fallback = True
                    elif expectation.action_id:
                        needs_fallback = True
                    else:
                        return None

                    if expectation.action_id and needs_fallback:
                        action_id = expectation.action_id
                        try:
                            fallback_params = (
                                json.dumps({"action_id": action_id}),
                                json.dumps({"draft_action_id": action_id}),
                                json.dumps({"email_action_id": action_id}),
                                json.dumps({"action_id": action_id}),
                                json.dumps({"draft_action_id": action_id}),
                                json.dumps({"email_action_id": action_id}),
                                json.dumps({"action_id": action_id}),
                            )
                        except Exception:
                            fallback_params = (
                                json.dumps({"action_id": str(action_id)}),
                                json.dumps({"draft_action_id": str(action_id)}),
                                json.dumps({"email_action_id": str(action_id)}),
                                json.dumps({"action_id": str(action_id)}),
                                json.dumps({"draft_action_id": str(action_id)}),
                                json.dumps({"email_action_id": str(action_id)}),
                                json.dumps({"action_id": str(action_id)}),
                            )

                        cur.execute(
                            """
                            SELECT COUNT(*)
                            FROM proc.draft_rfq_emails
                            WHERE sent = TRUE
                              AND (
                                payload::jsonb @> %s::jsonb
                                OR payload::jsonb @> %s::jsonb
                                OR payload::jsonb @> %s::jsonb
                                OR payload::jsonb -> 'metadata' @> %s::jsonb
                                OR payload::jsonb -> 'metadata' @> %s::jsonb
                                OR payload::jsonb -> 'metadata' @> %s::jsonb
                                OR payload::jsonb -> 'dispatch_metadata' @> %s::jsonb
                              )
                            """,
                            fallback_params,
                        )
                        row = cur.fetchone()
                        sent_counts.append(int(row[0]) if row else 0)

                    if not sent_counts:
                        return 0

                    return max(sent_counts)
        except Exception:
            logger.exception(
                "Failed to count sent drafts for action=%s",
                expectation.action_id,
            )
            return None

    def _wait_for_dispatch_completion(
        self, expectation: "_DispatchExpectation"
    ) -> bool:
        if expectation.draft_count <= 0:
            return True

        timeout = max(self._dispatch_wait_seconds, expectation.draft_count * 60)
        timeout = min(timeout, 300)
        base_interval = self._dispatch_wait_seconds if self._dispatch_wait_seconds > 0 else 5.0
        poll_interval = max(1.0, min(10.0, float(base_interval)))
        deadline = time.time() + timeout

        while True:
            sent_count = self._count_sent_drafts(expectation)
            if sent_count is None:
                return False
            if sent_count >= expectation.draft_count:
                logger.info(
                    "Detected %d/%d dispatched RFQ emails for action=%s (workflow=%s); proceeding with inbound poll",
                    sent_count,
                    expectation.draft_count,
                    expectation.action_id,
                    expectation.workflow_id,
                )
                if self._post_dispatch_settle_seconds > 0:
                    logger.info(
                        "Waiting %ds after dispatch completion before polling inbound for mailbox %s",
                        self._post_dispatch_settle_seconds,
                        self.mailbox_address,
                    )
                    time.sleep(self._post_dispatch_settle_seconds)
                return True

            now = time.time()
            if now >= deadline:
                logger.warning(
                    "Timed out waiting for %d dispatched RFQ emails (current=%d) for action=%s",
                    expectation.draft_count,
                    sent_count,
                    expectation.action_id,
                )
                return False

            sleep_for = min(poll_interval, max(0.5, deadline - now))
            logger.debug(
                "Waiting %.1fs for dispatched RFQ emails (current=%d/%d, action=%s)",
                sleep_for,
                sent_count,
                expectation.draft_count,
                expectation.action_id,
            )
            time.sleep(sleep_for)

    def _acknowledge_recent_dispatch(
        self,
        expectation: "_DispatchExpectation",
        completed: bool,
    ) -> None:
        expected_count = max(0, expectation.draft_count)
        if expected_count == 0:
            return

        drafts = self._fetch_recent_dispatched_drafts(expectation, expected_count)
        if not drafts:
            return

        unmatched: List[SESEmailWatcher._DraftSnapshot] = list(drafts)

        def _record_message(
            source_label: str,
            message: Dict[str, object],
            match: SESEmailWatcher._DraftSnapshot,
        ) -> None:
            message_id = str(message.get("id") or "")
            if not message_id:
                return
            workflow_id = match.workflow_id or expectation.workflow_id
            if workflow_id:
                self._remember_workflow_run(match.run_id or match.dispatch_token, workflow_id)
            metadata = {
                "status": "dispatch_copy",
                "draft_id": match.id,
                "rfq_id": match.rfq_id,
                "run_id": match.run_id,
                "matched_via": match.matched_via,
                "dispatch_completed": completed,
            }
            if workflow_id:
                metadata["workflow_id"] = workflow_id
            if match.run_id:
                metadata["dispatch_run_id"] = match.run_id
            if self.state_store is not None:
                self.state_store.add(message_id, metadata)
            last_modified = message.get("_last_modified")
            if isinstance(last_modified, datetime):
                self._update_watermark(last_modified, message_id)
            prefix_hint = message.get("_prefix")
            if isinstance(prefix_hint, str):
                watcher = self._s3_prefix_watchers.get(prefix_hint)
                if watcher is not None:
                    watcher.mark_known(
                        message_id,
                        last_modified if isinstance(last_modified, datetime) else None,
                    )
            logger.debug(
                "Recorded dispatched email copy %s for draft_id=%s (matched_via=%s, source=%s)",
                message_id,
                match.id,
                match.matched_via,
                source_label,
            )

        def _consume_batch(
            source_label: str, messages: Iterable[Dict[str, object]]
        ) -> None:
            for message in messages:
                if not unmatched:
                    break
                match = self._match_dispatched_message(message, unmatched)
                if match is None:
                    continue
                unmatched.remove(match)
                _record_message(source_label, message, match)

        loaders: List[Tuple[str, Callable[[], List[Dict[str, object]]]]] = []
        if self._imap_configured():
            loaders.append(
                (
                    "imap",
                    lambda: self._load_from_imap(
                        expected_count, mark_seen=False, since=None
                    ),
                )
            )
        if self.bucket:
            loaders.append(
                (
                    "s3",
                    lambda: self._load_from_s3(
                        expected_count,
                        prefixes=self._prefixes,
                        newest_first=True,
                    ),
                )
            )

        for source_label, loader in loaders:
            if not unmatched:
                break
            try:
                messages = loader() or []
            except Exception:
                logger.exception(
                    "Failed to load recent dispatch copies from %s for action=%s",
                    source_label.upper(),
                    expectation.action_id,
                )
                continue

            if not messages:
                logger.debug(
                    "No dispatched copies retrieved from %s for action=%s",
                    source_label.upper(),
                    expectation.action_id,
                )
                continue

            _consume_batch(source_label, messages)

    def _snapshot_from_draft_row(
        self,
        row: Sequence[object],
    ) -> Optional["_DraftSnapshot"]:
        if not row:
            return None

        try:
            draft_id = row[0]
            rfq_id = row[1]
            supplier_id = row[2] if len(row) > 2 else None
            subject = row[3] if len(row) > 3 else ""
            body = row[4] if len(row) > 4 else ""
            payload_raw = row[5] if len(row) > 5 else None
        except Exception:
            return None

        payload_doc = self._safe_parse_json(payload_raw) if payload_raw else None

        dispatch_token: Optional[str] = None
        run_id: Optional[str] = None
        recipients: Tuple[str, ...] = ()
        workflow_id: Optional[str] = None
        if isinstance(payload_doc, dict):
            workflow_id = self._extract_workflow_id_from_payload(payload_doc) or None
            dispatch_meta = payload_doc.get("dispatch_metadata")
            if isinstance(dispatch_meta, dict):
                dispatch_token = dispatch_meta.get("dispatch_token") or dispatch_meta.get("token")
                run_id = (
                    dispatch_meta.get("run_id")
                    or dispatch_meta.get("dispatch_run_id")
                    or run_id
                )
            meta_field = payload_doc.get("metadata")
            if isinstance(meta_field, dict):
                if not dispatch_token:
                    dispatch_token = meta_field.get("dispatch_token")
                if run_id is None:
                    run_id = meta_field.get("run_id")
            payload_subject = payload_doc.get("subject")
            payload_body = payload_doc.get("body") or payload_doc.get("negotiation_message")
            if not subject and isinstance(payload_subject, str):
                subject = payload_subject
            if isinstance(payload_body, str) and payload_body.strip():
                body = payload_body
            recipients_field = payload_doc.get("recipients")
            if isinstance(recipients_field, (list, tuple)):
                recipients = tuple(
                    str(item).strip()
                    for item in recipients_field
                    if isinstance(item, str) and item.strip()
                )

        if run_id is None and dispatch_token:
            run_id = dispatch_token

        try:
            draft_id_int = int(draft_id)
        except Exception:
            logger.debug("Ignoring draft with non-numeric id: %r", draft_id)
            return None

        return self._DraftSnapshot(
            id=draft_id_int,
            rfq_id=str(rfq_id) if rfq_id is not None else None,
            subject=str(subject or ""),
            body=str(body or ""),
            dispatch_token=dispatch_token,
            run_id=run_id,
            recipients=recipients,
            supplier_id=str(supplier_id) if supplier_id is not None else None,
            workflow_id=workflow_id,
        )

    def _fetch_recent_dispatched_drafts(
        self,
        expectation: "_DispatchExpectation",
        limit: int,
    ) -> List["_DraftSnapshot"]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return []

        query: str
        params: Tuple[object, ...]
        if expectation.draft_ids:
            query = (
                """
                SELECT id, rfq_id, supplier_id, subject, body, payload
                FROM proc.draft_rfq_emails
                WHERE id = ANY(%s)
                ORDER BY COALESCE(sent_on, updated_on) DESC, updated_on DESC, created_on DESC
                LIMIT %s
                """
            )
            params = (list(expectation.draft_ids), limit)
        else:
            query = (
                """
                SELECT id, rfq_id, supplier_id, subject, body, payload
                FROM proc.draft_rfq_emails
                WHERE sent = TRUE
                ORDER BY COALESCE(sent_on, updated_on) DESC, updated_on DESC, created_on DESC
                LIMIT %s
                """
            )
            params = (limit,)

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall() or []
        except Exception:
            logger.exception(
                "Failed to fetch dispatched draft rows for action=%s",
                expectation.action_id,
            )
            return []

        snapshots: List[SESEmailWatcher._DraftSnapshot] = []
        for row in rows:
            snapshot = self._snapshot_from_draft_row(row)
            if snapshot is not None:
                if expectation.workflow_id and not snapshot.workflow_id:
                    snapshot.workflow_id = expectation.workflow_id
                if snapshot.workflow_id:
                    self._remember_workflow_run(
                        snapshot.run_id or snapshot.dispatch_token,
                        snapshot.workflow_id,
                    )
                snapshots.append(snapshot)

        return snapshots

    def _load_recent_drafts_for_fallback(
        self,
        supplier_filter: Optional[str],
        *,
        limit: int = 10,
        within_minutes: int = 10,
    ) -> List["_DraftSnapshot"]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return []

        if not supplier_filter:
            return []

        try:
            interval_minutes = max(0, int(within_minutes))
        except Exception:
            interval_minutes = 0

        if interval_minutes <= 0:
            return []

        query = (
            """
            SELECT id, rfq_id, supplier_id, subject, body, payload
            FROM proc.draft_rfq_emails
            WHERE sent = TRUE
              AND COALESCE(sent_on, updated_on, created_on) >= (
                CURRENT_TIMESTAMP - (%s * INTERVAL '1 minute')
              )
              AND LOWER(supplier_id) = %s
            ORDER BY COALESCE(sent_on, updated_on) DESC, updated_on DESC, created_on DESC
            LIMIT %s
            """
        )

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (interval_minutes, supplier_filter, limit))
                    rows = cur.fetchall() or []
        except Exception:
            logger.exception(
                "Failed to load recent draft rows for fallback supplier match (supplier=%s)",
                supplier_filter,
            )
            return []

        snapshots: List[SESEmailWatcher._DraftSnapshot] = []
        for row in rows:
            snapshot = self._snapshot_from_draft_row(row)
            if snapshot is not None:
                snapshots.append(snapshot)

        return snapshots

    def _fallback_match_via_recent_drafts(
        self,
        processed_payload: Dict[str, object],
        filters: Dict[str, object],
    ) -> bool:
        if not processed_payload or not filters:
            return False

        supplier_filter = self._normalise_filter_value(
            processed_payload.get("supplier_id")
            or filters.get("supplier_id")
        )
        if not supplier_filter:
            return False

        drafts = self._load_recent_drafts_for_fallback(
            supplier_filter,
            limit=10,
            within_minutes=10,
        )
        if not drafts:
            return False

        match = None
        for draft in drafts:
            if draft.supplier_norm == supplier_filter:
                match = draft
                break

        if match is None:
            return False

        if match.rfq_id and not processed_payload.get("rfq_id"):
            processed_payload["rfq_id"] = match.rfq_id
        if match.supplier_id and not processed_payload.get("supplier_id"):
            processed_payload["supplier_id"] = match.supplier_id
        if not processed_payload.get("dispatch_run_id"):
            run_value = match.run_id or match.dispatch_token
            if run_value:
                processed_payload["dispatch_run_id"] = run_value

        processed_payload["matched_via"] = "fallback"
        processed_payload["match_score"] = 0.5

        return True

    def _hydrate_payload_from_marker(
        self,
        payload: Optional[Dict[str, object]],
        message: Dict[str, object],
    ) -> None:
        if not payload:
            return

        body_source = payload.get("message_body") or message.get("body") or ""
        body_text = str(body_source)
        if not body_text:
            return

        comment, _ = split_hidden_marker(body_text)
        if not comment:
            return

        if not payload.get("dispatch_run_id"):
            run_identifier = extract_run_id(comment)
            if run_identifier:
                payload["dispatch_run_id"] = run_identifier

        if not payload.get("dispatch_token"):
            token = extract_marker_token(comment)
            if token:
                payload.setdefault("dispatch_token", token)

        if not payload.get("rfq_id"):
            rfq_hint = extract_rfq_id(comment)
            if rfq_hint:
                payload["rfq_id"] = rfq_hint

    def _match_dispatched_message(
        self,
        message: Dict[str, object],
        drafts: List["_DraftSnapshot"],
    ) -> Optional["_DraftSnapshot"]:
        if not drafts:
            return None

        body = str(message.get("body") or "")
        comment, remainder = split_hidden_marker(body)
        token = extract_marker_token(comment)
        run_identifier = extract_run_id(comment)
        subject_norm = _norm(str(message.get("subject") or ""))
        body_norm = _norm(remainder or body)

        # 1) Match by run identifier / dispatch token
        identifier = run_identifier or token
        if identifier:
            for draft in drafts:
                candidate = draft.run_id or draft.dispatch_token
                if candidate and identifier == candidate:
                    object.__setattr__(draft, "matched_via", "dispatch_token")
                    return draft

        # 2) Match by normalised subject/body
        for draft in drafts:
            if draft.subject_norm and subject_norm:
                if subject_norm == draft.subject_norm:
                    object.__setattr__(draft, "matched_via", "subject")
                    return draft
            if draft.body_norm and body_norm and draft.body_norm == body_norm:
                object.__setattr__(draft, "matched_via", "body")
                return draft
            if draft.body_norm and body_norm and (
                draft.body_norm in body_norm or body_norm in draft.body_norm
            ):
                object.__setattr__(draft, "matched_via", "body_contains")
                return draft

        return None

    def _respect_post_dispatch_wait(
        self, filters: Optional[Dict[str, object]] = None
    ) -> Tuple[Optional["_DispatchExpectation"], bool]:
        if self._dispatch_wait_seconds <= 0:
            return None, False

        candidate_time: Optional[float] = self._last_dispatch_notified_at
        agent_time = getattr(self.agent_nick, "email_dispatch_last_sent_at", None)
        if isinstance(agent_time, (int, float)):
            agent_value = float(agent_time)
            candidate_time = agent_value if candidate_time is None else max(candidate_time, agent_value)

        if candidate_time is None:
            return None, False

        if (
            self._last_dispatch_wait_acknowledged is not None
            and candidate_time <= self._last_dispatch_wait_acknowledged
        ):
            return None, False

        expectation = self._resolve_dispatch_expectation(filters)
        completed = False
        if expectation is not None and expectation.draft_count > 0:
            completed = self._wait_for_dispatch_completion(expectation)
            if expectation.action_id:
                self._dispatch_expectations.pop(expectation.action_id, None)
                self._completed_dispatch_actions.add(expectation.action_id)
                if not completed:
                    logger.debug(
                        "Dispatch count check for action=%s ended without reaching target",
                        expectation.action_id,
                    )
            self._last_dispatch_wait_acknowledged = candidate_time
            return expectation, completed

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
        return expectation, False

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

    def _peek_recent_s3_messages(
        self,
        limit: Optional[int] = None,
        *,
        prefixes: Optional[Sequence[str]] = None,
        parser: Optional[Callable[[bytes], Dict[str, object]]] = None,
        newest_first: bool = True,
    ) -> List[Dict[str, object]]:
        if not self.bucket:
            logger.warning("SES inbound bucket not configured; skipping poll")
            return []

        active_prefixes = list(prefixes) if prefixes is not None else list(self._prefixes)
        if not active_prefixes:
            return []

        for prefix in active_prefixes:
            if prefix not in self._s3_prefix_watchers:
                self._s3_prefix_watchers[prefix] = S3ObjectWatcher(
                    limit=self._s3_watch_history_limit
                )

        client = self._get_s3_client()
        parser_fn = parser or self._parse_inbound_object

        object_refs = self._scan_recent_s3_objects(
            client,
            active_prefixes,
            watermark_ts=self._last_watermark_ts,
            watermark_key=self._last_watermark_key,
            enforce_window=True,
        )

        if not object_refs:
            return []

        collected: List[Tuple[Optional[datetime], Dict[str, object]]] = []
        seen_keys: Set[str] = set()

        for prefix, key, last_modified, etag in object_refs:
            if key in seen_keys:
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
            parsed["_prefix"] = prefix
            if size_bytes is not None:
                parsed["_content_length"] = size_bytes
            collected.append((last_modified, parsed))
            seen_keys.add(key)

            if limit is not None and len(collected) >= limit:
                break

        if limit is not None and len(collected) >= limit:
            logger.debug(
                "Reached dispatch peek limit (%s) for mailbox %s",
                limit,
                self.mailbox_address,
            )

        collected.sort(key=lambda item: item[0] or datetime.min, reverse=newest_first)
        return [payload for _, payload in collected]

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
            parsed["_prefix"] = prefix
            if size_bytes is not None:
                parsed["_content_length"] = size_bytes

            message_identifier = parsed.get("message_id")
            if isinstance(message_identifier, bytes):
                try:
                    message_identifier = message_identifier.decode()
                except Exception:
                    message_identifier = str(message_identifier)
            if isinstance(message_identifier, str):
                message_identifier = message_identifier.strip()
            if (
                isinstance(message_identifier, str)
                and message_identifier
                and self.state_store
                and message_identifier in self.state_store
            ):
                watcher.mark_known(key, last_modified)
                logger.debug(
                    "Skipping S3 object %s for mailbox %s; message_id %s already processed",
                    key,
                    self.mailbox_address,
                    message_identifier,
                )
                continue

            body_text = str(parsed.get("body") or "")
            comment, _body_remainder = split_hidden_marker(body_text)
            token = extract_marker_token(comment)
            run_identifier = extract_run_id(comment) or token
            if run_identifier:
                parsed["dispatch_run_id"] = run_identifier
                parsed.setdefault("dispatch_token", run_identifier)
            if token and "dispatch_token" not in parsed:
                parsed["dispatch_token"] = token
            if comment and "rfq_id" not in parsed:
                rfq_hint = extract_rfq_id(comment)
                if rfq_hint:
                    parsed["rfq_id"] = rfq_hint
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
        if messages:
            ids = [
                str(message.get("message_id") or message.get("id") or "<no-id>")
                for message in messages
            ]
            times = [
                str(message.get("received_at") or message.get("_last_modified") or "?")
                for message in messages
            ]
            logger.debug(
                "S3 fallback retrieved %d message(s) for mailbox %s: ids=%s times=%s",
                len(messages),
                self.mailbox_address,
                ids,
                times,
            )
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
        if candidate == reference:
            new_uid = _extract_imap_uid(key)
            old_uid = _extract_imap_uid(watermark_key)
            if new_uid is not None and old_uid is not None:
                return new_uid > old_uid
            if key > watermark_key:
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
        else:
            reference = self._last_watermark_ts
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)

            if candidate > reference:
                self._last_watermark_ts = candidate
                self._last_watermark_key = key
            elif candidate == reference:
                new_uid = _extract_imap_uid(key)
                old_uid = _extract_imap_uid(self._last_watermark_key)
                if new_uid is not None and old_uid is not None:
                    if new_uid > old_uid:
                        self._last_watermark_ts = candidate
                        self._last_watermark_key = key
                elif key > self._last_watermark_key:
                    self._last_watermark_ts = candidate
                    self._last_watermark_key = key

        new_uid_value = _extract_imap_uid(key)
        if new_uid_value is not None:
            previous_uid = self._last_uid or 0
            if new_uid_value > previous_uid:
                self._last_uid = new_uid_value

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
        in_reply_to_header = message.get("In-Reply-To") or message.get("in-reply-to")
        references_raw = message.get_all("References", []) or []
        references: List[str] = []
        for entry in references_raw:
            if entry in (None, ""):
                continue
            try:
                text = str(entry).strip()
            except Exception:
                continue
            if text:
                references.append(text)
        headers: Dict[str, object] = {}
        for key in message.keys():
            if key in headers:
                continue
            try:
                headers[key] = message.get(key)
            except Exception:
                headers[key] = message.get(key, "")
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
            "in_reply_to": in_reply_to_header,
            "references": references,
            "headers": headers,
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

    def _resolve_rfq_from_thread_headers(
        self, message: Dict[str, object]
    ) -> Optional[Tuple[str, Optional[str]]]:
        identifiers = self._collect_thread_identifiers(message)
        if not identifiers:
            return None

        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn) or not self._thread_table_name:
            return None

        try:
            with get_conn() as conn:
                if not self._ensure_thread_table_ready(conn):
                    return None
                thread_info = lookup_thread_metadata(
                    conn,
                    self._thread_table_name,
                    identifiers,
                    logger=logger,
                )
        except Exception:
            logger.exception(
                "Failed to resolve RFQ from thread headers for mailbox %s", self.mailbox_address
            )
            return None

        if not thread_info:
            return None

        rfq_id, supplier_id = thread_info
        if not rfq_id:
            return None

        supplier_value = str(supplier_id).strip() if supplier_id not in (None, "") else None
        return str(rfq_id), supplier_value

    def _collect_thread_identifiers(self, message: Dict[str, object]) -> List[str]:
        raw_values: List[str] = []

        def _collect(value: object) -> None:
            if value in (None, ""):
                return
            if isinstance(value, (list, tuple, set)):
                for entry in value:
                    _collect(entry)
                return
            try:
                text = str(value)
            except Exception:
                return
            trimmed = text.strip()
            if trimmed:
                raw_values.append(trimmed)

        _collect(message.get("in_reply_to"))
        _collect(message.get("references"))

        headers = message.get("headers")
        if isinstance(headers, dict):
            _collect(headers.get("In-Reply-To") or headers.get("in-reply-to"))
            reference_values = headers.get("References") or headers.get("references")
            _collect(reference_values)

        if not raw_values:
            return []

        message_ids: List[str] = []
        seen: Set[str] = set()
        pattern = re.compile(r"<([^>]+)>")
        for value in raw_values:
            matches = pattern.findall(value)
            if matches:
                for candidate in matches:
                    trimmed = candidate.strip()
                    if not trimmed:
                        continue
                    lowered = trimmed.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    message_ids.append(trimmed)
                continue

            trimmed = value.strip()
            if trimmed.startswith("<") and trimmed.endswith(">"):
                trimmed = trimmed[1:-1].strip()
            if not trimmed:
                continue
            lowered = trimmed.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            message_ids.append(trimmed)

        return message_ids

    def _build_thread_headers_for_reply(
        self, message: Dict[str, object]
    ) -> Optional[Dict[str, object]]:
        message_identifier = message.get("message_id")
        if isinstance(message_identifier, bytes):
            try:
                message_identifier = message_identifier.decode()
            except Exception:
                message_identifier = str(message_identifier)
        if not isinstance(message_identifier, str):
            return None

        trimmed_id = message_identifier.strip()
        if not trimmed_id:
            return None

        def _strip_brackets(value: str) -> str:
            token = value.strip()
            if token.startswith("<") and token.endswith(">"):
                token = token[1:-1].strip()
            return token

        def _ensure_brackets(value: str) -> str:
            token = value.strip()
            if token.startswith("<") and token.endswith(">"):
                return token
            return f"<{token}>"

        base_identifier = _strip_brackets(trimmed_id)
        references: List[str] = []
        for identifier in self._collect_thread_identifiers(message):
            stripped = _strip_brackets(identifier)
            if not stripped:
                continue
            formatted = _ensure_brackets(stripped)
            if formatted not in references:
                references.append(formatted)

        formatted_base = _ensure_brackets(base_identifier or trimmed_id)
        if formatted_base not in references:
            references.append(formatted_base)

        thread_payload: Dict[str, object] = {"message_id": formatted_base}
        if references:
            thread_payload["references"] = references
        return thread_payload

    def _collect_body_message_ids(self, message: Dict[str, object]) -> List[str]:
        body = message.get("body")
        if not isinstance(body, str) or "<" not in body:
            return []

        pattern = re.compile(r"<([^>]+)>")
        collected: List[str] = []
        seen: Set[str] = set()
        for match in pattern.findall(body):
            candidate = match.strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            collected.append(candidate)
        return collected

    def _priority_for(self, matched_via: str) -> int:
        return self._MATCH_PRIORITIES.get(matched_via, 10)

    def _prefer_candidate(
        self,
        new: "SESEmailWatcher._MatchCandidate",
        existing: "SESEmailWatcher._MatchCandidate",
    ) -> bool:
        if new.score > existing.score:
            return True
        if new.score < existing.score:
            return False

        if new.priority < existing.priority:
            return True
        if new.priority > existing.priority:
            return False

        if (
            new.matched_via == existing.matched_via == "dispatch_chain"
        ):
            new_index = new.thread_index or 0
            existing_index = existing.thread_index or 0
            if new_index > existing_index:
                return True
            if new_index < existing_index:
                return False

            new_created = new.created_at or datetime.min.replace(tzinfo=timezone.utc)
            existing_created = existing.created_at or datetime.min.replace(
                tzinfo=timezone.utc
            )
            if new_created > existing_created:
                return True
            if new_created < existing_created:
                return False

        if new.score == existing.score and new.priority == existing.priority:
            if new.order < existing.order:
                return True

        return False

    def _resolve_rfq_via_dispatch_chain(
        self,
        message: Dict[str, object],
        thread_identifiers: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        get_conn = getattr(self.agent_nick, "get_db_connection", None)
        if not callable(get_conn):
            return []

        supplier_hint = self._normalise_filter_value(message.get("supplier_id"))
        identifiers: List[str] = []
        if thread_identifiers:
            identifiers.extend(thread_identifiers)
        identifiers.extend(self._collect_body_message_ids(message))

        seen: Set[str] = set()
        ordered_identifiers: List[str] = []
        for value in identifiers:
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered_identifiers.append(value)

        matches: List[Dict[str, object]] = []

        lookback_days = max(getattr(self.settings, "dispatch_chain_lookback_days", 5), 1)
        lookback_text = str(lookback_days)

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    if ordered_identifiers:
                        cur.execute(
                            """
                            SELECT rfq_id, supplier_id, dispatch_metadata, thread_index, created_at, message_id
                            FROM proc.email_dispatch_chains
                            WHERE awaiting_response = TRUE
                              AND message_id = ANY(%s)
                              AND (
                                  dispatch_metadata ->> 'mailbox' IS NULL
                                  OR dispatch_metadata ->> 'mailbox' = %s
                              )
                            ORDER BY thread_index DESC, created_at DESC
                            LIMIT 1
                            """,
                            (ordered_identifiers, self.mailbox_address),
                        )
                        direct_row = cur.fetchone()
                    else:
                        direct_row = None

                    if direct_row:
                        matches.append(
                            {
                                "rfq_id": direct_row[0],
                                "supplier_id": direct_row[1],
                                "dispatch_metadata": direct_row[2],
                                "thread_index": direct_row[3],
                                "created_at": direct_row[4],
                                "message_id": direct_row[5],
                                "score": 0.75,
                            }
                        )
                        return matches

                    cur.execute(
                        """
                        SELECT rfq_id, supplier_id, dispatch_metadata, thread_index, created_at, message_id
                        FROM proc.email_dispatch_chains
                        WHERE awaiting_response = TRUE
                          AND created_at >= NOW() - (%s || ' days')::INTERVAL
                          AND (
                              dispatch_metadata ->> 'mailbox' IS NULL
                              OR dispatch_metadata ->> 'mailbox' = %s
                          )
                          AND (%s IS NULL OR supplier_id = %s)
                        ORDER BY thread_index DESC, created_at DESC
                        LIMIT 5
                        """,
                        (lookback_text, self.mailbox_address, supplier_hint, supplier_hint),
                    )
                    rows = cur.fetchall() or []
        except Exception:
            logger.debug("Failed to resolve RFQ via dispatch chain", exc_info=True)
            return matches

        for row in rows:
            score = 0.6
            if supplier_hint:
                score = max(score, 0.7)
            matches.append(
                {
                    "rfq_id": row[0],
                    "supplier_id": row[1],
                    "dispatch_metadata": row[2],
                    "thread_index": row[3],
                    "created_at": row[4],
                    "message_id": row[5],
                    "score": score,
                }
            )

        return matches

    def _ensure_thread_table_ready(self, conn) -> bool:
        if self._thread_table_ready:
            return True

        try:
            ensure_thread_table(conn, self._thread_table_name, logger=logger)
            self._thread_table_ready = True
            return True
        except Exception:
            logger.exception(
                "Failed to ensure email thread table %s", self._thread_table_name
            )
            try:
                conn.rollback()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Rollback failed after ensuring thread table")
            return False

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

