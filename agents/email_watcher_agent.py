"""Agent that monitors supplier reply mailboxes in real time."""

from __future__ import annotations

import contextlib
import imaplib
import logging
import random
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from services.event_bus import get_event_bus
from services.imap_supplier_response_watcher import (
    DatabaseBackend,
    DispatchContext,
    DispatchRecord,
    SupplierResponseRecord,
)
from services.email_watcher_v2 import EmailResponse, _parse_email
from utils.email_markers import (
    extract_marker_token,
    extract_rfq_id,
    extract_run_id,
    extract_supplier_id,
    split_hidden_marker,
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text.upper() or None


def _normalise_subject(subject: Optional[str]) -> Optional[str]:
    if not subject:
        return None
    cleaned = " ".join(str(subject).split())
    return cleaned.lower() or None


@dataclass
class MatchResult:
    dispatch: Optional[DispatchRecord]
    score: float
    reasons: List[str]
    supplier_id: Optional[str]
    rfq_id: Optional[str]
    workflow_id: Optional[str]

    def accepted(self, threshold: float) -> bool:
        return self.dispatch is not None and self.score >= threshold


class ImapIdleFetcher:
    """Fetch emails using IMAP IDLE when possible, falling back to polling."""

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        mailbox: str = "INBOX",
        port: int = 993,
        use_ssl: bool = True,
        login: Optional[str] = None,
        idle_timeout: int = 30,
        poll_interval: int = 60,
        jitter_seconds: float = 5.0,
        backoff_cap_seconds: int = 300,
        backward_window_seconds: int = 120,
        now: Callable[[], datetime] = _now,
        sleep: Callable[[float], None] = time.sleep,
        client_factory: Optional[Callable[[], imaplib.IMAP4]] = None,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.mailbox = mailbox
        self.port = port
        self.use_ssl = use_ssl
        self.login = login
        self.idle_timeout = max(5, idle_timeout)
        self.poll_interval = max(1, poll_interval)
        self.jitter_seconds = max(0.0, jitter_seconds)
        self.backoff_cap_seconds = max(self.poll_interval, backoff_cap_seconds)
        self.backward_window_seconds = max(0, backward_window_seconds)
        self._now = now
        self._sleep = sleep
        self._client_factory = client_factory
        self._client: Optional[imaplib.IMAP4] = None
        self._last_uid: Optional[int] = None
        self._idle_supported: Optional[bool] = None
        self._next_attempt: Optional[datetime] = None
        self._backoff: int = self.poll_interval
        self._selected_mailbox: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> imaplib.IMAP4:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            client = self._client_factory()
        elif self.use_ssl:
            client = imaplib.IMAP4_SSL(self.host, self.port)
        else:
            client = imaplib.IMAP4(self.host, self.port)
        login_name = self.login or self.username
        client.login(login_name, self.password)
        self._client = client
        self._idle_supported = "IDLE" in (getattr(client, "capabilities", lambda: [])() or [])
        return client

    def _ensure_selected(self) -> imaplib.IMAP4:
        client = self._connect()
        if self._selected_mailbox == self.mailbox:
            return client
        try:
            client.select(self.mailbox, readonly=True)
            self._selected_mailbox = self.mailbox
        except Exception:
            self._selected_mailbox = None
            raise
        return client

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            if self._selected_mailbox:
                with contextlib.suppress(Exception):
                    client.close()
        finally:
            self._selected_mailbox = None
            with contextlib.suppress(Exception):
                client.logout()

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def __call__(self, *, since: datetime) -> List[EmailResponse]:
        now = self._now()
        if self._next_attempt and now < self._next_attempt:
            return []
        try:
            client = self._ensure_selected()
        except Exception as exc:
            logger.warning("EmailWatcherAgent failed to connect to IMAP: %s", exc)
            self._schedule_backoff()
            return []

        window_start = since - timedelta(seconds=self.backward_window_seconds)
        if window_start.tzinfo is None:
            window_start = window_start.replace(tzinfo=timezone.utc)
        search_criteria: bytes
        if self._last_uid is None:
            since_str = window_start.strftime("%d-%b-%Y")
            search_criteria = f"(SINCE {since_str})".encode()
        else:
            search_criteria = f"(UID {self._last_uid + 1}:*)".encode()

        try:
            status, data = client.uid("search", None, search_criteria)
        except imaplib.IMAP4.abort as exc:
            logger.warning("EmailWatcherAgent IMAP session aborted: %s", exc)
            self._reset_connection()
            self._schedule_backoff()
            return []
        except (imaplib.IMAP4.error, socket.error) as exc:
            logger.warning("EmailWatcherAgent IMAP error during search: %s", exc)
            self._schedule_backoff()
            return []

        if status != "OK":
            logger.debug("EmailWatcherAgent IMAP search returned status %s", status)
            return []

        raw_ids = (data[0] or b"").decode().split()
        if not raw_ids:
            self._backoff = self.poll_interval
            return []

        responses: List[EmailResponse] = []
        for raw_uid in raw_ids:
            try:
                uid = int(raw_uid)
            except ValueError:
                continue
            if self._last_uid is not None and uid <= self._last_uid:
                continue
            status, payload = client.uid("fetch", raw_uid, "(RFC822)")
            if status != "OK" or not payload or not isinstance(payload[0], tuple):
                continue
            raw_email = payload[0][1]
            try:
                email = _parse_email(raw_email)
            except Exception:
                logger.exception("EmailWatcherAgent failed to parse IMAP payload")
                continue
            responses.append(email)
            logger.info(
                "EmailWatcherAgent fetched message uid=%s message_id=%s", raw_uid, email.message_id
            )
            if self._last_uid is None or uid > self._last_uid:
                self._last_uid = uid

        self._backoff = self.poll_interval
        self._next_attempt = None
        return responses

    def wait(self) -> None:
        jitter = random.uniform(0.0, self.jitter_seconds) if self.jitter_seconds else 0.0
        wait_seconds = float(self.poll_interval) + jitter
        if self._idle_supported:
            try:
                client = self._ensure_selected()
                self._idle(client, wait_seconds)
                return
            except Exception:
                logger.debug("EmailWatcherAgent falling back to polling", exc_info=True)
                self._idle_supported = False
        self._sleep(wait_seconds)

    def _idle(self, client: imaplib.IMAP4, duration: float) -> None:
        try:
            if hasattr(client, "idle") and callable(getattr(client, "idle")):
                client.idle(timeout=duration)  # type: ignore[attr-defined]
                return
        except Exception:
            logger.debug("EmailWatcherAgent native IDLE failed", exc_info=True)
        start = time.time()
        tag = client._new_tag()  # type: ignore[attr-defined]
        command = f"{tag} IDLE\r\n".encode()
        try:
            client.send(command)  # type: ignore[attr-defined]
            client.readline()  # type: ignore[attr-defined]
            remaining = max(0.0, duration - (time.time() - start))
            if remaining:
                self._sleep(remaining)
            client.send(b"DONE\r\n")  # type: ignore[attr-defined]
            client.readline()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("EmailWatcherAgent raw IDLE unsupported", exc_info=True)
            self._idle_supported = False

    def _schedule_backoff(self) -> None:
        self._backoff = min(self.backoff_cap_seconds, max(self.poll_interval, self._backoff * 2))
        self._next_attempt = self._now() + timedelta(seconds=self._backoff)
        logger.info("EmailWatcherAgent backing off for %.1fs", float(self._backoff))

    def _reset_connection(self) -> None:
        if self._client is None:
            return
        with contextlib.suppress(Exception):
            self._client.logout()
        self._client = None
        self._selected_mailbox = None


class EmailWatcherAgent(BaseAgent):
    """Monitor IMAP mailboxes for supplier responses after dispatch."""

    AGENTIC_PLAN_STEPS: Tuple[str, ...] = (
        "Confirm the dispatch round has completed before accessing the mailbox.",
        "Monitor the IMAP mailbox for new supplier replies and correlate them to dispatch records.",
        "Persist matched responses to the supplier response table and report completion status.",
    )

    def __init__(
        self,
        agent_nick,
        *,
        fetcher_factory: Optional[Callable[..., ImapIdleFetcher]] = None,
        now: Callable[[], datetime] = _now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(agent_nick)
        self._fetcher_factory = fetcher_factory or ImapIdleFetcher
        self._now = now
        self._sleep = sleep

    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        workflow_id = context.workflow_id
        payload = dict(context.input_data or {})
        round_number = payload.get("round") or payload.get("round_number")
        dispatch_run_id = payload.get("dispatch_run_id") or payload.get("run_id")
        expected = payload.get("expected_responses") or payload.get("expected_count")
        timeout_seconds = int(payload.get("timeout_seconds") or 1800)
        poll_seconds = int(
            payload.get("poll_seconds")
            or getattr(self.settings, "email_response_poll_seconds", 60)
            or 60
        )
        initial_wait = payload.get("initial_wait_seconds")
        if initial_wait is None:
            initial_wait = getattr(self.settings, "email_inbound_initial_wait_seconds", 90)
        mailbox = payload.get("mailbox") or getattr(self.settings, "imap_mailbox", "INBOX")
        threshold = float(payload.get("match_threshold") or 0.65)
        skip_barrier = bool(payload.get("skip_dispatch_barrier"))

        db = DatabaseBackend()
        db.ensure_schema()
        dispatch_records, last_dispatch = db.fetch_dispatch_context(
            workflow_id=workflow_id,
            action_id=str(round_number) if round_number is not None else None,
            run_id=dispatch_run_id,
        )
        dispatch_context = DispatchContext.build(dispatch_records)

        expected_total = expected or len(dispatch_context.records)
        if expected_total <= 0:
            logger.info(
                "EmailWatcherAgent no expected responses workflow=%s round=%s", workflow_id, round_number
            )
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.SUCCESS,
                    data={
                        "workflow_id": workflow_id,
                        "round": round_number,
                        "expected_responses": 0,
                        "responses_received": 0,
                        "timeout": False,
                        "supplier_responses": [],
                    },
                ),
            )

        if not skip_barrier:
            self._wait_for_dispatch_barrier(
                workflow_id=workflow_id,
                round_number=round_number,
                expected=expected_total,
                wait_seconds=max(0, int(initial_wait or 0)),
            )

        email_fetcher = payload.get("email_fetcher")
        fetcher_instance: Optional[ImapIdleFetcher] = None
        if email_fetcher is None:
            host = payload.get("imap_host") or getattr(self.settings, "imap_host", None)
            username = payload.get("imap_username") or getattr(self.settings, "imap_username", None)
            password = payload.get("imap_password") or getattr(self.settings, "imap_password", None)
            if not all([host, username, password]):
                raise RuntimeError("IMAP credentials are required for EmailWatcherAgent")
            fetcher_instance = self._fetcher_factory(
                host=str(host),
                username=str(username),
                password=str(password),
                mailbox=str(mailbox),
                port=int(payload.get("imap_port") or getattr(self.settings, "imap_port", 993) or 993),
                use_ssl=bool(payload.get("imap_use_ssl", True)),
                login=payload.get("imap_login") or getattr(self.settings, "imap_login", None),
                idle_timeout=poll_seconds,
                poll_interval=poll_seconds,
                jitter_seconds=payload.get("poll_jitter", 5.0) or 5.0,
                backward_window_seconds=payload.get("lookback_seconds", 120) or 120,
                now=self._now,
                sleep=self._sleep,
            )
            email_fetcher = fetcher_instance

        summary = self._collect_responses(
            workflow_id=workflow_id,
            round_number=round_number,
            dispatch_run_id=dispatch_run_id,
            dispatch_context=dispatch_context,
            expected=expected_total,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            threshold=threshold,
            mailbox=str(mailbox),
            email_fetcher=email_fetcher,
            last_dispatch_time=last_dispatch,
        )

        if fetcher_instance is not None:
            fetcher_instance.close()

        status = AgentStatus.SUCCESS
        if summary["timeout"] and summary["responses_received"] < expected_total:
            status = AgentStatus.FAILED

        return self._with_plan(
            context,
            AgentOutput(
                status=status,
                data=summary,
            ),
        )

    # ------------------------------------------------------------------
    def _collect_responses(
        self,
        *,
        workflow_id: str,
        round_number: Optional[int],
        dispatch_run_id: Optional[str],
        dispatch_context: DispatchContext,
        expected: int,
        timeout_seconds: int,
        poll_seconds: int,
        threshold: float,
        mailbox: str,
        email_fetcher: Callable[..., List[EmailResponse]],
        last_dispatch_time: Optional[datetime],
    ) -> Dict[str, object]:
        responses_received = 0
        matched_dispatches: Set[str] = set()
        seen_messages: Set[str] = set()
        stored_rows: List[Dict[str, object]] = []
        unmatched_messages: List[str] = []
        timeout = False

        cursor_time = last_dispatch_time or (self._now() - timedelta(minutes=5))
        if cursor_time.tzinfo is None:
            cursor_time = cursor_time.replace(tzinfo=timezone.utc)
        since_cursor = cursor_time

        deadline = self._now() + timedelta(seconds=max(timeout_seconds, poll_seconds))

        while responses_received < expected and self._now() < deadline:
            try:
                batch = email_fetcher(since=since_cursor)
            except Exception:
                logger.exception("EmailWatcherAgent email fetch failed")
                batch = []
            if batch:
                for email in batch:
                    message_id = email.message_id or ""
                    if message_id and message_id in seen_messages:
                        continue
                    if message_id:
                        seen_messages.add(message_id)
                    match = self._match_email(dispatch_context, email)
                    accepted = match.accepted(threshold)
                    record = self._to_supplier_record(
                        workflow_id=workflow_id,
                        dispatch_run_id=dispatch_run_id,
                        mailbox=mailbox,
                        match=match,
                        email=email,
                        accepted=accepted,
                    )
                    DatabaseBackend().upsert_response(record)
                    if accepted:
                        dispatch_key = (
                            match.dispatch.message_id
                            or match.dispatch.normalised_token()
                            or match.dispatch.rfq_id
                        )
                        if dispatch_key and dispatch_key not in matched_dispatches:
                            matched_dispatches.add(dispatch_key)
                            responses_received += 1
                        stored_rows.append(
                            {
                                "unique_id": dispatch_key,
                                "message_id": email.message_id,
                                "score": match.score,
                                "matched_on": list(match.reasons),
                            }
                        )
                    else:
                        unmatched_messages.append(email.message_id or "")
                    if email.received_at and email.received_at > since_cursor:
                        since_cursor = email.received_at
                logger.info(
                    "EmailWatcherAgent progress workflow=%s round=%s received=%d expected=%d",
                    workflow_id,
                    round_number,
                    responses_received,
                    expected,
                )
            if responses_received >= expected:
                break
            if hasattr(email_fetcher, "wait") and callable(getattr(email_fetcher, "wait")):
                try:
                    email_fetcher.wait()  # type: ignore[attr-defined]
                    continue
                except Exception:
                    logger.debug("EmailWatcherAgent fetcher wait failed", exc_info=True)
            self._sleep(poll_seconds)

        if responses_received < expected and self._now() >= deadline:
            timeout = True

        if responses_received >= expected:
            self._signal_completion(workflow_id, round_number, expected, responses_received)

        return {
            "workflow_id": workflow_id,
            "round": round_number,
            "dispatch_run_id": dispatch_run_id,
            "expected_responses": expected,
            "responses_received": responses_received,
            "timeout": timeout,
            "supplier_responses": stored_rows,
            "unmatched_messages": [msg for msg in unmatched_messages if msg],
        }

    # ------------------------------------------------------------------
    def _match_email(
        self,
        dispatch_context: DispatchContext,
        email: EmailResponse,
    ) -> MatchResult:
        reasons: List[str] = []
        best_score = 0.0
        best_dispatch: Optional[DispatchRecord] = None

        comment, _ = split_hidden_marker(email.body or "")
        marker_token = extract_marker_token(comment)
        marker_supplier = extract_supplier_id(comment)
        marker_rfq = extract_rfq_id(comment)
        marker_run = extract_run_id(comment)

        tokens = {
            (_normalise_identifier(email.unique_id) or "").lower(),
            (_normalise_identifier(marker_token) or "").lower(),
            (_normalise_identifier(marker_run) or "").lower(),
        }
        tokens.discard("")

        supplier_candidates = {
            (_normalise_identifier(email.supplier_id) or "").lower(),
            (_normalise_identifier(marker_supplier) or "").lower(),
        }
        supplier_candidates.discard("")

        rfq_candidates = {
            _normalise_identifier(email.rfq_id),
            _normalise_identifier(marker_rfq),
        }
        rfq_candidates.discard(None)

        workflow_candidates = {
            _normalise_identifier(email.workflow_id),
        }
        workflow_candidates.discard(None)

        thread_ids: Set[str] = set()
        for header in list(email.in_reply_to) + list(email.references):
            normalised = _normalise_identifier(header)
            if normalised:
                thread_ids.add(normalised)

        subject_norm = _normalise_subject(email.subject)

        TOKEN_WEIGHT = 0.55
        THREAD_WEIGHT = 0.35
        SUPPLIER_WEIGHT = 0.25
        WORKFLOW_WEIGHT = 0.1
        RFQ_WEIGHT = 0.25
        SUBJECT_WEIGHT = 0.1

        for record in dispatch_context.records:
            score = 0.0
            matched_reasons: List[str] = []
            token_norm = record.normalised_token()
            if token_norm and token_norm in tokens:
                score += TOKEN_WEIGHT
                matched_reasons.append("token")
            if thread_ids and record.message_id:
                message_norm = _normalise_identifier(record.message_id)
                if message_norm and message_norm in thread_ids:
                    score += THREAD_WEIGHT
                    matched_reasons.append("thread")
            supplier_norm = record.normalised_supplier()
            if supplier_norm and supplier_norm in supplier_candidates:
                score += SUPPLIER_WEIGHT
                matched_reasons.append("supplier")
            if workflow_candidates and record.workflow_id:
                record_workflow = _normalise_identifier(record.workflow_id)
                if record_workflow and record_workflow in workflow_candidates:
                    score += WORKFLOW_WEIGHT
                    matched_reasons.append("workflow")
            if rfq_candidates and record.rfq_id:
                record_rfq = _normalise_identifier(record.rfq_id)
                if record_rfq and record_rfq in rfq_candidates:
                    score += RFQ_WEIGHT
                    matched_reasons.append("rfq")
            if subject_norm and record.rfq_id:
                record_subject = _normalise_subject(record.rfq_id)
                if record_subject and record_subject in subject_norm:
                    score += SUBJECT_WEIGHT
                    matched_reasons.append("subject")
            if len(matched_reasons) >= 2:
                score += 0.05
            if score > best_score:
                best_score = score
                best_dispatch = record
                reasons = matched_reasons

        return MatchResult(
            dispatch=best_dispatch,
            score=round(best_score, 3),
            reasons=list(dict.fromkeys(reasons)),
            supplier_id=_pick_first(supplier_candidates) or None,
            rfq_id=_pick_first(rfq_candidates),
            workflow_id=_pick_first(workflow_candidates),
        )

    def _to_supplier_record(
        self,
        *,
        workflow_id: str,
        dispatch_run_id: Optional[str],
        mailbox: str,
        match: MatchResult,
        email: EmailResponse,
        accepted: bool,
    ) -> SupplierResponseRecord:
        dispatch = match.dispatch if accepted else None
        rfq_id = match.rfq_id or (dispatch.rfq_id if dispatch else (email.rfq_id or ""))
        supplier_id = match.supplier_id or (
            dispatch.supplier_id if dispatch else (email.supplier_id or None)
        )
        subject = email.subject or (getattr(dispatch, "subject", "") if dispatch else "")
        body_text = email.body_text or email.body or ""
        headers = {
            key: ", ".join(values)
            for key, values in (email.headers or {}).items()
            if values
        }
        if email.message_id and "Message-ID" not in headers:
            headers["Message-ID"] = email.message_id

        return SupplierResponseRecord(
            workflow_id=workflow_id,
            action_id=str(dispatch.action_id) if dispatch and dispatch.action_id else None,
            run_id=dispatch_run_id or (dispatch.token if dispatch else None),
            rfq_id=rfq_id or "",
            supplier_id=supplier_id,
            message_id=email.message_id,
            subject=subject or "",
            body=body_text,
            from_address=email.from_address,
            received_at=email.received_at or self._now(),
            headers=headers,
            mailbox=mailbox,
            match_score=match.score,
            match_method="accepted" if accepted else "needs_review",
            matched_on=match.reasons,
            match_confidence=match.score if accepted else 0.0,
        )

    # ------------------------------------------------------------------
    def _wait_for_dispatch_barrier(
        self,
        *,
        workflow_id: str,
        round_number: Optional[int],
        expected: int,
        wait_seconds: int,
    ) -> None:
        if wait_seconds <= 0:
            return
        event = threading.Event()
        bus = None
        try:
            bus = get_event_bus()
        except Exception:
            bus = None
        start = self._now()
        logger.info(
            "EmailWatcherAgent waiting for dispatch completion workflow=%s round=%s expected=%s",
            workflow_id,
            round_number,
            expected,
        )
        if bus is None:
            self._sleep(wait_seconds)
            return

        specific_event = f"round_dispatch_complete:{workflow_id}:{round_number}"

        def _handler(payload: Dict[str, object]) -> None:
            match_workflow = str(payload.get("workflow_id")) if payload else None
            match_round = payload.get("round")
            if match_workflow and _normalise_identifier(match_workflow) != _normalise_identifier(workflow_id):
                return
            if round_number is not None and match_round not in (round_number, str(round_number)):
                return
            event.set()

        bus.subscribe("round_dispatch_complete", _handler, once=True)
        bus.subscribe(specific_event, _handler, once=True)
        event.wait(timeout=wait_seconds)
        if not event.is_set():
            self._sleep(wait_seconds)
        elapsed = (self._now() - start).total_seconds()
        logger.info(
            "EmailWatcherAgent dispatch barrier cleared workflow=%s round=%s waited=%.1fs",
            workflow_id,
            round_number,
            elapsed,
        )

    def _signal_completion(
        self,
        workflow_id: str,
        round_number: Optional[int],
        expected: int,
        received: int,
    ) -> None:
        payload = {
            "workflow_id": workflow_id,
            "round": round_number,
            "expected_responses": expected,
            "responses_received": received,
        }
        try:
            bus = get_event_bus()
        except Exception:
            bus = None
        if bus:
            bus.publish("responses_complete", payload)
            bus.publish(f"responses_complete:{workflow_id}:{round_number}", payload)
        logger.info(
            "EmailWatcherAgent responses complete workflow=%s round=%s received=%d expected=%d",
            workflow_id,
            round_number,
            received,
            expected,
        )


def _pick_first(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


__all__ = ["EmailWatcherAgent", "ImapIdleFetcher"]
