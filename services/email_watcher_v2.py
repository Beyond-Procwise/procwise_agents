"""Compatibility layer exposing the legacy EmailWatcherV2 interface."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from services.email_watcher import (
    ContinuousEmailWatcher,
    EmailDispatchRecord,
    EmailResponse,
    EmailWatcher,
    EmailWatcherConfig,
    ImapEmailFetcher,
    WorkflowTracker,
    embed_unique_id_in_email_body,
    generate_unique_email_id,
    get_supplier_responses,
    mark_interaction_processed,
    parse_email_bytes,
    register_sent_email,
    wait_for_responses,
)


def _parse_email(raw: bytes) -> EmailResponse:
    """Backwards-compatible helper for tests importing the private parser."""

    return parse_email_bytes(raw)


<<<<<<< HEAD
class EmailWatcherV2(EmailWatcher):
    """Thin wrapper around :class:`services.email_watcher.EmailWatcher`."""
=======
def _normalise_thread_header(value) -> Sequence[str]:
    if value in (None, ""):
        return tuple()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v).strip("<> ") for v in value if v)
    return (str(value).strip("<> "),)


def _normalise_identifier(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text.upper() or None


def _normalise_email_address(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    _, address = parseaddr(str(value))
    address = address.strip().lower()
    return address or None


def _default_fetcher(
    *,
    host: str,
    username: str,
    password: str,
    mailbox: str,
    since: datetime,
    port: int = 993,
    use_ssl: bool = True,
    login: Optional[str] = None,
) -> List[EmailResponse]:
    client = _imap_client(host, username, password, port=port, use_ssl=use_ssl, login=login)
    try:
        client.select(mailbox, readonly=True)
        since_str = since.strftime("%d-%b-%Y")
        typ, data = client.search(None, f'(SINCE {since_str})')
        if typ != "OK":
            return []
        ids = (data[0] or b"").decode().split()
        responses: List[EmailResponse] = []
        for message_id in ids:
            typ, payload = client.fetch(message_id, "(RFC822)")
            if typ != "OK" or not payload or not isinstance(payload[0], tuple):
                continue
            raw = payload[0][1]
            responses.append(_parse_email(raw))
        return responses
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            client.logout()
        except Exception:
            pass


def _score_to_confidence(score: float) -> Decimal:
    """Convert a floating point score into a quantised confidence value.

    The score produced by the matcher is normalised into the range 0-1 so it can
    be persisted as a numeric confidence value for downstream processing.  Any
    unexpected inputs are treated defensively and clamped into range before
    quantisation to two decimal places.
    """

    try:
        if score != score:  # NaN check
            score = 0.0
    except Exception:  # pragma: no cover - defensive
        score = 0.0

    normalised = max(0.0, min(float(score), 1.0))
    decimal_score = Decimal(str(normalised))
    return decimal_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _calculate_match_score(dispatch: EmailDispatchRecord, email_response: EmailResponse) -> float:
    score = 0.0

    if email_response.unique_id and email_response.unique_id == dispatch.unique_id:
        return 1.0

    if (
        email_response.supplier_id
        and dispatch.supplier_id
        and email_response.supplier_id == dispatch.supplier_id
    ):
        score += 0.65

    thread_ids = set(dispatch.thread_headers.get("references", ())) | set(
        dispatch.thread_headers.get("in_reply_to", ())
    )
    if dispatch.message_id:
        thread_ids.add(dispatch.message_id)

    reply_headers = set(email_response.in_reply_to) | set(email_response.references)
    if dispatch.message_id and dispatch.message_id in reply_headers:
        score += 0.8
    elif thread_ids & reply_headers:
        score += 0.8

    if dispatch.supplier_email and email_response.from_address:
        dispatch_email = email.utils.parseaddr(str(dispatch.supplier_email))[1].lower()
        response_email = email.utils.parseaddr(str(email_response.from_address))[1].lower()
        if dispatch_email and response_email:
            if dispatch_email == response_email:
                score += 0.6
            else:
                dispatch_domain = dispatch_email.split("@")[-1]
                response_domain = response_email.split("@")[-1]
                if dispatch_domain and response_domain and dispatch_domain == response_domain:
                    score += 0.35

    if dispatch.subject and email_response.subject:
        normalised_subject = dispatch.subject.lower()
        if normalised_subject in email_response.subject.lower():
            score += 0.5

    # Add thread matching logic
    if dispatch.message_id and email_response.in_reply_to:
        if dispatch.message_id in email_response.in_reply_to:
            score += 0.9

    # Add reference chain matching
    common_refs = set(dispatch.thread_headers.get("references", [])) & set(email_response.references)
    if common_refs:
        score += 0.7

    # Add subject line matching with RE:/FW: handling
    if dispatch.subject and email_response.subject:
        clean_dispatch = re.sub(r"^(RE|FW):\s*", "", dispatch.subject, flags=re.IGNORECASE)
        clean_response = re.sub(r"^(RE|FW):\s*", "", email_response.subject, flags=re.IGNORECASE)
        if clean_dispatch.lower() == clean_response.lower():
            score += 0.6

    return score


class EmailWatcherV2:
    """Workflow-aware watcher coordinating dispatch/response tracking."""
>>>>>>> f6b29da (updated changes)

    def __init__(
        self,
        *,
        supplier_agent: Optional[SupplierInteractionAgent] = None,
        negotiation_agent: Optional[NegotiationAgent] = None,
        dispatch_wait_seconds: int = 90,
        poll_interval_seconds: int = 30,
        max_poll_attempts: int = 10,
        match_threshold: float = 0.8,
        email_fetcher: Optional[Callable[..., List[EmailResponse]]] = None,
        mailbox: Optional[str] = None,
        imap_host: Optional[str] = None,
        imap_username: Optional[str] = None,
        imap_password: Optional[str] = None,
        imap_port: Optional[int] = None,
        imap_use_ssl: Optional[bool] = None,
        imap_login: Optional[str] = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
<<<<<<< HEAD
        lookback_minutes: int = 240,
        max_fetch: Optional[int] = None,
        poll_backoff_factor: Optional[float] = None,
        poll_jitter_seconds: Optional[float] = None,
        poll_max_interval_seconds: Optional[int] = None,
        poll_timeout_seconds: Optional[int] = None,
    ) -> None:
        config = EmailWatcherConfig(
            imap_host=imap_host,
            imap_username=imap_username,
            imap_password=imap_password,
            imap_port=imap_port or 993,
            imap_use_ssl=True if imap_use_ssl is None else imap_use_ssl,
            imap_login=imap_login,
            imap_mailbox=mailbox or "INBOX",
            dispatch_wait_seconds=max(0, dispatch_wait_seconds),
            poll_interval_seconds=max(1, poll_interval_seconds),
            max_poll_attempts=max(1, max_poll_attempts),
            match_threshold=match_threshold,
            lookback_minutes=max(1, lookback_minutes),
            max_fetch=max_fetch,
            poll_backoff_factor=poll_backoff_factor if poll_backoff_factor is not None else 1.8,
            poll_jitter_seconds=poll_jitter_seconds if poll_jitter_seconds is not None else 2.0,
            poll_max_interval_seconds=(
                poll_max_interval_seconds
                if poll_max_interval_seconds is not None
                else max(1, int((poll_interval_seconds or 1) * 6))
            ),
            poll_timeout_seconds=poll_timeout_seconds,
        )
        super().__init__(
            config=config,
            supplier_agent=supplier_agent,
            negotiation_agent=negotiation_agent,
            fetcher=email_fetcher,
            sleep=sleep,
            now=now,
        )

        # legacy attribute compatibility
        self.dispatch_wait_seconds = config.dispatch_wait_seconds
        self.poll_interval_seconds = config.poll_interval_seconds
        self.max_poll_attempts = config.max_poll_attempts
        self.match_threshold = config.match_threshold
        self._mailbox = config.imap_mailbox
        self._imap_host = config.imap_host
        self._imap_username = config.imap_username
        self._imap_password = config.imap_password
        self._imap_port = config.imap_port
        self._imap_use_ssl = config.imap_use_ssl
        self._imap_login = config.imap_login
        self._fetcher = email_fetcher
        self.poll_backoff_factor = config.poll_backoff_factor
        self.poll_jitter_seconds = config.poll_jitter_seconds
        self.poll_max_interval_seconds = config.poll_max_interval_seconds
        self.poll_timeout_seconds = config.poll_timeout_seconds
=======
        workflow_memory: Optional[Any] = None,
    ) -> None:
        self.supplier_agent = supplier_agent
        self.negotiation_agent = negotiation_agent
        self.dispatch_wait_seconds = max(0, dispatch_wait_seconds)
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self.max_poll_attempts = max(1, max_poll_attempts)
        self.match_threshold = match_threshold
        self._fetcher = email_fetcher
        self._mailbox = mailbox or "INBOX"
        self._imap_host = imap_host
        self._imap_username = imap_username
        self._imap_password = imap_password
        self._imap_port = imap_port
        self._imap_use_ssl = imap_use_ssl
        self._imap_login = imap_login
        self._sleep = sleep
        self._now = now
        self._trackers: Dict[str, WorkflowTracker] = {}
        self.workflow_memory = workflow_memory
        self._thread_states: Dict[str, EmailThreadState] = {}

        tracking_repo.init_schema()
        supplier_response_repo.init_schema()

    def _record_response_memory(
        self,
        workflow_id: Optional[str],
        unique_id: Optional[str],
        response_row: SupplierResponseRow,
    ) -> None:
        if not workflow_id or not unique_id:
            return
        if not self.workflow_memory or not getattr(self.workflow_memory, "enabled", False):
            return
        body_text = response_row.response_text or ""
        if body_text and len(body_text) > 4000:
            body_text = body_text[:4000] + "..."
        metadata = {
            "supplier_id": response_row.supplier_id,
            "from": response_row.response_from,
            "subject": response_row.response_subject,
            "message_id": response_row.response_message_id,
        }
        payload = {
            "role": "supplier",
            "subject": response_row.response_subject,
            "body_html": body_text,
            "body_text": body_text,
            "metadata": metadata,
        }
        try:
            self.workflow_memory.record_email_message(workflow_id, unique_id, payload)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to persist supplier response to workflow memory (wf=%s uid=%s)",
                workflow_id,
                unique_id,
                exc_info=True,
            )

    def _ensure_tracker(self, workflow_id: str) -> WorkflowTracker:
        tracker = self._trackers.get(workflow_id)
        if tracker is not None:
            return tracker
        tracker = WorkflowTracker(workflow_id=workflow_id)
        rows = tracking_repo.load_workflow_rows(workflow_id=workflow_id)
        if rows:
            dispatches = [
                EmailDispatchRecord(
                    unique_id=row.unique_id,
                    supplier_id=row.supplier_id,
                    supplier_email=row.supplier_email,
                    message_id=row.message_id,
                    subject=row.subject,
                    thread_headers=row.thread_headers or {},
                    dispatched_at=row.dispatched_at,
                )
                for row in rows
            ]
            tracker.register_dispatches(dispatches)
            matched = [row for row in rows if row.matched]
            for row in matched:
                tracker.record_response(
                    row.unique_id,
                    EmailResponse(
                        unique_id=row.unique_id,
                        supplier_id=row.supplier_id,
                        supplier_email=row.supplier_email,
                        from_address=None,
                        message_id=row.response_message_id,
                        subject=None,
                        body="",
                        received_at=row.responded_at or row.dispatched_at,
                    ),
                )
        self._trackers[workflow_id] = tracker
        return tracker

    def register_workflow_dispatch(
        self,
        workflow_id: str,
        dispatches: Sequence[Dict[str, object]],
    ) -> WorkflowTracker:
        if not workflow_id:
            raise ValueError("workflow_id is required to register dispatches")

        tracker = self._ensure_tracker(workflow_id)
        records: List[EmailDispatchRecord] = []
        repo_rows: List[WorkflowDispatchRow] = []

        for payload in dispatches:
            unique_id = str(payload.get("unique_id") or uuid.uuid4().hex)
            supplier_id = payload.get("supplier_id")
            supplier_email = payload.get("supplier_email")
            message_id = payload.get("message_id")
            subject = payload.get("subject")
            dispatched_at = payload.get("dispatched_at")
            rfq_id = payload.get("rfq_id")
            dispatch_key = payload.get("dispatch_key")
            raw_thread_headers = (
                payload.get("thread_headers") if isinstance(payload.get("thread_headers"), dict) else {}
            )
            if isinstance(dispatched_at, datetime):
                dispatched_dt = dispatched_at if dispatched_at.tzinfo else dispatched_at.replace(tzinfo=timezone.utc)
            else:
                dispatched_dt = self._now()

            record = EmailDispatchRecord(
                unique_id=unique_id,
                supplier_id=str(supplier_id) if supplier_id else None,
                supplier_email=str(supplier_email) if supplier_email else None,
                message_id=str(message_id) if message_id else None,
                subject=str(subject) if subject else None,
                rfq_id=str(rfq_id) if rfq_id else None,
                thread_headers={
                    str(k): _normalise_thread_header(v) for k, v in raw_thread_headers.items()
                },
                dispatched_at=dispatched_dt,
            )
            records.append(record)
            repo_rows.append(
                WorkflowDispatchRow(
                    workflow_id=workflow_id,
                    unique_id=unique_id,
                    dispatch_key=str(dispatch_key).strip() if dispatch_key else None,
                    supplier_id=record.supplier_id,
                    supplier_email=record.supplier_email,
                    message_id=record.message_id,
                    subject=record.subject,
                    dispatched_at=dispatched_dt,
                    responded_at=None,
                    response_message_id=None,
                    matched=False,
                    thread_headers={
                        key: list(value) for key, value in record.thread_headers.items()
                    }
                    if record.thread_headers
                    else None,
                )
            )

        tracker.register_dispatches(records)
        tracking_repo.record_dispatches(workflow_id=workflow_id, dispatches=repo_rows)
        return tracker

    def _fetch_emails(self, since: datetime) -> List[EmailResponse]:
        if self._fetcher:
            return self._fetcher(since=since)

        if not all([self._imap_host, self._imap_username, self._imap_password]):
            raise RuntimeError("IMAP credentials must be supplied when no custom fetcher is provided")

        return _default_fetcher(
            host=self._imap_host,
            username=self._imap_username,
            password=self._imap_password,
            mailbox=self._mailbox,
            since=since,
            port=self._imap_port or 993,
            use_ssl=True if self._imap_use_ssl is None else self._imap_use_ssl,
            login=self._imap_login,
        )

    def _match_responses(
        self, tracker: WorkflowTracker, responses: Iterable[EmailResponse]
    ) -> List[SupplierResponseRow]:
        matched_rows: List[SupplierResponseRow] = []
        for email in responses:
            matched_id: Optional[str] = None
            best_score = 0.0
            best_dispatch: Optional[EmailDispatchRecord] = None
            for unique_id, dispatch in tracker.email_records.items():
                if unique_id in tracker.matched_responses:
                    continue
                score = _calculate_match_score(dispatch, email)
                if score > best_score:
                    matched_id = unique_id
                    best_score = score
                    best_dispatch = dispatch
            if (not matched_id or best_score < self.match_threshold) and email.supplier_id:
                supplier_matches = [
                    uid
                    for uid, dispatch in tracker.email_records.items()
                    if uid not in tracker.matched_responses
                    and dispatch.supplier_id
                    and dispatch.supplier_id == email.supplier_id
                ]
                if len(supplier_matches) == 1:
                    matched_id = supplier_matches[0]
                    best_score = max(best_score, self.match_threshold)
                    best_dispatch = tracker.email_records.get(matched_id)
            if (not matched_id or best_score < self.match_threshold) and email.rfq_id:
                normalised_rfq = _normalise_identifier(email.rfq_id)
                if normalised_rfq:
                    rfq_candidates = [
                        uid
                        for uid, dispatch in tracker.email_records.items()
                        if uid not in tracker.matched_responses
                        and dispatch.rfq_id
                        and _normalise_identifier(dispatch.rfq_id) == normalised_rfq
                    ]
                    if len(rfq_candidates) == 1:
                        matched_id = rfq_candidates[0]
                        best_score = max(best_score, self.match_threshold)
                        best_dispatch = tracker.email_records.get(matched_id)

            if not matched_id and len([
                uid for uid in tracker.email_records.keys() if uid not in tracker.matched_responses
            ]) == 1:
                remaining_uid = next(
                    uid
                    for uid in tracker.email_records.keys()
                    if uid not in tracker.matched_responses
                )
                matched_id = remaining_uid
                logger.debug(
                    "Defaulting response assignment for workflow=%s to remaining unique_id=%s",
                    tracker.workflow_id,
                    matched_id,
                )
                best_score = max(best_score, self.match_threshold)
                best_dispatch = tracker.email_records.get(matched_id)

            if matched_id and best_score >= self.match_threshold:
                logger.debug(
                    "Matched response for workflow=%s unique_id=%s score=%.2f",
                    tracker.workflow_id,
                    matched_id,
                    best_score,
                )
                if best_dispatch is None:
                    best_dispatch = tracker.email_records.get(matched_id)
                if best_dispatch is None:
                    logger.warning(
                        "Matched response for workflow=%s unique_id=%s but no dispatch record found",
                        tracker.workflow_id,
                        matched_id,
                    )
                    continue
                tracker.record_response(matched_id, email)
                tracking_repo.mark_response(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    responded_at=email.received_at,
                    response_message_id=email.message_id,
                )
                supplier_email = (
                    email.supplier_email
                    or _normalise_email_address(email.from_address)
                    or best_dispatch.supplier_email
                )
                supplier_id = email.supplier_id or best_dispatch.supplier_id
                response_time: Optional[Decimal] = None
                if best_dispatch.dispatched_at and email.received_at:
                    try:
                        delta_seconds = (email.received_at - best_dispatch.dispatched_at).total_seconds()
                        if delta_seconds >= 0:
                            response_time = Decimal(str(delta_seconds))
                    except Exception:  # pragma: no cover - defensive conversion
                        response_time = None
                response_row = SupplierResponseRow(
                    workflow_id=tracker.workflow_id,
                    unique_id=matched_id,
                    supplier_id=supplier_id,
                    supplier_email=supplier_email,
                    response_text=email.body,
                    received_time=email.received_at,
                    response_time=response_time,
                    response_message_id=email.message_id,
                    response_subject=email.subject,
                    response_from=email.from_address,
                    original_message_id=best_dispatch.message_id,
                    original_subject=best_dispatch.subject,
                    match_confidence=_score_to_confidence(best_score),
                    processed=False,
                )
                supplier_response_repo.insert_response(response_row)
                self._record_response_memory(tracker.workflow_id, matched_id, response_row)
                matched_rows.append(response_row)
            else:
                logger.warning(
                    "Unable to confidently match supplier email for workflow=%s message_id=%s (best_score=%.2f) supplier_id=%s rfq_id=%s unique_id=%s",
                    tracker.workflow_id,
                    email.message_id,
                    best_score,
                    email.supplier_id,
                    email.rfq_id,
                    email.unique_id,
                )
        return matched_rows
>>>>>>> f6b29da (updated changes)

    def wait_and_collect_responses(self, workflow_id: str) -> Dict[str, object]:
        return self.wait_for_responses(workflow_id)


<<<<<<< HEAD
__all__ = [
    "EmailWatcherV2",
    "EmailWatcher",
    "EmailWatcherConfig",
    "WorkflowTracker",
    "EmailDispatchRecord",
    "EmailResponse",
    "ImapEmailFetcher",
    "ContinuousEmailWatcher",
    "generate_unique_email_id",
    "embed_unique_id_in_email_body",
    "register_sent_email",
    "get_supplier_responses",
    "mark_interaction_processed",
    "wait_for_responses",
    "_parse_email",
]
=======
def generate_unique_email_id(
    workflow_id: str,
    supplier_id: Optional[str],
    *,
    round_number: int = 1,
) -> str:
    """Generate a unique identifier encoding workflow, supplier, and round."""

    supplier_hint = f"{supplier_id or 'anon'}-r{max(1, round_number)}"
    return email_tracking.generate_unique_email_id(workflow_id, supplier_hint)


def embed_unique_id_in_email_body(body: Optional[str], unique_id: str) -> str:
    """Expose :func:`utils.email_tracking.embed_unique_id_in_email_body`."""

    return email_tracking.embed_unique_id_in_email_body(body, unique_id)


def register_sent_email(
    agent_nick: str,
    workflow_id: str,
    supplier_id: Optional[str],
    supplier_email: Optional[str],
    unique_id: str,
    *,
    round_number: int = 1,
    interaction_type: str = "initial",
    subject: Optional[str] = None,
    body: Optional[str] = None,
    message_id: Optional[str] = None,
    rfq_id: Optional[str] = None,
    thread_headers: Optional[Dict[str, Iterable[str]]] = None,
) -> None:
    """Persist a dispatched email so inbound replies can be reconciled."""

    supplier_interaction_repo.init_schema()

    headers = thread_headers or {}
    record = SupplierInteractionRow(
        workflow_id=workflow_id,
        unique_id=unique_id,
        supplier_id=supplier_id,
        supplier_email=supplier_email,
        round_number=max(1, round_number),
        direction="outbound",
        interaction_type=interaction_type,
        status="pending",
        subject=subject,
        body=body,
        message_id=message_id,
        in_reply_to=list(headers.get("in_reply_to", [])) if headers else None,
        references=list(headers.get("references", [])) if headers else None,
        rfq_id=rfq_id,
        metadata={"agent": agent_nick},
    )
    supplier_interaction_repo.register_outbound(record)


def get_supplier_responses(
    agent_nick: str,
    workflow_id: str,
    *,
    interaction_type: Optional[str] = None,
    round_number: Optional[int] = None,
    status: str = "received",
) -> List[Dict[str, object]]:
    """Return inbound supplier responses filtered by status."""

    supplier_interaction_repo.init_schema()
    rows = supplier_interaction_repo.fetch_by_status(
        workflow_id=workflow_id,
        status=status,
        interaction_type=interaction_type,
        round_number=round_number,
    )
    logger.debug(
        "Loaded %d supplier interaction rows for workflow=%s status=%s agent=%s",
        len(rows),
        workflow_id,
        status,
        agent_nick,
    )
    return rows


def mark_interaction_processed(agent_nick: str, workflow_id: str, unique_id: str) -> None:
    """Mark a supplier response as processed by downstream agents."""

    if not unique_id:
        return
    logger.debug(
        "Marking supplier interaction processed agent=%s workflow=%s unique_id=%s",
        agent_nick,
        workflow_id,
        unique_id,
    )
    supplier_interaction_repo.mark_status(
        unique_ids=[unique_id],
        direction="inbound",
        status="processed",
        processed_at=datetime.now(timezone.utc),
    )


def wait_for_responses(
    workflow_id: str,
    *,
    round_number: Optional[int] = None,
    interaction_type: Optional[str] = None,
    timeout_minutes: int = 60,
    poll_interval_seconds: int = 30,
) -> List[Dict[str, object]]:
    """Block until responses arrive or timeout is reached."""

    deadline = datetime.now(timezone.utc) + timedelta(minutes=max(1, timeout_minutes))
    while datetime.now(timezone.utc) < deadline:
        rows = supplier_interaction_repo.fetch_by_status(
            workflow_id=workflow_id,
            status="received",
            round_number=round_number,
            interaction_type=interaction_type,
        )
        if rows:
            return rows
        time.sleep(max(1, poll_interval_seconds))
    return []


class ContinuousEmailWatcher:
    """Background IMAP poller that persists supplier responses immediately."""

    def __init__(
        self,
        agent_nick: str,
        *,
        poll_interval_seconds: int = 60,
        email_fetcher: Optional[Callable[..., List[EmailResponse]]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.agent_nick = agent_nick
        self.poll_interval_seconds = max(1, poll_interval_seconds)
        self._email_fetcher = email_fetcher
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn

    def _fetch(self, **kwargs) -> List[EmailResponse]:
        if self._email_fetcher is not None:
            return self._email_fetcher(**kwargs)
        return _default_fetcher(**kwargs)

    def _sleep(self) -> None:
        try:
            self._sleep_fn(self.poll_interval_seconds)
        except Exception:  # pragma: no cover - defensive
            time.sleep(self.poll_interval_seconds)

    def _row_to_outbound(self, row: Dict[str, Any]) -> SupplierInteractionRow:
        return SupplierInteractionRow(
            workflow_id=row["workflow_id"],
            unique_id=row["unique_id"],
            supplier_id=row.get("supplier_id"),
            supplier_email=row.get("supplier_email"),
            round_number=row.get("round_number", 1),
            direction="outbound",
            interaction_type=row.get("interaction_type", "initial"),
            status=row.get("status", "pending"),
            subject=row.get("subject"),
            body=row.get("body"),
            message_id=row.get("message_id"),
            in_reply_to=list(row.get("in_reply_to") or []),
            references=list(row.get("references") or row.get("reference_ids") or []),
            rfq_id=row.get("rfq_id"),
            metadata=row.get("metadata"),
        )

    def _resolve_outbound(self, response: EmailResponse) -> Optional[SupplierInteractionRow]:
        row: Optional[Dict[str, Any]] = None
        if response.unique_id:
            row = supplier_interaction_repo.lookup_outbound(response.unique_id)

        if row is None and response.rfq_id:
            rfq_id = response.rfq_id
            candidates = supplier_interaction_repo.find_pending_by_rfq(rfq_id)
            if not candidates and rfq_id.upper() != rfq_id:
                candidates = supplier_interaction_repo.find_pending_by_rfq(rfq_id.upper())
            if len(candidates) == 1:
                row = candidates[0]

        if row is None:
            return None

        try:
            return self._row_to_outbound(row)
        except KeyError:
            logger.warning(
                "Malformed outbound interaction row encountered agent=%s row_keys=%s",
                self.agent_nick,
                sorted(row.keys()),
            )
            return None

    def run_once(self, workflow_id: str) -> List[Dict[str, object]]:
        responses = self._fetch(workflow_id=workflow_id)
        if not responses:
            return []

        supplier_interaction_repo.init_schema()
        stored: List[Dict[str, object]] = []
        for response in responses:
            stored_record = self._handle_response(response)
            if stored_record:
                stored.append(stored_record)
        return stored

    def _handle_response(self, response: EmailResponse) -> Optional[Dict[str, object]]:
        outbound = self._resolve_outbound(response)
        if outbound is None:
            logger.warning(
                "Could not match supplier response to dispatch agent=%s unique_id=%s rfq_id=%s",
                self.agent_nick,
                response.unique_id,
                response.rfq_id,
            )
            return None

        supplier_interaction_repo.record_inbound_response(
            outbound=outbound,
            message_id=response.message_id or uuid.uuid4().hex,
            subject=response.subject,
            body=response.body,
            from_address=response.from_address,
            received_at=response.received_at,
            in_reply_to=response.in_reply_to,
            references=response.references,
            rfq_id=response.rfq_id,
            metadata={"agent": self.agent_nick},
        )

        logger.info(
            "Recorded supplier response workflow=%s unique_id=%s agent=%s",
            outbound.workflow_id,
            outbound.unique_id,
            self.agent_nick,
        )

        return {
            "unique_id": outbound.unique_id,
            "supplier_id": outbound.supplier_id,
            "workflow_id": outbound.workflow_id,
            "status": "received",
            "subject": response.subject,
        }

    def run_continuously(
        self,
        workflow_id: str,
        *,
        until: Optional[datetime] = None,
        timeout_minutes: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        collected: List[Dict[str, object]] = []
        start = self._now_fn()
        while True:
            collected.extend(self.run_once(workflow_id))
            if collected:
                return collected
            now = self._now_fn()
            if until and now >= until:
                return collected
            if timeout_minutes and (now - start) >= timedelta(minutes=timeout_minutes):
                return collected
            self._sleep()


@dataclass
class EmailThreadState:
    workflow_id: str
    thread_id: str
    subject: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_message_id: Optional[str] = None

    def add_message(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
        self.last_message_id = message.get("message_id")
        if not self.subject and message.get("subject"):
            self.subject = message["subject"]


def _ensure_re_prefix(subject: str) -> str:
    if not subject:
        return "RE: No Subject"
    normalized = subject.strip().upper()
    if normalized.startswith("RE:"):
        return subject
    return f"RE: {subject}"


def _get_thread_state(self, workflow_id: str, thread_id: str) -> EmailThreadState:
    key = f"{workflow_id}:{thread_id}"
    if key not in self._thread_states:
        self._thread_states[key] = EmailThreadState(
            workflow_id=workflow_id,
            thread_id=thread_id,
            subject=""
        )
    return self._thread_states[key]


def _prepare_email_body(self, new_content: str, thread_state: EmailThreadState) -> str:
    parts = [new_content]
    for message in reversed(thread_state.messages):
        parts.append("\n" + "-" * 60 + "\n")
        parts.append(f"From: {message.get('from', 'Unknown')}")
        parts.append(f"Sent: {message.get('timestamp', 'Unknown')}")
        parts.append(f"Subject: {message.get('subject', 'No Subject')}\n")
        parts.append(message.get('body', ''))
    return "\n".join(parts)
>>>>>>> f6b29da (updated changes)
