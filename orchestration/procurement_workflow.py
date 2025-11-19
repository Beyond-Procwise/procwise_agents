"""Lightweight procurement workflow orchestration primitives.

This module provides a deliberately small, dependency free implementation of
the multi-agent procurement orchestration pipeline used in documentation and
unit tests.  The production system exposes significantly more functionality,
however the objects here focus on the coordination mechanics that are required
by the tests in ``tests/test_procurement_workflow.py``.

The design mimics the real stack:

* ``EmailDraftingAgent`` generates per-supplier drafts and persists them with a
  ``workflow_id``/``unique_id`` tuple.  The drafts are stored with
  ``sent_status=False`` and the body contains a hidden tracking marker so that
  replies can be reconciled to the correct thread.
* ``SupplierInteractionAgent`` blocks on two gates.  The dispatch gate ensures
  that all expected drafts have been sent while the response gate waits until
  the reply count matches the dispatched unique identifiers.
* ``NegotiationAgent`` accepts batches of supplier responses and uses a
  ``ThreadPoolExecutor`` to draft counter offers in parallel.  Each generated
  draft re-uses the existing ``unique_id`` so downstream components can follow
  the same email thread.
* ``QuoteComparisonAgent`` performs a final consolidation step once the
  negotiation rounds have concluded.

The implementation purposefully avoids any network or database dependencies –
everything is stored in-memory – but the API mirrors the behaviour of the
production services so the orchestration logic can be validated in isolation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.email_markers import attach_hidden_marker
from utils.email_tracking import generate_unique_email_id
from services.negotiation_email_templates import NegotiationEmailTemplateRenderer

logger = logging.getLogger(__name__)


STATUS_REQUEST_SEPARATOR = "::round::"


def build_status_request_id(session_id: str, round_num: int) -> str:
    """Create deterministic request identifiers per round for status tracking."""

    return f"{session_id}{STATUS_REQUEST_SEPARATOR}{round_num}"


@dataclass
class EmailThread:
    """Represents a single supplier communication thread across the workflow."""

    workflow_id: str
    supplier_id: str
    supplier_email: Optional[str] = None
    unique_id: str = ""
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    supplier_unique_id: str = field(
        default_factory=lambda: f"SUP-{uuid.uuid4().hex[:10].upper()}"
    )
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_round: int = 0

    def __post_init__(self) -> None:
        if not self.unique_id:
            self.unique_id = generate_unique_email_id(self.workflow_id, self.supplier_id)

    def add_message(
        self,
        message_type: str,
        content: str,
        action_id: str,
        *,
        round_number: Optional[int] = None,
        headers: Optional[Dict[str, Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        """Attach a message to the thread without mutating identifiers."""

        effective_round = self.current_round if round_number is None else max(round_number, 0)
        self.current_round = max(self.current_round, effective_round)

        entry: Dict[str, Any] = {
            "thread_id": self.thread_id,
            "workflow_id": self.workflow_id,
            "supplier_id": self.supplier_id,
            "supplier_unique_id": self.supplier_unique_id,
            "unique_id": self.unique_id,
            "message_type": message_type,
            "content": content,
            "action_id": action_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": effective_round,
        }
        if headers:
            entry["headers"] = dict(headers)

        self.messages.append(entry)
        return entry

    def get_full_thread(self) -> str:
        """Format the thread for inclusion in downstream prompts."""

        header = (
            f"Thread ID: {self.thread_id}\n"
            f"Workflow ID: {self.workflow_id}\n"
            f"Supplier ID: {self.supplier_id}\n"
            f"Supplier Unique ID: {self.supplier_unique_id}\n"
            f"Unique ID: {self.unique_id}\n"
        )
        body_segments = []
        for message in self.messages:
            body_segments.append(
                f"[Round {message['round']}] {message['message_type']}\n{message['content']}"
            )
        return header + "\n".join(body_segments)


@dataclass
class DraftEmail:
    workflow_id: str
    unique_id: str
    supplier_id: str
    supplier_email: Optional[str]
    subject: str
    body: str
    round_number: int
    action_id: str
    sent_status: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dispatched_at: Optional[datetime] = None
    message_id: Optional[str] = None
    thread_headers: Dict[str, Sequence[str]] = field(default_factory=dict)


@dataclass
class DispatchRecord:
    workflow_id: str
    unique_id: str
    supplier_id: str
    message_id: Optional[str]
    dispatched_at: Optional[datetime]
    responded_at: Optional[datetime] = None
    thread_headers: Dict[str, Sequence[str]] = field(default_factory=dict)


@dataclass
class SupplierResponse:
    workflow_id: str
    unique_id: str
    supplier_id: str
    round_number: int
    payload: Dict[str, Any]
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RoundTrackingState:
    request_id: str
    workflow_id: str
    round_number: int
    expected_unique_ids: Set[str]
    expected_count: int
    responses: Dict[str, SupplierResponse] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def responses_received(self) -> int:
        return len(self.responses)

    @property
    def is_complete(self) -> bool:
        if self.expected_unique_ids:
            return self.expected_unique_ids.issubset(set(self.responses))
        return self.responses_received >= self.expected_count > 0

    def register_response(self, response: SupplierResponse) -> None:
        self.responses[response.unique_id] = response
        self.last_updated = datetime.now(timezone.utc)
        if self.is_complete and self.completed_at is None:
            self.completed_at = self.last_updated


@dataclass
class NegotiationSession:
    session_id: str
    workflow_id: str
    start_time: str
    current_round: int = 0
    supplier_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ProcurementStateStore:
    """In-memory backing store that mimics the shape of the production tables."""

    def __init__(self) -> None:
        self.drafts: Dict[str, DraftEmail] = {}
        self.workflow_index: Dict[str, Set[str]] = {}
        self.dispatches: Dict[str, DispatchRecord] = {}
        self.rounds: Dict[str, RoundTrackingState] = {}
        self.responses: List[SupplierResponse] = []

    # ------------------------------------------------------------------
    # Draft management
    # ------------------------------------------------------------------
    def record_draft(self, draft: DraftEmail) -> None:
        self.drafts[draft.unique_id] = draft
        self.workflow_index.setdefault(draft.workflow_id, set()).add(draft.unique_id)

    def mark_dispatched(
        self,
        *,
        workflow_id: str,
        unique_id: str,
        message_id: str,
        thread_headers: Optional[Dict[str, Sequence[str]]] = None,
        dispatched_at: Optional[datetime] = None,
    ) -> None:
        draft = self.drafts.get(unique_id)
        if draft:
            draft.sent_status = True
            draft.message_id = message_id
            draft.dispatched_at = dispatched_at or datetime.now(timezone.utc)
            if thread_headers:
                draft.thread_headers = dict(thread_headers)

        record = self.dispatches.get(unique_id)
        if record is None:
            record = DispatchRecord(
                workflow_id=workflow_id,
                unique_id=unique_id,
                supplier_id=draft.supplier_id if draft else "",
                message_id=message_id,
                dispatched_at=dispatched_at or datetime.now(timezone.utc),
                thread_headers=dict(thread_headers or {}),
            )
        else:
            record.message_id = message_id
            record.dispatched_at = dispatched_at or datetime.now(timezone.utc)
            if thread_headers:
                record.thread_headers = dict(thread_headers)
        self.dispatches[unique_id] = record

    def dispatch_summary(self, workflow_id: str) -> Tuple[List[DispatchRecord], Set[str]]:
        records = [
            record
            for record in self.dispatches.values()
            if record.workflow_id == workflow_id and record.dispatched_at
        ]
        return records, {record.unique_id for record in records}

    # ------------------------------------------------------------------
    # Round tracking
    # ------------------------------------------------------------------
    def initialise_round(
        self,
        *,
        workflow_id: str,
        round_number: int,
        expected_unique_ids: Iterable[str],
        request_id: str,
        expected_count: int,
    ) -> None:
        state = RoundTrackingState(
            request_id=request_id,
            workflow_id=workflow_id,
            round_number=round_number,
            expected_unique_ids={uid for uid in expected_unique_ids if uid},
            expected_count=max(int(expected_count), 0),
        )
        self.rounds[request_id] = state

    def get_round_state(self, request_id: str) -> Optional[RoundTrackingState]:
        return self.rounds.get(request_id)

    def record_response(self, response: SupplierResponse) -> None:
        self.responses.append(response)
        dispatch = self.dispatches.get(response.unique_id)
        if dispatch:
            dispatch.responded_at = response.received_at

        request_id = build_status_request_id(response.workflow_id, response.round_number)
        state = self.rounds.get(request_id)
        if state is None:
            state = RoundTrackingState(
                request_id=request_id,
                workflow_id=response.workflow_id,
                round_number=response.round_number,
                expected_unique_ids={response.unique_id},
            )
            self.rounds[request_id] = state
        state.register_response(response)

    def list_responses(
        self, *, workflow_id: str, round_number: int
    ) -> List[SupplierResponse]:
        return [
            response
            for response in self.responses
            if response.workflow_id == workflow_id
            and response.round_number == round_number
        ]


class MockDatabaseConnection:
    """Async wrapper around :class:`ProcurementStateStore` used in tests."""

    def __init__(self) -> None:
        self.store = ProcurementStateStore()

    # ------------------------- draft management -------------------------
    async def register_draft(self, draft: DraftEmail) -> None:
        self.store.record_draft(draft)

    async def mark_dispatched(
        self,
        *,
        workflow_id: str,
        unique_id: str,
        message_id: str,
        thread_headers: Optional[Dict[str, Sequence[str]]] = None,
        dispatched_at: Optional[datetime] = None,
    ) -> None:
        self.store.mark_dispatched(
            workflow_id=workflow_id,
            unique_id=unique_id,
            message_id=message_id,
            thread_headers=thread_headers,
            dispatched_at=dispatched_at,
        )

    async def load_dispatch_summary(
        self, workflow_id: str
    ) -> Tuple[List[DispatchRecord], Set[str]]:
        return self.store.dispatch_summary(workflow_id)

    # ------------------------ response tracking -------------------------
    async def initialise_round_status(
        self,
        *,
        workflow_id: str,
        round_number: int,
        expected_unique_ids: Iterable[str],
        request_id: str,
        expected_count: int,
    ) -> None:
        self.store.initialise_round(
            workflow_id=workflow_id,
            round_number=round_number,
            expected_unique_ids=expected_unique_ids,
            request_id=request_id,
            expected_count=expected_count,
        )

    async def record_supplier_response(
        self,
        *,
        workflow_id: str,
        unique_id: str,
        supplier_id: str,
        round_number: int,
        payload: Dict[str, Any],
    ) -> SupplierResponse:
        response = SupplierResponse(
            workflow_id=workflow_id,
            unique_id=unique_id,
            supplier_id=supplier_id,
            round_number=round_number,
            payload=payload,
        )
        self.store.record_response(response)
        return response

    async def get_round_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        state = self.store.get_round_state(request_id)
        if not state:
            return None
        return {
            "request_id": state.request_id,
            "workflow_id": state.workflow_id,
            "round_number": state.round_number,
            "responses_received": state.responses_received,
            "expected_responses": len(state.expected_unique_ids),
            "is_complete": state.is_complete,
            "last_updated": state.last_updated,
            "completed_at": state.completed_at,
            "responses": state.responses,
        }

    async def list_round_responses(
        self, *, workflow_id: str, round_number: int
    ) -> List[SupplierResponse]:
        return self.store.list_responses(workflow_id=workflow_id, round_number=round_number)


class WorkflowContextManager:
    """Maintains small pieces of shared workflow context for prompts."""

    def __init__(self, llm_model: str = "qwen3:30b") -> None:
        self.model = llm_model
        self.workflow_context = {
            "phases": [
                "ranking",
                "initial_email",
                "negotiation_rounds",
                "evaluation",
            ],
            "max_negotiation_rounds": 3,
            "human_checkpoints": [
                "email_review",
                "negotiation_review",
                "final_review",
            ],
        }
        self.embeddings: Dict[str, Any] = {}

    async def initialize_workflow_understanding(self) -> Dict[str, Any]:
        workflow_prompt = json.dumps(self.workflow_context, indent=2)
        self.embeddings["workflow"] = await self._create_embedding(workflow_prompt)
        self.embeddings["negotiation_playbook"] = await self._load_negotiation_playbook()
        return self.embeddings

    async def _create_embedding(self, text: str) -> Dict[str, Any]:
        return {"text": text, "timestamp": datetime.now(timezone.utc).isoformat()}

    async def _load_negotiation_playbook(self) -> Dict[str, Dict[str, str]]:
        return {
            "round_1": {
                "strategy": "Express interest, ask clarifications, gather information",
                "tone": "Inquisitive, professional",
            },
            "round_2": {
                "strategy": "Present counter-offer, negotiate terms, provide justification",
                "tone": "Confident, data-driven",
            },
            "round_3": {
                "strategy": "Best and final offer, set deadline, firm stance",
                "tone": "Decisive, respectful but firm",
            },
        }


class EmailDraftingAgent:
    """Generate procurement RFQ drafts while anchoring unique identifiers."""

    def __init__(
        self,
        db: MockDatabaseConnection,
        context: WorkflowContextManager,
        *,
        template_renderer: Optional[NegotiationEmailTemplateRenderer] = None,
    ) -> None:
        self.db = db
        self.context = context
        self.template_renderer = template_renderer or NegotiationEmailTemplateRenderer.from_default()

    async def generate_initial_email(
        self, workflow_id: str, supplier: Dict[str, Any]
    ) -> DraftEmail:
        supplier_id = supplier.get("id") or supplier.get("supplier_id")
        supplier_email = supplier.get("email")
        unique_id = generate_unique_email_id(workflow_id, supplier_id)

        supplier_payload = dict(supplier)
        supplier_payload.setdefault("supplier_id", supplier_id)
        subject, base_body = self.template_renderer.initial_quote_request(supplier_payload)
        marked_body, marker_token = attach_hidden_marker(
            base_body,
            supplier_id=supplier_id,
            unique_id=unique_id,
            run_id=uuid.uuid4().hex,
        )
        metadata = {
            "workflow_id": workflow_id,
            "supplier_id": supplier_id,
            "marker_token": marker_token,
            "sent_status": False,
        }
        draft = DraftEmail(
            workflow_id=workflow_id,
            unique_id=unique_id,
            supplier_id=supplier_id,
            supplier_email=supplier_email,
            subject=subject,
            body=marked_body,
            round_number=0,
            action_id=f"EMAIL-{uuid.uuid4().hex[:8].upper()}",
            metadata=metadata,
        )

        await self.db.register_draft(draft)
        return draft


class SupplierInteractionAgent:
    """Polls for dispatched messages and matching supplier responses."""

    def __init__(self, db: MockDatabaseConnection) -> None:
        self.db = db

    async def _await_dispatch_gate(
        self,
        workflow_id: str,
        *,
        timeout: float,
        poll_interval: float,
    ) -> Set[str]:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        while True:
            dispatches, unique_ids = await self.db.load_dispatch_summary(workflow_id)
            ready = all(record.dispatched_at for record in dispatches)
            if ready:
                return unique_ids
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError(
                    f"Dispatch gate timed out for workflow {workflow_id}: missing dispatches"
                )
            await asyncio.sleep(poll_interval)

    async def _await_response_gate(
        self,
        workflow_id: str,
        *,
        round_number: int,
        expected_unique_ids: Set[str],
        timeout: float,
        poll_interval: float,
    ) -> Dict[str, SupplierResponse]:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        request_id = build_status_request_id(workflow_id, round_number)
        while True:
            state = await self.db.get_round_status(request_id)
            if state and state["is_complete"]:
                responses = state["responses"]
                return {uid: responses[uid] for uid in expected_unique_ids if uid in responses}

            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError(
                    f"Response gate timed out for workflow {workflow_id} round {round_number}"
                )
            await asyncio.sleep(poll_interval)

    async def collect_round_responses(
        self,
        workflow_id: str,
        *,
        round_number: int,
        expected_unique_ids: Set[str],
        timeout: float,
        poll_interval: float,
    ) -> Dict[str, SupplierResponse]:
        return await self._await_response_gate(
            workflow_id,
            round_number=round_number,
            expected_unique_ids=expected_unique_ids,
            timeout=timeout,
            poll_interval=poll_interval,
        )


class NegotiationAgent:
    """Generate counter proposals for each supplier response."""

    def __init__(
        self,
        db: MockDatabaseConnection,
        context: WorkflowContextManager,
        *,
        template_renderer: Optional[NegotiationEmailTemplateRenderer] = None,
    ) -> None:
        self.db = db
        self.context = context
        self.template_renderer = template_renderer or NegotiationEmailTemplateRenderer.from_default()

    async def run(
        self,
        workflow_id: str,
        round_number: int,
        responses: Sequence[SupplierResponse],
        threads: Dict[str, EmailThread],
    ) -> List[DraftEmail]:
        drafts: List[DraftEmail] = []
        loop = asyncio.get_running_loop()
        tasks = []
        for response in responses:
            thread = threads[response.supplier_id]
            tasks.append(
                loop.run_in_executor(
                    None,
                    self._generate_negotiation_email,
                    workflow_id,
                    round_number,
                    thread,
                    response,
                )
            )

        for future in await asyncio.gather(*tasks):
            await self.db.register_draft(future)
            drafts.append(future)
        return drafts

    def _generate_negotiation_email(
        self,
        workflow_id: str,
        round_number: int,
        thread: EmailThread,
        response: SupplierResponse,
    ) -> DraftEmail:
        supplier_payload: Dict[str, Any] = {
            "supplier_id": thread.supplier_id,
            "name": response.payload.get("supplier_name") or thread.supplier_id,
            "contact_name": response.payload.get("contact_name"),
            "commodity": response.payload.get("commodity")
            or response.payload.get("item_name"),
        }
        if round_number <= 1:
            subject, content = self.template_renderer.renegotiation_request(
                supplier_payload
            )
        else:
            subject, content = self.template_renderer.final_review_request(
                supplier_payload
            )
        annotated_body, marker_token = attach_hidden_marker(
            content,
            supplier_id=thread.supplier_id,
            unique_id=thread.unique_id,
            run_id=uuid.uuid4().hex,
        )
        thread.add_message(
            f"negotiation_round_{round_number}",
            annotated_body,
            action_id=f"NEG-R{round_number}-{thread.supplier_unique_id}-{uuid.uuid4().hex[:6].upper()}",
            round_number=round_number,
        )
        metadata = {
            "workflow_id": workflow_id,
            "round_number": round_number,
            "marker_token": marker_token,
            "sent_status": False,
        }
        return DraftEmail(
            workflow_id=workflow_id,
            unique_id=thread.unique_id,
            supplier_id=thread.supplier_id,
            supplier_email=thread.supplier_email,
            subject=subject,
            body=annotated_body,
            round_number=round_number,
            action_id=f"NEG-{uuid.uuid4().hex[:8].upper()}",
            metadata=metadata,
        )


class QuoteComparisonAgent:
    """Produce a simple comparison across the final supplier offers."""

    def __init__(self, db: MockDatabaseConnection) -> None:
        self.db = db

    async def compare(self, workflow_id: str) -> Dict[str, Any]:
        final_round = max(
            (response.round_number for response in self.db.store.responses if response.workflow_id == workflow_id),
            default=0,
        )
        responses = await self.db.list_round_responses(
            workflow_id=workflow_id, round_number=final_round
        )
        summary = {}
        for response in responses:
            offer = response.payload.get("offer")
            summary[response.supplier_id] = {
                "unique_id": response.unique_id,
                "offer": offer,
                "received_at": response.received_at.isoformat(),
            }
        return {
            "workflow_id": workflow_id,
            "final_round": final_round,
            "offers": summary,
        }


class WorkflowOrchestrator:
    """Coordinates workflow phases while owning the workflow identifier."""

    def __init__(self, db: MockDatabaseConnection, wait_timeout: float = 300.0):
        self.db = db
        self.wait_timeout = wait_timeout
        self.check_interval = 1.0
        self.workflow_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.session = NegotiationSession(
            session_id=self.session_id,
            workflow_id=self.workflow_id,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self.context_manager = WorkflowContextManager()
        self.template_renderer = NegotiationEmailTemplateRenderer.from_default()
        self.threads: Dict[str, EmailThread] = {}
        self.drafting_agent = EmailDraftingAgent(
            db,
            self.context_manager,
            template_renderer=self.template_renderer,
        )
        self.interaction_agent = SupplierInteractionAgent(db)
        self.negotiation_agent = NegotiationAgent(
            db,
            self.context_manager,
            template_renderer=self.template_renderer,
        )
        self.comparison_agent = QuoteComparisonAgent(db)

    async def _initialize_response_tracking(
        self, round_num: int, expected_count: int, *, expected_unique_ids: Iterable[str] | None = None
    ) -> None:
        expected_ids = [uid for uid in expected_unique_ids or [] if uid]
        if not expected_ids and expected_count > 0:
            expected_ids = [
                thread.unique_id for thread in self.threads.values()
            ][:expected_count]
        request_id = build_status_request_id(self.session_id, round_num)
        await self.db.initialise_round_status(
            workflow_id=self.session_id,
            round_number=round_num,
            expected_unique_ids=expected_ids,
            request_id=request_id,
            expected_count=expected_count,
        )

    async def wait_for_responses(
        self, expected_count: int, context: str = "", round_num: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        request_id = build_status_request_id(self.session_id, round_num)
        elapsed = 0.0
        while elapsed < self.wait_timeout:
            state = await self.db.get_round_status(request_id)
            if state and state["is_complete"]:
                responses = state["responses"]
                return {
                    response.supplier_id: {
                        "supplier_id": response.supplier_id,
                        "unique_id": response.unique_id,
                        "response_data": response.payload,
                        "received_at": response.received_at,
                    }
                    for response in responses.values()
                }
            await asyncio.sleep(self.check_interval)
            elapsed += self.check_interval
        raise TimeoutError(
            f"Not all supplier responses for Round {round_num} received within {self.wait_timeout}s"
        )

    def create_thread_for_supplier(self, supplier: Dict[str, Any]) -> EmailThread:
        supplier_id = supplier.get("id") or supplier.get("supplier_id")
        thread = EmailThread(
            workflow_id=self.workflow_id,
            supplier_id=supplier_id,
            supplier_email=supplier.get("email"),
        )
        self.threads[supplier_id] = thread
        self.session.supplier_states[supplier_id] = {
            "thread_id": thread.thread_id,
            "unique_id": thread.unique_id,
            "rounds_completed": 0,
            "responses_received": 0,
        }
        return thread

    def build_approval_email(self) -> Dict[str, str]:
        """Return the approval email payload derived from the workflow template."""

        subject, body = self.template_renderer.approval_request_email()
        return {"subject": subject, "body": body}


async def handle_supplier_response(
    db: MockDatabaseConnection,
    session_id: str,
    round_num: int,
    supplier_id: str,
    unique_id: str,
    response_data: Dict[str, Any],
) -> SupplierResponse:
    """Persist a supplier response and update round tracking state."""

    return await db.record_supplier_response(
        workflow_id=session_id,
        unique_id=unique_id,
        supplier_id=supplier_id,
        round_number=round_num,
        payload=response_data,
    )


async def validate_implementation() -> None:
    """Run a lightweight validation similar to the legacy smoke test."""

    db = MockDatabaseConnection()
    orchestrator = WorkflowOrchestrator(db=db, wait_timeout=0.1)

    assert orchestrator.workflow_id, "workflow_id must be generated"
    assert orchestrator.workflow_id != orchestrator.session.session_id

    thread = orchestrator.create_thread_for_supplier({"id": "SUP-001", "email": "test@example.com"})
    assert thread.unique_id, "thread must provide a unique_id"

    await orchestrator._initialize_response_tracking(
        round_num=0,
        expected_count=1,
        expected_unique_ids={thread.unique_id},
    )

    await handle_supplier_response(
        db,
        orchestrator.session_id,
        0,
        thread.supplier_id,
        thread.unique_id,
        {"content": "hello"},
    )

    responses = await orchestrator.wait_for_responses(1, "validation", round_num=0)
    assert thread.supplier_id in responses

    approval_preview = orchestrator.build_approval_email()
    assert approval_preview.get("subject")
    assert approval_preview.get("body")


__all__ = [
    "DraftEmail",
    "DispatchRecord",
    "EmailDraftingAgent",
    "EmailThread",
    "NegotiationAgent",
    "NegotiationSession",
    "QuoteComparisonAgent",
    "SupplierInteractionAgent",
    "WorkflowContextManager",
    "WorkflowOrchestrator",
    "MockDatabaseConnection",
    "handle_supplier_response",
    "validate_implementation",
    "build_status_request_id",
]
