# ProcWise/api/routers/workflows.py
"""API routes exposing the agent workflows."""
from __future__ import annotations

import json
import os
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator, ConfigDict

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline
from services.opportunity_service import record_opportunity_feedback
from services.email_dispatch_service import EmailDispatchService

# Ensure GPU-related environment variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


logger = logging.getLogger(__name__)
def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator service is not available.")
    return orchestrator


def get_rag_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline service is not available.")
    return pipeline


def get_agent_nick(request: Request):
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if not agent_nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return agent_nick


class AskRequest(BaseModel):
    query: str
    user_id: str
    model_name: Optional[str] = None
    doc_type: Optional[str] = None
    product_type: Optional[str] = None
    file_path: Optional[str] = Field(default=None, description="Optional local file path", json_schema_extra={"example": None})

    @field_validator("doc_type", "product_type", "file_path", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() == "string":
                return None
        return value

    @field_validator("doc_type", "product_type")
    @classmethod
    def _normalize_case(cls, value: Optional[str]) -> Optional[str]:
        return value.lower() if isinstance(value, str) else value


class RankingRequest(BaseModel):
    query: str


class ExtractRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


class OpportunityMiningRequest(BaseModel):
    """Parameters for opportunity mining workflow."""

    workflow: str
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Policy specific parameters keyed by requirement name.",
    )
    min_financial_impact: float = Field(
        default=100.0,
        ge=0,
        description="Minimum savings required for an opportunity to be returned.",
    )

    @field_validator("workflow", mode="before")
    @classmethod
    def _normalise_workflow(cls, value: Any) -> str:
        if value is None:
            raise ValueError("workflow is required")
        if not isinstance(value, str):
            value = str(value)
        workflow = value.strip()
        if not workflow:
            raise ValueError("workflow must not be empty")
        return workflow

    @field_validator("conditions", mode="before")
    @classmethod
    def _default_conditions(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("conditions must be a mapping of field names to values")


class QuoteEvaluationRequest(BaseModel):
    """Input parameters for quote evaluation workflow."""

    supplier_names: Optional[List[str]] = None
    product_category: Optional[str] = None


class NegotiationRequest(BaseModel):
    supplier: str
    current_offer: float
    target_price: float
    user_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    amount: float
    supplier_id: Optional[str] = None
    threshold: Optional[float] = None
    user_id: Optional[str] = None


class SupplierInteractionRequest(BaseModel):
    message: str
    supplier_id: Optional[str] = None
    user_id: Optional[str] = None


class DiscrepancyRequest(BaseModel):
    extracted_docs: List[Dict[str, Any]]
    user_id: Optional[str] = None


class AgentType(BaseModel):
    agentId: int
    agentType: str
    description: str
    dependencies: List[str]


class OpportunityRejectionRequest(BaseModel):
    reason: Optional[str] = Field(
        default=None,
        description="Optional feedback describing why the opportunity was rejected.",
        max_length=2000,
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Identifier for the user submitting the feedback.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to persist alongside the feedback.",
    )
    opportunity_ref_id: Optional[str] = Field(
        default=None,
        description="Original opportunity reference identifier from the mining agent.",
    )


router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])


class EmailDraftPayload(BaseModel):
    """Structured representation of a stored RFQ draft."""

    unique_id: Optional[str] = Field(
        default=None,
        description="Unique identifier assigned to the draft (PROC-WF-XXXXXX).",
    )
    rfq_id: Optional[str] = Field(
        default=None,
        description="Legacy RFQ identifier used as a fallback when unique_id is absent.",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        description="Supplier reference associated with the draft.",
    )
    recipients: Optional[List[EmailStr]] = Field(
        default=None,
        description="Explicit list of recipient email addresses to override the stored draft values.",
    )
    sender: Optional[EmailStr] = Field(
        default=None,
        description="Sender email address override.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject override to apply when dispatching the draft.",
    )
    body: Optional[str] = Field(
        default=None,
        description="Body override to apply when dispatching the draft.",
    )
    action_id: Optional[str] = Field(
        default=None,
        description="Action identifier emitted by upstream orchestration steps.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary metadata captured alongside the draft for auditing.",
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("unique_id", "rfq_id", "supplier_id", "action_id", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("recipients", mode="before")
    @classmethod
    def _normalise_recipients(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            emails = [part.strip() for part in value.split(",") if part.strip()]
            return emails or None
        if isinstance(value, (list, tuple)):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned or None
        raise TypeError("recipients must be a string, list, or tuple of email addresses")

    def resolved_identifier(self) -> Optional[str]:
        for candidate in (self.unique_id, self.rfq_id):
            if candidate:
                identifier = candidate.strip()
                if identifier:
                    return identifier
        return None

    def resolved_recipients(self) -> Optional[List[str]]:
        if not self.recipients:
            return None
        return [str(email).strip() for email in self.recipients if str(email).strip()]

    def resolved_sender(self) -> Optional[str]:
        if self.sender is None:
            return None
        sender = str(self.sender).strip()
        return sender or None

    def resolved_subject(self) -> Optional[str]:
        if self.subject is None:
            return None
        return str(self.subject).strip()

    def resolved_body(self) -> Optional[str]:
        if self.body is None:
            return None
        return str(self.body)


class EmailDispatchRequest(BaseModel):
    """Request payload for dispatching stored RFQ drafts."""

    unique_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the draft to dispatch (PROC-WF-XXXXXX).",
    )
    rfq_id: Optional[str] = Field(
        default=None,
        description="Fallback RFQ identifier when unique_id is unavailable.",
    )
    recipients: Optional[List[EmailStr]] = Field(
        default=None,
        description="Override recipient list for the dispatched email.",
    )
    sender: Optional[EmailStr] = Field(
        default=None,
        description="Override sender email address.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional subject override.",
    )
    body: Optional[str] = Field(
        default=None,
        description="Optional body override.",
    )
    action_id: Optional[str] = Field(
        default=None,
        description="Identifier linking this dispatch request to upstream workflow actions.",
    )
    draft: Optional[EmailDraftPayload] = Field(
        default=None,
        description="Single draft payload when the request originates from the drafting agent.",
    )
    drafts: Optional[List[EmailDraftPayload]] = Field(
        default=None,
        description="Collection of draft payloads for batch dispatch scenarios.",
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _extract_identifier(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        if values.get("unique_id") or values.get("rfq_id"):
            return values

        draft_obj = values.get("draft")
        if isinstance(draft_obj, dict):
            unique_id = draft_obj.get("unique_id") or draft_obj.get("rfq_id")
            if unique_id:
                values["unique_id"] = unique_id
                return values

        drafts_array = values.get("drafts")
        if isinstance(drafts_array, list) and drafts_array:
            first_draft = drafts_array[0]
            if isinstance(first_draft, dict):
                unique_id = first_draft.get("unique_id") or first_draft.get("rfq_id")
                if unique_id:
                    values["unique_id"] = unique_id
                    return values

        if "unique_id" in values or "supplier_id" in values:
            return values

        available_fields = list(values.keys())
        raise ValueError(
            "No identifier found in request. Provide 'unique_id' or 'rfq_id'. "
            f"Available fields: {available_fields}"
        )

    @field_validator("unique_id", "rfq_id", "action_id", mode="before")
    @classmethod
    def _strip_identifiers(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("recipients", mode="before")
    @classmethod
    def _normalise_recipients(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            emails = [part.strip() for part in value.split(",") if part.strip()]
            return emails or None
        if isinstance(value, (list, tuple)):
            cleaned: List[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned or None
        raise TypeError("recipients must be provided as a string, list, or tuple of emails")

    def get_identifier(self) -> str:
        for candidate in (self.unique_id, self.rfq_id):
            if candidate:
                identifier = candidate.strip()
                if identifier:
                    return identifier

        if self.draft:
            identifier = self.draft.resolved_identifier()
            if identifier:
                return identifier

        if self.drafts:
            for draft in self.drafts:
                identifier = draft.resolved_identifier()
                if identifier:
                    return identifier

        return ""

    def resolve_recipients(self) -> Optional[List[str]]:
        if self.recipients:
            return [str(email).strip() for email in self.recipients if str(email).strip()]
        if self.draft:
            recipients = self.draft.resolved_recipients()
            if recipients:
                return recipients
        return None

    def resolve_sender(self) -> Optional[str]:
        if self.sender is not None:
            sender = str(self.sender).strip()
            return sender or None
        if self.draft:
            return self.draft.resolved_sender()
        return None

    def resolve_subject(self) -> Optional[str]:
        if self.subject is not None:
            return str(self.subject).strip()
        if self.draft:
            return self.draft.resolved_subject()
        return None

    def resolve_body(self) -> Optional[str]:
        if self.body is not None:
            return str(self.body)
        if self.draft:
            return self.draft.resolved_body()
        return None

    def resolve_action_id(self) -> Optional[str]:
        if self.action_id:
            return self.action_id
        if self.draft and self.draft.action_id:
            return self.draft.action_id
        return None


class EmailBatchDispatchRequest(BaseModel):
    """Request payload for batch email dispatch operations."""

    drafts: List[EmailDraftPayload] = Field(
        ..., description="List of drafts to dispatch in a single batch operation."
    )

    model_config = ConfigDict(extra="allow")


class EmailDispatchResponse(BaseModel):
    success: bool
    unique_id: str
    sent: bool
    message_id: Optional[str] = None
    recipients: List[str]
    sender: str
    subject: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/ask")
async def ask_question(
    req: AskRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    file_data: List[tuple[bytes, str]] = []
    if req.file_path:
        if not os.path.isfile(req.file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {req.file_path}")
        try:
            with open(req.file_path, "rb") as f:
                file_data.append((f.read(), os.path.basename(req.file_path)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    result = await run_in_threadpool(
        pipeline.answer_question,
        query=req.query,
        user_id=req.user_id,
        model_name=req.model_name,
        files=file_data or None,
        doc_type=req.doc_type,
        product_type=req.product_type,
    )
    return result


@router.post("/rank")
def rank_suppliers(
    req: RankingRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    # Use the public workflow entry point so that the orchestrator can build a
    # proper ``AgentContext``. Calling internal methods directly would bypass
    # this setup and lead to attribute errors such as ``'str' object has no
    # attribute 'input_data'`` when the workflow tries to access context fields.
    return orchestrator.execute_workflow("supplier_ranking", {"query": req.query})


@router.post("/quotes/evaluate")
def evaluate_quotes(
    req: QuoteEvaluationRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Evaluate and compare supplier quotes."""
    return orchestrator.execute_workflow("quote_evaluation", req.model_dump())


@router.post("/opportunities")
def mine_opportunities(
    req: OpportunityMiningRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    prs = orchestrator.agent_nick.process_routing_service
    process_id = prs.log_process(
        process_name="opportunity_mining",
        process_details=req.model_dump(),
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")

    action_id = prs.log_action(
        process_id=process_id,
        agent_type="opportunity_miner",
        action_desc=req.model_dump(),
        status="started",
    )
    try:
        result = orchestrator.execute_workflow(
            "opportunity_mining", req.model_dump()
        )
        prs.log_action(
            process_id=process_id,
            agent_type="opportunity_miner",
            action_desc=req.model_dump(),
            process_output=result,
            status="completed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, 1)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        prs.log_action(
            process_id=process_id,
            agent_type="opportunity_miner",
            action_desc=str(exc),
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/opportunities/{opportunity_id}/reject")
def reject_opportunity(
    opportunity_id: str,
    req: OpportunityRejectionRequest,
    agent_nick=Depends(get_agent_nick),
):
    if not opportunity_id or not opportunity_id.strip():
        raise HTTPException(status_code=400, detail="opportunity_id must be provided")

    try:
        record = record_opportunity_feedback(
            agent_nick,
            opportunity_id.strip(),
            opportunity_ref_id=req.opportunity_ref_id,
            status="rejected",
            reason=req.reason,
            user_id=req.user_id,
            metadata=req.metadata,
        )
    except Exception as exc:  # pragma: no cover - database/network
        logger.exception("Failed to record rejection for opportunity %s", opportunity_id)
        raise HTTPException(status_code=500, detail="Failed to record opportunity feedback") from exc

    updated_on = record.get("updated_on")
    if isinstance(updated_on, (bytes, str)):
        updated_iso = str(updated_on)
    elif updated_on is not None:
        updated_iso = updated_on.isoformat()
    else:
        updated_iso = None

    payload = {
        "opportunity_id": record.get("opportunity_id"),
        "opportunity_ref_id": record.get("opportunity_ref_id"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "user_id": record.get("user_id"),
        "metadata": record.get("metadata"),
        "updated_on": updated_iso,
    }
    return {"status": "success", "feedback": payload}


@router.post("/extract")
async def extract_documents(
    req: ExtractRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    prs = orchestrator.agent_nick.process_routing_service
    process_id = prs.log_process(
        process_name="document_extraction",
        process_details={
            "s3_prefix": req.s3_prefix,
            "s3_object_key": req.s3_object_key,
        },
        # process_status=1,
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")

    action_id = prs.log_action(
        process_id=process_id,
        agent_type="document_extraction",
        action_desc={
            "s3_prefix": req.s3_prefix,
            "s3_object_key": req.s3_object_key,
        },
        status="started",
    )

    async def run_flow() -> None:
        try:
            result = await run_in_threadpool(
                orchestrator.execute_extraction_flow,
                req.s3_prefix,
                req.s3_object_key,
            )
            prs.log_action(
                process_id=process_id,
                agent_type="document_extraction",
                action_desc={
                    "s3_prefix": req.s3_prefix,
                    "s3_object_key": req.s3_object_key,
                },
                process_output=result,
                status="completed",
                action_id=action_id,
            )
            prs.update_process_status(process_id, 1)
        except Exception as exc:  # pragma: no cover - network/runtime
            prs.log_action(
                process_id=process_id,
                agent_type="document_extraction",
                action_desc=str(exc),
                status="failed",
                action_id=action_id,
            )
            prs.update_process_status(process_id, -1)

    asyncio.create_task(run_flow())
    return {"status": "process started", "process_id": process_id}


# ---------------------------------------------------------------------------
# Email dispatch endpoint
# ---------------------------------------------------------------------------
@router.post("/email")
async def send_email(
    request: EmailDispatchRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    agent_nick=Depends(get_agent_nick),
):
    """Send a previously drafted RFQ email using the dispatch service."""

    identifier = request.get_identifier()
    if not identifier:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing Identifier",
                "message": "No unique_id or rfq_id provided",
                "hint": "Send: {\"unique_id\": \"PROC-WF-xxx\"}",
            },
        )

    input_data = request.model_dump(exclude_none=True)

    prs = orchestrator.agent_nick.process_routing_service

    initial_details = {
        "input": input_data,
        "agents": [],
        "output": {},
        "status": "saved",
    }
    process_id = prs.log_process(
        process_name="email_dispatch", process_details=initial_details
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")

    action_reference = request.resolve_action_id()

    action_id = prs.log_action(
        process_id=process_id,
        agent_type="email_dispatch",
        action_desc=input_data,
        status="started",
        action_id=action_reference,
    )

    dispatch_service = EmailDispatchService(agent_nick)

    try:
        result = await run_in_threadpool(
            dispatch_service.send_draft,
            identifier=identifier,
            recipients=request.resolve_recipients(),
            sender=request.resolve_sender(),
            subject_override=request.resolve_subject(),
            body_override=request.resolve_body(),
        )

        dispatch_timestamp = time.time()
        setattr(agent_nick, "dispatch_service_started", True)
        setattr(agent_nick, "email_dispatch_last_sent_at", dispatch_timestamp)

        raw_recipients = result.get("recipients")
        if isinstance(raw_recipients, str):
            response_recipients = [raw_recipients]
        elif isinstance(raw_recipients, list):
            response_recipients = raw_recipients
        elif raw_recipients:
            response_recipients = list(raw_recipients)
        else:
            response_recipients = request.resolve_recipients() or []

        response = EmailDispatchResponse(
            success=bool(result.get("sent", False)),
            unique_id=result.get("unique_id", identifier),
            sent=bool(result.get("sent", False)),
            message_id=result.get("message_id"),
            recipients=[str(r) for r in response_recipients],
            sender=str(result.get("sender") or request.resolve_sender() or ""),
            subject=str(result.get("subject") or request.resolve_subject() or ""),
            error=result.get("error"),
        )

        prs.log_action(
            process_id=process_id,
            agent_type="email_dispatch",
            action_desc=input_data,
            process_output=response.model_dump(),
            status="completed" if response.sent else "failed",
            action_id=action_id,
        )

        final_details = {
            "input": input_data,
            "agents": [],
            "output": response.model_dump(),
            "status": "completed" if response.sent else "failed",
        }
        prs.update_process_details(process_id, final_details)
        prs.update_process_status(process_id, 1 if response.sent else -1)

    except ValueError as exc:
        prs.log_action(
            process_id=process_id,
            agent_type="email_dispatch",
            action_desc=input_data,
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)

        error_message = str(exc)
        status_code = 404 if "No stored draft" in error_message else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": "Draft Dispatch Failed",
                "message": error_message,
                "identifier": identifier,
            },
        )

    except Exception as exc:  # pragma: no cover - network/runtime
        prs.log_action(
            process_id=process_id,
            agent_type="email_dispatch",
            action_desc=input_data,
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)
        raise HTTPException(status_code=500, detail=str(exc))

    return response


@router.post("/email/batch")
async def dispatch_batch_emails(
    request: EmailBatchDispatchRequest,
    agent_nick=Depends(get_agent_nick),
):
    if not request.drafts:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No Drafts",
                "message": "No drafts found in request body",
                "hint": "Send EmailDraftingAgent output with 'drafts' array",
            },
        )

    dispatch_service = EmailDispatchService(agent_nick)

    results: List[Dict[str, Any]] = []
    for draft in request.drafts:
        identifier = draft.resolved_identifier()
        if not identifier:
            logger.warning(
                "Skipping draft without identifier: supplier_id=%s", draft.supplier_id
            )
            results.append(
                {
                    "unique_id": None,
                    "sent": False,
                    "error": "Draft missing unique identifier",
                    "supplier_id": draft.supplier_id,
                    "subject": draft.resolved_subject(),
                }
            )
            continue

        try:
            result = await run_in_threadpool(
                dispatch_service.send_draft,
                identifier=identifier,
                recipients=draft.resolved_recipients(),
                sender=draft.resolved_sender(),
                subject_override=draft.resolved_subject(),
                body_override=draft.resolved_body(),
            )
            results.append(
                {
                    "unique_id": identifier,
                    "sent": bool(result.get("sent")),
                    "message_id": result.get("message_id"),
                    "supplier_id": draft.supplier_id,
                    "subject": result.get("subject") or draft.resolved_subject(),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.error("Failed to dispatch %s: %s", identifier, str(exc))
            results.append(
                {
                    "unique_id": identifier,
                    "sent": False,
                    "error": str(exc),
                    "supplier_id": draft.supplier_id,
                    "subject": draft.resolved_subject(),
                }
            )

    success_count = sum(1 for r in results if r.get("sent"))

    return {
        "success": success_count == len(results) if results else False,
        "total": len(results),
        "sent": success_count,
        "failed": len(results) - success_count,
        "results": results,
    }


@router.post("/{workflow_id}/email/dispatch-all")
async def dispatch_workflow_drafts(
    workflow_id: str,
    agent_nick=Depends(get_agent_nick),
):
    try:
        with agent_nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT unique_id, supplier_id, subject, sent
                    FROM proc.draft_rfq_emails
                    WHERE workflow_id = %s
                      AND sent = FALSE
                    ORDER BY id ASC
                    """,
                    (workflow_id,),
                )
                draft_rows = cur.fetchall()

        if not draft_rows:
            return {
                "success": True,
                "workflow_id": workflow_id,
                "total": 0,
                "sent": 0,
                "failed": 0,
                "message": "No unsent drafts found for workflow",
            }

        dispatch_service = EmailDispatchService(agent_nick)

        results: List[Dict[str, Any]] = []
        for unique_id, supplier_id, subject, sent in draft_rows:
            try:
                result = await run_in_threadpool(
                    dispatch_service.send_draft,
                    identifier=unique_id,
                    subject_override=subject,
                )
                results.append(
                    {
                        "unique_id": unique_id,
                        "sent": bool(result.get("sent")),
                        "message_id": result.get("message_id"),
                        "supplier_id": supplier_id,
                        "subject": subject,
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.error("Failed to dispatch %s: %s", unique_id, str(exc))
                results.append(
                    {
                        "unique_id": unique_id,
                        "sent": False,
                        "error": str(exc),
                        "supplier_id": supplier_id,
                        "subject": subject,
                    }
                )

        success_count = sum(1 for r in results if r.get("sent"))

        return {
            "success": success_count == len(results),
            "workflow_id": workflow_id,
            "total": len(results),
            "sent": success_count,
            "failed": len(results) - success_count,
            "results": results,
        }

    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("Workflow dispatch failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Workflow Dispatch Failed",
                "message": str(exc),
            },
        )



@router.post("/negotiate")
def negotiate(
    req: NegotiationRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute the negotiation agent."""
    return orchestrator.execute_workflow("negotiation", req.model_dump())


@router.post("/approvals")
def approvals(
    req: ApprovalRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Run the approvals agent."""
    return orchestrator.execute_workflow("approvals", req.model_dump())


@router.post("/supplier-interaction")
def supplier_interaction(
    req: SupplierInteractionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Trigger the supplier interaction agent."""
    return orchestrator.execute_workflow("supplier_interaction", req.model_dump())


@router.post("/discrepancy")
def detect_discrepancy(
    req: DiscrepancyRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Expose the discrepancy detection agent."""
    return orchestrator.execute_workflow("discrepancy_detection", req.model_dump())


@router.get(
    "/types",
    response_model=List[AgentType],
    summary="Get agent types and their resource dependencies",
)
def get_agent_types():
    """Return the agent catalogue defined in ``agent_definitions.json``."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "agent_definitions.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    return [AgentType(**agent) for agent in data]
