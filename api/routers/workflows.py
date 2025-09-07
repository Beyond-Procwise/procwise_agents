# ProcWise/api/routers/workflows.py
"""API routes exposing the agent workflows."""
from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline

# Ensure GPU-related environment variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------
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

    min_financial_impact: float = 100.0


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


router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])


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
# Email drafting endpoint
# ---------------------------------------------------------------------------
@router.post("/email")
async def draft_email(
    subject: str = Form(...),
    recipient: str = Form(...),
    sender: Optional[str] = Form(None),
    body: Optional[str] = Form(None),
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Draft and send an email using the EmailDraftingAgent."""
    input_data = {
        "subject": subject,
        "recipient": recipient,
        "sender": sender,
        "body": body,
    }
    result = await run_in_threadpool(
        orchestrator.execute_workflow, "email_drafting", input_data
    )
    return result


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
