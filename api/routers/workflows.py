# ProcWise/api/routers/workflows.py
"""API routes exposing the agent workflows."""
from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline


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


class QuestionParams:
    """Utility class used with ``Depends`` to parse multipart form requests."""

    def __init__(
        self,
        query: str = Form(...),
        user_id: str = Form(...),
        model_name: Optional[str] = Form(None),
        doc_type: Optional[str] = Form(None),
        product_type: Optional[str] = Form(None),
        files: List[UploadFile] | None = File(None),
    ):
        self.query = query
        self.user_id = user_id
        self.model_name = model_name
        self.doc_type = doc_type
        self.product_type = product_type
        self.files = files or []


class RankingRequest(BaseModel):
    query: str


class ExtractRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


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
    params: QuestionParams = Depends(),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    if len(params.files) > 3:
        raise HTTPException(status_code=400, detail="Max 3 files allowed.")
    file_data = [(await f.read(), f.filename) for f in params.files if f.filename]
    result = pipeline.answer_question(
        query=params.query,
        user_id=params.user_id,
        model_name=params.model_name,
        files=file_data,
        doc_type=params.doc_type,
        product_type=params.product_type,
    )
    return result


@router.post("/rank")
def rank_suppliers(
    req: RankingRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    return orchestrator.execute_ranking_flow(req.query)


@router.post("/extract")
def extract_documents(
    req: ExtractRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    return orchestrator.execute_extraction_flow(req.s3_prefix, req.s3_object_key)


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
