# ProcWise/api/routers/workflows.py

import json
import os
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class RankingRequest(BaseModel):
    query: str


class ExtractRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


class QuestionParams:
    def __init__(
        self,
        query: str = Form(...),
        user_id: str = Form("default_user"),
        model_name: Optional[str] = Form(None),
        doc_type: Optional[str] = Form(None),
        product_type: Optional[str] = Form(None),
        files: List[UploadFile] = File(default=[]),
    ):
        self.query = query
        self.user_id = user_id
        self.model_name = model_name
        self.doc_type = doc_type
        self.product_type = product_type
        self.files = files


class AgentType(BaseModel):
    agentId: int
    agentType: str
    description: str
    dependencies: List[str]


def load_from_json(file_path: str):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail=f"Configuration file not found: {file_path}"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail=f"Error decoding JSON from file: {file_path}"
        )


router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])


def get_orchestrator(req: Request):
    return req.app.state.orchestrator


def get_rag_pipeline(req: Request):
    return req.app.state.rag_pipeline


@router.post("/rank-suppliers")
def rank_suppliers(
    req: RankingRequest, orch: Orchestrator = Depends(get_orchestrator)
):
    return orch.execute_ranking_flow(req.query)


@router.post("/ask")
async def ask_question(
    params: QuestionParams = Depends(),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    if len(params.files) > 3:
        raise HTTPException(400, "Max 3 files allowed.")
    file_data = [(await f.read(), f.filename) for f in params.files if f.filename]
    return pipeline.answer_question(
        query=params.query,
        user_id=params.user_id,
        model_name=params.model_name,
        files=file_data,
        doc_type=params.doc_type,
        product_type=params.product_type,
    )


@router.post("/extract", status_code=202)
def extract_docs(
    req: ExtractRequest,
    bg: BackgroundTasks,
    orch: Orchestrator = Depends(get_orchestrator),
):
    bg.add_task(
        orch.execute_extraction_flow,
        s3_prefix=req.s3_prefix,
        s3_object_key=req.s3_object_key,
    )
    return {"status": "Extraction initiated."}


@router.get(
    "/types",
    response_model=List[AgentType],
    summary="Get agent types and their resource dependencies",
)
def get_agent_types():
    """
    Returns a list of all defined agent types, their descriptions,
    and their dependencies from the agent_definitions.json file.
    """

    agent_definitions_path = os.path.join(
        PROJECT_ROOT, "prompts", "agent_definitions.json"
    )
    return load_from_json(agent_definitions_path)

