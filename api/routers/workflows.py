# ProcWise/api/routers/workflows.py

import json
import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Form, UploadFile, File
from pydantic import BaseModel
from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline
from typing import Optional, List

# --- UPGRADE: Fix Hardcoded File Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- UPGRADE: HITL Pydantic Model (Import existing schemas) ---
from agents.schemas import HeaderData, LineItem, ValidationResult

class RankingRequest(BaseModel):
    query: str

class ExtractRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None

class CorrectedExtractionData(BaseModel):
    headerData: HeaderData
    lineItems: List[LineItem]
    validation: ValidationResult

class QuestionParams:
    def __init__(self, query: str = Form(...), user_id: str = Form("default_user"), files: List[UploadFile] = File(default=[])):
        self.query, self.user_id, self.files = query, user_id, files

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
        raise HTTPException(status_code=500, detail=f"Configuration file not found: {file_path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error decoding JSON from file: {file_path}")

router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])

def get_orchestrator(req: Request) -> Orchestrator:
    if not hasattr(req.app.state, 'orchestrator') or not req.app.state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator service is not available.")
    return req.app.state.orchestrator

def get_rag_pipeline(req: Request) -> RAGPipeline:
    if not hasattr(req.app.state, 'rag_pipeline') or not req.app.state.rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline service is not available.")
    return req.app.state.rag_pipeline

@router.post("/rank-suppliers")
def rank_suppliers(req: RankingRequest, orch: Orchestrator = Depends(get_orchestrator)):
    result = orch.execute_ranking_flow(req.query)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/ask")
async def ask_question(params: QuestionParams = Depends(), pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    if len(params.files) > 3: raise HTTPException(400, "Max 3 files allowed.")
    file_data = [(await f.read(), f.filename) for f in params.files if f.filename]
    return pipeline.answer_question(query=params.query, user_id=params.user_id, files=file_data)

@router.post("/extract", status_code=202)
def extract_docs(req: ExtractRequest, bg: BackgroundTasks, orch: Orchestrator = Depends(get_orchestrator)):
    bg.add_task(orch.execute_extraction_flow, s3_prefix=req.s3_prefix, s3_object_key=req.s3_object_key)
    return {"message": "Extraction initiated. This is an asynchronous process."}

@router.post("/correct-extraction/{extraction_id}", summary="Submit corrections for an extraction")
def correct_extraction(extraction_id: str, corrected_data: CorrectedExtractionData, orch: Orchestrator = Depends(get_orchestrator)):
    """
    Allows a user to submit corrected data for an extraction that was flagged
    for review (status='needs_review'). The `extraction_id` is the invoice_id or po_number.
    """
    result = orch.execute_correction_flow(extraction_id, corrected_data.model_dump(by_alias=True))
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("reason", "Correction failed."))
    return result

@router.get("/types", response_model=List[AgentType], summary="Get agent types and their resource dependencies")
def get_agent_types():
    """
    Returns a list of all defined agent types, their descriptions,
    and their dependencies from the agent_definitions.json file.
    """
    definitions_path = os.path.join(PROJECT_ROOT, 'prompts', 'agent_definitions.json')
    return load_from_json(definitions_path)
