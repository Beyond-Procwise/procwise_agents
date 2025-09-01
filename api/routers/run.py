import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator

# Ensure GPU-related environment variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

router = APIRouter(tags=["Run"], prefix="")


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator service is not available.")
    return orchestrator


class RunRequest(BaseModel):
    workflow: str
    payload: Dict[str, Any] = {}
    user_id: Optional[str] = None


@router.post("/run")
def run_agents(
    req: RunRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute a workflow using the orchestrator."""
    return orchestrator.execute_workflow(req.workflow, req.payload, user_id=req.user_id)
