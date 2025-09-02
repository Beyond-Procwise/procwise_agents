import os
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator
from utils.gpu import configure_gpu

# Ensure GPU-related environment variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")
configure_gpu()

router = APIRouter(tags=["Run"], prefix="")


class RunRequest(BaseModel):
    """Request schema for the ``/run`` endpoint."""

    process_id: int


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(
            status_code=503, detail="Orchestrator service is not available."
        )
    return orchestrator


@router.post("/run")
def run_agents(
    req: RunRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute an agent flow fetched from the routing table."""
    prs = getattr(orchestrator.agent_nick, "process_routing_service", None)
    if not prs:
        raise HTTPException(
            status_code=503, detail="Process routing service is not available."
        )

    details = prs.get_process_details(req.process_id)
    if details is None:
        raise HTTPException(status_code=404, detail="Process not found")
    if details.get("status") != "saved":
        raise HTTPException(status_code=409, detail="Process not in saved status")

    prs.update_process_status(req.process_id, 1)
    result = orchestrator.execute_agent_flow(details)
    if isinstance(result, dict):
        details["status"] = (
            "completed" if result.get("status") != "failed" else "failed"
        )
        prs.update_process_details(req.process_id, details)
    prs.update_process_status(
        req.process_id, 2 if result.get("status") != "failed" else 3
    )
    return result
