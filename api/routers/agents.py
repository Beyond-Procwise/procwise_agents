from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional
import logging
import os
from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator

# Ensure GPU-related environment variables are set for agent operations
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])


def get_agent_nick(request: Request):
    """Get AgentNick from app state"""
    if not hasattr(request.app.state, 'agent_nick') or not request.app.state.agent_nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return request.app.state.agent_nick


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator service is not available.")
    return orchestrator


@router.get("/list")
async def list_agents(agent_nick=Depends(get_agent_nick)):
    """List all available agents"""
    return {
        "agents": list(agent_nick.agents.keys()),
        "total": len(agent_nick.agents)
    }


@router.get("/status/{agent_name}")
async def get_agent_status(agent_name: str, agent_nick=Depends(get_agent_nick)):
    """Get status of specific agent"""
    if agent_name not in agent_nick.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

    agent = agent_nick.agents[agent_name]
    return {
        "name": agent_name,
        "class": agent.__class__.__name__,
        "status": "ready"
    }


@router.post("/reload-policies")
async def reload_policies(agent_nick=Depends(get_agent_nick)):
    """Reload policy configurations"""
    try:
        agent_nick.policy_engine.reload_policies()
        return {"status": "success", "message": "Policies reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AgentExecutionRequest(BaseModel):
    agent_type: str
    payload: Dict[str, Any] = {}


class DocumentProcessRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


@router.post("/process-document")
def process_document(
    req: DocumentProcessRequest, orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Convenience endpoint to run the document extraction workflow."""
    payload = {"s3_prefix": req.s3_prefix, "s3_object_key": req.s3_object_key}
    return orchestrator.execute_workflow("document_extraction", payload)


@router.post("/execute")
def execute_agent(
    req: AgentExecutionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute a specified agent workflow."""
    prs = orchestrator.agent_nick.process_routing_service
    process_id = prs.log_process(
        process_name=req.agent_type,
        process_details=req.payload,
        process_status=1,
    )
    action_id = prs.log_action(
        process_id=process_id,
        agent_type=req.agent_type,
        action_desc=req.payload,
        status="started",
    )
    try:
        result = orchestrator.execute_workflow(req.agent_type, req.payload)
        prs.log_action(
            process_id=process_id,
            agent_type=req.agent_type,
            action_desc=req.payload,
            process_output=result,
            status="completed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, 1)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        prs.log_action(
            process_id=process_id,
            agent_type=req.agent_type,
            action_desc=str(exc),
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)
        raise HTTPException(status_code=500, detail=str(exc))
