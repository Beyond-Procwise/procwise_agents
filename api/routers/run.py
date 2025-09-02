import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

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
        raise HTTPException(
            status_code=503, detail="Orchestrator service is not available."
        )
    return orchestrator


class AgentProperty(BaseModel):
    """Configuration information for an agent.

    The minimal data required to spin up an agent in the ProcWise
    framework.  This mirrors the ``agent_property`` structure supplied by
    the front end and is validated here before orchestration begins.
    """

    llm: str
    memory: Optional[str] = ""
    prompts: List[int] = Field(default_factory=list)
    policies: List[int] = Field(default_factory=list)


class AgentNode(BaseModel):
    """Recursive description of an agent flow node."""

    status: str
    agent_type: str = Field(alias="agent_type")
    agent_property: AgentProperty
    onNormal: Optional[Any] = None
    onFailure: Optional["AgentNode"] = None
    onSuccess: Optional["AgentNode"] = None


# Resolve forward references for self-referential model
AgentNode.model_rebuild()


class RunRequest(BaseModel):
    """Request schema for the ``/run`` endpoint.

    ``workflow`` is preserved for backwards compatibility with the
    existing API.  The new ``agent_flow`` field enables clients to submit
    a full agent orchestration graph which will be validated before
    execution.
    """

    workflow: Optional[str] = None
    payload: Dict[str, Any] = {}
    user_id: Optional[str] = None
    agent_flow: Optional[AgentNode] = None


@router.post("/run")
def run_agents(
    req: RunRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute a workflow using the orchestrator."""
    # When an agent flow is supplied we validate and execute it using the
    # orchestrator.  Otherwise fall back to the legacy ``workflow`` based
    # execution path.
    if req.agent_flow is not None:
        return orchestrator.execute_agent_flow(req.agent_flow.model_dump())
    if not req.workflow:
        raise HTTPException(
            status_code=400, detail="Either workflow or agent_flow must be provided"
        )
    return orchestrator.execute_workflow(req.workflow, req.payload, user_id=req.user_id)
