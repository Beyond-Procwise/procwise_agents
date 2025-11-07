"""Training management API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.model_training_endpoint import ModelTrainingEndpoint

router = APIRouter(prefix="/training", tags=["Model Training"])


class TrainingJobSummary(BaseModel):
    """Lightweight view of a dispatched training job."""

    job_id: int = Field(..., description="Identifier of the training job queue entry")
    workflow_id: Optional[str] = Field(
        None, description="Workflow identifier associated with the job"
    )
    agent_slug: str = Field(..., description="Agent that owns the training routine")
    policy_id: Optional[str] = Field(None, description="Related policy identifier, if any")
    status: str = Field(
        "completed",
        description="Resulting status of the job after dispatch",
    )


class TrainingDispatchRequest(BaseModel):
    limit: Optional[int] = Field(
        None,
        ge=1,
        description="Optional maximum number of queued jobs to execute",
    )


class TrainingDispatchResponse(BaseModel):
    dispatched: int = Field(..., description="Number of training jobs executed")
    jobs: List[TrainingJobSummary] = Field(
        default_factory=list,
        description="Summaries of the dispatched training jobs",
    )
    relationship_jobs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summaries of supplier relationship refresh executions",
    )
    negotiation_learnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Negotiation learning records processed during dispatch",
    )
    workflow_context: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent workflow context captured from the learning repository",
    )
    phi4_fine_tuning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary of the phi4 humanisation fine-tuning pipeline run",
    )
    rag_qwen30b_finetuning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary of the RAG qwen3-30b instruction example generation run",
    )


def _get_training_endpoint(request: Request) -> ModelTrainingEndpoint:
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if agent_nick is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent context is not initialised",
        )
    endpoint = getattr(request.app.state, "model_training_endpoint", None)
    if not isinstance(endpoint, ModelTrainingEndpoint):
        try:
            endpoint = ModelTrainingEndpoint(agent_nick)
            request.app.state.model_training_endpoint = endpoint
        except Exception as exc:  # pragma: no cover - defensive initialisation
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialise training service: {exc}",
            ) from exc
    return endpoint


@router.post(
    "/dispatch",
    response_model=TrainingDispatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute queued model training jobs immediately",
)
def trigger_training_dispatch(
    request: TrainingDispatchRequest = Body(
        default_factory=TrainingDispatchRequest,
        description="Optional dispatch configuration",
    ),
    endpoint: ModelTrainingEndpoint = Depends(_get_training_endpoint),
) -> TrainingDispatchResponse:
    """Trigger queued model training jobs on demand."""

    limit = request.limit
    result = endpoint.dispatch(force=True, limit=limit)
    jobs = result.get("training_jobs", [])
    relationship_jobs = result.get("relationship_jobs", [])
    negotiation_learnings = result.get("negotiation_learnings", [])
    workflow_context = result.get("workflow_context", [])
    phi4_fine_tuning = result.get("phi4_fine_tuning")
    rag_qwen30b_finetuning = result.get("rag_qwen30b_finetuning")
    summaries = [
        TrainingJobSummary(
            job_id=job.get("job_id"),
            workflow_id=job.get("workflow_id"),
            agent_slug=job.get("agent_slug", ""),
            policy_id=job.get("policy_id"),
        )
        for job in jobs
    ]
    return TrainingDispatchResponse(
        dispatched=len(summaries),
        jobs=summaries,
        relationship_jobs=relationship_jobs,
        negotiation_learnings=negotiation_learnings,
        workflow_context=workflow_context,
        phi4_fine_tuning=phi4_fine_tuning,
        rag_qwen30b_finetuning=rag_qwen30b_finetuning,
    )
