"""Email-related API routes."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from services.email_watcher_service import run_email_watcher_for_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["Email"])


class EmailWatcherTriggerRequest(BaseModel):
    """Request payload for triggering the EmailWatcher."""

    workflow_id: str = Field(
        ..., min_length=1, description="Workflow identifier used to scope email polling."
    )

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> str:
        if value is None:
            raise ValueError("workflow_id is required")
        if not isinstance(value, str):
            value = str(value)
        workflow = value.strip()
        if not workflow:
            raise ValueError("workflow_id must not be empty")
        return workflow


class EmailWatcherTriggerResponse(BaseModel):
    """Structured response returned by the EmailWatcher trigger endpoint."""

    status: str = Field(..., description="High-level execution status.")
    processed_count: int = Field(
        ..., ge=0, description="Number of supplier responses processed for the workflow."
    )
    expected_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of expected supplier responses for the workflow if known.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Collection of errors or outstanding conditions encountered during processing.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw watcher response payload for downstream inspection.",
    )


def _resolve_agent_registry(request: Request) -> Any:
    agent_registry = getattr(request.app.state, "agent_registry", None)
    if agent_registry is None:
        agent_nick = getattr(request.app.state, "agent_nick", None)
        agent_registry = getattr(agent_nick, "agents", None)
    return agent_registry


def _get_dependency_or_503(value: Any, name: str) -> Any:
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} is not available",
        )
    return value


async def _run_email_watcher(
    *,
    workflow_id: str,
    request: Request,
) -> Dict[str, Any]:
    orchestrator = _get_dependency_or_503(
        getattr(request.app.state, "orchestrator", None), "Orchestrator service"
    )
    agent_registry = _resolve_agent_registry(request)
    supplier_agent = getattr(request.app.state, "supplier_interaction_agent", None)
    negotiation_agent = getattr(request.app.state, "negotiation_agent", None)

    runner = getattr(request.app.state, "email_watcher_runner", None) or run_email_watcher_for_workflow

    return await run_in_threadpool(
        runner,
        workflow_id=workflow_id,
        run_id=None,
        agent_registry=agent_registry,
        orchestrator=orchestrator,
        supplier_agent=supplier_agent,
        negotiation_agent=negotiation_agent,
    )


@router.post("/emailwatcher", response_model=EmailWatcherTriggerResponse)
async def trigger_email_watcher(
    payload: EmailWatcherTriggerRequest,
    request: Request,
) -> EmailWatcherTriggerResponse:
    """Trigger the EmailWatcher for a specific workflow."""

    workflow_id = payload.workflow_id.strip()
    logger.info("EmailWatcher trigger received", extra={"workflow_id": workflow_id})
    try:
        result = await _run_email_watcher(workflow_id=workflow_id, request=request)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("EmailWatcher run failed for workflow %s", workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email watcher execution failed: {exc}",
        ) from exc

    processed = int(result.get("found") or 0)
    rows = result.get("rows")
    if processed == 0 and isinstance(rows, list):
        processed = len(rows)

    errors: List[str] = []
    reason = result.get("reason")
    status_value = result.get("status") or result.get("watcher_status") or "unknown"
    if reason and status_value not in {"processed", "completed"}:
        errors.append(str(reason))

    missing_ids = result.get("missing_unique_ids")
    if missing_ids:
        errors.append(
            "Missing responses for unique IDs: " + ", ".join(str(item) for item in missing_ids)
        )

    details = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "status",
            "watcher_status",
            "found",
            "rows",
            "reason",
            "missing_unique_ids",
        }
    }
    details["watcher_status"] = result.get("watcher_status")
    details["raw_rows"] = rows
    if reason:
        details["reason"] = reason
    if missing_ids:
        details["missing_unique_ids"] = missing_ids

    return EmailWatcherTriggerResponse(
        status=status_value,
        processed_count=processed,
        expected_count=result.get("expected"),
        errors=errors,
        details=details,
    )
