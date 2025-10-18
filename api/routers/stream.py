"""Streaming endpoints for plan execution events."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Streaming"])


class PlanStreamRequest(BaseModel):
    """Request payload for plan streaming."""

    task: str = Field(..., description="Description of the procurement task")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional contextual metadata to inform the plan",
    )


def _format_sse(event: str, payload: Dict[str, Any]) -> str:
    """Convert an event payload into Server-Sent Events wire format."""

    enriched = dict(payload)
    enriched.setdefault("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    message = json.dumps(enriched, ensure_ascii=False)
    return f"event: {event}\ndata: {message}\n\n"


async def _emit_plan_stream(
    orchestrator: Optional[Any],
    task: str,
    context: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Yield a deterministic sequence of plan execution events."""

    # Baseline plan tailored for Beyond ProcWise procurement flows.
    plan_steps: List[Dict[str, Any]] = [
        {
            "agent": "data_extraction",
            "action": "Synthesize supplier intelligence from internal sources",
            "status": "running",
        },
        {
            "agent": "negotiation",
            "action": "Draft human-in-the-loop negotiation outreach for suppliers",
            "status": "running",
        },
        {
            "agent": "quote_evaluation",
            "action": "Route refined quotes for comparative evaluation",
            "status": "running",
        },
    ]

    total_steps = len(plan_steps)

    yield _format_sse(
        "connected",
        {
            "message": "🤖 Connected to Beyond ProcWise Agentic System",
            "status": "running",
        },
    )

    yield _format_sse(
        "planning",
        {
            "message": "🧠 Analyzing your request and crafting a tailored execution plan...",
            "status": "running",
        },
    )

    # Placeholder integration point for a real orchestrator.
    if orchestrator is not None:
        logger.debug("Orchestrator available for task '%s' with context %s", task, context)

    yield _format_sse(
        "plan_created",
        {
            "message": "✅ Execution plan created",
            "status": "running",
            "total_steps": total_steps,
            "plan": [{"agent": step["agent"], "action": step["action"]} for step in plan_steps],
        },
    )

    for index, step in enumerate(plan_steps, start=1):
        yield _format_sse(
            "step_start",
            {
                "message": f"▶️ Step {index}/{total_steps}: {step['action']}",
                "status": "running",
                "agent": step["agent"],
                "step_number": index,
                "total_steps": total_steps,
            },
        )

        yield _format_sse(
            "agent_thinking",
            {
                "message": f"🤔 {step['agent']} is orchestrating procurement insights...",
                "status": "running",
                "agent": step["agent"],
                "step_number": index,
                "total_steps": total_steps,
            },
        )

        # Short pause to mimic real-time streaming cadence.
        await asyncio.sleep(0.05)

        yield _format_sse(
            "step_complete",
            {
                "message": f"✓ Completed: {step['action']}",
                "status": "completed",
                "agent": step["agent"],
                "step_number": index,
                "total_steps": total_steps,
            },
        )

    yield _format_sse(
        "execution_complete",
        {
            "message": "🎉 All tasks completed successfully!",
            "status": "success",
        },
    )


@router.post("/plan", response_class=StreamingResponse, status_code=status.HTTP_200_OK)
async def stream_plan(request: Request, payload: PlanStreamRequest) -> StreamingResponse:
    """Stream the execution plan for a procurement task via SSE."""

    async def event_publisher() -> AsyncGenerator[str, None]:
        try:
            orchestrator = getattr(request.app.state, "orchestrator", None)
            async for event in _emit_plan_stream(orchestrator, payload.task, payload.context):
                yield event
        except Exception as exc:  # pragma: no cover - safeguard for unexpected failures
            logger.exception("Plan streaming failed: %s", exc)
            yield _format_sse(
                "error",
                {
                    "message": f"❌ Error occurred: {exc}",
                    "status": "error",
                },
            )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers=headers,
    )
