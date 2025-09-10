import os
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional
import threading
import asyncio
import inspect
from orchestration.orchestrator import Orchestrator
from utils.gpu import configure_gpu

# Ensure GPU-related environment variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")
configure_gpu()

router = APIRouter(tags=["Run"], prefix="")

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    """Request schema for the ``/run`` endpoint."""

    process_id: int
    payload: Optional[Dict[str, Any]] = None


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

    # Initialise workflow progress to ``0`` before kicking off the background
    # execution so that callers can poll for real-time updates.
    details["status"] = 0
    try:
        prs.update_process_details(req.process_id, details)
    except Exception:
        logger.exception(
            "Failed to mark process %s as started", req.process_id
        )

    # Run the long-running execution in background

    def _background_run(details_obj, payload, process_id):
        logger.info("Starting background run for process %s", process_id)
        try:
            result = orchestrator.execute_agent_flow(
                details_obj, payload, process_id=process_id, prs=prs
            )
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
            logger.debug("Execution result for process %s: %s", process_id, result)
            if isinstance(result, dict):
                final = result.get("status")
            else:
                final = 0
            prs.update_process_status(process_id, 1 if final == 100 else -1)
            logger.info(
                "Completed process %s with final status %s", process_id, final
            )
        except Exception as exc:
            logger.exception(
                "Process %s raised an exception during execution", process_id
            )
            try:
                prs.update_process_details(
                    process_id,
                    {"status": 0, "error": str(exc)},
                )
            except Exception:
                logger.exception(
                    "Failed to persist error details for process %s", process_id
                )
            try:
                prs.update_process_status(process_id, -1)
            except Exception:
                logger.exception(
                    "Failed to update process %s to failed status", process_id
                )

    t = threading.Thread(target=_background_run, args=(details, req.payload or {}, req.process_id), daemon=True)
    t.start()

    return {"status": "started"}


