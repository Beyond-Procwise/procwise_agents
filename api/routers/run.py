import os
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
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

    # Run the long-running execution in background

    def _background_run(details_obj, process_id):
        logger.info("Starting background run for process %s", process_id)
        try:
            result = orchestrator.execute_agent_flow(details_obj)
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
            logger.debug("Execution result for process %s: %s", process_id, result)
            if isinstance(result, dict):
                try:
                    prs.update_process_details(process_id, result)
                except Exception:
                    logger.exception(
                        "Failed to update process %s details", process_id
                    )
                final = result.get("status")
            else:
                final = "failed"
            prs.update_process_status(process_id, 1 if final != "failed" else -1)
            logger.info(
                "Completed process %s with final status %s", process_id, final
            )
        except Exception:
            logger.exception(
                "Process %s raised an exception during execution", process_id
            )
            try:
                prs.update_process_status(process_id, -1)
            except Exception:
                logger.exception(
                    "Failed to update process %s to failed status", process_id
                )

    t = threading.Thread(target=_background_run, args=(details, req.process_id), daemon=True)
    t.start()

    return {"status": "started"}


