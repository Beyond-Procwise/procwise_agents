"""Coordinated access point for model training flows.

The endpoint wraps :class:`ModelTrainingService` so that all background
training routines, batch dispatches, and process refreshes execute through a
single orchestrated interface.  This keeps training logic detached from
individual agents while still allowing shared infrastructure (policy engine,
DB connections) to be reused via ``agent_nick``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.model_training_service import ModelTrainingService

logger = logging.getLogger(__name__)


class ModelTrainingEndpoint:
    """Facade that centralises training operations."""

    def __init__(self, agent_nick: Any) -> None:
        self._agent_nick = agent_nick
        self._service: Optional[ModelTrainingService] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_service(self, *, auto_subscribe: bool = False) -> ModelTrainingService:
        service = self._service
        if not isinstance(service, ModelTrainingService):
            service = ModelTrainingService(self._agent_nick, auto_subscribe=auto_subscribe)
            self._service = service
        elif auto_subscribe:
            # Ensure capture hooks are enabled when auto_subscribe is requested
            try:
                service.enable_workflow_capture()
            except Exception:  # pragma: no cover - defensive synchronisation
                logger.exception("Failed to enable workflow capture on training service")
        return service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def configure_capture(self, enable: bool) -> None:
        """Toggle workflow capture listeners for training."""

        service = self._resolve_service(auto_subscribe=enable)
        try:
            if enable:
                service.enable_workflow_capture()
            else:
                service.disable_workflow_capture()
        except Exception:  # pragma: no cover - defensive toggling
            logger.exception("Failed to update workflow capture state for training endpoint")

    def dispatch(self, *, force: bool = True, limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute queued training jobs and related refresh routines."""

        service = self._resolve_service()
        return service.dispatch_training_and_refresh(force=force, limit=limit)

    def queue_negotiation_learning(
        self,
        *,
        workflow_id: Optional[str],
        rfq_id: Optional[str],
        supplier_id: Optional[str],
        decision: Dict[str, Any],
        state: Dict[str, Any],
        awaiting_response: bool,
        supplier_reply_registered: bool,
    ) -> None:
        """Capture negotiation learnings for later dispatch processing."""

        service = self._resolve_service()
        service.queue_negotiation_learning(
            workflow_id=workflow_id,
            rfq_id=rfq_id,
            supplier_id=supplier_id,
            decision=decision,
            state=state,
            awaiting_response=awaiting_response,
            supplier_reply_registered=supplier_reply_registered,
        )

    def get_service(self) -> ModelTrainingService:
        """Expose the underlying service for advanced integrations."""

        return self._resolve_service()

    def reset(self) -> None:
        """Detach the underlying service instance."""

        self._service = None

