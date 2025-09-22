"""Model training scheduler integrating workflow outputs with policy definitions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Coordinate model training jobs derived from workflow outputs."""

    NON_BUSINESS_START: time = time(19, 0)  # 19:00 UTC
    NON_BUSINESS_END: time = time(7, 0)  # 07:00 UTC

    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick
        self.policy_engine = getattr(agent_nick, "policy_engine", None)
        if self.policy_engine is None:
            try:  # pragma: no cover - defensive import
                from engines.policy_engine import PolicyEngine

                self.policy_engine = PolicyEngine(agent_nick)
                setattr(agent_nick, "policy_engine", self.policy_engine)
            except Exception:
                logger.exception("Failed to initialise PolicyEngine for model training")
                self.policy_engine = None

    # ------------------------------------------------------------------
    # Database utilities
    # ------------------------------------------------------------------
    def _ensure_queue_table(self) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.agent_training_queue (
                            job_id BIGSERIAL PRIMARY KEY,
                            workflow_id TEXT,
                            agent_slug TEXT NOT NULL,
                            policy_id TEXT,
                            payload JSONB,
                            scheduled_for TIMESTAMPTZ NOT NULL,
                            status TEXT NOT NULL DEFAULT 'pending',
                            result_message TEXT,
                            created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                conn.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to ensure agent training queue table")

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------
    def _is_non_business_hour(self, moment: datetime) -> bool:
        ts = moment.astimezone(timezone.utc)
        current = ts.time()
        start = self.NON_BUSINESS_START
        end = self.NON_BUSINESS_END
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _next_non_business_window(self, moment: Optional[datetime] = None) -> datetime:
        moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self._is_non_business_hour(moment):
            return moment
        start = moment.replace(
            hour=self.NON_BUSINESS_START.hour,
            minute=self.NON_BUSINESS_START.minute,
            second=0,
            microsecond=0,
        )
        if moment.time() > self.NON_BUSINESS_START:
            start = start + timedelta(days=1)
        return start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enqueue_training_job(
        self,
        *,
        workflow_id: str,
        agent_slug: str,
        policy_id: Optional[str],
        payload: Dict[str, Any],
        scheduled_for: Optional[datetime] = None,
    ) -> None:
        self._ensure_queue_table()
        schedule_ts = scheduled_for or self._next_non_business_window()
        try:
            serialised = json.dumps(payload, default=str)
        except TypeError:  # pragma: no cover - defensive serialisation
            serialised = json.dumps({"payload": str(payload)})

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.agent_training_queue
                            (workflow_id, agent_slug, policy_id, payload, scheduled_for)
                        VALUES (%s, %s, %s, %s::jsonb, %s)
                        """,
                        (workflow_id, agent_slug, policy_id, serialised, schedule_ts),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to enqueue model training job")

    def dispatch_due_jobs(self, moment: Optional[datetime] = None) -> List[Dict[str, Any]]:
        moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not self._is_non_business_hour(moment):
            logger.info("Skipping training dispatch; current time within business hours")
            return []

        self._ensure_queue_table()
        jobs: List[Dict[str, Any]] = []
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT job_id, workflow_id, agent_slug, policy_id, payload
                        FROM proc.agent_training_queue
                        WHERE status = 'pending' AND scheduled_for <= %s
                        ORDER BY scheduled_for, job_id
                        """,
                        (moment,),
                    )
                    rows = cur.fetchall() or []
                for row in rows:
                    job_payload = {
                        "job_id": row[0],
                        "workflow_id": row[1],
                        "agent_slug": row[2],
                        "policy_id": row[3],
                        "payload": self._decode_payload(row[4]),
                    }
                    jobs.append(job_payload)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load due training jobs")
            return []

        for job in jobs:
            try:
                self._run_training_job(job)
                self._update_job_status(job["job_id"], "completed", "Training executed")
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Training job %s failed", job["job_id"])
                self._update_job_status(job["job_id"], "failed", str(exc))
        return jobs

    def record_workflow_outcome(
        self, workflow_name: str, workflow_id: str, result: Dict[str, Any]
    ) -> None:
        if not isinstance(result, dict):
            return
        opportunities = result.get("opportunities")
        if not isinstance(opportunities, dict):
            return
        policy_opps = opportunities.get("policy_opportunities", {})
        if not policy_opps:
            return
        metadata = opportunities.get("policy_metadata", {})
        data_profile = opportunities.get("data_profile")
        timestamp = datetime.utcnow().isoformat() + "Z"

        for display, findings in policy_opps.items():
            if not findings:
                continue
            policy_meta = metadata.get(display, {})
            policy_id = policy_meta.get("policy_id")
            payload = {
                "policy_display": display,
                "policy_metadata": policy_meta,
                "findings": findings,
                "workflow_name": workflow_name,
                "data_profile": data_profile,
                "recorded_at": timestamp,
            }
            self.enqueue_training_job(
                workflow_id=workflow_id,
                agent_slug="opportunity_miner",
                policy_id=str(policy_id) if policy_id is not None else None,
                payload=payload,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decode_payload(self, payload: Any) -> Dict[str, Any]:
        if payload in (None, ""):
            return {}
        if isinstance(payload, dict):
            return dict(payload)
        try:
            return json.loads(payload)
        except Exception:  # pragma: no cover - defensive parsing
            logger.debug("Unable to decode training payload: %s", payload)
            return {"raw": str(payload)}

    def _run_training_job(self, job: Dict[str, Any]) -> None:
        agent_slug = job.get("agent_slug")
        payload = job.get("payload") or {}
        policy_id = job.get("policy_id")
        logger.info(
            "Running model training job %s for %s (policy %s)",
            job.get("job_id"),
            agent_slug,
            policy_id,
        )
        if agent_slug == "opportunity_miner":
            query_engine = getattr(self.agent_nick, "query_engine", None)
            if query_engine is not None and hasattr(query_engine, "train_procurement_context"):
                query_engine.train_procurement_context()
            else:
                logger.info("No query engine available for opportunity miner training")
        else:
            logger.info("No training routine configured for agent '%s'", agent_slug)
        logger.debug("Training payload: %s", payload)

    def _update_job_status(self, job_id: int, status: str, message: Optional[str]) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.agent_training_queue
                           SET status = %s,
                               result_message = %s,
                               updated_on = NOW()
                         WHERE job_id = %s
                        """,
                        (status, message, job_id),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to update training job %s status", job_id)
