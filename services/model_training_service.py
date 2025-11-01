from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from copy import deepcopy

from services.event_bus import get_event_bus
from models.supplier_ranking_trainer import SupplierRankingTrainer
from models.context_trainer import ContextTrainer, TrainingConfig
from services.rag_training_pipeline import LayeredRAGTrainer

logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Coordinate model training jobs derived from workflow outputs."""

    NON_BUSINESS_START: time = time(19, 0)  # 19:00 UTC
    NON_BUSINESS_END: time = time(7, 0)  # 07:00 UTC

    def __init__(self, agent_nick: Any, *, auto_subscribe: bool = False) -> None:
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
        self._event_bus = get_event_bus()
        self._subscription_registered = False
        self._supplier_trainer = SupplierRankingTrainer()
        self._learning_repo = getattr(agent_nick, "learning_repository", None)
        self._pending_negotiation_learnings: List[Dict[str, Any]] = []
        self._context_trainer: Optional[ContextTrainer] = None
        self._queued_context_workflows: set[str] = set()
        self._rag_trainer = LayeredRAGTrainer(agent_nick)
        if auto_subscribe:
            self.enable_workflow_capture()

    def _get_relationship_scheduler(self):
        try:
            from services.supplier_relationship_service import SupplierRelationshipScheduler

            scheduler = getattr(self.agent_nick, "supplier_relationship_scheduler", None)
            if not isinstance(scheduler, SupplierRelationshipScheduler):
                scheduler = SupplierRelationshipScheduler(self.agent_nick)
                setattr(self.agent_nick, "supplier_relationship_scheduler", scheduler)
            return scheduler
        except Exception:
            logger.exception("Failed to initialise SupplierRelationshipScheduler for training dispatch")
            return None

    def _resolve_context_trainer(self) -> ContextTrainer:
        trainer = self._context_trainer
        if not isinstance(trainer, ContextTrainer):
            data_dir = getattr(
                self.agent_nick.settings,
                "context_training_data_dir",
                TrainingConfig().data_dir,
            )
            output_dir = getattr(
                self.agent_nick.settings,
                "context_model_output_dir",
                TrainingConfig().output_dir,
            )
            base_cfg = TrainingConfig(data_dir=data_dir, output_dir=output_dir)
            trainer = ContextTrainer(base_cfg)
            self._context_trainer = trainer
        return trainer

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

    def _ensure_supplier_model_table(self) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.supplier_ranking_model (
                            model_id BIGSERIAL PRIMARY KEY,
                            workflow_id TEXT,
                            metrics TEXT[],
                            label_column TEXT,
                            policy_weights JSONB,
                            learned_weights JSONB,
                            trained_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                conn.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to ensure supplier ranking model table")

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def enable_workflow_capture(self) -> None:
        """Register listeners that queue training jobs from workflow outcomes."""

        if self._subscription_registered:
            return
        try:
            self._event_bus.subscribe("workflow.complete", self._handle_workflow_complete)
            self._subscription_registered = True
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to subscribe to workflow completion events")

    def disable_workflow_capture(self) -> None:
        """Detach workflow listeners so training never hooks into live flows."""

        if not self._subscription_registered:
            return
        try:
            self._event_bus.unsubscribe("workflow.complete", self._handle_workflow_complete)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to unsubscribe from workflow completion events")
        finally:
            self._subscription_registered = False

    def _handle_workflow_complete(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        if event.get("status") != "completed":
            return
        workflow_id = event.get("workflow_id")
        if not workflow_id:
            return
        result = event.get("result")
        if not isinstance(result, dict):
            return
        workflow_name = event.get("workflow_name") or ""
        try:
            self.record_workflow_outcome(workflow_name, workflow_id, result)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to handle workflow completion for training queue")

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

    def dispatch_due_jobs(
        self,
        moment: Optional[datetime] = None,
        *,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not force and not self._is_non_business_hour(moment):
            logger.info("Skipping training dispatch; current time within business hours")
            return []

        self._ensure_queue_table()
        jobs: List[Dict[str, Any]] = []
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    params: List[Any]
                    if force:
                        query = (
                            """
                            SELECT job_id, workflow_id, agent_slug, policy_id, payload
                              FROM proc.agent_training_queue
                             WHERE status = 'pending'
                             ORDER BY scheduled_for, job_id
                            """
                        )
                        params = []
                    else:
                        query = (
                            """
                            SELECT job_id, workflow_id, agent_slug, policy_id, payload
                              FROM proc.agent_training_queue
                             WHERE status = 'pending' AND scheduled_for <= %s
                             ORDER BY scheduled_for, job_id
                            """
                        )
                        params = [moment]
                    if limit is not None:
                        try:
                            limit_value = int(limit)
                        except Exception:
                            limit_value = 0
                        if limit_value > 0:
                            query += " LIMIT %s"
                            params.append(limit_value)
                    cur.execute(query, params)  # type: ignore[arg-type]
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

    def dispatch_training_and_refresh(
        self,
        *,
        force: bool = True,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Dispatch model training jobs and refresh supplier relationships together."""

        training_jobs = self.dispatch_due_jobs(force=force, limit=limit)
        relationship_jobs: List[Dict[str, Any]] = []
        workflow_context: List[Dict[str, Any]] = []
        scheduler = self._get_relationship_scheduler()
        if scheduler is not None:
            try:
                relationship_jobs = scheduler.dispatch_due_jobs(force=force)
                if not relationship_jobs:
                    summary = scheduler.run_immediate_refresh()
                    if summary:
                        relationship_jobs = [summary]
            except Exception:
                logger.exception("Supplier relationship refresh during dispatch failed")
        if self._learning_repo is not None:
            try:
                workflow_context = self._learning_repo.get_recent_learnings(limit=25)
            except Exception:
                logger.exception("Failed to gather workflow learning context during dispatch")
        negotiation_learnings: List[Dict[str, Any]] = []
        if self._pending_negotiation_learnings:
            pending_items = self._pending_negotiation_learnings
            self._pending_negotiation_learnings = []
            requeue: List[Dict[str, Any]] = []
            for item in pending_items:
                payload = {
                    "workflow_id": item.get("workflow_id"),
                    "rfq_id": item.get("rfq_id"),
                    "supplier_id": item.get("supplier_id"),
                    "decision": deepcopy(item.get("decision") or {}),
                    "state": deepcopy(item.get("state") or {}),
                    "awaiting_response": item.get("awaiting_response", False),
                    "supplier_reply_registered": item.get(
                        "supplier_reply_registered", False
                    ),
                }
                status = "skipped"
                point_id: Optional[str] = None
                if self._learning_repo is not None:
                    try:
                        point_id = self._learning_repo.record_negotiation_learning(**payload)
                        status = "recorded"
                    except Exception:
                        logger.exception(
                            "Failed to persist negotiation learning for %s/%s",
                            payload.get("rfq_id"),
                            payload.get("supplier_id"),
                        )
                        status = "failed"
                        requeue.append(item)
                else:
                    requeue.append(item)
                negotiation_learnings.append(
                    {
                        "rfq_id": payload.get("rfq_id"),
                        "supplier_id": payload.get("supplier_id"),
                        "workflow_id": payload.get("workflow_id"),
                        "round": payload["decision"].get("round"),
                        "status": status,
                        "point_id": point_id,
                    }
                )
            if requeue:
                self._pending_negotiation_learnings.extend(requeue)
        return {
            "training_jobs": training_jobs,
            "relationship_jobs": relationship_jobs,
            "workflow_context": workflow_context,
            "negotiation_learnings": negotiation_learnings,
        }

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
        """Store a negotiation learning snapshot for later dispatch."""

        record = {
            "workflow_id": workflow_id,
            "rfq_id": rfq_id,
            "supplier_id": supplier_id,
            "decision": deepcopy(decision or {}),
            "state": deepcopy(state or {}),
            "awaiting_response": awaiting_response,
            "supplier_reply_registered": supplier_reply_registered,
        }
        self._pending_negotiation_learnings.append(record)

    def record_workflow_outcome(
        self, workflow_name: str, workflow_id: str, result: Dict[str, Any]
    ) -> None:
        if not isinstance(result, dict):
            return

        if self._learning_repo is not None:
            try:
                self._learning_repo.record_workflow_learning(
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    result=result,
                )
            except Exception:
                logger.exception(
                    "Failed to capture workflow learning context for %s", workflow_id
                )

        snapshot = result.get("context_snapshot") or {}
        conversation_history = result.get("conversation_history")
        workflow_has_context = bool(conversation_history or snapshot)
        if workflow_has_context and workflow_id not in self._queued_context_workflows:
            data_dir = getattr(
                self.agent_nick.settings,
                "context_training_data_dir",
                TrainingConfig().data_dir,
            )
            output_dir = getattr(
                self.agent_nick.settings,
                "context_model_output_dir",
                TrainingConfig().output_dir,
            )
            payload = {
                "data_dir": data_dir,
                "output_dir": output_dir,
            }
            self.enqueue_training_job(
                workflow_id=workflow_id,
                agent_slug="context_trainer",
                policy_id=None,
                payload=payload,
            )
            self._queued_context_workflows.add(workflow_id)

        if workflow_name == "supplier_ranking":
            snapshot = result.get("training_snapshot")
            if isinstance(snapshot, dict) and snapshot.get("rows"):
                payload = {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "training_snapshot": snapshot,
                }
                self.enqueue_training_job(
                    workflow_id=workflow_id,
                    agent_slug="supplier_ranking",
                    policy_id=None,
                    payload=payload,
                )

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

    def _persist_supplier_model(
        self,
        workflow_id: Optional[str],
        metrics: List[Any],
        label_column: str,
        policy_weights: Dict[str, Any],
        learned_weights: Dict[str, float],
    ) -> None:
        self._ensure_supplier_model_table()
        try:
            serialised_policy = json.dumps(policy_weights or {}, default=str)
            serialised_learned = json.dumps(learned_weights or {}, default=float)
        except TypeError:
            serialised_policy = json.dumps({"policy_weights": str(policy_weights)})
            serialised_learned = json.dumps({"learned_weights": str(learned_weights)})
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.supplier_ranking_model
                            (workflow_id, metrics, label_column, policy_weights, learned_weights)
                        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                        """,
                        (
                            workflow_id,
                            list(metrics),
                            label_column,
                            serialised_policy,
                            serialised_learned,
                        ),
                    )
                conn.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to persist supplier ranking model weights")

    def _run_training_job(self, job: Dict[str, Any]) -> None:
        agent_slug = job.get("agent_slug")
        payload = job.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        job["payload"] = payload
        if self._learning_repo is not None:
            try:
                context_snapshot = self._learning_repo.build_context_snapshot(
                    workflow_id=job.get("workflow_id")
                )
            except Exception:
                logger.exception("Failed to build learning context snapshot for job %s", job.get("job_id"))
                context_snapshot = None
            if context_snapshot:
                payload = dict(payload)
                payload["learning_context"] = context_snapshot
                job["payload"] = payload
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
        elif agent_slug == "supplier_ranking":
            snapshot = payload.get("training_snapshot") or {}
            metrics = snapshot.get("metrics") or []
            rows = snapshot.get("rows") or []
            label_column = snapshot.get("label_column") or self._supplier_trainer.config.label_column
            learned = self._supplier_trainer.train_from_records(rows, metrics, label_column)
            if learned:
                policy_weights = snapshot.get("policy_weights") or {}
                self._persist_supplier_model(
                    job.get("workflow_id"), metrics, label_column, policy_weights, learned
                )
                logger.info(
                    "Supplier ranking model trained with metrics: %s",
                    ", ".join(sorted(learned.keys())),
                )
            else:
                logger.info(
                    "Supplier ranking training skipped due to insufficient labelled samples"
                )
        elif agent_slug == "rag_agent":
            summary = self._rag_trainer.train(payload)
            payload["training_result"] = summary
            logger.info("RAG agent layered training summary: %s", summary)
        elif agent_slug == "context_trainer":
            trainer = self._resolve_context_trainer()
            overrides = {
                key: payload.get(key)
                for key in (
                    "data_dir",
                    "output_dir",
                    "model_name",
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "max_length",
                )
                if payload.get(key) is not None
            }
            summary = trainer.train(**overrides)
            payload["training_result"] = summary
            logger.info("Context trainer executed: %s", summary)
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
