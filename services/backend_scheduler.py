import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from services.email_watcher_service import EmailWatcherService

from services.model_training_endpoint import ModelTrainingEndpoint
from utils.gpu import configure_gpu

logger: logging.Logger = logging.getLogger(__name__)


class _ScheduledJob:
    def __init__(
        self,
        name: str,
        runner: Callable[[], None],
        interval: timedelta,
        initial_delay: Optional[timedelta] = None,
        *,
        one_shot: bool = False,
    ) -> None:
        self.name = name
        self.runner = runner
        self.interval = interval
        delay = initial_delay or timedelta(0)
        self.next_run = datetime.now(timezone.utc) + delay
        self._lock = threading.Lock()
        self.one_shot = one_shot

    def due(self, moment: datetime) -> bool:
        with self._lock:
            return moment >= self.next_run

    def mark_executed(self, executed_at: datetime) -> None:
        with self._lock:
            self.next_run = executed_at + self.interval


class BackendScheduler:
    _instance: Optional["BackendScheduler"] = None
    _instance_lock = threading.Lock()
    _poll_seconds: float = 60.0

    def __init__(
        self,
        agent_nick,
        *,
        training_endpoint: Optional[ModelTrainingEndpoint] = None,
    ) -> None:
        self.agent_nick = agent_nick
        self._jobs: Dict[str, _ScheduledJob] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._relationship_scheduler = self._init_relationship_scheduler()
        self._training_endpoint = training_endpoint
        self._email_watcher_service: Optional[EmailWatcherService] = None
        self._email_watcher_lock = threading.Lock()
        self._register_default_jobs()
        self.start()
        self._ensure_email_watcher_service()

    @classmethod
    def ensure(
        cls,
        agent_nick,
        *,
        training_endpoint: Optional[ModelTrainingEndpoint] = None,
    ) -> "BackendScheduler":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    agent_nick, training_endpoint=training_endpoint
                )
            else:
                cls._instance._update_agent(
                    agent_nick, training_endpoint=training_endpoint
                )
        return cls._instance

    def _update_agent(
        self,
        agent_nick,
        *,
        training_endpoint: Optional[ModelTrainingEndpoint] = None,
    ) -> None:
        if self.agent_nick is agent_nick:
            if training_endpoint is not None:
                self._training_endpoint = training_endpoint
            return
        self.agent_nick = agent_nick
        self._relationship_scheduler = self._init_relationship_scheduler()
        if training_endpoint is not None:
            self._training_endpoint = training_endpoint

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="procwise-backend-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        with self._email_watcher_lock:
            watcher = self._email_watcher_service
        if watcher is not None:
            try:
                watcher.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop email watcher service")

    def register_job(
        self,
        name: str,
        runner: Callable[[], None],
        interval: timedelta,
        initial_delay: Optional[timedelta] = None,
        *,
        one_shot: bool = False,
    ) -> None:
        job = _ScheduledJob(
            name,
            runner,
            interval,
            initial_delay,
            one_shot=one_shot,
        )
        with self._lock:
            self._jobs[name] = job

    def submit_once(
        self,
        name: str,
        runner: Callable[[], None],
        *,
        initial_delay: Optional[timedelta] = None,
    ) -> None:
        """Schedule ``runner`` to execute once in the background."""

        self.register_job(
            name,
            runner,
            interval=timedelta(days=365 * 100),
            initial_delay=initial_delay,
            one_shot=True,
        )

    def _register_default_jobs(self) -> None:
        training_delay = timedelta(minutes=15)
        self.register_job(
            "context-training-dispatch",
            self._run_model_training,
            interval=timedelta(hours=6),
            initial_delay=training_delay,
        )

    def _ensure_email_watcher_service(self) -> EmailWatcherService:
        with self._email_watcher_lock:
            if self._email_watcher_service is None:
                self._email_watcher_service = EmailWatcherService()
            service = self._email_watcher_service
        service.start()
        return service

    def notify_email_dispatch(self, workflow_id: str) -> None:
        workflow_key = (workflow_id or "").strip()
        if not workflow_key:
            return
        service = self._ensure_email_watcher_service()
        try:
            service.notify_workflow(workflow_key)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to notify email watcher for workflow %s", workflow_key
            )

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._poll_seconds):
            now = datetime.now(timezone.utc)
            jobs_snapshot = None
            with self._lock:
                jobs_snapshot = list(self._jobs.values())
            for job in jobs_snapshot:
                if not job.due(now):
                    continue
                self._execute_job(job)

    def _deregister_job(self, name: str) -> None:
        with self._lock:
            self._jobs.pop(name, None)

    def _execute_job(self, job: _ScheduledJob) -> None:
        try:
            configure_gpu()
            job.runner()
        except Exception:
            logger.exception("Backend job %s failed", job.name)
        finally:
            executed_at = datetime.now(timezone.utc)
            job.mark_executed(executed_at)
            if job.one_shot:
                self._deregister_job(job.name)

    def _run_supplier_refresh(self) -> None:
        scheduler = self._relationship_scheduler
        if scheduler is None:
            scheduler = self._init_relationship_scheduler()
            self._relationship_scheduler = scheduler
        if scheduler is None:
            return
        try:
            scheduler.schedule_daily_refresh()
            scheduler.dispatch_due_jobs()
        except Exception:
            logger.exception("Supplier relationship refresh dispatch failed")

    def _run_model_training(self) -> None:
        endpoint = self._resolve_training_endpoint()
        if endpoint is None:
            return
        try:
            endpoint.dispatch(force=False)
        except Exception:
            logger.exception("Model training dispatch failed")

    def _init_relationship_scheduler(self):
        try:
            from services.supplier_relationship_service import SupplierRelationshipScheduler

            scheduler = getattr(self.agent_nick, "supplier_relationship_scheduler", None)
            if scheduler is None:
                scheduler = SupplierRelationshipScheduler(self.agent_nick)
                setattr(self.agent_nick, "supplier_relationship_scheduler", scheduler)
            return scheduler
        except Exception:
            logger.exception("Failed to initialise SupplierRelationshipScheduler")
            return None

    def _resolve_training_endpoint(self) -> Optional[ModelTrainingEndpoint]:
        endpoint = self._training_endpoint
        if isinstance(endpoint, ModelTrainingEndpoint):
            return endpoint
        try:
            endpoint = ModelTrainingEndpoint(self.agent_nick)
            self._training_endpoint = endpoint
            return endpoint
        except Exception:  # pragma: no cover - defensive import/initialisation
            logger.exception("Failed to initialise ModelTrainingEndpoint")
            return None
