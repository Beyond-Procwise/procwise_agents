import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


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

    def __init__(self, agent_nick) -> None:
        self.agent_nick = agent_nick
        self._jobs: Dict[str, _ScheduledJob] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._relationship_scheduler = self._init_relationship_scheduler()
        self._training_service = self._init_training_service()
        self._register_default_jobs()
        self.start()

    @classmethod
    def ensure(cls, agent_nick) -> "BackendScheduler":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(agent_nick)
            else:
                cls._instance._update_agent(agent_nick)
        return cls._instance

    def _update_agent(self, agent_nick) -> None:
        if self.agent_nick is agent_nick:
            return
        self.agent_nick = agent_nick
        self._relationship_scheduler = self._init_relationship_scheduler()
        self._training_service = self._init_training_service()

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
        logger.debug(
            "BackendScheduler default jobs disabled; training and relationship refresh handled via explicit dispatch"
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
        service = self._training_service
        if service is None:
            service = self._init_training_service()
            self._training_service = service
        if service is None:
            return
        try:
            service.dispatch_due_jobs()
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

    def _init_training_service(self):
        try:
            from services.model_training_service import ModelTrainingService

            service = getattr(self.agent_nick, "model_training_service", None)
            settings = getattr(self.agent_nick, "settings", None)
            learning_enabled = bool(getattr(settings, "enable_learning", False))

            if not isinstance(service, ModelTrainingService):
                service = ModelTrainingService(
                    self.agent_nick, auto_subscribe=learning_enabled
                )
                setattr(self.agent_nick, "model_training_service", service)

            try:
                if learning_enabled:
                    service.enable_workflow_capture()
                else:
                    service.disable_workflow_capture()
            except Exception:
                logger.exception(
                    "Failed to synchronise workflow capture state on training service"
                )
            return service
        except Exception:
            logger.exception("Failed to initialise ModelTrainingService")
            return None
