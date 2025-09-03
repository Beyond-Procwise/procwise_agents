import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


class ProcessRoutingService:
    """Service to log agent processing state into proc.routing table."""

    def __init__(self, agent_nick):
        self.agent_nick = agent_nick
        self.settings = agent_nick.settings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Best effort serialization for complex objects.

        Converts pandas objects, numpy types and dataclasses into plain
        Python data structures so they can be dumped to JSON. Falls back to
        ``str(obj)`` for unknown types.
        """
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.generic):
            return obj.item()
        if is_dataclass(obj):
            return asdict(obj)
        return str(obj)

    @classmethod
    def _safe_dumps(cls, data: Any) -> str:
        """JSON dump that tolerates DataFrames and other complex objects."""
        return json.dumps(data, default=cls._serialize)

    def log_process(
        self,
        process_name: str,
        process_details: Dict[str, Any],
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        created_by: Optional[str] = None,
        process_status: Optional[int] = None,
    ) -> Optional[int]:
        """Insert a process routing record and return the new ``process_id``.

        The ``proc.routing`` table now serves as both the catalogue of
        processes and the repository for individual run details.  An initial
        ``process_status`` may be supplied to seed the run state, while
        subsequent status updates are handled via :meth:`update_process_status`
        or :meth:`log_run_detail`.
        """

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO proc.routing
                            (process_name, process_details, created_by, user_id, user_name, process_status)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING process_id
                        """,
                        (
                            process_name,
                            self._safe_dumps(process_details),
                            created_by or self.settings.script_user,
                            user_id,
                            user_name,
                            process_status,
                        ),
                    )
                    process_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(
                        "Logged process %s with id %s", process_name, process_id
                    )
                    return process_id
        except Exception as exc:
            logger.error("Failed to log process %s: %s", process_name, exc)
            return None

    def get_process_details(self, process_id: int) -> Optional[Dict[str, Any]]:
        """Fetch the ``process_details`` blob for a given ``process_id``."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT process_details FROM proc.routing WHERE process_id = %s",
                        (process_id,),
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        value = row[0]
                        if isinstance(value, (str, bytes, bytearray)):
                            return json.loads(value)
                        return value
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to fetch process %s: %s", process_id, exc)
        return None

    def update_process_details(
        self,
        process_id: int,
        process_details: Dict[str, Any],
        modified_by: Optional[str] = None,
    ) -> None:
        """Persist updated ``process_details`` for a process."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_details = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (
                            self._safe_dumps(process_details),
                            modified_by or self.settings.script_user,
                            process_id,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Updated process %s details", process_id
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to update details for process %s: %s", process_id, exc
            )

    def update_process_status(self, process_id: int, status: int, modified_by: Optional[str] = None) -> None:
        """Update process status in ``proc.routing``.

        Only ``1`` (success) and ``-1`` (failure) are valid. Any other value
        is coerced to ``-1`` if negative or ``1`` if positive."""
        if status not in (1, -1):
            coerced = 1 if status > 0 else -1
            logger.warning(
                "Updated process %s to invalid status %s - coercing to %s",
                process_id, status, coerced,
            )
            status = coerced
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_status = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (status, modified_by or self.settings.script_user, process_id),
                    )
                    conn.commit()
                    logger.info(
                        "Updated process %s to status %s", process_id, status
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to update status for process %s: %s", process_id, exc
            )


    def log_action(
        self,
        process_id: int,
        agent_type: str,
        action_desc: Any,
        process_output: Optional[Any] = None,
        status: str = "running",
        action_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Insert or update an action record in ``proc.action``.

        If ``action_id`` is provided, the existing record is updated with the
        new ``process_output`` and ``status``. Otherwise a new record is
        inserted and its ``action_id`` returned.
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    if not action_id:
                        action_id = str(uuid.uuid4())
                        run_id = run_id or str(uuid.uuid4())
                        cursor.execute(
                            """
                            INSERT INTO proc.action (
                                action_id, process_id, run_id, agent_type,
                                process_output, status, action_desc, action_date
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                action_id,
                                str(process_id),
                                run_id,
                                agent_type,
                                self._safe_dumps(process_output) if process_output is not None else None,
                                status,
                                action_desc if isinstance(action_desc, str) else self._safe_dumps(action_desc),
                                datetime.utcnow(),
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            UPDATE proc.action
                            SET process_output = %s,
                                status = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE action_id = %s
                            """,
                            (
                                self._safe_dumps(process_output) if process_output is not None else None,
                                status,
                                action_id,
                            ),
                        )
                    conn.commit()
                    logger.info(
                        "Logged action %s for process %s with status %s",
                        action_id,
                        process_id,
                        status,
                    )
                    return action_id
        except Exception as exc:
            logger.error(
                "Failed to log action for process %s: %s", process_id, exc
            )
            return None

    def log_run_detail(
        self,
        process_id: int,
        process_status: str,
        run_id: Optional[str] = None,
        process_details: Optional[Dict[str, Any]] = None,
        process_start_ts: Optional[datetime] = None,
        process_end_ts: Optional[datetime] = None,
        triggered_by: Optional[str] = None,
    ) -> Optional[str]:
        """Update ``proc.routing`` with run metadata for an execution.

        The new schema stores a single record per process. Execution run
        information is persisted as ``raw_data`` and status as an integer.
        ``run_id`` is still returned so that related tables (e.g. ``proc.action``)
        can reference the run.
        """

        run_id = run_id or str(uuid.uuid4())
        process_start_ts = process_start_ts or datetime.utcnow()
        process_end_ts = process_end_ts or datetime.utcnow()
        duration = (
            process_end_ts - process_start_ts
            if process_end_ts and process_start_ts
            else None
        )
        # Coerce textual statuses to the integer mapping expected by the table
        status_int = 1 if str(process_status).lower() in ("1", "success", "completed") else -1

        raw_payload = {
            "run_id": run_id,
            "process_start_ts": process_start_ts.isoformat(),
            "process_end_ts": process_end_ts.isoformat(),
            "duration": duration.total_seconds() if duration else None,
            "triggered_by": triggered_by or self.settings.script_user,
        }

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE proc.routing
                        SET process_status = %s,
                            process_details = %s,
                            raw_data = %s,
                            modified_on = CURRENT_TIMESTAMP,
                            modified_by = %s
                        WHERE process_id = %s
                        """,
                        (
                            status_int,
                            self._safe_dumps(process_details) if process_details is not None else None,
                            self._safe_dumps(raw_payload),
                            triggered_by or self.settings.script_user,
                            process_id,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Recorded run %s for process %s with status %s",
                        run_id,
                        process_id,
                        status_int,
                    )
                    return run_id
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to log run for process %s: %s", process_id, exc
            )
            return None
