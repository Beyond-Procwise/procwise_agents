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
        process_status: int = 1,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a process routing record and return the new process_id."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO proc.routing
                            (process_name, process_details, process_status, created_by, user_id, user_name)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING process_id
                        """,
                        (
                            process_name,
                            self._safe_dumps(process_details),
                            process_status,
                            created_by or self.settings.script_user,
                            user_id,
                            user_name,
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

    # python
    def update_process_status(
            self,
            process_id: int,
            process_status: int,
            modified_by: Optional[str] = None,
    ) -> None:
        """Update the status of an existing process routing entry.

        Maps `process_status` to 1 (passed) or 0 (failed) before persisting.
        """
        try:
            status_value = 1 if bool(process_status) else 0
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
                        (
                            status_value,
                            modified_by or self.settings.script_user,
                            process_id,

                        ),
                    )
                    conn.commit()
                    logger.info(
                        "Updated process %s to status %s", process_id, process_status
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to update process %s: %s", process_id, exc
            )
