from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from config.settings import settings  # type: ignore
except Exception:  # pragma: no cover - settings import is best effort
    settings = None  # type: ignore


class _ColumnDescriptor:
    """Minimal shim replicating psycopg2 cursor description entries."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCursor:
    def __init__(self, store: "_FakePostgresStore") -> None:
        self._store = store
        self._results: List[Tuple[Any, ...]] = []
        self.description: Optional[List[_ColumnDescriptor]] = None

    # -- public cursor API -------------------------------------------------
    def execute(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        statements = [stmt.strip() for stmt in query.split(";") if stmt.strip()]
        if len(statements) > 1:
            for stmt in statements:
                self.execute(stmt, params if len(statements) == 1 else None)
            return

        statement = statements[0] if statements else query.strip()
        self._results = []
        self.description = None
        self._dispatch(statement, params or tuple())

    def executemany(
        self, query: str, param_list: Iterable[Tuple[Any, ...]]
    ) -> None:
        for params in param_list:
            self.execute(query, params)

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        return self._results[0] if self._results else None

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._results)

    def close(self) -> None:  # pragma: no cover - noop for compatibility
        return

    # -- query dispatch helpers -------------------------------------------
    def _dispatch(self, statement: str, params: Tuple[Any, ...]) -> None:
        upper_stmt = statement.upper()

        if upper_stmt.startswith("SELECT TO_REGCLASS"):
            exists = hasattr(self._store, "workflow_email_tracking")
            value = "proc.workflow_email_tracking" if exists else None
            self._results = [(value,)]
            self.description = [_ColumnDescriptor("to_regclass")]
            return

        if upper_stmt.startswith("CREATE SCHEMA"):
            return
        if upper_stmt.startswith("CREATE TABLE") or upper_stmt.startswith(
            "CREATE INDEX"
        ) or upper_stmt.startswith("CREATE UNIQUE INDEX"):
            self._store.ensure_tables()
            return
        if upper_stmt.startswith("WITH RANKED AS"):
            # Deduplication CTE is a no-op in the in-memory store.
            return
        if upper_stmt.startswith("WITH DUPLICATES AS") and "DELETE FROM PROC.SUPPLIER_RESPONSE" in upper_stmt:
            # The in-memory store never creates duplicate response rows so the
            # clean-up CTE can be safely ignored during tests.
            return
        if "INFORMATION_SCHEMA.COLUMNS" in upper_stmt:
            if params:
                schema, table, column = params
                exists = self._store.column_exists(schema, table, column)
                self._results = [(1,)] if exists else []
                self.description = [_ColumnDescriptor("exists")]
            else:
                schema = "proc"
                table = "workflow_email_tracking"
                columns = self._store.list_columns(schema, table)
                self._results = [(name,) for name in columns]
                self.description = [_ColumnDescriptor("column_name")]
            return
        if (
            "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in upper_stmt
            and "PRIMARY KEY" in upper_stmt
        ):
            self._results = [
                ("workflow_email_tracking_pkey", "workflow_id", 1),
                ("workflow_email_tracking_pkey", "unique_id", 2),
            ]
            self.description = [
                _ColumnDescriptor("constraint_name"),
                _ColumnDescriptor("column_name"),
                _ColumnDescriptor("ordinal_position"),
            ]
            return
        if upper_stmt.startswith("ALTER TABLE"):
            # schema migrations are no-ops for the in-memory store because
            # columns are eagerly present.
            self._store.ensure_tables()
            return

        # Data manipulation commands for supplier responses
        if upper_stmt.startswith("INSERT INTO PROC.SUPPLIER_RESPONSE"):
            column_order = [
                "workflow_id",
                "supplier_id",
                "supplier_email",
                "rfq_id",
                "unique_id",
                "response_text",
                "response_body",
                "body_html",
                "response_message_id",
                "response_subject",
                "response_from",
                "response_date",
                "original_message_id",
                "original_subject",
                "match_confidence",
                "match_score",
                "price",
                "currency",
                "payment_terms",
                "warranty",
                "validity",
                "exceptions",
                "lead_time",
                "response_time",
                "received_time",
                "match_evidence",
                "matched_on",
                "raw_headers",
                "tables",
                "attachments",
                "processed",
            ]
            values = {key: params[idx] if idx < len(params) else None for idx, key in enumerate(column_order)}
            workflow_id = values.get("workflow_id")
            unique_id = values.get("unique_id")
            if workflow_id is None or unique_id is None:
                return
            self._store.upsert_supplier_response(
                workflow_id,
                unique_id,
                values,
            )
            return

        if upper_stmt.startswith("UPDATE PROC.SUPPLIER_RESPONSE SET SUPPLIER_ID"):
            update_keys = [
                "supplier_id",
                "supplier_email",
                "rfq_id",
                "response_text",
                "response_body",
                "body_html",
                "response_subject",
                "response_from",
                "response_date",
                "original_message_id",
                "original_subject",
                "match_confidence",
                "match_score",
                "price",
                "currency",
                "payment_terms",
                "warranty",
                "validity",
                "exceptions",
                "lead_time",
                "response_time",
                "received_time",
                "match_evidence",
                "matched_on",
                "raw_headers",
                "tables",
                "attachments",
                "processed",
                "workflow_id",
                "response_message_id",
            ]
            values = {key: params[idx] if idx < len(params) else None for idx, key in enumerate(update_keys)}
            workflow_id = values.pop("workflow_id")
            response_message_id = values.pop("response_message_id")
            if workflow_id is None or response_message_id is None:
                return
            self._store.update_supplier_response_by_message_id(
                workflow_id,
                response_message_id,
                values,
            )
            return

        if upper_stmt.startswith("UPDATE PROC.SUPPLIER_RESPONSE SET PROCESSED"):
            workflow_id, ids = params
            self._store.mark_supplier_responses_processed(workflow_id, list(ids))
            return

        if upper_stmt.startswith("DELETE FROM PROC.SUPPLIER_RESPONSE"):
            (workflow_id,) = params
            self._store.delete_supplier_responses(workflow_id, None)
            return

        if upper_stmt.startswith("INSERT INTO PROC.WORKFLOW_LIFECYCLE"):
            column_segment = statement.split("(", 1)[1].split(")", 1)[0]
            columns = [col.strip().strip('"') for col in column_segment.split(",")]
            values = {col: params[idx] if idx < len(params) else None for idx, col in enumerate(columns)}
            workflow_id = values.get("workflow_id")
            payload = {key: value for key, value in values.items() if key != "workflow_id"}
            if workflow_id is not None:
                self._store.upsert_workflow_lifecycle(str(workflow_id), payload)
            return

        if upper_stmt.startswith("UPDATE PROC.WORKFLOW_LIFECYCLE SET"):
            set_clause = statement.split("SET", 1)[1]
            set_part, _, _ = set_clause.partition("WHERE")
            assignments = [item.strip() for item in set_part.split(",") if item.strip()]
            columns = [assign.split("=")[0].strip().strip('"') for assign in assignments]
            updates = {columns[idx]: params[idx] for idx in range(len(columns))}
            workflow_id = params[len(columns)] if len(params) > len(columns) else None
            if workflow_id is not None:
                self._store.upsert_workflow_lifecycle(str(workflow_id), updates)
            return

        if upper_stmt.startswith("DELETE FROM PROC.WORKFLOW_LIFECYCLE"):
            if params:
                workflow_id = params[0]
                if workflow_id is not None:
                    self._store.delete_workflow_lifecycle(str(workflow_id))
            return

        if upper_stmt.startswith("SELECT") and "FROM PROC.WORKFLOW_LIFECYCLE" in upper_stmt:
            select_part = statement.split("SELECT", 1)[1].split("FROM", 1)[0]
            columns = [col.strip().strip('"') for col in select_part.split(",")]
            workflow_id = params[0] if params else None
            row = None
            if workflow_id is not None:
                row = self._store.fetch_workflow_lifecycle(str(workflow_id))
            if row is None:
                self._results = []
            else:
                payload: List[Any] = []
                for column in columns:
                    value = row.get(column)
                    if column == "metadata" and isinstance(value, dict):
                        payload.append(json.dumps(value))
                    else:
                        payload.append(value)
                self._results = [tuple(payload)]
            self.description = [_ColumnDescriptor(col) for col in columns]
            return

        if upper_stmt.startswith("SELECT WORKFLOW_ID, SUPPLIER_ID") and (
            "FROM PROC.SUPPLIER_RESPONSE" in upper_stmt
        ):
            (workflow_id,) = params
            pending_only = "COALESCE(PROCESSED" in upper_stmt
            rows = self._store.fetch_supplier_responses(
                workflow_id, pending_only=pending_only
            )
            columns = [
                "workflow_id",
                "supplier_id",
                "supplier_email",
                "rfq_id",
                "unique_id",
                "response_text",
                "response_body",
                "body_html",
                "response_message_id",
                "response_subject",
                "response_from",
                "response_date",
                "original_message_id",
                "original_subject",
                "match_confidence",
                "match_score",
                "match_evidence",
                "matched_on",
                "raw_headers",
                "price",
                "currency",
                "payment_terms",
                "warranty",
                "validity",
                "exceptions",
                "lead_time",
                "response_time",
                "received_time",
                "tables",
                "attachments",
                "processed",
            ]
            self.description = [
                _ColumnDescriptor(col) for col in columns
            ]
            self._results = [
                tuple(row.get(col) for col in columns)
                for row in rows
            ]
            return

        if upper_stmt.startswith("SELECT UNIQUE_ID FROM PROC.SUPPLIER_RESPONSE"):
            workflow_id, message_id = params
            unique_id = self._store.lookup_unique_by_message_id(workflow_id, message_id)
            self.description = [_ColumnDescriptor("unique_id")]
            if unique_id is None:
                self._results = []
            else:
                self._results = [(unique_id,)]
            return

        # Workflow email tracking commands
        if upper_stmt.startswith("INSERT INTO PROC.WORKFLOW_EMAIL_TRACKING"):
            (
                workflow_id,
                unique_id,
                dispatch_key,
                supplier_id,
                supplier_email,
                recipient_emails,
                message_id,
                subject,
                dispatched_at,
                responded_at,
                response_message_id,
                matched,
                thread_headers,
            ) = params
            self._store.upsert_workflow_tracking(
                workflow_id,
                unique_id,
                {
                    "workflow_id": workflow_id,
                    "unique_id": unique_id,
                    "dispatch_key": dispatch_key,
                    "supplier_id": supplier_id,
                    "supplier_email": supplier_email,
                    "recipient_emails": recipient_emails,
                    "message_id": message_id,
                    "subject": subject,
                    "dispatched_at": dispatched_at,
                    "responded_at": responded_at,
                    "response_message_id": response_message_id,
                    "matched": matched,
                    "thread_headers": thread_headers,
                },
            )
            return

        if upper_stmt.startswith("SELECT DISTINCT WORKFLOW_ID") or (
            upper_stmt.startswith("SELECT WORKFLOW_ID FROM PROC.WORKFLOW_EMAIL_TRACKING")
            and "GROUP BY" in upper_stmt
        ):
            rows = self._store.distinct_active_workflow_ids()
            self.description = [_ColumnDescriptor("workflow_id")]
            self._results = [(workflow_id,) for workflow_id in rows]
            return

        if upper_stmt.startswith("SELECT WORKFLOW_ID, UNIQUE_ID") and (
            "FROM PROC.WORKFLOW_EMAIL_TRACKING" in upper_stmt
        ):
            (workflow_id,) = params
            rows = self._store.fetch_workflow_tracking(workflow_id)
            columns = [
                "workflow_id",
                "unique_id",
                "dispatch_key",
                "supplier_id",
                "supplier_email",
                "recipient_emails",
                "message_id",
                "subject",
                "dispatched_at",
                "responded_at",
                "response_message_id",
                "matched",
                "thread_headers",
            ]
            self.description = [
                _ColumnDescriptor(col) for col in columns
            ]
            self._results = [
                tuple(row.get(col) for col in columns)
                for row in rows
            ]
            return

        if upper_stmt.startswith("UPDATE PROC.WORKFLOW_EMAIL_TRACKING SET RESPONDED_AT"):
            responded_at, response_message_id, workflow_id, unique_id = params
            self._store.mark_workflow_response(
                workflow_id,
                unique_id,
                responded_at,
                response_message_id,
            )
            return

        if upper_stmt.startswith("DELETE FROM PROC.WORKFLOW_EMAIL_TRACKING"):
            (workflow_id,) = params
            self._store.delete_workflow_rows(workflow_id)
            return

        if upper_stmt.startswith("INSERT INTO PROC.SUPPLIER_RISK_SIGNALS"):
            (
                supplier_id,
                signal_type,
                severity,
                source,
                payload,
                occurred_at,
            ) = params
            self._store.insert_risk_signal(
                supplier_id,
                {
                    "supplier_id": supplier_id,
                    "signal_type": signal_type,
                    "severity": severity,
                    "source": source,
                    "payload": payload,
                    "occurred_at": occurred_at,
                },
            )
            return

        if upper_stmt.startswith("SELECT SUPPLIER_ID, SIGNAL_TYPE") and (
            "FROM PROC.SUPPLIER_RISK_SIGNALS" in upper_stmt
        ):
            supplier_id, limit = params
            rows = self._store.fetch_risk_signals(supplier_id, int(limit))
            columns = [
                "supplier_id",
                "signal_type",
                "severity",
                "source",
                "payload",
                "occurred_at",
            ]
            self.description = [_ColumnDescriptor(col) for col in columns]
            self._results = [
                (
                    row["supplier_id"],
                    row["signal_type"],
                    row["severity"],
                    row["source"],
                    row.get("payload"),
                    row["occurred_at"],
                )
                for row in rows
            ]
            return

        if upper_stmt.startswith("INSERT INTO PROC.SUPPLIER_RISK_SCORES"):
            (
                supplier_id,
                score,
                model_version,
                feature_summary,
                computed_at,
            ) = params
            self._store.upsert_risk_score(
                supplier_id,
                {
                    "supplier_id": supplier_id,
                    "score": score,
                    "model_version": model_version,
                    "feature_summary": feature_summary,
                    "computed_at": computed_at,
                },
            )
            return

        if upper_stmt.startswith("SELECT SUPPLIER_ID, SCORE") and (
            "FROM PROC.SUPPLIER_RISK_SCORES" in upper_stmt
        ):
            (supplier_id,) = params
            row = self._store.fetch_risk_score(supplier_id)
            if row is None:
                self._results = []
            else:
                self._results = [
                    (
                        row["supplier_id"],
                        row["score"],
                        row["model_version"],
                        row.get("feature_summary"),
                        row["computed_at"],
                    )
                ]
            columns = [
                "supplier_id",
                "score",
                "model_version",
                "feature_summary",
                "computed_at",
            ]
            self.description = [_ColumnDescriptor(col) for col in columns]
            return

        raise NotImplementedError(f"Unsupported fake query: {statement}")


class _FakePostgresStore:
    def __init__(self) -> None:
        self.ensure_tables()

    def ensure_tables(self) -> None:
        if not hasattr(self, "supplier_response"):
            self.supplier_response = []  # type: ignore[attr-defined]
        if not hasattr(self, "workflow_email_tracking"):
            self.workflow_email_tracking = []  # type: ignore[attr-defined]
        if not hasattr(self, "workflow_lifecycle"):
            self.workflow_lifecycle = []  # type: ignore[attr-defined]
        if not hasattr(self, "supplier_risk_signals"):
            self.supplier_risk_signals = []  # type: ignore[attr-defined]
        if not hasattr(self, "supplier_risk_scores"):
            self.supplier_risk_scores = {}  # type: ignore[attr-defined]

    # -- information schema helpers --------------------------------------
    def _schema_columns(self) -> Dict[str, List[str]]:
        return {
            "proc.supplier_response": [
                "workflow_id",
                "supplier_id",
                "supplier_email",
                "rfq_id",
                "unique_id",
                "response_text",
                "response_body",
                "body_html",
                "response_message_id",
                "response_subject",
                "response_from",
                "response_date",
                "original_message_id",
                "original_subject",
                "match_confidence",
                "match_score",
                "match_evidence",
                "matched_on",
                "raw_headers",
                "price",
                "currency",
                "payment_terms",
                "warranty",
                "validity",
                "exceptions",
                "lead_time",
                "response_time",
                "received_time",
                "tables",
                "attachments",
                "processed",
            ],
            "proc.workflow_email_tracking": [
                "workflow_id",
                "unique_id",
                "dispatch_key",
                "supplier_id",
                "supplier_email",
                "recipient_emails",
                "message_id",
                "subject",
                "dispatched_at",
                "responded_at",
                "response_message_id",
                "matched",
                "created_at",
                "thread_headers",
            ],
            "proc.workflow_lifecycle": [
                "workflow_id",
                "supplier_agent_status",
                "supplier_agent_updated_at",
                "negotiation_status",
                "negotiation_updated_at",
                "watcher_status",
                "watcher_started_at",
                "watcher_stopped_at",
                "watcher_runtime_seconds",
                "expected_responses",
                "received_responses",
                "last_event",
                "last_event_at",
                "metadata",
            ],
            "proc.supplier_risk_signals": [
                "id",
                "supplier_id",
                "signal_type",
                "severity",
                "source",
                "payload",
                "occurred_at",
            ],
            "proc.supplier_risk_scores": [
                "supplier_id",
                "score",
                "model_version",
                "feature_summary",
                "computed_at",
            ],
        }

    def column_exists(self, schema: str, table: str, column: str) -> bool:
        full_table = f"{schema}.{table}"
        columns = self._schema_columns().get(full_table, [])
        return column in columns

    def list_columns(self, schema: str, table: str) -> List[str]:
        full_table = f"{schema}.{table}"
        return list(self._schema_columns().get(full_table, []))

    # -- supplier response operations ------------------------------------
    def upsert_supplier_response(
        self, workflow_id: str, unique_id: str, row: Dict[str, Any]
    ) -> None:
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        message_id = row.get("response_message_id")
        for existing in table:
            if existing["workflow_id"] != workflow_id:
                continue
            if existing["unique_id"] == unique_id or (
                message_id and existing.get("response_message_id") == message_id
            ):
                existing.update(row)
                return
        record = dict(row)
        record["workflow_id"] = workflow_id
        record["unique_id"] = unique_id
        if "processed" in record:
            record["processed"] = bool(record.get("processed"))
        table.append(record)

    def update_supplier_response_by_message_id(
        self, workflow_id: str, message_id: Optional[str], updates: Dict[str, Any]
    ) -> None:
        if not message_id:
            return
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        for existing in table:
            if existing["workflow_id"] != workflow_id:
                continue
            if existing.get("response_message_id") == message_id:
                existing.update(updates)
                if updates.get("processed"):
                    existing["processed"] = bool(
                        existing.get("processed") or updates["processed"]
                    )
                return

    def lookup_unique_by_message_id(
        self, workflow_id: str, message_id: Optional[str]
    ) -> Optional[str]:
        if not message_id:
            return None
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        for existing in table:
            if existing["workflow_id"] != workflow_id:
                continue
            if existing.get("response_message_id") == message_id:
                return existing.get("unique_id")
        return None

    def mark_supplier_responses_processed(
        self, workflow_id: str, unique_ids: Optional[List[str]]
    ) -> None:
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        ids = set(unique_ids or [])
        for row in table:
            if row["workflow_id"] != workflow_id:
                continue
            if unique_ids is None or row["unique_id"] in ids:
                row["processed"] = True

    def delete_supplier_responses(
        self, workflow_id: str, unique_ids: Optional[List[str]]
    ) -> None:
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        if unique_ids is None:
            self.supplier_response = [
                row for row in table if row["workflow_id"] != workflow_id
            ]
        else:
            ids = set(unique_ids)
            self.supplier_response = [
                row
                for row in table
                if not (
                    row["workflow_id"] == workflow_id and row["unique_id"] in ids
                )
            ]

    def fetch_supplier_responses(
        self, workflow_id: str, *, pending_only: bool
    ) -> List[Dict[str, Any]]:
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        results = []
        for row in table:
            if row["workflow_id"] != workflow_id:
                continue
            if pending_only and row.get("processed"):
                continue
            results.append(row.copy())
        return results

    # -- workflow tracking operations ------------------------------------
    def upsert_workflow_tracking(
        self, workflow_id: str, unique_id: str, row: Dict[str, Any]
    ) -> None:
        self.ensure_tables()
        table = self.workflow_email_tracking  # type: ignore[attr-defined]
        for existing in table:
            if existing["workflow_id"] == workflow_id and existing["unique_id"] == unique_id:
                existing.update(row)
                return
        table.append(dict(row))

    def distinct_active_workflow_ids(self) -> List[str]:
        self.ensure_tables()
        table = self.workflow_email_tracking  # type: ignore[attr-defined]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in table:
            workflow_id = row.get("workflow_id")
            if not workflow_id:
                continue
            grouped.setdefault(workflow_id, []).append(row)

        workflow_ids: List[str] = []
        for workflow_id, rows in grouped.items():
            has_pending = any(
                (r.get("responded_at") is None)
                or (not bool(r.get("matched")))
                for r in rows
            )
            if not has_pending:
                continue

            all_dispatched = all(r.get("dispatched_at") for r in rows)
            if not all_dispatched:
                continue

            workflow_ids.append(workflow_id)

        return workflow_ids

    def fetch_workflow_tracking(self, workflow_id: str) -> List[Dict[str, Any]]:
        self.ensure_tables()
        table = self.workflow_email_tracking  # type: ignore[attr-defined]
        return [row.copy() for row in table if row["workflow_id"] == workflow_id]

    def mark_workflow_response(
        self,
        workflow_id: str,
        unique_id: str,
        responded_at: Any,
        response_message_id: Optional[str],
    ) -> None:
        self.ensure_tables()
        table = self.workflow_email_tracking  # type: ignore[attr-defined]
        for row in table:
            if row["workflow_id"] == workflow_id and row["unique_id"] == unique_id:
                row["responded_at"] = responded_at
                row["response_message_id"] = response_message_id
                row["matched"] = True
                return

    def delete_workflow_rows(self, workflow_id: str) -> None:
        self.ensure_tables()
        table = self.workflow_email_tracking  # type: ignore[attr-defined]
        self.workflow_email_tracking = [
            row for row in table if row["workflow_id"] != workflow_id
        ]

    # -- workflow lifecycle operations ---------------------------------

    def upsert_workflow_lifecycle(
        self, workflow_id: str, row: Dict[str, Any]
    ) -> None:
        self.ensure_tables()
        table = self.workflow_lifecycle  # type: ignore[attr-defined]
        serialised = dict(row)
        if "metadata" in serialised and isinstance(serialised["metadata"], str):
            try:
                serialised["metadata"] = json.loads(serialised["metadata"])
            except Exception:
                pass
        for existing in table:
            if existing.get("workflow_id") == workflow_id:
                for key, value in serialised.items():
                    existing[key] = value
                return
        record = {"workflow_id": workflow_id}
        record.update(serialised)
        table.append(record)

    def fetch_workflow_lifecycle(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        self.ensure_tables()
        table = self.workflow_lifecycle  # type: ignore[attr-defined]
        for row in table:
            if row.get("workflow_id") == workflow_id:
                return dict(row)
        return None

    def delete_workflow_lifecycle(self, workflow_id: str) -> None:
        self.ensure_tables()
        table = self.workflow_lifecycle  # type: ignore[attr-defined]
        self.workflow_lifecycle = [
            row for row in table if row.get("workflow_id") != workflow_id
        ]

    # -- risk intelligence operations -----------------------------------

    def insert_risk_signal(self, supplier_id: str, row: Dict[str, Any]) -> None:
        self.ensure_tables()
        table = self.supplier_risk_signals  # type: ignore[attr-defined]
        record = dict(row)
        record.setdefault("supplier_id", supplier_id)
        record.setdefault("occurred_at", datetime.utcnow())
        table.append(record)

    def fetch_risk_signals(self, supplier_id: str, limit: int) -> List[Dict[str, Any]]:
        self.ensure_tables()
        table = self.supplier_risk_signals  # type: ignore[attr-defined]
        rows = [row.copy() for row in table if row.get("supplier_id") == supplier_id]
        rows.sort(key=lambda item: item.get("occurred_at"), reverse=True)
        return rows[:limit]

    def upsert_risk_score(self, supplier_id: str, row: Dict[str, Any]) -> None:
        self.ensure_tables()
        table = self.supplier_risk_scores  # type: ignore[attr-defined]
        table[supplier_id] = dict(row)

    def fetch_risk_score(self, supplier_id: str) -> Optional[Dict[str, Any]]:
        self.ensure_tables()
        table = self.supplier_risk_scores  # type: ignore[attr-defined]
        result = table.get(supplier_id)
        return dict(result) if result is not None else None


_FAKE_DB_STORE: Optional[_FakePostgresStore] = None


def _fake_store() -> _FakePostgresStore:
    global _FAKE_DB_STORE
    if _FAKE_DB_STORE is None:
        _FAKE_DB_STORE = _FakePostgresStore()
    return _FAKE_DB_STORE


class _FakeConnection:
    """Very small in-memory stand-in used for offline unit tests."""

    autocommit = True

    def __init__(self) -> None:
        self._store = _fake_store()

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._store)

    def close(self) -> None:  # pragma: no cover - compatibility no-op
        return

try:
    import psycopg2  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore

# Maintained for backward compatibility with callers that guard against the
# previous SQLite fallback. PostgreSQL is now mandatory so the flag is always
# ``False``.
USING_SQLITE = False

def _pg_dsn() -> Optional[str]:
    host = os.environ.get("PGHOST")
    db = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    pwd = os.environ.get("PGPASSWORD")
    port = os.environ.get("PGPORT")
    sslmode = os.environ.get("PGSSLMODE")

    if settings is not None:
        host = host or getattr(settings, "db_host", None)
        db = db or getattr(settings, "db_name", None)
        user = user or getattr(settings, "db_user", None)
        pwd = pwd or getattr(settings, "db_password", None)
        port = port or str(getattr(settings, "db_port", "5432"))

    if not host:
        return None

    port = port or "5432"
    parts = [f"host={host}", f"port={port}"]
    if db:
        parts.append(f"dbname={db}")
    if user:
        parts.append(f"user={user}")
    if pwd:
        parts.append(f"password={pwd}")
    if sslmode:
        parts.append(f"sslmode={sslmode}")
    return " ".join(parts)

@contextmanager
def get_conn():
    """Yield a PostgreSQL connection using environment derived DSN."""

    dsn = _pg_dsn()
    if not dsn or psycopg2 is None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            conn = _FakeConnection()
            try:
                yield conn
            finally:
                conn.close()
            return
        raise RuntimeError(
            "PostgreSQL connection parameters are not configured. "
            "Set PGHOST/PGDATABASE/PGUSER/PGPASSWORD to continue."
        )

    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
    except Exception:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            conn = _FakeConnection()
        else:
            raise

    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
