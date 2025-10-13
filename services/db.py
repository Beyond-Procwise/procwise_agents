from __future__ import annotations

import os
from contextlib import contextmanager
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

        if upper_stmt.startswith("CREATE SCHEMA"):
            return
        if upper_stmt.startswith("CREATE TABLE") or upper_stmt.startswith(
            "CREATE INDEX"
        ):
            self._store.ensure_tables()
            return
        if "INFORMATION_SCHEMA.COLUMNS" in upper_stmt:
            schema, table, column = params
            exists = self._store.column_exists(schema, table, column)
            self._results = [(1,)] if exists else []
            self.description = [_ColumnDescriptor("exists")]
            return
        if upper_stmt.startswith("ALTER TABLE"):
            # schema migrations are no-ops for the in-memory store because
            # columns are eagerly present.
            self._store.ensure_tables()
            return

        # Data manipulation commands for supplier responses
        if upper_stmt.startswith("INSERT INTO PROC.SUPPLIER_RESPONSE"):
            (
                workflow_id,
                supplier_id,
                unique_id,
                response_text,
                price,
                lead_time,
                received_time,
            ) = params
            self._store.upsert_supplier_response(
                workflow_id,
                unique_id,
                {
                    "workflow_id": workflow_id,
                    "supplier_id": supplier_id,
                    "unique_id": unique_id,
                    "response_text": response_text,
                    "price": price,
                    "lead_time": lead_time,
                    "received_time": received_time,
                },
            )
            return

        if upper_stmt.startswith("DELETE FROM PROC.SUPPLIER_RESPONSE"):
            if "UNIQUE_ID = ANY" in upper_stmt:
                workflow_id, ids = params
                self._store.delete_supplier_responses(workflow_id, list(ids))
            else:
                (workflow_id,) = params
                self._store.delete_supplier_responses(workflow_id, None)
            return

        if upper_stmt.startswith("SELECT WORKFLOW_ID, SUPPLIER_ID") and (
            "FROM PROC.SUPPLIER_RESPONSE" in upper_stmt
        ):
            (workflow_id,) = params
            rows = self._store.fetch_supplier_responses(workflow_id)
            columns = [
                "workflow_id",
                "supplier_id",
                "unique_id",
                "response_text",
                "price",
                "lead_time",
                "received_time",
            ]
            self.description = [
                _ColumnDescriptor(col) for col in columns
            ]
            self._results = [
                tuple(row.get(col) for col in columns)
                for row in rows
            ]
            return

        # Workflow email tracking commands
        if upper_stmt.startswith("INSERT INTO PROC.WORKFLOW_EMAIL_TRACKING"):
            (
                workflow_id,
                unique_id,
                supplier_id,
                supplier_email,
                message_id,
                subject,
                dispatched_at,
                responded_at,
                response_message_id,
                matched,
            ) = params
            self._store.upsert_workflow_tracking(
                workflow_id,
                unique_id,
                {
                    "workflow_id": workflow_id,
                    "unique_id": unique_id,
                    "supplier_id": supplier_id,
                    "supplier_email": supplier_email,
                    "message_id": message_id,
                    "subject": subject,
                    "dispatched_at": dispatched_at,
                    "responded_at": responded_at,
                    "response_message_id": response_message_id,
                    "matched": matched,
                },
            )
            return

        if upper_stmt.startswith("SELECT DISTINCT WORKFLOW_ID"):
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
                "supplier_id",
                "supplier_email",
                "message_id",
                "subject",
                "dispatched_at",
                "responded_at",
                "response_message_id",
                "matched",
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

        raise NotImplementedError(f"Unsupported fake query: {statement}")


class _FakePostgresStore:
    def __init__(self) -> None:
        self.ensure_tables()

    def ensure_tables(self) -> None:
        if not hasattr(self, "supplier_response"):
            self.supplier_response = []  # type: ignore[attr-defined]
        if not hasattr(self, "workflow_email_tracking"):
            self.workflow_email_tracking = []  # type: ignore[attr-defined]

    # -- information schema helpers --------------------------------------
    def column_exists(self, schema: str, table: str, column: str) -> bool:
        full_table = f"{schema}.{table}"
        columns = {
            "proc.supplier_response": {
                "workflow_id",
                "supplier_id",
                "unique_id",
                "response_text",
                "price",
                "lead_time",
                "received_time",
            },
            "proc.workflow_email_tracking": {
                "workflow_id",
                "unique_id",
                "supplier_id",
                "supplier_email",
                "message_id",
                "subject",
                "dispatched_at",
                "responded_at",
                "response_message_id",
                "matched",
                "created_at",
            },
        }
        return column in columns.get(full_table, set())

    # -- supplier response operations ------------------------------------
    def upsert_supplier_response(
        self, workflow_id: str, unique_id: str, row: Dict[str, Any]
    ) -> None:
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        for existing in table:
            if existing["workflow_id"] == workflow_id and existing["unique_id"] == unique_id:
                existing.update(row)
                return
        table.append(dict(row))

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
            self.supplier_response = [
                row
                for row in table
                if not (
                    row["workflow_id"] == workflow_id
                    and row["unique_id"] in set(unique_ids)
                )
            ]

    def fetch_supplier_responses(self, workflow_id: str) -> List[Dict[str, Any]]:
        self.ensure_tables()
        table = self.supplier_response  # type: ignore[attr-defined]
        return [row.copy() for row in table if row["workflow_id"] == workflow_id]

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
        workflow_ids = []
        for row in table:
            dispatched_at = row.get("dispatched_at")
            responded_at = row.get("responded_at")
            matched = row.get("matched")
            if dispatched_at and (responded_at is None or not matched):
                if row["workflow_id"] not in workflow_ids:
                    workflow_ids.append(row["workflow_id"])
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
