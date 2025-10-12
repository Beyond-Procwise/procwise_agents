from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore

USING_SQLITE = False

def _pg_dsn() -> Optional[str]:
    host = os.environ.get("PGHOST")
    if not host:
        return None
    db = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    pwd = os.environ.get("PGPASSWORD")
    port = os.environ.get("PGPORT", "5432")
    sslmode = os.environ.get("PGSSLMODE")
    parts = [f"host={host}", f"port={port}"]
    if db: parts.append(f"dbname={db}")
    if user: parts.append(f"user={user}")
    if pwd: parts.append(f"password={pwd}")
    if sslmode: parts.append(f"sslmode={sslmode}")
    return " ".join(parts)

@contextmanager
def get_conn():
    """
    Yields a DB connection. Prefers PostgreSQL via psycopg2.
    Falls back to SQLite (file ./procwise_dev.sqlite) for local dev if PG env not set.
    """
    global USING_SQLITE
    dsn = _pg_dsn()
    if dsn and psycopg2 is not None:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        USING_SQLITE = False
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass
    else:
        USING_SQLITE = True
        path = os.path.abspath("./procwise_dev.sqlite")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass
