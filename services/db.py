from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

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
    """Yield a PostgreSQL connection using environment derived DSN."""

    dsn = _pg_dsn()
    if not dsn or psycopg2 is None:
        raise RuntimeError(
            "PostgreSQL connection parameters are not configured. "
            "Set PGHOST/PGDATABASE/PGUSER/PGPASSWORD to continue."
        )

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
