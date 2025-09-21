"""Database utility helpers for pandas compatibility."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def read_sql_compat(sql: str, conn: Any, params: Optional[Any] = None) -> pd.DataFrame:
    """Execute ``sql`` using ``conn`` returning a DataFrame without pandas warnings.

    pandas emits ``UserWarning`` messages when supplied with raw DBAPI
    connections because their behaviour is not part of the library's
    compatibility guarantees.  This helper inspects the connection and falls
    back to manual cursor execution for DBAPI objects while still delegating to
    :func:`pandas.read_sql` for SQLAlchemy engines or connection strings.  The
    return value always mirrors :func:`pandas.read_sql` so callers can use it as
    a drop-in replacement.
    """

    if hasattr(conn, "cursor") and callable(getattr(conn, "cursor")):
        with conn.cursor() as cursor:
            if params is not None:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            description = cursor.description or []
            columns = [col[0] for col in description]
        return pd.DataFrame(rows, columns=columns)

    if params is not None:
        return pd.read_sql(sql, conn, params=params)
    return pd.read_sql(sql, conn)
