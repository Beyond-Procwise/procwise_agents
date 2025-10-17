"""Database utility helpers for pandas compatibility."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pandas as pd


def _normalize_params(params: Any) -> Any:
    """Normalize DB-API parameters for compatibility across engines.

    ``pandas.read_sql`` ultimately forwards parameters to SQLAlchemy which
    expects an ``executemany`` style structure when it receives a ``list``.
    Our callers typically pass a list of values because ``psycopg`` adapts
    Python lists to ``ARRAY`` types (used with ``= ANY(%s)`` predicates).  When
    such a list is supplied directly SQLAlchemy raises ``ArgumentError``
    complaining that "List argument must consist only of tuples or
    dictionaries".  Converting the outer container to a tuple tells SQLAlchemy
    to treat it as the argument list for a single execution while keeping the
    inner lists intact so array adaptation still works.
    """

    if params is None:
        return None

    if isinstance(params, Mapping):
        return params

    if isinstance(params, Sequence) and not isinstance(
        params, (str, bytes, bytearray)
    ):
        # ``list`` triggers the SQLAlchemy executemany path unless each element
        # is already a mapping/tuple.  Converting to ``tuple`` maintains the
        # semantics while satisfying the DB-API expectations.
        if isinstance(params, list):
            if params and all(
                isinstance(item, (Mapping, tuple)) for item in params
            ):
                return params
            return tuple(params)

    return params


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
            normalized_params = _normalize_params(params)
            if normalized_params is not None:
                cursor.execute(sql, normalized_params)
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            description = cursor.description or []
            columns = [col[0] for col in description]
        return pd.DataFrame(rows, columns=columns)

    normalized_params = _normalize_params(params)
    if normalized_params is not None:
        return pd.read_sql(sql, conn, params=normalized_params)
    return pd.read_sql(sql, conn)
