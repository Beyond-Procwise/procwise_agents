import pandas as pd
import pytest

from utils.db import read_sql_compat


class _DummyCursor:
    def __init__(self, rows, description):
        self._rows = rows
        self.description = description

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed = (query, params)

    def fetchall(self):
        return self._rows


class _DummyConnection:
    def __init__(self, rows, description):
        self._rows = rows
        self._description = description
        self.cursor_calls = 0

    def cursor(self):
        self.cursor_calls += 1
        return _DummyCursor(self._rows, self._description)


def test_read_sql_compat_uses_cursor_for_dbapi(monkeypatch):
    conn = _DummyConnection([(1, "alpha")], [("id",), ("name",)])

    def fail_read_sql(*args, **kwargs):  # pragma: no cover - ensure not called
        raise AssertionError("pandas.read_sql should not be invoked for DBAPI connections")

    monkeypatch.setattr(pd, "read_sql", fail_read_sql)

    df = read_sql_compat("SELECT * FROM dummy", conn)
    assert conn.cursor_calls == 1
    assert list(df.columns) == ["id", "name"]
    assert df.iloc[0]["name"] == "alpha"


def test_read_sql_compat_delegates_to_pandas(monkeypatch):
    sentinel = object()
    calls = {}

    def fake_read_sql(query, conn, params=None):
        calls["invoked"] = (query, conn, params)
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df = read_sql_compat("SELECT 1", sentinel, params={"a": 1})
    assert calls["invoked"] == ("SELECT 1", sentinel, {"a": 1})
    assert df.iloc[0]["value"] == 1
