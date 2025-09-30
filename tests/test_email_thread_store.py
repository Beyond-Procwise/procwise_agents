import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(Path(__file__).resolve().parent, "..")))

from services import email_thread_store


def test_sanitise_thread_table_name_valid_schema():
    result = email_thread_store.sanitise_thread_table_name("proc.custom_threads")
    assert result == "proc.custom_threads"


def test_sanitise_thread_table_name_invalid_logs_warning(caplog):
    caplog.set_level(logging.WARNING)
    result = email_thread_store.sanitise_thread_table_name("invalid-name!", logger=logging.getLogger("test"))
    assert result == email_thread_store.DEFAULT_THREAD_TABLE
    assert "Invalid email thread table name" in caplog.text


@pytest.mark.parametrize(
    "value",
    [None, "", "  "],
)
def test_sanitise_thread_table_name_defaults(value):
    assert email_thread_store.sanitise_thread_table_name(value) == email_thread_store.DEFAULT_THREAD_TABLE
