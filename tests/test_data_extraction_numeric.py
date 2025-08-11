"""Tests for numeric sanitation helpers in DataExtractionAgent."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.data_extraction_agent import DataExtractionAgent


class _DummyNick:
    """Minimal stub of AgentNick required for instantiation in tests."""

    def __init__(self):
        self.settings = type("S", (), {"extraction_model": "dummy"})()


agent = DataExtractionAgent(_DummyNick())


def test_clean_numeric_percent():
    assert agent._clean_numeric("20%") == 20.0


def test_clean_numeric_currency_and_negative():
    assert agent._clean_numeric("£1,234.50") == 1234.50
    assert agent._clean_numeric("(100)") == -100.0


def test_sanitize_value_tax_percent():
    assert agent._sanitize_value("20%", "tax_percent") == 20.0


def test_clean_numeric_mixed_values():
    assert agent._clean_numeric("Tax (20%) £150") == 20.0


def test_clean_date_trailing_dot():
    from datetime import date

    assert agent._clean_date("30 MARCH. 2024") == date(2024, 3, 30)


def test_sanitize_currency_symbols():
    assert agent._sanitize_value("£", "currency") == "GBP"
    assert agent._sanitize_value("USD", "currency") == "USD"
