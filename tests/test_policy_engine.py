import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.policy_engine import PolicyEngine


def test_valid_criteria_allowed():
    engine = PolicyEngine()
    result = engine.validate_workflow('supplier_ranking', 'user1', {'criteria': ['price', 'delivery']})
    assert result['allowed'] is True


def test_invalid_criteria_blocked():
    engine = PolicyEngine()
    result = engine.validate_workflow('supplier_ranking', 'user1', {'criteria': ['unknown']})
    assert result['allowed'] is False
