import os
import sys

import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.policy_engine import PolicyEngine


def _policy_rows():
    details = {
        "rules": {
            "default_weights": {"price": 0.6, "delivery": 0.4},
        }
    }
    return [
        {
            "policy_id": 1,
            "policy_name": "WeightAllocationPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Default supplier ranking weights",
            "policy_details": json.dumps(details),
        }
    ]


def test_valid_criteria_allowed():
    engine = PolicyEngine(policy_rows=_policy_rows())
    result = engine.validate_workflow('supplier_ranking', 'user1', {'criteria': ['price', 'delivery']})
    assert result['allowed'] is True


def test_invalid_criteria_blocked():
    engine = PolicyEngine(policy_rows=_policy_rows())
    result = engine.validate_workflow('supplier_ranking', 'user1', {'criteria': ['unknown']})
    assert result['allowed'] is False


def test_missing_criteria_defaults_to_policy_weights():
    engine = PolicyEngine(policy_rows=_policy_rows())
    result = engine.validate_workflow('supplier_ranking', 'user1', {})
    assert result['allowed'] is True
