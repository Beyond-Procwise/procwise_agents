import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.opportunity_miner_agent import OpportunityMinerAgent
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.settings = SimpleNamespace(script_user="tester")


def create_agent(monkeypatch, tables: Optional[Dict[str, Any]] = None):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick, min_financial_impact=0)
    monkeypatch.setattr(agent, "_output_excel", lambda findings: None)
    monkeypatch.setattr(agent, "_output_feed", lambda findings: None)
    data = tables if tables is not None else agent._mock_data()
    monkeypatch.setattr(
        agent,
        "_ingest_data",
        lambda: {name: df.copy() for name, df in data.items()},
    )
    return agent


def build_context(workflow: str, conditions: Optional[Dict[str, Any]]) -> AgentContext:
    return AgentContext(
        workflow_id="wf-test",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"workflow": workflow, "conditions": conditions or {}},
    )


def test_missing_workflow_blocks_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = AgentContext(
        workflow_id="wf-missing",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"conditions": {}},
    )

    output = agent.run(context)

    assert output.status == AgentStatus.FAILED
    assert output.data["blocked_reason"]
    assert output.data["policy_events"]


def test_price_variance_detection_generates_finding(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "price_variance_check",
        {
            "supplier_id": "SI0001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    pb = [f for f in output.data["findings"] if f["detector_type"] == "Price Benchmark Variance"]
    assert pb
    assert pb[0]["supplier_name"] == "Supplier One"
    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_volume_consolidation_identifies_costlier_supplier(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "volume_consolidation_check", {"minimum_volume_gbp": 50}
    )

    output = agent.run(context)

    vc = [f for f in output.data["findings"] if f["detector_type"] == "Volume Consolidation"]
    assert vc
    assert all(f["supplier_id"] for f in vc)


def test_supplier_risk_alert_threshold(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "supplier_risk_check",
        {"risk_threshold": 0.5, "risk_weight": 1000},
    )

    output = agent.run(context)

    alerts = [f for f in output.data["findings"] if f["detector_type"] == "Supplier Risk Alert"]
    assert alerts
    assert alerts[0]["supplier_id"] == "SI0001"
    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_maverick_spend_detection_flags_po(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "maverick_spend_check", {"minimum_value_gbp": 120}
    )

    output = agent.run(context)

    findings = [
        f for f in output.data["findings"] if f["detector_type"] == "Maverick Spend Detection"
    ]
    assert findings
    assert findings[0]["supplier_id"] == "SI0002"


def test_category_overspend_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "category_overspend_check", {"category_budgets": {"CatA": 90}}
    )

    output = agent.run(context)

    overspend = [
        f for f in output.data["findings"] if f["detector_type"] == "Category Overspend"
    ]
    assert overspend
    assert overspend[0]["supplier_id"] == "SI0001"


def test_unused_contract_value_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "unused_contract_value_check", {"minimum_unused_value_gbp": 50}
    )

    output = agent.run(context)

    unused = [
        f for f in output.data["findings"] if f["detector_type"] == "Unused Contract Value"
    ]
    assert unused
    assert unused[0]["supplier_id"] == "SI0002"
    assert unused[0]["supplier_name"] == "Supplier Two"


def test_esg_opportunity_creates_event(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "esg_opportunity_check",
        {
            "esg_scores": [{"supplier_id": "SI0003", "score": 0.9}],
            "incumbent_score": 0.6,
            "esg_threshold": 0.8,
            "estimated_switch_savings_gbp": 2000,
            "category_id": "CatA",
        },
    )

    output = agent.run(context)

    esg = [f for f in output.data["findings"] if f["detector_type"] == "ESG Opportunity"]
    assert esg
    assert esg[0]["supplier_id"] == "SI0003"
    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_inflation_passthrough_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "inflation_passthrough_check",
        {"market_inflation_pct": 0.02, "tolerance_pct": 0.0},
    )

    output = agent.run(context)

    inflation = [
        f for f in output.data["findings"] if f["detector_type"] == "Inflation Pass-Through"
    ]
    assert inflation
    assert inflation[0]["supplier_id"] == "SI0001"
