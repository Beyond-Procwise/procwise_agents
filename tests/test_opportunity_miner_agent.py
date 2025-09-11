import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.opportunity_miner_agent import OpportunityMinerAgent
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.settings = SimpleNamespace(script_user="tester")


def test_opportunity_miner_returns_findings(monkeypatch):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick, min_financial_impact=0)

    # Avoid file system writes during tests
    monkeypatch.setattr(agent, "_output_excel", lambda findings: None)
    monkeypatch.setattr(agent, "_output_feed", lambda findings: None)
    monkeypatch.setattr(agent, "_ingest_data", agent._mock_data)

    context = AgentContext(
        workflow_id="wf1",
        agent_id="opportunity_miner",
        user_id="u1",
        input_data={},
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
    assert output.data["opportunity_count"] >= 0
    assert isinstance(output.data["findings"], list)


def test_supplier_consolidation_sets_supplier_id(monkeypatch):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick, min_financial_impact=0)

    monkeypatch.setattr(agent, "_output_excel", lambda findings: None)
    monkeypatch.setattr(agent, "_output_feed", lambda findings: None)
    monkeypatch.setattr(agent, "_ingest_data", agent._mock_data)

    context = AgentContext(
        workflow_id="wf2",
        agent_id="opportunity_miner",
        user_id="u1",
        input_data={},
    )

    output = agent.run(context)
    sc = [f for f in output.data["findings"] if f["detector_type"] == "Supplier Consolidation"]
    assert sc and sc[0]["supplier_id"] is not None


def test_opportunity_miner_handles_missing_columns(monkeypatch):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    monkeypatch.setattr(agent, "_output_excel", lambda findings: None)
    monkeypatch.setattr(agent, "_output_feed", lambda findings: None)

    mock_tables = agent._mock_data()
    mock_tables["purchase_orders"] = mock_tables["purchase_orders"][["po_id", "supplier_id", "currency"]]
    monkeypatch.setattr(agent, "_ingest_data", lambda: mock_tables)

    context = AgentContext(
        workflow_id="wf1",
        agent_id="opportunity_miner",
        user_id="u1",
        input_data={},
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS


def test_opportunity_miner_handles_none_payment_terms(monkeypatch):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    monkeypatch.setattr(agent, "_output_excel", lambda findings: None)
    monkeypatch.setattr(agent, "_output_feed", lambda findings: None)

    mock_tables = agent._mock_data()
    mock_tables["invoices"].loc[0, "payment_terms"] = None
    monkeypatch.setattr(agent, "_ingest_data", lambda: mock_tables)

    context = AgentContext(
        workflow_id="wf1",
        agent_id="opportunity_miner",
        user_id="u1",
        input_data={},
    )

    output = agent.run(context)
    assert output.status == AgentStatus.SUCCESS
