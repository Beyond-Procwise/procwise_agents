import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.supplier_ranking_agent import SupplierRankingAgent
from agents.base_agent import AgentContext
from engines.policy_engine import PolicyEngine


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.settings = SimpleNamespace(extraction_model="llama3", script_user="tester")


def test_top_n_parsed_from_query(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame({
        "supplier_name": [f"S{i}" for i in range(6)],
        "price": [60, 50, 40, 30, 20, 10],
    })

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="u1",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price"]}},
            "query": "Rank top 5 suppliers by price",
        },
    )

    output = agent.run(context)
    assert len(output.data["ranking"]) == 5
