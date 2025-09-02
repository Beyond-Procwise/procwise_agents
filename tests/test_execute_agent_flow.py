import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from agents.base_agent import AgentOutput, AgentStatus


class DummyAgent:
    def __init__(self):
        self.ran = False

    def execute(self, context):  # pragma: no cover - simple stub
        self.ran = True
        return AgentOutput(status=AgentStatus.SUCCESS, data={})


def test_execute_agent_flow_runs_and_updates_status():
    agent = DummyAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"data_extraction": agent},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )
    orchestrator = Orchestrator(nick)
    flow = {
        "status": "saved",
        "agent_type": "1",
        "agent_property": {"llm": "m", "prompts": [1], "policies": [1]},
    }
    result = orchestrator.execute_agent_flow(flow)
    assert result["status"] == "completed"
    assert flow["status"] == "completed"
    assert agent.ran is True
