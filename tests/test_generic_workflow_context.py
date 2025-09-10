import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


class RecordingAgent:
    def __init__(self, next_agents=None, pass_fields=None):
        self.calls = []
        self.received = []
        self.next_agents = next_agents or []
        self.pass_fields = pass_fields or {}

    def execute(self, context):  # pragma: no cover - simple behaviour
        self.calls.append(context.agent_id)
        self.received.append(dict(context.input_data))
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={},
            next_agents=self.next_agents,
            pass_fields=self.pass_fields,
        )


def test_generic_workflow_uses_child_context_and_pass_fields():
    agent1 = RecordingAgent(next_agents=["second"], pass_fields={"foo": "bar"})
    agent2 = RecordingAgent()
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"start": agent1, "second": agent2},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model={}),
    )
    orchestrator = Orchestrator(nick)
    context = AgentContext(
        workflow_id="wf", agent_id="start", user_id="tester", input_data={}
    )
    orchestrator._execute_generic_workflow("start", context)

    assert agent1.calls == ["start"]
    assert agent2.calls == ["second"]
    assert agent2.received[0].get("foo") == "bar"
