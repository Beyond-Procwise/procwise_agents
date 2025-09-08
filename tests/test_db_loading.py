import json
from types import SimpleNamespace

from orchestration.orchestrator import Orchestrator


class DummyCursor:
    """Cursor that returns rows based on the executed query."""

    def __init__(self, rows, agent_rows):
        self._rows = rows
        self._agent_rows = agent_rows
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, query, params=None):
        self.query = query

    def fetchall(self):
        if "FROM proc.agent" in self.query:
            return self._agent_rows
        return self._rows


class DummyConn:
    def __init__(self, rows, agent_rows):
        self._rows = rows
        self._agent_rows = agent_rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def cursor(self):
        return DummyCursor(self._rows, self._agent_rows)


def make_orchestrator(rows, agent_rows):
    def get_conn():
        return DummyConn(rows, agent_rows)

    agent_nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        agents={},
        get_db_connection=get_conn,
    )
    return Orchestrator(agent_nick)


def test_load_prompts_from_db():
    prompt_rows = [(1, "hello", "{1}")]
    agent_rows = [(1, "AgentOne")]
    orchestrator = make_orchestrator(prompt_rows, agent_rows)
    prompts = orchestrator._load_prompts()
    assert prompts[1]["template"] == "hello"
    assert prompts[1]["agents"][0]["agent_name"] == "AgentOne"


def test_load_policies_from_db():
    policy_rows = [(2, "Example policy", "{1}")]
    agent_rows = [(1, "AgentOne")]
    orchestrator = make_orchestrator(policy_rows, agent_rows)
    policies = orchestrator._load_policies()
    assert policies[2]["description"] == "Example policy"
    assert policies[2]["agents"][0]["agent_name"] == "AgentOne"


def test_load_agent_definitions_from_db():
    agent_rows = [(1, "AgentOne"), (2, "AgentTwo")]
    orchestrator = make_orchestrator([], agent_rows)
    defs = orchestrator._load_agent_definitions()
    assert defs["1"] == "AgentOne"
    assert defs["2"] == "AgentTwo"

