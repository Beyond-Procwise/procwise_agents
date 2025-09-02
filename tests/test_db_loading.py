import json
from types import SimpleNamespace

from orchestration.orchestrator import Orchestrator


class DummyCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, query, params=None):
        self.query = query

    def fetchall(self):
        return self._rows


class DummyConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def cursor(self):
        return DummyCursor(self._rows)


def make_orchestrator(rows):
    def get_conn():
        return DummyConn(rows)

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
    rows = [(1, json.dumps({"promptId": 1, "content": "hello"}))]
    orchestrator = make_orchestrator(rows)
    prompts = orchestrator._load_prompts()
    assert prompts[1]["content"] == "hello"


def test_load_policies_from_db():
    rows = [(2, json.dumps({"policyName": "Example"}))]
    orchestrator = make_orchestrator(rows)
    policies = orchestrator._load_policies()
    assert policies[2]["policyName"] == "Example"


def test_invalid_prompt_json_is_wrapped():
    rows = [(99, "not-json")]
    orchestrator = make_orchestrator(rows)
    prompts = orchestrator._load_prompts()
    assert prompts[99]["template"] == "not-json"


def test_invalid_policy_json_is_wrapped():
    rows = [(99, "not-json")]
    orchestrator = make_orchestrator(rows)
    policies = orchestrator._load_policies()
    assert policies[99]["description"] == "not-json"

