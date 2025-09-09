import json
from types import SimpleNamespace

from orchestration.orchestrator import Orchestrator


class DummyCursor:
    """Cursor that returns preset rows for executed queries."""

    def __init__(self, rows):
        self._rows = rows
        self.query = ""

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
    prompt_rows = [(1, "hello", "{supplier_ranking}")]
    orchestrator = make_orchestrator(prompt_rows)
    prompts = orchestrator._load_prompts()
    assert prompts[1]["template"] == "hello"
    assert prompts[1]["agents"][0]["agent_name"] == "SupplierRankingAgent"


def test_load_policies_from_db():
    policy_rows = [(2, "Example policy", "{supplier_ranking}")]
    orchestrator = make_orchestrator(policy_rows)
    policies = orchestrator._load_policies()
    assert policies[2]["description"] == "Example policy"
    assert policies[2]["agents"][0]["agent_name"] == "SupplierRankingAgent"


def test_load_agent_definitions_from_file():
    orchestrator = make_orchestrator([])
    defs = orchestrator._load_agent_definitions()
    assert defs["supplier_ranking"] == "SupplierRankingAgent"
    assert "admin_supplier_ranking" not in defs

