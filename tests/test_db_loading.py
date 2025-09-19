import json
from types import SimpleNamespace

from orchestration.orchestrator import Orchestrator
from orchestration.prompt_engine import PromptEngine


class DummyCursor:
    """Cursor that returns preset rows for executed queries."""

    def __init__(self, data_map):
        self._data_map = data_map
        self._rows: list = []
        self.query = ""
        self.description = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, query, params=None):
        self.query = query
        if "proc.prompt" in query:
            self._rows = self._data_map.get("prompt", [])
            self.description = [
                ("prompt_id", None, None, None, None, None, None),
                ("prompt_name", None, None, None, None, None, None),
                ("prompt_type", None, None, None, None, None, None),
                ("prompt_linked_agents", None, None, None, None, None, None),
                ("prompts_desc", None, None, None, None, None, None),
                ("prompts_status", None, None, None, None, None, None),
                ("created_date", None, None, None, None, None, None),
                ("created_by", None, None, None, None, None, None),
                ("last_modified_date", None, None, None, None, None, None),
                ("last_modified_by", None, None, None, None, None, None),
                ("version", None, None, None, None, None, None),
            ]
        elif "proc.policy" in query:
            self._rows = self._data_map.get("policy", [])
            self.description = [
                ("policy_id", None, None, None, None, None, None),
                ("policy_desc", None, None, None, None, None, None),
                ("policy_details", None, None, None, None, None, None),
                ("policy_linked_agents", None, None, None, None, None, None),
            ]
        else:
            self._rows = self._data_map.get("default", [])
            self.description = []

    def fetchall(self):
        return self._rows


class DummyConn:
    def __init__(self, data_map):
        self._data_map = data_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def cursor(self):
        return DummyCursor(self._data_map)


def make_orchestrator(prompt_rows=None, policy_rows=None):
    data_map = {
        "prompt": prompt_rows or [],
        "policy": policy_rows or [],
    }

    def get_conn():
        return DummyConn(data_map)

    agent_nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        agents={},
        get_db_connection=get_conn,
        prompt_engine=PromptEngine(connection_factory=get_conn),
    )
    return Orchestrator(agent_nick)


def test_load_prompts_from_db():
    prompt_rows = [
        (
            1,
            "Opportunity Prompt",
            "template",
            "{supplier_ranking}",
            "hello",
            1,
            None,
            None,
            None,
            None,
            None,
        )
    ]
    orchestrator = make_orchestrator(prompt_rows=prompt_rows)
    prompts = orchestrator._load_prompts()
    assert prompts[1]["template"] == "hello"
    assert prompts[1]["agents"][0]["agent_name"] == "SupplierRankingAgent"


def test_load_policies_from_db():
    policy_rows = [
        (2, "Example policy", json.dumps({"rules": {}}), "{supplier_ranking}"),
    ]
    orchestrator = make_orchestrator(policy_rows=policy_rows)
    policies = orchestrator._load_policies()
    assert policies[2]["description"] == "Example policy"
    assert policies[2]["details"] == {"rules": {}}
    assert policies[2]["agents"][0]["agent_name"] == "SupplierRankingAgent"


def test_load_agent_definitions_from_file():
    orchestrator = make_orchestrator()
    defs = orchestrator._load_agent_definitions()
    assert defs["supplier_ranking"] == "SupplierRankingAgent"
    assert "admin_supplier_ranking" not in defs

