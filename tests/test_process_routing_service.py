import os
import sys
import json
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.process_routing_service import ProcessRoutingService


class DummyCursor:
    def __init__(self):
        self.params = None

    def execute(self, sql, params):
        self.params = params

    def fetchone(self):
        return [42]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyConn:
    def __init__(self):
        self.cursor_obj = DummyCursor()

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def test_log_process_defaults_status_zero():
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    pid = prs.log_process("foo", {"a": 1})
    assert pid == 42
    # process_status should default to 0 and not be None
    assert conn.cursor_obj.params[5] == 0


def test_convert_agents_to_flow_builds_tree():
    details = {
        "status": "saved",
        "agents": [
            {
                "agent": "1",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                "dependencies": {"onSuccess": ["2"], "onFailure": ["3"]},
            },
            {
                "agent": "2",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                "dependencies": {},
            },
            {
                "agent": "3",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                "dependencies": {},
            },
        ],
    }

    flow = ProcessRoutingService.convert_agents_to_flow(details)
    assert flow["agent_type"] == "1"
    assert flow["onSuccess"]["agent_type"] == "2"
    assert flow["onFailure"]["agent_type"] == "3"


class FetchCursor:
    def __init__(self, proc_details, prompt_rows, policy_rows):
        self.proc_details = proc_details
        self.prompt_rows = prompt_rows
        self.policy_rows = policy_rows
        self.query = ""

    def execute(self, sql, params=None):
        self.query = sql

    def fetchone(self):
        if "FROM proc.routing" in self.query:
            return [json.dumps(self.proc_details)]
        return None

    def fetchall(self):
        if "FROM proc.prompt" in self.query:
            return self.prompt_rows
        if "FROM proc.policy" in self.query:
            return self.policy_rows
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class FetchConn:
    def __init__(self, proc_details, prompt_rows, policy_rows):
        self.proc_details = proc_details
        self.prompt_rows = prompt_rows
        self.policy_rows = policy_rows

    def cursor(self):
        return FetchCursor(self.proc_details, self.prompt_rows, self.policy_rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def test_get_process_details_enriches_agent_data():
    proc_details = {
        "status": "saved",
        "agent_type": "supplier_ranking_123",
        "agent_property": {"llm": None, "prompts": [], "policies": []},
    }
    conn = FetchConn(
        proc_details,
        [(1, "{supplier_ranking}")],
        [(2, "{supplier_ranking}")],
    )
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    details = prs.get_process_details(1)
    assert details["agent_type"] == "SupplierRankingAgent"
    assert details["agent_property"]["prompts"] == [1]
    assert details["agent_property"]["policies"] == [2]
    assert details["process_status"] == 0
