import os
import sys
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
