import os
import sys
import json
from datetime import datetime
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


def test_get_process_details_handles_prefixed_agent_names():
    proc_details = {
        "status": "saved",
        "agent_type": "admin_quote_agent_000067_1757404210002",
        "agent_property": {"llm": None, "prompts": [], "policies": []},
    }
    conn = FetchConn(proc_details, [], [])
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs._load_agent_links = lambda: (
        {"quote_evaluation": "QuoteEvaluationAgent"},
        {},
        {},
    )
    details = prs.get_process_details(1)
    assert details["agent_type"] == "QuoteEvaluationAgent"


def test_update_agent_status_preserves_structure():
    initial = {
        "status": "",
        "agents": [
            {"agent": "A1", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "1"},
            {"agent": "A2", "dependencies": {"onSuccess": ["A1"], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "2"},
            {"agent": "A3", "dependencies": {"onSuccess": [], "onFailure": ["A1"], "onCompletion": []}, "status": "saved", "agent_ref_id": "3"},
        ],
    }
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs.get_process_details = lambda pid, **kwargs: initial
    prs.update_agent_status(1, "A2", "validated")
    updated = json.loads(conn.cursor_obj.params[0])
    assert updated["agents"][1]["status"] == "validated"
    assert updated["agents"][0]["status"] == "saved"
    assert updated["agents"][2]["dependencies"]["onFailure"] == ["A1"]
    assert updated["status"] == "saved"



def test_update_agent_status_preserves_structure_scenario2():
    initial = {
        "status": "",
        "agents": [
            {"agent": "A1", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "1"},
            {"agent": "A2", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "2"},
            {"agent": "A3", "dependencies": {"onSuccess": ["A1"], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "3"},
            {"agent": "A4", "dependencies": {"onSuccess": [], "onFailure": ["A1", "A2"], "onCompletion": []}, "status": "saved", "agent_ref_id": "4"},
        ],
    }
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs.get_process_details = lambda pid, **kwargs: initial

    prs.update_agent_status(1, "A4", "validated")
    updated = json.loads(conn.cursor_obj.params[0])
    assert updated["agents"][3]["status"] == "validated"
    assert updated["agents"][2]["dependencies"]["onSuccess"] == ["A1"]
    assert updated["agents"][3]["dependencies"]["onFailure"] == ["A1", "A2"]
    assert updated["status"] == "saved"



def test_update_agent_status_updates_overall_status():
    initial = {
        "status": "saved",
        "agents": [
            {"agent": "A1", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "1"},
            {"agent": "A2", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "2"},
        ],
    }
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    holder = {"value": initial}

    def get_details(pid, **kwargs):
        return holder["value"]

    def upd_details(pid, details, modified_by=None):
        holder["value"] = json.loads(json.dumps(details))

    def upd_status(pid, status, modified_by=None, process_details=None):
        holder["value"] = json.loads(json.dumps(process_details))
        conn.cursor_obj.params = (status, json.dumps(process_details))

    prs.get_process_details = get_details
    prs.update_process_details = upd_details
    prs.update_process_status = upd_status

    prs.update_agent_status(1, "A1", "completed")
    assert holder["value"]["status"] == "saved"

    prs.update_agent_status(1, "A2", "completed")
    assert holder["value"]["status"] == "completed"
    assert conn.cursor_obj.params[0] == 1

    prs.update_agent_status(1, "A1", "failed")
    assert holder["value"]["status"] == "failed"
    assert conn.cursor_obj.params[0] == -1


def test_update_process_status_updates_process_details():
    initial = {"status": "saved", "agents": []}
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs.get_process_details = lambda pid, **kwargs: initial

    prs.update_process_status(1, 1)
    params = conn.cursor_obj.params
    assert params[0] == 1
    stored = json.loads(params[1])
    assert stored["status"] == "completed"


def test_log_run_detail_keeps_agent_statuses():
    details = {
        "status": "",
        "agents": [
            {"agent": "A1", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "validated", "agent_ref_id": "1"},
            {"agent": "A2", "dependencies": {"onSuccess": ["A1"], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "2"},
        ],
    }
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs.log_run_detail(
        process_id=1,
        process_status="success",
        process_details=details,
        process_start_ts=datetime.utcnow(),
        process_end_ts=datetime.utcnow(),
        triggered_by="tester",
    )
    updated = json.loads(conn.cursor_obj.params[1])
    assert updated["status"] == "saved"

    assert updated["agents"][0]["status"] == "validated"
    assert updated["agents"][1]["status"] == "saved"
