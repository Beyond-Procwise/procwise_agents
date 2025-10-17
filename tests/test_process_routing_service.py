import os
import sys
import json
from datetime import datetime
from types import SimpleNamespace
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Enable GPU for tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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


def test_convert_agents_to_flow_respects_list_order():
    """Even when dependencies reference upstream agents, the first listed
    agent should remain the starting point of the flow."""

    details = {
        "status": "saved",
        "agents": [
            {
                "agent": "A1",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                "dependencies": {},
            },
            {
                "agent": "A2",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                # Incorrectly points back to the first agent
                "dependencies": {"onSuccess": ["A1"]},
            },
        ],
    }

    flow = ProcessRoutingService.convert_agents_to_flow(details)
    assert flow["agent"] == "A1"
    assert flow["onSuccess"]["agent"] == "A2"


def test_convert_agents_to_flow_handles_reverse_dependencies():
    details = {
        "status": "saved",
        "agents": [
            {
                "agent": "A1",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                "dependencies": {},
            },
            {
                "agent": "A2",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                # Declares dependency on A1's success
                "dependencies": {"onSuccess": ["A1"]},
            },
            {
                "agent": "A3",
                "status": "saved",
                "agent_property": {"llm": "m", "prompts": [], "policies": []},
                # Runs when A2 fails
                "dependencies": {"onFailure": ["A2"]},
            },
        ],
    }

    flow = ProcessRoutingService.convert_agents_to_flow(details)
    assert flow["agent"] == "A1"
    assert flow["onSuccess"]["agent"] == "A2"
    assert flow["onSuccess"]["onFailure"]["agent"] == "A3"


def test_convert_agents_to_flow_normalises_properties():
    details = {
        "status": "saved",
        "agents": [
            {
                "agent": "A1",
                "status": "saved",
                "agent_property": {
                    "llm": "gpt-oss:latest",
                    "memory": "3.2B",
                    "prompts": ["1", "2", "1"],
                    "policies": [{"policyId": "3"}, "4"],
                },
                "dependencies": {},
            }
        ],
    }

    flow = ProcessRoutingService.convert_agents_to_flow(details)
    props = flow["agent_property"]
    assert props["llm"] == "gpt-oss"
    assert props["prompts"] == [1, 2]
    assert props["policies"] == [3, 4]
    assert "memory" not in props


def test_convert_agents_to_flow_preserves_agent_ref_id():
    details = {
        "status": "saved",
        "agents": [
            {
                "agent": "A1",
                "agent_ref_id": "123",
                "agent_id": "123",
                "status": "saved",
                "agent_property": {},
                "dependencies": {},
            }
        ],
    }

    flow = ProcessRoutingService.convert_agents_to_flow(details)
    assert flow["agent_ref_id"] == "123"
    assert flow["agent_id"] == "123"


def test_canonical_key_matches_class_name():
    defs = {"opportunity_miner": "OpportunityMinerAgent"}
    slug = ProcessRoutingService._canonical_key("OpportunityMinerAgent", defs)
    assert slug == "opportunity_miner"


def test_normalise_agent_properties_defaults_llm():
    props = ProcessRoutingService._normalise_agent_properties({"prompts": [1]})
    assert props["llm"] == ProcessRoutingService.DEFAULT_LLM_MODEL


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
        "agent_property": {
            "llm": "gpt-oss:latest",
            "memory": "3.2B",
            "prompts": [],
            "policies": [],
        },
    }
    prompt_rows = [(1, "hello", "{supplier_ranking}")]
    policy_rows = [
        (
            2,
            "ExamplePolicy",
            "Example policy",
            json.dumps({"rules": {}}),
            "{supplier_ranking}",
        ),
    ]
    conn = FetchConn(proc_details, prompt_rows, policy_rows)
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs._load_agent_links = lambda: (
        {"supplier_ranking": "SupplierRankingAgent"},
        {"supplier_ranking": [1]},
        {"supplier_ranking": [2]},
    )
    details = prs.get_process_details(1)
    assert details["agent_type"] == "SupplierRankingAgent"
    assert details["agent_property"]["prompts"] == [1]
    assert details["agent_property"]["policies"] == [2]
    assert details["agent_property"]["llm"] == "gpt-oss"
    assert "memory" not in details["agent_property"]


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


def test_load_agent_links_normalises_proc_agent_type():
    agent_id = "admin_testopp_000092_1758530338949"
    raw_agent_type = agent_id
    agent_name = "OpportunityMinerAgent"
    props_payload = json.dumps({"llm": "llama-2", "prompts": [7]})
    timestamp = datetime.utcnow()

    class AgentCursor:
        def __init__(self):
            self.query = ""

        def execute(self, sql, params=None):
            self.query = sql

        def fetchall(self):
            if "FROM proc.prompt" in self.query:
                return []
            if "FROM proc.policy" in self.query:
                return []
            if "FROM proc.agent" in self.query:
                return [
                    (
                        agent_id,
                        raw_agent_type,
                        agent_name,
                        props_payload,
                        timestamp,
                        timestamp,
                    )
                ]
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    class AgentConn:
        def cursor(self):
            return AgentCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    agent = SimpleNamespace(
        get_db_connection=lambda: AgentConn(),
        settings=SimpleNamespace(script_user="tester"),
    )

    prs = ProcessRoutingService(agent)
    agent_defs, prompt_map, policy_map = prs._load_agent_links()

    assert agent_defs.get("opportunity_miner") == "OpportunityMinerAgent"
    assert prs._agent_type_cache_by_id[agent_id] == "OpportunityMinerAgent"
    assert "opportunity_miner" in prs._agent_defaults_cache
    assert prs._agent_defaults_cache["opportunity_miner"]["llm"] == "llama-2"
    assert prompt_map == {}
    assert policy_map == {}


def test_enrich_node_applies_agent_defaults():
    agent = SimpleNamespace(settings=SimpleNamespace(script_user="tester"))
    prs = ProcessRoutingService(agent)
    prs._agent_defaults_cache = {
        "opportunity_miner": {
            "llm": "gpt-oss",
            "prompts": [50],
            "policies": [9],
            "conditions": {"negotiation_window_days": 30},
        }
    }

    node = {
        "agent_type": "opportunity_miner",
        "agent_property": {"prompts": [51], "conditions": {"negotiation_window_days": 60}},
    }

    agent_defs = {"opportunity_miner": "OpportunityMinerAgent"}
    prompt_map = {"opportunity_miner": [52]}
    policy_map = {"opportunity_miner": [10]}

    prs._enrich_node(node, agent_defs, prompt_map, policy_map)

    props = node["agent_property"]
    assert props["llm"] == "gpt-oss"
    assert props["prompts"] == [50, 51, 52]
    assert props["policies"] == [9, 10]
    assert props["conditions"]["negotiation_window_days"] == 60


def test_enrich_node_uses_base_agent_ref_id():
    agent = SimpleNamespace(settings=SimpleNamespace(script_user="tester"))
    prs = ProcessRoutingService(agent)
    prs._agent_property_cache_by_id = {
        "admin_opportunity_000065": {
            "llm": "llama-db",
            "prompts": [9],
            "policies": [12],
        }
    }
    prs._agent_type_cache_by_id = {
        "admin_opportunity_000065": "OpportunityMinerAgent"
    }
    prs._prompt_id_catalog = {9}
    prs._policy_id_catalog = {12}

    node = {
        "agent_ref_id": "admin_opportunity_000065_1758278380887",
        "agent_property": {},
        "agent_type": "opportunity_miner",
    }

    prs._enrich_node(
        node,
        {"opportunity_miner": "OpportunityMinerAgent"},
        {},
        {},
    )

    props = node["agent_property"]
    assert props["llm"] == "llama-db"
    assert props["prompts"] == [9]
    assert props["policies"] == [12]
    assert node["agent_id"] == "admin_opportunity_000065"


def test_enrich_node_replaces_dynamic_agent_type_with_db_value():
    agent = SimpleNamespace(settings=SimpleNamespace(script_user="tester"))
    prs = ProcessRoutingService(agent)
    dynamic_id = "admin_testopp_000092_1758530338949"
    prs._agent_property_cache_by_id = {
        dynamic_id: {"llm": "legacy-llm", "prompts": [], "policies": []}
    }
    prs._agent_type_cache_by_id = {dynamic_id: "OpportunityMinerAgent"}
    prs._prompt_id_catalog = set()
    prs._policy_id_catalog = set()

    node = {
        "agent_id": dynamic_id,
        "agent_type": dynamic_id,
        "agent_property": {},
    }

    prs._enrich_node(
        node,
        {"opportunity_miner": "OpportunityMinerAgent"},
        {},
        {},
    )

    assert node["agent_type"] == "OpportunityMinerAgent"
    assert node["agent_property"]["llm"] == "legacy-llm"


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
    prs.update_agent_status(1, "A1", "completed")
    prs.update_agent_status(1, "A2", "validated")
    updated = json.loads(conn.cursor_obj.params[0])
    assert updated["agents"][1]["status"] == "validated"
    assert updated["agents"][0]["status"] == "completed"
    assert updated["agents"][2]["dependencies"]["onFailure"] == ["A1"]
    assert updated["status"] == "running"



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

    prs.update_agent_status(1, "A1", "completed")
    prs.update_agent_status(1, "A2", "completed")
    prs.update_agent_status(1, "A3", "completed")
    prs.update_agent_status(1, "A4", "validated")
    updated = json.loads(conn.cursor_obj.params[0])
    assert updated["agents"][3]["status"] == "validated"
    assert updated["agents"][2]["dependencies"]["onSuccess"] == ["A1"]
    assert updated["agents"][3]["dependencies"]["onFailure"] == ["A1", "A2"]
    assert updated["status"] == "running"



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
    assert holder["value"]["status"] == "running"

    prs.update_agent_status(1, "A2", "completed")
    assert holder["value"]["status"] == "completed"
    assert conn.cursor_obj.params[0] == 1

    prs.update_agent_status(1, "A1", "failed")
    assert holder["value"]["status"] == "failed"
    assert conn.cursor_obj.params[0] == -1


def test_update_agent_status_enforces_sequence():
    initial = {
        "status": "saved",
        "agents": [
            {"agent": "A1", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "1"},
            {"agent": "A2", "dependencies": {"onSuccess": ["A1"], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "2"},
        ],
    }
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs.get_process_details = lambda pid, **kwargs: initial
    with pytest.raises(ValueError):
        prs.update_agent_status(1, "A2", "validated")


def test_update_agent_status_allows_when_previous_started():
    initial = {
        "status": "saved",
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
    prs.get_process_details = lambda pid, **kwargs: initial
    prs.update_agent_status(1, "A2", "validated")
    updated = json.loads(conn.cursor_obj.params[0])
    assert updated["agents"][1]["status"] == "validated"


def test_update_agent_status_ignores_unrelated_predecessors():
    initial = {
        "status": "running",
        "agents": [
            {"agent": "Root", "dependencies": {"onSuccess": [], "onFailure": [], "onCompletion": []}, "status": "failed", "agent_ref_id": "root"},
            {"agent": "SuccessBranch", "dependencies": {"onSuccess": ["Root"], "onFailure": [], "onCompletion": []}, "status": "saved", "agent_ref_id": "success"},
            {"agent": "FailureBranch", "dependencies": {"onSuccess": [], "onFailure": ["Root"], "onCompletion": []}, "status": "saved", "agent_ref_id": "failure"},
        ],
    }
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)
    prs.get_process_details = lambda pid, **kwargs: initial

    prs.update_agent_status(1, "FailureBranch", "validated")
    params = conn.cursor_obj.params
    assert params[0] == -1
    updated = json.loads(params[1])
    assert updated["agents"][2]["status"] == "validated"


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


def test_update_process_status_accepts_textual_tokens():
    conn = DummyConn()
    agent = SimpleNamespace(
        get_db_connection=lambda: conn,
        settings=SimpleNamespace(script_user="tester"),
    )
    prs = ProcessRoutingService(agent)

    prs.update_process_status(5, "completed", process_details={"status": "saved", "agents": []})
    first_params = conn.cursor_obj.params
    assert first_params[0] == 1
    stored = json.loads(first_params[1])
    assert stored["status"] == "completed"

    prs.update_process_status(6, "failed", process_details={"status": "saved", "agents": []})
    second_params = conn.cursor_obj.params
    assert second_params[0] == -1
    stored_fail = json.loads(second_params[1])
    assert stored_fail["status"] == "failed"


def test_classify_completion_status_handles_tokens():
    success_numeric = ProcessRoutingService.classify_completion_status(100)
    assert success_numeric == (1, "completed", True)

    success_text = ProcessRoutingService.classify_completion_status("success")
    assert success_text == (1, "completed", True)

    failure_text = ProcessRoutingService.classify_completion_status("error")
    assert failure_text == (-1, "failed", True)

    unknown = ProcessRoutingService.classify_completion_status("maybe")
    assert unknown[0] == -1
    assert unknown[1] == "failed"
    assert unknown[2] is False


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
