from services.process_routing_service import ProcessRoutingService


def test_normalize_process_details_scenario1():
    raw = {
        "agents": [
            {"agent": "A1"},
            {"agent": "A2", "dependencies": {"onSuccess": ["A1"]}},
            {"agent": "A3", "dependencies": {"onFailure": ["A1"]}},
        ],
    }

    details = ProcessRoutingService.normalize_process_details(raw)

    assert details["status"] == ""
    assert details["process_status"] == 0
    assert len(details["agents"]) == 3
    assert details["agents"][0]["dependencies"] == {
        "onSuccess": [],
        "onFailure": [],
        "onCompletion": [],
    }
    assert details["agents"][1]["dependencies"]["onSuccess"] == ["A1"]
    assert details["agents"][2]["dependencies"]["onFailure"] == ["A1"]
    assert all(agent["status"] == "saved" for agent in details["agents"])
    assert all("agent_ref_id" in agent for agent in details["agents"])


def test_normalize_process_details_scenario2():
    raw = {
        "status": "saved",
        "agents": [
            {"agent": "A1"},
            {"agent": "A2"},
            {"agent": "A3", "dependencies": {"onSuccess": ["A1"]}},
            {"agent": "A4", "dependencies": {"onFailure": ["A1", "A2"]}},
        ],
    }

    details = ProcessRoutingService.normalize_process_details(raw)

    assert details["status"] == "saved"
    assert details["process_status"] == 0
    assert len(details["agents"]) == 4
    assert details["agents"][2]["dependencies"]["onSuccess"] == ["A1"]
    assert details["agents"][3]["dependencies"]["onFailure"] == ["A1", "A2"]
