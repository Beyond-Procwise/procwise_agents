import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator


def make_orchestrator():
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model={}),
    )
    return Orchestrator(nick)


def test_canonical_key_handles_decorated_names():
    orch = make_orchestrator()
    defs = orch._load_agent_definitions()

    slug1 = orch._canonical_key("SupplierRankingAgent", defs)
    slug2 = orch._canonical_key("user_supplier_ranking_agent_1", defs)

    assert slug1 == "supplier_ranking"
    assert slug2 == "supplier_ranking"


def test_get_agent_details_parses_array_string():
    orch = make_orchestrator()
    details = orch._get_agent_details("{supplier_ranking,quote_evaluation}")
    types = {d["agent_type"] for d in details}
    assert {"supplier_ranking", "quote_evaluation"} <= types


def test_canonical_key_fuzzy_matches_dynamic_ids():
    orch = make_orchestrator()
    defs = orch._load_agent_definitions()

    slug_quotes = orch._canonical_key(
        "keerthi_quotes_agent_test_000074_1757585776760", defs
    )
    slug_negotiation = orch._canonical_key(
        "keerthi_negotitiation_test_000078_1757586202506", defs
    )

    assert slug_quotes == "quote_evaluation"
    assert slug_negotiation == "negotiation"
