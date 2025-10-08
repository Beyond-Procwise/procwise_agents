import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.agent_manifest import AgentManifestService


def _make_nick(policies=None):
    policy_engine = SimpleNamespace(_policies=policies or [])
    return SimpleNamespace(policy_engine=policy_engine)


def test_manifest_exposes_task_policy_and_knowledge():
    policies = [
        {
            "policyId": "101",
            "policyName": "Supplier Quality Policy",
            "policy_desc": "Assess supplier performance",
            "policy_type": "governance",
            "policy_linked_agents": ["supplier_ranking"],
        }
    ]
    nick = _make_nick(policies)
    manifest_service = AgentManifestService(nick)

    manifest = manifest_service.build_manifest("SupplierRankingAgent")

    task = manifest["task"]
    assert task["agent_type"].lower().startswith("suppli")
    assert "dependencies" in task

    policy_bundle = manifest["policies"]
    assert any(policy["policyId"] == "101" for policy in policy_bundle)

    knowledge = manifest["knowledge"]
    assert "proc.invoice_agent" in knowledge["tables"]
    assert any(
        rel["from"] == "proc.contracts.supplier_id" and rel["to"] == "proc.supplier.supplier_id"
        for rel in knowledge["relationships"]
    )


def test_manifest_for_unknown_agent_returns_defaults():
    nick = _make_nick()
    manifest_service = AgentManifestService(nick)

    manifest = manifest_service.build_manifest("custom_test_agent")

    assert manifest["task"]["agent_type"] == "custom_test_agent"
    assert isinstance(manifest["policies"], list)
    assert "proc.supplier" in manifest["knowledge"]["tables"]
