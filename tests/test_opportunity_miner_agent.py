import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.opportunity_miner_agent import OpportunityMinerAgent
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.settings = SimpleNamespace(script_user="tester")


def test_price_expression_falls_back_to_unit_price(monkeypatch):
    nick = DummyNick()
    nick.query_engine = None
    agent = OpportunityMinerAgent(nick)

    monkeypatch.setattr(
        agent, "_get_table_columns", lambda schema, table: {"unit_price"}
    )

    expr = agent._price_expression("proc", "po_line_items_agent", "li")
    assert expr == "li.unit_price"


def _sample_tables() -> Dict[str, Any]:
    purchase_orders = pd.DataFrame(
        [
            {
                "po_id": "PO000001",
                "supplier_id": "SI0001",
                "currency": "GBP",
                "total_amount": 100.0,
                "payment_terms": "15",
                "contract_id": "CO000001",
            },
            {
                "po_id": "PO000002",
                "supplier_id": "SI0002",
                "currency": "GBP",
                "total_amount": 80.0,
                "payment_terms": "30",
                "contract_id": "CO000002",
            },
            {
                "po_id": "PO000003",
                "supplier_id": "SI0002",
                "currency": "GBP",
                "total_amount": 150.0,
                "payment_terms": "45",
                "contract_id": None,
            },
        ]
    )

    invoices = pd.DataFrame(
        [
            {
                "invoice_id": "IN000001",
                "po_id": "PO000001",
                "supplier_id": "SI0001",
                "currency": "GBP",
                "invoice_amount": 110.0,
                "invoice_total_incl_tax": 110.0,
                "payment_terms": "15",
            },
            {
                "invoice_id": "IN000002",
                "po_id": "PO000002",
                "supplier_id": "SI0002",
                "currency": "GBP",
                "invoice_amount": 96.0,
                "invoice_total_incl_tax": 96.0,
                "payment_terms": "30",
            },
        ]
    )

    purchase_order_lines = pd.DataFrame(
        [
            {
                "po_line_id": "POL000001",
                "po_id": "PO000001",
                "item_id": "ITM-001",
                "item_description": "Logistics Support",
                "quantity": 10,
                "unit_price": 10.0,
                "line_amount": 100.0,
                "total_amount_incl_tax": 100.0,
                "currency": "GBP",
            },
            {
                "po_line_id": "POL000002",
                "po_id": "PO000002",
                "item_id": "ITM-001",
                "item_description": "Logistics Support",
                "quantity": 5,
                "unit_price": 12.0,
                "line_amount": 60.0,
                "total_amount_incl_tax": 60.0,
                "currency": "GBP",
            },
            {
                "po_line_id": "POL000003",
                "po_id": "PO000003",
                "item_id": "ITM-002",
                "item_description": "Adhoc Consulting",
                "quantity": 3,
                "unit_price": 50.0,
                "line_amount": 150.0,
                "total_amount_incl_tax": 150.0,
                "currency": "GBP",
            },
        ]
    )

    invoice_lines = pd.DataFrame(
        [
            {
                "invoice_line_id": "INL000001",
                "invoice_id": "IN000001",
                "po_id": "PO000001",
                "item_id": "ITM-001",
                "item_description": "Logistics Support",
                "quantity": 10,
                "unit_price": 11.0,
                "line_amount": 110.0,
                "total_amount_incl_tax": 110.0,
                "currency": "GBP",
            },
            {
                "invoice_line_id": "INL000002",
                "invoice_id": "IN000002",
                "po_id": "PO000002",
                "item_id": "ITM-001",
                "item_description": "Logistics Support",
                "quantity": 5,
                "unit_price": 13.0,
                "line_amount": 65.0,
                "total_amount_incl_tax": 65.0,
                "currency": "GBP",
            },
        ]
    )

    contracts = pd.DataFrame(
        [
            {
                "contract_id": "CO000001",
                "contract_title": "Contract 1",
                "contract_type": "Services",
                "supplier_id": "SI0001",
                "buyer_org_id": "BUY1",
                "contract_start_date": pd.Timestamp("2024-01-01"),
                "contract_end_date": pd.Timestamp("2024-09-30"),
                "currency": "GBP",
                "total_contract_value": 100.0,
                "spend_category": "CatA",
                "payment_terms": "15",
            },
            {
                "contract_id": "CO000002",
                "contract_title": "Contract 2",
                "contract_type": "Goods",
                "supplier_id": "SI0002",
                "buyer_org_id": "BUY1",
                "contract_start_date": pd.Timestamp("2024-01-01"),
                "contract_end_date": pd.Timestamp("2024-12-31"),
                "currency": "GBP",
                "total_contract_value": 300.0,
                "spend_category": "CatA",
                "payment_terms": "30",
            },
        ]
    )

    indices = pd.DataFrame(
        [
            {
                "index_name": "FX_GBP",
                "value": 1.0,
                "effective_date": "2024-01-01",
                "currency": "GBP",
            }
        ]
    )

    shipments = pd.DataFrame(
        [
            {
                "shipment_id": "SH1",
                "po_id": "PO1",
                "logistics_cost": 5.0,
                "currency": "GBP",
                "delivery_date": "2024-01-10",
            }
        ]
    )

    supplier_master = pd.DataFrame(
        [
            {
                "supplier_id": "SI0001",
                "supplier_name": "Supplier One",
                "risk_score": 0.8,
            },
            {
                "supplier_id": "SI0002",
                "supplier_name": "Supplier Two",
                "risk_score": 0.2,
            },
            {
                "supplier_id": "SI0003",
                "supplier_name": "Supplier Three",
                "risk_score": 0.65,
            },
        ]
    )

    product_mapping = pd.DataFrame(
        [
            {
                "product": "Logistics Support - Catalog",
                "category_level_1": "Logistics",
                "category_level_2": "Support",
                "category_level_3": None,
                "category_level_4": None,
                "category_level_5": None,
            },
            {
                "product": "Adhoc Consulting Services",
                "category_level_1": "Professional Services",
                "category_level_2": "Consulting",
                "category_level_3": None,
                "category_level_4": None,
                "category_level_5": None,
            },
        ]
    )

    return {
        "purchase_orders": purchase_orders,
        "purchase_order_lines": purchase_order_lines,
        "invoices": invoices,
        "invoice_lines": invoice_lines,
        "product_mapping": product_mapping,
        "contracts": contracts,
        "indices": indices,
        "shipments": shipments,
        "supplier_master": supplier_master,
    }

    assert output.status == AgentStatus.FAILED
    assert output.data["blocked_reason"]
    assert output.data["policy_events"]


def test_price_variance_detection_generates_finding(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "price_variance_check",
        {
            "supplier_id": "SI0001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

def create_agent(monkeypatch, tables: Optional[Dict[str, Any]] = None):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick, min_financial_impact=0)
    monkeypatch.setattr(agent, "_output_excel", lambda findings: None)
    monkeypatch.setattr(agent, "_output_feed", lambda findings: None)
    data = tables if tables is not None else _sample_tables()
    monkeypatch.setattr(
        agent,
        "_ingest_data",
        lambda: {name: df.copy() for name, df in data.items()},
    )
    return agent


def build_context(workflow: str, conditions: Optional[Dict[str, Any]]) -> AgentContext:
    return AgentContext(
        workflow_id="wf-test",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"workflow": workflow, "conditions": conditions or {}},
    )


def test_missing_workflow_blocks_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = AgentContext(
        workflow_id="wf-missing",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"conditions": {}},
    )

    output = agent.run(context)

    assert output.status == AgentStatus.FAILED
    assert output.data["blocked_reason"]
    assert output.data["policy_events"]


def test_price_variance_detection_generates_finding(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "price_variance_check",
        {
            "supplier_id": "SI0001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    pb = [f for f in output.data["findings"] if f["detector_type"] == "Price Benchmark Variance"]
    assert pb
    assert pb[0]["supplier_name"] == "Supplier One"
    directory = output.data.get("supplier_directory")
    assert directory
    assert any(
        entry["supplier_id"] == "SI0001" and entry.get("supplier_name") == "Supplier One"
        for entry in directory
    )
    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_candidate_supplier_lookup_uses_original_item(monkeypatch):
    agent = create_agent(monkeypatch)
    calls: list[Any] = []

    def fake_find(item_id, supplier_id, sources):  # pragma: no cover - simple stub
        calls.append(item_id)
        return []

    monkeypatch.setattr(agent, "_find_candidate_suppliers", fake_find)

    context = build_context(
        "price_variance_check",
        {
            "supplier_id": "SI0001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert calls
    assert all(call == "ITM-001" for call in calls)
    finding = next(
        f for f in output.data["findings"] if f["detector_type"] == "Price Benchmark Variance"
    )
    assert finding["item_id"] == "Logistics Support - Catalog"
    assert finding.get("item_reference") == "ITM-001"


def test_min_financial_impact_override_filters_findings(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "price_variance_check",
        {
            "supplier_id": "SI0001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )
    context.input_data["min_financial_impact"] = 50.0

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["opportunity_count"] == 0


def test_dynamic_policies_surface_opportunities(monkeypatch):
    tables = _sample_tables()
    agent = create_agent(monkeypatch, tables)
    policies = [
        {
            "policyId": 201,
            "policyName": "VolumeDiscountOpportunity",
            "details": {
                "rules": {
                    "parameters": {
                        "minimum_quantity_threshold": 5,
                        "minimum_spend_threshold": 50,
                        "target_discount_pct": 10,
                    }
                }
            },
        },
        {
            "policyId": 202,
            "policyName": "SupplierConsolidationOpportunity",
            "details": {
                "rules": {
                    "parameters": {
                        "minimum_supplier_count": 2,
                        "consolidation_savings_pct": 8,
                    }
                }
            },
        },
    ]
    context = AgentContext(
        workflow_id="wf-dynamic",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"policies": policies},
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    detectors = {finding["detector_type"] for finding in output.data["findings"]}
    assert "VolumeDiscountOpportunity" in detectors
    assert "SupplierConsolidationOpportunity" in detectors
    policy_suppliers = output.data.get("policy_suppliers")
    assert policy_suppliers
    for name in ("VolumeDiscountOpportunity", "SupplierConsolidationOpportunity"):
        suppliers = policy_suppliers.get(name)
        assert suppliers
        assert any(supplier.startswith("SI") for supplier in suppliers)
    assert output.data.get("policy_opportunities")
    assert output.data.get("supplier_candidates")


def test_volume_consolidation_identifies_costlier_supplier(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "volume_consolidation_check", {"minimum_volume_gbp": 50}

    )

    output = agent.run(context)

    vc = [f for f in output.data["findings"] if f["detector_type"] == "Volume Consolidation"]
    assert vc
    assert all(f["supplier_id"] for f in vc)


def test_contract_expiry_injects_default_window(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "contract_expiry_check", {"reference_date": "2024-07-02"}
    )
    context.input_data["conditions"].pop("negotiation_window_days", None)

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    findings = [
        f
        for f in output.data["findings"]
        if f["detector_type"] == "Contract Expiry Opportunity"
    ]
    assert findings
    assert context.input_data["conditions"]["negotiation_window_days"] == 90


def test_contract_expiry_fallback_surfaces_top_suppliers(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "contract_expiry_check",
        {"reference_date": "2030-01-01", "negotiation_window_days": 30},
    )
    context.input_data["ranking"] = [
        {"supplier_id": "SI0001", "final_score": 19.5, "justification": "Top"},
        {"supplier_id": "SI0002", "final_score": 18.7, "justification": "Strong"},
    ]

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    findings = [
        f
        for f in output.data["findings"]
        if f["detector_type"] == "Contract Expiry Opportunity - Portfolio"
    ]
    assert findings
    assert all(
        f["calculation_details"].get("analysis_type") == "contract_portfolio_fallback"
        for f in findings
    )
    assert any(evt["status"] == "fallback" for evt in output.data["policy_events"])


def test_supplier_risk_alert_threshold(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "supplier_risk_check",
        {"risk_threshold": 0.5, "risk_weight": 1000},

    )

    output = agent.run(context)

    alerts = [f for f in output.data["findings"] if f["detector_type"] == "Supplier Risk Alert"]
    assert alerts
    assert alerts[0]["supplier_id"] == "SI0001"
    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_maverick_spend_detection_flags_po(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "maverick_spend_check", {"minimum_value_gbp": 120}

    )

    output = agent.run(context)

    findings = [
        f for f in output.data["findings"] if f["detector_type"] == "Maverick Spend Detection"
    ]
    assert findings
    assert findings[0]["supplier_id"] == "SI0002"


def test_category_overspend_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "category_overspend_check", {"category_budgets": {"CatA": 90}}
    )

    output = agent.run(context)

    overspend = [
        f for f in output.data["findings"] if f["detector_type"] == "Category Overspend"
    ]
    assert overspend
    assert overspend[0]["supplier_id"] == "SI0001"


def test_unused_contract_value_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "unused_contract_value_check", {"minimum_unused_value_gbp": 50}

    )

    output = agent.run(context)

    unused = [
        f for f in output.data["findings"] if f["detector_type"] == "Unused Contract Value"
    ]
    assert unused
    assert unused[0]["supplier_id"] == "SI0002"
    assert unused[0]["supplier_name"] == "Supplier Two"


def test_esg_opportunity_creates_event(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "esg_opportunity_check",
        {
            "esg_scores": [{"supplier_id": "SI0003", "score": 0.9}],
            "incumbent_score": 0.6,
            "esg_threshold": 0.8,
            "estimated_switch_savings_gbp": 2000,
            "category_id": "CatA",
        },
    )

    output = agent.run(context)

    esg = [f for f in output.data["findings"] if f["detector_type"] == "ESG Opportunity"]
    assert esg
    assert esg[0]["supplier_id"] == "SI0003"
    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_inflation_passthrough_detection(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "inflation_passthrough_check",
        {"market_inflation_pct": 0.02, "tolerance_pct": 0.0},
    )

    output = agent.run(context)

    inflation = [
        f for f in output.data["findings"] if f["detector_type"] == "Inflation Pass-Through"
    ]
    assert inflation
    assert inflation[0]["supplier_id"] == "SI0001"
