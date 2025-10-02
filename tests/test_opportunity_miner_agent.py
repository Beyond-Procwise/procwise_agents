import json
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.opportunity_miner_agent import (
    OpportunityMinerAgent,
    Finding,
    _PURCHASE_LINE_VALUE_COLUMNS,
)
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine


def _opportunity_policy_rows():
    def row(pid: int, identifier: str, name: str, parameters: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None):
        details: Dict[str, Any] = {
            "policy_identifier": identifier,
            "rules": {"parameters": parameters},
        }
        if defaults:
            details["rules"]["default_conditions"] = defaults
        return {
            "policy_id": pid,
            "policy_name": name,
            "policy_type": "opportunity_mining",
            "policy_desc": identifier,
            "policy_details": json.dumps(details),
        }

    return [
        row(
            1,
            "oppfinderpolicy_001_price_benchmark_variance_detection",
            "Price Benchmark Variance",
            {"variance_threshold_pct": 0.05},
        ),
        row(
            3,
            "oppfinderpolicy_003_volume_consolidation",
            "Volume Consolidation",
            {"minimum_volume_gbp": 1000},
        ),
        row(
            4,
            "oppfinderpolicy_004_contract_expiry_opportunity",
            "Contract Expiry Opportunity",
            {"negotiation_window_days": 90},
            {"negotiation_window_days": 90},
        ),
        row(
            5,
            "oppfinderpolicy_005_supplier_risk_alert",
            "Supplier Risk Alert",
            {"risk_threshold": 0.7},
        ),
        row(
            6,
            "oppfinderpolicy_006_maverick_spend_detection",
            "Maverick Spend Detection",
            {"minimum_value_gbp": 5000},
        ),
        row(
            7,
            "oppfinderpolicy_007_duplicate_supplier",
            "Duplicate Supplier",
            {"minimum_overlap_gbp": 10000},
        ),
        row(
            8,
            "oppfinderpolicy_008_category_overspend",
            "Category Overspend",
            {"category_budgets": {"CatA": 100000}},
        ),
        row(
            9,
            "oppfinderpolicy_009_inflation_pass-through",
            "Inflation Pass-Through",
            {"market_inflation_pct": 0.02},
        ),
        row(
            10,
            "oppfinderpolicy_010_unused_contract_value",
            "Unused Contract Value",
            {"minimum_unused_value_gbp": 1000},
        ),
        row(
            11,
            "oppfinderpolicy_011_supplier_performance_deviation",
            "Supplier Performance Deviation",
            {"performance_records": {}},
        ),
        row(
            12,
            "oppfinderpolicy_012_esg_opportunity",
            "ESG Opportunity",
            {"esg_scores": []},
        ),
    ]


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine(policy_rows=_opportunity_policy_rows())
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


def test_build_finding_includes_policy_identifier():
    nick = DummyNick()
    nick.query_engine = None
    agent = OpportunityMinerAgent(nick)

    finding_a = agent._build_finding(
        "VolumeDiscountOpportunity",
        "SI0001",
        "CatA",
        "Item-1",
        100.0,
        {},
        ["PO1"],
        policy_id="policy_9",
        policy_name="Volume Discount Opportunity",
    )
    finding_b = agent._build_finding(
        "VolumeDiscountOpportunity",
        "SI0001",
        "CatA",
        "Item-1",
        100.0,
        {},
        ["PO1"],
        policy_id="policy_10",
        policy_name="Volume Discount Opportunity",
    )

    assert finding_a.policy_id == "policy_9"
    assert finding_b.policy_id == "policy_10"
    assert finding_a.opportunity_id != finding_b.opportunity_id
    assert finding_a.opportunity_id == "1"
    assert finding_b.opportunity_id == "2"
    assert "policy_9" in finding_a.opportunity_ref_id
    assert "policy_10" in finding_b.opportunity_ref_id


def test_policy_category_limits_deduplicate_same_supplier():
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    observed = datetime.now(timezone.utc)
    finding_low = Finding(
        opportunity_id="OPP-001",
        opportunity_ref_id="OPP-REF-001",
        detector_type="Volume Discount",
        supplier_id="SUP-1",
        category_id="CAT-1",
        item_id="ITEM-1",
        financial_impact_gbp=100.0,
        calculation_details={"source": "policy_a"},
        source_records=["rec-1"],
        detected_on=observed,
        policy_id="POL-A",
    )
    finding_high = Finding(
        opportunity_id="OPP-001",
        opportunity_ref_id="OPP-REF-001",
        detector_type="Supplier Consolidation",
        supplier_id="SUP-1",
        category_id="CAT-1",
        item_id="ITEM-1",
        financial_impact_gbp=250.0,
        calculation_details={"source": "policy_b"},
        source_records=["rec-2"],
        detected_on=observed,
        policy_id="POL-B",
    )

    per_policy = {
        "Volume Discount": [finding_low],
        "Supplier Consolidation": [finding_high],
    }

    aggregated, category_map = agent._apply_policy_category_limits(per_policy)

    assert len(aggregated) == 1
    chosen = aggregated[0]
    assert chosen.financial_impact_gbp == pytest.approx(250.0)
    assert sorted(chosen.source_records) == ["rec-1", "rec-2"]
    assert chosen.calculation_details.get("related_policies") == ["POL-A", "POL-B"]

    assert category_map["Volume Discount"]["all"] == [finding_low]
    assert category_map["Supplier Consolidation"]["all"] == [finding_high]


def test_build_finding_normalises_decimal_inputs():
    nick = DummyNick()
    nick.query_engine = None
    agent = OpportunityMinerAgent(nick)

    details = {"savings": Decimal("12.34"), "nested": {"delta": Decimal("1.23")}}
    sources = ["PO-1", Decimal("2")]

    finding = agent._build_finding(
        "SupplierConsolidationOpportunity",
        "SI0001",
        "Category-1",
        "Item-1",
        Decimal("99.50"),
        details,
        sources,
        policy_id="policy_12",
        policy_name="Supplier Consolidation",
    )

    assert isinstance(finding.financial_impact_gbp, float)
    assert finding.financial_impact_gbp == pytest.approx(99.5)
    assert isinstance(finding.calculation_details["savings"], float)
    assert finding.calculation_details["nested"]["delta"] == pytest.approx(1.23)
    assert all(isinstance(src, str) for src in finding.source_records)


def test_normalise_numeric_dataframe_converts_decimals():
    nick = DummyNick()
    nick.query_engine = None
    agent = OpportunityMinerAgent(nick)

    df = pd.DataFrame(
        {
            "line_total": [Decimal("10.50"), Decimal("5.25"), None],
            "po_id": ["PO-1", "PO-2", "PO-3"],
            "line_number": [1, 2, None],
        }
    )

    normalised = agent._normalise_numeric_dataframe(df.copy())

    assert normalised["line_total"].dtype == float
    assert normalised["line_total"].iloc[0] == pytest.approx(10.5)
    assert normalised["line_number"].dtype == float
    assert normalised["line_number"].iloc[1] == pytest.approx(2.0)


def test_normalise_currency_handles_decimal_sources():
    nick = DummyNick()
    nick.query_engine = None
    agent = OpportunityMinerAgent(nick)

    tables = {
        "purchase_orders": pd.DataFrame(
            [
                {
                    "po_id": "PO-1",
                    "currency": "USD",
                    "total_amount": Decimal("100.00"),
                }
            ]
        ),
        "indices": pd.DataFrame([
            {"currency": "USD", "value": Decimal("0.80")}
        ]),
    }

    normalised = agent._normalise_currency(tables)
    purchase_orders = normalised["purchase_orders"]

    assert "total_amount_gbp" in purchase_orders.columns
    assert pytest.approx(80.0) == purchase_orders.loc[0, "total_amount_gbp"]


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


def test_resolve_supplier_id_matches_supplier_aliases():
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    tables = {
        "supplier_master": pd.DataFrame(
            [
                {
                    "supplier_id": "SI0001",
                    "supplier_name": "Acme Industrial Ltd",
                    "trading_name": "ACME LTD",
                },
                {
                    "supplier_id": "SI0002",
                    "supplier_name": "Beta Manufacturing",
                },
            ]
        ),
        "contracts": pd.DataFrame(),
        "purchase_orders": pd.DataFrame(),
        "invoices": pd.DataFrame(),
    }

    agent._build_supplier_lookup(tables)

    assert agent._resolve_supplier_id("SI0001") == "SI0001"
    assert agent._resolve_supplier_id("acme industrial ltd") == "SI0001"
    assert agent._resolve_supplier_id("ACME LTD") == "SI0001"
    assert agent._resolve_supplier_id("Beta Manufacturing") == "SI0002"
    assert agent._resolve_supplier_id("Unknown Supplier") is None


def test_choose_first_column_handles_line_total_column():
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    df = pd.DataFrame({"po_id": ["PO1"], "item_id": ["Item"], "line_total": [100.0]})

    selected = agent._choose_first_column(df, _PURCHASE_LINE_VALUE_COLUMNS)
    assert selected == "line_total"


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

def test_instruction_overrides_populate_price_variance_fields(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context("price_variance_check", {})

    instructions = {
        "supplier_id": "SI0001",
        "item_id": "ITM-001",
        "actual_price": "11.25",
        "benchmark_price": "9.50",
        "variance_threshold_pct": "0.10",
        "quantity": "15",
    }

    agent._apply_instruction_overrides(context, instructions)

    conditions = context.input_data["conditions"]
    assert conditions["supplier_id"] == "SI0001"
    assert conditions["item_id"] == "ITM-001"
    assert conditions["actual_price"] == 11.25
    assert conditions["benchmark_price"] == 9.50
    assert conditions["variance_threshold_pct"] == 0.10
    assert conditions["quantity"] == 15


def test_price_variance_uses_top_level_fields_when_conditions_missing(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context("price_variance_check", {})

    context.input_data.update(
        {
            "supplier_id": "SI0001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        }
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    conditions = context.input_data["conditions"]
    assert conditions["supplier_id"] == "SI0001"
    assert conditions["item_id"] == "ITM-001"
    assert conditions["actual_price"] == 11.0


def test_price_variance_accepts_camel_case_conditions(monkeypatch):
    agent = create_agent(monkeypatch)
    context = AgentContext(
        workflow_id="wf-camel",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={
            "workflow": "price_variance_check",
            "conditions": {},
            "supplierId": "SI0001",
            "itemId": "ITM-001",
            "actualPrice": "11.50",
            "benchmarkPrice": "9.25",
            "quantity": "12",
            "varianceThresholdPct": "0.05",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    conditions = context.input_data["conditions"]
    assert conditions["supplier_id"] == "SI0001"
    assert conditions["item_id"] == "ITM-001"
    assert conditions["actual_price"] == 11.5
    assert conditions["benchmark_price"] == 9.25
    assert conditions["quantity"] == 12.0
    assert conditions["variance_threshold_pct"] == 0.05


def test_policy_conditions_merge_into_context(monkeypatch):
    agent = create_agent(monkeypatch)
    policy_payload = {
        "policyId": 9,
        "policyName": "Price Benchmark Variance",
        "policy_desc": "oppfinderpolicy_001_price_benchmark_variance_detection",
        "conditions": {
            "supplierId": "SI0001",
            "itemId": "ITM-001",
            "actualPrice": 11.0,
            "benchmarkPrice": 9.0,
            "quantity": 10,
            "varianceThresholdPct": 0.05,
        },
    }

    context = AgentContext(
        workflow_id="wf-policy",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={
            "workflow": "price_variance_check",
            "conditions": {},
            "policies": [policy_payload],
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    conditions = context.input_data["conditions"]
    assert conditions["supplier_id"] == "SI0001"
    assert conditions["benchmark_price"] == 9.0
    assert any(
        f["detector_type"] == "Price Benchmark Variance"
        for f in output.data["findings"]
    )
    assert conditions["benchmark_price"] == 9.0


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
        input_data={"conditions": {"actual_price": 1.0}},
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
    assert "supplier_directory" not in output.data
    assert "policy_metadata" not in output.data
    assert "policy_top_opportunities" not in output.data
    assert "data_profile" not in output.data
    assert "data_flow_snapshot" not in output.data

    assert any(evt["status"] == "escalated" for evt in output.data["policy_events"])


def test_price_variance_blocks_when_supplier_missing(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "price_variance_check",
        {
            "actual_price": 12.5,
            "benchmark_price": 9.5,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert not output.data["findings"]
    events = output.data.get("policy_events") or []
    assert events
    blocked = [evt for evt in events if evt["status"] == "blocked"]
    assert blocked
    message = blocked[0]["message"]
    assert "supplier" in message.lower()
    details = blocked[0].get("details", {})
    assert "supplier_id" in details.get("missing_fields", [])


def test_price_variance_infers_supplier_from_contract(monkeypatch):
    tables = _sample_tables()
    agent = create_agent(monkeypatch, tables)
    context = build_context(
        "price_variance_check",
        {
            "contract_id": "CO000001",
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    findings = [
        f for f in output.data["findings"] if f["detector_type"] == "Price Benchmark Variance"
    ]
    assert findings
    assert findings[0]["supplier_id"] == "SI0001"
    assert context.input_data["conditions"]["supplier_id"] == "SI0001"


def test_price_variance_infers_supplier_from_item(monkeypatch):
    tables = _sample_tables()
    agent = create_agent(monkeypatch, tables)
    context = build_context(
        "price_variance_check",
        {
            "item_id": "ITM-001",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    findings = [
        f
        for f in output.data["findings"]
        if f["detector_type"] == "Price Benchmark Variance"
    ]
    assert findings
    assert findings[0]["supplier_id"] == "SI0001"
    assert context.input_data["conditions"]["supplier_id"] == "SI0001"


def test_price_variance_infers_supplier_from_description(monkeypatch):
    tables = _sample_tables()
    agent = create_agent(monkeypatch, tables)
    context = build_context(
        "price_variance_check",
        {
            "item_description": "Logistics Support",
            "actual_price": 11.0,
            "benchmark_price": 9.0,
            "quantity": 10,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    findings = [
        f
        for f in output.data["findings"]
        if f["detector_type"] == "Price Benchmark Variance"
    ]
    assert findings
    assert findings[0]["supplier_id"] == "SI0001"
    assert context.input_data["conditions"]["supplier_id"] == "SI0001"


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
    supplier_gaps = output.data.get("policy_supplier_gaps")
    assert supplier_gaps is not None
    for name, suppliers in policy_suppliers.items():
        if suppliers:
            assert name not in supplier_gaps
    assert output.data.get("policy_opportunities")
    category_map = output.data.get("policy_category_opportunities")
    assert category_map
    for categories in category_map.values():
        for entries in categories.values():
            if entries:
                assert 1 <= len(entries) <= 2
    assert output.data.get("supplier_candidates")
    assert "policy_top_opportunities" not in output.data
    assert "data_flow_snapshot" not in output.data


def test_policy_supplier_gap_reason_when_no_suppliers(monkeypatch):
    tables = _sample_tables()
    agent = create_agent(monkeypatch, tables)
    context = build_context(
        "price_variance_check",
        {
            "supplier_id": "SI9999",
            "item_id": "ITM-999",
            "actual_price": 10.0,
            "benchmark_price": 9.0,
            "quantity": 5,
            "variance_threshold_pct": 0.05,
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    suppliers_by_policy = output.data.get("policy_suppliers")
    assert suppliers_by_policy
    for suppliers in suppliers_by_policy.values():
        assert isinstance(suppliers, list)
        assert len(suppliers) == 0
    supplier_gaps = output.data.get("policy_supplier_gaps")
    assert supplier_gaps
    reason = next(iter(supplier_gaps.values()))
    assert isinstance(reason, str)
    assert reason



def test_policy_category_limits_caps_results():
    agent = OpportunityMinerAgent(DummyNick())
    now = datetime.utcnow()

    per_policy = {
        "PolicyA": [
            Finding(
                opportunity_id="a1",
                detector_type="Test",
                supplier_id="SI001",
                category_id="CatA",
                item_id="Item1",
                financial_impact_gbp=120.0,
                calculation_details={},
                source_records=["PO1"],
                detected_on=now,
            ),
            Finding(
                opportunity_id="a2",
                detector_type="Test",
                supplier_id="SI002",
                category_id="CatA",
                item_id="Item2",
                financial_impact_gbp=110.0,
                calculation_details={},
                source_records=["PO2"],
                detected_on=now,
            ),
            Finding(
                opportunity_id="a3",
                detector_type="Test",
                supplier_id="SI003",
                category_id="CatA",
                item_id="Item3",
                financial_impact_gbp=90.0,
                calculation_details={},
                source_records=["PO3"],
                detected_on=now,
            ),
            Finding(
                opportunity_id="b1",
                detector_type="Test",
                supplier_id="SI004",
                category_id="CatB",
                item_id="Item4",
                financial_impact_gbp=150.0,
                calculation_details={},
                source_records=["PO4"],
                detected_on=now,
            ),
        ]
    }

    aggregated, category_map = agent._apply_policy_category_limits(per_policy)

    assert "PolicyA" in category_map
    categories = category_map["PolicyA"]
    assert set(categories.keys()) == {"all", "CatA", "CatB"}
    assert [f.opportunity_id for f in categories["all"]] == ["b1", "a1"]
    assert len(categories["CatA"]) == 3
    assert len(categories["CatB"]) == 1
    aggregated_ids = {finding.opportunity_id for finding in aggregated}
    assert aggregated_ids == {"a1", "b1"}
    retained_ids = {finding.opportunity_id for finding in per_policy["PolicyA"]}
    assert retained_ids == aggregated_ids


def test_volume_consolidation_identifies_costlier_supplier(monkeypatch):
    agent = create_agent(monkeypatch)
    context = build_context(
        "volume_consolidation_check", {"minimum_volume_gbp": 50}

    )

    output = agent.run(context)

    vc = [f for f in output.data["findings"] if f["detector_type"] == "Volume Consolidation"]
    assert vc
    assert all(f["supplier_id"] for f in vc)


def test_volume_consolidation_uses_line_level_supplier_ids(monkeypatch):
    tables = _sample_tables()
    purchase_orders = tables["purchase_orders"].drop(columns=["supplier_id"])
    purchase_orders = purchase_orders.assign(contract_id=None)
    tables["purchase_orders"] = purchase_orders
    po_lines = tables["purchase_order_lines"].copy()
    po_lines["supplier_id"] = ["SI0001", "SI0002", "SI0002"]
    tables["purchase_order_lines"] = po_lines

    agent = create_agent(monkeypatch, tables)
    context = build_context(
        "volume_consolidation_check", {"minimum_volume_gbp": 50}
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    vc = [
        f
        for f in output.data["findings"]
        if f["detector_type"] == "Volume Consolidation"
    ]
    assert vc
    assert all(f["supplier_id"] for f in vc)


def test_volume_consolidation_handles_misaligned_po_ids(monkeypatch):
    tables = _sample_tables()
    po_lines = tables["purchase_order_lines"].copy()
    po_lines["po_id"] = po_lines["po_id"].str.lower() + "  "
    tables["purchase_order_lines"] = po_lines

    agent = create_agent(monkeypatch, tables)
    context = build_context(
        "volume_consolidation_check", {"minimum_volume_gbp": 50}
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    vc = [
        f
        for f in output.data["findings"]
        if f["detector_type"] == "Volume Consolidation"
    ]
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


def test_apply_instruction_settings_updates_context():
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    context = AgentContext(
        workflow_id="wf",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={
            "prompts": [
                {
                    "promptId": 1,
                    "prompts_desc": "workflow: custom_flow\nmin_financial_impact: 2500\nlookback_period_days: 45",
                }
            ],
            "policies": [
                {
                    "policyId": 4,
                    "policy_desc": "{\"parameters\": {\"negotiation_window_days\": 60}}",
                }
            ],
        },
    )

    agent._apply_instruction_settings(context)

    assert context.input_data["workflow"] == "custom_flow"
    assert context.input_data["min_financial_impact"] == 2500
    assert context.input_data["conditions"]["lookback_period_days"] == 45
    assert context.input_data["conditions"]["negotiation_window_days"] == 60


def test_assemble_policy_registry_filters_static_entries():
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    input_data = {
        "policies": [
            {
                "policyId": "oppfinderpolicy_004_contract_expiry_opportunity",
                "policyName": "Contract Expiry",
                "policy_desc": "{\"parameters\": {\"negotiation_window_days\": 75}}",
            }
        ]
    }

    registry, provided = agent._assemble_policy_registry(input_data)
    assert len(provided) == 1
    assert list(registry.keys()) == ["contract_expiry_check"]
    entry = registry["contract_expiry_check"]
    assert entry["policy_id"] == "oppfinderpolicy_004_contract_expiry_opportunity"
    assert entry.get("parameters", {}).get("negotiation_window_days") == 75


def test_registry_matches_numeric_policy_identifiers_without_catalog():
    class CataloglessNick:
        def __init__(self):
            self.settings = SimpleNamespace(script_user="tester")
            self.prompt_engine = SimpleNamespace()
            self.policy_engine = None
            self.process_routing_service = SimpleNamespace(
                log_process=lambda **kwargs: None,
                log_run_detail=lambda **kwargs: None,
                log_action=lambda **kwargs: None,
                update_process_status=lambda **kwargs: None,
            )

    agent = OpportunityMinerAgent(CataloglessNick())

    input_data = {
        "policies": [
            {
                "policyId": 9,
                "policyName": "oppfinderpolicy_001_price_benchmark_variance_detection",
                "policy_desc": "Identify suppliers charging above benchmark.",
            },
            {
                "policyId": 10,
                "policyName": "oppfinderpolicy_003_volume_consolidation",
                "policy_desc": "Identify consolidation opportunities across multiple suppliers.",
            },
        ]
    }

    registry, provided = agent._assemble_policy_registry(dict(input_data))

    assert {"price_variance_check", "volume_consolidation_check"} == set(registry.keys())
    first_entry = registry["price_variance_check"]
    assert first_entry["policy_id"] == "oppfinderpolicy_001_price_benchmark_variance_detection"
    assert first_entry["policy_slug"] == "price_variance_check"
    second_entry = registry["volume_consolidation_check"]
    assert second_entry["policy_id"] == "oppfinderpolicy_003_volume_consolidation"
    assert second_entry["policy_slug"] == "volume_consolidation_check"
    assert len(provided) == 2

def test_opportunity_miner_resolves_supplier_from_db(monkeypatch):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    draft_calls = {"count": 0}

    def fake_lookup(rfq_id):
        draft_calls["count"] += 1
        return "SUP-DB"

    monkeypatch.setattr(agent, "_lookup_supplier_from_drafts", fake_lookup)
    monkeypatch.setattr(agent, "_lookup_supplier_from_processed", lambda rfq: None)

    def boom_ingest():
        raise RuntimeError("ingest stop")

    monkeypatch.setattr(agent, "_ingest_data", boom_ingest)

    context = AgentContext(
        workflow_id="wf1",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"rfq_id": "RFQ-20240101-XYZ", "conditions": {}},
    )

    output = agent.run(context)

    assert draft_calls["count"] == 1
    assert context.input_data["supplier_id"] == "SUP-DB"
    assert output.status == AgentStatus.FAILED
    assert "skipped" not in output.data
    assert output.error == "ingest stop"


def test_opportunity_miner_skips_without_supplier(monkeypatch, caplog):
    nick = DummyNick()
    agent = OpportunityMinerAgent(nick)

    monkeypatch.setattr(agent, "_lookup_supplier_from_drafts", lambda rfq: None)
    monkeypatch.setattr(agent, "_lookup_supplier_from_processed", lambda rfq: None)

    ingest_called = {"value": False}

    def mark_ingest():
        ingest_called["value"] = True
        return {}

    monkeypatch.setattr(agent, "_ingest_data", mark_ingest)

    context = AgentContext(
        workflow_id="wf-skip",
        agent_id="opportunity_miner",
        user_id="tester",
        input_data={"rfq_id": "RFQ-20240101-MISS"},
    )

    with caplog.at_level(logging.INFO):
        output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data == {"skipped": True, "reason": "missing_supplier_id"}
    assert ingest_called["value"] is False
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
