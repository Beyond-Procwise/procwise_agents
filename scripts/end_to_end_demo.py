"""End-to-end demonstration harness for the ProcWise knowledge graph stack."""
from __future__ import annotations

import logging
import os
from pprint import pprint

from procurement_knowledge_graph.config import AppConfig
from procurement_knowledge_graph.procurement_knowledge_graph import ProcurementKnowledgeGraph
from procurement_knowledge_graph.procwise_kg_integration import ProcWiseKnowledgeGraphAgent
from services.email_thread import EmailThread, make_action_id

logging.basicConfig(level=os.environ.get("PROCWISE_LOG_LEVEL", "INFO"))


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_demo() -> None:
    """Load source systems, refresh the KG, and showcase high-level queries."""

    config = AppConfig.from_env()
    logging.info("Refreshing procurement knowledge graph across Postgres, Neo4j, and Qdrant")
    builder = ProcurementKnowledgeGraph.from_config(config)
    try:
        builder.full_refresh()
    finally:
        builder.close()

    logging.info("Running hybrid query, negotiation readiness, and strategy samples")
    agent = ProcWiseKnowledgeGraphAgent(config)
    try:
        readiness = agent.should_trigger_negotiation(
            "workflow-123",
            rfq_id="rfq-456",
            min_responses_required=3,
        )
        _print_section("Negotiation Readiness")
        pprint(readiness)

        strategy = agent.generate_negotiation_strategy(
            rfq_id="rfq-456",
            supplier_id="SUP-101",
            target_price=95_000,
            current_quote=100_000,
        )
        _print_section("Strategy Proposal")
        pprint(strategy["context"])
        pprint(strategy["strategy"])

        answer = agent.engine.ask(
            "Summarise open supplier invoices this quarter.",
            filters={"label": "Invoice"},
        )
        _print_section("Semantic Answer")
        pprint(answer)
    finally:
        agent.close()

    thread = EmailThread(workflow_id="workflow-123", supplier_id="SUP-101")
    dispatch_id = make_action_id(0, thread.supplier_unique_id)
    thread.add_message(
        "dispatch_round_0",
        "Initial RFQ issued with Net 30 terms and 12-day target lead time.",
        dispatch_id,
        round_num=0,
    )
    response_id = make_action_id(1, thread.supplier_unique_id)
    thread.add_message(
        "supplier_response_round_1",
        "Supplier confirms shipment in 10 business days with a 3% discount.",
        response_id,
        round_num=1,
    )
    _print_section("Email Thread Snapshot")
    print(thread.get_full_thread())

    print("\nDemo complete. The system is ready for HITL orchestration.")


if __name__ == "__main__":
    run_demo()
