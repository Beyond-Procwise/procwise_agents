"""End-to-end demonstration of the ProcWise knowledge graph stack."""
from __future__ import annotations

import logging
import os
from pprint import pprint

from procurement_knowledge_graph import build_default_builder
from hybrid_query_engine import build_default_engine
from procwise_kg_integration import ProcwiseIntegrationFacade, EmailThreadingUtilities

logging.basicConfig(level=os.environ.get("PROCWISE_LOG_LEVEL", "INFO"))


def run_demo() -> None:
    logging.info("Starting procurement data build...")
    builder = build_default_builder()
    builder.refresh()
    builder.close()

    logging.info("Running hybrid query examples...")
    engine = build_default_engine()

    facade = ProcwiseIntegrationFacade(engine=engine)

    spend_answer = facade.ask("How much did we spend with preferred suppliers last quarter?", filters={"label": "PurchaseOrder"})
    print("\n=== Spend Analysis ===")
    print(spend_answer["answer_html"])
    print("Supporting facts:")
    pprint(spend_answer["supporting_facts"])

    readiness = facade.should_trigger_negotiation("workflow-123", "rfq-456", min_responses=5)
    print("\n=== Negotiation Readiness ===")
    pprint(readiness)

    strategy = facade.generate_strategy("workflow-123", "rfq-456", context="Targeting 5% savings on logistics lanes")
    print("\n=== Strategy Proposal ===")
    pprint(strategy)

    headers = EmailThreadingUtilities.build_outbound_headers("workflow-123")
    print("\n=== Email Headers ===")
    pprint(headers)
    mapped = EmailThreadingUtilities.parse_incoming_headers(headers["Message-ID"], headers.get("References"))
    print("Mapped reply context:")
    pprint(mapped)

    print("\nDemo complete. The system is ready for HITL orchestration.")


if __name__ == "__main__":
    run_demo()
