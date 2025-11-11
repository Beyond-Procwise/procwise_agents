# Procurement Knowledge Graph Platform

This package delivers a production-ready procurement intelligence platform that combines Neo4j, Qdrant, and Ollama to enrich the ProcWise multi-agent ecosystem. It consolidates structured procurement data, negotiation telemetry, and semantic embeddings to unlock advanced supplier analytics, negotiation strategy recommendations, and natural language interactions.

## Key Capabilities

- **Unified Knowledge Graph**: Loads suppliers, contracts, POs, invoices, quotes, negotiation sessions, workflow telemetry, and product taxonomy into Neo4j with strict schema constraints and indexes.
- **Semantic Retrieval**: Uses Qdrant with MiniLM embeddings to power similarity search for suppliers, contracts, and negotiations.
- **Local LLM Integration**: Streams prompts to Ollama for strategy generation, negotiation analysis, and natural language interfaces with resilient retry logic.
- **Agent-Ready APIs**: Provides drop-in integration for the existing ProcWise NegotiationAgent, EmailDraftingAgent, and orchestration services.
- **Workflow Safety Nets**: Prevents premature negotiation triggers, enforces email threading, and synchronises workflow telemetry across agents.

## Repository Layout

```
procurement_knowledge_graph/
├── procurement_knowledge_graph.py    # Graph builder and sync engine
├── hybrid_query_engine.py           # Unified query surface across Neo4j/Qdrant/Ollama
├── procwise_kg_integration.py       # ProcWise agent façade
├── end_to_end_demo.py               # Demonstration workflow
├── docker-compose.yml               # Infrastructure stack
├── setup.sh                         # Bootstrap script
├── config.py                        # Configuration dataclasses
├── README.md                        # This document
├── QUICK_REFERENCE.md              # Operator cheat sheet
├── GETTING_STARTED.md              # Step-by-step setup
└── __init__.py                      # Package marker
```

## Architecture Overview

The solution introduces three cooperating layers:

1. **Data Ingestion** – `ProcurementKnowledgeGraph` extracts tables from PostgreSQL using incremental watermarks, builds analytical views, and writes nodes/relationships into Neo4j within batch transactions. It concurrently prepares embedding payloads and upserts them into Qdrant.
2. **Hybrid Query Engine** – `HybridProcurementQueryEngine` orchestrates graph lookups, vector searches, and LLM requests. It implements the twelve critical contract-negotiation methods described in the project brief.
3. **Agent Integration Layer** – `ProcWiseKnowledgeGraphAgent` exposes agent-friendly APIs that extend current ProcWise workflows without breaking existing contracts.

## Core Design Decisions

- **Workflow-Oriented Keys**: All negotiation and email interactions are keyed by `workflow_id`, ensuring the platform enforces consistent threading across agents.
- **Fuzzy Supplier Matching**: Purchase order records are linked to suppliers using deterministic ID joins when available, with RapidFuzz-powered fallback matching for legacy name-based joins.
- **Resilient LLM Handling**: Every Ollama invocation uses exponential backoff, JSON schema validation, and automatic repair prompts when responses are malformed.
- **Performance Conscious**: Cypher queries employ appropriate indexes, multi-hop traversals are pre-optimised, and vector payloads are compressed via `orjson` before transmission.
- **Security-Aware**: Credentials are injected via environment variables, all SQL uses bound parameters, and the platform avoids logging sensitive payloads.

## Getting Started

1. Ensure the root-level `.env` file contains the Neo4j, Qdrant, Ollama, and PostgreSQL configuration that ProcWise already uses.
2. Execute `./setup.sh` to launch Docker services, create a Python virtual environment, install dependencies, and validate all connections.
3. Activate the virtual environment: `source .venv/bin/activate`.
4. Populate PostgreSQL with procurement data or run the demo script for synthetic data: `python procurement_knowledge_graph/end_to_end_demo.py`.
5. Integrate with ProcWise agents by importing `ProcWiseKnowledgeGraphAgent`.

Detailed setup and maintenance commands are available in [GETTING_STARTED.md](GETTING_STARTED.md) and [QUICK_REFERENCE.md](QUICK_REFERENCE.md).

## Testing Strategy

- **Unit Tests**: Cover SQL extraction, Cypher generation, vector payload serialization, and LLM response validation.
- **Integration Tests**: Validate negotiation readiness gating, email threading reconstruction, and natural language query pathways using synthetic fixtures.
- **Performance Benchmarks**: Provide scripts to load 10k suppliers, 50k purchase orders, and 100k line items to stress-test the system.

## License

This module inherits the licensing terms of the parent ProcWise project.
