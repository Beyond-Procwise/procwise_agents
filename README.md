# ProcWise Knowledge Graph Platform

This repository extends the ProcWise agents platform with a production-grade
knowledge graph, semantic retrieval, and negotiation intelligence stack. The
solution links Neo4j, Qdrant, and Ollama to unlock rich reasoning for sourcing
and procurement workflows.

## Architecture Overview

```
PostgreSQL  ──► procurement_knowledge_graph.py ──► Neo4j
      │                                           │
      └────► Qdrant embeddings ◄── Ollama ◄───────┘
                                │
                                ▼
Hybrid Query Engine ──► Strategy + Workflow APIs
```

* **procurement_knowledge_graph.py** performs incremental extraction from
  Postgres, updates the Neo4j graph schema, and stores vector embeddings with
  rich payloads in Qdrant.
* **hybrid_query_engine.py** blends graph queries, vector retrieval, and LLM
  reasoning to answer spend, supplier, and negotiation questions.
* **procwise_kg_integration.py** exposes ProcWise-ready orchestration hooks for
  negotiation gating, email threading, and AI-assisted strategy creation.
* **end_to_end_demo.py** demonstrates the full pipeline from data ingest to
  question answering and strategy generation.

## Services

A Docker Compose stack ships with managed services for:

| Service  | Purpose                         | Default Port |
|----------|---------------------------------|--------------|
| Postgres | Operational procurement source  | 5432         |
| Neo4j    | Knowledge graph backend         | 7474 / 7687  |
| Qdrant   | Vector store for embeddings     | 6333 / 6334  |
| Ollama   | Local LLM + embedding provider  | 11434        |
| App      | Executes the end-to-end demo    | N/A          |

## Key Capabilities

* **Idempotent graph sync** with resumable checkpoints per entity.
* **Hybrid retrieval** that merges graph insights and semantic context with
  re-ranking and HTML-formatted answers.
* **Negotiation workflows** that respect minimum supplier response thresholds
  and preserve threaded email chains.
* **LLM-driven strategy assistance** using historical graph knowledge and
  semantic neighbors for counter-offer guidance.

## Getting Started

1. Run `./setup.sh` to launch the infrastructure, seed demo data, and pull the
   required Ollama models.
2. Execute `docker compose up app` to run the end-to-end demonstration.
3. Explore APIs and sample queries inside the `docs/` directory and
   `QUICK_REFERENCE.md`.

## Production Notes

* All configuration uses environment variables to remain compatible with the
  existing ProcWise deployment pipeline.
* Answers from the hybrid engine avoid leaking internal identifiers or raw
  table names.
* The retrieval stack defaults to `phi4:latest` with a semi-formal, concise
  tone tuned for procurement operations.
* The Qdrant collection prefix may be customised per environment to support
  multi-tenant deployments.
