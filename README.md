# ProcWise Knowledge Graph Platform

This repository extends the ProcWise agents platform with a production-grade
knowledge graph, semantic retrieval, and negotiation intelligence stack. The
solution links Neo4j, Qdrant, and Ollama to unlock rich reasoning for sourcing
and procurement workflows.

## Architecture Overview

```
PostgreSQL  ──► procurement_knowledge_graph/procurement_knowledge_graph.py ──► Neo4j
      │                                           │
      └────► Qdrant embeddings ◄── Ollama ◄───────┘
                                │
                                ▼
Hybrid Query Engine ──► Strategy + Workflow APIs
```

* **procurement_knowledge_graph/procurement_knowledge_graph.py** performs incremental extraction from
  Postgres, updates the Neo4j graph schema, and stores vector embeddings with
  rich payloads in Qdrant.
* **procurement_knowledge_graph/hybrid_query_engine.py** blends graph queries, vector retrieval, and LLM
  reasoning to answer spend, supplier, and negotiation questions.
* **procurement_knowledge_graph/procwise_kg_integration.py** exposes ProcWise-ready orchestration hooks for
  negotiation gating, email threading, and AI-assisted strategy creation.
* **scripts/end_to_end_demo.py** demonstrates the full pipeline from data ingest
  to question answering and strategy generation.

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
2. Execute `docker compose up app` (or run `python scripts/end_to_end_demo.py`)
   to exercise the full demo.
3. Explore APIs and sample queries inside the `docs/` directory, including
   `docs/QUICK_REFERENCE.md`.

## Production Notes

* All configuration uses environment variables to remain compatible with the
  existing ProcWise deployment pipeline.
* Answers from the hybrid engine avoid leaking internal identifiers or raw
  table names.
* The retrieval stack defaults to `phi4:latest` with a semi-formal, concise
  tone tuned for procurement operations.
* The Qdrant collection prefix may be customised per environment to support
  multi-tenant deployments.

## Model Training & Evaluation

The `training/pipeline.py` module now exposes a modular workflow for exporting
instruction data, fine-tuning with Unsloth-powered QLoRA, rendering Ollama
Modelfiles, and validating the resulting RAG behaviour.

Key commands (run via `python -m training.pipeline <command> ...`):

- `export-data` – stream instruction data from Postgres directly to JSONL.
- `train` – launch Unsloth + TRL supervised fine-tuning with reproducible
  configs and deterministic logging.
- `convert-gguf` – convert merged HF weights into GGUF (and optionally quantize)
  for Ollama.
- `render-modelfile` – fill `models/Modelfile.template` with the new weight
  path, temperature, and context window. This produces the Modelfile that
  should be registered as `qwen3-30b-procwise` for the RAG agent.
- `rag-eval` – execute the upgraded retrieval pipeline across multiple Qdrant
  collections using a JSON/JSONL list of benchmark queries. The command writes
  aggregate metrics (doc diversity, latency, refusal rate) plus optional deltas
  versus a baseline report.

See `services/RAG_README.md` for environment knobs. After each training run,
render a Modelfile, `ollama create qwen3-30b-procwise -f <Modelfile>`, and run
`rag-eval` with your preferred factory (e.g. `services.rag_service:build_rag`)
to confirm the model understands, retrieves, and cites context correctly across
all configured collections.
