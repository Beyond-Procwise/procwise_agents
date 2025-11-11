# Getting Started

This guide walks through launching the procurement knowledge graph stack, seeding data, and integrating with existing ProcWise agents.

## 1. Prerequisites

- Docker and Docker Compose v2
- Python 3.10+
- At least 12 GB RAM and 30 GB disk for Neo4j, Qdrant, and Ollama models
- Access to the ProcWise PostgreSQL database or sample datasets

## 2. Bootstrap the Environment

```bash
cd procurement_knowledge_graph
./setup.sh
```

The setup script will:

1. Launch PostgreSQL, Neo4j, Qdrant, Ollama, and pgAdmin containers.
2. Wait until each service reports healthy status.
3. Pull the required Ollama models (`llama3.2`, `all-minilm`).
4. Create a Python virtual environment under `.venv` and install dependencies from the root `requirements.txt`.
5. Validate connections to all back-end services using the shared ProcWise configuration.

## 3. Load Sample Data (Optional)

If you do not have production data, execute the demo script to populate synthetic tables and graph nodes:

```bash
source .venv/bin/activate
python procurement_knowledge_graph/end_to_end_demo.py --load-sample-data
```

The script provisions 10 suppliers, 50 purchase orders, 20 quotes, and associated negotiation telemetry. It then triggers a full graph rebuild, Qdrant embedding sync, and demonstrates negotiation strategy generation.

## 4. Perform a Full Graph Build

```bash
python -m procurement_knowledge_graph.procurement_knowledge_graph --full-refresh
```

This command truncates existing graph entities (via APOC) and rebuilds the entire knowledge graph with indexes, constraints, and vector collections. For subsequent updates use incremental mode:

```bash
python -m procurement_knowledge_graph.procurement_knowledge_graph --incremental --since "2025-01-01T00:00:00"
```

## 5. Integrate with ProcWise Agents

```python
from procurement_knowledge_graph.procwise_kg_integration import ProcWiseKnowledgeGraphAgent

kg_agent = ProcWiseKnowledgeGraphAgent()
status = kg_agent.get_workflow_status("WF_20250115_001")
print(status)
```

The agent façade shares the same signatures as the project brief, ensuring NegotiationAgent and EmailDraftingAgent can adopt the new capabilities with minimal code changes.

## 6. Running the Demo Workflow

```bash
python procurement_knowledge_graph/end_to_end_demo.py --demo
```

The demo script orchestrates the following steps:

1. Seeds sample procurement data (if requested).
2. Executes a negotiation readiness check.
3. Generates negotiation strategies for responding suppliers.
4. Crafts counter-offer emails with proper threading headers.
5. Performs a natural language query summarising supplier performance.

All interactions use the live Neo4j, Qdrant, and Ollama services.

## 7. Maintenance Tips

- Rebuild Neo4j indexes after bulk data imports: `python -m procurement_knowledge_graph.procurement_knowledge_graph --rebuild-indexes`.
- Compact Qdrant collections weekly: `qdrant-cli snapshot --collection suppliers`.
- Monitor Ollama logs for GPU memory pressure and adjust model roster if needed.
- Rotate environment variables and secrets regularly; never commit credentials.

## 8. Troubleshooting

- **Neo4j authentication failures** – Ensure `NEO4J_USERNAME` and `NEO4J_PASSWORD` match the running container.
- **Qdrant connection refused** – Check port mapping `6333:6333` and firewall settings.
- **Slow LLM responses** – Confirm `OLLAMA_KEEP_ALIVE` is set (see docker-compose). Consider enabling GPU acceleration.
- **Threading header mismatch** – Run `get_email_thread_context` to inspect recorded message IDs and ensure dispatch tracking is enabled.

## 9. Next Steps

- Hook the knowledge graph into CI pipelines for nightly refreshes.
- Extend embeddings to include policy documents for governance-aware negotiations.
- Explore additional Ollama models for specialised drafting or classification tasks.

You are now ready to enrich ProcWise with data-driven negotiation intelligence.
