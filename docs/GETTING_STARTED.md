# Getting Started

This guide walks through deploying the ProcWise knowledge graph stack locally
or within a containerised environment.

## Prerequisites

* Docker and the Docker Compose plugin
* At least 16 GB RAM for comfortable Ollama model execution
* Bash-compatible shell

## 1. Bootstrap the stack

```bash
./setup.sh
```

The script will:

1. Validate Docker availability
2. Launch Postgres, Neo4j, Qdrant, and Ollama containers
3. Create the required procurement tables and seed demo data in Postgres
4. Verify Neo4j connectivity
5. Pull the `phi4:latest` and `nomic-embed-text` models for Ollama

## 2. Run the end-to-end demo

Start the demo container once infrastructure is ready:

```bash
docker compose up app
```

The `app` service executes `scripts/end_to_end_demo.py`, which will:

* Refresh the knowledge graph and Qdrant embeddings
* Execute a hybrid spend analysis query
* Evaluate negotiation readiness safeguards
* Generate a negotiation strategy proposal
* Produce stable email threading headers

## 3. Develop locally without Docker

1. Create a Python 3.11 virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Start local Postgres, Neo4j, Qdrant, and Ollama instances (or reuse the
   docker-compose services).

3. Export the configuration variables:

   ```bash
   export POSTGRES_DSN=postgresql://procwise:procwise@localhost:5432/procwise
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password
   export QDRANT_URL=http://localhost:6333
   export QDRANT_API_KEY=
   export QDRANT_COLLECTION_PREFIX=dev_
   export OLLAMA_HOST=http://localhost:11434
   export LLM_DEFAULT_MODEL=phi4:latest
   export OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

4. Load data and test the engine:

   ```bash
   python -m procurement_knowledge_graph.procurement_knowledge_graph --full-refresh
   python scripts/end_to_end_demo.py
   ```

## 4. Production deployment tips

* Schedule `procurement_knowledge_graph.procurement_knowledge_graph` on a timer or event trigger to keep
  the graph current. The script is idempotent and resumes from the last synced
  timestamp per entity.
* Use distinct `QDRANT_COLLECTION_PREFIX` values for staging vs production to
  avoid cross-environment collisions.
* Configure Ollama model caching volumes on persistent storage to prevent
  repeated downloads.
* Integrate `ProcWiseKnowledgeGraphAgent` functions into existing services by
  importing `procurement_knowledge_graph.procwise_kg_integration` and reusing
  the shared environment variables.
