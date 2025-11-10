# Quick Reference

## Common Commands

```bash
./setup.sh                          # Launch infra, seed data, pull models
python procurement_knowledge_graph.py # Refresh KG + embeddings locally
python end_to_end_demo.py            # Run the hybrid demo script
docker compose up app               # Execute demo inside container stack
docker compose logs -f app          # Stream app logs
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `POSTGRES_DSN` | PostgreSQL connection string used for extraction and workflow checks |
| `NEO4J_URI` | Bolt endpoint for Neo4j |
| `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j credentials |
| `QDRANT_URL` | Base URL for Qdrant vector store |
| `QDRANT_API_KEY` | Optional API key for managed Qdrant deployments |
| `QDRANT_COLLECTION_PREFIX` | Prefix applied to the procurement embeddings collection |
| `OLLAMA_HOST` | HTTP endpoint for Ollama |
| `LLM_DEFAULT_MODEL` | Default chat model (phi4:latest recommended) |
| `OLLAMA_EMBED_MODEL` | Embedding model for vector creation |

## Sample Graph Queries

```cypher
// Suppliers with active contracts
MATCH (s:Supplier)-[r:HAS_CONTRACT]->(c:Contract)
WHERE c.end_date > date()
RETURN s.name, c.contract_number, r.total_value
ORDER BY r.total_value DESC;

// Spend per category
MATCH (po:PurchaseOrder)-[:ORDERED_UNDER]->(cat:Category)
RETURN cat.name AS category, sum(po.total_amount) AS spend
ORDER BY spend DESC;
```

## Hybrid Query API Samples

```python
from hybrid_query_engine import build_default_engine

engine = build_default_engine()
response = engine.ask("Show supplier performance for LogiTrans in 2025")
print(response["answer_html"])

similar = engine.find_similar_negotiations(rfq_id="rfq-456")
print(similar)
```

## Troubleshooting

* **Vectors missing** – Verify `OLLAMA_EMBED_MODEL` is available (`ollama list`).
* **Neo4j auth errors** – Confirm `NEO4J_AUTH` in docker-compose matches exported
  credentials.
* **Slow answers** – Ensure Ollama has pre-pulled models and adequate CPU/GPU.
* **Email threading mismatches** – Use `EmailThreadingUtilities.parse_incoming_headers`
  to debug tokens and workflow identifiers.
