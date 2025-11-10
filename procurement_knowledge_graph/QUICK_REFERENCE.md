# Quick Reference Guide

## Environment

```bash
# Environment configuration is shared with the root ProcWise `.env`
source .venv/bin/activate
```

## Containers

```bash
# Start stack
docker compose -f procurement_knowledge_graph/docker-compose.yml up -d

# Stop stack
docker compose -f procurement_knowledge_graph/docker-compose.yml down
```

## Data Sync

```bash
python -m procurement_knowledge_graph.procurement_knowledge_graph --full-refresh
python -m procurement_knowledge_graph.procurement_knowledge_graph --incremental --since "2025-01-01"
```

## Negotiation Safety

```python
from procurement_knowledge_graph.procwise_kg_integration import ProcWiseKnowledgeGraphAgent

agent = ProcWiseKnowledgeGraphAgent()
trigger = agent.should_trigger_negotiation("WF_20250115_001", "RFQ-7788")
if trigger["should_trigger_negotiation"]:
    print("Ready for negotiation")
else:
    print(trigger["recommendation"])
```

## Email Threading

```python
context = agent.get_email_thread_context("RFQ-7788", "SUP-1001")
headers = context["threading_headers"]
```

## Similar Negotiations

```python
engine = agent.engine
similar = engine.find_similar_negotiations(
    "Round 2 counter-offer for ISO supplier in Europe with target price 10000"
)
```

## Natural Language Query

```python
answer = engine.natural_language_query("Which suppliers have the best on-time delivery rate?")
print(answer["answer"])
```

## Maintenance Commands

```bash
# Recreate Neo4j indexes
python -m procurement_knowledge_graph.procurement_knowledge_graph --rebuild-indexes

# Vacuum supplier responses table
psql $POSTGRES_DSN -c 'VACUUM ANALYZE proc.supplier_response'
```
