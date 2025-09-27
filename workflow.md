# ProcWise Agentic Workflow Reference

## Procurement Data Foundation
- **Source tables** follow the canonical schemas documented in `docs/procurement_table_reference.md`. Key relationships include contracts → suppliers → purchase orders → line items → invoices, along with product taxonomy joins for enriched categorisation.【F:docs/procurement_table_reference.md†L1-L44】
- `DataFlowManager.build_data_flow_map` ingests sampled records from these tables, reconciles supplier identifiers, and derives per-supplier flow dictionaries that capture contract, order, invoice, quote, and product coverage metrics for downstream agents.【F:services/data_flow_manager.py†L372-L520】【F:services/data_flow_manager.py†L827-L1108】

## Knowledge Graph & Vector Indexing
- `QueryEngine.train_procurement_context` orchestrates data sampling, schema documentation, knowledge graph construction, and supplier flow extraction before embedding all artefacts into Qdrant. The method also triggers procurement flow summaries for retrieval agents.【F:engines/query_engine.py†L640-L739】
- `SupplierRelationshipService.index_supplier_flows` converts each supplier flow into natural-language summaries, normalised metadata, and relationship statements. It also generates an aggregate overview document that captures coverage ratios, record counts, and top-performing suppliers for use in retrieval-augmented prompts.【F:services/supplier_relationship_service.py†L26-L173】

## Background Refresh Operations
- `BackendScheduler` runs as a daemon thread, registers supplier relationship refresh and model-training dispatch jobs, and invokes GPU configuration before each execution so maintenance stays isolated from foreground workflows.【F:services/backend_scheduler.py†L1-L153】
- `SupplierRelationshipScheduler` persists a refresh queue, schedules daily jobs in the next non-business window, and executes them by calling the procurement training pipeline. Each run updates job status metadata and records the latest supplier relationship overview for all agents.【F:services/supplier_relationship_service.py†L175-L324】

## Retrieval-Augmented Intelligence
- `RAGAgent` treats supplier flow, relationship, and overview documents as knowledge graph sources. Overview payloads now surface aggregate coverage and spend highlights, enriching the context outline and answer planning for supplier-sensitive questions.【F:agents/rag_agent.py†L17-L29】【F:agents/rag_agent.py†L380-L453】

## Workflow Execution Lifecycle
1. Incoming workflow requests are normalised, GPU configuration is ensured, and the backend scheduler is initialised so maintenance jobs continue independently of policy validation and execution.【F:orchestration/orchestrator.py†L16-L54】【F:orchestration/orchestrator.py†L63-L72】
2. Once validated, the orchestrator routes tasks through specialised agents (extraction, ranking, quotes, opportunities) and logs outcomes for policy-aware retraining cues.【F:orchestration/orchestrator.py†L170-L214】【F:orchestration/orchestrator.py†L214-L242】【F:services/model_training_service.py†L13-L120】

## Maintenance Guidelines
- Update this document whenever new agents, background jobs, or procurement data sources are introduced so downstream developers can quickly align enhancements with the existing workflow.
- When commissioning a new supplier mailbox, follow `docs/ses_inbound_pipeline.md` to verify SES → S3 → SQS delivery so the `SESEmailWatcher` receives replies without delay.【F:docs/ses_inbound_pipeline.md†L1-L58】【F:services/email_watcher.py†L137-L214】
