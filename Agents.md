Agentic Workflow Architecture — ProcWise Agents

This document describes all active agents in the Beyond ProcWise ecosystem, their responsibilities, key interactions, and configuration parameters.
It reflects the post–October 2025 refactor removing RFQ ID usage and standardizing on workflow_id.

🔁 Overview

Each agent in ProcWise follows a common pattern:

Input Context: Provided via routing (workflow_id, agent_name, supplier_id, round, status, payload).

Output Context: Persisted to Postgres (e.g., process details, drafts, session state) and optionally dispatched via the email chain.

Idempotency: All writes use ON CONFLICT DO UPDATE aligned with real unique indexes.

Locking: Agents that modify shared state (e.g., NegotiationAgent) use Postgres advisory locks to prevent duplicate executions.

🧩 Agent Catalog
Agent	Purpose	Inputs	Outputs	Dependencies
OpportunityMinerAgent	Mines historical spend, contracts, emails, and embeddings to surface new sourcing/negotiation opportunities and supplier candidates. De-duplicates by opportunity signature and routes to the orchestrator.	Historical transactions, contract metadata, supplier catalogs, email corpus, embeddings, workflow_id	Opportunity candidate list, suggested categories/suppliers, routing events	qdrant_client, embedding service, spend/contract loaders
DataExtractionAgent	Extracts tabular/semantic content from uploaded documents (PDF/DOCX/XLSX) and stores structured JSON & embeddings.	Document path/type, workflow_id	Parsed JSON + vector embeddings	pdfplumber, fitz, OCR, qdrant_client
SupplierRankingAgent	Scores and ranks suppliers using multi-criteria evaluation (price, lead time, risk, terms).	Supplier offers, policy weights, workflow_id	Ranked supplier table + best supplier suggestion	Policy engine & scoring utilities
NegotiationAgent	Handles multi-round negotiation logic. Determines counter-offers, applies playbooks, and triggers drafting. One active run per (workflow_id, supplier_id, round) via advisory lock.	Supplier proposals, policy thresholds, workflow_id	Counter-offer decisions, session state, multi-supplier draft bundle	EmailDraftingAgent, ProcessRoutingService
EmailDraftingAgent	Generates structured/contextual emails (supports multi-supplier bundle per round).	workflow_id, supplier_id, session_reference, round, prompt/context	HTML drafts stored in drafts table	Template/Jinja, Ollama LLMs
EmailDispatchAgent	Sends finalized drafts and records each dispatch in the dispatch chain.	Draft details, workflow_id, supplier_id, session_reference	Sent message + message_id recorded	SMTP/SES, Postgres
EmailWatcherAgent	Monitors inbound mail for supplier responses. Maps replies deterministically via message_id or embedded PROC-WF-* token.	Raw emails, workflow_id	Parsed responses, updated negotiation sessions	Email libs, Postgres chain mapping
OrchestratorAgent	Central controller coordinating agents according to workflow states in routing.	Routing JSON	Sequenced/parallel agent invocations	ProcessRoutingService

Note: The OpportunityMinerAgent is upstream of sourcing/negotiation and can autonomously seed new workflows or enrich existing ones with candidate suppliers and demand signals.

🧭 End-to-End Flow
graph TD
    subgraph Discovery
      OM[OpportunityMinerAgent]
      DE[DataExtractionAgent]
    end
    RANK[SupplierRankingAgent]
    NEG[NegotiationAgent]
    DRAFT[EmailDraftingAgent]
    SEND[EmailDispatchAgent]
    WATCH[EmailWatcherAgent]

    OM --> RANK
    DE --> RANK
    RANK --> NEG
    NEG --> DRAFT
    DRAFT --> SEND
    SEND --> WATCH
    WATCH --> NEG


Discovery: OpportunityMinerAgent and/or DataExtractionAgent surface candidate items and suppliers.

Negotiation: Capped at three rounds per supplier with human-in-the-loop checkpoints.

Threading: Replies loop back deterministically to the same (workflow_id, supplier_id, session_reference).

⚙️ Database Keys & Constraints (canonical)
Table (concept)	Unique Key	Purpose
Negotiation Sessions	(workflow_id, supplier_id, round)	Session row per supplier per round
Negotiation Session State	(workflow_id, supplier_id)	Latest state for each supplier
Email Drafts	(workflow_id, unique_id, supplier_id)	Draft per supplier per workflow (per unique)
Email Dispatch Chain	(workflow_id, supplier_id, unique_id)	Maps outbound message_id + session_reference for threading
Routing (process)	(workflow_id, agent_name, supplier_id, round)	Prevent duplicate agent execution per supplier round
Opportunity Candidates (if persisted)	(workflow_id, opportunity_signature)	De-dupe mined opportunities in a workflow (signature strategy defined by implementation)

If you persist opportunities, ensure the signature is stable (e.g., normalized category + spec hash + time window) to avoid duplicates.

🔒 Locking and Idempotency

NegotiationAgent advisory lock per (workflow_id, supplier_id, round):
SELECT pg_try_advisory_lock(crc32(f"{workflow_id}:{supplier_id}:{round}") & 0x7FFFFFFF);

All inserts use UPSERT matching real unique indexes to avoid UniqueViolation or InvalidColumnReference.

📨 Email Threading Chain
Step	Source	Target	Key Columns
Draft Creation	EmailDraftingAgent	Drafts store	workflow_id, unique_id, supplier_id
Dispatch Record	EmailDispatchAgent	Dispatch chain	workflow_id, supplier_id, unique_id, message_id, session_reference
Reply Mapping	EmailWatcherAgent	Joins chain + drafts	message_id or PROC-WF-* token → (workflow_id, supplier_id, session_reference)

Every outbound email appends a marker in the HTML:

<!-- PROCWISE_MARKER:TRACKING:{unique_id}|SUPPLIER:{supplier_id}|TOKEN:{token} -->

🤖 Model Configuration (Ollama)

Agents discover installed local models via:

ollama list


Supported roster (graceful fallback handled in code/config):

qwen3:30b
mixtral:8x7b
phi4:latest
gemma3:1b-it-qat
gemma3:latest
mistral:latest
gpt-oss:latest
llama3.2:latest


Tune preferred models per agent where relevant (e.g., drafting vs. extraction) and ensure environment hooks surface all required tags.

✅ Quality & Stability Checks
Check	Description	Expected Outcome
Opportunity De-duplication	Miner uses stable signature to avoid duplicate opportunities	No repeated candidates in the same workflow
Advisory Locking	Prevent duplicate NegotiationAgent runs	Only one per (workflow_id, supplier_id, round)
Upserts	ON CONFLICT targets real unique indexes	No InvalidColumnReference
Supplier Propagation	Drafts/dispatches must include supplier_id	No supplier=None
Thread Integrity	Replies map deterministically	Watcher resolves supplier/session every time
Noise Reduction	FAISS GPU fallback warns once	Clean logs, minimal spam
🧩 OpportunityMinerAgent Details

Role:
Proactively identifies sourcing/negotiation opportunities (e.g., expiring contracts, category consolidation, price variance, demand spikes) and proposes suppliers.

Core Steps:

Ingest & Embed: Pull spend/contract/email snippets; refresh or query embeddings via Qdrant.

Detect Opportunities: Rules + LLM heuristics (e.g., anomalous price bands, long-tail consolidation).

Candidate Suppliers: Map items/categories to suppliers; enrich with performance/risk context.

De-duplicate & Persist: Use an opportunity_signature; upsert candidates (optional table) and/or route directly.

Route: Post routing events to Orchestrator to trigger SupplierRankingAgent (and downstream flow).

Inputs:
Spend lines, contract metadata (end dates/renewals), catalog SKUs, historical quotes/emails, embeddings, workflow_id.

Outputs:

List of opportunities (category, items, suggested suppliers, rationale)

Routing payload for SupplierRankingAgent / NegotiationAgent

Optional persistence for audit/tracking (opportunity candidates)

Idempotency:

Upsert by (workflow_id, opportunity_signature) if persisted.

Avoid re-routing the same signature twice.

📌 Implementation Notes

No RFQ identifiers are used; workflow_id is the canonical key across all agents and tables.

Ensure OpportunityMinerAgent can start:

As a workflow entry-point (creating a new workflow_id), or

As an enrichment step inside an existing workflow (receiving workflow_id from routing).

When OpportunityMinerAgent proposes suppliers for a category, it should populate routing payloads that the SupplierRankingAgent can consume without additional normalization.