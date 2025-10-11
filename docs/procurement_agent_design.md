# Procurement Agent System Design Proposal

This document summarises the target architecture for the ProcWise procurement
agent stack. It complements the canonical data definitions captured in
[`docs/procurement_table_reference.md`](procurement_table_reference.md) and
focuses on four critical dimensions needed to operationalise supplier RFQ
negotiations end-to-end:

1. Reliable email ingestion across IMAP and S3 sources.
2. Persistent threading for all supplier correspondence.
3. Context-rich agent orchestration with governed prompts.
4. Negotiation intelligence, governance, and logging safeguards.

## 1. Reliable Email Ingestion (IMAP + S3)

To guarantee that supplier responses are captured exactly once, the watcher
service ingests inbound emails from an IMAP inbox and an SES-backed S3
quarantine.

- **Primary IMAP polling** – The watcher polls the configured IMAP mailbox for
  unseen messages. Each message is parsed (RFC5322) and marked as seen once it
  has been persisted to the workflow state to prevent duplicates.
- **S3 safety net** – When the IMAP mailbox is unavailable or yields no new data,
  the watcher transparently falls back to scanning the configured SES bucket.
  Only objects whose `LastModified` timestamp and key are newer than the stored
  watermark are processed. The watermark (timestamp + key) is persisted so restarts
  resume from the last known message.
- **Hidden RFQ markers** – The email drafting agent injects hidden HTML comments
  (`<!-- RFQ-ID: ... -->`) into outbound RFQs. During ingestion the parser checks
  for this marker first, then inspects dedicated headers, the subject, the body,
  and lastly SES metadata until an RFQ identifier match is found. Messages that
  fail to match are treated as noise and ignored.
- **Source-based deduplication** – Every processed S3 object is recorded in both
  memory and the `proc.processed_emails` registry (bucket, key, ETag). IMAP
  message-ids are tracked in-memory as well. This double-entry filter prevents
  reprocessing while making late-arriving objects discoverable.
- **Content validation** – Only emails that include the RFQ token and contain
  a substantive response (pricing, terms, or meaningful reply text) are passed
  into downstream agents. Invalid payloads are logged for review but do not
  block the ingestion loop.
- **Resilience** – Errors are logged and skipped so that malformed payloads do
  not halt processing. IAM role assumptions, thread pools, and retry guards
  allow the watcher to handle bursts of supplier replies in parallel.

## 2. Persistent Email Threading

Maintaining a single thread per RFQ keeps negotiations contextual for suppliers
and internal buyers.

- **Message-ID capture** – The initial RFQ dispatch stores the generated
  Message-ID in `proc.email_thread` alongside the RFQ metadata. This becomes the
  anchor for subsequent replies.
- **Reply headers** – Every follow-up email includes `In-Reply-To` and
  `References` headers derived from the last received supplier message. This
  ensures that email clients group communications correctly.
- **Subject normalisation** – Subjects are stored per RFQ so replies can safely
  prefix `Re:` while preserving the base description (e.g. `Re: Request for
  Quotation – Item ABC`).

## 3. Context-Rich Agent Execution

Agents never operate in isolation. The orchestrator passes a structured
`AgentContext` that accumulates knowledge as the workflow progresses.

- **Context propagation** – Outputs from one agent (e.g. parsed quote price,
  lead time, supplier sentiments) become `pass_fields` for the next agent.
- **Prompt governance** – All prompts live in Postgres (`proc.prompt`) and are
  fetched at runtime by the `PromptEngine`. Updates go live without redeploys
  and each invocation records the prompt version used.
- **Knowledge retrieval** – The `QueryEngine` enriches contexts with historical
  purchase data (Postgres) and semantic memories (Qdrant). Agents therefore act
  with awareness of policies, benchmark prices, and supplier performance.
- **Professional tone** – System prompts reinforce a "senior procurement"
  persona so generated outputs remain consistent and policy-aligned.

## 4. Negotiation Intelligence & Governance

Negotiations combine deterministic tactics with adaptive LLM guidance.

- **Negotiation playbook** – Strategy parameters per round (anchor, midpoint,
  best-and-final) are defined in configuration (JSON or `proc.negotiation_strategy`).
  The agent’s deterministic logic consumes these settings to calculate counters
  and recommended negotiation plays.
- **Dynamic prompt tuning** – LLM prompts adapt based on negotiation round,
  supplier sentiment, and detected keywords (e.g. "final offer"). This keeps
  tone and concessions aligned with supplier behaviour.
- **Exit criteria** – Hard caps on negotiation rounds and walk-away thresholds
  ensure the agent gracefully concludes or escalates when stalemates occur.
- **Audit trail** – Every agent action writes structured logs into Postgres
  (e.g. `proc.negotiation_log`) with RFQ IDs, strategy labels, counter values,
  prompt identifiers, and outcomes. Combined with the stored email thread data
  this delivers a complete audit trail.
- **Human oversight** – Complex replies, legal clauses, or anomalous quotes can
  be flagged for human approval. The workflow records the human decision to
  complete the audit chain.
- **Learning loops** – Final negotiation outcomes feed the learning repository
  so strategy parameters and prompts can be refined based on real performance
  metrics.

## 5. Testing & Quality Targets

To sign-off the implementation, the ingestion and extraction stack must satisfy
key operational KPIs:

- Header field extraction F1 ≥ 0.95 across golden invoice samples.
- Line reconciliation mismatch rate < 2%.
- Automated regression tests cover at least two scanned and two digital invoice
  samples within `tests/test_extraction.py`.

Meeting these targets, while respecting existing API contracts and database
schemas outlined in the reference document, ensures the procurement agent system
is production-grade, auditable, and adaptable.
