# Email Response Capture Research

## Objectives
- Maintain a one-to-one mapping between dispatched supplier emails and inbound responses.
- Reduce ambiguity caused by shared RFQ identifiers across workflows or suppliers.
- Ensure deterministic processing order when aggregating multi-supplier workflows.

## Key Findings
1. **Hidden JSON Markers**
   - Embed a HTML comment `<!-- PROCWISE:{...} -->` at the top of every outbound email body.
   - Include `workflow_id`, `unique_id`, `supplier_id`, and tracking tokens to simplify IMAP matching.
2. **Header Parity**
   - Mirror the marker values inside custom headers (`X-Procwise-Workflow-Id`, `X-Procwise-Unique-Id`, `X-Procwise-Supplier-Id`).
   - Header parity guards against email client truncation and allows server-side filtering.
3. **Dispatch Registry**
   - Persist `workflow_id`, `unique_id`, `run_id`, `supplier_id`, and `dispatched_at` inside `proc.draft_rfq_emails` for every sent message.
   - The IMAP watcher queries this registry to determine the expected response count and enforce counters.
4. **Response Ingestion Workflow**
   - Wait at least 90 seconds after the last dispatch before scanning IMAP to accommodate provider delays.
   - Compare the number of unique responses discovered with the dispatch counter; process only when counts align.
   - Persist matched responses to `proc.supplier_responses` and mark them as processed to avoid replays.
5. **Processing Strategies**
   - Prioritise direct `unique_id` matches; fall back to `supplier_id` only when a unique mapping is guaranteed.
   - Feed response payloads through the Supplier Interaction agent (or orchestrator) for structured parsing.

## Recommended Enhancements
- Adopt message hashing in addition to unique IDs to detect duplicate replies.
- Store IMAP UIDs and mailbox identifiers for audit-friendly traceability.
- Expand automated tests to include multi-supplier workflows with staggered replies.
- Monitor the delay between `dispatched_at` and `received_at` to tune the wait window dynamically.

