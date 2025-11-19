# Domain Finetuning Workflow

This repository now ships with a full finetuning toolchain that extracts
domain-labelled supervision from PostgreSQL, enriches each sample with the
supporting passages stored in Qdrant, and trains a LoRA adapter that can be
merged and imported into LM Studio.

## 1. Build a domain dataset

The `build-domain-dataset` subcommand reads any SQL query that returns
`instruction`/`output` (and optional `input`) columns, looks up matching
payloads in Qdrant, and appends the retrieved evidence to each sample.

```bash
python -m training.pipeline build-domain-dataset \
  --dsn postgresql://procwise:procwise@localhost:5432/procwise \
  --query "$(cat sql/export_instructions.sql)" \
  --output data/domain_dataset.jsonl \
  --join-column doc_id \
  --context-header "ProcWise knowledge" \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection procwise_document_embeddings \
  --qdrant-match-field doc_id \
  --qdrant-context-field content \
  --qdrant-title-field title \
  --qdrant-limit 4
```

The query can join any business tables as long as it emits the required
columns. A typical template is:

```sql
SELECT
  prompt AS instruction,
  COALESCE(input_summary, '') AS input,
  answer AS output,
  doc_id
FROM proc.policy_response_log
WHERE answer IS NOT NULL
```

The `doc_id` column is used to match the Qdrant payload field defined by
`--qdrant-match-field`, ensuring that the appended passages stay aligned with
the original evidence. If `--include-context-metadata` is set the raw payloads
and titles are stored in `context_sources` for downstream auditing.

## 2. Run QLoRA finetuning

Once the dataset has been enriched you can launch supervised finetuning using
any Hugging Face base model that you ultimately want to host inside LM Studio.

```bash
python -m training.pipeline train \
  --base-model microsoft/phi-4-mini-128k-instruct \
  --train-file data/domain_dataset.jsonl \
  --output-dir outputs/phi4-procwise-lora \
  --system-prompt "You are ProcWise's sourcing copilot..." \
  --num-train-epochs 3 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

The command defaults to QLoRA + Unsloth. Disable `--use-unsloth` if you prefer
standard `transformers` execution.

## 3. Merge weights and import into LM Studio

After finetuning finishes you can merge the adapter, convert to GGUF if needed,
and render an LM Studio modelfile descriptor:

```bash
python -m training.pipeline merge \
  --base-model microsoft/phi-4-mini-128k-instruct \
  --adapter-path outputs/phi4-procwise-lora \
  --output-dir outputs/phi4-procwise-merged

python -m training.pipeline render-modelfile \
  --template models/Modelfile.template \
  --output models/phi4-procwise.Modelfile \
  --weights outputs/phi4-procwise-merged
```

Import the rendered Modelfile (or a quantized GGUF) into LM Studio via the
“Models → Add Custom Model” dialog. Point `LMSTUDIO_CHAT_MODEL` at the new
identifier and restart the ProcWise services; all local calls now route through
the domain-adapted weights that were grounded in the PostgreSQL/Qdrant corpus.
