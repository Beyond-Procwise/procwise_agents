# Phi-4 Humanization Lightweight Fine-Tuning Plan

This plan describes how to adapt the `phi4` base model into an empathetic and concise assistant (`phi4-joshi`) optimised for Beyond ProcWise negotiation agents. The focus is on short acknowledgements, grounded recommendations, and zero disclosure of internal controls. The approach balances dataset curation, supervision, preference optimisation, and deployment with Ollama.

## 1. Data Strategy

### 1.1 Supervised Fine-Tuning (SFT) Dataset
- **Volume:** Target 200–800 instruction samples distilled from authentic ProcWise chats.
- **Schema:** Each record should contain `{instruction, context, ideal_answer}`. Context must include negotiation state (round, supplier sentiment, outstanding asks) whenever available.
- **Style Guidelines:**
  - No boilerplate greetings or closings; responses should begin with a brief, empathetic acknowledgement (≤20 tokens).
  - Provide exactly one actionable follow-up suggestion.
  - Avoid referencing internal tooling, guardrails, or escalation paths.
- **Curation Workflow:**
  - Mine existing transcripts via the Negotiation and Supplier Interaction agents; remove sensitive identifiers, redact pricing with ranges where necessary.
  - Annotators validate each sample for empathy tone, factual grounding, and compliance with the above rules.
  - Store curated samples in a versioned dataset repository (`datasets/phi4_humanization/sft_v1.jsonl`).

### 1.2 Preference Dataset (DPO/ORPO)
- **Volume:** 200–500 preference pairs.
- **Structure:** `{prompt, chosen, rejected}` where the chosen response is concise, warm, and grounded, and the rejected response is robotic, verbose, or leaks internals.
- **Generation:**
  - Use disagreement harvesting from human review notes and existing red-team transcripts.
  - Optionally synthesise negative samples by prompting the base model with high temperature and tagging failure modes.
  - Record rationales for preference decisions to help future auditors.

### 1.3 Data Governance
- Maintain an internal README documenting sample provenance, redaction policies, and reviewer approvals.
- Store embeddings of dataset metadata in Qdrant (`learning` collection) for discovery alongside other workflow learnings using the `LearningRepository.record_model_plan` helper.

## 2. Training Pipeline

### 2.1 Environment
- Use the existing LoRA/QLoRA training stack (PyTorch + PEFT + bitsandbytes) within the `model-training` workspace.
- Hardware: single A100-40GB or equivalent (QLoRA reduces VRAM pressure).

### 2.2 Supervised Fine-Tuning Pass
- **Configuration:**
  - Base model: `phi4` (latest Ollama export).
  - Adapter rank: 64, alpha: 16, dropout: 0.05.
  - Learning rate: `2e-4` with cosine decay and 100 warmup steps.
  - Batch size: 128 sequences (gradient accumulation permitted).
  - Max sequence length: 4,096 tokens (truncate context beyond this window).
- **Loss:** Standard cross-entropy over the response tokens.
- **Validation:** hold-out 10% of SFT samples; metric is BLEU-style token overlap plus human spot checks for empathy tone.

### 2.3 Preference Optimisation (Optional)
- Apply DPO or ORPO on the SFT-adapted model using the preference dataset.
- Objective emphasises shorter responses by penalising completions >180 tokens.
- Run for 1–3 epochs with learning rate `5e-6`; evaluate against the validation split and manual review set.

### 2.4 Checkpoints and Versioning
- Save adapters at `{artifacts}/phi4-joshi/sft` and `{artifacts}/phi4-joshi/dpo`.
- Track experiment metadata in Weights & Biases (project `procwise/phi4-humanization`).

## 3. Evaluation Criteria
- **Automatic:**
  - Toxicity and leakage checks using internal red-team harness.
  - Response length percentile (P90 < 160 tokens).
  - Task success rate across negotiation reply benchmarks.
- **Human Review:**
  - 30-sample blind review by negotiation specialists focusing on tone, grounding, and actionable next steps.
  - Verify zero references to internal agent architecture.

## 4. Deployment Plan (Ollama / Local)

1. Export merged adapter weights and apply to the `phi4` base model.
2. Quantise to GGUF (Q4_K_M recommended) using `llama.cpp` conversion utilities.
3. Place the quantised model at `models/phi4-joshi.gguf`.
4. Register the model with Ollama using the following Modelfile:
   ```
   FROM ./phi4-joshi.gguf
   PARAMETER temperature 0.2
   PARAMETER top_p 0.9
   ```
5. Update ProcWise environment variable `LLM_DEFAULT_MODEL=phi4-joshi`.
6. Smoke-test with Supplier Interaction and Negotiation agents to ensure behavioural alignment.

## 5. Operational Considerations
- **Monitoring:**
  - Log structured feedback via the learning repository to capture future preference pairs.
  - Integrate daily drift detection comparing phi4-joshi outputs to baseline transcripts.
- **Rollback:** retain previous production adapter and quantised artefact; switching models is a single environment variable change.
- **Retraining Cadence:** monthly refresh or when tone drift >10% on empathy rubric.

## 6. Qdrant Memory Registration
Use the new `LearningRepository.record_model_plan` helper to persist this plan as vectorised memory. Supply `model_name="phi4-joshi"`, include `plan_version`, `dataset_counts`, and a short `deployment_status`. Agents can later call `LearningRepository.fetch_model_plans` to retrieve the latest plan when reasoning about negotiation tone or debugging behaviour.

