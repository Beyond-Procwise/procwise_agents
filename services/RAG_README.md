# RAG Pipeline Controls

The qwen3-30b-procwise RAG service now enforces a deterministic, high-precision pipeline. The knobs below can be tuned via environment variables (defaults shown).

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_TOP_K` | `10` | Dense candidate count pulled from Qdrant before rerank. |
| `RAG_RERANK_TOP_N` | `8` | Number of chunks kept after cross-encoder rerank. |
| `RAG_PER_DOC_CAP` | `2` | Maximum chunks admitted per `doc_id` before diversity relax kicks in. |
| `RAG_CONTEXT_MAX_CHARS` | `12000` | Hard cap for the packed context sent to the generator. |
| `COMPRESS_PER_CHUNK_CHARS` | `1200` | Extractive compression ceiling per chunk prior to packing. |
| `QDRANT_VECTOR_NAME` | `text` | Named vector to query in Qdrant (override when collections store multiple vectors). |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder used for reranking and sentence-level scoring. |
| `RERANK_BATCH` | `16` | Batch size for reranker/scoring calls. |
| `RERANK_FP16` | `1` | Set to `0` to force full precision on devices that do not support fp16. |

Operational safeguards:

1. Retrieval → rerank uses `QDRANT_VECTOR_NAME` (default `text`) before handing off to the cross-encoder.
2. Candidates pass through exact-hash and semantic dedupe, then a per-doc cap that still guarantees breadth.
3. Compression scores sentences with the reranker + MMR to retain only salient evidence, respecting `COMPRESS_PER_CHUNK_CHARS`.
4. Context packing enforces the `RAG_CONTEXT_MAX_CHARS` budget and guarantees at least three distinct `doc_id` sources whenever available.
5. Generation always goes through the LM Studio-hosted `qwen3-30b-procwise` model with strict refusal fallback when no context survives.

Make sure `LMSTUDIO_MODEL=qwen3-30b-procwise` (or set `LMSTUDIO_CHAT_MODEL`) and `LMSTUDIO_BASE_URL` are set appropriately wherever the service runs.

After each fine-tune, render a fresh Modelfile via `python -m training.pipeline render-modelfile --template models/Modelfile.template --weights <path to gguf>` and import it into LM Studio (`Models > Add Custom Model`). This ensures the ChatML instructions enforced by the Modelfile stay consistent with the service contract.
