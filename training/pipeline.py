"""Command line utilities for ProcWise model fine-tuning.

This module implements the end-to-end workflow requested by the
customer:

* Export instruction tuning data from Postgres to JSONL.
* Fine-tune a base model with QLoRA using ``transformers`` + ``trl``.
* Optionally leverage ``unsloth`` for improved training throughput.
* Merge LoRA adapters back into full Hugging Face weights.
* Convert merged weights into GGUF format and optionally quantize them
  for Ollama consumption.

The module provides a CLI with dedicated sub-commands so each step can
be orchestrated from automation or executed manually when needed.
"""
from __future__ import annotations
from unsloth import FastLanguageModel  # type: ignore
import argparse
import importlib
import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, replace, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import statistics

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
try:  # pragma: no cover - datasets is optional until training runs
    from datasets import DatasetDict, load_dataset
except ModuleNotFoundError:  # pragma: no cover - import guard for --help invocations
    DatasetDict = Any  # type: ignore
    load_dataset = None  # type: ignore

from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          PreTrainedTokenizerBase)

if TYPE_CHECKING:  # pragma: no cover - optional import for typing only
    from services.rag_qwen30b import RAGQwen30b


try:  # pragma: no cover - optional import guard
    from peft import LoraConfig, PeftModel
except ModuleNotFoundError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    PeftModel = None  # type: ignore

try:  # pragma: no cover - optional import guard
    from trl import SFTConfig, SFTTrainer
except ModuleNotFoundError:  # pragma: no cover
    SFTConfig = None  # type: ignore
    SFTTrainer = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for exporting training data from Postgres."""

    dsn: str
    query: str
    output_path: Path
    chunk_size: int = 1_000
    id_column: Optional[str] = None


@dataclass
class ExportResult:
    """Details about an export run."""

    output_path: Path
    count: int
    record_ids: List[int]


@dataclass
class QdrantContextConfig:
    """Configuration for fetching supporting passages from Qdrant."""

    url: str
    collection: str
    api_key: Optional[str] = None
    match_payload_key: str = "doc_id"
    context_payload_key: str = "content"
    title_payload_key: Optional[str] = None
    limit: int = 5


@dataclass
class DomainDatasetConfig:
    """Configuration for building a domain-specialised dataset."""

    export: ExportConfig
    qdrant: Optional[QdrantContextConfig] = None
    join_column: Optional[str] = None
    context_header: str = "Domain context"
    max_context_chars: int = 2400
    include_context_metadata: bool = False


@dataclass
class TrainConfig:
    """Configuration for QLoRA fine-tuning."""

    base_model: str
    train_file: Path
    output_dir: Path
    eval_file: Optional[Path] = None
    system_prompt: Optional[str] = None
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    save_steps: int = 500
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    bf16: bool = True
    packing: bool = True
    use_unsloth: bool = True
    chat_template: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class MergeConfig:
    """Configuration for merging LoRA adapters with base weights."""

    base_model: str
    adapter_path: Path
    output_dir: Path
    safe_serialization: bool = True


@dataclass
class GGUFConfig:
    """Configuration for converting HF weights to GGUF and quantizing."""

    llama_cpp_dir: Path
    hf_model_dir: Path
    gguf_output: Path
    quantize: Optional[str] = None
    quantized_output: Optional[Path] = None


@dataclass
class EvaluationQuery:
    """Single evaluation prompt specification."""

    query: str
    ensure_min_docs: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGEvaluationConfig:
    """Configuration for validating RAG behaviour using a set of queries."""

    queries_path: Path
    output_path: Path
    collections: Sequence[str]
    ensure_min_docs: int = 3
    baseline_report: Optional[Path] = None
    max_queries: Optional[int] = None


@dataclass
class RAGEvaluationResult:
    """Result of a RAG evaluation run."""

    output_path: Path
    aggregate: Dict[str, Any]
    query_metrics: List[Dict[str, Any]]
    comparison: Optional[Dict[str, Any]] = None


@dataclass
class ModelfileConfig:
    """Configuration for rendering an Ollama Modelfile pointing at new weights."""

    template_path: Path
    output_path: Path
    model_name: str = "qwen3-30b-procwise"
    context_window: int = 8192
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRunConfig:
    """Configuration for the complete fine-tuning workflow."""

    export: ExportConfig
    train: TrainConfig
    merge: Optional[MergeConfig] = None
    gguf: Optional[GGUFConfig] = None
    evaluation: Optional[RAGEvaluationConfig] = None
    modelfile: Optional[ModelfileConfig] = None
    min_records: int = 1


@dataclass
class PipelineResult:
    """Result summary for the end-to-end fine-tuning pipeline."""

    export: ExportResult
    adapter_dir: Optional[Path] = None
    merged_model_dir: Optional[Path] = None
    gguf_model_path: Optional[Path] = None
    quantized_model_path: Optional[Path] = None
    modelfile_path: Optional[Path] = None
    evaluation_report: Optional[Path] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_extra_parameters(param_args: Optional[Sequence[str]]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for raw in param_args or []:
        if "=" not in raw:
            raise ValueError(f"Parameter '{raw}' must be in KEY=VALUE format")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Parameter key cannot be empty")
        params[key] = value.strip()
    return params


def _load_callable(dotted: str) -> Callable[[], Any]:
    """Import a zero-argument factory from a dotted path (module:function)."""

    if ":" not in dotted:
        raise ValueError("Callable path must be in module.submodule:factory format")
    module_name, attr_name = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"{dotted} is not callable")
    return factory


def export_training_data(config: ExportConfig) -> ExportResult:
    """Export data from Postgres and persist it as JSONL."""
    LOGGER.info("Exporting training data to %s", config.output_path)
    exported = 0
    record_ids: List[int] = []
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg2.connect(config.dsn) as conn, conn.cursor(name="procwise_export") as cursor:
        cursor.itersize = config.chunk_size
        cursor.execute(config.query)
        field_names = [desc.name for desc in cursor.description]
        if not {"instruction", "output"}.issubset(field_names):
            raise ValueError(
                "Query must return at least 'instruction' and 'output' columns. Columns present: %s"
                % ", ".join(field_names)
            )
        optional_input = "input" in field_names
        if config.id_column and config.id_column not in field_names:
            raise ValueError(
                "Query must include the id_column '%s'. Columns present: %s"
                % (config.id_column, ", ".join(field_names))
            )
        with config.output_path.open("w", encoding="utf-8") as handle:
            for row in cursor:
                row_dict = dict(zip(field_names, row))
                record = {
                    "instruction": row_dict.get("instruction") or "",
                    "output": row_dict.get("output") or "",
                }
                if optional_input:
                    record["input"] = row_dict.get("input") or ""
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
                if config.id_column:
                    identifier = row_dict.get(config.id_column)
                    if identifier is not None:
                        try:
                            record_ids.append(int(identifier))
                        except (TypeError, ValueError):
                            LOGGER.debug("Unable to coerce %s to int for exported id", identifier)
        LOGGER.info("Exported %d records", exported)
    return ExportResult(config.output_path, exported, record_ids)


def _get_nested_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _fetch_qdrant_contexts(
    client: QdrantClient,
    cfg: QdrantContextConfig,
    match_value: Any,
) -> List[Dict[str, Any]]:
    if match_value in (None, "", []):
        return []
    condition = qmodels.FieldCondition(
        key=cfg.match_payload_key,
        match=qmodels.MatchValue(value=match_value),
    )
    records, _ = client.scroll(
        collection_name=cfg.collection,
        scroll_filter=qmodels.Filter(must=[condition]),
        with_payload=True,
        with_vectors=False,
        limit=cfg.limit,
    )
    contexts: List[Dict[str, Any]] = []
    for record in records:
        payload = record.payload or {}
        text = _get_nested_value(payload, cfg.context_payload_key)
        if not text:
            continue
        snippet = {
            "text": str(text).strip(),
            "payload": payload,
        }
        if cfg.title_payload_key:
            snippet["title"] = payload.get(cfg.title_payload_key)
        contexts.append(snippet)
    return contexts


def _append_context_to_input(
    base_input: str,
    contexts: List[Dict[str, Any]],
    cfg: DomainDatasetConfig,
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    if not contexts:
        return base_input, None
    lines: List[str] = []
    consumed = 0
    for chunk in contexts:
        text = chunk.get("text", "")
        if not text:
            continue
        line = text
        if chunk.get("title"):
            line = f"{chunk['title']}: {text}"
        remaining = cfg.max_context_chars - consumed if cfg.max_context_chars else None
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and len(line) > remaining:
            line = line[:remaining]
        lines.append(line.strip())
        consumed += len(line)
        if cfg.max_context_chars and consumed >= cfg.max_context_chars:
            break
    if not lines:
        return base_input, None
    context_text = f"{cfg.context_header}\n- " + "\n- ".join(lines)
    combined = context_text if not base_input else f"{base_input}\n\n{context_text}"
    metadata = contexts if cfg.include_context_metadata else None
    return combined, metadata


def build_domain_dataset(config: DomainDatasetConfig) -> ExportResult:
    """Export training data augmented with Qdrant domain context."""

    qdrant_client: Optional[QdrantClient] = None
    if config.qdrant:
        qdrant_client = QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key,
        )

    export_cfg = config.export
    LOGGER.info("Building domain dataset to %s", export_cfg.output_path)
    export_cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with psycopg2.connect(export_cfg.dsn) as conn, conn.cursor(name="procwise_domain_dataset") as cursor:
        cursor.itersize = export_cfg.chunk_size
        cursor.execute(export_cfg.query)
        field_names = [desc.name for desc in cursor.description]
        required = {"instruction", "output"}
        if not required.issubset(field_names):
            raise ValueError(
                "Query must include %s columns. Found: %s"
                % (", ".join(sorted(required)), ", ".join(field_names))
            )
        join_column = config.join_column
        if join_column and join_column not in field_names:
            raise ValueError(
                f"join_column '{join_column}' not returned by query. Columns: {', '.join(field_names)}"
            )
        optional_input = "input" in field_names
        with export_cfg.output_path.open("w", encoding="utf-8") as handle:
            for row in cursor:
                row_data = dict(zip(field_names, row))
                record_input = row_data.get("input") or "" if optional_input else ""
                context_metadata: Optional[List[Dict[str, Any]]] = None
                if qdrant_client and config.qdrant and join_column:
                    join_value = row_data.get(join_column)
                    snippets = _fetch_qdrant_contexts(qdrant_client, config.qdrant, join_value)
                    record_input, context_metadata = _append_context_to_input(record_input, snippets, config)
                entry: Dict[str, Any] = {
                    "instruction": row_data.get("instruction") or "",
                    "output": row_data.get("output") or "",
                }
                if record_input:
                    entry["input"] = record_input
                if context_metadata is not None:
                    entry["context_sources"] = context_metadata
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                exported += 1
    LOGGER.info("Domain dataset contains %d samples", exported)
    return ExportResult(export_cfg.output_path, exported, [])


def _format_messages(
    sample: dict,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TrainConfig,
) -> str:
    instruction = (sample.get(cfg.instruction_field) or "").strip()
    input_part = (sample.get(cfg.input_field) or "").strip()
    output = (sample.get(cfg.output_field) or "").strip()

    user_content = instruction
    if input_part:
        user_content = f"{instruction}\n\n{input_part}" if instruction else input_part

    messages = []
    if cfg.system_prompt:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})

    if cfg.chat_template is not None or getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": False}
        if cfg.chat_template:
            kwargs["chat_template"] = cfg.chat_template
        return tokenizer.apply_chat_template(messages, **kwargs)

    # Default to a simple [INST] template compatible with Mistral/Llama style models
    prompt = """
<s>[INST] {user} [/INST]
{assistant}</s>
""".strip()
    return prompt.format(user=user_content, assistant=output)


def _load_dataset(cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    if load_dataset is None:
        raise ModuleNotFoundError(
            "datasets is not installed. Install the requirements listed in requirements.txt before training."
        )
    data_files = {"train": str(cfg.train_file)}
    if cfg.eval_file is not None:
        data_files["validation"] = str(cfg.eval_file)
    dataset = load_dataset("json", data_files=data_files)

    def _formatter(example: dict) -> dict:
        return {"text": _format_messages(example, tokenizer, cfg)}

    formatted = dataset.map(_formatter, remove_columns=[col for col in dataset["train"].column_names])
    return formatted


def _split_target_modules(modules: Sequence[str]) -> List[str]:
    items: List[str] = []
    for module in modules:
        if isinstance(module, str):
            items.extend(part.strip() for part in module.split(",") if part.strip())
    return items


def load_evaluation_queries(path: Path) -> List[EvaluationQuery]:
    """Load evaluation prompts from a JSON or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation queries file not found: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("Evaluation queries file is empty")

    records: List[Any] = []
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "queries" in payload:
            queries = payload["queries"]
            if isinstance(queries, list):
                records = queries
        else:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        # Fall back to JSONL parsing
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    queries: List[EvaluationQuery] = []
    for record in records:
        query_text = ""
        ensure_docs: Optional[int] = None
        metadata: Dict[str, Any] = {}
        if isinstance(record, str):
            query_text = record.strip()
        elif isinstance(record, dict):
            query_text = str(record.get("query") or record.get("prompt") or "").strip()
            ensure_docs = record.get("ensure_min_docs")
            metadata = {
                k: v
                for k, v in record.items()
                if k not in {"query", "prompt", "ensure_min_docs"}
            }
        if not query_text:
            continue
        queries.append(EvaluationQuery(query=query_text, ensure_min_docs=ensure_docs, metadata=metadata))

    if not queries:
        raise ValueError(f"No valid queries found in {path}")
    return queries


def _safe_numeric(values: Iterable[Any]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            cleaned.append(float(value))
            continue
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return cleaned


def _mean(values: Iterable[Any]) -> float:
    nums = _safe_numeric(values)
    return float(statistics.mean(nums)) if nums else 0.0


def _median(values: Iterable[Any]) -> float:
    nums = _safe_numeric(values)
    return float(statistics.median(nums)) if nums else 0.0


def _ratio(rows: Iterable[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if predicate(row))
    return hits / len(rows)


def _load_baseline_metrics(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Unable to parse baseline report at %s", path)
        return None
    aggregate = data.get("aggregate")
    return aggregate if isinstance(aggregate, dict) else None


def evaluate_rag_model(
    rag_factory: Callable[[], "RAGQwen30b"],
    cfg: RAGEvaluationConfig,
) -> RAGEvaluationResult:
    """Run the RAG pipeline over a benchmark set and persist metrics."""

    queries = load_evaluation_queries(cfg.queries_path)
    if cfg.max_queries:
        queries = queries[: cfg.max_queries]
    rag = rag_factory()
    query_metrics: List[Dict[str, Any]] = []
    for query in queries:
        ensure_docs = query.ensure_min_docs or cfg.ensure_min_docs
        response = rag.answer(
            query.query,
            ensure_min_docs=ensure_docs,
            collections=list(cfg.collections),
        )
        diagnostics = dict(response.get("diagnostics") or {})
        diagnostics["query"] = query.query
        diagnostics["sources"] = response.get("sources", [])
        diagnostics["answer_preview"] = (response.get("answer") or "")[:512]
        diagnostics["doc_diversity"] = len(set(diagnostics["sources"]))
        diagnostics["multi_doc"] = diagnostics["doc_diversity"] >= max(ensure_docs or 3, 3)
        answer_text = (response.get("answer") or "").strip()
        diagnostics["refused"] = answer_text.startswith("I don't have enough information")
        diagnostics["metadata"] = query.metadata
        query_metrics.append(diagnostics)

    aggregate = {
        "queries_evaluated": len(query_metrics),
        "avg_dense_candidates": _mean(row.get("dense") for row in query_metrics),
        "avg_reranked": _mean(row.get("after_rerank") for row in query_metrics),
        "avg_deduped": _mean(row.get("after_dedupe") for row in query_metrics),
        "avg_capped": _mean(row.get("after_cap") for row in query_metrics),
        "median_packed_chars": _median(row.get("packed_chars") for row in query_metrics),
        "avg_latency_seconds": _mean(row.get("elapsed_seconds") for row in query_metrics),
        "doc_diversity_avg": _mean(row.get("doc_diversity") for row in query_metrics),
        "multi_doc_rate": _ratio(query_metrics, lambda row: bool(row.get("multi_doc"))),
        "refusal_rate": _ratio(query_metrics, lambda row: bool(row.get("refused"))),
    }

    baseline = _load_baseline_metrics(cfg.baseline_report)
    comparison: Optional[Dict[str, Dict[str, float]]] = None
    if baseline:
        comparison = {}
        for key, current in aggregate.items():
            baseline_value = baseline.get(key)
            if baseline_value is None:
                continue
            comparison[key] = {
                "current": current,
                "baseline": float(baseline_value),
                "delta": current - float(baseline_value),
            }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "aggregate": aggregate,
        "queries": query_metrics,
        "comparison": comparison,
        "collections": list(cfg.collections),
    }
    cfg.output_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "RAG evaluation complete: %d queries, multi-doc rate %.2f, refusal rate %.2f",
        aggregate["queries_evaluated"],
        aggregate["multi_doc_rate"],
        aggregate["refusal_rate"],
    )
    return RAGEvaluationResult(
        output_path=cfg.output_path,
        aggregate=aggregate,
        query_metrics=query_metrics,
        comparison=comparison,
    )


def _format_extra_parameters(params: Mapping[str, Any]) -> str:
    if not params:
        return ""
    lines = []
    for key, value in params.items():
        lines.append(f"PARAMETER {key} {value}")
    return "\n".join(lines)


def write_modelfile(cfg: ModelfileConfig, weights_path: Path) -> Path:
    """Render a Modelfile from the provided template."""

    if not cfg.template_path.exists():
        raise FileNotFoundError(f"Template not found at {cfg.template_path}")
    template = cfg.template_path.read_text(encoding="utf-8")
    rendered = template.format(
        MODEL_PATH=weights_path.as_posix(),
        MODEL_NAME=cfg.model_name,
        CONTEXT_WINDOW=cfg.context_window,
        TEMPERATURE=cfg.temperature,
        TOP_P=cfg.top_p,
        REPEAT_PENALTY=cfg.repeat_penalty,
        EXTRA_PARAMETERS=_format_extra_parameters(cfg.extra_parameters),
    )
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Wrote Modelfile to %s pointing at %s", cfg.output_path, weights_path)
    return cfg.output_path


def train_model(cfg: TrainConfig) -> Path:
    LOGGER.info("Starting QLoRA training for base model %s", cfg.base_model)

    if cfg.use_unsloth and FastLanguageModel is None:
        raise RuntimeError("unsloth is not installed but --use-unsloth was requested")
    if SFTTrainer is None or SFTConfig is None:
        raise ModuleNotFoundError("trl is not installed. Install the training dependencies before running.")
    if not cfg.use_unsloth and LoraConfig is None:
        raise ModuleNotFoundError("peft is not installed. Install the training dependencies before running.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    if cfg.use_unsloth:
        assert FastLanguageModel is not None  # for mypy
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            dtype="bfloat16" if cfg.bf16 else "float16",
            load_in_4bit=True,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=_split_target_modules(cfg.target_modules),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=quant_config,
            device_map="auto",
        )

    dataset = _load_dataset(cfg, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=None if cfg.use_unsloth else LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_split_target_modules(cfg.target_modules),
        ),
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            bf16=cfg.bf16,
            packing=cfg.packing,
            max_seq_length=cfg.max_seq_length,
            seed=cfg.seed,
        ),
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)
    LOGGER.info("Saved LoRA adapter to %s", cfg.output_dir)
    return cfg.output_dir


def merge_lora(config: MergeConfig) -> Path:
    if PeftModel is None:
        raise ModuleNotFoundError("peft is not installed. Install the training dependencies before running merge.")
    LOGGER.info("Merging adapter %s into base model %s", config.adapter_path, config.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model, device_map="cpu")
    lora_model = PeftModel.from_pretrained(base_model, str(config.adapter_path))
    merged = lora_model.merge_and_unload()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(config.output_dir, safe_serialization=config.safe_serialization)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(config.adapter_path), use_fast=True)
    except OSError:
        LOGGER.warning("Tokenizer not found in adapter path, falling back to base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    tokenizer.save_pretrained(config.output_dir)

    LOGGER.info("Merged model saved to %s", config.output_dir)
    return config.output_dir


def convert_to_gguf(cfg: GGUFConfig) -> Path:
    LOGGER.info("Converting %s to GGUF via llama.cpp at %s", cfg.hf_model_dir, cfg.llama_cpp_dir)
    converter = cfg.llama_cpp_dir / "convert-hf-to-gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"convert-hf-to-gguf.py not found at {converter}")

    cfg.gguf_output.parent.mkdir(parents=True, exist_ok=True)

    convert_cmd = [
        "python",
        str(converter),
        str(cfg.hf_model_dir),
        "--outfile",
        str(cfg.gguf_output),
    ]
    LOGGER.debug("Running converter command: %s", " ".join(shlex.quote(part) for part in convert_cmd))
    subprocess.run(convert_cmd, check=True)

    if cfg.quantize:
        quantize_bin = cfg.llama_cpp_dir / "quantize"
        if not quantize_bin.exists():
            raise FileNotFoundError(f"quantize binary not found at {quantize_bin}")
        if cfg.quantized_output is None:
            raise ValueError("quantized_output must be provided when quantize is set")
        quant_cmd = [
            str(quantize_bin),
            str(cfg.gguf_output),
            str(cfg.quantized_output),
            cfg.quantize,
        ]
        LOGGER.debug("Running quantize command: %s", " ".join(shlex.quote(part) for part in quant_cmd))
        subprocess.run(quant_cmd, check=True)
        LOGGER.info("Quantized model written to %s", cfg.quantized_output)
        return cfg.quantized_output

    LOGGER.info("GGUF model written to %s", cfg.gguf_output)
    return cfg.gguf_output


def run_full_pipeline(
    config: PipelineRunConfig,
    rag_factory: Optional[Callable[[], "RAGQwen30b"]] = None,
) -> PipelineResult:
    """Execute the export → train → merge → convert workflow."""

    export_result = export_training_data(config.export)
    if export_result.count < config.min_records:
        LOGGER.info(
            "Exported %d records which is below the minimum %d; skipping fine-tuning",
            export_result.count,
            config.min_records,
        )
        return PipelineResult(export=export_result)

    train_cfg = replace(config.train, train_file=config.export.output_path)
    adapter_dir = train_model(train_cfg)

    merged_dir: Optional[Path] = None
    if config.merge is not None:
        merge_cfg = replace(
            config.merge,
            base_model=train_cfg.base_model,
            adapter_path=adapter_dir,
        )
        merged_dir = merge_lora(merge_cfg)
    else:
        merged_dir = None

    gguf_path: Optional[Path] = None
    quantized_path: Optional[Path] = None
    if config.gguf is not None:
        if merged_dir is None:
            raise ValueError("GGUF conversion requested without merged model directory")
        gguf_cfg = replace(config.gguf, hf_model_dir=merged_dir)
        gguf_result = convert_to_gguf(gguf_cfg)
        if gguf_cfg.quantize:
            quantized_path = gguf_result
            gguf_path = gguf_cfg.gguf_output
        else:
            gguf_path = gguf_result

    modelfile_path: Optional[Path] = None
    weights_for_inference = quantized_path or gguf_path or merged_dir
    if config.modelfile is not None:
        if weights_for_inference is None:
            raise ValueError("Modelfile rendering requested but no model weights are available")
        modelfile_path = write_modelfile(config.modelfile, weights_for_inference)

    evaluation_report: Optional[Path] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    if config.evaluation is not None:
        if rag_factory is None:
            LOGGER.warning("Evaluation config supplied without rag_factory; skipping RAG validation stage.")
        else:
            eval_result = evaluate_rag_model(rag_factory, config.evaluation)
            evaluation_report = eval_result.output_path
            evaluation_metrics = eval_result.aggregate

    return PipelineResult(
        export=export_result,
        adapter_dir=adapter_dir,
        merged_model_dir=merged_dir,
        gguf_model_path=gguf_path,
        quantized_model_path=quantized_path,
        modelfile_path=modelfile_path,
        evaluation_report=evaluation_report,
        evaluation_metrics=evaluation_metrics,
    )


def _add_common_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model to fine-tune")
    parser.add_argument("--train-file", required=True, type=Path, help="Path to training JSONL file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write LoRA adapter")
    parser.add_argument("--eval-file", type=Path, help="Optional validation JSONL file")
    parser.add_argument("--system-prompt", help="Optional system prompt to prepend in chat templates")
    parser.add_argument("--instruction-field", default="instruction", help="Column containing instructions")
    parser.add_argument("--input-field", default="input", help="Column containing optional inputs")
    parser.add_argument("--output-field", default="output", help="Column containing responses")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma separated list of target modules for LoRA",
    )
    parser.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 training")
    parser.add_argument("--no-packing", action="store_true", help="Disable sequence packing")
    parser.add_argument("--use-unsloth", action="store_true", help="Use unsloth FastLanguageModel for faster training")
    parser.add_argument("--chat-template", help="Force usage of tokenizer chat template by id")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-data", help="Export instruction tuning data from Postgres")
    export_parser.add_argument("--dsn", required=True, help="Postgres DSN e.g. postgresql://user:pass@host/db")
    export_parser.add_argument("--query", required=True, help="SQL query returning instruction/input/output columns")
    export_parser.add_argument("--output", required=True, type=Path, help="Destination JSONL file")
    export_parser.add_argument("--chunk-size", type=int, default=1000, help="Streaming chunk size for Postgres cursor")
    export_parser.add_argument(
        "--id-column",
        help="Optional identifier column to track which records were exported",
    )

    domain_parser = subparsers.add_parser(
        "build-domain-dataset",
        help="Export Postgres instruction data enriched with Qdrant context",
    )
    domain_parser.add_argument("--dsn", required=True, help="Postgres DSN for sourcing instruction data")
    domain_parser.add_argument("--query", required=True, help="SQL query returning instruction/input/output columns")
    domain_parser.add_argument("--output", required=True, type=Path, help="Destination JSONL file")
    domain_parser.add_argument("--chunk-size", type=int, default=1000, help="Streaming chunk size for Postgres cursor")
    domain_parser.add_argument(
        "--join-column",
        default="doc_id",
        help="Column from the SQL result used to look up context in Qdrant payloads",
    )
    domain_parser.add_argument(
        "--context-header",
        default="Domain context",
        help="Header inserted before appended Qdrant passages",
    )
    domain_parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2400,
        help="Maximum cumulative characters of appended Qdrant context",
    )
    domain_parser.add_argument(
        "--include-context-metadata",
        action="store_true",
        help="Include raw context payload metadata alongside each record",
    )
    domain_parser.add_argument("--qdrant-url", required=True, help="Qdrant HTTP URL, e.g. http://localhost:6333")
    domain_parser.add_argument("--qdrant-api-key", help="Optional Qdrant API key")
    domain_parser.add_argument("--qdrant-collection", required=True, help="Qdrant collection to source passages from")
    domain_parser.add_argument(
        "--qdrant-match-field",
        default="doc_id",
        help="Payload field used to match the join column value",
    )
    domain_parser.add_argument(
        "--qdrant-context-field",
        default="content",
        help="Payload field containing the passage text to inject",
    )
    domain_parser.add_argument(
        "--qdrant-title-field",
        help="Optional payload field for a human readable title prefixed to each passage",
    )
    domain_parser.add_argument(
        "--qdrant-limit",
        type=int,
        default=5,
        help="Maximum number of passages to retrieve per record",
    )

    train_parser = subparsers.add_parser("train", help="Run QLoRA fine-tuning with TRL")
    _add_common_train_arguments(train_parser)

    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base model weights")
    merge_parser.add_argument("--base-model", required=True, help="Base model name or path")
    merge_parser.add_argument("--adapter-path", required=True, type=Path, help="Directory containing adapter weights")
    merge_parser.add_argument("--output-dir", required=True, type=Path, help="Where to write merged model")
    merge_parser.add_argument(
        "--unsafe-serialization", action="store_true", help="Use legacy torch.save instead of safetensors"
    )

    gguf_parser = subparsers.add_parser("convert-gguf", help="Convert merged model to GGUF")
    gguf_parser.add_argument("--llama-cpp-dir", required=True, type=Path, help="Path to llama.cpp repository")
    gguf_parser.add_argument("--hf-model-dir", required=True, type=Path, help="Path to merged Hugging Face model")
    gguf_parser.add_argument("--gguf-output", required=True, type=Path, help="Path to write float GGUF weights")
    gguf_parser.add_argument("--quantize", help="Quantization preset, e.g. Q4_K_M")
    gguf_parser.add_argument("--quantized-output", type=Path, help="Output path for quantized GGUF weights")

    modelfile_parser = subparsers.add_parser("render-modelfile", help="Render an Ollama Modelfile from a template")
    modelfile_parser.add_argument("--template", required=True, type=Path, help="Path to Modelfile template")
    modelfile_parser.add_argument("--output", required=True, type=Path, help="Destination Modelfile path")
    modelfile_parser.add_argument("--weights", required=True, type=Path, help="Model weights to reference (GGUF or HF)")
    modelfile_parser.add_argument("--model-name", default="qwen3-30b-procwise", help="Name to register with Ollama")
    modelfile_parser.add_argument("--context-window", type=int, default=8192)
    modelfile_parser.add_argument("--temperature", type=float, default=0.2)
    modelfile_parser.add_argument("--top-p", type=float, default=0.9)
    modelfile_parser.add_argument("--repeat-penalty", type=float, default=1.05)
    modelfile_parser.add_argument(
        "--parameter",
        action="append",
        help="Additional PARAMETER lines in KEY=VALUE form (repeatable)",
    )

    eval_parser = subparsers.add_parser("rag-eval", help="Evaluate the RAG pipeline across multiple collections")
    eval_parser.add_argument("--queries", required=True, type=Path, help="JSON/JSONL file containing evaluation prompts")
    eval_parser.add_argument("--collections", required=True, nargs="+", help="Qdrant collections to evaluate against")
    eval_parser.add_argument("--output", required=True, type=Path, help="Where to store the evaluation report JSON")
    eval_parser.add_argument("--baseline", type=Path, help="Optional previous report JSON for comparison")
    eval_parser.add_argument("--max-queries", type=int, help="Optional cap on number of queries to run")
    eval_parser.add_argument("--ensure-min-docs", type=int, default=3, help="Minimum distinct doc_ids to enforce")
    eval_parser.add_argument(
        "--rag-factory",
        required=True,
        help="Dotted path to a zero-arg callable returning services.rag_qwen30b.RAGQwen30b",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> Path | ExportResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "export-data":
        cfg = ExportConfig(
            dsn=args.dsn,
            query=args.query,
            output_path=args.output,
            chunk_size=args.chunk_size,
            id_column=args.id_column,
        )
        return export_training_data(cfg)

    if args.command == "build-domain-dataset":
        export_cfg = ExportConfig(
            dsn=args.dsn,
            query=args.query,
            output_path=args.output,
            chunk_size=args.chunk_size,
        )
        qdrant_cfg = QdrantContextConfig(
            url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            collection=args.qdrant_collection,
            match_payload_key=args.qdrant_match_field,
            context_payload_key=args.qdrant_context_field,
            title_payload_key=args.qdrant_title_field,
            limit=args.qdrant_limit,
        )
        cfg = DomainDatasetConfig(
            export=export_cfg,
            qdrant=qdrant_cfg,
            join_column=args.join_column,
            context_header=args.context_header,
            max_context_chars=args.max_context_chars,
            include_context_metadata=args.include_context_metadata,
        )
        return build_domain_dataset(cfg)

    if args.command == "train":
        cfg = TrainConfig(
            base_model=args.base_model,
            train_file=args.train_file,
            output_dir=args.output_dir,
            eval_file=args.eval_file,
            system_prompt=args.system_prompt,
            instruction_field=args.instruction_field,
            input_field=args.input_field,
            output_field=args.output_field,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","),
            bf16=not args.no_bf16,
            packing=not args.no_packing,
            use_unsloth=args.use_unsloth,
            chat_template=args.chat_template,
            seed=args.seed,
        )
        return train_model(cfg)

    if args.command == "merge":
        cfg = MergeConfig(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            safe_serialization=not args.unsafe_serialization,
        )
        return merge_lora(cfg)

    if args.command == "convert-gguf":
        cfg = GGUFConfig(
            llama_cpp_dir=args.llama_cpp_dir,
            hf_model_dir=args.hf_model_dir,
            gguf_output=args.gguf_output,
            quantize=args.quantize,
            quantized_output=args.quantized_output,
        )
        return convert_to_gguf(cfg)

    if args.command == "render-modelfile":
        params = _parse_extra_parameters(args.parameter)
        cfg = ModelfileConfig(
            template_path=args.template,
            output_path=args.output,
            model_name=args.model_name,
            context_window=args.context_window,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            extra_parameters=params,
        )
        return write_modelfile(cfg, args.weights)

    if args.command == "rag-eval":
        rag_factory = _load_callable(args.rag_factory)
        cfg = RAGEvaluationConfig(
            queries_path=args.queries,
            output_path=args.output,
            collections=args.collections,
            ensure_min_docs=args.ensure_min_docs,
            baseline_report=args.baseline,
            max_queries=args.max_queries,
        )
        return evaluate_rag_model(rag_factory, cfg).output_path

    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
