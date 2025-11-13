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

import argparse
import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence

import psycopg2
try:  # pragma: no cover - datasets is optional until training runs
    from datasets import DatasetDict, load_dataset
except ModuleNotFoundError:  # pragma: no cover - import guard for --help invocations
    DatasetDict = Any  # type: ignore
    load_dataset = None  # type: ignore

from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          PreTrainedTokenizerBase)

try:  # pragma: no cover - optional import
    from unsloth import FastLanguageModel  # type: ignore
except Exception:  # pragma: no cover - optional import
    FastLanguageModel = None

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
    use_unsloth: bool = False
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
class PipelineRunConfig:
    """Configuration for the complete fine-tuning workflow."""

    export: ExportConfig
    train: TrainConfig
    merge: Optional[MergeConfig] = None
    gguf: Optional[GGUFConfig] = None
    min_records: int = 1


@dataclass
class PipelineResult:
    """Result summary for the end-to-end fine-tuning pipeline."""

    export: ExportResult
    adapter_dir: Optional[Path] = None
    merged_model_dir: Optional[Path] = None
    gguf_model_path: Optional[Path] = None
    quantized_model_path: Optional[Path] = None


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


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


def run_full_pipeline(config: PipelineRunConfig) -> PipelineResult:
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

    return PipelineResult(
        export=export_result,
        adapter_dir=adapter_dir,
        merged_model_dir=merged_dir,
        gguf_model_path=gguf_path,
        quantized_model_path=quantized_path,
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

    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
