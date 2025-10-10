"""Context-aware model fine-tuning utilities for ProcWise agents."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # Optional heavy dependencies – import lazily and tolerate absence.
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except Exception:  # pragma: no cover - executed only when deps unavailable
    torch = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialise_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class TrainingConfig:
    data_dir: str = "./data/conversations"
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    output_dir: str = "./output/context-aware-model"
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-5
    max_length: int = 2048


class ConversationDatasetWriter:
    """Persist conversation snapshots as supervised fine-tuning records."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        _ensure_dir(self.data_dir)

    def write_record(
        self,
        *,
        context_text: str,
        response_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        if not context_text.strip() or not response_text.strip():
            raise ValueError("Context and response text must be non-empty")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        record_id = uuid.uuid4().hex
        payload: Dict[str, Any] = {
            "context": context_text.strip(),
            "response": response_text.strip(),
        }
        if metadata:
            payload["metadata"] = metadata
        path = self.data_dir / f"conversation-{timestamp}-{record_id}.json"
        path.write_text(_serialise_json(payload), encoding="utf-8")
        return path


def load_conversations(data_dir: str) -> Dataset:
    if Dataset is None:  # pragma: no cover - triggered only without datasets dependency
        raise RuntimeError("datasets library not available; cannot load conversations")
    records: List[Dict[str, str]] = []
    directory = Path(data_dir)
    if not directory.exists():
        raise RuntimeError(f"Training data directory {data_dir} does not exist")
    for file in directory.glob("*.json"):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse training conversation %s", file)
            continue
        context = str(payload.get("context", "")).strip()
        response = str(payload.get("response", "")).strip()
        if not context or not response:
            continue
        text = f"### Context:\n{context}\n\n### Response:\n{response}"
        records.append({"text": text})
    if not records:
        raise RuntimeError(f"No training records found in {data_dir}")
    return Dataset.from_list(records)


def _tokenize_examples(
    examples: Dict[str, List[str]],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, Any]:
    encodings = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def fine_tune_model(cfg: TrainingConfig) -> Dict[str, Any]:
    if any(dep is None for dep in (Dataset, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)):
        raise RuntimeError(
            "Transformers and datasets libraries are required for context training"
        )

    dataset = load_conversations(cfg.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    tokenised = dataset.map(
        lambda batch: _tokenize_examples(batch, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=["text"],
    )
    tokenised.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
        evaluation_strategy="no",
        fp16=torch.cuda.is_available() if torch else False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenised)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return {
        "output_dir": cfg.output_dir,
        "model_name": cfg.model_name,
        "epochs": cfg.epochs,
        "records": len(dataset),
    }


class ContextTrainer:
    """High-level facade for context-aware fine-tuning."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()

    def train(self, **overrides: Any) -> Dict[str, Any]:
        cfg = self.config
        if overrides:
            cfg = replace(
                cfg,
                **{
                    key: overrides[key]
                    for key in overrides
                    if hasattr(cfg, key)
                },
            )
        try:
            return fine_tune_model(cfg)
        except RuntimeError as exc:
            logger.warning("Context training skipped: %s", exc)
            return {"status": "skipped", "reason": str(exc)}
        except Exception:  # pragma: no cover - heavy training failures logged once
            logger.exception("Context training failed")
            return {"status": "failed"}

    def build_writer(self, data_dir: Optional[str] = None) -> ConversationDatasetWriter:
        target_dir = data_dir or self.config.data_dir
        return ConversationDatasetWriter(target_dir)


if __name__ == "__main__":  # pragma: no cover - manual training entrypoint
    cfg = TrainingConfig(
        data_dir=os.getenv("PROCWISE_CONTEXT_DATA", TrainingConfig.data_dir),
        output_dir=os.getenv("PROCWISE_CONTEXT_MODEL", TrainingConfig.output_dir),
        model_name=os.getenv("PROCWISE_CONTEXT_BASE_MODEL", TrainingConfig.model_name),
    )
    trainer = ContextTrainer(cfg)
    summary = trainer.train()
    print(json.dumps(summary, indent=2))

