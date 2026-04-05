"""Unified Hugging Face Trainer + LoRA fine-tuning for DynaHate and HateCheck."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Literal

import evaluate as hf_evaluate
import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from hatelens.datasets import create_dynahate_dataset, create_hatecheck_dataset
from hatelens.modeling import ID2LABEL, LABEL2ID
from hatelens.paths import data_dir, repo_root

logger = logging.getLogger(__name__)

DatasetName = Literal["dynahate", "hatecheck"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_wandb_init(project: str, config: dict[str, Any]) -> bool:
    """Return True if W&B logging is active (opt-in via WANDB_ENABLED=1)."""
    if os.environ.get("WANDB_ENABLED", "").lower() not in ("1", "true", "yes"):
        os.environ["WANDB_MODE"] = "disabled"
        return False
    try:
        import wandb
    except ImportError:
        logger.warning("WANDB_ENABLED set but wandb not installed; disabling.")
        os.environ["WANDB_MODE"] = "disabled"
        return False
    entity = os.environ.get("WANDB_ENTITY")
    kwargs: dict[str, Any] = {"project": project, "config": config}
    if entity:
        kwargs["entity"] = entity
    wandb.init(**kwargs)
    return True


def _load_splits(dataset: DatasetName):
    if dataset == "dynahate":
        return create_dynahate_dataset(data_dir()), "text"
    return create_hatecheck_dataset(data_dir()), "test_case"


def run_training(config_path: Path, dataset: DatasetName, *, seed: int = 123) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_path = config_path.resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_checkpoint = cfg["model_checkpoint"]
    model_name = cfg.get("model_name", "model")

    use_wandb = _maybe_wandb_init(f"hatlens-{model_name}-{dataset}", cfg)
    set_seed(int(cfg.get("seed", seed)))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    peft_config = LoraConfig(
        task_type=cfg["task_type"],
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    raw, text_key = _load_splits(dataset)

    def tokenize_fn(examples: dict[str, list]):
        return tokenizer(
            examples[text_key],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized = raw.map(tokenize_fn, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = hf_evaluate.load("accuracy")
    precision = hf_evaluate.load("precision")
    recall = hf_evaluate.load("recall")
    f1 = hf_evaluate.load("f1")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "precision": precision.compute(predictions=predictions, references=labels)["precision"],
            "recall": recall.compute(predictions=predictions, references=labels)["recall"],
            "f1": f1.compute(predictions=predictions, references=labels)["f1"],
        }

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    log_dir = Path(cfg["logging_dir"])
    if not log_dir.is_absolute():
        log_dir = repo_root() / log_dir

    use_fp16 = torch.cuda.is_available() and bool(cfg.get("fp16", True))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8 if dataset == "dynahate" else 2))
    train_bs = int(cfg.get("per_device_train_batch_size", 8 if dataset == "dynahate" else 4))
    eval_bs = int(cfg.get("per_device_eval_batch_size", train_bs))
    workers = min(os.cpu_count() or 1, int(cfg.get("dataloader_num_workers", 8)))

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=float(cfg["num_train_epochs"]),
        weight_decay=float(cfg["weight_decay"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.get("metric_for_best_model", "f1"),
        greater_is_better=True,
        logging_dir=str(log_dir),
        logging_steps=int(cfg["logging_steps"]),
        report_to="wandb" if use_wandb else "none",
        fp16=use_fp16,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=workers,
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Starting training dataset=%s config=%s", dataset, config_path)
    trainer.train()
    best_path = out_dir / "best_checkpoint"
    best_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_path))
    logger.info("Saved best model to %s", best_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="HateLens LoRA fine-tuning")
    p.add_argument("config_file", type=Path, help="YAML config (see experiments/*/config.yaml)")
    p.add_argument(
        "--dataset",
        choices=("dynahate", "hatecheck"),
        required=True,
        help="Which dataset to fine-tune on",
    )
    args = p.parse_args(argv)
    run_training(args.config_file, args.dataset)


if __name__ == "__main__":
    main()
