"""Unified Hugging Face Trainer + LoRA fine-tuning (binary or structured multi-head)."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from hatelens.datasets import (
    create_dynahate_dataset,
    create_hatecheck_dataset,
    create_hateeval_dataset,
)
from hatelens.evaluation import classification_metrics
from hatelens.paths import data_dir, repo_root
from hatelens.peft_factory import apply_peft, build_base_sequence_classifier, build_peft_config
from hatelens.registry import resolve_model_id
from hatelens.training_artifacts import write_config_resolved, write_train_metrics_json

logger = logging.getLogger(__name__)

BinaryDatasetName = Literal["dynahate", "hatecheck", "hateeval"]
StructuredDatasetName = Literal["dynahate", "hateeval", "hatexplain", "dynahate_hatexplain"]
TrainDatasetName = Literal[
    "dynahate",
    "hatecheck",
    "hateeval",
    "hatexplain",
    "dynahate_hatexplain",
]

BINARY_DATASETS: frozenset[str] = frozenset({"dynahate", "hatecheck", "hateeval"})
STRUCTURED_DATASETS: frozenset[str] = frozenset(
    {"dynahate", "hateeval", "hatexplain", "dynahate_hatexplain"}
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    canonical = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _softmax_probs_hate(logits: np.ndarray) -> np.ndarray:
    """Column 1 probability for binary logits (N, 2)."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    return p[:, 1]


def _is_smoke(cfg: dict[str, Any]) -> bool:
    if os.environ.get("HATELENS_SMOKE", "").lower() in ("1", "true", "yes"):
        return True
    return bool(cfg.get("smoke_test", False))


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


def _load_splits(dataset: BinaryDatasetName):
    if dataset == "dynahate":
        return create_dynahate_dataset(data_dir()), "text"
    if dataset == "hatecheck":
        return create_hatecheck_dataset(data_dir()), "test_case"
    return create_hateeval_dataset(data_dir()), "text"


def run_training(config_path: Path, dataset: TrainDatasetName, *, seed: int = 123) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_path = config_path.resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training_mode = str(cfg.get("training_mode", "binary")).lower()
    if training_mode == "structured":
        if dataset not in STRUCTURED_DATASETS:
            raise ValueError(
                f"training_mode=structured requires --dataset in {sorted(STRUCTURED_DATASETS)}; "
                f"got {dataset!r}."
            )
        from hatelens.structured_train import run_structured_training

        run_structured_training(config_path, str(dataset), seed=seed)
        return

    if dataset not in BINARY_DATASETS:
        raise ValueError(
            f"training_mode=binary requires --dataset in {sorted(BINARY_DATASETS)}; "
            f"got {dataset!r}. Use training_mode: structured for hatexplain / dynahate_hatexplain."
        )

    model_checkpoint = resolve_model_id(str(cfg["model_checkpoint"]))
    model_name = cfg.get("model_name", "model")
    root = repo_root()
    cfg_fp = _config_fingerprint(cfg)
    rev = _git_head(root)
    logger.info(
        "Run metadata: config_sha256_16=%s git_head=%s model=%s dataset=%s",
        cfg_fp,
        rev or "(unknown)",
        model_checkpoint,
        dataset,
    )

    use_wandb = _maybe_wandb_init(f"hatlens-{model_name}-{dataset}", cfg)
    set_seed(int(cfg.get("seed", seed)))

    quant = str(cfg.get("quantization", "none"))
    if quant not in ("none", "4bit", "8bit"):
        quant = "none"
    model = build_base_sequence_classifier(model_checkpoint, quantization=quant)  # type: ignore[arg-type]
    peft_type = str(cfg.get("peft_type", "lora"))
    peft_config = build_peft_config(
        cfg["task_type"],
        peft_type=peft_type,
        r=int(cfg["r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["target_modules"]),
        use_rslora=bool(cfg.get("use_rslora", False)),
    )
    model = apply_peft(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    raw, text_key = _load_splits(cast(BinaryDatasetName, dataset))

    smoke = _is_smoke(cfg)
    if smoke:
        n_tr = min(int(cfg.get("max_train_samples", 64)), len(raw["train"]))
        n_ev = min(int(cfg.get("max_eval_samples", 32)), len(raw["validation"]))
        raw["train"] = raw["train"].select(range(n_tr))
        raw["validation"] = raw["validation"].select(range(n_ev))
        if "test" in raw:
            n_te = min(int(cfg.get("max_test_samples", 32)), len(raw["test"]))
            raw["test"] = raw["test"].select(range(n_te))
        logger.info(
            "Smoke mode: truncated splits before tokenization "
            "(set HATELENS_SMOKE=0 or smoke_test=false to disable)"
        )

    def tokenize_fn(examples: dict[str, list]):
        return tokenizer(
            examples[text_key],
            padding="max_length",
            truncation=True,
            max_length=int(cfg.get("max_length", 512)),
        )

    tokenized = raw.map(tokenize_fn, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits = np.asarray(eval_pred.predictions)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        labels = np.asarray(eval_pred.label_ids)
        preds = np.argmax(logits, axis=-1)
        probs = _softmax_probs_hate(logits)
        metrics = classification_metrics(labels, preds, probs)
        # HF Trainer logs floats; omit NaN keys for ROC when a single class is present
        return {k: v for k, v in metrics.items() if not (isinstance(v, float) and np.isnan(v))}

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir = out_dir / dataset
    log_dir = Path(cfg["logging_dir"])
    if not log_dir.is_absolute():
        log_dir = repo_root() / log_dir
    log_dir = log_dir / dataset

    use_fp16 = torch.cuda.is_available() and bool(cfg.get("fp16", True))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8 if dataset == "dynahate" else 2))
    train_bs = int(cfg.get("per_device_train_batch_size", 8 if dataset == "dynahate" else 4))
    eval_bs = int(cfg.get("per_device_eval_batch_size", train_bs))
    workers = min(os.cpu_count() or 1, int(cfg.get("dataloader_num_workers", 8)))

    metric_name = cfg.get("metric_for_best_model", "f1")
    num_epochs = float(cfg["num_train_epochs"])
    max_steps_cfg = cfg.get("max_steps")
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else -1
    if smoke:
        num_epochs = float(cfg.get("smoke_num_epochs", 1))
        if max_steps < 0 and cfg.get("smoke_max_steps") is not None:
            max_steps = int(cfg["smoke_max_steps"])

    ta_kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=num_epochs,
        weight_decay=float(cfg["weight_decay"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
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
    if max_steps > 0:
        ta_kwargs["max_steps"] = max_steps

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Starting training dataset=%s config=%s", dataset, config_path)
    train_out = trainer.train()
    best_path = out_dir / "best_checkpoint"
    best_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_path))
    write_config_resolved(config_path, out_dir)
    write_train_metrics_json(out_dir, train_metrics=dict(train_out.metrics))
    logger.info("Saved best model to %s", best_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="HateLens LoRA fine-tuning")
    p.add_argument(
        "config_file",
        type=Path,
        help="YAML config path (default example: configs/models/tinyllama.yaml)",
    )
    p.add_argument(
        "--dataset",
        choices=(
            "dynahate",
            "hatecheck",
            "hateeval",
            "hatexplain",
            "dynahate_hatexplain",
        ),
        required=True,
        help="Dataset key; combine with training_mode in YAML (binary vs structured).",
    )
    args = p.parse_args(argv)
    run_training(args.config_file, args.dataset)


if __name__ == "__main__":
    main()
