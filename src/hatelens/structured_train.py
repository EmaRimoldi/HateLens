"""Structured multi-task training (invoked from ``train_pipeline`` when ``training_mode: structured``)."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, TrainingArguments

from hatelens.evaluation import classification_metrics
from hatelens.modeling_structured import StructuredHateModel
from hatelens.paths import repo_root
from hatelens.peft_factory import apply_peft, build_base_transformer_backbone, build_peft_config
from hatelens.registry import resolve_model_id
from hatelens.structured_collator import StructuredCollator
from hatelens.structured_data import build_structured_dataset_dict
from hatelens.structured_trainer import StructuredTrainer

logger = logging.getLogger(__name__)


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
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    return p[:, 1]


def _is_smoke(cfg: dict[str, Any]) -> bool:
    if os.environ.get("HATELENS_SMOKE", "").lower() in ("1", "true", "yes"):
        return True
    return bool(cfg.get("smoke_test", False))


def _maybe_wandb_init(project: str, config: dict[str, Any]) -> bool:
    if os.environ.get("WANDB_ENABLED", "").lower() not in ("1", "true", "yes"):
        os.environ["WANDB_MODE"] = "disabled"
        return False
    try:
        import wandb
    except ImportError:
        os.environ["WANDB_MODE"] = "disabled"
        return False
    wandb.init(project=project, config=config, entity=os.environ.get("WANDB_ENTITY"))
    return True


def run_structured_training(
    config_path: Path,
    dataset_key: str,
    *,
    seed: int = 123,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_path = config_path.resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    root = repo_root()
    model_checkpoint = resolve_model_id(str(cfg["model_checkpoint"]))
    model_name = cfg.get("model_name", "model")
    cfg_fp = _config_fingerprint(cfg)
    rev = _git_head(root)
    training_mode = str(cfg.get("training_mode", "structured"))
    logger.info(
        "Structured run: config_sha256_16=%s git_head=%s model=%s dataset=%s mode=%s",
        cfg_fp,
        rev or "(unknown)",
        model_checkpoint,
        dataset_key,
        training_mode,
    )
    logger.info("Resolved config keys: training_mode=%s use_rationale=%s use_consistency=%s", training_mode, cfg.get("use_rationale", True), cfg.get("use_consistency", False))

    use_wandb = _maybe_wandb_init(f"hatlens-structured-{model_name}-{dataset_key}", cfg)
    seed_i = int(cfg.get("seed", seed))
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(cfg.get("max_length", 512))
    use_rationale = bool(cfg.get("use_rationale", True))
    tokenized, vocabs = build_structured_dataset_dict(
        dataset_key,
        tokenizer,
        max_length=max_length,
        use_rationale=use_rationale,
    )

    smoke = _is_smoke(cfg)
    if smoke:
        n_tr = min(int(cfg.get("max_train_samples", 64)), len(tokenized["train"]))
        n_ev = min(int(cfg.get("max_eval_samples", 32)), len(tokenized["validation"]))
        tokenized["train"] = tokenized["train"].select(range(n_tr))
        tokenized["validation"] = tokenized["validation"].select(range(n_ev))
        if "test" in tokenized:
            n_te = min(int(cfg.get("max_test_samples", 32)), len(tokenized["test"]))
            tokenized["test"] = tokenized["test"].select(range(n_te))
        logger.info("Smoke mode: truncated structured splits")

    quant = str(cfg.get("quantization", "none"))
    if quant not in ("none", "4bit", "8bit"):
        quant = "none"
    backbone = build_base_transformer_backbone(model_checkpoint, quantization=quant)  # type: ignore[arg-type]
    task_type = str(cfg.get("task_type", "FEATURE_EXTRACTION"))
    peft_type = str(cfg.get("peft_type", "lora"))
    peft_config = build_peft_config(
        task_type,
        peft_type=peft_type,
        r=int(cfg["r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["target_modules"]),
        use_rslora=bool(cfg.get("use_rslora", False)),
    )
    backbone = apply_peft(backbone, peft_config)
    hidden = int(backbone.config.hidden_size)
    model = StructuredHateModel(
        backbone,
        hidden_size=hidden,
        n_main=2,
        n_target_group=vocabs.target_group.num_labels(),
        n_hate_type=vocabs.hate_type.num_labels(),
        n_explicitness=vocabs.explicitness.num_labels(),
        n_rationale_token=2,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: trainable=%s total=%s", f"{n_trainable:,}", f"{n_total:,}")

    collator = StructuredCollator(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits = np.asarray(eval_pred.predictions)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        labels = np.asarray(eval_pred.label_ids)
        preds = np.argmax(logits, axis=-1)
        probs = _softmax_probs_hate(logits)
        metrics = classification_metrics(labels, preds, probs)
        return {k: v for k, v in metrics.items() if not (isinstance(v, float) and np.isnan(v))}

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir / f"structured_{dataset_key}"
    log_dir = Path(cfg["logging_dir"])
    if not log_dir.is_absolute():
        log_dir = root / log_dir
    log_dir = log_dir / f"structured_{dataset_key}"

    use_fp16 = torch.cuda.is_available() and bool(cfg.get("fp16", True))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    train_bs = int(cfg.get("per_device_train_batch_size", 4))
    eval_bs = int(cfg.get("per_device_eval_batch_size", train_bs))
    workers = min(os.cpu_count() or 1, int(cfg.get("dataloader_num_workers", 4)))
    metric_name = cfg.get("metric_for_best_model", "f1_macro")
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
        remove_unused_columns=False,
    )
    if max_steps > 0:
        ta_kwargs["max_steps"] = max_steps

    args = TrainingArguments(**ta_kwargs)

    trainer = StructuredTrainer(
        lambda_aux=float(cfg.get("aux_loss_weight", cfg.get("lambda_aux", 0.5))),
        lambda_rat=float(cfg.get("rationale_loss_weight", cfg.get("lambda_rat", 0.3))),
        lambda_cons=float(cfg.get("consistency_loss_weight", cfg.get("lambda_cons", 0.2))),
        use_rationale=bool(cfg.get("use_rationale", True)),
        use_consistency=bool(cfg.get("use_consistency", False)),
        vocabs=vocabs,
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Starting structured training dataset=%s config=%s", dataset_key, config_path)
    trainer.train()

    best_path = out_dir / "best_checkpoint"
    best_path.mkdir(parents=True, exist_ok=True)
    vocabs.save_dir(best_path / "vocab")
    torch.save(model.state_dict(), best_path / "structured_model.pt")
    if hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(best_path / "peft_adapter")
    model.save_heads(best_path / "structured_heads.pt")
    logger.info("Saved structured checkpoint to %s", best_path)
