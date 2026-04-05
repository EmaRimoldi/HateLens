"""Load sequence classifiers from HF hub or local dirs (full weights or PEFT LoRA adapters)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hatelens.paths import repo_root

logger = logging.getLogger(__name__)

CLASS_NAMES = ("nothate", "hate")
ID2LABEL = {0: "nothate", 1: "hate"}
LABEL2ID = {"nothate": 0, "hate": 1}


def _is_local_adapter(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def load_sequence_classifier(
    checkpoint: str | Path,
    *,
    device: torch.device | None = None,
    num_labels: int = 2,
    local_files_only: bool | None = None,
    merge_adapters: bool = True,
) -> tuple[Any, Any]:
    """
    Load tokenizer + model. If ``checkpoint`` is a PEFT adapter directory, load base weights
    from ``base_model_name_or_path`` in adapter_config.json and attach (optionally merge) adapters.
    """
    ckpt = Path(checkpoint).resolve()
    if local_files_only is None:
        local_files_only = ckpt.is_dir()

    if _is_local_adapter(ckpt):
        cfg_path = ckpt / "adapter_config.json"
        with open(cfg_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg["base_model_name_or_path"]
        logger.info("Loading LoRA adapter from %s (base=%s)", ckpt, base_id)
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            base_id,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, str(ckpt))
        if merge_adapters:
            model = model.merge_and_unload()
        model.eval()
    else:
        tid = str(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(tid, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            tid,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=local_files_only,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def default_checkpoints() -> dict[str, dict[str, str]]:
    """Built-in paths matching the original repository layout (override with env if needed)."""
    root = repo_root()
    base = os.environ.get("HATELENS_BASE_MODEL", "PY007/TinyLlama-1.1B-step-50K-105b")
    return {
        "base": {"id": base},
        "post": {
            "dynahate": os.environ.get(
                "HATELENS_CKPT_DYNAHATE",
                str(root / "checkpoints/TinyLlama/dynahate/best_checkpoint_42"),
            ),
            "hatecheck": os.environ.get(
                "HATELENS_CKPT_HATECHECK",
                str(root / "checkpoints/TinyLlama/hatecheck/best_checkpoint_33"),
            ),
        },
    }
