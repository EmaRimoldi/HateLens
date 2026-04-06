"""Save resolved config and training metrics next to checkpoints for reproducibility."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def write_config_resolved(config_path: Path, dest_dir: Path) -> Path:
    """Copy the training YAML used for this run as ``config_resolved.yaml``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "config_resolved.yaml"
    shutil.copy2(config_path.resolve(), out)
    return out


def write_train_metrics_json(
    dest_dir: Path,
    *,
    train_metrics: dict[str, Any] | None = None,
    eval_summary: dict[str, Any] | None = None,
) -> Path:
    """Write ``train_metrics.json`` (runtime + optional eval summary from Trainer)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if train_metrics:
        payload["train_metrics"] = train_metrics
    if eval_summary:
        payload["eval_summary"] = eval_summary
    out = dest_dir / "train_metrics.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def write_eval_summary_json(dest_dir: Path, summary: dict[str, Any]) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out


def dump_resolved_config_dict(cfg: dict[str, Any], dest_dir: Path) -> Path:
    """Write a resolved config dict (already merged) as YAML."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "config_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
