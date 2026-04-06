"""Repository root resolution (works for editable installs and cwd-based runs)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the HateLens repository root directory."""
    env = os.environ.get("HATELENS_ROOT")
    if env:
        return Path(env).resolve()
    # src/hatelens/paths.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def outputs_dir() -> Path:
    """Root for all generated artifacts (eval, LIME pickles, training runs)."""
    return repo_root() / "outputs"


def eval_runs_dir() -> Path:
    """Unified evaluation runner output root (``metrics.json``, ``predictions.jsonl``, …)."""
    return outputs_dir() / "eval_runs"


def runs_dir() -> Path:
    """Training runs root (``outputs/runs``)."""
    return outputs_dir() / "runs"
