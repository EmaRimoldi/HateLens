"""Aggregate in-domain, calibration, and optional cross-dataset metrics into JSON-serializable dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hatelens.evaluation import classification_metrics
from hatelens.evaluation_calibration import brier_score_binary, expected_calibration_error


def run_binary_eval_bundle(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> dict[str, Any]:
    """Single-dataset evaluation block (classification + calibration)."""
    cls = classification_metrics(labels, preds, probs)
    cls["ece"] = expected_calibration_error(labels, probs)
    cls["brier"] = brier_score_binary(labels, probs)
    return cls


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
