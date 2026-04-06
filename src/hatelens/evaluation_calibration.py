"""Calibration metrics: ECE, Brier score (binary hate probability vs label)."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Standard ECE for binary labels in {0,1} and ``probs`` = P(hate)."""
    labels = np.asarray(labels).astype(np.float64)
    probs = np.clip(np.asarray(probs).astype(np.float64), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        w = np.sum(mask) / n
        ece += w * abs(acc - conf)
    return float(ece)


def brier_score_binary(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.float64)
    probs = np.asarray(probs).astype(np.float64)
    return float(np.mean((probs - labels) ** 2))
