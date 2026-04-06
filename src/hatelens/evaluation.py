"""Batched inference and metric computation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Binary hate detection: label 0 = not hate, 1 = hate.
# Unless noted otherwise, precision/recall/f1 are **hate-class (positive) binary** metrics
# (sklearn ``average="binary"``), which matches common hate-speech reporting.


@torch.inference_mode()
def predict_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    *,
    max_length: int = 512,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_labels [N], proba_hate [N])."""
    preds: list[int] = []
    probs: list[float] = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Batched inference", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        pr = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        probs.extend(pr)
        preds.extend(pred)
    return np.array(preds, dtype=np.int64), np.array(probs, dtype=np.float64)


def classification_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> dict[str, float]:
    """
    Standardized binary metrics for hate vs not-hate.

    - ``precision``, ``recall``, ``f1``: hate class (positive), ``average="binary"``.
    - ``f1_macro``, ``precision_macro``, ``recall_macro``: unweighted class average.
    - ``roc_auc`` / ``pr_auc``: ranking metrics when probabilities are valid.
    """
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "f1_binary": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["roc_auc"] = float("nan")
    pr_p, pr_r, _ = precision_recall_curve(labels, probs)
    out["pr_auc"] = float(auc(pr_r, pr_p))
    return out


def log_metrics(prefix: str, m: dict[str, float]) -> None:
    for k, v in m.items():
        if np.isnan(v):
            logger.info("%s %s: nan", prefix, k)
        else:
            logger.info("%s %s: %.4f", prefix, k, v)
