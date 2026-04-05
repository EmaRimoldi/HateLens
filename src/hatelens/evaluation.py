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
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
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
