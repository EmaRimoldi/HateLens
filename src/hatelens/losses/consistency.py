"""Pairwise consistency losses for invariance / directional (counterfactual) pairs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def js_divergence_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric Jensen–Shannon divergence between softmax distributions (batch mean)."""
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = (m + 1e-12).log()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()


def kl_divergence_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


def pairwise_ce_directional(
    logits_b: torch.Tensor,
    labels_b: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for the second element of a directed pair (expects batch of paired rows)."""
    return F.cross_entropy(logits_b, labels_b)


def numpy_js_divergence_from_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Utility for evaluation / tests without torch."""
    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    m = 0.5 * (p + q)
    kl_p = np.sum(p * (np.log(p) - np.log(m)))
    kl_q = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_p + kl_q))
