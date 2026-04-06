"""Multi-head structured hate-speech model sharing one PEFT-tuned backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


@dataclass
class StructuredOutputs:
    """``logits`` duplicates ``logits_main`` so HuggingFace ``Trainer`` eval can read ``outputs.logits``."""

    logits: torch.Tensor
    logits_main: torch.Tensor
    logits_target_group: torch.Tensor
    logits_hate_type: torch.Tensor
    logits_explicitness: torch.Tensor
    logits_rationale: torch.Tensor  # (B, T, 2) token-level binary


def pool_last_non_pad(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last real token (right-padded)."""
    seq_lens = attention_mask.sum(dim=1).clamp(min=1) - 1
    b = torch.arange(hidden.size(0), device=hidden.device, dtype=torch.long)
    return hidden[b, seq_lens]


class StructuredHateModel(nn.Module):
    """
    Backbone (e.g. LlamaModel + PEFT) + classification heads + token-level rationale head.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        *,
        hidden_size: int,
        n_main: int,
        n_target_group: int,
        n_hate_type: int,
        n_explicitness: int,
        n_rationale_token: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.main_head = nn.Linear(hidden_size, n_main)
        self.target_group_head = nn.Linear(hidden_size, n_target_group)
        self.hate_type_head = nn.Linear(hidden_size, n_hate_type)
        self.explicitness_head = nn.Linear(hidden_size, n_explicitness)
        self.rationale_head = nn.Linear(hidden_size, n_rationale_token)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> StructuredOutputs:
        # Drop keys passed by Trainer that the backbone does not accept
        fwd_kw = {k: v for k, v in kwargs.items() if k in ("position_ids", "inputs_embeds")}
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            **fwd_kw,
        )
        h = out.last_hidden_state
        if attention_mask is None:
            raise ValueError("attention_mask required")
        pooled = pool_last_non_pad(h, attention_mask)
        logits_r = self.rationale_head(h)
        lm = self.main_head(pooled)
        return StructuredOutputs(
            logits=lm,
            logits_main=lm,
            logits_target_group=self.target_group_head(pooled),
            logits_hate_type=self.hate_type_head(pooled),
            logits_explicitness=self.explicitness_head(pooled),
            logits_rationale=logits_r,
        )

    def save_heads(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict_heads(), path)

    def load_heads(self, path: Path, *, map_location: str | torch.device = "cpu") -> None:
        try:
            sd = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd, strict=False)

    def state_dict_heads(self) -> dict[str, torch.Tensor]:
        keys = [
            "main_head",
            "target_group_head",
            "hate_type_head",
            "explicitness_head",
            "rationale_head",
        ]
        out: dict[str, torch.Tensor] = {}
        sd = self.state_dict()
        for k, v in sd.items():
            if any(k.startswith(p) for p in keys):
                out[k] = v
        return out


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean CE over elements where targets != ignore_index."""
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")
    mask = targets.ne(ignore_index)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return loss.masked_select(mask).mean()


def rationale_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """logits: (B,T,C), labels: (B,T)"""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def js_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch JS divergence for soft probabilities p,q shape (B, C)."""
    p = p.clamp(min=eps, max=1 - eps)
    q = q.clamp(min=eps, max=1 - eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()
