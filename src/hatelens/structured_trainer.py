"""Custom Trainer: multi-task + optional rationale + pair consistency."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer

from hatelens.labels import IGNORE_INDEX, StructuredVocabBundle
from hatelens.modeling_structured import (
    StructuredHateModel,
    js_divergence_probs,
    masked_ce,
    rationale_token_loss,
)

logger = logging.getLogger(__name__)


class StructuredTrainer(Trainer):
    def __init__(
        self,
        *,
        lambda_aux: float = 0.5,
        lambda_rat: float = 0.3,
        lambda_cons: float = 0.2,
        use_rationale: bool = True,
        use_consistency: bool = False,
        vocabs: StructuredVocabBundle,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lambda_aux = lambda_aux
        self.lambda_rat = lambda_rat
        self.lambda_cons = lambda_cons
        self.use_rationale = use_rationale
        self.use_consistency = use_consistency
        self.vocabs = vocabs
        st = vocabs.pair_relation.stoi
        self._inv_idx = st.get("invariant", -1)
        self._flip_h_idx = st.get("flip_to_hate", -1)
        self._flip_nh_idx = st.get("flip_to_non_hate", -1)
        self._consistency_log_counter = 0

    def compute_loss(
        self,
        model: StructuredHateModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        inputs.pop("labels", None)
        pair_ids = inputs.pop("pair_ids")
        has_rationale = inputs.pop("has_rationale")
        main_labels = inputs.pop("main_labels")
        target_group_labels = inputs.pop("target_group_labels")
        hate_type_labels = inputs.pop("hate_type_labels")
        explicitness_labels = inputs.pop("explicitness_labels")
        pair_relation_ids = inputs.pop("pair_relation_ids")
        rationale_token_labels = inputs.pop("rationale_token_labels")

        fwd = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        outputs = model(**fwd)

        loss_main = F.cross_entropy(outputs.logits_main, main_labels)
        loss_tg = masked_ce(outputs.logits_target_group, target_group_labels, ignore_index=IGNORE_INDEX)
        loss_ht = masked_ce(outputs.logits_hate_type, hate_type_labels, ignore_index=IGNORE_INDEX)
        loss_ex = masked_ce(outputs.logits_explicitness, explicitness_labels, ignore_index=IGNORE_INDEX)
        loss_aux = (loss_tg + loss_ht + loss_ex) / 3.0

        loss_rat = torch.tensor(0.0, device=loss_main.device)
        if self.use_rationale and has_rationale.any():
            sel = has_rationale
            if sel.any():
                loss_rat = rationale_token_loss(
                    outputs.logits_rationale[sel],
                    rationale_token_labels[sel],
                    ignore_index=IGNORE_INDEX,
                )

        loss_cons = torch.tensor(0.0, device=loss_main.device)
        n_inv = 0
        n_dir = 0
        if self.use_consistency and pair_ids is not None:
            probs = F.softmax(outputs.logits_main, dim=-1)
            by_pair: dict[str, list[int]] = {}
            for i, pid in enumerate(pair_ids):
                if pid:
                    by_pair.setdefault(str(pid), []).append(i)
            for pid, idxs in by_pair.items():
                if len(idxs) < 2:
                    continue
                i0, i1 = idxs[0], idxs[1]
                rel = int(pair_relation_ids[i0].item())
                if rel == self._inv_idx and self._inv_idx >= 0:
                    p0 = probs[i0 : i0 + 1]
                    p1 = probs[i1 : i1 + 1]
                    loss_cons = loss_cons + js_divergence_probs(p0, p1)
                    n_inv += 1
                elif rel == self._flip_h_idx and self._flip_h_idx >= 0:
                    loss_cons = loss_cons + F.cross_entropy(
                        outputs.logits_main[i1 : i1 + 1],
                        torch.ones(1, dtype=torch.long, device=loss_main.device),
                    )
                    n_dir += 1
                elif rel == self._flip_nh_idx and self._flip_nh_idx >= 0:
                    loss_cons = loss_cons + F.cross_entropy(
                        outputs.logits_main[i1 : i1 + 1],
                        torch.zeros(1, dtype=torch.long, device=loss_main.device),
                    )
                    n_dir += 1
            if n_inv + n_dir > 0:
                loss_cons = loss_cons / float(n_inv + n_dir)

        loss = (
            loss_main
            + self.lambda_aux * loss_aux
            + (self.lambda_rat * loss_rat if self.use_rationale else 0.0)
            + (self.lambda_cons * loss_cons if self.use_consistency else 0.0)
        )

        self._consistency_log_counter += 1
        if self.state.global_step and self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "loss_main": loss_main.detach().item(),
                    "loss_aux": loss_aux.detach().item(),
                    "loss_rat": loss_rat.detach().item(),
                    "loss_cons": loss_cons.detach().item(),
                    "consistency_invariant_pairs": float(n_inv),
                    "consistency_directional_pairs": float(n_dir),
                }
            )

        if return_outputs:
            return loss, outputs
        return loss
