"""Pad structured batches (rationale token labels, pair ids)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StructuredCollator:
    """Pad ``input_ids`` / ``attention_mask`` and stack auxiliary fields."""

    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        rat_rows: list[list[int]] = []
        for f in features:
            rl = list(f["rationale_token_labels"])
            if len(rl) < max_len:
                rl = rl + [-100] * (max_len - len(rl))
            else:
                rl = rl[:max_len]
            rat_rows.append(rl)
        batch["rationale_token_labels"] = torch.tensor(rat_rows, dtype=torch.long)
        main_t = torch.tensor([int(f["main_labels"]) for f in features], dtype=torch.long)
        batch["main_labels"] = main_t
        batch["labels"] = main_t
        batch["target_group_labels"] = torch.tensor(
            [int(f["target_group_labels"]) for f in features], dtype=torch.long
        )
        batch["hate_type_labels"] = torch.tensor([int(f["hate_type_labels"]) for f in features], dtype=torch.long)
        batch["explicitness_labels"] = torch.tensor(
            [int(f["explicitness_labels"]) for f in features], dtype=torch.long
        )
        batch["pair_relation_ids"] = torch.tensor(
            [int(f["pair_relation_ids"]) for f in features], dtype=torch.long
        )
        batch["pair_ids"] = [str(f.get("pair_ids") or "") for f in features]
        batch["has_rationale"] = torch.tensor([bool(f.get("has_rationale")) for f in features], dtype=torch.bool)
        return batch
