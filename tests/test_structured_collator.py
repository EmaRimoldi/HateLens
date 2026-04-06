"""Structured collator stacks labels and pads rationale rows."""

from unittest.mock import MagicMock

import torch

from hatelens.structured_collator import StructuredCollator


def test_structured_collator_aligns_labels_and_rationale_pad():
    tok = MagicMock()
    tok.pad = MagicMock(
        side_effect=lambda feats, **kw: {
            "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        }
    )
    c = StructuredCollator(tokenizer=tok)
    feats = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "main_labels": 1,
            "labels": 1,
            "target_group_labels": 0,
            "hate_type_labels": 1,
            "explicitness_labels": 2,
            "pair_relation_ids": 0,
            "pair_ids": "p1",
            "rationale_token_labels": [-100, -100],
            "has_rationale": False,
        },
        {
            "input_ids": [3],
            "attention_mask": [1],
            "main_labels": 0,
            "labels": 0,
            "target_group_labels": 1,
            "hate_type_labels": 0,
            "explicitness_labels": 0,
            "pair_relation_ids": 0,
            "pair_ids": "",
            "rationale_token_labels": [1, 0],
            "has_rationale": True,
        },
    ]
    b = c(feats)
    assert torch.equal(b["main_labels"], b["labels"])
    assert b["rationale_token_labels"].shape == (2, 3)
    assert b["rationale_token_labels"][1, 0].item() == 1
    assert b["pair_ids"] == ["p1", ""]
