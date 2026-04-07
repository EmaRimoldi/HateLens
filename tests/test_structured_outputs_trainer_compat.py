"""StructuredOutputs tuple-like API for Hugging Face Trainer.prediction_step."""

import torch

from hatelens.modeling_structured import StructuredOutputs


def _dummy(b: int = 2, t: int = 4) -> StructuredOutputs:
    return StructuredOutputs(
        logits=torch.zeros(b, 2),
        logits_main=torch.zeros(b, 2),
        logits_target_group=torch.zeros(b, 3),
        logits_hate_type=torch.zeros(b, 3),
        logits_explicitness=torch.zeros(b, 3),
        logits_rationale=torch.zeros(b, t, 2),
    )


def test_structured_outputs_len_and_index():
    o = _dummy()
    assert len(o) == 1
    assert o[0].shape == (2, 2)


def test_structured_outputs_slice_one_suffix():
    o = _dummy()
    tail = o[1:]
    assert isinstance(tail, tuple) and len(tail) == 1
    assert tail[0].shape == (2, 2)
