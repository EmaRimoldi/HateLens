"""Ensure binary and structured training entrypoints import cleanly together."""

from hatelens.structured_train import run_structured_training
from hatelens.train_pipeline import run_training


def test_training_entrypoints_are_callable():
    assert callable(run_training)
    assert callable(run_structured_training)
