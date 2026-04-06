"""Standardized binary vs macro metrics."""

import numpy as np

from hatelens.evaluation import classification_metrics


def test_classification_metrics_binary_and_macro():
    # Perfect classifier on balanced 4-example set
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.9, 0.8])
    m = classification_metrics(labels, preds, probs)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["f1_binary"] == 1.0
    assert m["f1_macro"] == 1.0


def test_macro_f1_differs_from_binary_when_only_negative_predicted():
    labels = np.array([0, 1])
    preds = np.array([0, 0])  # false negative on hate
    probs = np.array([0.2, 0.3])
    m = classification_metrics(labels, preds, probs)
    assert m["f1_binary"] == 0.0
    assert m["f1_macro"] > m["f1_binary"]  # macro averages both classes; binary F1 is hate-only
