import numpy as np

from hatelens.evaluation_calibration import brier_score_binary, expected_calibration_error


def test_ece_finite_and_reasonable():
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
    ece = expected_calibration_error(labels, probs, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_brier():
    y = np.array([0.0, 1.0])
    p = np.array([0.0, 1.0])
    assert abs(brier_score_binary(y, p)) < 1e-9
