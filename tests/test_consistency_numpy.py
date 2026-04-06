import numpy as np

from hatelens.losses.consistency import numpy_js_divergence_from_probs


def test_js_divergence_identical():
    p = np.array([0.5, 0.5])
    assert numpy_js_divergence_from_probs(p, p) < 1e-9
