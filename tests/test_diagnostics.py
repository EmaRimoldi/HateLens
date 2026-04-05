import numpy as np
import pandas as pd

from hatelens.diagnostics import hatecheck_functionality_report


def test_functionality_report_basic():
    df = pd.DataFrame(
        {
            "functionality": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
        }
    )
    preds = np.array([0, 1, 1, 1])
    rep = hatecheck_functionality_report(df, preds)
    assert len(rep) == 2
    assert set(rep["functionality"]) == {"a", "b"}
