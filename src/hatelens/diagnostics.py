"""HateCheck-specific diagnostic metrics (per-functionality breakdown)."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


def hatecheck_functionality_report(
    df: pd.DataFrame,
    preds: np.ndarray,
    *,
    label_col: str = "label",
    functionality_col: str = "functionality",
) -> pd.DataFrame:
    """
    Aggregate accuracy / F1 by HateCheck ``functionality`` (functional test type).

    Expects rows aligned with ``preds`` (same order as evaluation dataframe).
    """
    if functionality_col not in df.columns:
        raise ValueError(f"Column {functionality_col!r} not in dataframe")
    y = df[label_col].astype(int).to_numpy()
    rows = []
    by_func: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, func in enumerate(df[functionality_col].astype(str)):
        by_func[func].append((int(y[i]), int(preds[i])))

    for func, pairs in sorted(by_func.items(), key=lambda x: -len(x[1])):
        ys = np.array([p[0] for p in pairs])
        ps = np.array([p[1] for p in pairs])
        rows.append(
            {
                "functionality": func,
                "n": len(pairs),
                "accuracy": accuracy_score(ys, ps),
                "f1": f1_score(ys, ps, zero_division=0),
            }
        )
    out = pd.DataFrame(rows)
    logger.info("HateCheck diagnostic: %d functionality buckets", len(out))
    return out
