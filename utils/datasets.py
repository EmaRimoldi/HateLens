# Compatibility shim: legacy ``from utils.datasets import ...`` when running from repo root.
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from hatelens.datasets import (  # noqa: E402
    create_dynahate_dataset,
    create_gab_dataset,
    create_hatecheck_dataset,
    create_hatecheck_dataset_with_metadata,
    describe_dataset,
    download_dynahate,
)

__all__ = [
    "create_dynahate_dataset",
    "create_hatecheck_dataset",
    "create_hatecheck_dataset_with_metadata",
    "create_gab_dataset",
    "describe_dataset",
    "download_dynahate",
]
