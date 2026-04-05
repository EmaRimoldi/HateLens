#!/usr/bin/env python3
"""Backward-compatible entrypoint. Prefer: ``hatelens train <config> --dataset dynahate``."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hatlens.train_pipeline import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trainer_dynahate.py <config.yaml>")
        sys.exit(1)
    sys.argv = [sys.argv[0], sys.argv[1], "--dataset", "dynahate"]
    main()
