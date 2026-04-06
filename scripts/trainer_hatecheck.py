#!/usr/bin/env python3
"""Legacy: forwards to ``hatelens train`` (hatecheck)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hatelens.train_pipeline import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: trainer_hatecheck.py <config.yaml>")
        sys.exit(1)
    sys.argv = [sys.argv[0], sys.argv[1], "--dataset", "hatecheck"]
    main()
