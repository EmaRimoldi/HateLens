#!/usr/bin/env python3
"""Legacy wrapper: ``python -m hatelens lime ...``."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hatelens.cli import main as cli_main


def main() -> None:
    argv = sys.argv[1:]
    if "--dynahate" not in argv and "--hatecheck" not in argv:
        if "--help" not in argv and "-h" not in argv:
            sys.stderr.write("Usage: compute_lime_scores.py --dynahate | --hatecheck\n")
            sys.exit(2)
    sys.argv = ["hatelens", "lime", *argv]
    cli_main()


if __name__ == "__main__":
    main()
