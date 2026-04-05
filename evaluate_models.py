#!/usr/bin/env python3
"""Backward-compatible entrypoint. Prefer: ``python -m hatlens evaluate --dynahate|--hatecheck``."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hatlens.cli import main as cli_main


def main() -> None:
    argv = sys.argv[1:]
    if "--dynahate" not in argv and "--hatecheck" not in argv:
        if "--help" not in argv and "-h" not in argv:
            sys.stderr.write(
                "Usage: evaluate_models.py --dynahate | --hatecheck [--plots] [--batch-size N]\n"
            )
            sys.exit(2)
    sys.argv = ["hatelens", "evaluate", *argv]
    cli_main()


if __name__ == "__main__":
    main()
