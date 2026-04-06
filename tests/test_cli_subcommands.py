"""CLI wiring smoke tests (no training)."""

import json
from pathlib import Path

from hatelens.cli import build_parser


def test_eval_run_subcommand_exists():
    p = build_parser()
    args = p.parse_args(["eval-run"])
    assert args.command == "eval-run"


def test_export_tables_subcommand_exists(tmp_path: Path):
    mj = tmp_path / "m.json"
    payload = {"in_domain": {}, "efficiency": {"total_parameters": 1}}
    mj.write_text(json.dumps(payload), encoding="utf-8")
    p = build_parser()
    args = p.parse_args(["export-tables", str(mj), "--kind", "efficiency"])
    assert args.command == "export-tables"
