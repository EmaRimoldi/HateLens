"""Tests for unified evaluation runner helpers (offline, no GPU required)."""

import json
from pathlib import Path

from hatelens.eval_runner import (
    EvalRunConfig,
    _cross_items_from_cli,
    checkpoint_size_bytes,
    parse_eval_config,
)
from hatelens.evaluation_suite import run_binary_eval_bundle
from hatelens.metrics_tables import build_comparison_table, flatten_in_domain_rows
from hatelens.paths import repo_root


def test_cross_items_from_cli_parses_pairs():
    out = _cross_items_from_cli(["dynahate:hateeval", "a:b"])
    assert out == [{"train": "dynahate", "test": "hateeval"}, {"train": "a", "test": "b"}]


def test_cross_items_empty():
    assert _cross_items_from_cli(None) == []
    assert _cross_items_from_cli([]) == []


def test_parse_eval_config_minimal():
    p = repo_root() / "configs/eval/minimal.yaml"
    cfg = parse_eval_config(p)
    assert cfg.run_name
    assert cfg.checkpoint.name == "best_checkpoint"
    assert "dynahate" in cfg.in_domain


def test_checkpoint_size_bytes_tmp(tmp_path: Path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "a.bin").write_bytes(b"abc")
    assert checkpoint_size_bytes(d) == 3


def test_flatten_in_domain_rows():
    payload = {
        "in_domain": {
            "dynahate": {"accuracy": 0.5, "f1": 0.4},
            "bad": {"error": "x"},
        }
    }
    rows = flatten_in_domain_rows(payload, source="run1")
    assert len(rows) == 1
    assert rows[0]["dataset"] == "dynahate"


def test_build_comparison_table_empty(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"in_domain": {}}), encoding="utf-8")
    df = build_comparison_table([p], kind="binary_vs_structured")
    assert len(df) == 0


def test_run_binary_eval_bundle_keys():
    y = __import__("numpy").array([0, 1, 1])
    pred = __import__("numpy").array([0, 1, 0])
    prob = __import__("numpy").array([0.1, 0.9, 0.4])
    b = run_binary_eval_bundle(y, pred, prob)
    assert "ece" in b and "brier" in b and "f1_macro" in b


def test_eval_run_config_defaults():
    c = EvalRunConfig(checkpoint=Path("/tmp/x"), run_name="t")
    assert c.split == "test"
