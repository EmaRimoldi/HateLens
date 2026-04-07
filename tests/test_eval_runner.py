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


def test_parse_eval_config_paper_followup():
    p = repo_root() / "configs/eval/paper_followup.yaml"
    cfg = parse_eval_config(p)
    assert any("dynahate_hatexplain" in str(x.get("train", "")) for x in cfg.cross_dataset)
    assert cfg.hatecheck is True


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


def test_export_tables_hatecheck_extracts_overall(tmp_path: Path):
    mj = tmp_path / "m.json"
    mj.write_text(
        json.dumps(
            {
                "hatecheck": {
                    "overall": {"accuracy": 0.5, "f1_macro": 0.4},
                    "per_functionality_csv": [{"functionality": "slur", "accuracy": 0.3}],
                }
            }
        ),
        encoding="utf-8",
    )
    df = build_comparison_table([mj], kind="hatecheck")
    assert len(df) == 1
    assert float(df.iloc[0]["accuracy"]) == 0.5


def test_export_tables_cross_dataset_extracts(tmp_path: Path):
    mj = tmp_path / "m.json"
    mj.write_text(
        json.dumps(
            {
                "cross_dataset": {
                    "train_a_test_b": {"accuracy": 0.9, "f1": 0.88},
                }
            }
        ),
        encoding="utf-8",
    )
    df = build_comparison_table([mj], kind="cross")
    assert len(df) == 1
    assert df.iloc[0]["split"] == "train_a_test_b"
    assert float(df.iloc[0]["f1"]) == 0.88


def test_export_tables_efficiency_extracts(tmp_path: Path):
    mj = tmp_path / "m.json"
    mj.write_text(
        json.dumps(
            {"efficiency": {"total_parameters": 1000, "trainable_parameters": 10}},
        ),
        encoding="utf-8",
    )
    df = build_comparison_table([mj], kind="efficiency")
    assert int(df.iloc[0]["total_parameters"]) == 1000


def test_efficiency_report_cpu_smoke(tmp_path: Path):
    import torch
    import torch.nn as nn

    from hatelens.eval_runner import efficiency_report

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, attention_mask=None, **_kwargs):
            return None

    model = Tiny()

    class Enc(dict):
        """BatchEncoding-like: ``.to(device)`` returns a dict for ``model(**enc)``."""

        def to(self, device):
            return Enc({k: v.to(device) for k, v in self.items()})

    class Tok:
        def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
            n = min(max_length, 8)
            return Enc(
                {
                    "input_ids": torch.ones(len(texts), n, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), n, dtype=torch.long),
                }
            )

    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "stub.bin").write_bytes(b"stub")

    rep = efficiency_report(
        model,
        ck,
        device=torch.device("cpu"),
        tokenizer=Tok(),
        train_runtime_seconds=123.0,
    )
    rep2 = efficiency_report(
        model,
        ck,
        device=torch.device("cpu"),
        tokenizer=Tok(),
        train_runtime_seconds=None,
    )
    assert rep["train_runtime_seconds"] == 123.0
    assert rep["total_parameters"] >= 1
    assert "inference_latency_ms_batch1" in rep
    assert rep2["checkpoint_size_bytes"] >= 0
