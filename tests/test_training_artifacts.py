"""Training artifact writers (no training)."""

import json
from pathlib import Path

from hatelens.training_artifacts import (
    eval_summary_from_trainer_state,
    write_eval_summary_json,
)


def test_eval_summary_from_trainer_state_filters_eval_keys(tmp_path: Path):
    run = tmp_path / "run"
    (run / "best_checkpoint").mkdir(parents=True)
    s = eval_summary_from_trainer_state(
        run,
        trainer_metrics={
            "train_loss": 0.1,
            "eval_f1": 0.8,
            "eval_loss": 0.2,
            "epoch": 1.0,
        },
        training_mode="binary",
    )
    assert s["training_mode"] == "binary"
    assert "eval_f1" in s["trainer_validation_metrics"]
    assert "train_loss" not in s["trainer_validation_metrics"]
    assert "best_checkpoint" in s["note"].lower() or "eval-run" in s["note"]


def test_write_eval_summary_json_roundtrip(tmp_path: Path):
    p = write_eval_summary_json(tmp_path, {"a": 1})
    assert p.name == "eval_summary.json"
    assert json.loads(p.read_text(encoding="utf-8"))["a"] == 1
