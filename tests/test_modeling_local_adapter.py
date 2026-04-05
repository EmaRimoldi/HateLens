"""Smoke test: PEFT adapter layout under checkpoints/ (no hub download)."""

from hatelens.paths import repo_root


def test_bundled_adapter_paths_exist():
    root = repo_root()
    for rel in (
        "checkpoints/TinyLlama/dynahate/best_checkpoint_42",
        "checkpoints/TinyLlama/hatecheck/best_checkpoint_33",
    ):
        p = root / rel
        if p.is_dir():
            assert (p / "adapter_config.json").exists()
            assert (p / "adapter_model.safetensors").exists()
