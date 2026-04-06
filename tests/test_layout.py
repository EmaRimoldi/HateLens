"""Repository layout and config presence."""

from hatelens.paths import data_dir, outputs_dir, repo_root


def test_default_configs_exist():
    root = repo_root()
    assert (root / "configs/models/tinyllama.yaml").is_file()
    assert (root / "configs/models/tinyllama-legacy.yaml").is_file()
    assert (root / "configs/models/phi-2.yaml").is_file()
    assert (root / "configs/models/opt-1.3b.yaml").is_file()
    assert (root / "configs/models/qwen2.5-1.5b.yaml").is_file()
    assert (root / "configs/models/tinyllama-structured.yaml").is_file()
    assert (root / "configs/eval/minimal.yaml").is_file()
    assert (root / "configs/experiments/paper_matrix/tinyllama_binary_compare.yaml").is_file()
    assert (root / "configs/smoke/tinyllama_dynahate.yaml").is_file()


def test_outputs_dir_is_under_repo():
    assert outputs_dir().is_relative_to(repo_root())


def test_data_dynahate_csv():
    p = data_dir() / "DynaHate/dynahate_v0.2.3.csv"
    assert p.is_file()
