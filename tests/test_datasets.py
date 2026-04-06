from hatelens.datasets import (
    create_dynahate_dataset,
    create_hatecheck_dataset,
    create_hatecheck_dataset_with_metadata,
)
from hatelens.paths import data_dir, repo_root


def test_repo_root_points_at_hatelens_repo():
    assert (repo_root() / "pyproject.toml").exists()


def test_dynahate_splits():
    ds = create_dynahate_dataset(data_dir())
    assert set(ds.keys()) == {"train", "validation", "test"}
    assert "text" in ds["train"].column_names
    assert "label" in ds["train"].column_names
    assert "target" in ds["train"].column_names
    assert "type" in ds["train"].column_names
    assert len(ds["test"]) > 0


def test_hatecheck_splits():
    ds = create_hatecheck_dataset(data_dir())
    assert "test_case" in ds["train"].column_names
    assert len(ds["test"]) > 0


def test_hatecheck_metadata_has_functionality():
    ds = create_hatecheck_dataset_with_metadata(data_dir())
    assert "functionality" in ds["test"].column_names
