"""Dataset builders for DynaHate and HateCheck."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final

import pandas as pd
from datasets import Dataset, DatasetDict

from hatelens.paths import data_dir

_REPO_DYNAHATE: Final = (
    "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset.git"
)
_DYNA_FILE: Final = "Dynamically Generated Hate Dataset v0.2.3.csv"


def download_dynahate(base_dir: Path | str | None = None) -> Path:
    """Download DynaHate v0.2.3 CSV into ``<base>/DynaHate/``."""
    dataset_dir = Path(base_dir or data_dir()).resolve() / "DynaHate"
    clone_root = dataset_dir / "_clone_sparse"
    if clone_root.exists():
        shutil.rmtree(clone_root)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            _REPO_DYNAHATE,
            str(clone_root),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone_root), "sparse-checkout", "set", "."],
        check=True,
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    src = clone_root / _DYNA_FILE
    dst = dataset_dir / "dynahate_v0.2.3.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)
    shutil.rmtree(clone_root, ignore_errors=True)
    return dataset_dir


def _split_and_binarise_dynahate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df[["text", "label", "split"]].copy()
    df["label"] = (df["label"].str.lower() == "hate").astype(int)
    df["split"] = df["split"].map({"train": "train", "dev": "validation", "test": "test"})
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        sub = df[df["split"] == split].drop(columns="split").reset_index(drop=True)
        out[split] = sub
    return out


def create_dynahate_dataset(base_dir: Path | str | None = None) -> DatasetDict:
    """Load DynaHate from ``<base>/DynaHate/dynahate_v0.2.3.csv`` with binary labels."""
    root = Path(base_dir or data_dir()).resolve()
    csv_path = root / "DynaHate" / "dynahate_v0.2.3.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Place the CSV there or call hatlens.datasets.download_dynahate()."
        )
    df = pd.read_csv(csv_path)
    splits = _split_and_binarise_dynahate(df)
    return DatasetDict({k: Dataset.from_pandas(v) for k, v in splits.items()})


def _hatecheck_split_path() -> Path:
    return data_dir() / "hatecheck" / "hatecheck_split.csv"


def create_hatecheck_dataset(base_dir: Path | str | None = None) -> DatasetDict:
    """
    Load stratified HateCheck splits (test_case + label).

    Expects ``data/hatecheck/hatecheck_split.csv`` (generated from the official suite).
    """
    _ = base_dir  # reserved for symmetry with DynaHate; paths are under repo data/
    split_csv_path = _hatecheck_split_path()
    if not split_csv_path.exists():
        raise FileNotFoundError(
            f"Missing {split_csv_path}. Run preprocessing under data/hatecheck/ or copy splits."
        )
    df = pd.read_csv(split_csv_path)
    splits = {
        "train": df[df["split"] == "train"],
        "validation": df[df["split"] == "validation"],
        "test": df[df["split"] == "test"],
    }
    return DatasetDict(
        {split: Dataset.from_pandas(sub[["test_case", "label"]]) for split, sub in splits.items()}
    )


def create_hatecheck_dataset_with_metadata(base_dir: Path | str | None = None) -> DatasetDict:
    """
    Same splits as ``create_hatecheck_dataset`` but keep HateCheck metadata columns
    for diagnostic evaluation (functionality, target group, etc.).
    """
    _ = base_dir
    split_csv_path = _hatecheck_split_path()
    if not split_csv_path.exists():
        raise FileNotFoundError(split_csv_path)
    df = pd.read_csv(split_csv_path)
    # Normalise optional index column noise from pandas exports
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    meta_cols = [
        c
        for c in (
            "functionality",
            "case_id",
            "target_ident",
            "direction",
            "templ_id",
            "test_case",
            "label",
        )
        if c in df.columns
    ]
    splits = {
        "train": df[df["split"] == "train"],
        "validation": df[df["split"] == "validation"],
        "test": df[df["split"] == "test"],
    }
    parts = {
        split: Dataset.from_pandas(sub[meta_cols].reset_index(drop=True))
        for split, sub in splits.items()
    }
    return DatasetDict(parts)


def create_gab_dataset(base_dir: Path | str | None = None) -> DatasetDict:
    """Load preprocessed Gab CSV with columns ``text``, ``hate_speech_idx``, ``split``."""
    root = Path(base_dir or data_dir()).resolve()
    csv_path = root / "gab" / "processed_gab_final.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. This optional dataset is not bundled with HateLens by default."
        )
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("The 'split' column is missing in the input file.")
    splits = {
        "train": df[df["split"] == "train"],
        "validation": df[df["split"] == "evaluation"],
        "test": df[df["split"] == "test"],
    }
    return DatasetDict(
        {
            split: Dataset.from_pandas(sub[["text", "hate_speech_idx"]].reset_index(drop=True))
            for split, sub in splits.items()
        }
    )


def describe_dataset(ds_dict: DatasetDict, name: str | None = None) -> None:
    """Pretty-print split sizes and first example."""
    title = f"Dataset structure – {name}" if name else "Dataset structure"
    bar = "═" * len(title)
    print(f"{bar}\n{title}\n{bar}")
    for split in ("train", "validation", "test"):
        if split not in ds_dict:
            print(f"• {split:<10}  (missing)\n")
            continue
        ds = ds_dict[split]
        n = len(ds)
        cols = ", ".join(ds.column_names)
        sample = f"        {ds[0]}" if n else "        (empty)"
        print(f"• {split:<10}  {n:>6,} rows   [columns: {cols}]\n{sample}\n")


__all__ = [
    "create_dynahate_dataset",
    "create_hatecheck_dataset",
    "create_hatecheck_dataset_with_metadata",
    "create_gab_dataset",
    "describe_dataset",
    "download_dynahate",
    "data_dir",
]
