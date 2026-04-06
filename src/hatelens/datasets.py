"""Dataset builders for DynaHate, HateCheck, and HateEval (SemEval 2019 English)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Final

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from hatelens.paths import data_dir

logger = logging.getLogger(__name__)

_REPO_DYNAHATE: Final = (
    "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset.git"
)
_DYNA_FILE: Final = "Dynamically Generated Hate Dataset v0.2.3.csv"

# Official SemEval 2019 Task 5 English TSVs (HateEval / HateEvalTeam, GitHub cicl2018/HateEvalTeam).
HATEEVAL_TRAIN_URL: Final[str] = (
    "https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/"
    "Data%20Files/Data%20Files/%232%20Development-English-A/train_en.tsv"
)
HATEEVAL_DEV_URL: Final[str] = (
    "https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/"
    "Data%20Files/Data%20Files/%232%20Development-English-A/dev_en.tsv"
)
# Official test release is unlabeled (id, text only); we do not use it for metric computation.
HATEEVAL_TEST_URL: Final[str] = (
    "https://raw.githubusercontent.com/cicl2018/HateEvalTeam/master/"
    "Data%20Files/Data%20Files/%233%20Evaluation-English-A/test_en.tsv"
)


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
    keep = ["text", "label", "split"]
    for extra in ("target", "type", "level"):
        if extra in df.columns:
            keep.append(extra)
    df = df[keep].copy()
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


def download_hateeval_tsvs(base_dir: Path | str | None = None, *, download_test_unlabeled: bool = False) -> Path:
    """
    Download English HateEval train/dev TSVs into ``data/HateEval/``.

    Labels: HS = hate speech (0/1). TR/AG kept in columns for optional structured use.
    """
    root = Path(base_dir or data_dir()).resolve() / "HateEval"
    root.mkdir(parents=True, exist_ok=True)
    for url, name in (
        (HATEEVAL_TRAIN_URL, "train_en.tsv"),
        (HATEEVAL_DEV_URL, "dev_en.tsv"),
    ):
        dest = root / name
        if not dest.exists():
            logger.info("Downloading %s -> %s", url, dest)
            urllib.request.urlretrieve(url, dest)  # noqa: S310
    if download_test_unlabeled:
        dest = root / "test_en_unlabeled.tsv"
        if not dest.exists():
            urllib.request.urlretrieve(HATEEVAL_TEST_URL, dest)  # noqa: S310
    return root


def _read_hateeval_tsv(path: Path) -> pd.DataFrame:
    """Tab-separated; columns: id, text, HS, TR, AG (test may be id, text only)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in {path}")
    df["text"] = df["text"].fillna("").astype(str)
    if "HS" in df.columns:
        df["label"] = pd.to_numeric(df["HS"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        df["label"] = -1  # unlabeled
    return df


def create_hateeval_dataset(base_dir: Path | str | None = None) -> DatasetDict:
    """
    English HateEval (SemEval 2019) with official train/dev splits.

    **Test split**: the public ``test_en.tsv`` has **no labels**. We expose **dev** as ``test``
    for metric computation (same labeled distribution as validation). This matches common
    replication setups when gold test labels are not bundled. For true test evaluation, obtain
    gold labels from the task organizers and place a ``test_en_labeled.tsv`` with columns
    ``id``, ``text``, ``HS``, ``TR``, ``AG`` — then extend this loader.

    Raises ``FileNotFoundError`` if TSVs are missing; call ``download_hateeval_tsvs()`` first.
    """
    root = Path(base_dir or data_dir()).resolve() / "HateEval"
    train_p = root / "train_en.tsv"
    dev_p = root / "dev_en.tsv"
    if not train_p.is_file() or not dev_p.is_file():
        raise FileNotFoundError(
            f"Missing {train_p} or {dev_p}. Run hatelens.datasets.download_hateeval_tsvs() "
            "or place official train_en.tsv and dev_en.tsv under data/HateEval/."
        )
    train_df = _read_hateeval_tsv(train_p)
    dev_df = _read_hateeval_tsv(dev_p)
    for d in (train_df, dev_df):
        if "TR" in d.columns:
            d["TR"] = pd.to_numeric(d["TR"], errors="coerce").fillna(0).astype(int)
        if "AG" in d.columns:
            d["AG"] = pd.to_numeric(d["AG"], errors="coerce").fillna(0).astype(int)
    keep = ["text", "label"] + [c for c in ("TR", "AG", "id") if c in train_df.columns]
    train_ds = Dataset.from_pandas(train_df[keep].reset_index(drop=True))
    # Split official dev into validation vs test (stratified) — public test_en.tsv is unlabeled.
    if len(dev_df) > 2:
        vdf, tdf = train_test_split(
            dev_df[keep],
            test_size=0.5,
            random_state=42,
            stratify=dev_df["label"],
        )
        val_ds = Dataset.from_pandas(vdf.reset_index(drop=True))
        test_ds = Dataset.from_pandas(tdf.reset_index(drop=True))
    else:
        val_ds = Dataset.from_pandas(dev_df[keep].reset_index(drop=True))
        test_ds = val_ds
    return DatasetDict(train=train_ds, validation=val_ds, test=test_ds)


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
    "create_hateeval_dataset",
    "create_gab_dataset",
    "describe_dataset",
    "download_dynahate",
    "download_hateeval_tsvs",
    "data_dir",
]
