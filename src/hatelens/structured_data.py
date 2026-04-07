"""Build tokenized structured datasets (multi-head labels + rationale + pair metadata)."""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from hatelens.datasets import (
    create_dynahate_dataset,
    create_hateeval_dataset,
    data_dir,
)
from hatelens.labels import IGNORE_INDEX, StructuredVocabBundle, build_vocabs_from_frequency
from hatelens.rationale_align import extract_char_spans_from_hatexplain_record, token_labels_from_char_spans

logger = logging.getLogger(__name__)


def _rows_dynahate(ds: DatasetDict) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ds}
    for split in ds:
        for i in range(len(ds[split])):
            row = ds[split][i]
            out[split].append(
                {
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                    "target_group": str(row.get("target", "unknown") or "unknown").lower(),
                    "hate_type": str(row.get("type", "unknown") or "unknown").lower(),
                    "explicitness": str(row.get("level", "unknown") or "unknown").lower(),
                    "pair_id": str(row.get("pair_id", "") or ""),
                    "pair_relation": str(row.get("pair_relation", "unknown") or "unknown"),
                    "dataset": "dynahate",
                    "raw": dict(row),
                }
            )
    return out


def _rows_hateeval(ds: DatasetDict) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ds}
    for split in ds:
        for i in range(len(ds[split])):
            row = ds[split][i]
            tr = row.get("TR", 0)
            out[split].append(
                {
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                    "target_group": "women_or_immigrants_task",
                    "hate_type": "targeted" if int(tr) == 1 else "non_targeted",
                    "explicitness": "unknown",
                    "pair_id": "",
                    "pair_relation": "unknown",
                    "dataset": "hateeval",
                    "raw": dict(row),
                }
            )
    return out


def _rows_hatexplain(ds: DatasetDict) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ds}
    for split in ds:
        for i in range(len(ds[split])):
            row = ds[split][i]
            lab = row.get("label", row.get("class", 0))
            if isinstance(lab, str):
                label = 1 if lab.lower() in ("1", "hate", "offensive") else 0
            else:
                label = 1 if int(lab) == 1 else 0
            out[split].append(
                {
                    "text": str(row.get("post", row.get("text", ""))),
                    "label": label,
                    "target_group": str(row.get("target", "unknown") or "unknown").lower(),
                    "hate_type": str(row.get("category", "unknown") or "unknown").lower(),
                    "explicitness": "unknown",
                    "pair_id": "",
                    "pair_relation": "unknown",
                    "dataset": "hatexplain",
                    "raw": dict(row),
                }
            )
    return out


def _collect_strings_for_vocab(rows_by_split: dict[str, list[dict[str, Any]]]) -> StructuredVocabBundle:
    train_rows = rows_by_split.get("train", [])
    tg, ht, ex = [], [], []
    for r in train_rows:
        tg.append(r["target_group"])
        ht.append(r["hate_type"])
        ex.append(r["explicitness"])
    # Ensure baseline tokens exist
    for x in ("unknown", "explicit", "implicit", "counter_speech"):
        ex.append(x)
    for x in ("unknown", "none", "other"):
        tg.append(x)
        ht.append(x)
    return build_vocabs_from_frequency(tg, ht, ex)


def _encode_row(
    row: dict[str, Any],
    tokenizer: Any,
    vocabs: StructuredVocabBundle,
    max_length: int,
    use_rationale: bool,
) -> dict[str, Any]:
    text = row["text"]
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )
    rid = enc.pop("offset_mapping")
    rationale_labels = [IGNORE_INDEX] * len(enc["input_ids"])
    has_rat = False
    if use_rationale and row.get("dataset") == "hatexplain":
        spans, ok = extract_char_spans_from_hatexplain_record(row["raw"], text)
        if ok and spans:
            rationale_labels = token_labels_from_char_spans(rid, spans, seq_len=len(enc["input_ids"]))
            has_rat = True
    pr = str(row.get("pair_relation", "unknown")).lower().replace("-", "_")
    if pr not in vocabs.pair_relation.stoi:
        pr = "unknown"
    lab = int(row["label"])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": lab,
        "main_labels": lab,
        "target_group_labels": vocabs.target_group.encode(row["target_group"]),
        "hate_type_labels": vocabs.hate_type.encode(row["hate_type"]),
        "explicitness_labels": vocabs.explicitness.encode(row["explicitness"]),
        "pair_relation_ids": vocabs.pair_relation.encode(pr),
        "pair_ids": row.get("pair_id") or "",
        "rationale_token_labels": rationale_labels,
        "has_rationale": has_rat,
    }


def _tokenize_split(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    vocabs: StructuredVocabBundle,
    max_length: int,
    use_rationale: bool,
) -> Dataset:
    records = [
        _encode_row(r, tokenizer, vocabs, max_length, use_rationale) for r in rows
    ]
    return Dataset.from_list(records)


def load_hatexplain_hf() -> DatasetDict:
    """
    HateXplain on the Hub still ships a legacy ``hatexplain.py`` loader; ``datasets`` 3.x
    rejects script-based datasets unless you use the parquet conversion branch.
    """
    last: Exception | None = None
    candidates: list[tuple[str, str | None]] = [
        ("hatexplain", "refs/convert/parquet"),
        ("bhavitvyahatekxp/hatexplain", "refs/convert/parquet"),
        ("hatexplain", None),
        ("bhavitvyahatekxp/hatexplain", None),
    ]
    for name, rev in candidates:
        try:
            if rev:
                return load_dataset(name, revision=rev)
            return load_dataset(name)
        except Exception as e:  # noqa: BLE001
            last = e
            logger.debug("hatexplain load name=%s revision=%s failed: %s", name, rev, e)
    raise RuntimeError(
        "Could not load HateXplain from Hugging Face. Try: cache with "
        "`load_dataset('hatexplain', revision='refs/convert/parquet')`, set HF_HOME, or see docs. "
        f"Last error: {last}"
    ) from last


def build_structured_dataset_dict(
    dataset_key: str,
    tokenizer: Any,
    *,
    max_length: int = 512,
    use_rationale: bool = True,
    vocabs: StructuredVocabBundle | None = None,
) -> tuple[DatasetDict, StructuredVocabBundle]:
    """
    ``dataset_key``: ``dynahate`` | ``hateeval`` | ``hatexplain`` | ``dynahate_hatexplain``.

    Returns (tokenized DatasetDict, vocab bundle).
    """
    rows_pack: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if dataset_key == "dynahate":
        d = create_dynahate_dataset(data_dir())
        rows_pack["single"] = _rows_dynahate(d)
    elif dataset_key == "hateeval":
        d = create_hateeval_dataset(data_dir())
        rows_pack["single"] = _rows_hateeval(d)
    elif dataset_key == "hatexplain":
        d = load_hatexplain_hf()
        rows_pack["single"] = _rows_hatexplain(d)
    elif dataset_key == "dynahate_hatexplain":
        d1 = create_dynahate_dataset(data_dir())
        d2 = load_hatexplain_hf()
        rows_pack["a"] = _rows_dynahate(d1)
        rows_pack["b"] = _rows_hatexplain(d2)
    else:
        raise ValueError(f"Unknown structured dataset_key={dataset_key!r}")

    if dataset_key == "dynahate_hatexplain":
        merged: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "validation", "test"):
            merged[split] = rows_pack["a"][split] + rows_pack["b"][split]
        rows_all = merged
    else:
        rows_all = rows_pack["single"]

    if vocabs is None:
        vocabs = _collect_strings_for_vocab(rows_all)

    parts: dict[str, Dataset] = {}
    for split in rows_all:
        parts[split] = _tokenize_split(rows_all[split], tokenizer, vocabs, max_length, use_rationale)

    return DatasetDict(parts), vocabs
