"""HateXplain: rationales + token-level targets (Hugging Face ``bhavitvyahatekxp/hatexplain`` or fallback)."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from hatelens.schema import UnifiedExample

logger = logging.getLogger(__name__)

# Common HF identifiers; first successful load wins.
_HF_CANDIDATES: tuple[str, ...] = (
    "bhavitvyahatekxp/hatexplain",
    "hatexplain",
)


def _row_to_unified(record: dict[str, Any], idx: int, split: str) -> UnifiedExample:
    text = str(record.get("post", record.get("text", "")))
    label_raw = record.get("label", record.get("class", ""))
    if isinstance(label_raw, (int, float)):
        label_s = "hate" if int(label_raw) == 1 else "non_hate"
    else:
        s = str(label_raw).lower()
        label_s = "hate" if s in ("1", "hate", "offensive", "off") else "non_hate"
    rationales = record.get("rationales", record.get("targetSpans"))
    rat_text = None
    spans: list[tuple[int, int]] | None = None
    if isinstance(rationales, str):
        rat_text = rationales
    elif isinstance(rationales, list) and rationales:
        rat_text = " ".join(str(x) for x in rationales[:3])
    return UnifiedExample(
        id=f"hatexplain-{split}-{idx}",
        text=text,
        label=label_s,
        dataset_name="hatexplain",
        split=split,
        target_group=str(record.get("target", "unknown")),
        hate_type=str(record.get("category", "unknown")),
        explicitness="unknown",
        rationale_spans=spans,
        rationale_text=rat_text,
        source_metadata=dict(record),
    )


def load_hatexplain_unified() -> tuple[DatasetDict, list[UnifiedExample]]:
    """
    Load HateXplain and return HF DatasetDict plus a small list of unified examples (for tests).

    Raises the last loader error if all candidates fail (offline / missing dependency).
    """
    last_err: Exception | None = None
    for name in _HF_CANDIDATES:
        try:
            ds = load_dataset(name)
            logger.info("Loaded HateXplain-like dataset: %s", name)
            unified_samples: list[UnifiedExample] = []
            for split in ds:
                for i, row in enumerate(ds[split].select(range(min(5, len(ds[split]))))):
                    unified_samples.append(_row_to_unified(row, i, split))
            return ds, unified_samples
        except Exception as e:  # noqa: BLE001 — try next candidate
            last_err = e
            logger.debug("Could not load %s: %s", name, e)
    raise RuntimeError(
        "Could not load HateXplain from Hugging Face. "
        "Check network access, HF_TOKEN for gated models, and install `datasets`. "
        f"Last error: {last_err}"
    ) from last_err
