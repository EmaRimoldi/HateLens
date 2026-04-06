"""Map raw dataset rows into :class:`hatelens.schema.UnifiedExample`."""

from __future__ import annotations

import json
from typing import Any

from hatelens.schema import UnifiedExample


def dynahate_row_to_unified(
    row: dict[str, Any],
    *,
    idx: int,
    split: str,
) -> UnifiedExample:
    """DynaHate CSV row after binarisation: ``label`` int 0/1, ``text`` str."""
    text = str(row.get("text", ""))
    lab = int(row["label"])
    label_s = "hate" if lab == 1 else "non_hate"
    meta: dict[str, Any] = {k: row[k] for k in row if k not in ("text", "label")}
    tg = row.get("target") or meta.get("target_group")
    ht = row.get("type") or meta.get("hate_type")
    return UnifiedExample(
        id=f"dynahate-{split}-{idx}",
        text=text,
        label=label_s,
        dataset_name="dynahate",
        split=split,
        target_group=str(tg).strip() if tg is not None and str(tg).strip() else "unknown",
        hate_type=str(ht).strip() if ht is not None and str(ht).strip() else "unknown",
        explicitness="unknown",
        source_metadata=meta,
    )


def hatecheck_row_to_unified(row: dict[str, Any], *, idx: int, split: str) -> UnifiedExample:
    """HateCheck split row (may include functionality, target_ident, etc.)."""
    text = str(row.get("test_case", row.get("text", "")))
    lab = int(row["label"])
    label_s = "hate" if lab == 1 else "non_hate"
    tg = str(row.get("target_ident", row.get("target_group", "unknown")))
    func = str(row.get("functionality", "unknown"))
    return UnifiedExample(
        id=f"hatecheck-{split}-{idx}",
        text=text,
        label=label_s,
        dataset_name="hatecheck",
        split=split,
        target_group=tg if tg else "unknown",
        hate_type=func if func != "unknown" else "unknown",
        explicitness="unknown",
        pair_id=str(row["case_id"]) if row.get("case_id") is not None else None,
        pair_relation="unknown",
        source_metadata={k: row[k] for k in row},
    )


def example_to_json_target(ex: UnifiedExample) -> str:
    """Compact supervision JSON for generative / instruction targets (Phase 3)."""
    return json.dumps(
        {
            "label": ex.label,
            "target_group": ex.target_group,
            "hate_type": ex.hate_type,
            "explicitness": ex.explicitness,
            "rationale": ex.rationale_text or "",
        },
        ensure_ascii=False,
    )
