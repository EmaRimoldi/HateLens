"""Map character-level rationale spans to per-token binary labels (1=rationale, 0=other, -100=mask)."""

from __future__ import annotations

from typing import Any

import torch


def token_labels_from_char_spans(
    offset_mapping: list[tuple[int, int]] | torch.Tensor,
    char_spans: list[tuple[int, int]],
    *,
    seq_len: int | None = None,
) -> list[int]:
    """
    Align rationale character spans to tokenizer offset_mapping.

    Special / padding tokens with (0,0) offsets stay at -100 when they are not real text.
    """
    if seq_len is None:
        seq_len = len(offset_mapping)
    if isinstance(offset_mapping, torch.Tensor):
        om = offset_mapping.tolist()
    else:
        om = offset_mapping
    labels: list[int] = []
    for i in range(min(seq_len, len(om))):
        off = om[i]
        if not off or len(off) < 2:
            labels.append(-100)
            continue
        c0, c1 = int(off[0]), int(off[1])
        if c0 == 0 and c1 == 0:
            # Could be CLS/PAD depending on tokenizer; mask from rationale loss
            labels.append(-100)
            continue
        inside = any(c0 < e and c1 > s for s, e in char_spans)
        labels.append(1 if inside else 0)
    while len(labels) < seq_len:
        labels.append(-100)
    return labels[:seq_len]


def count_skipped_no_span(rationale_flags: list[bool]) -> int:
    return sum(1 for x in rationale_flags if not x)


def extract_char_spans_from_hatexplain_record(record: dict[str, Any], text: str) -> tuple[list[tuple[int, int]], bool]:
    """
    Try to build char spans for HateXplain-style records.

    Returns (spans, ok). If ``ok`` is False, skip rationale loss for this row.
    """
    # Some HF releases store token-level rationale lists
    if "rationale" in record and isinstance(record["rationale"], list):
        spans = []
        for item in record["rationale"]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                s, e = int(item[0]), int(item[1])
                spans.append((s, e))
        if spans:
            return spans, True
    # Annotated token spans as list of words — locate in text (first occurrence)
    rats = record.get("rationales")
    if isinstance(rats, list) and rats:
        spans = []
        pos = 0
        for w in rats:
            if not isinstance(w, str):
                continue
            w = w.strip()
            if not w:
                continue
            idx = text.find(w, pos)
            if idx < 0:
                continue
            spans.append((idx, idx + len(w)))
            pos = idx + len(w)
        if spans:
            return spans, True
    return [], False
