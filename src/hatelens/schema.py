"""Unified training/evaluation example schema (beyond binary labels)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final


@dataclass
class UnifiedExample:
    """Single row in the cross-dataset canonical format."""

    id: str
    text: str
    label: str  # "hate" | "non_hate"
    dataset_name: str
    split: str
    target_group: str = "unknown"
    protected_attribute: str | None = None
    hate_type: str = "unknown"
    explicitness: str = "unknown"
    rationale_spans: list[tuple[int, int]] | None = None
    rationale_text: str | None = None
    pair_id: str | None = None
    pair_relation: str = "unknown"  # invariant | flip_to_hate | flip_to_non_hate | unknown
    policy_text: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


UNKNOWN: Final[str] = "unknown"
