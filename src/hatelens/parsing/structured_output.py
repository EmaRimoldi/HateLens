"""Parse and validate model-generated structured JSON supervision targets."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_structured_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output; return None if invalid."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def normalize_prediction(obj: dict[str, Any]) -> dict[str, str]:
    """Map parsed JSON into canonical string fields."""
    lab = str(obj.get("label", "non_hate")).lower()
    if lab not in ("hate", "non_hate"):
        lab = "hate" if lab in ("1", "true", "yes", "offensive") else "non_hate"
    return {
        "label": lab,
        "target_group": str(obj.get("target_group", "unknown")),
        "hate_type": str(obj.get("hate_type", "unknown")),
        "explicitness": str(obj.get("explicitness", "unknown")),
        "rationale": str(obj.get("rationale", "")),
    }
