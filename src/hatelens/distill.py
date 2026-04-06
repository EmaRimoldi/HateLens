"""Optional teacher–student distillation scaffolding (Phase 4). Cache teacher outputs on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hatelens.paths import repo_root


def default_distill_cache_dir() -> Path:
    return repo_root() / "outputs" / "distill_cache"


def cache_teacher_batch(path: Path, batch_id: str, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{batch_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cached_teacher(path: Path, batch_id: str) -> dict[str, Any] | None:
    f = path / f"{batch_id}.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8"))
