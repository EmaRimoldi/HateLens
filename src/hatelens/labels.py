"""Stable label vocabularies and ignore index for masked multi-task losses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# PyTorch CrossEntropyLoss default ignore
IGNORE_INDEX: int = -100


@dataclass
class LabelVocabulary:
    """Maps string labels to contiguous ids; index 0 reserved for padding/unknown class."""

    name: str
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_labels(
        cls,
        name: str,
        labels: list[str],
        *,
        unknown_token: str = "unknown",
        ordered: list[str] | None = None,
    ) -> LabelVocabulary:
        if ordered is not None:
            uniq = [unknown_token] + [x for x in ordered if x != unknown_token]
        else:
            uniq = sorted({unknown_token, *labels})
        stoi = {s: i for i, s in enumerate(uniq)}
        itos = {i: s for s, i in stoi.items()}
        return cls(name=name, stoi=stoi, itos=itos)

    def encode(self, label: str | None) -> int:
        if not label or str(label).strip() == "":
            return self.stoi.get("unknown", 0)
        s = str(label).strip().lower()
        return self.stoi.get(s, self.stoi.get("unknown", 0))

    def num_labels(self) -> int:
        return len(self.stoi)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> LabelVocabulary:
        itos = {int(k): v for k, v in d["itos"].items()}
        stoi = dict(d["stoi"])
        return cls(name=d["name"], stoi=stoi, itos=itos)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LabelVocabulary:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


def default_main_binary() -> LabelVocabulary:
    # Two logits only: 0 = non_hate, 1 = hate (matches hatelens.modeling.LABEL2ID)
    return LabelVocabulary(
        name="main",
        stoi={"non_hate": 0, "hate": 1},
        itos={0: "non_hate", 1: "hate"},
    )


@dataclass
class StructuredVocabBundle:
    main: LabelVocabulary
    target_group: LabelVocabulary
    hate_type: LabelVocabulary
    explicitness: LabelVocabulary
    pair_relation: LabelVocabulary = field(default_factory=lambda: _default_pair_rel())

    def save_dir(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.main.save(directory / "main.json")
        self.target_group.save(directory / "target_group.json")
        self.hate_type.save(directory / "hate_type.json")
        self.explicitness.save(directory / "explicitness.json")
        self.pair_relation.save(directory / "pair_relation.json")

    @classmethod
    def load_dir(cls, directory: Path) -> StructuredVocabBundle:
        return cls(
            main=LabelVocabulary.load(directory / "main.json"),
            target_group=LabelVocabulary.load(directory / "target_group.json"),
            hate_type=LabelVocabulary.load(directory / "hate_type.json"),
            explicitness=LabelVocabulary.load(directory / "explicitness.json"),
            pair_relation=LabelVocabulary.load(directory / "pair_relation.json"),
        )


def _default_pair_rel() -> LabelVocabulary:
    return LabelVocabulary.from_labels(
        "pair_relation",
        ["unknown", "none", "invariant", "flip_to_hate", "flip_to_non_hate"],
        unknown_token="unknown",
    )


def build_vocabs_from_frequency(
    target_groups: list[str],
    hate_types: list[str],
    explicitnesses: list[str],
) -> StructuredVocabBundle:
    """Build vocabs from collected raw strings (training-time)."""
    return StructuredVocabBundle(
        main=default_main_binary(),
        target_group=LabelVocabulary.from_labels("target_group", target_groups),
        hate_type=LabelVocabulary.from_labels("hate_type", hate_types),
        explicitness=LabelVocabulary.from_labels("explicitness", explicitnesses),
        pair_relation=_default_pair_rel(),
    )
