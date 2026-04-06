"""Load binary or structured checkpoints for unified evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from hatelens.labels import StructuredVocabBundle
from hatelens.modeling import load_sequence_classifier
from hatelens.modeling_structured import StructuredHateModel
from hatelens.peft_factory import build_base_transformer_backbone

logger = logging.getLogger(__name__)

CheckpointMode = Literal["binary", "structured"]


def detect_checkpoint_mode(checkpoint_dir: Path) -> CheckpointMode:
    """Infer ``binary`` (sequence-classification + PEFT) vs ``structured`` (multi-head) layout."""
    p = Path(checkpoint_dir).resolve()
    if (p / "structured_heads.pt").is_file() or (p / "structured_model.pt").is_file():
        return "structured"
    if (p / "adapter_config.json").is_file():
        return "binary"
    raise FileNotFoundError(
        f"Unrecognized checkpoint layout at {p}: expected adapter_config.json (binary) or "
        "structured_heads.pt / structured_model.pt (structured)."
    )


def load_binary_eval(
    checkpoint_dir: Path | str,
    *,
    device: torch.device,
    merge_adapters: bool = True,
) -> tuple[Any, Any]:
    """Load tokenizer + sequence-classification model (existing behavior)."""
    return load_sequence_classifier(checkpoint_dir, device=device, merge_adapters=merge_adapters)


def load_structured_eval(
    checkpoint_dir: Path | str,
    *,
    device: torch.device,
    quantization: str = "none",
) -> tuple[StructuredHateModel, Any, StructuredVocabBundle]:
    """
    Load ``StructuredHateModel`` from a training run under ``best_checkpoint/``.

    Expects ``peft_adapter/``, ``vocab/``, and ``structured_model.pt`` or ``structured_heads.pt``.
    """
    ckpt = Path(checkpoint_dir).resolve()
    peft_dir = ckpt / "peft_adapter"
    vocab_dir = ckpt / "vocab"
    full_sd = ckpt / "structured_model.pt"
    heads_pt = ckpt / "structured_heads.pt"

    if not peft_dir.is_dir():
        raise FileNotFoundError(f"Missing PEFT adapter dir: {peft_dir}")
    if not vocab_dir.is_dir():
        raise FileNotFoundError(f"Missing vocab dir: {vocab_dir}")

    cfg_path = peft_dir / "adapter_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        adapter_cfg: dict[str, Any] = json.load(f)
    base_id = str(adapter_cfg["base_model_name_or_path"])

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q = quantization if quantization in ("none", "4bit", "8bit") else "none"
    backbone = build_base_transformer_backbone(base_id, quantization=q)  # type: ignore[arg-type]
    backbone = PeftModel.from_pretrained(backbone, str(peft_dir))

    vocabs = StructuredVocabBundle.load_dir(vocab_dir)
    hidden = int(backbone.config.hidden_size)
    model = StructuredHateModel(
        backbone,
        hidden_size=hidden,
        n_main=2,
        n_target_group=vocabs.target_group.num_labels(),
        n_hate_type=vocabs.hate_type.num_labels(),
        n_explicitness=vocabs.explicitness.num_labels(),
        n_rationale_token=2,
    )

    if full_sd.is_file():
        try:
            sd = torch.load(full_sd, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(full_sd, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(
                "structured load_state_dict missing keys (sample): %s",
                list(missing)[:5],
            )
        if unexpected:
            logger.warning(
                "structured load_state_dict unexpected keys (sample): %s",
                list(unexpected)[:5],
            )
    elif heads_pt.is_file():
        model.load_heads(heads_pt, map_location="cpu")
    else:
        raise FileNotFoundError(f"Need {full_sd} or {heads_pt} for structured evaluation.")

    model.to(device)
    model.eval()
    return model, tokenizer, vocabs


def load_for_eval(
    checkpoint_dir: Path | str,
    *,
    device: torch.device,
    mode: CheckpointMode | Literal["auto"] = "auto",
    merge_binary_adapters: bool = True,
    quantization: str = "none",
) -> tuple[Any, Any, StructuredVocabBundle | None, CheckpointMode]:
    """
    Load model + tokenizer for evaluation.

    Returns ``(model, tokenizer, vocabs_or_none, mode)``.
    """
    ckpt = Path(checkpoint_dir).resolve()
    detected = detect_checkpoint_mode(ckpt)
    if mode in ("binary", "structured") and mode != detected:
        raise ValueError(
            f"Checkpoint under {ckpt} looks like {detected}, but mode={mode!r} was requested."
        )
    resolved: CheckpointMode = detected if mode == "auto" else mode  # type: ignore[assignment]
    if resolved == "binary":
        m, t = load_binary_eval(ckpt, device=device, merge_adapters=merge_binary_adapters)
        return m, t, None, "binary"
    m, t, v = load_structured_eval(ckpt, device=device, quantization=quantization)
    return m, t, v, "structured"
