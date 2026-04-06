"""Configurable model IDs — legacy paper models + modern small LMs (no hard-coded training logic)."""

from __future__ import annotations

from typing import Final

# Keys are stable config identifiers; values are Hugging Face hub IDs (or local paths).
MODEL_REGISTRY: Final[dict[str, str]] = {
    # Legacy (HateTinyLLM-class)
    "tinyllama-1.1b": "PY007/TinyLlama-1.1B-step-50K-105b",
    "phi-2": "microsoft/phi-2",
    "opt-1.3b": "facebook/opt-1.3b",
    # Modern small open models (verify availability before large sweeps)
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "gemma-3-4b": "google/gemma-3-4b-it",
    # Reference encoder baseline (encoder-only; separate loader path)
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "roberta-large": "roberta-large",
}


def resolve_model_id(model_key_or_id: str) -> str:
    """Return hub ID if ``model_key_or_id`` is a registry key, else pass through as path/id."""
    return MODEL_REGISTRY.get(model_key_or_id, model_key_or_id)
