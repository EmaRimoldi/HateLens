"""PEFT configuration and model wrapping (LoRA, QLoRA, DoRA; optional AdaLoRA / PiSSA)."""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

from hatelens.modeling import ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)

PeftKind = Literal["lora", "qlora", "dora", "adalora", "pissa"]


def _parse_task_type(task_type: str) -> Any:
    """Map YAML strings (``SEQ_CLS``, ``FEATURE_EXTRACTION``, …) to ``peft.TaskType``."""
    return getattr(TaskType, str(task_type), TaskType.SEQ_CLS)


def _bnb_config(
    load_in_4bit: bool,
    bnb_4bit_compute_dtype: str = "bfloat16",
) -> Any | None:
    try:
        import bitsandbytes as bnb  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning(
            "bitsandbytes not installed — QLoRA disabled. Install: pip install bitsandbytes"
        )
        return None
    dt = torch.bfloat16 if bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dt,
    )


def build_base_sequence_classifier(
    model_checkpoint: str,
    *,
    quantization: Literal["none", "4bit", "8bit"] = "none",
) -> PreTrainedModel:
    """Load ``AutoModelForSequenceClassification`` with optional QLoRA-style quantization."""
    kwargs: dict[str, Any] = {
        "num_labels": 2,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
    if quantization == "4bit":
        bnb = _bnb_config(load_in_4bit=True)
        if bnb is None:
            logger.warning("Falling back to full precision (no bitsandbytes).")
        else:
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = "auto"
    elif quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            BitsAndBytesConfig = None  # type: ignore[misc, assignment]
        if BitsAndBytesConfig is None:
            pass
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, **kwargs)
    return model


def build_base_transformer_backbone(
    model_checkpoint: str,
    *,
    quantization: Literal["none", "4bit", "8bit"] = "none",
) -> PreTrainedModel:
    """Decoder/transformer backbone without classification head (for multi-head structured models)."""
    kwargs: dict[str, Any] = {}
    if quantization == "4bit":
        bnb = _bnb_config(load_in_4bit=True)
        if bnb is None:
            logger.warning("Falling back to full precision (no bitsandbytes).")
        else:
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = "auto"
    elif quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            BitsAndBytesConfig = None  # type: ignore[misc, assignment]
        if BitsAndBytesConfig is not None:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
    return AutoModel.from_pretrained(model_checkpoint, **kwargs)


def build_peft_config(
    task_type: str,
    *,
    peft_type: PeftKind | str = "lora",
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    use_rslora: bool = False,
) -> Any:
    """Return a PEFT config. Unsupported kinds fall back to LoRA with a log message."""
    tm = target_modules or ["k_proj", "v_proj"]
    peft_type = str(peft_type).lower()
    tt = _parse_task_type(str(task_type))
    if peft_type in ("lora", "qlora"):
        return LoraConfig(
            task_type=tt,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=tm,
            use_rslora=use_rslora,
        )
    if peft_type == "dora":
        try:
            from peft import LoraConfig as LC
            # DoRA: PEFT 0.12+ exposes use_dora on LoraConfig
            return LC(
                task_type=tt,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=tm,
                use_dora=True,
            )
        except (TypeError, ImportError) as e:
            logger.warning("DoRA not available (%s); using plain LoRA.", e)
            return LoraConfig(
                task_type=tt,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=tm,
            )
    if peft_type == "adalora":
        try:
            from peft import AdaLoraConfig

            return AdaLoraConfig(
                task_type=tt,
                r=r,
                lora_alpha=lora_alpha,
                target_modules=tm,
                lora_dropout=lora_dropout,
            )
        except ImportError:
            logger.warning("AdaLoraConfig missing; using LoRA.")
    if peft_type == "pissa":
        try:
            return LoraConfig(
                task_type=tt,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=tm,
                init_lora_weights="pissa",
            )
        except TypeError:
            logger.warning("PiSSA init not supported; using LoRA defaults.")
    return LoraConfig(
        task_type=tt,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=tm,
    )


def apply_peft(
    model: PreTrainedModel,
    peft_cfg: Any,
) -> PeftModel | PreTrainedModel:
    return get_peft_model(model, peft_cfg)
