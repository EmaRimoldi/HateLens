"""
Unified evaluation: in-domain, cross-dataset, HateCheck, rationale, calibration, efficiency.

Writes under ``outputs/eval_runs/<run_name>/`` (see ``paths.eval_runs_dir``).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from hatelens.datasets import (
    create_dynahate_dataset,
    create_hatecheck_dataset,
    create_hatecheck_dataset_with_metadata,
    create_hateeval_dataset,
    data_dir,
)
from hatelens.diagnostics import hatecheck_functionality_report
from hatelens.eval_loading import load_for_eval
from hatelens.evaluation import classification_metrics, predict_batch
from hatelens.evaluation_calibration import brier_score_binary, expected_calibration_error
from hatelens.evaluation_suite import run_binary_eval_bundle, write_results_json
from hatelens.labels import IGNORE_INDEX
from hatelens.modeling_structured import StructuredHateModel
from hatelens.paths import eval_runs_dir, repo_root
from hatelens.rationale_align import (
    extract_char_spans_from_hatexplain_record,
    token_labels_from_char_spans,
)
from hatelens.structured_data import load_hatexplain_hf

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_nan_dict(d: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            continue
        out[k] = v
    return out


def _subset_n(texts: list[str], labels: np.ndarray, n: int | None) -> tuple[list[str], np.ndarray]:
    if n is None or n >= len(texts):
        return texts, labels
    return texts[:n], labels[:n]


def load_texts_labels_for_dataset(
    dataset_name: str,
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> tuple[list[str], np.ndarray, str]:
    """
    Return (texts, binary labels, text column name) for supported datasets.

    Raises FileNotFoundError / RuntimeError with an actionable message if data is missing.
    """
    root = data_dir()
    if dataset_name == "dynahate":
        ds = create_dynahate_dataset(root)
        key = "text"
    elif dataset_name == "hatecheck":
        ds = create_hatecheck_dataset(root)
        key = "test_case"
    elif dataset_name == "hateeval":
        ds = create_hateeval_dataset(root)
        key = "text"
    elif dataset_name == "hatexplain":
        d = load_hatexplain_hf()
        if split not in d:
            raise KeyError(f"HateXplain split {split!r} not in {list(d.keys())}")
        part = d[split]
        key = "post" if "post" in part.column_names else "text"
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}")

    if dataset_name != "hatexplain":
        if split not in ds:
            raise KeyError(f"Split {split!r} missing for {dataset_name}")
        part = ds[split]

    df = part.to_pandas()
    texts = df[key].astype(str).tolist()
    lab_col = "label" if "label" in df.columns else "class"
    if lab_col not in df.columns:
        raise ValueError(f"No label column in {dataset_name} split {split}")
    labels = np.array(df[lab_col].astype(int))
    texts, labels = _subset_n(texts, labels, max_samples)
    return texts, labels, key


@dataclass
class EvalRunConfig:
    checkpoint: Path
    run_name: str
    mode: str = "auto"  # auto | binary | structured
    in_domain: list[str] = field(default_factory=list)
    cross_dataset: list[dict[str, str]] = field(default_factory=list)
    hatecheck: bool = False
    rationale: dict[str, Any] = field(default_factory=dict)
    calibration: bool = True
    efficiency: bool = True
    max_samples: int | None = None
    split: str = "test"
    batch_size: int = 8
    merge_binary_adapters: bool = True
    output_root: Path | None = None
    quantization: str = "none"


def _default_eval_yaml_schema() -> dict[str, Any]:
    return {
        "checkpoint": "outputs/runs/tinyllama/dynahate/best_checkpoint",
        "run_name": "example",
        "mode": "auto",
        "in_domain": ["dynahate", "hateeval"],
        "cross_dataset": [{"train": "dynahate", "test": "hateeval"}],
        "hatecheck": True,
        "rationale": {
            "enabled": False,
            "dataset": "hatexplain",
            "split": "test",
            "max_samples": 64,
        },
        "calibration": True,
        "efficiency": True,
        "max_samples": None,
        "split": "test",
        "batch_size": 8,
    }


def parse_eval_config(path: Path) -> EvalRunConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ck = Path(raw["checkpoint"])
    if not ck.is_absolute():
        ck = repo_root() / ck
    out_root = raw.get("output_root")
    orp = Path(out_root).resolve() if out_root else None
    cross = raw.get("cross_dataset") or []
    normalized: list[dict[str, str]] = []
    for item in cross:
        if isinstance(item, dict) and "train" in item and "test" in item:
            normalized.append({"train": str(item["train"]), "test": str(item["test"])})
    return EvalRunConfig(
        checkpoint=ck,
        run_name=str(raw.get("run_name", path.stem)),
        mode=str(raw.get("mode", "auto")),
        in_domain=list(raw.get("in_domain") or []),
        cross_dataset=normalized,
        hatecheck=bool(raw.get("hatecheck", False)),
        rationale=dict(raw.get("rationale") or {}),
        calibration=bool(raw.get("calibration", True)),
        efficiency=bool(raw.get("efficiency", True)),
        max_samples=raw.get("max_samples"),
        split=str(raw.get("split", "test")),
        batch_size=int(raw.get("batch_size", 8)),
        merge_binary_adapters=bool(raw.get("merge_binary_adapters", True)),
        output_root=orp,
        quantization=str(raw.get("quantization", "none")),
    )


def checkpoint_size_bytes(checkpoint_dir: Path) -> int:
    total = 0
    for p in Path(checkpoint_dir).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def efficiency_report(
    model: Any,
    checkpoint_dir: Path,
    *,
    device: torch.device,
    tokenizer: Any,
    train_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Parameter counts, disk size, and simple inference timing."""
    ckpt = Path(checkpoint_dir).resolve()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    rep: dict[str, Any] = {
        "total_parameters": int(total_p),
        "trainable_parameters": int(trainable),
        "checkpoint_size_bytes": int(checkpoint_size_bytes(ckpt)),
        "checkpoint_dir": str(ckpt),
        "train_runtime_seconds": train_runtime_seconds,
    }

    model.eval()
    texts = ["hello world", "this is a slightly longer test for batching"]
    enc = tokenizer(
        texts[:1],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    latency_ms: float | None = None
    throughput_sps: float | None = None
    peak_mem_bytes: int | None = None

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Latency batch 1
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(10):
                _ = model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - t0) / 10.0 * 1000.0

        enc2 = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(20):
                _ = model(**enc2)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t1) / 20.0
        throughput_sps = len(texts) / dt if dt > 0 else None

        if device.type == "cuda":
            peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))
    except Exception as e:  # noqa: BLE001
        logger.warning("Efficiency profiling failed: %s", e)
        rep["efficiency_error"] = str(e)

    rep["inference_latency_ms_batch1"] = latency_ms
    rep["inference_throughput_samples_per_s_batch_gt1"] = throughput_sps
    rep["peak_memory_bytes_cuda"] = peak_mem_bytes
    return rep


def _predict_structured_main(
    model: StructuredHateModel,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[int] = []
    probs: list[float] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            out = model(**enc)
            logits = out.logits
            pr = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
            pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        probs.extend(pr)
        preds.extend(pred)
    return np.array(preds, dtype=np.int64), np.array(probs, dtype=np.float64)


def _rationale_token_metrics(
    model: StructuredHateModel,
    tokenizer: Any,
    device: torch.device,
    *,
    max_samples: int,
    split: str,
) -> dict[str, Any]:
    """Token-level P/R/F1 on HateXplain rationale spans (structured models only)."""
    d = load_hatexplain_hf()
    if split not in d:
        return {"error": f"split {split!r} not in HateXplain: {list(d.keys())}"}
    part = d[split]
    n = min(max_samples, len(part))
    tp = fp = fn = 0
    examples: list[dict[str, Any]] = []
    model.eval()
    for i in range(n):
        row = part[i]
        text = str(row.get("post", row.get("text", "")))
        raw = dict(row)
        spans, ok = extract_char_spans_from_hatexplain_record(raw, text)
        if not ok or not spans:
            continue
        enc = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        om = enc.pop("offset_mapping")[0]
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        gold = token_labels_from_char_spans(om.tolist(), spans, seq_len=int(input_ids.size(1)))
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_tok = out.logits_rationale.argmax(dim=-1)[0].cpu().tolist()
        for j, g in enumerate(gold):
            if g == IGNORE_INDEX:
                continue
            p = int(pred_tok[j]) if j < len(pred_tok) else 0
            g_bin = 1 if g == 1 else 0
            if g_bin == 1 and p == 1:
                tp += 1
            elif g_bin == 0 and p == 1:
                fp += 1
            elif g_bin == 1 and p == 0:
                fn += 1
        if len(examples) < 20:
            examples.append({"text": text[:400], "spans_ok": ok, "tp": tp, "fp": fp, "fn": fn})
    denom_p = tp + fp
    denom_r = tp + fn
    prec = tp / denom_p if denom_p else 0.0
    rec = tp / denom_r if denom_r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "token_precision": float(prec),
        "token_recall": float(rec),
        "token_f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "examples_preview": examples,
    }


def run_eval(cfg: EvalRunConfig) -> Path:
    """Execute a full eval run; returns the output directory."""
    device = _device()
    root_out = cfg.output_root or eval_runs_dir()
    out_dir = (Path(root_out) / cfg.run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = cfg.checkpoint.resolve()
    mode = cfg.mode
    if mode not in ("auto", "binary", "structured"):
        raise ValueError(f"Invalid mode={mode!r}")

    load_mode: str = mode if mode in ("binary", "structured") else "auto"
    model, tokenizer, _vocabs, loaded_mode = load_for_eval(
        ckpt,
        device=device,
        mode=load_mode,
        merge_binary_adapters=cfg.merge_binary_adapters,
        quantization=cfg.quantization,
    )

    results: dict[str, Any] = {
        "run_name": cfg.run_name,
        "checkpoint": str(ckpt),
        "loaded_mode": loaded_mode,
        "device": str(device),
    }

    predictions_rows: list[dict[str, Any]] = []

    def run_split(name: str, split: str, tag: str) -> dict[str, Any]:
        texts, labels, _tk = load_texts_labels_for_dataset(
            name, split=split, max_samples=cfg.max_samples
        )
        if loaded_mode == "structured":
            preds, probs = _predict_structured_main(
                model, tokenizer, texts, device, batch_size=cfg.batch_size
            )
        else:
            preds, probs = predict_batch(
                model, tokenizer, texts, device, batch_size=cfg.batch_size
            )
        bundle = run_binary_eval_bundle(labels, preds, probs)
        bundle = _safe_nan_dict(bundle)
        for i in range(len(texts)):
            predictions_rows.append(
                {
                    "tag": tag,
                    "dataset": name,
                    "split": split,
                    "label": int(labels[i]),
                    "pred": int(preds[i]),
                    "prob_hate": float(probs[i]),
                    "text": texts[i][:2000],
                }
            )
        return bundle

    in_domain_metrics: dict[str, Any] = {}
    for ds in cfg.in_domain:
        try:
            tag = f"in_domain:{ds}"
            in_domain_metrics[ds] = run_split(ds, cfg.split, tag)
        except Exception as e:  # noqa: BLE001
            logger.exception("In-domain eval failed for %s", ds)
            in_domain_metrics[ds] = {"error": str(e)}

    cross_metrics: dict[str, Any] = {}
    for pair in cfg.cross_dataset:
        tr = pair["train"]
        te = pair["test"]
        key = f"train_{tr}_test_{te}"
        try:
            cross_metrics[key] = run_split(te, cfg.split, f"cross:{key}")
        except Exception as e:  # noqa: BLE001
            logger.exception("Cross eval failed for %s", key)
            cross_metrics[key] = {"error": str(e)}

    hatecheck_block: dict[str, Any] | None = None
    if cfg.hatecheck:
        try:
            meta = create_hatecheck_dataset_with_metadata(data_dir())
            df = meta["test"].to_pandas()
            texts = df["test_case"].astype(str).tolist()
            labels = np.array(df["label"].astype(int))
            texts, labels = _subset_n(texts, labels, cfg.max_samples)
            if loaded_mode == "structured":
                preds, probs = _predict_structured_main(
                    model, tokenizer, texts, device, batch_size=cfg.batch_size
                )
            else:
                preds, probs = predict_batch(
                    model, tokenizer, texts, device, batch_size=cfg.batch_size
                )
            overall = _safe_nan_dict(classification_metrics(labels, preds, probs))
            overall.update(
                {
                    "ece": expected_calibration_error(labels, probs),
                    "brier": brier_score_binary(labels, probs),
                }
            )
            df_hc = df.iloc[: len(preds)].reset_index(drop=True)
            rep = hatecheck_functionality_report(df_hc, preds)
            hatecheck_block = {
                "overall": overall,
                "per_functionality_csv": rep.to_dict(orient="records"),
            }
            for i in range(len(texts)):
                predictions_rows.append(
                    {
                        "tag": "hatecheck",
                        "dataset": "hatecheck",
                        "split": "test",
                        "label": int(labels[i]),
                        "pred": int(preds[i]),
                        "prob_hate": float(probs[i]),
                        "text": texts[i][:2000],
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("HateCheck eval failed")
            hatecheck_block = {"error": str(e)}

    rationale_block: dict[str, Any] | None = None
    rat = cfg.rationale or {}
    if rat.get("enabled") and loaded_mode == "structured":
        rationale_block = _rationale_token_metrics(
            model,
            tokenizer,
            device,
            max_samples=int(rat.get("max_samples", 64)),
            split=str(rat.get("split", "test")),
        )
    elif rat.get("enabled") and loaded_mode != "structured":
        rationale_block = {
            "skipped": (
                "Rationale metrics require a structured checkpoint with token rationale head."
            )
        }

    cal_block: dict[str, Any] | None = None
    if cfg.calibration and cfg.in_domain:
        # Report calibration on first successful in-domain dataset
        for ds, block in in_domain_metrics.items():
            if isinstance(block, dict) and "ece" in block:
                cal_block = {"reference_dataset": ds, "ece": block["ece"], "brier": block["brier"]}
                break
        if cal_block is None:
            cal_block = {"note": "No in-domain metrics with calibration keys; run in_domain lists."}

    eff_block: dict[str, Any] | None = None
    if cfg.efficiency:
        tr_time: float | None = None
        tm_path = ckpt.parent / "train_metrics.json"
        if tm_path.is_file():
            try:
                tm = json.loads(tm_path.read_text(encoding="utf-8"))
                tr_time = None
                if "train_metrics" in tm:
                    tr_time = float(tm["train_metrics"]["train_runtime"])
            except (KeyError, ValueError, TypeError):
                tr_time = None
        eff_block = efficiency_report(
            model,
            ckpt,
            device=device,
            tokenizer=tokenizer,
            train_runtime_seconds=tr_time,
        )

    results["in_domain"] = in_domain_metrics
    results["cross_dataset"] = cross_metrics
    results["hatecheck"] = hatecheck_block
    results["rationale"] = rationale_block
    results["calibration"] = cal_block
    results["efficiency"] = eff_block

    write_results_json(out_dir / "metrics.json", results)

    # Flat CSV (one row per metric group)
    flat_rows: list[dict[str, Any]] = []
    for ds, m in in_domain_metrics.items():
        if isinstance(m, dict) and "error" not in m:
            row = {"group": "in_domain", "dataset": ds, **m}
            flat_rows.append(row)
    for k, m in cross_metrics.items():
        if isinstance(m, dict) and "error" not in m:
            row = {"group": "cross_dataset", "dataset": k, **m}
            flat_rows.append(row)
    if hatecheck_block and "overall" in (hatecheck_block or {}):
        flat_rows.append(
            {"group": "hatecheck", "dataset": "hatecheck", **hatecheck_block["overall"]}
        )
    pd.DataFrame(flat_rows).to_csv(out_dir / "metrics.csv", index=False)

    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in predictions_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rat_path = out_dir / "rationale_examples.jsonl"
    with open(rat_path, "w", encoding="utf-8") as f:
        if rationale_block and "examples_preview" in (rationale_block or {}):
            for ex in rationale_block.get("examples_preview", []):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        elif rationale_block:
            f.write(json.dumps(rationale_block, default=str) + "\n")

    if eff_block is not None:
        write_results_json(out_dir / "efficiency.json", eff_block)

    logger.info("Wrote eval run to %s", out_dir)
    return out_dir


def add_eval_cli_arguments(p: argparse.ArgumentParser) -> None:
    """Flags for ``hatelens eval-run`` (also used by standalone ``build_eval_arg_parser``)."""
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file (see configs/eval/minimal.yaml). CLI flags override when also passed.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Training best_checkpoint directory",
    )
    p.add_argument("--run-name", type=str, default=None, help="Subfolder under outputs/eval_runs/")
    p.add_argument("--mode", type=str, default="auto", choices=("auto", "binary", "structured"))
    p.add_argument(
        "--in-domain",
        nargs="*",
        default=None,
        help="Datasets: dynahate hatecheck hateeval hatexplain",
    )
    p.add_argument(
        "--cross",
        nargs="*",
        default=None,
        metavar="TRAIN:TEST",
        help="Cross-dataset pairs, e.g. dynahate:hateeval hateeval:dynahate",
    )
    p.add_argument(
        "--hatecheck",
        action="store_true",
        help="Run HateCheck test + per-functionality breakdown",
    )
    p.add_argument(
        "--rationale",
        action="store_true",
        help="Token rationale metrics (structured checkpoints)",
    )
    p.add_argument("--rationale-max-samples", type=int, default=64)
    p.add_argument("--no-efficiency", action="store_true")
    p.add_argument("--no-calibration-summary", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-root", type=Path, default=None, help="Default: outputs/eval_runs")
    p.add_argument("--quantization", type=str, default="none", choices=("none", "4bit", "8bit"))


def build_eval_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified HateLens evaluation (eval-run)")
    add_eval_cli_arguments(p)
    return p


def _cross_items_from_cli(raw: list[str] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not raw:
        return out
    for item in raw:
        if ":" not in item:
            raise SystemExit(f"Invalid --cross item {item!r}; expected train_ds:test_ds")
        a, b = item.split(":", 1)
        out.append({"train": a.strip(), "test": b.strip()})
    return out


def run_eval_from_namespace(ns: argparse.Namespace) -> Path:
    root = repo_root()

    if ns.config:
        cfg = parse_eval_config(ns.config.resolve())
        if ns.checkpoint is not None:
            ck = ns.checkpoint
            cfg.checkpoint = ck if ck.is_absolute() else (root / ck).resolve()
        if ns.run_name is not None:
            cfg.run_name = ns.run_name
        if ns.in_domain is not None:
            cfg.in_domain = list(ns.in_domain)
        if ns.cross is not None:
            cfg.cross_dataset = _cross_items_from_cli(ns.cross)
        if ns.hatecheck:
            cfg.hatecheck = True
        if ns.rationale:
            cfg.rationale = {
                **(cfg.rationale or {}),
                "enabled": True,
                "max_samples": ns.rationale_max_samples,
            }
        if ns.no_efficiency:
            cfg.efficiency = False
        if ns.no_calibration_summary:
            cfg.calibration = False
        if ns.max_samples is not None:
            cfg.max_samples = ns.max_samples
        if ns.split:
            cfg.split = ns.split
        if ns.batch_size:
            cfg.batch_size = ns.batch_size
        if ns.output_root is not None:
            o = ns.output_root
            cfg.output_root = o if o.is_absolute() else (root / o).resolve()
        cfg.mode = ns.mode
        cfg.quantization = ns.quantization
        return run_eval(cfg)

    if ns.checkpoint is None or ns.run_name is None:
        raise SystemExit("Provide --config, or both --checkpoint and --run-name.")

    ck = ns.checkpoint
    ck_resolved = ck if ck.is_absolute() else (root / ck).resolve()
    in_dom = list(ns.in_domain) if ns.in_domain else ["dynahate", "hateeval"]
    out_root: Path | None = None
    if ns.output_root is not None:
        o = ns.output_root
        out_root = o.resolve() if o.is_absolute() else (root / o).resolve()

    cfg = EvalRunConfig(
        checkpoint=ck_resolved,
        run_name=ns.run_name,
        mode=ns.mode,
        in_domain=in_dom,
        cross_dataset=_cross_items_from_cli(ns.cross),
        hatecheck=bool(ns.hatecheck),
        rationale={
            "enabled": bool(ns.rationale),
            "max_samples": int(ns.rationale_max_samples),
            "split": ns.split,
        },
        calibration=not ns.no_calibration_summary,
        efficiency=not ns.no_efficiency,
        max_samples=ns.max_samples,
        split=ns.split,
        batch_size=ns.batch_size,
        output_root=out_root,
        quantization=ns.quantization,
    )
    return run_eval(cfg)


def run_eval_from_argv(argv: list[str] | None) -> Path:
    return run_eval_from_namespace(build_eval_arg_parser().parse_args(argv))
