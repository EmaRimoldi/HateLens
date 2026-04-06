"""Command-line entry: evaluate, diagnose, LIME (optional), train."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hatelens import __version__
from hatelens.datasets import (
    create_dynahate_dataset,
    create_hatecheck_dataset,
    create_hatecheck_dataset_with_metadata,
    data_dir,
)
from hatelens.diagnostics import hatecheck_functionality_report
from hatelens.eval_runner import add_eval_cli_arguments
from hatelens.evaluation import classification_metrics, log_metrics, predict_batch
from hatelens.metrics_tables import build_export_arg_parser
from hatelens.modeling import default_checkpoints, load_sequence_classifier
from hatelens.paths import outputs_dir


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_evaluate(args: argparse.Namespace) -> None:
    name = "dynahate" if args.dynahate else "hatecheck"
    ck = default_checkpoints()
    base_id = ck["base"]["id"]
    post_path = args.adapter or ck["post"][name]

    if name == "dynahate":
        ds = create_dynahate_dataset(data_dir())
        text_key = "text"
        seed = 42
    else:
        ds = create_hatecheck_dataset(data_dir())
        text_key = "test_case"
        seed = 33

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    df = ds["test"].to_pandas()
    texts = df[text_key].astype(str).tolist()
    labels = np.array(df["label"].astype(int))

    logging.info("Evaluating base model: %s", base_id)
    m_base, tok_base = load_sequence_classifier(base_id, device=device)
    pred_b, prob_b = predict_batch(m_base, tok_base, texts, device, batch_size=args.batch_size)
    metrics_b = classification_metrics(labels, pred_b, prob_b)
    log_metrics("pre_ft", metrics_b)

    rows = [{"stage": "pre_ft", **metrics_b}]
    if not Path(post_path).exists():
        logging.warning("Post-FT checkpoint missing at %s — skipping post-FT eval.", post_path)
    else:
        logging.info("Evaluating fine-tuned adapter: %s", post_path)
        m_post, tok_post = load_sequence_classifier(post_path, device=device)
        pred_p, prob_p = predict_batch(m_post, tok_post, texts, device, batch_size=args.batch_size)
        metrics_p = classification_metrics(labels, pred_p, prob_p)
        log_metrics("post_ft", metrics_p)
        rows.append({"stage": "post_ft", **metrics_p})

    eval_root = Path(args.eval_output).resolve() if args.eval_output else outputs_dir() / "eval"
    out = eval_root / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics_summary.csv", index=False)
    logging.info("Wrote %s", out / "metrics_summary.csv")

    if args.plots and len(rows) > 1:
        post_m = {k: v for k, v in rows[1].items() if k != "stage"}
        _plot_simple_compare(metrics_b, post_m, out / "comparison_bar.png")


def _plot_simple_compare(pre: dict, post: dict, path: Path) -> None:
    keys = ["accuracy", "f1", "precision", "recall"]
    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [pre[k] for k in keys], w, label="pre-FT")
    ax.bar(x + w / 2, [post[k] for k in keys], w, label="post-FT")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("HateLens — validation metrics (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Wrote %s", path)


def cmd_diagnose(args: argparse.Namespace) -> None:
    ck = default_checkpoints()
    post_path = args.adapter or ck["post"]["hatecheck"]
    if not Path(post_path).exists():
        raise SystemExit(
            f"Adapter not found: {post_path}\n"
            "Train first (hatelens train ...) or pass --adapter / set HATELENS_CKPT_HATECHECK."
        )
    device = _device()
    ds = create_hatecheck_dataset_with_metadata(data_dir())
    df = ds["test"].to_pandas()
    texts = df["test_case"].astype(str).tolist()
    labels = np.array(df["label"].astype(int))

    m, tok = load_sequence_classifier(post_path, device=device)
    preds, probs = predict_batch(m, tok, texts, device, batch_size=args.batch_size)
    overall = classification_metrics(labels, preds, probs)
    log_metrics("hatecheck_test", overall)

    rep = hatecheck_functionality_report(df, preds)
    eval_root = Path(args.eval_output).resolve() if args.eval_output else outputs_dir() / "eval"
    out = eval_root / "hatecheck"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "functionality_breakdown.csv"
    rep.to_csv(csv_path, index=False)
    logging.info("Wrote %s (%d rows)", csv_path, len(rep))


def cmd_lime(args: argparse.Namespace) -> None:
    try:
        from hatelens.lime_scores import run_lime_for_dataset
    except ImportError as e:
        raise SystemExit(
            "LIME extras not installed. Run: pip install 'hatelens[lime]' or pip install lime"
        ) from e
    name = "dynahate" if args.dynahate else "hatecheck"
    run_lime_for_dataset(
        name,
        n_samples=args.n_samples,
        num_features=args.num_features,
        post_adapter_override=args.adapter,
    )


def cmd_train(args: argparse.Namespace) -> None:
    from hatelens.train_pipeline import run_training

    run_training(Path(args.config), args.dataset)


def cmd_eval_run(args: argparse.Namespace) -> None:
    from hatelens.eval_runner import run_eval_from_namespace

    run_eval_from_namespace(args)


def cmd_export_tables(args: argparse.Namespace) -> None:
    from hatelens.metrics_tables import export_tables_from_namespace

    export_tables_from_namespace(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hatelens", description="HateLens CLI")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="Batched pre/post FT metrics + optional plots")
    g = ev.add_mutually_exclusive_group(required=True)
    g.add_argument("--dynahate", action="store_true")
    g.add_argument("--hatecheck", action="store_true")
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Post-LoRA adapter dir (default: outputs/runs/tinyllama/<dataset>/best_checkpoint)",
    )
    ev.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Root directory for metrics CSV/plots (default: outputs/eval)",
    )
    ev.add_argument(
        "--plots",
        action="store_true",
        help="Save comparison_bar.png under the dataset subfolder",
    )
    ev.set_defaults(func=cmd_evaluate)

    dg = sub.add_parser(
        "diagnose-hatecheck",
        help="Per-functionality HateCheck table (requires fine-tuned checkpoint)",
    )
    dg.add_argument("--batch-size", type=int, default=8)
    dg.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Post-LoRA adapter path (default: env or tinyllama hatecheck run)",
    )
    dg.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Root for functionality_breakdown.csv (default: outputs/eval)",
    )
    dg.set_defaults(func=cmd_diagnose)

    lm = sub.add_parser("lime", help="Signed LIME word weights (optional dependency)")
    g2 = lm.add_mutually_exclusive_group(required=True)
    g2.add_argument("--dynahate", action="store_true")
    g2.add_argument("--hatecheck", action="store_true")
    lm.add_argument("--n-samples", type=int, default=500)
    lm.add_argument("--num-features", type=int, default=10)
    lm.add_argument("--adapter", type=str, default=None, help="Override post-LoRA adapter path")
    lm.set_defaults(func=cmd_lime)

    tr = sub.add_parser("train", help="LoRA fine-tuning via Hugging Face Trainer")
    tr.add_argument("config", type=str, help="Path to config YAML")
    tr.add_argument(
        "--dataset",
        choices=(
            "dynahate",
            "hatecheck",
            "hateeval",
            "hatexplain",
            "dynahate_hatexplain",
        ),
        required=True,
        help="Must match training_mode in the config (binary: dynahate/hatecheck/hateeval; "
        "structured: dynahate/hateeval/hatexplain/dynahate_hatexplain).",
    )
    tr.set_defaults(func=cmd_train)

    er = sub.add_parser(
        "eval-run",
        help=(
            "Unified evaluation: in-domain, cross-dataset, HateCheck, "
            "rationale, calibration, efficiency"
        ),
    )
    add_eval_cli_arguments(er)
    er.set_defaults(func=cmd_eval_run)

    ex = sub.add_parser(
        "export-tables",
        help="Aggregate metrics.json files into CSV/Markdown/LaTeX tables",
        parents=[build_export_arg_parser()],
        add_help=False,
        conflict_handler="resolve",
    )
    ex.set_defaults(func=cmd_export_tables)

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
