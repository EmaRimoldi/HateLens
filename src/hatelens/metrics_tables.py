"""
Aggregate exported ``metrics.json`` files into manuscript-friendly tables.

Outputs: CSV, Markdown, or minimal LaTeX ``tabular`` text.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_metrics_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_in_domain_rows(payload: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block = payload.get("in_domain") or {}
    for ds, m in block.items():
        if not isinstance(m, dict) or "error" in m:
            continue
        row = {
            "source": source,
            "dataset": ds,
            **{k: v for k, v in m.items() if k not in ("error",)},
        }
        rows.append(row)
    return rows


def flatten_cross_dataset_rows(payload: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block = payload.get("cross_dataset") or {}
    for split_key, m in block.items():
        if not isinstance(m, dict) or "error" in m:
            continue
        rows.append(
            {
                "source": source,
                "split": split_key,
                **{k: v for k, v in m.items() if k not in ("error",)},
            }
        )
    return rows


def build_comparison_table(paths: list[Path], *, kind: str) -> pd.DataFrame:
    """
    ``kind``: one of ``binary_vs_structured``, ``rationale``, ``consistency``,
    ``cross``, ``hatecheck``, ``efficiency``.

    ``cross`` reads the ``cross_dataset`` block (train→test splits), not ``in_domain``.

    ``consistency`` uses the same in-domain flattening as ``binary_vs_structured``:
    pass one ``metrics.json`` per ablation run and compare rows by ``source``
    (eval run directory name).

    Expects each path to be an ``eval_runs/*/metrics.json`` file.
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        pl = load_metrics_json(p)
        src = p.parent.name
        if kind in ("binary_vs_structured", "rationale"):
            rows.extend(flatten_in_domain_rows(pl, source=src))
        elif kind == "cross":
            rows.extend(flatten_cross_dataset_rows(pl, source=src))
        elif kind == "hatecheck":
            hc = pl.get("hatecheck") or {}
            ov = hc.get("overall") if isinstance(hc, dict) else None
            if isinstance(ov, dict):
                rows.append({"source": src, **ov})
        elif kind == "efficiency":
            ef = pl.get("efficiency") or {}
            if isinstance(ef, dict) and "error" not in ef:
                rows.append({"source": src, **ef})
        else:
            rows.extend(flatten_in_domain_rows(pl, source=src))
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = ["| " + " | ".join(cols) + " |", sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_simple(df: pd.DataFrame, path: Path) -> None:
    """Write a minimal LaTeX ``tabular`` (escape manually if cells contain ``&``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    header = " & ".join(cols) + r" \\"
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", header, r"\hline"]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[c]) for c in cols) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_export_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate eval metrics.json files into tables")
    p.add_argument("metrics_json", nargs="+", type=Path, help="One or more metrics.json paths")
    p.add_argument(
        "--kind",
        default="binary_vs_structured",
        choices=[
            "binary_vs_structured",
            "rationale",
            "consistency",
            "cross",
            "hatecheck",
            "efficiency",
        ],
        help="Which section of metrics.json to emphasize",
    )
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-tex", type=Path, default=None)
    return p


def export_tables_from_namespace(args: argparse.Namespace) -> None:
    df = build_comparison_table([x.resolve() for x in args.metrics_json], kind=args.kind)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        logger.info("Wrote %s", args.out_csv)
    if args.out_md:
        write_markdown_table(df, args.out_md)
        logger.info("Wrote %s", args.out_md)
    if args.out_tex:
        write_latex_simple(df, args.out_tex)
        logger.info("Wrote %s", args.out_tex)
    if not any([args.out_csv, args.out_md, args.out_tex]):
        print(df.to_string(index=False))


def export_tables_main(argv: list[str] | None = None) -> None:
    args = build_export_arg_parser().parse_args(argv)
    export_tables_from_namespace(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_tables_main()


if __name__ == "__main__":
    main()
