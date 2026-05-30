"""Filter generation output by accuracy.

Reads all per-batch parquet files produced by ``verl.trainer.main_generation``
from a directory, concatenates them, and drops rows whose accuracy (the
``test_score.mean_score`` field, populated when generation was run with
``is_eval=True``) is strictly greater than ``--threshold``. The remaining
rows form the "hard set" that the model has not yet solved reliably and is
typically used as the next round's training data.

Usage:

    python filter_by_accuracy.py \
        --input-dir  $HOME/LLM/Train/verl/logs/eval_<MODEL>_<DATA>/save_data \
        --output     $HOME/LLM/Data/deepmath_hard.parquet
        # default --threshold 0.5, --metric mean_score
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-batch parquet files written by main_generation.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination single parquet file.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Drop rows with metric > threshold (default: {DEFAULT_THRESHOLD}).",
    )
    p.add_argument(
        "--metric",
        choices=["mean_score", "max_score"],
        default="mean_score",
        help="Which test_score field to compare against threshold (default: %(default)s).",
    )
    p.add_argument(
        "--drop-test-score",
        action="store_true",
        help="Drop the test_score column from the output (smaller file, but loses accuracy info).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {args.input_dir}")

    files = sorted(args.input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {args.input_dir}")

    print(f"Reading {len(files)} parquet shard(s) from {args.input_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  -> {len(df):,} rows total, columns: {df.columns.tolist()}")

    if "test_score" not in df.columns:
        raise SystemExit(
            "Column 'test_score' not found; was generation run with is_eval=True?"
        )

    metric_vals = df["test_score"].apply(
        lambda x: x.get(args.metric) if isinstance(x, dict) else None
    ).astype("float64")

    if metric_vals.isna().any():
        n_bad = int(metric_vals.isna().sum())
        print(f"  WARNING: {n_bad} rows have no '{args.metric}' field; treating as 0.0.")
        metric_vals = metric_vals.fillna(0.0)

    mask = metric_vals <= args.threshold
    n_kept = int(mask.sum())
    n_dropped = len(df) - n_kept
    pct = lambda n: 100 * n / max(1, len(df))

    print(f"Filter: keep rows with {args.metric} <= {args.threshold}")
    print(f"  kept    : {n_kept:>6,} ({pct(n_kept):5.1f}%)")
    print(f"  dropped : {n_dropped:>6,} ({pct(n_dropped):5.1f}%)")
    if n_kept:
        kept = metric_vals[mask]
        print(
            f"  kept {args.metric}: "
            f"min={kept.min():.3f} mean={kept.mean():.3f} max={kept.max():.3f}"
        )

    out = df[mask].reset_index(drop=True)
    if args.drop_test_score:
        out = out.drop(columns=["test_score"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB, {len(out):,} rows)")


if __name__ == "__main__":
    main()
