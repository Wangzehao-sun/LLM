"""Filter generation output by accuracy.

Reads all per-batch parquet files produced by ``verl.trainer.main_generation``
from a directory, concatenates them, and drops rows whose accuracy (the
``test_score.mean_score`` field, populated when generation was run with
``is_eval=True``) is strictly greater than ``--threshold``. The remaining
rows form the "hard set" that the model has not yet solved reliably and is
typically used as the next round's training data.

Always prints a per-bucket distribution: for each possible number of correct
rollouts (0, 1, ..., n) how many questions have that count and its percentage.

Two filtering modes:
  * threshold (default): keep rows with metric <= --threshold, optional --limit cap.
  * mix (--mix-ratios + --target-size): build a fixed-size set where each
    correct-count bucket contributes a specified fraction, e.g.
    '--mix-ratios 0:0.1,1:0.2,2:0.3,3:0.2,4:0.2 --target-size 5000'.
    Buckets short on questions contribute all they have (no top-up).

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
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the final number of saved rows. If fewer rows pass the filter, "
             "all are kept. Default: no cap.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --limit subsamples the kept rows (default: %(default)s).",
    )
    p.add_argument(
        "--mix-ratios",
        type=str,
        default=None,
        help="Enable MIX filtering mode instead of the threshold filter. Comma-separated "
             "'correct_count:ratio' pairs, e.g. '0:0.1,1:0.2,2:0.3,3:0.2,4:0.2'. Each "
             "bucket's quota = round(--target-size * ratio); rows are sampled from the "
             "questions whose number of correct rollouts (scores_per_response == 1) equals "
             "that count. If a bucket has fewer questions than its quota, all available are "
             "kept (no top-up). Requires --target-size. Ignores --threshold / --limit.",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Only used with --mix-ratios. Target total number of rows in the output; "
             "each bucket contributes round(target_size * ratio) rows.",
    )
    return p.parse_args()


def _correct_count(test_score) -> int | None:
    """Number of correct rollouts for one row = count of scores_per_response == 1.

    Returns None if the field is missing/unusable so callers can decide how to
    treat it (we drop such rows from bucket stats/sampling).
    """
    if not isinstance(test_score, dict):
        return None
    spr = test_score.get("scores_per_response")
    if spr is None:
        return None
    try:
        return int(sum(1 for s in spr if float(s) == 1.0))
    except (TypeError, ValueError):
        return None


def _print_bucket_stats(counts: pd.Series, n_total: int) -> None:
    """Print per-bucket (correct-count) question counts and percentages."""
    valid = counts.dropna().astype(int)
    n_valid = len(valid)
    print("Correct-rollout distribution (scores_per_response == 1 per question):")
    if n_valid == 0:
        print("  (no rows with usable scores_per_response)")
        return
    n_max = int(valid.max())
    for k in range(n_max + 1):
        n_k = int((valid == k).sum())
        pct = 100 * n_k / max(1, n_valid)
        print(f"  correct={k}: {n_k:>6,} ({pct:5.1f}%)")
    if len(counts) != n_valid:
        n_bad = len(counts) - n_valid
        print(f"  (unusable/missing scores_per_response: {n_bad})")


def _parse_mix_ratios(spec: str) -> dict[int, float]:
    """Parse '0:0.1,1:0.2,...' into {correct_count: ratio}."""
    out: dict[int, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"--mix-ratios entry must be 'count:ratio', got {part!r}")
        k_str, r_str = part.split(":", 1)
        try:
            k, r = int(k_str), float(r_str)
        except ValueError:
            raise SystemExit(f"--mix-ratios entry must be 'int:float', got {part!r}")
        if k < 0:
            raise SystemExit(f"--mix-ratios count must be >= 0, got {k}")
        if r < 0:
            raise SystemExit(f"--mix-ratios ratio must be >= 0, got {r}")
        out[k] = r
    if not out:
        raise SystemExit("--mix-ratios parsed empty")
    return out


def _mix_filter(
    df: pd.DataFrame, counts: pd.Series, ratios: dict[int, float],
    target_size: int, seed: int,
) -> pd.DataFrame:
    """Sample rows so each correct-count bucket contributes ~round(target*ratio).

    Buckets with fewer questions than their quota contribute all they have
    (no top-up from other buckets).
    """
    valid_counts = counts.dropna().astype(int)
    parts = []
    print(f"Mix filter: target_size={target_size:,}")
    for k in sorted(ratios):
        quota = int(round(target_size * ratios[k]))
        bucket_idx = valid_counts.index[valid_counts == k]
        n_avail = len(bucket_idx)
        take = min(quota, n_avail)
        if take > 0:
            chosen = df.loc[bucket_idx].sample(n=take, random_state=seed)
            parts.append(chosen)
        short = "" if take >= quota else f"  (SHORT: only {n_avail} available)"
        print(f"  correct={k}: quota={quota:>6,} take={take:>6,}{short}")
    if not parts:
        return df.iloc[0:0]
    out = pd.concat(parts, ignore_index=True)
    # Shuffle so buckets are interleaved, not blocked by correct-count.
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"  total kept: {len(out):,}")
    return out


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

    # --- Statistics: per-bucket correct-rollout distribution (always printed) ---
    correct_counts = df["test_score"].apply(_correct_count)
    _print_bucket_stats(correct_counts, len(df))

    # --- MIX mode: sample by per-bucket ratios up to a target size ---
    if args.mix_ratios is not None:
        if args.target_size is None or args.target_size <= 0:
            raise SystemExit("--mix-ratios requires --target-size > 0")
        ratios = _parse_mix_ratios(args.mix_ratios)
        out = _mix_filter(df, correct_counts, ratios, args.target_size, args.seed)
        if args.drop_test_score and "test_score" in out.columns:
            out = out.drop(columns=["test_score"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.output, index=False)
        size_mb = args.output.stat().st_size / 1e6
        print(f"Wrote {args.output} ({size_mb:.1f} MB, {len(out):,} rows)")
        return

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

    if args.limit is not None and len(out) > args.limit:
        out = out.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
        print(f"Limit: subsampled to {args.limit:,} rows (seed={args.seed}).")

    if args.drop_test_score:
        out = out.drop(columns=["test_score"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB, {len(out):,} rows)")


if __name__ == "__main__":
    main()
