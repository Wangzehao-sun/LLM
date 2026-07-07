"""Drop rows whose summarize prompt columns are empty.

The summarize-prompt generation script (``prepare_summarize_prompts.py``) can
fail to render a prompt for a few rows, leaving ``summarize_prompt`` /
``summarize_prompts`` as empty arrays (``shape == (0,)``). Such rows are fine
during normal steps (the dataset pads them), but crash the failure-recycle
``np.stack`` collate in ``new_ray_trainer.py`` when they land in the failure
buffer alongside normal rows (mismatched shapes).

This script removes those rows so the training data is clean at the source.

Usage:

    python Data/filter_empty_summarize.py \
        --input  openr1_hard_thinkonly_split_summarize_new.parquet \
        --output openr1_hard_thinkonly_split_summarize_new_clean.parquet
        # default columns: summarize_prompt, summarize_prompts (any empty -> drop)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _is_empty(v) -> bool:
    """True if the cell is a missing/empty summarize prompt."""
    if v is None:
        return True
    if isinstance(v, (list, tuple)):
        return len(v) == 0
    if isinstance(v, np.ndarray):
        return v.size == 0
    # scalar NaN
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Input parquet.")
    p.add_argument("--output", type=Path, required=True, help="Cleaned output parquet.")
    p.add_argument(
        "--columns",
        nargs="+",
        default=["summarize_prompt", "summarize_prompts"],
        help="Columns to check; a row is dropped if ANY of these is empty "
             "(default: summarize_prompt summarize_prompts).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"--input is not a file: {args.input}")

    df = pd.read_parquet(args.input)
    n0 = len(df)
    print(f"Read {n0:,} rows from {args.input}; columns: {df.columns.tolist()}")

    cols = [c for c in args.columns if c in df.columns]
    missing = [c for c in args.columns if c not in df.columns]
    if missing:
        print(f"  WARNING: columns not found, ignored: {missing}")
    if not cols:
        raise SystemExit(f"None of the requested columns present: {args.columns}")

    # Row is bad if ANY checked column is empty.
    bad = pd.Series(False, index=df.index)
    for c in cols:
        col_bad = df[c].apply(_is_empty)
        n_bad_c = int(col_bad.sum())
        print(f"  '{c}': {n_bad_c} empty row(s)")
        bad = bad | col_bad

    bad_idx = df.index[bad].tolist()
    n_drop = len(bad_idx)
    print(f"Dropping {n_drop} row(s) with empty summarize prompt (positions: {bad_idx[:20]}"
          f"{' ...' if n_drop > 20 else ''}).")

    out = df[~bad].reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB, {len(out):,} rows; removed {n0 - len(out):,}).")


if __name__ == "__main__":
    main()
