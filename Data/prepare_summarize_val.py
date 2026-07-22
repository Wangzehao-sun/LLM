"""Carve a FROZEN summarize-val subset out of a summarize parquet.

Slices a fixed, reproducible subset (default 128 rows) from a parquet that
already carries the ``summarize_prompts`` column (produced by
``Data/prepare_summarize_prompts.py``). This subset is used by the trainer's
``_validate_summarize`` to observe the rephraser/summarize accuracy on a
constant set of questions during RL, isolating ability drift from data drift.

Writes TWO files (both default to sit next to --input):
  1. --output       : the 128-row frozen val subset. Defaults to
                      ``<input_stem>_val<N>.parquet``.
  2. --train-output : the input MINUS those 128 rows, so the val questions are
                      never trained on (zero leakage). Defaults to
                      ``<input_stem>_excl_val<N>.parquet``.

Rows without a usable ``summarize_prompts`` array are never picked for the val
subset (but are kept in the train-output). Both outputs keep the input schema
byte-for-byte — only rows are selected/dropped, never altered. A dedicated
fixed seed (distinct from the train-filter seed) keeps the subset stable
across regenerations.

CLI:

    python Data/prepare_summarize_val.py                 # 128 rows, seed 1234
    python Data/prepare_summarize_val.py --num-samples 256
    python Data/prepare_summarize_val.py --input /path/to/summarize.parquet
    python Data/prepare_summarize_val.py --output /path/to/val.parquet
    python Data/prepare_summarize_val.py --train-output /path/to/train.parquet
    python Data/prepare_summarize_val.py --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("/Users/zenohaoz/LLM/Data/deepmath_dgt6_n10000_summarize.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Source summarize parquet with a summarize_prompts column (default: %(default)s)")
    p.add_argument("--output", type=Path, default=None,
                   help="Destination parquet for the frozen val subset. "
                        "Defaults to <input_stem>_val<N>.parquet next to the input.")
    p.add_argument("--train-output", type=Path, default=None,
                   help="Destination parquet for the input MINUS the val rows. "
                        "Defaults to <input_stem>_excl_val<N>.parquet next to the input.")
    p.add_argument("--summarize-prompts-key", type=str, default="summarize_prompts",
                   help="Column holding the rendered summarize prompts (default: %(default)s)")
    p.add_argument("--num-samples", type=int, default=128, help="Rows to sample (default: %(default)s)")
    p.add_argument("--seed", type=int, default=1234,
                   help="Fixed random seed so the subset is frozen/reproducible (default: %(default)s)")
    return p.parse_args()


def _has_prompt(v) -> bool:
    if isinstance(v, np.ndarray):
        return v.size >= 1
    if isinstance(v, (list, tuple)):
        return len(v) >= 1
    return False


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  -> {len(df):,} rows")

    if args.summarize_prompts_key not in df.columns:
        raise SystemExit(
            f"Column '{args.summarize_prompts_key}' not in {args.input}. "
            "Run Data/prepare_summarize_prompts.py first."
        )

    # 只从"有可用 summarize_prompts"的行里抽 val；保留原始行索引以便从 df 精确剔除。
    eligible = df[df[args.summarize_prompts_key].apply(_has_prompt)]
    print(f"Rows with usable '{args.summarize_prompts_key}': {len(eligible):,}")

    if len(eligible) == 0:
        raise SystemExit("No rows carry a summarize prompt — refusing to write an empty parquet.")

    n = min(args.num_samples, len(eligible))
    if n < args.num_samples:
        print(f"  WARNING: only {n} rows available; capping --num-samples accordingly.")

    sampled = eligible.sample(n=n, random_state=args.seed)  # 保留原始索引，勿 reset
    val_df = sampled.reset_index(drop=True)
    # 从完整 df 按原始索引剔除 val 行（没有 summarize_prompts 的行仍留在 train）。
    train_df = df.drop(index=sampled.index).reset_index(drop=True)

    # 无泄漏自检：val 与 train 行数之和 == 原始行数，且无索引交集。
    assert len(val_df) + len(train_df) == len(df), "row count mismatch after split"

    if args.output is None:
        out_path = args.input.with_name(f"{args.input.stem}_val{n}.parquet")
    else:
        out_path = args.output

    if args.train_output is None:
        train_out = args.input.with_name(f"{args.input.stem}_excl_val{n}.parquet")
    else:
        train_out = args.train_output

    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_parquet(out_path, index=False)
    train_df.to_parquet(train_out, index=False)

    print(f"Wrote VAL   {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  rows    : {len(val_df):,}  (seed={args.seed}, frozen)")
    print(f"Wrote TRAIN {train_out} ({train_out.stat().st_size / 1e6:.2f} MB)")
    print(f"  rows    : {len(train_df):,}  (= {len(df):,} input - {len(val_df):,} val)")
    print(f"  columns : {val_df.columns.tolist()}")


if __name__ == "__main__":
    main()

