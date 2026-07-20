"""Carve a FROZEN summarize-val subset out of a summarize parquet.

Slices a fixed, reproducible subset (default 128 rows) from a parquet that
already carries the ``summarize_prompts`` column (produced by
``Data/prepare_summarize_prompts.py``). This subset is used by the trainer's
``_validate_summarize`` to observe the rephraser/summarize accuracy on a
constant set of questions during RL, isolating ability drift from data drift.

Rows without a usable ``summarize_prompts`` array are dropped before sampling.
The output schema is byte-for-byte identical to the input — only rows are
selected, never altered. A dedicated fixed seed (distinct from the train-filter
seed) keeps the subset stable across regenerations.

CLI:

    python Data/prepare_summarize_val.py                 # 128 rows, seed 1234
    python Data/prepare_summarize_val.py --num-samples 256
    python Data/prepare_summarize_val.py --input /path/to/summarize.parquet
    python Data/prepare_summarize_val.py --output /path/to/val.parquet
    python Data/prepare_summarize_val.py --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("/Users/zenohaoz/LLM/Data/deepmath_dgt6_n10000_summarize.parquet")
DEFAULT_OUTPUT = Path("/Users/zenohaoz/LLM/Data/deepmath_dgt6_summarize_val128.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Source summarize parquet with a summarize_prompts column (default: %(default)s)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Destination parquet (default: %(default)s)")
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

    mask = df[args.summarize_prompts_key].apply(_has_prompt)
    filtered = df[mask].reset_index(drop=True)
    print(f"Rows with usable '{args.summarize_prompts_key}': {len(filtered):,}")

    if len(filtered) == 0:
        raise SystemExit("No rows carry a summarize prompt — refusing to write an empty parquet.")

    n = min(args.num_samples, len(filtered))
    if n < args.num_samples:
        print(f"  WARNING: only {n} rows available; capping --num-samples accordingly.")

    sampled = filtered.sample(n=n, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.output, index=False)

    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    print(f"  rows    : {len(sampled):,}  (seed={args.seed}, frozen)")
    print(f"  columns : {sampled.columns.tolist()}")


if __name__ == "__main__":
    main()
