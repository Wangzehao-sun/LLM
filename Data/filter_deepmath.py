"""Filter deepmath.parquet by difficulty and sub-sample.

Default behaviour: keep rows where ``extra_info.difficulty > 6`` and
randomly sample 1000 of them, writing the result next to the source
parquet.

CLI overrides:

    python filter_deepmath.py                       # d>6, n=1000
    python filter_deepmath.py --min-difficulty 7    # d>7
    python filter_deepmath.py --num-samples 5000    # 5k samples
    python filter_deepmath.py --output /tmp/x.parquet
    python filter_deepmath.py --seed 0
    python filter_deepmath.py --inclusive           # use d>=threshold

The output schema is byte-for-byte the same as deepmath.parquet (and
therefore the same as openr1.parquet) — we only drop rows, never alter
columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("/Users/zenohaoz/LLM/Data/deepmath.parquet")
DEFAULT_OUTPUT_DIR = Path("/Users/zenohaoz/LLM/Data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source parquet (default: %(default)s)")
    p.add_argument("--output", type=Path, default=None, help="Destination parquet. Defaults to "
                   "<input_dir>/deepmath_d{thr}_n{n}.parquet")
    p.add_argument("--min-difficulty", type=float, default=6.0,
                   help="Difficulty threshold (default: %(default)s)")
    p.add_argument("--inclusive", action="store_true",
                   help="Use difficulty >= threshold instead of strict >")
    p.add_argument("--num-samples", type=int, default=10000, help="Rows to sample (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  -> {len(df):,} rows")

    # Difficulty lives inside the dict-typed `extra_info` column.
    difficulty = df["extra_info"].apply(lambda x: x.get("difficulty") if isinstance(x, dict) else None)

    op = ">=" if args.inclusive else ">"
    mask = difficulty >= args.min_difficulty if args.inclusive else difficulty > args.min_difficulty
    filtered = df[mask].reset_index(drop=True)
    print(f"Filter difficulty {op} {args.min_difficulty}: {len(filtered):,} rows match")

    if len(filtered) == 0:
        raise SystemExit("No rows match the filter — refusing to write an empty parquet.")

    n = min(args.num_samples, len(filtered))
    if n < args.num_samples:
        print(f"  WARNING: only {n} rows available; capping --num-samples accordingly.")

    sampled = filtered.sample(n=n, random_state=args.seed).reset_index(drop=True)

    if args.output is None:
        thr_str = f"{args.min_difficulty:g}".replace(".", "p")
        prefix = "ge" if args.inclusive else "gt"
        out_path = DEFAULT_OUTPUT_DIR / f"deepmath_d{prefix}{thr_str}_n{n}.parquet"
    else:
        out_path = args.output

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(out_path, index=False)

    sampled_diff = sampled["extra_info"].apply(lambda x: x["difficulty"])
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  rows                : {len(sampled):,}")
    print(f"  difficulty min/mean/max: "
          f"{sampled_diff.min():.2f} / {sampled_diff.mean():.2f} / {sampled_diff.max():.2f}")
    print(f"  columns             : {sampled.columns.tolist()}")


if __name__ == "__main__":
    main()
