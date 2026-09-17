#!/usr/bin/env python3
"""Build a Prefix-RFT training parquet from one of this repo's `target`-carrying parquets.

WHY THIS EXISTS. Prefix-RFT (github.com/ZeroYuHuang/prefix_rft) reads a `demos` column --
a LIST of reference solutions per question, from which its dataset samples one and takes a
token prefix. Our parquets carry a single `target` message list instead. This is the same
transform the upstream repo does in recipe/prepare_data/luffy_train.py:42:

    demos = [target[0]['content']]
    example["demos"] = demos

plus the two things upstream folds into the same pass:

  * the instruction is APPENDED to the question, not prepended. Upstream uses
    `raw_prompt + '\\nPlease reason step by step, and put your final answer within \\boxed{}'`
    and drops the system turn, so the prompt is a single user message. Reproducing the
    method means reproducing that rendering -- the prefix is later appended as an
    unfinished assistant turn via `continue_final_message=True`, and a stray system turn
    changes what the model conditions on.
  * `demos_corr` (per-demo correctness flags). Upstream leaves this absent and its dataset
    then defaults every demo to True, so we write it explicitly rather than relying on that
    fallback -- a demo that does not verify is a prefix that teaches a wrong derivation.
    Pass --skip-verify to write all-True without checking (much faster, matches upstream's
    own default path, which has the correctness check commented out).

Run from the repo root:

    python Data/prepare_prefix_rft.py \\
        --input  Data/dapo_math/dapo_en_math_solution_9k_random_thinkonly.parquet \\
        --output /tmp/repro/prefix_rft_data/train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Upstream's exact suffix (recipe/prepare_data/luffy_train.py:25). No trailing period, and a
# leading newline -- reproduced byte for byte because it is part of the prompt the model sees.
INST_AFTER_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}"

# Prefixes our own parquets may already carry, which have to come off before the suffix goes
# on -- otherwise the question is wrapped twice and no longer matches upstream's rendering.
KNOWN_PREFIXES = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n",
    "Please reason step by step, and put your final answer within \\boxed{}.\n",
    "Please reason step by step, and put your final answer within \\boxed{}.",
)


def _messages(value) -> list:
    """A prompt/target cell as a list of dicts (parquet gives ndarray, list, or bare dict)."""
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _question(prompt_value) -> str:
    """The bare question: last user message, with any instruction wrapper stripped."""
    msgs = _messages(prompt_value)
    text = ""
    for msg in reversed(msgs):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = msg.get("content") or ""
            break
    else:
        if msgs and isinstance(msgs[-1], dict):
            text = msgs[-1].get("content") or ""
    text = text.strip()
    for prefix in KNOWN_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _target_text(target_value) -> str:
    msgs = _messages(target_value)
    if not msgs or not isinstance(msgs[0], dict):
        return ""
    content = msgs[0].get("content")
    return content.strip() if isinstance(content, str) else ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="parquet with prompt / target / reward_model")
    ap.add_argument("--output", required=True, help="destination train.parquet")
    ap.add_argument(
        "--val-output",
        default=None,
        help="also write a validation split here (rows are taken off the end)",
    )
    ap.add_argument("--val-rows", type=int, default=0, help="rows for --val-output")
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="write demos_corr all-True without running math_verify (upstream's default path)",
    )
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N rows")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    for col in ("prompt", "target", "reward_model"):
        if col not in df.columns:
            raise SystemExit(f"{args.input} has no {col!r} column (columns: {list(df.columns)})")
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)

    verify_fn = None
    if not args.skip_verify:
        try:
            from math_verify import parse, verify

            def verify_fn(solution: str, gt: str) -> bool:
                try:
                    return bool(verify(parse(f"${gt}$"), parse(solution)))
                except Exception:
                    # A demo whose answer cannot be parsed is not evidence of a WRONG
                    # derivation, so it stays True -- the same default upstream falls back to.
                    return True
        except ImportError:
            print("[warn] math_verify not installed; writing demos_corr all-True", flush=True)

    prompts, demos_col, corr_col = [], [], []
    n_empty = n_wrong = 0

    for i in range(len(df)):
        question = _question(df["prompt"].iloc[i])
        solution = _target_text(df["target"].iloc[i])
        if not solution:
            n_empty += 1

        # Single user turn, instruction appended -- upstream's rendering exactly.
        prompts.append(np.array([{"role": "user", "content": question + INST_AFTER_PROMPT}], dtype=object))
        demos_col.append(np.array([solution], dtype=object))

        ok = True
        if verify_fn is not None and solution:
            gt = df["reward_model"].iloc[i]
            gt = gt.get("ground_truth", "") if hasattr(gt, "get") else ""
            ok = verify_fn(solution, str(gt))
            if not ok:
                n_wrong += 1
        corr_col.append(np.array([ok], dtype=bool))

    out = df.copy()
    out["prompt"] = prompts
    out["demos"] = demos_col
    out["demos_corr"] = corr_col

    print(
        f"[prepare] {len(out):,} rows: {n_empty:,} with an empty demo"
        + (f", {n_wrong:,} whose demo does not verify" if verify_fn is not None else " (unverified)")
    )
    lens = np.array([len(d[0]) for d in demos_col])
    print(f"[prepare] demo chars p50/p90/max: {np.percentile(lens,50):,.0f}/"
          f"{np.percentile(lens,90):,.0f}/{lens.max():,}")
    print(f"[prepare] sample prompt: {out['prompt'].iloc[0][0]['content'][:110]!r}")

    if args.val_output and args.val_rows > 0:
        n = min(args.val_rows, len(out) - 1)
        val, train = out.iloc[-n:].reset_index(drop=True), out.iloc[:-n].reset_index(drop=True)
        Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
        val.to_parquet(args.val_output, index=False)
        print(f"[prepare] wrote {args.val_output} ({len(val):,} rows)")
    else:
        train = out

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output, index=False)
    print(f"[prepare] wrote {args.output} ({len(train):,} rows)")


if __name__ == "__main__":
    main()
