#!/usr/bin/env python3
"""Restore the full `target` column of a rollout parquet from its source dataset.

WHAT IS MISSING. A rollout/generation parquet carries a `target` that has been reduced to
the REASONING only -- the interior of the `<think>...</think>` block -- while the source
dataset's `target` is the whole assistant turn:

    source (file 1):  "<think>\\nOkay, so I need to ...\\n</think>\\n\\nLet \\( v \\) be ..."
    rollout (file 2):  "Okay, so I need to ...
                        **Final Answer**\\nThe speed ... \\boxed{10} ..."

So the `<think>` markers AND the rewritten solution that follows `</think>` are both gone.
Neither can be reconstructed from the rollout file alone, which is why the source is needed.

HOW ROWS ARE MATCHED. Not by position -- a rollout file is usually a shard, a filtered
subset, or a reordered sample, and a positional join would silently pair the wrong rows.
Two content keys are tried per row, in order:

  1. the target's own text, stripped to the `<think>` interior. This is exactly what the
     rollout file holds, so it is the most direct key available.
  2. the question (the last user message, with any `Please reason step by step ...`
     instruction prefix removed, since the two files render that differently).

A row that neither key resolves is left untouched and counted, never guessed at.

Run from the repo root:

    python Data/restore_target.py \\
        --source  ~/Desktop/rollout_data/openr1.parquet \\
        --input   ~/Desktop/0.parquet \\
        --output  ~/Desktop/0_target_restored.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

# The rollout renderer prepends this to the question; the source dataset does not. Stripping
# it is what lets the question serve as a join key across the two files.
INSTRUCTION_PREFIXES = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n",
    "Please reason step by step, and put your final answer within \\boxed{}.\n",
)


def _messages(value) -> list:
    """A prompt/target cell as a list of {role, content} dicts.

    Parquet round-trips these as numpy arrays, plain lists, or (for a single turn) a bare
    dict, so all three have to be accepted.
    """
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _content(value) -> str:
    """The text of a single-turn `target` cell."""
    msgs = _messages(value)
    if not msgs:
        return ""
    text = msgs[-1].get("content")
    return text if isinstance(text, str) else ""


def _question(value) -> str:
    """The question, from the last user message, without the instruction prefix."""
    msgs = _messages(value)
    text = ""
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            text = msg.get("content") or ""
            break
    else:
        if msgs:
            text = msgs[-1].get("content") or ""
    for prefix in INSTRUCTION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


def _think_interior(text: str) -> str:
    """The reasoning inside `<think>...</think>`, which is what the rollout file kept.

    Returns the whole string when there is no think block, so a source row that was never
    wrapped still produces a usable key rather than an empty one.
    """
    if "</think>" not in text:
        return text.strip()
    head = text.split("</think>", 1)[0]
    if head.startswith("<think>"):
        head = head[len("<think>"):]
    return head.strip()


def build_index(source: pd.DataFrame) -> tuple[dict, dict, Counter]:
    """Map both keys -> source row index.

    First writer wins on a collision. Duplicates are counted and reported: with an ambiguous
    key the restored target may come from either twin, and that is worth knowing even though
    both carry the same question.
    """
    by_interior: dict[str, int] = {}
    by_question: dict[str, int] = {}
    dupes = Counter()

    for i in range(len(source)):
        interior = _think_interior(_content(source["target"].iloc[i]))
        if interior:
            if interior in by_interior:
                dupes["interior"] += 1
            else:
                by_interior[interior] = i

        question = _question(source["prompt"].iloc[i])
        if question:
            if question in by_question:
                dupes["question"] += 1
            else:
                by_question[question] = i

    return by_interior, by_question, dupes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="parquet holding the intact `target` (file 1)")
    ap.add_argument("--input", required=True, help="parquet whose `target` is reduced (file 2)")
    ap.add_argument("--output", required=True, help="where to write the restored copy")
    ap.add_argument(
        "--keep-column",
        default=None,
        metavar="NAME",
        help="also keep the reduced target under this name (e.g. target_reasoning)",
    )
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero if any row could not be matched, instead of leaving it as-is",
    )
    args = ap.parse_args()

    source = pd.read_parquet(args.source)
    target_df = pd.read_parquet(args.input)
    for name, df, path in (("source", source, args.source), ("input", target_df, args.input)):
        for col in ("prompt", "target"):
            if col not in df.columns:
                raise SystemExit(f"{name} {path} has no {col!r} column (columns: {list(df.columns)})")

    by_interior, by_question, dupes = build_index(source)
    print(
        f"[index] {len(source):,} source rows -> {len(by_interior):,} interior keys, "
        f"{len(by_question):,} question keys"
        + (f" (collisions: interior={dupes['interior']}, question={dupes['question']})" if dupes else "")
    )

    restored = list(target_df["target"])
    reduced = list(target_df["target"])
    n_interior = n_question = n_missed = n_already = 0
    missed_rows: list[int] = []

    for i in range(len(target_df)):
        current = _content(target_df["target"].iloc[i])
        if "</think>" in current:
            # Already intact -- re-restoring would be a no-op, but counting it separately
            # keeps "nothing to do" from reading like a successful repair.
            n_already += 1
            continue

        src = by_interior.get(current.strip())
        if src is not None:
            n_interior += 1
        else:
            src = by_question.get(_question(target_df["prompt"].iloc[i]))
            if src is not None:
                n_question += 1

        if src is None:
            n_missed += 1
            if len(missed_rows) < 10:
                missed_rows.append(i)
            continue

        restored[i] = source["target"].iloc[src]

    out = target_df.copy()
    if args.keep_column:
        out[args.keep_column] = reduced
    out["target"] = restored

    print(
        f"[restore] {len(target_df):,} rows: matched {n_interior:,} by target text, "
        f"{n_question:,} by question, {n_already:,} already intact, {n_missed:,} unmatched"
    )
    if missed_rows:
        print(f"[restore] first unmatched row indices: {missed_rows}")

    # Report the size change: the whole point is that the solution after </think> comes back,
    # so a restored file whose targets did not grow means the join found the wrong rows.
    before = float(np.mean([len(_content(v)) for v in target_df["target"]]))
    after = float(np.mean([len(_content(v)) for v in restored]))
    n_think = sum("</think>" in _content(v) for v in restored)
    print(
        f"[restore] mean target chars {before:,.0f} -> {after:,.0f}; "
        f"{n_think:,}/{len(restored):,} now carry a <think> block"
    )

    if n_missed and args.require_all:
        raise SystemExit(f"--require-all: {n_missed} row(s) unmatched, nothing written")

    out.to_parquet(args.output, index=False)
    print(f"[restore] wrote {args.output}")


if __name__ == "__main__":
    main()
