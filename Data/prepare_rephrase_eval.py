"""Build a rephrase evaluation set from a rendered summarize parquet.

Two jobs, both about making the measurement mean something:

1. **Flatten the prompt.** ``prepare_summarize_prompts.py`` writes
   ``summarize_prompt`` as a length-1 array wrapping one messages list (and
   ``summarize_prompts`` as a length-K array). ``verl.trainer.main_generation``
   reads ``data.prompt_key`` and hands it straight to ``apply_chat_template``, so
   it needs a flat ``[system, user]`` list. This copies the chosen prompt into the
   ``prompt`` column.

2. **Check the prompt matches training.** If the eval prompt differs from the one
   the SFT data was built with, the accuracy number measures prompt drift rather
   than model quality. Pass ``--train-parquet`` and this refuses to run when the
   two ``prompt_id`` values disagree.

The original ``prompt`` column (the bare question) is preserved as
``question_prompt``, flattened the same way, so ONE file can measure two things:
the rephrase task, and the model's plain problem-solving ability on the same
questions. Choose which column to evaluate with ``sweep_sft_checkpoints.sh``'s
``PROMPT_KEY``.

Usage:

    python Data/prepare_rephrase_eval.py \\
        --input  Data/eval_rephrase.parquet \\
        --output Data/eval_rephrase_flat.parquet \\
        --train-parquet Data/rephraser_sft.parquet

    # then, at evaluation time:
    #   PROMPT_KEY=prompt          -> the rephrase task
    #   PROMPT_KEY=question_prompt -> the bare question
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def flatten_prompt(cell, index: int):
    """Return a flat list of ``{role, content}`` dicts from a prompt cell.

    Handles both shapes the renderer produces: a length-1 array wrapping the
    messages (``summarize_prompt``), and a length-K array of which ``index`` is
    taken (``summarize_prompts``). A cell that is already flat is returned as-is.
    """
    items = list(cell)
    if not items:
        raise ValueError("empty prompt cell")
    if isinstance(items[0], dict) and "role" in items[0]:
        return [dict(m) for m in items]  # already flat
    if index >= len(items):
        raise ValueError(f"--prompt-index {index} out of range for a length-{len(items)} cell")
    return [dict(m) for m in list(items[index])]


def read_prompt_id(path: Path) -> str | None:
    """Read the template name a parquet was rendered with, if it records one."""
    try:
        df = pd.read_parquet(path, columns=["prompt_id"])
    except Exception:
        return None
    ids = df["prompt_id"].dropna().unique()
    if len(ids) == 0:
        return None
    if len(ids) > 1:
        raise SystemExit(f"{path} mixes {len(ids)} template(s): {list(ids)}; it must use exactly one")
    return str(ids[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Rendered summarize parquet.")
    ap.add_argument("--output", type=Path, required=True, help="Destination parquet for evaluation.")
    ap.add_argument("--prompt-key", default="summarize_prompt",
                    help="Column holding the rendered rephrase prompt (default: %(default)s).")
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="Which entry to take from a length-K prompt column (default: %(default)s).")
    ap.add_argument("--output-key", default="prompt",
                    help="Column to write the flattened rephrase prompt into (default: %(default)s). "
                         "Evaluation reads this one by default.")
    ap.add_argument("--question-key", default="question_prompt",
                    help="Column to preserve the bare question in, flattened and ready to "
                         "evaluate (default: %(default)s). Use it to measure plain "
                         "problem-solving on the same questions.")
    ap.add_argument("--train-parquet", type=Path, default=None,
                    help="SFT parquet used for training. Its prompt_id must match this eval set's.")
    ap.add_argument("--allow-template-mismatch", action="store_true",
                    help="Proceed even when the eval and training prompt_id differ. Only for a "
                         "deliberate cross-prompt comparison.")
    ap.add_argument("--limit", type=int, default=None, help="Keep only the first N rows.")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    print(f"[eval] {len(df):,} rows from {args.input}")
    if args.prompt_key not in df.columns:
        raise SystemExit(
            f"no '{args.prompt_key}' column; available: {list(df.columns)}. "
            f"Render the eval set with Data/prepare_summarize_prompts.py first."
        )

    eval_id = read_prompt_id(args.input)
    print(f"[eval] prompt_id: {eval_id or '(not recorded)'}")
    if args.train_parquet is not None:
        train_id = read_prompt_id(args.train_parquet)
        print(f"[train] prompt_id: {train_id or '(not recorded)'}")
        if eval_id is None or train_id is None:
            print("[warn] cannot compare prompt_id -- one side does not record it; "
                  "verify by hand that both used the same --template")
        elif eval_id != train_id:
            message = (
                f"prompt mismatch: training used {train_id!r}, this eval set uses {eval_id!r}. "
                f"The accuracy would reflect the prompt change, not the model. Re-render the eval "
                f"set with --template {train_id}."
            )
            if not args.allow_template_mismatch:
                raise SystemExit(f"[error] {message}")
            print(f"[warn] {message}")
        else:
            print(f"[ok] training and evaluation both use {eval_id!r}")

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"[eval] limited to {len(df):,} rows")
    else:
        df = df.copy()

    # Both prompts are kept side by side so one file can measure two things: the
    # rephrase task, and the model's plain problem-solving ability on the same
    # questions. Pick the column at evaluation time (sweep_sft_checkpoints.sh
    # PROMPT_KEY). The bare question is flattened too, so either column can be fed
    # to generation without further processing.
    if "prompt" in df.columns:
        df[args.question_key] = [flatten_prompt(cell, 0) for cell in df["prompt"]]
        print(f"[eval] kept the bare question as '{args.question_key}' (flattened)")

    df[args.output_key] = [
        flatten_prompt(cell, args.prompt_index) for cell in df[args.prompt_key]
    ]

    roles = [m["role"] for m in df.iloc[0][args.output_key]]
    print(f"[eval] flattened '{args.prompt_key}'[{args.prompt_index}] -> '{args.output_key}', roles={roles}")
    if roles and roles[-1] == "assistant":
        raise SystemExit(
            "the flattened prompt ends with an assistant turn; generation expects the "
            "conversation to stop after the user turn"
        )
    chars = pd.Series(
        [len(m["content"]) for p in df[args.output_key] for m in p if m["role"] == "user"]
    )
    print(f"[eval] user-turn chars: p50={int(chars.median()):,} max={int(chars.max()):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[write] {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
