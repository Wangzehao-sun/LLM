"""Build a prefill-continuation eval set: the instruct model resumes a student's answer.

The question being asked: given the first R% of a wrong student attempt, does a stronger
instruct model steer it to a correct answer? The prefix must reach the model as text it
already appears to have written -- NOT as something described in the prompt -- so that the
continuation is a natural continuation rather than a reaction to an instruction.

That distinction is mechanical, not stylistic. A chat template renders

    [system, user, assistant(prefix)]      + add_generation_prompt=True
      -> ...<|im_start|>assistant\\nPREFIX<|im_end|>\\n<|im_start|>assistant\\n

which CLOSES the prefix as a finished turn and starts a fresh one: the model writes from
scratch and can see that it already "answered" once. What is needed instead is

    [system, user, assistant(prefix)]      + continue_final_message=True
      -> ...<|im_start|>assistant\\nPREFIX

with no end-of-turn token, so the cursor sits at the end of the prefix. The two flags are
mutually exclusive in transformers (passing both raises), which is why
verl/trainer/main_generation.py needed a switch -- see PREFILL_NOTE below.

This script only produces the data. It writes ``prompt`` as ``[system, user,
assistant(prefix)]`` and records what was cut, plus a ``question_prompt`` column holding
the bare ``[system, user]`` so the SAME file can measure the no-prefix baseline.

The prefix is cut in TOKEN space and then backed up to a sentence boundary. Both matter:

  * token space, because that is what "15% of the reasoning" means to the model, and
    because character counts are badly skewed in LaTeX-dense text (``\\frac`` is many
    characters and few tokens). It also matches how the rest of the repo defines prefixes
    (add_token_split_points.py, prepare_summarize_prompts.py).
  * sentence boundary, because an arbitrary cut lands mid-formula -- measured on
    self_rollout_wrong.parquet, 15% falls at ``e^{-(x - a)^2/2}\\n\\n### Step 2: Substi``.
    Resuming from a half-written word makes the model repair the fragment first, which
    contaminates what the experiment is trying to measure. Reuses
    ``_backup_to_sentence_boundary`` from prepare_summarize_prompts.py.

Usage:

    # 15% prefix, the default
    python Data/prepare_prefill_continue.py \\
        --input  Data/self_rollout_wrong.parquet \\
        --output Data/self_rollout_wrong_prefill15.parquet \\
        --tokenizer-path /home/data/shared/Qwen3-4B-Instruct

    # sweep several ratios into separate files
    for r in 0.10 0.15 0.30; do
        python Data/prepare_prefill_continue.py --input ... \\
            --ratio $r --output Data/prefill_$r.parquet --tokenizer-path ...
    done

    # --ratio 0 writes an empty prefix: the control group, same rows, no prefill
    python Data/prepare_prefill_continue.py --input ... --ratio 0 --output Data/prefill_00.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from prepare_summarize_prompts import _backup_to_sentence_boundary  # noqa: E402


def as_messages(cell) -> list[dict]:
    """Normalize a parquet ``prompt`` cell to a plain list of message dicts."""
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if not isinstance(cell, (list, tuple)):
        raise ValueError(f"prompt cell is not a message list: {type(cell).__name__}")
    out = []
    for m in cell:
        if isinstance(m, dict):
            out.append({"role": str(m["role"]), "content": str(m["content"])})
        elif hasattr(m, "dtype") and getattr(m.dtype, "names", None):
            out.append({"role": str(m["role"]), "content": str(m["content"])})
        else:
            raise ValueError(f"cannot read a message from {type(m).__name__}")
    return out


def _delimiters_balanced(text: str) -> bool:
    """True if no math/code block is left open.

    A sentence boundary is not enough on its own. ``_backup_to_sentence_boundary``
    accepts a newline, and in these traces a newline very often comes right after ``\\[``
    opening a display-math block -- measured on self_rollout_wrong.parquet, 13 of 128
    prefixes ended inside unclosed math. Resuming there makes the model finish someone
    else's broken formula, which is not the continuation being studied.
    """
    if text.count(r"\[") != text.count(r"\]"):
        return False
    if text.count("$$") % 2:
        return False
    # Single $ ... $, ignoring the $$ pairs already counted above.
    if len(re.findall(r"(?<!\$)\$(?!\$)", text)) % 2:
        return False
    return text.count("```") % 2 == 0


def _rewind_to_safe_boundary(prefix: str) -> str:
    """Rewind to a sentence boundary that also leaves math/code delimiters balanced.

    Repeatedly backs up: sentence boundary first (reusing the repo's existing helper), and
    if that still leaves a block open, keep rewinding to earlier boundaries. Returns the
    empty string when no safe boundary exists, so the caller can decide.
    """
    candidate = _backup_to_sentence_boundary(prefix)
    while candidate.strip():
        if _delimiters_balanced(candidate):
            return candidate
        shorter = _backup_to_sentence_boundary(candidate[:-1])
        if shorter == candidate:  # no progress; stop rather than spin
            break
        candidate = shorter
    return candidate if _delimiters_balanced(candidate) else ""


def cut_prefix(text: str, ratio: float, encode, decode, backup: bool) -> tuple[str, int, int]:
    """Return ``(prefix, kept_tokens, total_tokens)``.

    Cut in token space, then optionally rewind to a boundary that is safe to resume from.
    ``kept_tokens`` is measured AFTER the rewind, so the reported ratio is the one the
    model actually sees rather than the one that was requested.
    """
    ids = encode(text)
    total = len(ids)
    if total == 0 or ratio <= 0:
        return "", 0, total
    if ratio >= 1.0:
        return text, total, total

    cut = max(1, min(int(round(ratio * total)), total))
    prefix = decode(ids[:cut])
    if backup:
        backed = _rewind_to_safe_boundary(prefix)
        # Never let the rewind empty the prefix: a trace with no safe boundary in its first
        # R% would otherwise silently become a control row. Keeping the raw cut is the
        # lesser evil, and n_unsafe_prefix reports how often it happens.
        if backed.strip():
            prefix = backed
    prefix = prefix.rstrip()
    return prefix, len(encode(prefix)), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Parquet with prompt + the student attempt column.")
    ap.add_argument("--output", type=Path, required=True, help="Destination parquet.")
    ap.add_argument("--rollout-key", default="wrong_rollout",
                    help="Column holding the student attempt to resume (default: %(default)s). "
                         "collect_wrong_rollouts.py already picked one median-length wrong "
                         "attempt per question, so no sampling is needed here.")
    ap.add_argument("--ratio", type=float, default=0.15,
                    help="Fraction of the attempt's TOKENS to keep as the prefix "
                         "(default: %(default)s). 0 = no prefix, i.e. the control group.")
    ap.add_argument("--no-backup", dest="backup", action="store_false",
                    help="Cut exactly at the ratio instead of rewinding to a sentence "
                         "boundary. Exact ratios, but prefixes may end mid-formula.")
    ap.add_argument("--tokenizer-path", default=None,
                    help="Tokenizer for the token-space cut. Without it the cut falls back "
                         "to characters, which is skewed in LaTeX-dense text.")
    ap.add_argument("--limit", type=int, default=None, help="Keep only the first N rows.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    if not 0.0 <= args.ratio <= 1.0:
        raise SystemExit(f"--ratio must be in [0, 1], got {args.ratio}")
    if not args.input.is_file():
        raise SystemExit(f"input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    else:
        df = df.copy()
    print(f"[data] {len(df):,} rows from {args.input}")
    if args.rollout_key not in df.columns:
        raise SystemExit(f"no {args.rollout_key!r} column; available: {list(df.columns)}")
    if "prompt" not in df.columns:
        raise SystemExit(f"no 'prompt' column; available: {list(df.columns)}")

    encode = decode = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731
            decode = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
            print(f"[cut] token space, tokenizer: {args.tokenizer_path}")
        except Exception as error:
            print(f"[cut] could not load {args.tokenizer_path} ({error}); falling back to characters")
    if encode is None:
        # Character fallback: encode/decode as a list of characters so the same
        # ratio-then-rewind code path works unchanged.
        encode = list
        decode = "".join
        print("[cut] CHARACTER space (no tokenizer): ratios are approximate in LaTeX-dense text")

    prompts, question_prompts = [], []
    prefixes, kept_list, total_list = [], [], []
    n_empty_attempt = n_backup_shrunk = 0

    for position in range(len(df)):
        row = df.iloc[position]
        msgs = as_messages(row["prompt"])
        if msgs and msgs[-1]["role"] == "assistant":
            raise SystemExit(
                f"row {position}: 'prompt' already ends with an assistant turn; this script "
                f"expects the bare [system, user] conversation"
            )

        attempt = row[args.rollout_key]
        attempt = attempt.strip() if isinstance(attempt, str) else ""
        if not attempt:
            n_empty_attempt += 1

        requested = int(round(args.ratio * len(encode(attempt)))) if attempt else 0
        prefix, kept, total = cut_prefix(attempt, args.ratio, encode, decode, args.backup)
        if args.backup and requested > 0 and kept < requested:
            n_backup_shrunk += 1

        question_prompts.append(msgs)
        # The prefix becomes a trailing ASSISTANT message. main_generation renders it with
        # continue_final_message=True, which strips the end-of-turn token so generation
        # resumes inside this message instead of starting a new one.
        prompts.append(msgs + [{"role": "assistant", "content": prefix}] if prefix else msgs)
        prefixes.append(prefix)
        kept_list.append(kept)
        total_list.append(total)

    df["question_prompt"] = question_prompts
    df["prompt"] = prompts
    df["prefill_prefix"] = prefixes
    df["prefill_tokens"] = kept_list
    df["prefill_total_tokens"] = total_list
    df["prefill_ratio"] = [
        (k / t if t else 0.0) for k, t in zip(kept_list, total_list)
    ]
    df["prefill_ratio_requested"] = args.ratio

    unit = "tokens" if args.tokenizer_path and encode is not list else "chars"
    kept = pd.Series(kept_list)
    achieved = pd.Series(df["prefill_ratio"])
    n_with_prefix = int((kept > 0).sum())
    print(
        f"[cut] ratio requested={args.ratio:.3f}  achieved p50={achieved.median():.3f} "
        f"min={achieved.min():.3f} max={achieved.max():.3f}\n"
        f"[cut] prefix {unit}: p50={int(kept.median()):,} max={int(kept.max()):,}; "
        f"{n_with_prefix:,}/{len(df):,} rows have a prefix"
    )
    if n_backup_shrunk:
        print(f"[cut] {n_backup_shrunk:,} prefix(es) shortened by the sentence-boundary rewind "
              f"(that is the point: no mid-formula cuts)")
    if n_empty_attempt:
        print(f"[warn] {n_empty_attempt:,} row(s) had an empty {args.rollout_key!r}; they get no "
              f"prefix and are effectively control rows")
    if args.ratio > 0 and n_with_prefix == 0:
        raise SystemExit(
            f"--ratio {args.ratio} produced no prefixes at all -- check that "
            f"{args.rollout_key!r} actually holds text"
        )

    roles = [m["role"] for m in df.iloc[0]["prompt"]]
    print(f"[schema] prompt roles (row 0): {roles}")
    if args.ratio > 0 and roles[-1] != "assistant":
        print("[warn] row 0 has no prefix, so its prompt ends at the user turn; mixed rows are "
              "fine (continue_final_message only affects rows that end with an assistant turn)")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[write] {len(df):,} rows ({args.output.stat().st_size / 1e6:.1f} MB) -> {args.output}")
    print(
        "[next] evaluate with:\n"
        "         CKPT_DIR=... EVAL_PATH=" + str(args.output) + " \\\n"
        "             PREFILL=1 bash Myverl/examples/custom/sweep_prefill_continue.sh\n"
        "       PROMPT_KEY=question_prompt gives the no-prefix baseline on the same rows."
    )


if __name__ == "__main__":
    main()
