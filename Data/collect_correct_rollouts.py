"""Collect one correct rollout per question from step-named JSONL dumps.

First step of the "student's own correct rollout as SFT target" route: the
rephraser should learn to write inside the student's own distribution, so the SFT
target has to be a solution the student itself produced and got right.

Input is a directory of trainer rollout dumps named by step (``40.jsonl``,
``41.jsonl``, ...), each line one rollout with ``{input, output, score}``. A step
range selects which dumps to read. Every question with at least one correct
rollout qualifies; ``--pick`` decides which single rollout is kept.

Those dumps carry no expert reasoning, so questions are matched back to a base
parquet (which has ``prompt`` and ``target``) to recover it -- the same join
``attach_rollout_responses.py`` does, by normalized question text, because the
dumps carry no uid.

Output is one row per question, keeping the base parquet's schema and adding:

    rollout          : str  -- the chosen correct rollout (the SFT target)
    rollout_step     : int  -- which step it came from
    rollout_n_correct: int  -- how many correct rollouts that question had
    rollout_n_total  : int  -- how many rollouts it had in total
    rollout_pass_rate: float

Then render the rephrase prompts with ``Data/prepare_summarize_prompts.py`` and
pair prompt with target using ``Data/prepare_sft.py --target-key rollout``.

A rollout with no ``\\boxed{}`` is dropped by default: a correct-scored response
without one was cut off by the response-length cap mid-derivation and never states
a final answer, so it would teach the rephraser to trail off.

Usage:

    # steps 40..60, keep at most 1024 questions
    python Data/collect_correct_rollouts.py \\
        --rollout-dir   ~/Desktop/rollout_data/<folder> \\
        --input-parquet Data/deepmath_hard_solonly.parquet \\
        --output        Data/rollouts_correct.parquet \\
        --min-step 40 --max-step 60 --limit 1024

    # or name the steps explicitly; --pick chooses among a question's correct ones
    python Data/collect_correct_rollouts.py ... --steps 40,45,50 --pick shortest
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

# The boxed-answer instruction prefixing every question. The parquet's user turn
# carries it twice, a dump transcript once, so it is stripped before matching.
BOILERPLATE_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}."

# _dump_generations writes a flattened transcript: "system\n...\nuser\n...\nassistant\n".
_INPUT_RE = re.compile(r"^user\n(.*)\nassistant\n?$", re.S | re.M)


def normalize_question(text: str) -> str:
    """Strip the repeated boilerplate prefix and collapse whitespace.

    This is a join key, not a display string: everything else is preserved.
    """
    if not isinstance(text, str):
        return ""
    stripped = text.lstrip()
    while stripped.startswith(BOILERPLATE_PREFIX):
        stripped = stripped[len(BOILERPLATE_PREFIX) :].lstrip()
    return " ".join(stripped.split())


def question_from_transcript(text: Any) -> str:
    """Pull the user turn out of a dump row's flattened ``input``."""
    if not isinstance(text, str):
        return ""
    match = _INPUT_RE.search(text)
    return match.group(1) if match else ""


def parquet_question(prompt: Any) -> str:
    """Take the last user turn of a parquet ``prompt`` as the question text."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, (list, tuple)):
        return ""
    contents = [message.get("content", "") for message in prompt if isinstance(message, dict) and message.get("role") == "user"]
    return normalize_question(contents[-1]) if contents else ""


def select_dump_files(
    rollout_dir: Path,
    steps: str | None,
    min_step: int | None,
    max_step: int | None,
) -> list[tuple[int, Path]]:
    """Pick the step-named dumps to read, ordered by step.

    Files are expected to be named ``<step>.jsonl``. A stem that is not an integer
    is reported and skipped rather than silently ignored, since a stray filename
    would otherwise quietly shrink the dataset.
    """
    if not rollout_dir.is_dir():
        raise SystemExit(f"--rollout-dir is not a directory: {rollout_dir}")

    found: dict[int, Path] = {}
    non_numeric: list[str] = []
    for path in sorted(rollout_dir.glob("*.jsonl")):
        try:
            found[int(path.stem)] = path
        except ValueError:
            non_numeric.append(path.name)
    if non_numeric:
        print(f"[scan] skipped {len(non_numeric)} non-step file(s): {', '.join(non_numeric[:5])}")
    if not found:
        raise SystemExit(f"no <step>.jsonl files in {rollout_dir}")

    if steps:
        wanted: list[int] = []
        for token in steps.replace(" ", "").split(","):
            if not token:
                continue
            try:
                wanted.append(int(token))
            except ValueError:
                raise SystemExit(f"--steps expects comma-separated integers, got {token!r}") from None
        missing = [s for s in wanted if s not in found]
        if missing:
            raise SystemExit(f"--steps names step(s) with no file: {missing}; available: {min(found)}..{max(found)}")
        selected = [(s, found[s]) for s in sorted(set(wanted))]
    else:
        selected = [(s, p) for s, p in sorted(found.items()) if (min_step is None or s >= min_step) and (max_step is None or s <= max_step)]
        if not selected:
            raise SystemExit(f"no step in [{min_step}, {max_step}] among available {min(found)}..{max(found)}")

    print(f"[scan] {len(found)} dump(s) in {rollout_dir} (steps {min(found)}..{max(found)}); using {len(selected)}: {', '.join(str(s) for s, _ in selected)}")
    return selected


def load_rollouts(
    files: list[tuple[int, Path]],
    input_field: str,
    output_field: str,
    score_field: str,
    correct_score: float,
    require_boxed: bool,
) -> tuple[OrderedDict[str, dict[str, Any]], dict[str, int]]:
    """Group rollouts by normalized question, in (step, line) order.

    Returns ``{question: {"correct": [...], "n_total": int}}``. ``n_total`` counts
    every scored rollout for the question, so the pass rate reported later is over
    the real group size rather than only the kept ones.

    Insertion order is preserved so ``--pick`` ties break deterministically:
    earliest step first, and within a step the earliest line.
    """
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    stats = {
        "rows": 0,
        "correct": 0,
        "bad_json": 0,
        "no_question": 0,
        "no_output": 0,
        "bad_score": 0,
        "unboxed": 0,
    }

    for step, path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                stats["rows"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                question = normalize_question(question_from_transcript(record.get(input_field)))
                if not question:
                    stats["no_question"] += 1
                    continue
                score = record.get(score_field)
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    stats["bad_score"] += 1
                    continue

                entry = grouped.setdefault(question, {"correct": [], "n_total": 0})
                entry["n_total"] += 1
                if float(score) != correct_score:
                    continue

                output = record.get(output_field)
                if not isinstance(output, str) or not output.strip():
                    stats["no_output"] += 1
                    continue
                output = output.strip()
                if require_boxed and "\\boxed" not in output:
                    # Scored correct but never states a final answer: the
                    # response-length cap truncated it mid-derivation.
                    stats["unboxed"] += 1
                    continue
                stats["correct"] += 1
                entry["correct"].append({"output": output, "step": step})

    return grouped, stats


def pick_rollout(candidates: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    """Choose ONE correct rollout to be the SFT target.

    Selection is by output length, which is the axis that matters for the
    rephraser: it sets how verbose the SFT'd model becomes.

      * ``median``   -- avoids both extremes, so the target style sits closest to
        the student's typical output. Default.
      * ``shortest`` -- mirrors the trainer's summarize_replace path, which scans
        candidates and takes the shortest correct one. Biases toward concise,
        low-detour solutions.
      * ``longest``  -- keeps the most detailed reasoning.
      * ``first``    -- earliest step, then earliest line; no length preference.
    """
    if strategy == "first":
        return candidates[0]
    ordered = sorted(candidates, key=lambda c: len(c["output"]))
    if strategy == "shortest":
        return ordered[0]
    if strategy == "longest":
        return ordered[-1]
    if strategy == "median":
        return ordered[(len(ordered) - 1) // 2]
    raise ValueError(f"unknown --pick strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout-dir", type=Path, required=True, help="Directory of <step>.jsonl trainer rollout dumps.")
    parser.add_argument("--input-parquet", type=Path, required=True, help="Base parquet with prompt / target, supplying the expert reasoning.")
    parser.add_argument("--output", type=Path, required=True, help="Destination parquet.")
    parser.add_argument("--min-step", type=int, default=None, help="Lowest step to include (inclusive).")
    parser.add_argument("--max-step", type=int, default=None, help="Highest step to include (inclusive).")
    parser.add_argument("--steps", default=None, help="Comma-separated step numbers, e.g. '40,45,50'. Overrides --min-step/--max-step.")
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N questions (= N rows). Default: all qualifying.")
    parser.add_argument("--pick", choices=["median", "shortest", "longest", "first"], default="median", help="Which of a question's correct rollouts becomes the target, by output length (default: %(default)s).")
    parser.add_argument("--input-field", default="input", help="JSONL transcript field (default: %(default)s).")
    parser.add_argument("--output-field", default="output", help="JSONL response field (default: %(default)s).")
    parser.add_argument("--score-field", default="score", help="JSONL score field (default: %(default)s).")
    parser.add_argument("--correct-score", type=float, default=1.0, help="Score counted as correct (default: %(default)s).")
    parser.add_argument("--allow-unboxed", action="store_true", help="Keep correct rollouts with no \\boxed{}. Off by default: those were length-capped mid-derivation and state no final answer.")
    parser.add_argument("--rollout-key", default="rollout", help="Column name for the chosen rollout (default: %(default)s). Pass this to prepare_sft.py as --target-key.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if not args.input_parquet.is_file():
        raise SystemExit(f"base parquet not found: {args.input_parquet}")

    files = select_dump_files(args.rollout_dir, args.steps, args.min_step, args.max_step)
    grouped, stats = load_rollouts(
        files,
        args.input_field,
        args.output_field,
        args.score_field,
        args.correct_score,
        not args.allow_unboxed,
    )
    qualifying = {q: e for q, e in grouped.items() if e["correct"]}
    print(f"[rollout] {stats['rows']:,} rows read over {len(grouped):,} question(s); {stats['correct']:,} correct rollout(s) over {len(qualifying):,} question(s)")
    for name, label in (
        ("unboxed", "correct but no \\boxed{}"),
        ("no_output", "empty output"),
        ("no_question", "unparseable transcript"),
        ("bad_score", "unusable score"),
        ("bad_json", "malformed line"),
    ):
        if stats[name]:
            print(f"           skipped {stats[name]:,} ({label})")
    if not qualifying:
        raise SystemExit("no question has a correct rollout; nothing to build")

    df = pd.read_parquet(args.input_parquet)
    if "prompt" not in df.columns:
        raise SystemExit("base parquet has no 'prompt' column; cannot match by question")
    if "target" not in df.columns:
        print("[warn] base parquet has no 'target' column -- prepare_summarize_prompts.py needs it for the expert-reasoning draft")
    print(f"[parquet] {len(df):,} rows from {args.input_parquet}")

    keep_rows: list[int] = []
    chosen: list[dict[str, Any]] = []
    matched: set[str] = set()
    duplicate_rows = 0

    for position, prompt in enumerate(df["prompt"]):
        if args.limit is not None and len(keep_rows) >= args.limit:
            break
        question = parquet_question(prompt)
        entry = qualifying.get(question)
        if entry is None:
            continue
        if question in matched:
            # The base parquet repeats this question; keep the first row only.
            duplicate_rows += 1
            continue
        matched.add(question)
        keep_rows.append(position)
        best = pick_rollout(entry["correct"], args.pick)
        chosen.append(
            {
                "output": best["output"],
                "step": best["step"],
                "n_correct": len(entry["correct"]),
                "n_total": entry["n_total"],
            }
        )

    if not keep_rows:
        raise SystemExit("no parquet row matched a rollout question; check that the dumps and the base parquet describe the same dataset")

    out_df = df.iloc[keep_rows].reset_index(drop=True)
    out_df[args.rollout_key] = [c["output"] for c in chosen]
    out_df["rollout_step"] = [c["step"] for c in chosen]
    out_df["rollout_n_correct"] = [c["n_correct"] for c in chosen]
    out_df["rollout_n_total"] = [c["n_total"] for c in chosen]
    out_df["rollout_pass_rate"] = [c["n_correct"] / c["n_total"] for c in chosen]

    lengths = pd.Series([len(c["output"]) for c in chosen])
    n_correct = pd.Series([c["n_correct"] for c in chosen])
    unmatched = [q for q in qualifying if q not in matched]
    print(
        f"[build] {len(out_df):,} question(s), one {args.rollout_key} each (--pick {args.pick})\n"
        f"        target chars:  p50={int(lengths.median()):,} max={int(lengths.max()):,}\n"
        f"        correct/question among those kept: min={n_correct.min()} "
        f"p50={int(n_correct.median())} max={n_correct.max()}\n"
        f"        steps used: {sorted({c['step'] for c in chosen})}"
    )
    if duplicate_rows:
        print(f"[warn] skipped {duplicate_rows:,} duplicate question row(s) in the base parquet")
    if unmatched:
        print(f"[warn] {len(unmatched):,} qualifying question(s) matched no parquet row, e.g.:")
        for question in unmatched[:3]:
            print(f"         {question[:100]}")
    if args.limit is not None and len(out_df) == args.limit and unmatched:
        print(f"[note] stopped at --limit {args.limit}; more questions were available")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    size_mb = args.output.stat().st_size / 1e6
    print(f"[write] {len(out_df):,} rows ({size_mb:.1f} MB) -> {args.output}")


if __name__ == "__main__":
    main()
