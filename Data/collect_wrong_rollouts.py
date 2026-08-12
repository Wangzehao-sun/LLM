"""Collect questions the model gets ENTIRELY wrong, with one representative attempt.

Companion to ``collect_correct_rollouts.py``: that one keeps questions with at
least one correct rollout, this one keeps questions with NONE. The result is an
error set -- the problems the model cannot solve at all -- plus one of its wrong
attempts per question, which is what a downstream repair/teacher prompt needs as
the target model's attempt.

"Entirely wrong" is judged over the SELECTED STEP RANGE, not per file. A question
solved at step 50 but missed at step 40 is NOT in the error set when both steps are
loaded, because the model demonstrably can solve it. Widening the range therefore
shrinks the set; narrow it to ask "what could the model not do at this point in
training".

Input is a directory of trainer rollout dumps named by step (``40.jsonl``,
``41.jsonl``, ...), each line one rollout with ``{input, output, score}``. Those
dumps carry no expert reasoning, so questions are matched back to a base parquet
(which has ``prompt`` and ``target``) by normalized question text, since the dumps
carry no uid.

Output is one row per question, keeping the base parquet's schema and adding:

    wrong_rollout       : str  -- the chosen wrong attempt
    wrong_rollout_step  : int  -- which step it came from
    wrong_n_total       : int  -- how many rollouts the question had (all wrong)

Usage:

    # steps 40..60, every question missed throughout
    python Data/collect_wrong_rollouts.py \\
        --rollout-dir   ~/rollout_data/<folder> \\
        --input-parquet Data/deepmath_hard_solonly.parquet \\
        --output        Data/rollouts_wrong.parquet \\
        --min-step 40 --max-step 60 --limit 1024

    # --pick chooses among a question's wrong attempts (median length by default)
    python Data/collect_wrong_rollouts.py ... --steps 40,45,50 --pick shortest
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
    would otherwise quietly change which questions look "entirely wrong".
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

    Returns ``{question: {"wrong": [...], "n_correct": int, "n_total": int}}``.
    ``n_correct`` is counted over EVERY scored rollout, before any boxed filtering,
    so a question is only called entirely-wrong when the model really never solved
    it -- not merely when its correct attempt happened to be truncated.

    Insertion order is preserved so ``--pick`` ties break deterministically:
    earliest step first, and within a step the earliest line.
    """
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    stats = {
        "rows": 0,
        "wrong": 0,
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

                entry = grouped.setdefault(question, {"wrong": [], "n_correct": 0, "n_total": 0})
                entry["n_total"] += 1
                if float(score) == correct_score:
                    entry["n_correct"] += 1
                    continue

                output = record.get(output_field)
                if not isinstance(output, str) or not output.strip():
                    stats["no_output"] += 1
                    continue
                output = output.strip()
                if require_boxed and "\\boxed" not in output:
                    # No final answer at all: the response-length cap truncated it
                    # mid-derivation. That is an incomplete attempt rather than a
                    # wrong one, and reads oddly as "the model's answer".
                    stats["unboxed"] += 1
                    continue
                stats["wrong"] += 1
                entry["wrong"].append({"output": output, "step": step})

    return grouped, stats


def pick_rollout(candidates: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    """Choose ONE wrong rollout to represent the question.

    Selection is by output length, the axis that decides how typical the kept
    attempt is:

      * ``median``   -- the middle-length attempt, closest to what the model
        usually produces on this question. Default.
      * ``shortest`` -- often the attempt that gave up earliest.
      * ``longest``  -- often the one that rambled or looped.
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
    parser.add_argument(
        "--pick",
        choices=["median", "shortest", "longest", "first"],
        default="median",
        help="Which of a question's wrong rollouts to keep, by output length (default: %(default)s).",
    )
    parser.add_argument("--min-rollouts", type=int, default=1, help="Require at least N rollouts for the question before calling it entirely wrong (default: %(default)s).")
    parser.add_argument("--input-field", default="input", help="JSONL transcript field (default: %(default)s).")
    parser.add_argument("--output-field", default="output", help="JSONL response field (default: %(default)s).")
    parser.add_argument("--score-field", default="score", help="JSONL score field (default: %(default)s).")
    parser.add_argument("--correct-score", type=float, default=1.0, help="Score counted as correct (default: %(default)s).")
    parser.add_argument(
        "--allow-unboxed",
        action="store_true",
        help="Keep wrong rollouts with no \\boxed{}. Off by default: those were length-capped mid-derivation and state no answer at all.",
    )
    parser.add_argument("--rollout-key", default="wrong_rollout", help="Column name for the chosen wrong rollout (default: %(default)s).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.min_rollouts < 1:
        raise SystemExit("--min-rollouts must be >= 1")
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

    # Entirely wrong = never solved anywhere in the selected range, and at least one
    # usable wrong attempt survives to represent it.
    solved = {q for q, e in grouped.items() if e["n_correct"] > 0}
    qualifying = {q: e for q, e in grouped.items() if q not in solved and e["wrong"] and e["n_total"] >= args.min_rollouts}
    too_few = sum(1 for q, e in grouped.items() if q not in solved and e["wrong"] and e["n_total"] < args.min_rollouts)
    no_usable = sum(1 for q, e in grouped.items() if q not in solved and not e["wrong"])

    print(f"[rollout] {stats['rows']:,} rows over {len(grouped):,} question(s); {len(solved):,} solved at least once, {len(grouped) - len(solved):,} never solved")
    print(f"[rollout] {len(qualifying):,} question(s) qualify ({stats['wrong']:,} wrong rollout(s) kept)")
    if no_usable:
        print(f"           {no_usable:,} never-solved question(s) had no usable wrong attempt (all filtered out)")
    if too_few:
        print(f"           {too_few:,} never-solved question(s) had fewer than --min-rollouts {args.min_rollouts} rollouts")
    for name, label in (
        ("unboxed", "wrong but no \\boxed{}"),
        ("no_output", "empty output"),
        ("no_question", "unparseable transcript"),
        ("bad_score", "unusable score"),
        ("bad_json", "malformed line"),
    ):
        if stats[name]:
            print(f"           skipped {stats[name]:,} ({label})")
    if not qualifying:
        raise SystemExit("no question is entirely wrong across the selected steps; nothing to build")

    df = pd.read_parquet(args.input_parquet)
    if "prompt" not in df.columns:
        raise SystemExit("base parquet has no 'prompt' column; cannot match by question")
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
        best = pick_rollout(entry["wrong"], args.pick)
        chosen.append({"output": best["output"], "step": best["step"], "n_total": entry["n_total"]})

    if not keep_rows:
        raise SystemExit("no parquet row matched an entirely-wrong question; check that the dumps and the base parquet describe the same dataset")

    out_df = df.iloc[keep_rows].reset_index(drop=True)
    out_df[args.rollout_key] = [c["output"] for c in chosen]
    out_df["wrong_rollout_step"] = [c["step"] for c in chosen]
    out_df["wrong_n_total"] = [c["n_total"] for c in chosen]

    lengths = pd.Series([len(c["output"]) for c in chosen])
    print(
        f"[build] {len(out_df):,} entirely-wrong question(s), one {args.rollout_key} each (--pick {args.pick})\n"
        f"        attempt chars: p50={int(lengths.median()):,} max={int(lengths.max()):,}\n"
        f"        steps used: {sorted({c['step'] for c in chosen})}"
    )
    if duplicate_rows:
        print(f"[warn] skipped {duplicate_rows:,} duplicate question row(s) in the base parquet")
    unmatched = [q for q in qualifying if q not in matched]
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
