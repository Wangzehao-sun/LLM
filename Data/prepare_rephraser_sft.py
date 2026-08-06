"""Build rephraser SFT data whose target is the STUDENT's own correct rollout.

Motivation
----------
The online rephraser (``teacher_api`` / summarize path) rewrites an expert
reasoning draft into a full solution that is then spliced back into the student's
rollout group. Its output is scored by the student's own log-probs, so the closer
the rephraser writes to the student's natural distribution, the smaller the
importance ratio ``exp(logp_short - logp_long)`` variance. SFT-ing the rephraser
on the *student's own correct rollouts* is a direct way to pull it there.

So, per question::

    input  = summarize/rephraser prompt(question, expert reasoning prefix)
    output = one CORRECT student rollout for that same question

The expert reasoning comes from the source parquet's ``target`` column (same
extraction + answer-truncation as ``prepare_summarize_prompts.py``, so no prefix
can leak the final answer). The correct rollouts come from a trainer rollout dump
JSONL (``_dump_generations`` output, e.g. ``rollout_data/80.jsonl``), whose rows
are ``{input, output, score, step}``.

Because that dump does NOT carry a uid/original_index, rows are joined back to the
parquet by NORMALIZED question text: the boxed-answer instruction is stripped (the
parquet often carries it twice, the dump once) and whitespace collapsed. The join
is verified to be exact -- any unmatched question is reported, and ``--require-all``
turns that into a hard error.

Output schema
-------------
The source parquet's columns, plus the three columns a rephraser-SFT parquet needs
(byte-identical layout to the reference
``deepmath_hard_thinkonly1024_summarize0_5_sft.parquet``):

    summarize_prompts : list[struct] -- a FLAT [system, user] messages list (the
                        rephraser prompt). NOTE: this shadows the length-K column
                        of the same name in the RL parquets; here it is one
                        messages list, matching the reference SFT file.
    output            : str          -- the chosen correct student rollout
    messages          : list[struct] -- summarize_prompts + [assistant: output],
                        i.e. [system, user, assistant], what MultiTurnSFTDataset
                        consumes (loss masked to the assistant turn).

One row per question by default (``--per-question 1``), like the reference file.
With ``--per-question N > 1`` a question yields up to N rows that share the same
``summarize_prompts`` but carry different ``output``s.

Usage
-----
    python Data/prepare_rephraser_sft.py \
        --rollout  /Users/zenohaoz/Desktop/rollout_data/80.jsonl \
        --parquet  Data/deepmath_hard_solonly_split_summarize_teacher.parquet \
        --output   Data/deepmath_hard_rephraser_sft_step80.parquet

    # --template takes a built-in name OR a path to a template text file (it must
    # contain {question} and {prefix}), so the prompt design can be swapped freely
    python Data/prepare_rephraser_sft.py \
        --rollout rollout_data/60.jsonl rollout_data/80.jsonl \
        --parquet Data/deepmath_hard_solonly_split_summarize_teacher.parquet \
        --output  Data/rephraser_sft.parquet \
        --template my_rephraser_template.txt \
        --pick median --per-question 2 --draft-ratio 0.5

    # inspect what would be built without writing anything
    python Data/prepare_rephraser_sft.py --rollout ... --parquet ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# Reuse the offline prompt machinery so the input side stays byte-identical to
# what prepare_summarize_prompts.py would render (same templates, same
# answer-truncation, same sentence-boundary backup).
from prepare_summarize_prompts import (  # noqa: E402
    DEFAULT_TEMPLATE,
    DEFAULT_TEMPLATE1,
    DEFAULT_TEMPLATE2,
    DEFAULT_TEMPLATE3,
    TEACHER_TEMPLATE_DEFAULT,
    TEACHER_TEMPLATE_DEFAULT1,
    _backup_to_sentence_boundary,
    _build_summarize_prompt,
    _extract_question,
    _extract_think_process,
    _system_message,
    _truncate_before_final_answer,
)

# The built-in templates, shared with prepare_summarize_prompts.py. ``--template``
# also accepts a PATH to a text file, so a template that lives outside this repo
# (e.g. the one the reference SFT parquet was rendered with, which is not any of
# these) can be used without editing code. A custom template must contain the
# {question} and {prefix} placeholders; see load_template.
TEMPLATES: Dict[str, str] = {
    "default": DEFAULT_TEMPLATE,
    "template1": DEFAULT_TEMPLATE1,
    "template2": DEFAULT_TEMPLATE2,
    "template3": DEFAULT_TEMPLATE3,
    "teacher": TEACHER_TEMPLATE_DEFAULT,
    "teacher1": TEACHER_TEMPLATE_DEFAULT1,
}

# The boxed-answer instruction that prefixes every question. The dump carries it
# once, the parquet sometimes twice (a known double-prefix in the source data), so
# it must be stripped before matching -- see _norm_question.
BOXED_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."

# _dump_generations writes the flattened chat text: "system\n...\nuser\n...\nassistant\n".
_INPUT_RE = re.compile(r"^user\n(.*)\nassistant\n?$", re.S | re.M)


# ---------------------------------------------------------------------------
# rollout dump parsing
# ---------------------------------------------------------------------------

def _norm_question(q: str) -> str:
    """Normalize a question for cross-source matching.

    The dump's flattened prompt and the parquet's ``prompt`` message differ in how
    many times the boxed instruction is prefixed and in incidental whitespace, so
    both are stripped. Everything else is preserved -- this is a join key, not a
    display string.
    """
    return " ".join(q.replace(BOXED_INSTR, " ").split())


def _question_from_dump_input(text: str) -> str | None:
    """Pull the user turn out of a dump row's flattened ``input``."""
    m = _INPUT_RE.search(text or "")
    return m.group(1) if m else None


def load_correct_rollouts(paths: Sequence[Path], score_threshold: float) -> Dict[str, List[Dict[str, Any]]]:
    """Read rollout dumps and group the CORRECT rollouts by normalized question.

    Returns ``{normalized_question: [row, ...]}``. Rows keep their ``output`` /
    ``score`` / ``step`` plus the source file, so selection and reporting can use
    them. Malformed lines and rows whose ``input`` does not parse are counted and
    reported rather than crashing the run.
    """
    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_total = n_correct = n_bad_json = n_bad_input = 0
    all_questions: set[str] = set()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_bad_json += 1
                    continue
                question = _question_from_dump_input(row.get("input", ""))
                if question is None:
                    n_bad_input += 1
                    continue
                key = _norm_question(question)
                all_questions.add(key)
                # score: 1.0 correct / 0.0 wrong / -1.0 format error.
                if float(row.get("score", 0.0)) < score_threshold:
                    continue
                out = str(row.get("output", "") or "")
                if not out.strip():
                    continue
                n_correct += 1
                by_question[key].append(
                    {"output": out, "score": float(row["score"]), "step": row.get("step"), "src": path.name}
                )

    print(
        f"[rollout] {len(paths)} file(s): {n_total} rows, {len(all_questions)} unique questions, "
        f"{n_correct} correct rollouts over {len(by_question)} questions"
    )
    if n_bad_json or n_bad_input:
        print(f"[rollout] skipped {n_bad_json} unparseable line(s), {n_bad_input} row(s) with unrecognized input format")
    return by_question


def pick_rollouts(cands: List[Dict[str, Any]], strategy: str, per_question: int) -> List[Dict[str, Any]]:
    """Choose up to ``per_question`` rollouts out of one question's correct set.

    Selection is by output LENGTH, which is the axis that matters for the
    rephraser: it controls how verbose the SFT'd rephraser becomes.

      * ``shortest`` -- mirrors the trainer's summarize_replace path, which scans
        candidates and takes the shortest correct one. Biases toward concise,
        low-detour solutions.
      * ``longest``  -- keeps the most detailed reasoning.
      * ``median``   -- avoids both extremes, so the target style sits closest to
        the student's typical output.
      * ``all``      -- length-ascending, no length preference.

    When ``per_question > 1`` the extra picks walk outward from the chosen one in
    the length-sorted list, so a 2-pick ``median`` yields the two most typical
    candidates rather than one typical and one extreme.
    """
    if not cands:
        return []
    ordered = sorted(cands, key=lambda c: len(c["output"]))
    if strategy == "all":
        return ordered[:per_question] if per_question > 0 else ordered
    if strategy == "shortest":
        ranked = ordered
    elif strategy == "longest":
        ranked = list(reversed(ordered))
    elif strategy == "median":
        mid = (len(ordered) - 1) // 2
        # Walk outward from the median index: mid, mid+1, mid-1, mid+2, ...
        idx: List[int] = []
        for off in range(len(ordered)):
            for cand_i in ((mid + (off + 1) // 2) if off % 2 == 0 else (mid - (off + 1) // 2),):
                if 0 <= cand_i < len(ordered) and cand_i not in idx:
                    idx.append(cand_i)
        ranked = [ordered[i] for i in idx]
    else:
        raise ValueError(f"unknown --pick strategy: {strategy}")
    return ranked[:per_question] if per_question > 0 else ranked


# ---------------------------------------------------------------------------
# prompt rendering
# ---------------------------------------------------------------------------

def build_draft(item: Dict[str, Any], ratio: float) -> str:
    """Cut the expert reasoning down to a ``ratio`` prefix, answer-free.

    Two steps, both borrowed from prepare_summarize_prompts.py so the rendered
    prompt matches the RL-time one:

    1. ``_truncate_before_final_answer`` drops the answer-revealing tail (the
       earliest ``\\boxed{}`` matching the final answer). If it returns empty, the
       answer is revealed in the very first sentence and there is no safe prefix --
       the caller must skip the row.
    2. The remainder is cut at ``ratio`` in CHARACTER space and backed up to the
       last sentence boundary. Character space (not token space) is deliberate: it
       reproduces the reference SFT parquet byte-for-byte and needs no tokenizer.
    """
    reasoning = _truncate_before_final_answer(_extract_think_process(item))
    if not reasoning:
        return ""
    if ratio >= 1.0:
        return reasoning
    cut = max(0, min(int(round(ratio * len(reasoning))), len(reasoning)))
    return _backup_to_sentence_boundary(reasoning[:cut]).strip()


def load_template(spec: str) -> tuple[str, bool]:
    """Resolve ``--template`` to ``(template_text, is_teacher)``.

    ``spec`` is either a key of ``TEMPLATES`` or a path to a UTF-8 text file
    holding a template. A file lets you use a template that does not live in this
    repo (the reference SFT parquet's template, for instance, matches none of the
    built-ins) without touching code.

    A custom template must contain ``{question}`` and ``{prefix}``. It is rendered
    through the teacher path (targeted str.replace) whenever it also contains a
    ``{style_example_*}`` placeholder or a literal ``\\boxed{}``, because
    str.format would raise on those; otherwise through str.format like the
    built-in summarize templates.
    """
    if spec in TEMPLATES:
        return TEMPLATES[spec], spec.startswith("teacher")

    path = Path(spec)
    if not path.exists():
        raise SystemExit(
            f"--template {spec!r} is neither a built-in ({', '.join(sorted(TEMPLATES))}) "
            f"nor an existing file"
        )
    text = path.read_text(encoding="utf-8")
    missing = [p for p in ("{question}", "{prefix}") if p not in text]
    if missing:
        raise SystemExit(f"custom template {path} is missing required placeholder(s): {', '.join(missing)}")
    # Literal \boxed{} / {style_example_N} braces are invalid str.format fields, so
    # such templates must take the replace-based render path.
    is_teacher = "{style_example_" in text or "\\boxed{}" in text
    print(f"[template] loaded custom template from {path} ({len(text)} chars, render={'replace' if is_teacher else 'format'})")
    return text, is_teacher


def _as_object_array(items: Sequence[Any]) -> np.ndarray:
    """Pack a list into a 1-D object ndarray.

    ``np.array(list_of_dicts, dtype=object)`` would infer a 2-D shape for
    uniform-key dicts, and pyarrow only accepts 1-D object columns ("Only 1D
    arrays accepted"). Filling an empty object array keeps each cell a dict so the
    column serializes as ``list[struct]``.
    """
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    return arr


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rollout", type=Path, nargs="+", required=True,
                   help="One or more rollout-dump JSONL files ({input,output,score,step} per line).")
    p.add_argument("--parquet", type=Path, required=True,
                   help="Source parquet supplying the questions and the expert reasoning (`target`).")
    p.add_argument("--output", type=Path, default=None,
                   help="Destination parquet. Defaults to <parquet_stem>_rephraser_sft.parquet next to --parquet.")
    p.add_argument("--template", default="template2",
                   help="Rephraser prompt template: a built-in name (%s) or a path to a "
                        "text file containing {question} and {prefix} (default: %%(default)s)."
                        % ", ".join(sorted(TEMPLATES)))
    p.add_argument("--draft-ratio", type=float, default=0.5,
                   help="Fraction of the answer-truncated expert reasoning to expose as the draft (default: %(default)s).")
    p.add_argument("--pick", choices=["median", "shortest", "longest", "all"], default="median",
                   help="Which correct rollout(s) become the SFT target, by output length (default: %(default)s).")
    p.add_argument("--per-question", type=int, default=1,
                   help="Max rows per question; >1 shares the prompt across different outputs (default: %(default)s).")
    p.add_argument("--score-threshold", type=float, default=1.0,
                   help="Minimum rollout score to count as correct (default: %(default)s).")
    p.add_argument("--min-output-chars", type=int, default=0,
                   help="Drop candidate rollouts shorter than this many characters (default: %(default)s = off).")
    p.add_argument("--allow-unboxed", action="store_true",
                   help="Keep correct rollouts that contain no \\boxed{}. Off by default: such a "
                        "rollout was cut off by the response-length cap mid-derivation and only "
                        "scored 1.0 incidentally, so it is a truncated, unusable SFT target.")
    p.add_argument("--require-all", action="store_true",
                   help="Fail if any dump question cannot be matched to a --parquet row.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be built; write nothing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.draft_ratio <= 1.0:
        raise ValueError(f"--draft-ratio must be in (0, 1], got {args.draft_ratio}")
    if args.per_question < 1:
        raise ValueError(f"--per-question must be >= 1, got {args.per_question}")
    for path in args.rollout:
        if not path.exists():
            raise FileNotFoundError(f"rollout dump not found: {path}")

    template, is_teacher = load_template(args.template)
    if "{style_example_" in template:
        # The teacher templates carry a {style_example_1} placeholder that the
        # trainer fills online with the student's own INCORRECT rollout. Offline we
        # have no such slot in the SFT prompt, so it would survive into the training
        # data as a literal placeholder.
        print(
            f"[warn] template '{args.template}' contains {{style_example_1}}, which is filled at "
            "RL time with the student's incorrect attempt. It stays UNFILLED here -- the SFT "
            "prompt will contain the literal placeholder."
        )

    by_question = load_correct_rollouts(args.rollout, args.score_threshold)
    if not by_question:
        raise SystemExit("no correct rollouts found; nothing to build")

    df = pd.read_parquet(args.parquet)
    print(f"[parquet] {len(df)} rows from {args.parquet}, columns={list(df.columns)}")

    # Map normalized question -> first parquet row index carrying it. The dump has
    # no uid/original_index, so question text is the only available join key.
    q_to_row: Dict[str, int] = {}
    for i, prompt in enumerate(df["prompt"]):
        key = _norm_question(_extract_question({"prompt": prompt}))
        if key:
            q_to_row.setdefault(key, i)

    rows: List[Dict[str, Any]] = []
    n_unmatched = n_no_draft = n_no_candidate = n_unboxed = 0
    unmatched_examples: List[str] = []

    for key, cands in by_question.items():
        row_i = q_to_row.get(key)
        if row_i is None:
            n_unmatched += 1
            if len(unmatched_examples) < 3:
                unmatched_examples.append(key[:120])
            continue

        src = df.iloc[row_i]
        item = {"prompt": src["prompt"], "target": src["target"]}
        draft = build_draft(item, args.draft_ratio)
        if not draft:
            # No answer-free prefix exists for this question (the reference solution
            # reveals its result in the first sentence). Rendering a prompt with an
            # empty draft would train the rephraser on a degenerate input.
            n_no_draft += 1
            continue

        pool = cands
        if args.min_output_chars > 0:
            pool = [c for c in pool if len(c["output"]) >= args.min_output_chars]
        if not args.allow_unboxed:
            # A correct-scored rollout with no \boxed{} was truncated by the
            # response-length cap mid-derivation; it never states a final answer, so
            # it would teach the rephraser to trail off.
            kept = [c for c in pool if "\\boxed" in c["output"]]
            n_unboxed += len(pool) - len(kept)
            pool = kept
        picks = pick_rollouts(pool, args.pick, args.per_question)
        if not picks:
            n_no_candidate += 1
            continue

        messages = _build_summarize_prompt(
            _system_message(item), _extract_question(item), draft, template, is_teacher=is_teacher
        )
        prompt_arr = _as_object_array(messages)

        for cand in picks:
            out = cand["output"]
            new_row = src.to_dict()
            new_row["summarize_prompts"] = prompt_arr
            new_row["output"] = out
            new_row["messages"] = _as_object_array(
                list(messages) + [{"role": "assistant", "content": out}]
            )
            rows.append(new_row)

    if n_unmatched:
        msg = f"{n_unmatched} dump question(s) had no matching --parquet row"
        if unmatched_examples:
            msg += "; e.g. " + " | ".join(repr(e) for e in unmatched_examples)
        if args.require_all:
            raise SystemExit(f"[error] {msg}")
        print(f"[warn] {msg} (skipped)")
    if n_no_draft:
        print(f"[warn] {n_no_draft} question(s) skipped: no answer-free expert-reasoning prefix")
    if n_unboxed:
        print(f"[warn] dropped {n_unboxed} correct rollout(s) with no \\boxed{{}} (length-capped mid-derivation)")
    if n_no_candidate:
        print(f"[warn] {n_no_candidate} question(s) skipped: no correct rollout passed the filters")
    if not rows:
        raise SystemExit("no SFT rows built; nothing to write")

    out_df = pd.DataFrame(rows)
    # Preserve the source column order, then append the three SFT columns, so the
    # schema lines up with the reference rephraser-SFT parquet.
    ordered = [c for c in df.columns if c not in ("summarize_prompts", "output", "messages")]
    out_df = out_df[ordered + ["summarize_prompts", "output", "messages"]]

    n_q = out_df["summarize_prompts"].map(lambda m: dict(m[-1])["content"]).nunique()
    out_len = out_df["output"].str.len()
    prompt_len = out_df["summarize_prompts"].map(lambda m: len(dict(m[-1])["content"]))
    print(
        f"[build] {len(out_df)} SFT rows over {n_q} question(s)\n"
        f"        prompt chars: p50={int(prompt_len.median())} max={int(prompt_len.max())}\n"
        f"        output chars: p50={int(out_len.median())} max={int(out_len.max())}\n"
        f"        template={args.template} draft_ratio={args.draft_ratio} "
        f"pick={args.pick} per_question={args.per_question}"
    )

    sample_roles = [dict(m)["role"] for m in out_df.iloc[0]["messages"]]
    assert sample_roles[-1] == "assistant", f"last message must be assistant for SFT masking, got {sample_roles}"
    print(f"[build] messages roles = {sample_roles}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    out_path = args.output or args.parquet.with_name(f"{args.parquet.stem}_rephraser_sft.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[write] {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
