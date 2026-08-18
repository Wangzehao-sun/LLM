"""Aggregate sweep_sft_checkpoints.sh output parquets into one SR-replacement dataset.

``verl.trainer.main_generation`` (which sweep_sft_checkpoints.sh drives) writes one
parquet per batch into its output dir, each carrying ``responses`` (N samples per
question) and ``test_score`` (per-sample scores + mean/max). This script walks those
shards, keeps one CORRECT and length-typical response per question, and writes a single
parquet with the base schema plus the SR columns.

The point is to move the summarize-replacement candidates OFFLINE. Today
``_summarize_replace_normal_step`` generates K candidates every step -- 128 questions x 8
candidates of up to 14k tokens -- and keeps one per question, discarding ~97%. When the
rephraser is frozen its output distribution never changes across training, so that
sampling is re-rolling the same dice every step. Generating once and reading a column
costs nothing per step, and lets the candidates be inspected and filtered with the usual
offline tools (``check_repetition.py``, the trajectory filter) before they ever reach a
gradient.

Output = the input schema (so it stays feedable to ``prepare_summarize_prompts.py``)
plus:

    sr_response        : str   -- the chosen correct, median-length response
    sr_response_len    : int   -- its length (tokens when available, else chars)
    sr_n_correct       : int   -- how many of the question's samples were correct
    sr_n_total         : int   -- how many samples the question had
    sr_pass_rate       : float -- sr_n_correct / sr_n_total
    sr_source_shard    : str   -- which shard parquet it came from, for tracing

Questions with no correct response are DROPPED by default (nothing to replace a rollout
with). ``--keep-unsolved`` keeps them with an empty ``sr_response`` instead, so one file
can carry both the SR set and the error set.

Usage:

    # aggregate one sweep run's output
    python Data/aggregate_sr_responses.py \\
        --input-dir  ~/LLM/Train/verl/logs/<sweep>/<label>/save_data \\
        --output     Data/deepmath_hard_sr.parquet

    # several checkpoints' outputs at once; later dirs win ties on pass rate
    python Data/aggregate_sr_responses.py \\
        --input-dir <ckpt1>/save_data --input-dir <ckpt2>/save_data \\
        --output Data/deepmath_hard_sr.parquet --tokenizer-path /home/data/shared/Qwen3-4b-base

    # keep only responses in the middle of the global length distribution
    python Data/aggregate_sr_responses.py --input-dir <dir> --output <out> \\
        --len-low-pct 10 --len-high-pct 90
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Columns main_generation adds on top of the source dataset. They are per-sweep artifacts,
# not part of the dataset, so they are dropped unless --keep-gen-columns is passed.
GEN_COLUMNS = ("responses", "test_score", "response_lengths", "input_lengths", "se_logits")

# The boxed-answer instruction prefixing every question, stripped before using the
# question text as a join key. Same constant as collect_correct_rollouts.py.
BOILERPLATE_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def normalize_question(text: Any) -> str:
    """Strip the repeated boilerplate prefix and collapse whitespace.

    A join key, not a display string. The prefix can appear more than once (the parquet's
    user turn carries it twice in some vintages), hence the loop.
    """
    if not isinstance(text, str):
        return ""
    stripped = text.lstrip()
    while stripped.startswith(BOILERPLATE_PREFIX):
        stripped = stripped[len(BOILERPLATE_PREFIX) :].lstrip()
    return " ".join(stripped.split())


def question_of(prompt: Any) -> str:
    """Take the last user turn of a ``prompt`` cell as the question text."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, (list, tuple)):
        return ""
    contents = [m.get("content", "") for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
    return normalize_question(contents[-1]) if contents else ""


def is_correct(score: Any) -> bool:
    """True when a per-sample score means "correct".

    main_generation stores whatever the reward fn returned, and reward_impl_version=4
    returns numpy bools -- so scores_per_response is often ``[False False True False]``
    rather than floats. Accepting both is not defensive coding: treating a numpy bool as
    a float via ``== 1.0`` happens to work, but ``float(np.False_)`` on a missing value
    does not, and a silently-empty correct set looks identical to "the model solved
    nothing".
    """
    if isinstance(score, bool):
        return score
    try:
        return float(score) == 1.0
    except (TypeError, ValueError):
        return False


def shard_files(dirs: list[Path]) -> list[Path]:
    """Collect the per-batch parquets, ordered by batch index within each dir.

    main_generation names them ``<batch_idx>.parquet``, so lexical sort would put 10
    before 2 and make --limit non-deterministic in a way that is easy to miss.
    """
    found: list[Path] = []
    for directory in dirs:
        if not directory.is_dir():
            raise SystemExit(f"--input-dir is not a directory: {directory}")
        shards = list(directory.glob("*.parquet"))
        if not shards:
            raise SystemExit(f"no *.parquet in {directory}; did the sweep write anything?")

        def batch_index(path: Path) -> tuple[int, str]:
            match = re.fullmatch(r"(\d+)", path.stem)
            return (int(match.group(1)) if match else 1 << 30, path.stem)

        found.extend(sorted(shards, key=batch_index))
    return found


def pick_median(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """The middle-length candidate -- the model's typical output on this question.

    Not the mean: lengths are heavily right-skewed (one rambling sample can be 5x the
    others), so an average would drag the choice toward the tail. With an even count take
    the lower middle, which biases very slightly short -- the safer direction, since a
    long response is the one at risk of hitting the training truncation cap.
    """
    ordered = sorted(candidates, key=lambda c: (c["length"], c["text"]))
    return ordered[(len(ordered) - 1) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, action="append", required=True,
                    help="A sweep output dir holding <batch>.parquet shards. Repeatable.")
    ap.add_argument("--output", type=Path, required=True, help="Destination parquet.")
    ap.add_argument("--response-key", default="sr_response",
                    help="Column name for the chosen response (default: %(default)s).")
    ap.add_argument("--limit", type=int, default=None, help="Keep at most N questions.")
    ap.add_argument("--keep-unsolved", action="store_true",
                    help="Keep questions with no correct response, with an empty response, instead "
                         "of dropping them. Useful for building an error set from the same run.")
    ap.add_argument("--keep-gen-columns", action="store_true",
                    help="Keep main_generation's responses / test_score / *_lengths columns. Off by "
                         "default: they are large and make the output no longer schema-compatible "
                         "with the source dataset.")
    ap.add_argument("--require-boxed", action="store_true", default=True,
                    help="Require \\boxed{} in the kept response (default: on).")
    ap.add_argument("--allow-unboxed", dest="require_boxed", action="store_false",
                    help="Keep responses with no \\boxed{}: those were length-capped mid-derivation "
                         "and state no final answer.")
    ap.add_argument("--len-low-pct", type=float, default=0.0,
                    help="Drop candidates below this global length percentile before picking, e.g. "
                         "10 to exclude give-up-early answers (default: %(default)s = off).")
    ap.add_argument("--len-high-pct", type=float, default=100.0,
                    help="Drop candidates above this global length percentile, e.g. 90 to exclude "
                         "rambling/truncated ones (default: %(default)s = off).")
    ap.add_argument("--tokenizer-path", default=None,
                    help="Tokenizer for exact token lengths, used when the shards lack a "
                         "response_lengths column. Without either, lengths are characters.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if not 0.0 <= args.len_low_pct < args.len_high_pct <= 100.0:
        raise SystemExit(f"need 0 <= --len-low-pct ({args.len_low_pct}) < --len-high-pct ({args.len_high_pct}) <= 100")

    files = shard_files(args.input_dir)
    print(f"[scan] {len(files)} shard(s) across {len(args.input_dir)} dir(s)")

    encode = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            encode = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
            print(f"[length] tokenizer: {args.tokenizer_path}")
        except Exception as error:
            print(f"[length] could not load {args.tokenizer_path} ({error}); falling back")

    # Group by question across every shard. A question appearing in several shards (e.g.
    # two checkpoints of the same eval set) contributes all of its samples to one pool,
    # so the median is taken over everything available rather than per shard.
    groups: dict[str, dict[str, Any]] = {}
    rows_by_question: dict[str, tuple[Path, int]] = {}
    stats = {"rows": 0, "no_question": 0, "no_score": 0, "unboxed": 0, "empty": 0, "used_token_lens": 0}

    for path in files:
        df = pd.read_parquet(path)
        if "responses" not in df.columns or "test_score" not in df.columns:
            raise SystemExit(
                f"{path} has no 'responses'/'test_score' column (got {list(df.columns)}); this "
                f"script expects verl.trainer.main_generation output with +is_eval=True"
            )
        has_token_lens = "response_lengths" in df.columns

        for position in range(len(df)):
            stats["rows"] += 1
            row = df.iloc[position]
            question = question_of(row.get("prompt"))
            if not question:
                stats["no_question"] += 1
                continue

            score_blob = row["test_score"]
            scores = score_blob.get("scores_per_response") if isinstance(score_blob, dict) else None
            if scores is None:
                stats["no_score"] += 1
                continue
            responses = row["responses"]
            if hasattr(responses, "tolist"):
                responses = responses.tolist()
            token_lens = row["response_lengths"] if has_token_lens else None
            if token_lens is not None and hasattr(token_lens, "tolist"):
                token_lens = token_lens.tolist()

            entry = groups.setdefault(question, {"correct": [], "n_correct": 0, "n_total": 0})
            # First shard to contain a question owns its base row, so the output keeps one
            # row per question with the schema it arrived with.
            rows_by_question.setdefault(question, (path, position))

            for i, response in enumerate(responses):
                entry["n_total"] += 1
                if not is_correct(scores[i] if i < len(scores) else None):
                    continue
                entry["n_correct"] += 1
                if not isinstance(response, str) or not response.strip():
                    stats["empty"] += 1
                    continue
                response = response.strip()
                if args.require_boxed and "\\boxed" not in response:
                    # Scored correct but states no final answer: the response-length cap cut
                    # it off mid-derivation. Reads oddly as "the model's solution".
                    stats["unboxed"] += 1
                    continue
                if token_lens is not None and i < len(token_lens):
                    length = int(token_lens[i])
                    stats["used_token_lens"] += 1
                elif encode is not None:
                    length = encode(response)
                else:
                    length = len(response)
                entry["correct"].append({"text": response, "length": length, "shard": path.name})

    solved = {q: e for q, e in groups.items() if e["correct"]}
    print(f"[pool] {stats['rows']:,} shard row(s) -> {len(groups):,} question(s); "
          f"{len(solved):,} have at least one usable correct response")
    for name, label in (
        ("unboxed", "correct but no \\boxed{}"),
        ("empty", "correct but empty text"),
        ("no_question", "unparseable prompt"),
        ("no_score", "missing test_score"),
    ):
        if stats[name]:
            print(f"        skipped {stats[name]:,} sample(s) ({label})")
    if stats["used_token_lens"]:
        print("[length] using main_generation's response_lengths (exact token counts)")
    elif encode is None:
        print("[length] no response_lengths column and no --tokenizer-path: lengths are CHARACTERS, "
              "so the percentile window is approximate")
    if not solved:
        raise SystemExit("no question has a usable correct response; nothing to build")

    # Percentile window is computed over the GLOBAL pool, not per question: a per-question
    # window would be meaningless (4-8 samples) and would not express "this answer is
    # unusually short/long for this dataset", which is the actual intent.
    if args.len_low_pct > 0.0 or args.len_high_pct < 100.0:
        all_lengths = pd.Series([c["length"] for e in solved.values() for c in e["correct"]])
        low = all_lengths.quantile(args.len_low_pct / 100.0)
        high = all_lengths.quantile(args.len_high_pct / 100.0)
        kept = 0
        for entry in solved.values():
            inside = [c for c in entry["correct"] if low <= c["length"] <= high]
            # Never empty a question's pool: an all-outside question falls back to its full
            # set rather than being silently dropped by a length filter.
            if inside:
                entry["correct"] = inside
            kept += len(inside)
        print(f"[length] window p{args.len_low_pct:g}-p{args.len_high_pct:g} = [{low:,.0f}, {high:,.0f}]: "
              f"{kept:,}/{len(all_lengths):,} candidate(s) inside")

    # Assemble output rows in shard order so the result is reproducible.
    ordered_questions = sorted(rows_by_question, key=lambda q: (rows_by_question[q][0].name, rows_by_question[q][1]))
    base_rows, sr_meta = [], []
    shard_cache: dict[Path, pd.DataFrame] = {}
    n_unsolved = 0

    for question in ordered_questions:
        if args.limit is not None and len(base_rows) >= args.limit:
            break
        entry = groups[question]
        chosen = pick_median(entry["correct"]) if entry["correct"] else None
        if chosen is None:
            n_unsolved += 1
            if not args.keep_unsolved:
                continue
        path, position = rows_by_question[question]
        if path not in shard_cache:
            shard_cache[path] = pd.read_parquet(path)
        base_rows.append(shard_cache[path].iloc[position])
        sr_meta.append({
            "text": chosen["text"] if chosen else "",
            "length": chosen["length"] if chosen else 0,
            "shard": chosen["shard"] if chosen else path.name,
            "n_correct": entry["n_correct"],
            "n_total": entry["n_total"],
        })

    if not base_rows:
        raise SystemExit("no rows to write")

    out_df = pd.DataFrame(base_rows).reset_index(drop=True)
    out_df[args.response_key] = [m["text"] for m in sr_meta]
    out_df["sr_response_len"] = [m["length"] for m in sr_meta]
    out_df["sr_n_correct"] = [m["n_correct"] for m in sr_meta]
    out_df["sr_n_total"] = [m["n_total"] for m in sr_meta]
    out_df["sr_pass_rate"] = [m["n_correct"] / max(1, m["n_total"]) for m in sr_meta]
    out_df["sr_source_shard"] = [m["shard"] for m in sr_meta]

    if not args.keep_gen_columns:
        dropped = [c for c in GEN_COLUMNS if c in out_df.columns]
        out_df = out_df.drop(columns=dropped)
        if dropped:
            print(f"[schema] dropped generation column(s): {', '.join(dropped)} "
                  f"(--keep-gen-columns to retain)")

    lengths = pd.Series([m["length"] for m in sr_meta if m["length"] > 0])
    rates = pd.Series([m["n_correct"] / max(1, m["n_total"]) for m in sr_meta])
    unit = "tokens" if (stats["used_token_lens"] or encode is not None) else "chars"
    print(
        f"[build] {len(out_df):,} row(s), one {args.response_key} each (median length per question)\n"
        f"        chosen length ({unit}): p50={int(lengths.median()):,} p90={int(lengths.quantile(0.9)):,} "
        f"max={int(lengths.max()):,}\n"
        f"        pass rate: p50={rates.median():.3f} mean={rates.mean():.3f}"
    )
    if n_unsolved:
        verb = "kept with an empty response" if args.keep_unsolved else "dropped"
        print(f"[build] {n_unsolved:,} question(s) had no usable correct response ({verb})")
    if args.limit is not None and len(out_df) == args.limit:
        print(f"[note] stopped at --limit {args.limit}; more questions were available")
    print(f"[schema] columns: {list(out_df.columns)}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"[write] {len(out_df):,} rows ({args.output.stat().st_size / 1e6:.1f} MB) -> {args.output}")


if __name__ == "__main__":
    main()
