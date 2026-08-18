"""Screen a text column for degenerate repetition before it becomes SFT target.

A target that keeps talking after it has answered teaches exactly that. Since the
route-B targets are the student's OWN rollouts, filtered only by ``score == 1``
and "contains ``\\boxed``", a response that answered correctly at 2k chars and
then looped to the response cap passes both checks and still poisons the SFT set.

Five signals, cheap enough to run over the whole parquet. None is conclusive
alone; a row tripping two or more is worth reading.

  chars after \\boxed{}   A finished solution stops shortly after boxing its
                         answer. Thousands of characters past it means the model
                         never stopped -- the single sharpest signal for math.
  \\boxed{} count         Re-answering over and over. Some legitimate solutions
                         box an intermediate result, so read the distribution
                         rather than flagging every count > 1.
  tail cycle             The literal degenerate loop: the response ends in k
                         consecutive copies of the same block. Finds the period.
  compression ratio      zlib size / raw size. Ordinary math prose sits near
                         0.30; a looped tail drags the whole response under
                         0.15. Catches loops that are not byte-exact.
  max n-gram repeat      Where the loop is, not just that it exists -- and it
                         finds repetition in the middle, which the tail signals
                         miss.

Report only: nothing is written back to the input, and no filtered copy is
produced. Use ``--dump`` to get the flagged tails in a file and read them.

Usage:

    # the student rollouts a route-B SFT set is built from
    python Data/check_repetition.py --input Data/rollouts_summarize.parquet --column rollout

    # the assembled SFT parquet: reads the assistant turn out of `messages`
    python Data/check_repetition.py --input Data/rephraser_sft.parquet --column messages

    # the expert reasoning that gets rendered into the prompt as the draft
    python Data/check_repetition.py --input Data/deepmath_hard_solonly.parquet --column target

    # write the flagged tails out to read them
    python Data/check_repetition.py --input <parquet> --column rollout \\
        --dump Data/repetition_flagged.txt
"""

from __future__ import annotations

import argparse
import zlib
from collections import Counter
from pathlib import Path

import pandas as pd


def _as_dict(message):
    """One message -> dict, tolerant of the numpy structured records parquet yields."""
    if isinstance(message, dict):
        return message
    if hasattr(message, "dtype") and getattr(message.dtype, "names", None):
        return {k: message[k] for k in message.dtype.names}
    return None


def extract_text(cell, role: str = "assistant") -> str:
    """Pull the text to screen out of a cell, whatever shape the column has.

    A plain string is itself. A messages list contributes only its ``role`` turns
    (assistant, by default) -- screening the prompt for repetition would flag the
    boilerplate every row shares. The extra length-1 array the renderer wraps
    messages in is unwrapped by recursing.
    """
    if isinstance(cell, str):
        return cell
    if cell is None:
        return ""
    if hasattr(cell, "tolist") and not hasattr(cell, "dtype"):
        cell = cell.tolist()
    as_dict = _as_dict(cell)
    if as_dict is not None:
        return str(as_dict.get("content") or "")
    if hasattr(cell, "tolist"):
        cell = cell.tolist()
    if isinstance(cell, (list, tuple)):
        parts = []
        for item in cell:
            message = _as_dict(item)
            if message is not None:
                if role is None or message.get("role") == role:
                    parts.append(str(message.get("content") or ""))
            else:
                parts.append(extract_text(item, role))
        return "\n".join(p for p in parts if p)
    return str(cell)


def boxed_stats(text: str) -> tuple[int, int]:
    """Return ``(number of \\boxed{}, characters after the last one closes)``.

    The closing brace is found by matching, not by ``str.find("}")``: answers like
    ``\\boxed{\\frac{1}{2}}`` nest, and stopping at the first ``}`` would report a
    tail that is really still part of the answer.
    """
    count = text.count("\\boxed")
    if count == 0:
        return 0, len(text)

    start = text.rfind("\\boxed")
    brace = text.find("{", start)
    if brace == -1:
        return count, len(text) - start

    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return count, len(text) - (i + 1)
    return count, 0  # unbalanced braces: treat the rest as part of the answer


def tail_cycle(text: str, window: int, min_repeats: int, max_period: int, min_span: int) -> int:
    """Shortest period whose repetition the response ends in, or 0 if none.

    ``min_span`` keeps trailing whitespace and a repeated ``$$`` from registering:
    a real loop repeats something substantial, so ``period * min_repeats`` has to
    cover at least that many characters.
    """
    tail = text.rstrip()[-window:]
    limit = min(max_period, len(tail) // min_repeats)
    for period in range(1, limit + 1):
        if period * min_repeats < min_span:
            continue
        if tail.endswith(tail[-period:] * min_repeats):
            return period
    return 0


def compress_ratio(text: str) -> float:
    """zlib size / raw size. Lower means more redundant; loops fall far below prose."""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / len(raw)


def max_ngram_repeat(text: str, n: int) -> tuple[int, str]:
    """Most-repeated word n-gram and its count.

    Word-level rather than character-level so that a loop whose whitespace or
    numbering drifts still collapses onto the same n-gram.
    """
    words = text.split()
    if len(words) < n:
        return 0, ""
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return count, " ".join(gram)


def describe(name: str, series: pd.Series, low: bool = False) -> None:
    """One distribution line. ``low=True`` reports the low tail instead of the high."""
    if low:
        print(
            f"[stat] {name:24s} p50={series.median():>8.2f}  p10={series.quantile(0.1):>8.2f}  "
            f"min={series.min():>8.2f}"
        )
    else:
        print(
            f"[stat] {name:24s} p50={int(series.median()):>8,}  p90={int(series.quantile(0.9)):>8,}  "
            f"p99={int(series.quantile(0.99)):>8,}  max={int(series.max()):>8,}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Parquet to screen.")
    ap.add_argument("--column", default="rollout", help="Column holding the text (default: %(default)s).")
    ap.add_argument("--role", default="assistant",
                    help="For a messages-shaped column, which role to screen (default: %(default)s).")
    ap.add_argument("--limit", type=int, default=None, help="Screen only the first N rows.")

    ap.add_argument("--tail-chars", type=int, default=300,
                    help="Flag when more than this many characters follow the last \\boxed{} "
                         "(default: %(default)s).")
    ap.add_argument("--ngram", type=int, default=20, help="n-gram size in words (default: %(default)s).")
    ap.add_argument("--max-repeat", type=int, default=3,
                    help="Flag when an n-gram appears at least this many times (default: %(default)s).")
    ap.add_argument("--compress-ratio", type=float, default=0.15,
                    help="Flag when zlib ratio falls below this (default: %(default)s).")
    ap.add_argument("--cycle-window", type=int, default=2000,
                    help="Trailing characters searched for a cycle (default: %(default)s).")
    ap.add_argument("--cycle-repeats", type=int, default=3,
                    help="Consecutive copies required to call it a cycle (default: %(default)s).")
    ap.add_argument("--cycle-max-period", type=int, default=400,
                    help="Longest cycle period to look for (default: %(default)s).")
    ap.add_argument("--cycle-min-span", type=int, default=40,
                    help="A cycle must span at least this many characters, so trailing "
                         "whitespace does not register (default: %(default)s).")

    ap.add_argument("--dump", type=Path, default=None,
                    help="Write the flagged rows' tails here for reading.")
    ap.add_argument("--dump-chars", type=int, default=1500,
                    help="Trailing characters written per flagged row (default: %(default)s).")
    ap.add_argument("--top", type=int, default=5, help="Worst rows to print (default: %(default)s).")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    if args.column not in df.columns:
        raise SystemExit(f"no '{args.column}' column; available: {list(df.columns)}")
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[data] {len(df):,} rows from {args.input}, column {args.column!r}")

    texts = [extract_text(cell, args.role) for cell in df[args.column]]
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"[warn] {empty:,} row(s) yielded no text for role={args.role!r}; they cannot be screened")

    rows = []
    for position, text in enumerate(texts):
        n_boxed, tail = boxed_stats(text)
        repeat, gram = max_ngram_repeat(text, args.ngram)
        rows.append({
            "row": position,
            "chars": len(text),
            "n_boxed": n_boxed,
            "tail": tail,
            "cycle": tail_cycle(text, args.cycle_window, args.cycle_repeats,
                                args.cycle_max_period, args.cycle_min_span),
            "ratio": compress_ratio(text),
            "repeat": repeat,
            "gram": gram,
        })
    stats = pd.DataFrame(rows)

    describe("chars", stats["chars"])
    describe("\\boxed{} count", stats["n_boxed"])
    describe("chars after last \\boxed", stats["tail"])
    describe(f"max {args.ngram}-gram repeat", stats["repeat"])
    describe("compress ratio", stats["ratio"], low=True)
    cycles = stats.loc[stats["cycle"] > 0, "cycle"]
    if len(cycles):
        print(f"[stat] {'tail cycle':24s} found in {len(cycles):,} row(s), "
              f"period p50={int(cycles.median()):,} max={int(cycles.max()):,}")
    else:
        print(f"[stat] {'tail cycle':24s} none found")

    # Each signal names a different way the same failure shows up, so they are
    # reported separately as well as unioned -- which one fires tells you whether
    # to fix the data or the generation cap.
    checks = {
        f"no \\boxed{{}} at all (truncated?)": stats["n_boxed"] == 0,
        f"more than {args.tail_chars:,} chars after \\boxed{{}}": stats["tail"] > args.tail_chars,
        "repeating tail cycle": stats["cycle"] > 0,
        f"{args.ngram}-gram repeated >= {args.max_repeat}x": stats["repeat"] >= args.max_repeat,
        f"compress ratio < {args.compress_ratio}": stats["ratio"] < args.compress_ratio,
    }
    flagged = pd.Series(False, index=stats.index)
    for mask in checks.values():
        flagged |= mask

    total = int(flagged.sum())
    print()
    if not total:
        print(f"[ok] no row tripped any signal over {len(stats):,} row(s)")
    else:
        print(f"[flag] {total:,}/{len(stats):,} row(s) ({100 * total / len(stats):.1f}%) tripped a signal:")
        for label, mask in checks.items():
            if int(mask.sum()):
                print(f"         {int(mask.sum()):>6,}  {label}")

        worst = stats[flagged].sort_values("tail", ascending=False).head(args.top)
        print(f"[flag] worst {len(worst)} by chars after \\boxed{{}}:")
        for _, r in worst.iterrows():
            print(f"         row {int(r['row']):<6} chars={int(r['chars']):>7,}  tail={int(r['tail']):>7,}  "
                  f"boxed={int(r['n_boxed']):>3}  cycle={int(r['cycle']):>4}  "
                  f"repeat={int(r['repeat']):>4}  ratio={r['ratio']:.3f}")

    if args.dump is not None and total:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as handle:
            for _, r in stats[flagged].sort_values("tail", ascending=False).iterrows():
                position = int(r["row"])
                tripped = [label for label, mask in checks.items() if bool(mask.iloc[position])]
                handle.write(f"{'=' * 78}\nrow {position}  chars={int(r['chars']):,}  "
                             f"tail={int(r['tail']):,}  boxed={int(r['n_boxed'])}  "
                             f"cycle={int(r['cycle'])}  repeat={int(r['repeat'])}  "
                             f"ratio={r['ratio']:.3f}\n")
                handle.write(f"tripped: {'; '.join(tripped)}\n")
                if r["gram"]:
                    handle.write(f"top {args.ngram}-gram: {r['gram'][:200]}\n")
                handle.write(f"--- last {args.dump_chars:,} chars ---\n{texts[position][-args.dump_chars:]}\n")
        print(f"[dump] {total:,} flagged row(s) -> {args.dump}")


if __name__ == "__main__":
    main()
