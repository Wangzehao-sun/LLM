"""Visualize a rollout-dump JSONL (the per-step files written by the trainer's
_dump_generations, e.g. rollout_data/normal/*.jsonl).

Each line is one rollout row with fields:
    input          : the prompt the model was conditioned on (rewritten/summarize
                     prompt for off rows, plain question for on rows)
    output         : the model's generated solution
    score          : reward for this rollout (1.0 correct / 0.0 wrong / -1.0 format-error)
    step           : training step
    reward_model   : {'ground_truth': ..., 'style': ...}
    data_source    : dataset tag
    original_index : row index in the source parquet
    uid            : question id (all rollouts of one question share it)
    extra_info     : dataset meta
    is_replaced    : True if this row is an SR-injected off-policy candidate
    target         : the reference solution (decoded tgt_input_ids)

Two outputs:
  1. A console summary (distribution of scores, per-uid solve stats, lengths,
     data-source breakdown, replaced-row stats).
  2. An interactive HTML report grouping rollouts by question (uid), showing
     each rollout's score + output side-by-side, plus the shared prompt / target /
     ground truth. Open it in a browser to browse individual examples.

Usage:

    python Data/visualize_rollouts.py \
        --input  /Users/zenohaoz/Desktop/1.jsonl \
        --output /Users/zenohaoz/Desktop/1_report.html
        # --limit-questions N   only render the first N questions in the HTML
        # --no-html             console summary only
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Rollout-dump JSONL file.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML report path (default: alongside input as <stem>_report.html).",
    )
    p.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Only render the first N questions (uids) in the HTML. Default: all.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Print the console summary only; skip HTML generation.",
    )
    return p.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"--input is not a file: {path}")
    recs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {i}: {e}")
    if not recs:
        raise SystemExit(f"No usable JSON lines in {path}")
    return recs


def _score_label(s: float) -> str:
    if s == 1.0:
        return "correct"
    if s == 0.0:
        return "wrong"
    if s == -1.0:
        return "format-error"
    return f"{s:g}"


def print_summary(recs: list[dict]) -> None:
    n = len(recs)
    has = lambda k: any(k in r for r in recs)  # noqa: E731

    print(f"\n{'=' * 60}\nRollout dump summary: {n:,} rows\n{'=' * 60}")

    # Score distribution
    if has("score"):
        sc = Counter(_score_label(r.get("score")) for r in recs if "score" in r)
        print("\nScore distribution:")
        for label, c in sorted(sc.items(), key=lambda x: -x[1]):
            print(f"  {label:>13}: {c:>6,} ({100 * c / n:5.1f}%)")

    # Steps
    if has("step"):
        steps = Counter(r.get("step") for r in recs if "step" in r)
        print(f"\nSteps present: {sorted(steps)}  (rows/step: {dict(sorted(steps.items()))})")

    # is_replaced
    if has("is_replaced"):
        rep = Counter(bool(r.get("is_replaced")) for r in recs if "is_replaced" in r)
        print(f"\nis_replaced: replaced={rep.get(True, 0):,}, on-policy={rep.get(False, 0):,}")

    # Data source
    if has("data_source"):
        ds = Counter(r.get("data_source") for r in recs if "data_source" in r)
        print("\nData source:")
        for src, c in ds.most_common():
            print(f"  {src:>18}: {c:>6,} ({100 * c / n:5.1f}%)")

    # Lengths (char-based, tokenizer-free)
    if has("output"):
        outs = [len(r["output"]) for r in recs if r.get("output")]
        if outs:
            print(f"\nOutput length (chars): min={min(outs):,} mean={round(mean(outs)):,} max={max(outs):,}")
    if has("target"):
        tgts = [len(r["target"]) for r in recs if r.get("target")]
        if tgts:
            print(f"Target length (chars): min={min(tgts):,} mean={round(mean(tgts)):,} max={max(tgts):,}")

    # Per-question (uid) solve stats
    if has("uid") and has("score"):
        by_uid: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            if "uid" in r and "score" in r:
                by_uid[r["uid"]].append(r["score"])
        n_q = len(by_uid)
        n_correct_per_q = [sum(1 for s in v if s == 1.0) for v in by_uid.values()]
        solve_dist = Counter(n_correct_per_q)
        rollouts_per_q = Counter(len(v) for v in by_uid.values())
        print(f"\nQuestions (uid): {n_q:,}  (rollouts/question: {dict(sorted(rollouts_per_q.items()))})")
        print("Correct-rollout distribution per question:")
        for k in range(max(solve_dist) + 1 if solve_dist else 0):
            c = solve_dist.get(k, 0)
            print(f"  {k} correct: {c:>5,} ({100 * c / max(1, n_q):5.1f}%)")
        all_wrong = sum(1 for v in n_correct_per_q if v == 0)
        all_right = sum(1 for c, tot in ((sum(1 for s in v if s == 1.0), len(v)) for v in by_uid.values()) if c == tot)
        print(f"  all-wrong questions : {all_wrong:,} ({100 * all_wrong / max(1, n_q):.1f}%)")
        print(f"  all-correct questions: {all_right:,} ({100 * all_right / max(1, n_q):.1f}%)")

        if has("is_replaced"):
            injected_uids = {
                r["uid"] for r in recs if r.get("uid") is not None and r.get("is_replaced")
            }
            n_inj = len(injected_uids)
            print(f"  SR-injected questions: {n_inj:,} ({100 * n_inj / max(1, n_q):.1f}%)")


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

_SCORE_COLOR = {
    "correct": "#1a7f37",       # green
    "wrong": "#b35900",         # orange
    "format-error": "#a40e26",  # red
}


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def _badge(score) -> str:
    label = _score_label(score)
    color = _SCORE_COLOR.get(label, "#555")
    return f'<span class="badge" style="background:{color}">{_esc(label)} ({_esc(score)})</span>'


def build_html(recs: list[dict], limit_questions: int | None) -> str:
    # Group by uid, preserving first-seen order.
    by_uid: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for r in recs:
        u = r.get("uid", "__no_uid__")
        if u not in by_uid:
            order.append(u)
        by_uid[u].append(r)
    if limit_questions is not None:
        order = order[:limit_questions]

    parts: list[str] = []
    parts.append(
        """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Rollout report</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
 header{position:sticky;top:0;background:#22272e;color:#fff;padding:12px 20px;z-index:10}
 header h1{margin:0;font-size:16px}
 header .meta{font-size:12px;opacity:.8;margin-top:4px}
 .q{background:#fff;margin:16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1);overflow:hidden}
 .q>summary{cursor:pointer;padding:12px 16px;font-weight:600;background:#eef1f5;list-style:none;display:flex;justify-content:space-between;align-items:center}
 .q>summary::-webkit-details-marker{display:none}
 .q>summary .qmeta{font-weight:400;font-size:12px;color:#555}
 .section{padding:10px 16px;border-top:1px solid #eee}
 .section h3{margin:0 0 6px;font-size:13px;color:#444;text-transform:uppercase;letter-spacing:.03em}
 pre{white-space:pre-wrap;word-wrap:break-word;background:#f8f9fa;border:1px solid #e3e6ea;
     border-radius:6px;padding:10px;margin:0;font-size:13px;line-height:1.45;max-height:340px;overflow:auto}
 .badge{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap}
 .roll{border-top:1px dashed #ddd;padding:10px 16px}
 .roll .rhead{display:flex;gap:10px;align-items:center;margin-bottom:6px;font-size:13px;color:#333}
 .gt{color:#1a7f37;font-weight:600}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 @media(max-width:900px){.cols{grid-template-columns:1fr}}
 .tag{display:inline-block;background:#dfe3e8;color:#333;padding:1px 7px;border-radius:8px;font-size:11px;margin-left:6px}
 .tag.sr{background:#8250df;color:#fff}
</style></head><body>"""
    )

    n_q = len(order)
    total_rows = sum(len(by_uid[u]) for u in order)
    parts.append(
        f'<header><h1>Rollout report</h1>'
        f'<div class="meta">{n_q:,} questions · {total_rows:,} rollouts rendered'
        f' · click a question to expand</div></header>'
    )

    for qi, uid in enumerate(order):
        rolls = by_uid[uid]
        first = rolls[0]
        gt = first.get("reward_model", {}).get("ground_truth") if isinstance(first.get("reward_model"), dict) else None
        ds = first.get("data_source", "")
        n_correct = sum(1 for r in rolls if r.get("score") == 1.0)
        n_tot = len(rolls)
        n_replaced = sum(1 for r in rolls if r.get("is_replaced"))
        # shared prompt/target (identical across a question's rollouts in typical dumps)
        prompt = first.get("input", "")
        target = first.get("target", "")

        sr_tag = (
            f'<span class="tag sr">SR-injected ×{n_replaced}</span>' if n_replaced else ""
        )
        parts.append('<details class="q">')
        parts.append(
            f'<summary><span>Q{qi + 1} <span class="tag">{_esc(ds)}</span>'
            f'<span class="tag">idx {_esc(first.get("original_index", "?"))}</span>{sr_tag}</span>'
            f'<span class="qmeta">solved {n_correct}/{n_tot} · GT '
            f'<span class="gt">{_esc(gt)}</span></span></summary>'
        )

        # Shared prompt + reference target side by side
        parts.append('<div class="section"><div class="cols">')
        parts.append(f'<div><h3>Prompt (input)</h3><pre>{_esc(prompt)}</pre></div>')
        if target:
            parts.append(f'<div><h3>Reference target</h3><pre>{_esc(target)}</pre></div>')
        parts.append('</div></div>')

        # Each rollout
        for ri, r in enumerate(rolls):
            rep = ' <span class="tag">replaced</span>' if r.get("is_replaced") else ""
            parts.append('<div class="roll">')
            parts.append(
                f'<div class="rhead">Rollout {ri + 1} {_badge(r.get("score"))}{rep}</div>'
            )
            parts.append(f'<pre>{_esc(r.get("output", ""))}</pre>')
            parts.append('</div>')

        parts.append('</details>')

    parts.append('</body></html>')
    return "".join(parts)


def main() -> None:
    args = parse_args()
    recs = load_records(args.input)
    print(f"Loaded {len(recs):,} rows from {args.input}")

    print_summary(recs)

    if args.no_html:
        return

    out = args.output or args.input.with_name(f"{args.input.stem}_report.html")
    html_str = build_html(recs, args.limit_questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str, encoding="utf-8")
    size_kb = out.stat().st_size / 1e3
    print(f"\nWrote HTML report: {out} ({size_kb:.0f} KB)")
    print(f"Open it in a browser:  open {out}")


if __name__ == "__main__":
    main()
