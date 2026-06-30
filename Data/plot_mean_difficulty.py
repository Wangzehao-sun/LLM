#!/usr/bin/env python3
"""Plot mean difficulty of "hard" questions across training steps.

Each rollout-dump file (one per step) holds N questions, every question
expanded into a fixed number of rollouts (8 by default) sharing the same
`uid` / `original_index`. A question is counted as "hard" when either:

  * any of its rollouts has is_replaced == True   (a correct off-policy
    answer had to be injected because the group was all-wrong), or
  * all of its rollouts scored 0                  (never solved on-policy).

For the hard questions in a file we take each question's difficulty once
(from extra_info.difficulty) and average over questions. The remaining
(non-hard) questions are averaged the same way. Both means are plotted as two
lines against the file's `step` (x-axis).

If a file is missing any required key it is skipped entirely.

Usage:
    python Data/plot_mean_difficulty.py --input-dir /path/to/folder
    python Data/plot_mean_difficulty.py --input-dir DIR --pattern '*.jsonl' \
        --output mean_difficulty.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Top-level keys every record must carry. extra_info.difficulty is checked
# separately because it is nested.
REQUIRED_TOP_KEYS = ("score", "is_replaced", "step", "extra_info")


def _load_records(path):
    """Yield parsed JSON objects from a .jsonl (or .json list) file."""
    with open(path, "r") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []
    # Support both jsonl (one object per line) and a single JSON array.
    if text[0] == "[":
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _group_key(rec):
    """Identify the question a rollout belongs to."""
    if rec.get("uid") is not None:
        return rec["uid"]
    return rec.get("original_index")


def _has_required_keys(records):
    """Return True iff every required key (incl. nested difficulty) exists."""
    if not records:
        return False
    sample = records[0]
    for k in REQUIRED_TOP_KEYS:
        if k not in sample:
            return False
    if _group_key(sample) is None:
        return False
    extra = sample.get("extra_info")
    if not isinstance(extra, dict) or "difficulty" not in extra:
        return False
    return True


def _question_difficulty(rollouts):
    """Per-question difficulty (first non-null among its rollouts)."""
    for r in rollouts:
        d = r.get("extra_info", {}).get("difficulty")
        if d is not None:
            return float(d)
    return None


def compute_mean_difficulty(records):
    """Split questions into hard / rest and average difficulty within each.

    Returns (hard_mean, rest_mean, n_hard, n_rest, n_total). A mean is None
    when that bucket is empty.
    """
    groups = defaultdict(list)
    for rec in records:
        groups[_group_key(rec)].append(rec)

    hard_difficulties = []
    rest_difficulties = []
    for rollouts in groups.values():
        is_replaced = any(bool(r.get("is_replaced")) for r in rollouts)
        all_zero = all(float(r.get("score", 0)) == 0.0 for r in rollouts)
        diff = _question_difficulty(rollouts)
        if diff is None:
            continue
        if is_replaced or all_zero:
            hard_difficulties.append(diff)
        else:
            rest_difficulties.append(diff)

    hard_mean = sum(hard_difficulties) / len(hard_difficulties) if hard_difficulties else None
    rest_mean = sum(rest_difficulties) / len(rest_difficulties) if rest_difficulties else None
    return hard_mean, rest_mean, len(hard_difficulties), len(rest_difficulties), len(groups)


def _step_of(records):
    """Representative step for a file (the most common value)."""
    counts = defaultdict(int)
    for r in records:
        counts[r["step"]] += 1
    return int(max(counts, key=counts.get))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="Folder containing rollout-dump files.")
    ap.add_argument("--pattern", default="*.jsonl",
                    help="Glob pattern for data files (default: *.jsonl).")
    ap.add_argument("--output", default=None,
                    help="Output image path (default: <input-dir>/mean_difficulty_vs_step.png).")
    ap.add_argument("--title", default="Mean difficulty of hard questions vs step")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files match {args.pattern!r} in {args.input_dir}")

    points = []  # (step, hard_mean, rest_mean)
    for path in paths:
        try:
            records = _load_records(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {os.path.basename(path)}: failed to parse ({e})")
            continue
        if not _has_required_keys(records):
            print(f"[skip] {os.path.basename(path)}: missing required key(s)")
            continue
        hard_mean, rest_mean, n_hard, n_rest, n_total = compute_mean_difficulty(records)
        step = _step_of(records)
        if hard_mean is None and rest_mean is None:
            print(f"[warn] {os.path.basename(path)}: step={step} has no usable questions, skipping point")
            continue
        points.append((step, hard_mean, rest_mean))
        hard_str = f"{hard_mean:.4f}" if hard_mean is not None else "n/a"
        rest_str = f"{rest_mean:.4f}" if rest_mean is not None else "n/a"
        print(f"[ok]   {os.path.basename(path)}: step={step} "
              f"hard={hard_str} ({n_hard}) rest={rest_str} ({n_rest}) total={n_total}")

    if not points:
        raise SystemExit("No valid data points to plot.")

    points.sort(key=lambda p: p[0])

    def _series(idx):
        # Keep only steps where this bucket has a value (avoid plotting None gaps).
        xs, ys = [], []
        for p in points:
            if p[idx] is not None:
                xs.append(p[0])
                ys.append(p[idx])
        return xs, ys

    hard_x, hard_y = _series(1)
    rest_x, rest_y = _series(2)

    out = args.output or os.path.join(args.input_dir, "mean_difficulty_vs_step.png")
    plt.figure(figsize=(9, 5))
    if hard_x:
        plt.plot(hard_x, hard_y, marker="o", linewidth=1.8, label="hard questions")
    if rest_x:
        plt.plot(rest_x, rest_y, marker="s", linewidth=1.8, label="remaining questions")
    plt.xlabel("step")
    plt.ylabel("mean difficulty")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out} ({len(points)} points)")


if __name__ == "__main__":
    main()
