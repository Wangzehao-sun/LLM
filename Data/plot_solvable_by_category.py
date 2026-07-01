#!/usr/bin/env python3
"""Plot the fraction of SOLVABLE questions per category across training steps.

Each rollout-dump file (one per step) holds N questions, every question
expanded into a fixed number of rollouts (8 by default) sharing the same
``uid`` / ``original_index``. A question is counted as SOLVABLE (solve@8) when
at least one of its *on-policy* rollouts scored > 0. Rollouts with
``is_replaced == True`` are injected off-policy correct answers and are NOT
counted -- only genuine on-policy solves matter.

For every step we split the questions into CATEGORIES and, within each category,
report the fraction of solvable questions::

    solvable_fraction[cat] = (# solvable questions in cat) / (# questions in cat)

Two category dimensions are used (one PNG each per folder):

  * topic       -- extra_info.topic, cut to a level (default the field after
                   "Mathematics", e.g. Calculus / Algebra / Geometry / ...).
  * difficulty  -- extra_info.difficulty (e.g. 6.5 / 7.0 / ... / 9.5).

TWO-FOLDER COMPARISON
---------------------
Pass a second folder with ``--input-dir2``. Its rollout files need NOT carry
category info (topic / difficulty); they borrow it from ``--input-dir`` by
STEP + POSITIONAL ORDER: for the same-step file, the i-th question (in
first-appearance order) inherits folder-1's i-th question's topic / difficulty.
This yields 4 PNGs -- {topic, difficulty} x {folder1, folder2} -- so you can
put the two runs side by side.

Each PNG has one line per category: x = step, y = solvable fraction in [0, 1].

A folder-1 file missing any required key (score / step / group-id / extra_info)
is skipped. A folder-2 file only needs score / step / group-id; if its step has
no folder-1 map it is skipped with a warning.

Usage:
    # single folder (2 PNGs)
    python Data/plot_solvable_by_category.py --input-dir /path/to/folder

    # two folders compared (4 PNGs)
    python Data/plot_solvable_by_category.py \
        --input-dir  DIR_WITH_CATEGORIES \
        --input-dir2 DIR_WITHOUT_CATEGORIES \
        --label1 baseline --label2 ours --output-dir OUT
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keys every record must carry. extra_info (with the nested category fields) is
# required only for the category-owning folder (folder 1).
REQUIRED_TOP_KEYS = ("score", "step")


def _load_records(path):
    """Yield parsed JSON objects from a .jsonl (or .json list) file."""
    with open(path, "r") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []
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


def _has_required_keys(records, need_extra):
    """True iff required keys exist. need_extra also demands extra_info (dict)."""
    if not records:
        return False
    sample = records[0]
    for k in REQUIRED_TOP_KEYS:
        if k not in sample:
            return False
    if _group_key(sample) is None:
        return False
    if need_extra and not isinstance(sample.get("extra_info"), dict):
        return False
    return True


def _ordered_question_rollouts(records):
    """Rollout-lists, one per question, in first-appearance order of the group id."""
    order = []
    groups = {}
    for rec in records:
        k = _group_key(rec)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(rec)
    return [groups[k] for k in order]


def _question_solvable(rollouts):
    """solve@8: True iff any *on-policy* rollout scored > 0.

    Rollouts with is_replaced == True are injected off-policy correct answers
    (a correct trajectory was spliced into an all-wrong group). They do NOT
    reflect the model actually solving the question, so they are excluded --
    only genuine on-policy rollouts count toward solvability.
    """
    return any(
        float(r.get("score", 0)) > 0.0 and not bool(r.get("is_replaced"))
        for r in rollouts
    )


def _topic_bucket(rollouts, level):
    """Category label from extra_info.topic, cut to ``level`` (1-indexed).

    topic looks like "Mathematics -> Calculus -> Integral Calculus -> ...".
    level=2 -> "Calculus"; level=1 -> "Mathematics". Falls back to the deepest
    available level when the topic has fewer parts. Returns None if absent.
    """
    for r in rollouts:
        topic = r.get("extra_info", {}).get("topic")
        if topic:
            parts = [p.strip() for p in str(topic).split("->") if p.strip()]
            if parts:
                return parts[min(level - 1, len(parts) - 1)]
    return None


def _difficulty_bucket(rollouts):
    """Category label from extra_info.difficulty (first non-null). None if absent."""
    for r in rollouts:
        d = r.get("extra_info", {}).get("difficulty")
        if d is not None:
            try:
                return f"{float(d):g}"
            except (TypeError, ValueError):
                return str(d)
    return None


def _own_label(rollouts, dimension, topic_level):
    """Category label of a question from its OWN extra_info."""
    if dimension == "topic":
        return _topic_bucket(rollouts, topic_level)
    return _difficulty_bucket(rollouts)


def build_label_maps(folder_data, topic_level):
    """Positional category maps for folder 1, keyed by step.

    folder_data: list of (step, records). Returns
    step -> {"topic": [labels...], "difficulty": [labels...]}, where the list
    is aligned to the questions' first-appearance order in that step's file.
    Later folders borrow these by position.
    """
    maps = {}
    for step, records in folder_data:
        qs = _ordered_question_rollouts(records)
        maps[step] = {
            "topic": [_own_label(q, "topic", topic_level) for q in qs],
            "difficulty": [_own_label(q, "difficulty", topic_level) for q in qs],
        }
    return maps


def aggregate(records, labels):
    """Per-category (solvable, total) counts for one file given a label list.

    ``labels`` is aligned to the questions' first-appearance order. Questions
    beyond the shorter of (questions, labels) are dropped (with the caller
    warning on length mismatch). None labels are skipped.
    """
    qs = _ordered_question_rollouts(records)
    n = min(len(qs), len(labels))
    counts = defaultdict(lambda: [0, 0])  # cat -> [solvable, total]
    for i in range(n):
        cat = labels[i]
        if cat is None:
            continue
        counts[cat][1] += 1
        if _question_solvable(qs[i]):
            counts[cat][0] += 1
    return counts, len(qs)


def _step_of(records):
    """Representative step for a file (the most common value)."""
    counts = defaultdict(int)
    for r in records:
        counts[r["step"]] += 1
    return int(max(counts, key=counts.get))


def _sorted_categories(cats, dimension):
    """difficulty sorts numerically; topic sorts alphabetically."""
    if dimension == "difficulty":
        def _key(c):
            try:
                return (0, float(c))
            except ValueError:
                return (1, c)
        return sorted(cats, key=_key)
    return sorted(cats)


def load_folder(paths, need_extra):
    """Load + validate a folder's files. Returns list of (step, records)."""
    out = []
    for path in paths:
        try:
            records = _load_records(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {os.path.basename(path)}: failed to parse ({e})")
            continue
        if not _has_required_keys(records, need_extra):
            print(f"[skip] {os.path.basename(path)}: missing required key(s)")
            continue
        out.append((_step_of(records), records))
    return out


def per_step_counts_own(folder_data, dimension, topic_level):
    """folder-1 aggregation: each file labels its questions from its own info."""
    per_step = []
    for step, records in folder_data:
        qs = _ordered_question_rollouts(records)
        labels = [_own_label(q, dimension, topic_level) for q in qs]
        counts, _ = aggregate(records, labels)
        if counts:
            per_step.append((step, counts))
    return per_step


def per_step_counts_borrowed(folder_data, dimension, label_maps, tag):
    """folder-2 aggregation: labels borrowed from folder-1 by step + position."""
    per_step = []
    for step, records in folder_data:
        if step not in label_maps:
            print(f"[skip] {tag}: step={step} has no folder-1 map to borrow, skipping")
            continue
        labels = label_maps[step][dimension]
        counts, n_q = aggregate(records, labels)
        if n_q != len(labels):
            print(f"[warn] {tag}: step={step} question count {n_q} != folder-1 "
                  f"{len(labels)}; aligned on the first {min(n_q, len(labels))}")
        if counts:
            per_step.append((step, counts))
    return per_step


def plot_dimension(per_step, dimension, min_questions, out_path, title):
    """per_step: list of (step, {cat: [solvable, total]}). Draw one line/cat."""
    series = defaultdict(list)  # cat -> [(step, fraction)]
    for step, counts in per_step:
        for cat, (n_solv, n_tot) in counts.items():
            if n_tot >= min_questions:
                series[cat].append((step, n_solv / n_tot))

    if not series:
        print(f"[warn] {dimension}: no category met --min-questions={min_questions}, "
              f"skipping {os.path.basename(out_path)}")
        return False

    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    for i, cat in enumerate(_sorted_categories(series.keys(), dimension)):
        pts = sorted(series[cat], key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker=markers[i % len(markers)], linewidth=1.7,
                 markersize=5, label=str(cat))

    plt.xlabel("step")
    plt.ylabel("solvable fraction (solve@8, on-policy)")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path} ({len(series)} categories)")
    return True


def _safe(name):
    """Filesystem-safe tag for filenames."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="Folder-1: rollout dumps WITH category info (topic/difficulty).")
    ap.add_argument("--input-dir2", default=None,
                    help="Folder-2 (optional): rollout dumps that borrow folder-1's "
                         "categories by step + positional order. Produces a second set "
                         "of PNGs for side-by-side comparison.")
    ap.add_argument("--pattern", default="*.jsonl",
                    help="Glob pattern for data files (default: *.jsonl).")
    ap.add_argument("--label1", default=None,
                    help="Name for folder-1 in titles/filenames (default: its basename).")
    ap.add_argument("--label2", default=None,
                    help="Name for folder-2 in titles/filenames (default: its basename).")
    ap.add_argument("--topic-level", type=int, default=2,
                    help="Which topic level to bucket on, 1-indexed "
                         "(default 2 = the field after 'Mathematics').")
    ap.add_argument("--min-questions", type=int, default=1,
                    help="Skip a (step, category) point with fewer than this many "
                         "questions (avoids noisy tiny buckets). Default 1.")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the PNGs (default: --input-dir).")
    args = ap.parse_args()

    paths1 = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths1:
        raise SystemExit(f"No files match {args.pattern!r} in {args.input_dir}")
    out_dir = args.output_dir or args.input_dir
    label1 = args.label1 or os.path.basename(os.path.normpath(args.input_dir))

    folder1 = load_folder(paths1, need_extra=True)
    if not folder1:
        raise SystemExit("No valid folder-1 files.")
    print(f"[folder1={label1}] {len(folder1)} step file(s)")

    label_maps = build_label_maps(folder1, args.topic_level)

    folder2 = None
    label2 = None
    if args.input_dir2:
        paths2 = sorted(glob.glob(os.path.join(args.input_dir2, args.pattern)))
        if not paths2:
            raise SystemExit(f"No files match {args.pattern!r} in {args.input_dir2}")
        label2 = args.label2 or os.path.basename(os.path.normpath(args.input_dir2))
        folder2 = load_folder(paths2, need_extra=False)
        if not folder2:
            raise SystemExit("No valid folder-2 files.")
        print(f"[folder2={label2}] {len(folder2)} step file(s)")

    for dim in ("topic", "difficulty"):
        p1 = per_step_counts_own(folder1, dim, args.topic_level)
        plot_dimension(
            p1, dim, args.min_questions,
            os.path.join(out_dir, f"solvable_by_{dim}_vs_step__{_safe(label1)}.png"),
            f"Solvable fraction (solve@8) by {dim} vs step -- {label1}",
        )
        if folder2 is not None:
            p2 = per_step_counts_borrowed(folder2, dim, label_maps, tag=label2)
            plot_dimension(
                p2, dim, args.min_questions,
                os.path.join(out_dir, f"solvable_by_{dim}_vs_step__{_safe(label2)}.png"),
                f"Solvable fraction (solve@8) by {dim} vs step -- {label2}",
            )


if __name__ == "__main__":
    main()
