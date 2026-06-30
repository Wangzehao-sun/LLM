#!/usr/bin/env python3
"""Box plot of the ON/OFF-policy token ratio across training steps.

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors`. For each step it gathers
the chosen bucket's token ratio and draws ONE box per (step, bucket), so the
x-axis is the training step and each box shows that step's ratio distribution
(median, quartiles, whiskers).

    ratio = exp(log_probs - old_log_probs)      # the PPO/IS ratio used in loss
    OFF tokens = prefix_mask & response_mask     # off-policy rows' valid tokens
    ON  tokens = (~prefix_mask) & response_mask  # on-policy rows' valid tokens

``--which``:
  * ``both`` (default): two side-by-side boxes per step -- on (blue, left) and
    off (orange, right) -- with a legend.
  * ``off`` / ``on``: a single box per step.

A red dashed line marks ratio=1 (on-distribution reference). Boxes are placed at
evenly spaced positions with the step number as the tick label (so far-apart
steps like 1 / 100 / 500 stay readable).

Note: with no off injection (prefix_mask all-False) the OFF bucket is empty for
every step (its box is skipped); the ON bucket then holds all valid tokens.

Usage:
    python Data/plot_off_ratio_box.py --save-dir LOGDIR/save_tensors
    python Data/plot_off_ratio_box.py --save-dir DIR --which both --steps 1,100,500 \
        --clamp 1e-3,10 --logy --output ratio_box.png
    python Data/plot_off_ratio_box.py --save-dir DIR --every 50   # subsample steps
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from load_saved_tensors import load_steps, token_ratio

# bucket -> box fill colour
_BUCKET_COLOR = {"on": "tab:blue", "off": "tab:orange"}


def bucket_ratio_values(step: dict, clamp: Optional[tuple], bucket: str) -> torch.Tensor:
    """1D tensor of ratio over this step's tokens in ``bucket``.

    ratio = exp(log_probs - old_log_probs), restricted to:
        off -> prefix_mask & response_mask
        on  -> (~prefix_mask) & response_mask
    Empty / missing keys -> empty tensor.
    """
    needed = ("log_probs", "old_log_probs", "prefix_mask", "response_mask")
    if any(k not in step for k in needed):
        return torch.empty(0)
    resp = step["response_mask"]
    if bucket == "off":
        tok = step["prefix_mask"] & resp
    else:
        tok = (~step["prefix_mask"]) & resp
    return token_ratio(step, clamp=clamp)[tok]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--save-dir", required=True, help="Directory holding tensors_step_*.pt")
    ap.add_argument(
        "--which",
        choices=["both", "off", "on"],
        default="both",
        help="Which bucket(s) to box. both (default): side-by-side on+off boxes "
             "per step. off / on: a single box per step.",
    )
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers, e.g. '1,100,500'. Default: ALL steps "
             "found (optionally thinned by --every).",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="When --steps is not given, keep every Nth step (1 = all). Useful "
             "when there are many dumps and the x-axis gets crowded.",
    )
    ap.add_argument(
        "--clamp",
        type=str,
        default="1e-3,10",
        help="Clamp the ratio to lo,hi before plotting (matches trainer). "
             "Pass 'none' to disable. Default '1e-3,10'.",
    )
    ap.add_argument("--logy", action="store_true", help="Log-scale the ratio (y) axis.")
    ap.add_argument(
        "--showfliers", action="store_true",
        help="Draw outlier points beyond the whiskers (off by default; off-ratio "
             "tails can be huge and swamp the boxes).",
    )
    ap.add_argument(
        "--whis",
        type=float,
        default=1.5,
        help="Whisker length as a multiple of the IQR (matplotlib default 1.5).",
    )
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/off_ratio_box.png).")
    args = ap.parse_args()

    clamp = None
    if args.clamp.strip().lower() != "none":
        try:
            lo, hi = (float(x) for x in args.clamp.split(","))
            clamp = (lo, hi)
        except ValueError:
            raise SystemExit("--clamp must be 'lo,hi' (e.g. '1e-3,10') or 'none'")

    steps = load_steps(args.save_dir)
    if not steps:
        raise SystemExit(f"No step dumps loaded from {args.save_dir}")
    avail = [int(s["step"]) for s in steps if "step" in s]
    by_step = {int(s["step"]): s for s in steps if "step" in s}

    if args.steps:
        try:
            want = [int(x) for x in args.steps.split(",") if x.strip() != ""]
        except ValueError:
            raise SystemExit("--steps must be comma-separated ints, e.g. '1,100,500'")
        missing = [w for w in want if w not in by_step]
        if missing:
            print(f"[warn] steps not found, skipping: {missing} (available: {avail})")
        want = [w for w in want if w in by_step]
    else:
        want = sorted(avail)[:: max(1, args.every)]
    if not want:
        raise SystemExit("No valid steps to plot.")

    buckets = ["on", "off"] if args.which == "both" else [args.which]

    # Gather per-step, per-bucket off/on-ratio arrays. Keep a step only if at
    # least one requested bucket has tokens.
    step_labels: List[str] = []
    per_bucket_data = {b: [] for b in buckets}   # bucket -> list aligned with step_labels
    for st in want:
        vals = {b: bucket_ratio_values(by_step[st], clamp=clamp, bucket=b).numpy() for b in buckets}
        if all(v.size == 0 for v in vals.values()):
            print(f"[skip] step {st}: no tokens in {buckets}")
            continue
        step_labels.append(str(st))
        for b in buckets:
            per_bucket_data[b].append(vals[b])
    if not step_labels:
        raise SystemExit("No tokens in any selected step/bucket (off injection disabled?).")

    n_steps = len(step_labels)
    centers = np.arange(1, n_steps + 1)  # evenly spaced step slots
    fig_w = max(8.0, 0.7 * n_steps + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    def _draw(positions, arrays, color, label):
        # boxplot needs non-empty arrays; substitute a tiny placeholder for empty
        # buckets but DON'T draw a misleading box -- we filter those positions out.
        pos_use, arr_use = [], []
        for p, a in zip(positions, arrays):
            if a.size > 0:
                pos_use.append(p)
                arr_use.append(a)
        if not arr_use:
            return None
        bp = ax.boxplot(
            arr_use, positions=pos_use, widths=width,
            showfliers=args.showfliers, whis=args.whis, patch_artist=True,
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.6)
        for med in bp["medians"]:
            med.set(color="black")
        bp["boxes"][0].set_label(label)  # one legend entry per bucket
        return bp

    if args.which == "both":
        width = 0.36
        _draw(centers - width / 2 - 0.02, per_bucket_data["on"], _BUCKET_COLOR["on"], "on")
        _draw(centers + width / 2 + 0.02, per_bucket_data["off"], _BUCKET_COLOR["off"], "off")
        ax.legend()
    else:
        width = 0.6
        b = buckets[0]
        _draw(centers, per_bucket_data[b], _BUCKET_COLOR[b], b)

    ax.axhline(1.0, color="red", lw=0.9, ls="--", alpha=0.7)  # ratio=1 reference
    if args.logy:
        ax.set_yscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(step_labels, rotation=45 if n_steps > 12 else 0, ha="right")
    ax.set_xlabel("training step")
    ax.set_ylabel("ratio = exp(log_probs - old_log_probs)")
    title_bucket = "on vs off" if args.which == "both" else args.which
    ax.set_title(f"{title_bucket} token ratio distribution per step")
    ax.grid(True, axis="y", alpha=0.3)

    default_name = f"ratio_box_{args.which}.png"
    out = args.output or os.path.join(args.save_dir, default_name)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} (which={args.which}, {n_steps} step(s): {step_labels})")


if __name__ == "__main__":
    main()
