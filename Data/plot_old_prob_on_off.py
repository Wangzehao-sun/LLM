#!/usr/bin/env python3
"""Plot ON vs OFF old_prob mean across training steps (two lines, one figure).

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors`. For every step it computes
the per-token behaviour-policy probability::

    old_prob = exp(old_log_probs)

and splits it into ON / OFF buckets (matching the trainer's ``batch/old_prob_on``
/ ``batch/old_prob_off_standard`` convention -- a ROW split by prefix_mask):

    OFF rows = prefix_mask.any(-1)      # rows carrying any off-policy token
    ON  rows = ~OFF rows                # everything else

Within each bucket ALL valid response tokens (response_mask) are pooled and
averaged. The result is two curves on one figure: x = step, y = mean old_prob,
one line for ON rows and one for OFF rows.

Note: with no off injection (prefix_mask all-False) every row is ON, so the OFF
line is empty for those steps (skipped). For OFF rows the saved old_log_probs is
whatever the trainer stored at call site A (post prompt-swap when the reshape is
not in the swap blacklist), i.e. the long-prompt behaviour prob.

Usage:
    python Data/plot_old_prob_on_off.py --save-dir LOGDIR/save_tensors
    python Data/plot_old_prob_on_off.py --save-dir DIR --every 10 --band \
        --output old_prob_on_off.png
"""
from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from load_saved_tensors import load_steps, split_row_masks, masked_token_stat

# bucket -> line colour
_BUCKET_COLOR = {"on": "tab:blue", "off": "tab:orange"}


def old_prob_on_off(step: dict):
    """(on_stat, off_stat) masked mean/std of old_prob for one step.

    old_prob = exp(old_log_probs). Rows are split by prefix_mask (row split):
    OFF = prefix_mask.any(-1), ON = the rest; within each bucket all valid
    response tokens are pooled. Returns two dicts {mean,var,std,count} (from
    masked_token_stat) or None for a bucket with no tokens / missing keys.
    """
    needed = ("old_log_probs", "prefix_mask", "response_mask")
    if any(k not in step for k in needed):
        return None, None
    old_prob = torch.exp(step["old_log_probs"])          # [B, L]
    resp = step["response_mask"]
    rows = split_row_masks(step, include_se=False)        # {"on":..., "off":...}
    out = {}
    for name in ("on", "off"):
        token_mask = rows[name].unsqueeze(-1) & resp
        stat = masked_token_stat(old_prob, token_mask)
        out[name] = stat if stat["count"].item() > 0 else None
    return out["on"], out["off"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--save-dir", required=True, help="Directory holding tensors_step_*.pt")
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers, e.g. '1,100,500'. Default: ALL steps "
             "(optionally thinned by --every).",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="When --steps is not given, keep every Nth step (1 = all).",
    )
    ap.add_argument(
        "--band", action="store_true",
        help="Shade a ±1 std band around each mean line.",
    )
    ap.add_argument("--logy", action="store_true", help="Log-scale the old_prob (y) axis.")
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/old_prob_on_off.png).")
    args = ap.parse_args()

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

    # bucket -> parallel lists aligned with its own kept steps
    series = {"on": {"x": [], "mean": [], "std": []},
              "off": {"x": [], "mean": [], "std": []}}
    for st in want:
        on_stat, off_stat = old_prob_on_off(by_step[st])
        for name, stat in (("on", on_stat), ("off", off_stat)):
            if stat is None:
                continue
            series[name]["x"].append(st)
            series[name]["mean"].append(float(stat["mean"].item()))
            series[name]["std"].append(float(stat["std"].item()))
        on_str = f"{on_stat['mean'].item():.4f}" if on_stat else "n/a"
        off_str = f"{off_stat['mean'].item():.4f}" if off_stat else "n/a"
        print(f"[ok] step {st}: on={on_str} off={off_str}")

    if not series["on"]["x"] and not series["off"]["x"]:
        raise SystemExit("No usable tokens in any selected step.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ("on", "off"):
        xs = series[name]["x"]
        if not xs:
            continue
        means = np.asarray(series[name]["mean"])
        stds = np.asarray(series[name]["std"])
        ax.plot(xs, means, marker="o", linewidth=1.8, color=_BUCKET_COLOR[name],
                label=f"{name} old_prob")
        if args.band:
            ax.fill_between(xs, means - stds, means + stds, alpha=0.18,
                            color=_BUCKET_COLOR[name])

    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("old_prob = exp(old_log_probs)")
    ax.set_title("ON vs OFF old_prob (row split by prefix_mask) vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = args.output or os.path.join(args.save_dir, "old_prob_on_off.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out} (on: {len(series['on']['x'])} pts, off: {len(series['off']['x'])} pts)")


if __name__ == "__main__":
    main()
