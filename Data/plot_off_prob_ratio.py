#!/usr/bin/env python3
"""Scatter plot of (prob, ratio) for OFF-policy tokens, per selected step.

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors`, then for each selected
step pulls the OFF-policy tokens and scatters:

    x = prob  = exp(log_probs)                 # current-policy prob of the token
    y = ratio = exp(log_probs - old_log_probs) # the PPO/IS ratio used in loss

OFF tokens = ``prefix_mask & response_mask`` (off-policy rows' valid response
tokens). One subplot per selected step.

Note: with no off injection (prefix_mask all-False) a step has zero off tokens
and is reported empty / skipped.

Usage:
    python Data/plot_off_prob_ratio.py --save-dir LOGDIR/save_tensors
    python Data/plot_off_prob_ratio.py --save-dir DIR --steps 1,100,500 \
        --max-points 20000 --output off_prob_ratio.png
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from load_saved_tensors import load_steps, token_ratio


def _pick_default_steps(all_steps: List[int], k: int = 3) -> List[int]:
    """When the user doesn't pass --steps, pick a spread (first / mid / last)."""
    if len(all_steps) <= k:
        return all_steps
    idxs = sorted({0, len(all_steps) // 2, len(all_steps) - 1})
    return [all_steps[i] for i in idxs]


def _off_prob_ratio(step: dict, clamp: Optional[tuple]):
    """Return (prob, ratio) 1D tensors over this step's OFF tokens.

    prob  = exp(log_probs) restricted to off tokens
    ratio = exp(log_probs - old_log_probs) restricted to off tokens
    off tokens = prefix_mask & response_mask. Empty -> (empty, empty).
    """
    if "log_probs" not in step or "old_log_probs" not in step:
        return torch.empty(0), torch.empty(0)
    if "prefix_mask" not in step or "response_mask" not in step:
        return torch.empty(0), torch.empty(0)
    off_tok = step["prefix_mask"] & step["response_mask"]
    prob = torch.exp(step["log_probs"])[off_tok]
    ratio = token_ratio(step, clamp=clamp)[off_tok]
    return prob, ratio


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--save-dir", required=True, help="Directory holding tensors_step_*.pt")
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers to plot, e.g. '1,100,500'. "
             "Default: first / middle / last available step.",
    )
    ap.add_argument(
        "--clamp",
        type=str,
        default="1e-3,10",
        help="Clamp the ratio to lo,hi before plotting (matches trainer). "
             "Pass 'none' to disable. Default '1e-3,10'.",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Subsample to at most this many off tokens per step (0 = no cap).",
    )
    ap.add_argument("--alpha", type=float, default=0.2, help="Scatter point alpha.")
    ap.add_argument("--point-size", type=float, default=3.0, help="Scatter point size.")
    ap.add_argument(
        "--logy", action="store_true", help="Use a log scale on the ratio (y) axis."
    )
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/off_prob_ratio.png).")
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
        want = _pick_default_steps(avail)
    if not want:
        raise SystemExit("No valid steps to plot.")

    gen = torch.Generator().manual_seed(0)  # reproducible subsampling
    n = len(want)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)

    for ax_idx, st in enumerate(want):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        prob, ratio = _off_prob_ratio(by_step[st], clamp=clamp)
        n_off = prob.numel()
        if n_off == 0:
            ax.set_title(f"step {st}: no off tokens")
            ax.text(0.5, 0.5, "no off tokens", ha="center", va="center", transform=ax.transAxes)
            continue
        if args.max_points and n_off > args.max_points:
            sel = torch.randperm(n_off, generator=gen)[: args.max_points]
            prob, ratio = prob[sel], ratio[sel]
            shown = args.max_points
        else:
            shown = n_off
        ax.scatter(prob.numpy(), ratio.numpy(), alpha=args.alpha, s=args.point_size, linewidths=0)
        ax.axhline(1.0, color="red", lw=0.8, ls="--", alpha=0.6)  # ratio=1 reference
        if args.logy:
            ax.set_yscale("log")
        ax.set_xlabel("prob = exp(log_probs)")
        ax.set_ylabel("ratio = exp(log_probs - old_log_probs)")
        ax.set_title(f"step {st}  (off tokens: {n_off}, shown {shown})")
        ax.grid(True, alpha=0.3)

    # hide any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    out = args.output or os.path.join(args.save_dir, "off_prob_ratio.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} ({len(want)} step(s): {want})")


if __name__ == "__main__":
    main()
