#!/usr/bin/env python3
"""Scatter plot of (prob, ratio) for OFF- and/or ON-policy tokens, per step.

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors`, then for each selected
step pulls the chosen token bucket and scatters:

    x = prob  = exp(log_probs)                 # current-policy prob of the token
    y = ratio = exp(log_probs - old_log_probs) # the PPO/IS ratio used in loss

Token buckets (both gated by response_mask = valid response tokens):
    OFF = prefix_mask & response_mask          # off-policy rows' valid tokens
    ON  = (~prefix_mask) & response_mask        # on-policy rows' valid tokens

``--which`` picks what to draw: ``both`` (default) puts one COLUMN per step
with the on bucket on the top row and the off bucket on the bottom row (no
overlay); ``off`` / ``on`` draw a single bucket in a grid.

Note: with no off injection (prefix_mask all-False) the OFF bucket is empty for
every step; the ON bucket then holds all valid tokens.

Usage:
    python Data/plot_off_prob_ratio.py --save-dir LOGDIR/save_tensors
    python Data/plot_off_prob_ratio.py --save-dir DIR --which both --steps 1,100,500 \
        --max-points 20000 --output prob_ratio.png
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

# bucket -> (scatter colour, ratio=1 reference line colour)
_BUCKET_STYLE = {
    "off": ("tab:orange", "darkorange"),
    "on": ("tab:blue", "navy"),
}


def _pick_default_steps(all_steps: List[int], k: int = 3) -> List[int]:
    """When the user doesn't pass --steps, pick a spread (first / mid / last)."""
    if len(all_steps) <= k:
        return all_steps
    idxs = sorted({0, len(all_steps) // 2, len(all_steps) - 1})
    return [all_steps[i] for i in idxs]


def _bucket_token_mask(step: dict, bucket: str) -> Optional[torch.Tensor]:
    """Boolean [B, L] token mask for the requested bucket, or None if keys
    are missing. off = prefix_mask & response_mask; on = ~prefix_mask & response_mask."""
    if "prefix_mask" not in step or "response_mask" not in step:
        return None
    resp = step["response_mask"]
    if bucket == "off":
        return step["prefix_mask"] & resp
    return (~step["prefix_mask"]) & resp


def _prob_ratio(step: dict, clamp: Optional[tuple], bucket: str):
    """Return (prob, ratio) 1D tensors over this step's tokens in ``bucket``.

    prob  = exp(log_probs) restricted to the bucket's tokens
    ratio = exp(log_probs - old_log_probs) restricted to the bucket's tokens
    Empty / missing keys -> (empty, empty).
    """
    if "log_probs" not in step or "old_log_probs" not in step:
        return torch.empty(0), torch.empty(0)
    tok = _bucket_token_mask(step, bucket)
    if tok is None:
        return torch.empty(0), torch.empty(0)
    prob = torch.exp(step["log_probs"])[tok]
    ratio = token_ratio(step, clamp=clamp)[tok]
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
        "--which",
        choices=["both", "off", "on"],
        default="both",
        help="Which token bucket to scatter. both (default): one COLUMN per "
             "step, top row = on, bottom row = off (no overlay). off / on: a "
             "single bucket in a grid.",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Subsample to at most this many tokens per bucket per step (0 = no cap).",
    )
    ap.add_argument("--alpha", type=float, default=0.2, help="Scatter point alpha.")
    ap.add_argument("--point-size", type=float, default=3.0, help="Scatter point size.")
    ap.add_argument(
        "--logy", action="store_true", help="Use a log scale on the ratio (y) axis."
    )
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/prob_ratio_<which>.png).")
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

    buckets = ["on", "off"] if args.which == "both" else [args.which]

    gen = torch.Generator().manual_seed(0)  # reproducible subsampling

    def _scatter_bucket(ax, step_dict, bucket: str) -> int:
        """Scatter one bucket on ``ax``; return the number of tokens it had."""
        prob, ratio = _prob_ratio(step_dict, clamp=clamp, bucket=bucket)
        n_tok = prob.numel()
        if n_tok == 0:
            ax.text(0.5, 0.5, f"no {bucket} tokens", ha="center", va="center",
                    transform=ax.transAxes)
            return 0
        if args.max_points and n_tok > args.max_points:
            sel = torch.randperm(n_tok, generator=gen)[: args.max_points]
            prob, ratio = prob[sel], ratio[sel]
        color, _ = _BUCKET_STYLE[bucket]
        ax.scatter(
            prob.numpy(), ratio.numpy(), alpha=args.alpha, s=args.point_size,
            linewidths=0, color=color,
        )
        return n_tok

    def _style_ax(ax, st, bucket, n_tok):
        ax.axhline(1.0, color="red", lw=0.8, ls="--", alpha=0.6)  # ratio=1 reference
        if args.logy:
            ax.set_yscale("log")
        ax.set_xlabel("prob = exp(log_probs)")
        ax.set_ylabel("ratio = exp(log_probs - old_log_probs)")
        ax.set_title(f"step {st} — {bucket} ({n_tok} tokens)")
        ax.grid(True, alpha=0.3)

    n = len(want)
    if args.which == "both":
        # One COLUMN per step; row 0 = on (top), row 1 = off (bottom). No overlay.
        fig, axes = plt.subplots(
            2, n, figsize=(5.0 * n, 4.2 * 2), squeeze=False, sharex="col",
        )
        for col, st in enumerate(want):
            for row, bucket in enumerate(["on", "off"]):
                ax = axes[row][col]
                n_tok = _scatter_bucket(ax, by_step[st], bucket)
                _style_ax(ax, st, bucket, n_tok)
    else:
        # Single bucket: grid, one subplot per step.
        bucket = buckets[0]
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
        for ax_idx, st in enumerate(want):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            n_tok = _scatter_bucket(ax, by_step[st], bucket)
            _style_ax(ax, st, bucket, n_tok)
        for j in range(n, nrows * ncols):  # hide unused axes
            axes[j // ncols][j % ncols].axis("off")

    default_name = f"prob_ratio_{args.which}.png"
    out = args.output or os.path.join(args.save_dir, default_name)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} (which={args.which}, {len(want)} step(s): {want})")


if __name__ == "__main__":
    main()
