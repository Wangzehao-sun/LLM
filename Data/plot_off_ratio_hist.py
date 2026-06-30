#!/usr/bin/env python3
"""Histogram of the OFF-policy token ratio, per selected step.

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors`, pulls the OFF-policy
tokens of each selected step, and histograms their ratio:

    ratio = exp(log_probs - old_log_probs)     # the PPO/IS ratio used in loss

OFF tokens = ``prefix_mask & response_mask`` (off-policy rows' valid response
tokens). Two layout modes:

  * ``--overlay`` (default): all selected steps on ONE axis, one colour each,
    so you can see how the off-ratio distribution shifts across training.
  * ``--subplots``: one panel per step.

A red dashed line marks ratio=1 (on-distribution reference).

Note: with no off injection (prefix_mask all-False) a step has zero off tokens
and is skipped.

Usage:
    python Data/plot_off_ratio_hist.py --save-dir LOGDIR/save_tensors
    python Data/plot_off_ratio_hist.py --save-dir DIR --steps 1,100,500 \
        --bins 80 --clamp 1e-3,10 --logx --output off_ratio_hist.png
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


def _pick_default_steps(all_steps: List[int], k: int = 3) -> List[int]:
    """When the user doesn't pass --steps, pick a spread (first / mid / last)."""
    if len(all_steps) <= k:
        return all_steps
    idxs = sorted({0, len(all_steps) // 2, len(all_steps) - 1})
    return [all_steps[i] for i in idxs]


def off_ratio_values(step: dict, clamp: Optional[tuple]) -> torch.Tensor:
    """1D tensor of ratio over this step's OFF tokens.

    ratio = exp(log_probs - old_log_probs), restricted to
    off tokens = prefix_mask & response_mask. Empty -> empty tensor.
    """
    needed = ("log_probs", "old_log_probs", "prefix_mask", "response_mask")
    if any(k not in step for k in needed):
        return torch.empty(0)
    off_tok = step["prefix_mask"] & step["response_mask"]
    return token_ratio(step, clamp=clamp)[off_tok]


def _bin_edges(all_vals: np.ndarray, bins: int, logx: bool) -> np.ndarray:
    """Shared bin edges across steps so overlaid histograms are comparable."""
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if logx:
        lo = max(lo, 1e-6)
        if hi <= lo:
            hi = lo * 10
        return np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, bins + 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--save-dir", required=True, help="Directory holding tensors_step_*.pt")
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers, e.g. '1,100,500'. "
             "Default: first / middle / last available step.",
    )
    ap.add_argument(
        "--clamp",
        type=str,
        default="1e-3,10",
        help="Clamp the ratio to lo,hi before histogramming (matches trainer). "
             "Pass 'none' to disable. Default '1e-3,10'.",
    )
    ap.add_argument("--bins", type=int, default=60, help="Number of histogram bins.")
    ap.add_argument("--logx", action="store_true", help="Log-scale ratio (x) axis + log-spaced bins.")
    ap.add_argument(
        "--density", action="store_true",
        help="Normalize to a probability density (useful when steps have different off-token counts).",
    )
    layout = ap.add_mutually_exclusive_group()
    layout.add_argument("--overlay", action="store_true", help="All steps on one axis (default).")
    layout.add_argument("--subplots", action="store_true", help="One panel per step.")
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/off_ratio_hist.png).")
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

    # Collect off-ratio arrays per step, dropping empty ones.
    per_step_vals = {}
    for st in want:
        v = off_ratio_values(by_step[st], clamp=clamp).numpy()
        if v.size == 0:
            print(f"[skip] step {st}: no off tokens")
            continue
        per_step_vals[st] = v
    if not per_step_vals:
        raise SystemExit("No off tokens in any selected step (off injection disabled?).")

    all_concat = np.concatenate(list(per_step_vals.values()))
    edges = _bin_edges(all_concat, args.bins, args.logx)

    use_subplots = args.subplots and not args.overlay
    out = args.output or os.path.join(args.save_dir, "off_ratio_hist.png")

    if use_subplots:
        n = len(per_step_vals)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
        for i, (st, v) in enumerate(per_step_vals.items()):
            ax = axes[i // ncols][i % ncols]
            ax.hist(v, bins=edges, density=args.density, alpha=0.85)
            ax.axvline(1.0, color="red", lw=0.9, ls="--", alpha=0.7)
            if args.logx:
                ax.set_xscale("log")
            ax.set_title(f"step {st}  (off tokens: {v.size})")
            ax.set_xlabel("off ratio = exp(log_probs - old_log_probs)")
            ax.set_ylabel("density" if args.density else "count")
            ax.grid(True, alpha=0.3)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for st, v in per_step_vals.items():
            ax.hist(
                v, bins=edges, density=args.density, alpha=0.45,
                label=f"step {st} (n={v.size})",
            )
        ax.axvline(1.0, color="red", lw=0.9, ls="--", alpha=0.7)
        if args.logx:
            ax.set_xscale("log")
        ax.set_xlabel("off ratio = exp(log_probs - old_log_probs)")
        ax.set_ylabel("density" if args.density else "count")
        ax.set_title("OFF-policy token ratio distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} ({len(per_step_vals)} step(s): {list(per_step_vals)})")


if __name__ == "__main__":
    main()
