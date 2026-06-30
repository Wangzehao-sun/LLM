#!/usr/bin/env python3
"""Plot OFF-policy ratio variance vs training step, one curve per clip upper bound.

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors`. For every step it computes
the RAW off-policy token ratio::

    ratio = exp(log_probs - old_log_probs)       # the PPO/IS ratio used in loss
    OFF tokens = prefix_mask & response_mask      # off-policy rows' valid tokens

then, for each requested clip upper bound, clamps the ratio to ``max=upper``
(mirroring the trainer's ``off_max_clip``) and computes its variance over the
step's off tokens. Variance uses the same masked, biased formula as the
trainer's metric (``masked_var(..., unbiased=False)`` = mean((x-mean)^2)).

The result is one line per clip upper bound (e.g. 1.2 and 1.28), x = step,
y = clipped off-ratio variance -- so you can see how tighter clipping shrinks
the off-ratio variance across training.

Note: clipping only caps the UPPER tail, so a smaller upper bound gives a
smaller (or equal) variance at every step. An optional ``--clip-lower`` also
caps the bottom (off by default; ratio is already > 0).

Usage:
    python Data/plot_off_ratio_var.py --save-dir LOGDIR/save_tensors
    python Data/plot_off_ratio_var.py --save-dir DIR --clip-uppers 1.2,1.28 \
        --every 10 --output off_ratio_var.png
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


def off_ratio_raw(step: dict) -> torch.Tensor:
    """RAW (unclamped) ratio over this step's OFF tokens, 1D.

    ratio = exp(log_probs - old_log_probs), restricted to
    off tokens = prefix_mask & response_mask. Empty -> empty tensor.
    """
    needed = ("log_probs", "old_log_probs", "prefix_mask", "response_mask")
    if any(k not in step for k in needed):
        return torch.empty(0)
    off_tok = step["prefix_mask"] & step["response_mask"]
    return token_ratio(step, clamp=None)[off_tok]


def clipped_var(values: torch.Tensor, upper: float, lower: Optional[float], unbiased: bool) -> float:
    """Variance of ``values`` after clamping to [lower?, upper].

    Matches the trainer's metric口径: biased by default
    (var = mean((x-mean)^2)); unbiased=True applies the Bessel correction.
    Empty input -> nan (caller drops it from the curve).
    """
    if values.numel() == 0:
        return float("nan")
    v = torch.clamp(values, max=upper)
    if lower is not None:
        v = torch.clamp(v, min=lower)
    mean = v.mean()
    var = ((v - mean) ** 2).mean()
    if unbiased and v.numel() > 1:
        var = var * (v.numel() / (v.numel() - 1))
    return float(var.item())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--save-dir", required=True, help="Directory holding tensors_step_*.pt")
    ap.add_argument(
        "--clip-uppers",
        type=str,
        default="1.2,1.28",
        help="Comma-separated clip upper bounds, one curve each. Default '1.2,1.28'. "
             "Each clamps the off ratio to max=upper before computing variance.",
    )
    ap.add_argument(
        "--clip-lower",
        type=float,
        default=None,
        help="Optional lower clamp applied to every curve (default: none).",
    )
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers to include, e.g. '1,100,500'. "
             "Default: ALL steps (optionally thinned by --every).",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="When --steps is not given, keep every Nth step (1 = all).",
    )
    ap.add_argument(
        "--unbiased",
        action="store_true",
        help="Use the unbiased (Bessel-corrected) variance. Default matches the "
             "trainer's metric (biased).",
    )
    ap.add_argument("--logy", action="store_true", help="Log-scale the variance (y) axis.")
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/off_ratio_var.png).")
    args = ap.parse_args()

    try:
        uppers = [float(x) for x in args.clip_uppers.split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit("--clip-uppers must be comma-separated floats, e.g. '1.2,1.28'")
    if not uppers:
        raise SystemExit("--clip-uppers is empty")

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

    # For each step, grab the raw off ratio once; reuse it for every clip upper.
    step_x: List[int] = []
    raw_per_step: List[torch.Tensor] = []
    for st in want:
        v = off_ratio_raw(by_step[st])
        if v.numel() == 0:
            print(f"[skip] step {st}: no off tokens")
            continue
        step_x.append(st)
        raw_per_step.append(v)
    if not step_x:
        raise SystemExit("No off tokens in any selected step (off injection disabled?).")

    # Build one variance curve per clip upper bound.
    curves = {}
    for upper in uppers:
        curves[upper] = [
            clipped_var(v, upper, args.clip_lower, args.unbiased) for v in raw_per_step
        ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for upper in uppers:
        ax.plot(step_x, curves[upper], marker="o", linewidth=1.8,
                label=f"clip max={upper:g}")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ylabel = "off-ratio variance" + (" (unbiased)" if args.unbiased else "")
    ax.set_ylabel(ylabel)
    lo_str = "none" if args.clip_lower is None else f"{args.clip_lower:g}"
    ax.set_title(f"OFF-policy ratio variance vs step  (clip lower={lo_str})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = args.output or os.path.join(args.save_dir, "off_ratio_var.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} ({len(step_x)} steps, clip uppers {uppers})")


if __name__ == "__main__":
    main()
