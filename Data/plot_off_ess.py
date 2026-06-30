#!/usr/bin/env python3
"""Plot off-policy ESS of REPLACED groups vs training step.

A "replaced group" is a GRPO group (rollouts sharing a ``uid``) that contains at
least one off-policy row -- i.e. summarize_replace / recycle injected a correct
off-policy candidate into it (its row has ``prefix_mask`` set). A typical group
is 8 rollouts = 7 on-policy + 1 injected off row.

For each replaced group we compute the TOKEN-LEVEL off-policy ESS over that
group's off-row tokens, exactly like the trainer's ``compute_ess``::

    r    = off_ratio = exp(log_probs - old_log_probs)   # on off tokens
    mask = prefix_mask & response_mask                  # off-row valid tokens
    ess  = (Σ r)^2 / (Σ r^2) / N_tok                    # normalized to [0, 1]

ESS near 1 means the off ratios are uniform (the injected trajectory's tokens
all carry similar importance weight); ESS near 0 means a few tokens dominate.

The curve is, per step, the MEAN ESS over all replaced groups, with an optional
±std band.

Reads the per-step tensor dumps written by the trainer's ``save_tensors_dir``
channel (call site A) via :mod:`load_saved_tensors` (needs the ``uid`` key).

Note: with no off injection (prefix_mask all-False) there are no replaced groups
and every step is skipped.

Usage:
    python Data/plot_off_ess.py --save-dir LOGDIR/save_tensors
    python Data/plot_off_ess.py --save-dir DIR --every 10 --band --output off_ess.png
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from load_saved_tensors import load_steps, token_ratio


def _token_ess(off_ratio_row_tokens: torch.Tensor, mask_row_tokens: torch.Tensor) -> float:
    """Trainer's compute_ess on a pooled set of off tokens (one group).

    ess = (Σ r)^2 / (Σ r^2) / N_tok, with r taken over masked tokens.
    Returns a [0, 1] scalar; nan when the group has no off tokens.
    """
    n = mask_row_tokens.sum()
    if n.item() <= 0:
        return float("nan")
    r = off_ratio_row_tokens * mask_row_tokens
    sum_r = r.sum()
    sum_r2 = (off_ratio_row_tokens ** 2 * mask_row_tokens).sum()
    ess = (sum_r ** 2) / (sum_r2 + 1e-8) / n
    return float(ess.item())


def replaced_group_ess(step: dict, clamp: Optional[tuple]) -> List[float]:
    """Per-replaced-group token-level off ESS for one step.

    Groups rows by ``uid``; a group is "replaced" if any of its rows is off
    (prefix_mask.any(-1)). For each such group the off-token mask is
    ``prefix_mask & response_mask`` restricted to the group's rows, pooled, and
    fed to :func:`_token_ess`. Returns a list of ESS values (one per replaced
    group). Empty when keys are missing or no group was replaced.
    """
    needed = ("log_probs", "old_log_probs", "prefix_mask", "response_mask", "uid")
    if any(k not in step for k in needed):
        return []
    prefix = step["prefix_mask"]               # [B, L] bool
    resp = step["response_mask"]               # [B, L] bool
    uids = step["uid"]                          # list[str] length B
    ratio = token_ratio(step, clamp=clamp)      # [B, L]
    off_tok = prefix & resp                     # [B, L] bool

    # group row indices by uid
    rows_by_uid = defaultdict(list)
    for i, u in enumerate(uids):
        rows_by_uid[u].append(i)

    ess_vals: List[float] = []
    row_is_off = prefix.any(dim=-1)             # [B] bool
    for u, rows in rows_by_uid.items():
        idx = torch.as_tensor(rows, dtype=torch.long)
        if not bool(row_is_off[idx].any()):
            continue  # not a replaced group
        grp_ratio = ratio[idx]                  # [g, L]
        grp_mask = off_tok[idx].to(grp_ratio.dtype)
        ess = _token_ess(grp_ratio, grp_mask)
        if not np.isnan(ess):
            ess_vals.append(ess)
    return ess_vals


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
        "--clamp",
        type=str,
        default="none",
        help="Clamp the off ratio to lo,hi before ESS (default 'none' to match "
             "the trainer's compute_ess, which does not clamp). e.g. '1e-3,10'.",
    )
    ap.add_argument(
        "--band", action="store_true",
        help="Shade a ±1 std band around the per-step mean ESS.",
    )
    ap.add_argument("--output", default=None, help="Output PNG (default: <save-dir>/off_ess.png).")
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

    step_x: List[int] = []
    mean_ess: List[float] = []
    std_ess: List[float] = []
    n_groups: List[int] = []
    for st in want:
        vals = replaced_group_ess(by_step[st], clamp=clamp)
        if not vals:
            print(f"[skip] step {st}: no replaced groups")
            continue
        arr = np.asarray(vals, dtype=float)
        step_x.append(st)
        mean_ess.append(float(arr.mean()))
        std_ess.append(float(arr.std()))
        n_groups.append(arr.size)
        print(f"[ok]   step {st}: {arr.size} replaced groups, mean ESS={arr.mean():.4f}")

    if not step_x:
        raise SystemExit("No replaced groups in any selected step (off injection disabled?).")

    mean_arr = np.asarray(mean_ess)
    std_arr = np.asarray(std_ess)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(step_x, mean_arr, marker="o", linewidth=1.8, label="mean off ESS (replaced groups)")
    if args.band:
        ax.fill_between(step_x, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2, label="±1 std")
    ax.set_ylim(0, 1.02)  # token-level ESS is normalized to [0, 1]
    ax.set_xlabel("training step")
    ax.set_ylabel("off-policy ESS (token-level, [0,1])")
    ax.set_title("Off-policy ESS of replaced groups vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = args.output or os.path.join(args.save_dir, "off_ess.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out} ({len(step_x)} steps; groups/step: {n_groups})")


if __name__ == "__main__":
    main()
