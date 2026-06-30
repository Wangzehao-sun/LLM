#!/usr/bin/env python3
"""Load + normalize the per-step tensor dumps written by the trainer's
``save_tensors_dir`` channel (call site A in new_ray_trainer.py).

Each training step writes one file::

    {save_tensors_dir}/tensors_step_{step:07d}.pt   # torch.save(dict)

For call site A the dict carries these keys (plus "step")::

    old_log_probs   [B, L]  fp16   行动前 log prob (off 行可能是长 prompt 值)
    log_probs       [B, L]  fp16   更新后重算的 log prob
    prefix_mask     [B, L]  fp16   off-policy token 掩码 (0/1)
    se_mask         [B, L]  fp16   self-explain 掩码 (0/1)
    response_mask   [B, L]  fp16   response 有效 token 掩码 (0/1)
    reward_sum      list[B]        每条 rollout 的总 reward (来自 non_tensor)
    uid             list[B]        GRPO 分组 id (来自 non_tensor)

This module turns those raw dicts into a clean, typed structure so downstream
stat/plot code never has to worry about fp16 masks, list-vs-tensor reward_sum,
or missing keys:

    * masks  -> bool tensors
    * logp   -> fp32 tensors
    * reward_sum -> 1D fp32 tensor [B]
    * uid    -> list[str] length B

It also exposes the few derived quantities those stats usually want (per-token
ratio, on/off row masks, masked per-row reduction) so you can go straight to
variance/plotting.

Example
-------
    from load_saved_tensors import (
        load_steps, token_ratio, on_off_token_stats, ratio_on_off_over_steps,
    )

    steps = load_steps("/path/to/save_tensors")          # list of dicts, sorted by step

    # per-step on/off split (mirrors trainer's ratio_on / ratio_off_standard):
    for s in steps:
        ratio = token_ratio(s, clamp=(1e-3, 10))
        buckets = on_off_token_stats(ratio, s, split="row")
        # buckets = {"on": {mean,var,std,count}, "off_standard": {...}, "off_se": {...}}

    # or get plot-ready cross-step curves in one call:
    series = ratio_on_off_over_steps(steps)              # {"step":[...], "ratio_on_var":[...], ...}
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional

import torch

# Keys saved by call site A (see new_ray_trainer.py:2718). Kept here so callers
# can introspect / validate, but normalize_step also passes through any extra
# keys it doesn't recognise (e.g. if you later switch to call site B).
CALL_SITE_A_KEYS = [
    "old_log_probs",
    "log_probs",
    "prefix_mask",
    "reward_sum",
    "uid",
    "se_mask",
    "response_mask",
]

_LOGP_KEYS = ("old_log_probs", "log_probs")
_MASK_KEYS = ("prefix_mask", "se_mask", "response_mask")

_STEP_RE = re.compile(r"tensors_step_(\d+)\.pt$")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def iter_step_files(save_dir: str, pattern: str = "tensors_step_*.pt") -> List[str]:
    """Return step-dump file paths in ascending step order.

    Sorts by the integer parsed from the filename (not lexicographically), so
    step 100 comes after step 99.
    """
    paths = glob.glob(os.path.join(save_dir, pattern))

    def _step_of(p: str) -> int:
        m = _STEP_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else -1

    return sorted(paths, key=_step_of)


# ---------------------------------------------------------------------------
# Per-key normalization
# ---------------------------------------------------------------------------

def _to_reward_sum_1d(val: Any) -> torch.Tensor:
    """reward_sum was stored from a non_tensor numpy array of shape [B, 1] via
    ``list(...)`` -> a list of B length-1 arrays. Collapse back to a 1D [B]
    fp32 tensor. Also tolerates an already-tensor / flat-list form.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().to(torch.float32).reshape(-1)
    # list of arrays / scalars, or a numpy array
    t = torch.as_tensor(
        [float(x.item()) if hasattr(x, "item") and getattr(x, "size", 1) == 1
         else float(x) if not hasattr(x, "__len__")
         else float(x[0])
         for x in val],
        dtype=torch.float32,
    )
    return t


def _to_uid_list(val: Any) -> List[str]:
    """uid was stored via ``list(non_tensor_array)`` -> list of B ids
    (numpy str / python str). Normalize to plain str list."""
    return [str(x) for x in val]


def normalize_step(raw: Dict[str, Any], to_fp32: bool = True) -> Dict[str, Any]:
    """Turn one raw torch.load dict into clean typed fields.

    * logp keys   -> fp32 tensors (if to_fp32) [B, L]
    * mask keys   -> bool tensors [B, L]
    * reward_sum  -> fp32 tensor [B]
    * uid         -> list[str] length B
    * step        -> int
    * unknown keys are passed through unchanged (so call site B's
      entropys/responses/decoded_responses/extra_info also survive).

    Missing keys are simply omitted from the result (no crash), so partial
    dumps still load.
    """
    out: Dict[str, Any] = {}

    if "step" in raw:
        try:
            out["step"] = int(raw["step"])
        except (TypeError, ValueError):
            out["step"] = raw["step"]

    for key, val in raw.items():
        if key == "step":
            continue
        if key in _LOGP_KEYS:
            t = val.detach() if isinstance(val, torch.Tensor) else torch.as_tensor(val)
            out[key] = t.to(torch.float32) if to_fp32 else t
        elif key in _MASK_KEYS:
            t = val.detach() if isinstance(val, torch.Tensor) else torch.as_tensor(val)
            # stored as fp16 0.0/1.0 -> bool
            out[key] = t.to(torch.bool)
        elif key == "reward_sum":
            out[key] = _to_reward_sum_1d(val)
        elif key == "uid":
            out[key] = _to_uid_list(val)
        else:
            out[key] = val  # pass through (responses / entropys / extra_info / ...)

    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_step_file(path: str, to_fp32: bool = True, normalize: bool = True) -> Dict[str, Any]:
    """Load a single tensors_step_*.pt file. ``normalize=False`` returns the
    raw torch.load dict untouched."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not normalize:
        return raw
    return normalize_step(raw, to_fp32=to_fp32)


def load_steps(
    save_dir: str,
    pattern: str = "tensors_step_*.pt",
    to_fp32: bool = True,
    normalize: bool = True,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load every step dump in ``save_dir``, ascending by step.

    Returns a list of (normalized) dicts. ``limit`` caps how many files are
    loaded (the first N by step), handy for quick smoke tests.
    """
    paths = iter_step_files(save_dir, pattern)
    if limit is not None:
        paths = paths[:limit]
    out = []
    for p in paths:
        try:
            out.append(load_step_file(p, to_fp32=to_fp32, normalize=normalize))
        except Exception as e:  # noqa: BLE001 - keep loading the rest
            print(f"[skip] {os.path.basename(p)}: failed to load ({e})")
    return out


# ---------------------------------------------------------------------------
# Derived quantities (the stuff variance / plotting usually wants)
# ---------------------------------------------------------------------------

def token_ratio(step: Dict[str, Any], clamp: Optional[tuple] = None) -> torch.Tensor:
    """Per-token IS/PPO ratio = exp(log_probs - old_log_probs), shape [B, L].

    Requires both logp keys. ``clamp=(lo, hi)`` optionally clamps the ratio
    (mirrors the trainer's clamp(1e-3, 10) before its variance metric).
    """
    if "log_probs" not in step or "old_log_probs" not in step:
        raise KeyError("token_ratio needs both 'log_probs' and 'old_log_probs'")
    ratio = torch.exp(step["log_probs"] - step["old_log_probs"])
    if clamp is not None:
        ratio = torch.clamp(ratio, clamp[0], clamp[1])
    return ratio


def row_off_mask(step: Dict[str, Any]) -> torch.Tensor:
    """Per-row off-policy mask [B] bool: True if any response token is off-policy
    (prefix_mask hit). All-False when there's no off injection."""
    if "prefix_mask" not in step:
        raise KeyError("row_off_mask needs 'prefix_mask'")
    return step["prefix_mask"].any(dim=-1)


def masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the True positions of ``mask``, per row -> [B].

    Rows with an empty mask return 0 (matches the trainer's +1e-8 denom guard).
    """
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum(dim=-1) / (mask_f.sum(dim=-1) + 1e-8)


# ---------------------------------------------------------------------------
# on / off split (by prefix_mask) — mirrors the trainer's ratio_on / ratio_off
# metrics in new_ray_trainer.py:_compute_ratio_metrics
# ---------------------------------------------------------------------------

def split_row_masks(step: Dict[str, Any], include_se: bool = True) -> Dict[str, torch.Tensor]:
    """Per-row [B] bool buckets, split by prefix_mask (the trainer's convention).

    A row is OFF if any of its response tokens is off-policy
    (``prefix_mask.any(-1)``); otherwise ON. When ``se_mask`` is present the OFF
    rows are further split into ``off_se`` (self-explain) and ``off_standard``
    (everything else) -- exactly matching ``standard_off_policy_mask`` /
    ``se_mask`` in _compute_ratio_metrics.

    Returns a dict with keys: on, off, and (if se present) off_standard, off_se.
    With no off injection (your current config) ``off`` is all-False and ``on``
    is all-True.
    """
    if "prefix_mask" not in step:
        raise KeyError("split_row_masks needs 'prefix_mask'")
    off = step["prefix_mask"].any(dim=-1)
    on = ~off
    out = {"on": on, "off": off}
    if include_se and "se_mask" in step:
        se = step["se_mask"].any(dim=-1)
        out["off_se"] = off & se
        out["off_standard"] = off & (~se)
    return out


def masked_token_stat(
    values: torch.Tensor, token_mask: torch.Tensor, unbiased: bool = False
) -> Dict[str, torch.Tensor]:
    """mean / var / std / count of ``values`` over True positions of
    ``token_mask`` (pooled across all selected tokens).

    Mirrors verl ``masked_mean`` / ``masked_var``. ``unbiased=False`` (default)
    matches the trainer's variance metrics (it passes unbiased=False). Returns
    0s for an empty bucket instead of dividing by zero.
    """
    m = token_mask.to(values.dtype)
    n = m.sum()
    z = values.new_tensor(0.0)
    if n.item() <= 0:
        return {"mean": z, "var": z, "std": z, "count": n}
    mean = (values * m).sum() / n
    var = ((values - mean) ** 2 * m).sum() / n
    if unbiased and n.item() > 1:
        var = var * (n / (n - 1))
    return {"mean": mean, "var": var, "std": var.clamp_min(0).sqrt(), "count": n}


def on_off_token_stats(
    values: torch.Tensor,
    step: Dict[str, Any],
    response_mask_key: str = "response_mask",
    split: str = "row",
    unbiased: bool = False,
    include_se: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split per-token ``values`` [B, L] into on/off buckets and compute
    masked mean/var/std/count for each.

    ``split``:
      * ``"row"`` (default, matches trainer ratio_on/ratio_off metrics):
        off = rows with any off token (prefix_mask.any(-1)); within each bucket
        ALL response tokens are pooled. Buckets: on, off_standard, off_se
        (or just on/off when se_mask absent).
      * ``"token"`` (matches the loss off region): off tokens = prefix_mask &
        response_mask; on tokens = (~prefix_mask) & response_mask. Buckets:
        on, off.

    Returns ``{bucket_name: {mean, var, std, count}}``.
    """
    if response_mask_key not in step:
        raise KeyError(f"on_off_token_stats needs '{response_mask_key}'")
    resp = step[response_mask_key]
    if "prefix_mask" not in step:
        raise KeyError("on_off_token_stats needs 'prefix_mask'")
    prefix = step["prefix_mask"]

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    if split == "token":
        on_tok = (~prefix) & resp
        off_tok = prefix & resp
        out["on"] = masked_token_stat(values, on_tok, unbiased)
        out["off"] = masked_token_stat(values, off_tok, unbiased)
    elif split == "row":
        rows = split_row_masks(step, include_se=include_se)
        for name, row_mask in rows.items():
            if name == "off":
                # When se split is available we expose off_standard/off_se
                # instead of the merged off, mirroring the trainer.
                if include_se and "se_mask" in step:
                    continue
            token_mask = row_mask.unsqueeze(-1) & resp
            out[name] = masked_token_stat(values, token_mask, unbiased)
    else:
        raise ValueError(f"split must be 'row' or 'token', got {split!r}")
    return out


def ratio_on_off_over_steps(
    steps: List[Dict[str, Any]],
    clamp: Optional[tuple] = (1e-3, 10.0),
    split: str = "row",
    unbiased: bool = False,
    include_se: bool = True,
) -> Dict[str, List[float]]:
    """Cross-step on/off ratio stats, ready to plot.

    For every loaded step computes the per-token ratio
    (``exp(log_probs - old_log_probs)``, clamped like the trainer) and splits it
    into on/off buckets via :func:`on_off_token_stats`. Returns parallel lists
    keyed by metric name, e.g.::

        {"step": [...],
         "ratio_on_mean": [...], "ratio_on_var": [...], "ratio_on_std": [...],
         "ratio_off_standard_mean": [...], "ratio_off_standard_var": [...], ...}

    These mirror the trainer's ``batch/ratio_on_var`` /
    ``batch/ratio_off_standard_var`` so you can overlay offline-recomputed
    curves on the logged ones. Steps missing the needed keys are skipped.
    """
    series: Dict[str, List[float]] = {"step": []}
    for s in steps:
        if "log_probs" not in s or "old_log_probs" not in s:
            continue
        ratio = token_ratio(s, clamp=clamp)
        try:
            buckets = on_off_token_stats(
                ratio, s, split=split, unbiased=unbiased, include_se=include_se
            )
        except KeyError:
            continue
        series["step"].append(int(s.get("step", -1)))
        for bucket, stat in buckets.items():
            for stat_name in ("mean", "var", "std"):
                col = f"ratio_{bucket}_{stat_name}"
                series.setdefault(col, []).append(float(stat[stat_name].item()))
    return series


def stack_over_steps(steps: List[Dict[str, Any]], key: str) -> Dict[int, Any]:
    """Convenience for plotting: {step_int -> value[key]} across all loaded
    steps (skips any step missing that key)."""
    return {s["step"]: s[key] for s in steps if "step" in s and key in s}


# ---------------------------------------------------------------------------
# CLI: quick inspection of a save_tensors dir
# ---------------------------------------------------------------------------

def _summarize(save_dir: str, limit: Optional[int]) -> None:
    paths = iter_step_files(save_dir)
    if not paths:
        raise SystemExit(f"No tensors_step_*.pt found under {save_dir}")
    print(f"Found {len(paths)} step files in {save_dir}")
    show = paths if limit is None else paths[:limit]
    for p in show:
        try:
            s = load_step_file(p)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        step = s.get("step", "?")
        parts = []
        for k, v in s.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}{tuple(v.shape)}:{v.dtype}".replace("torch.", ""))
            elif isinstance(v, list):
                parts.append(f"{k}[list:{len(v)}]")
            else:
                parts.append(f"{k}[{type(v).__name__}]")
        print(f"  step {step}: " + ", ".join(parts))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Inspect / sanity-check a trainer save_tensors_dir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--save-dir", required=True, help="Directory holding tensors_step_*.pt")
    ap.add_argument("--limit", type=int, default=None, help="Only inspect the first N steps.")
    args = ap.parse_args()
    _summarize(args.save_dir, args.limit)


if __name__ == "__main__":
    main()
