"""Prefix-length scheduling for Prefix-RFT (arXiv:2507.01679).

WHAT THIS DECIDES. How much of the reference solution the off-policy row is handed, per
sample, per step. A FIXED fraction is two different experiments at two points in training:
early on the model cannot solve the question at all, so a long prefix is the only thing that
produces a usable trajectory; later the same prefix does the work the policy should be doing
itself and the off row stops teaching anything.

Prefix-RFT's answer is not a schedule on the fraction directly but on a WINDOW:

    ratio ~ Beta(alpha, beta),  mapped into [low_ctrl(step), high_ctrl(step)]

The two bounds are separately scheduled, so the window can widen, narrow, or slide. Sampling
inside it rather than taking its midpoint is what keeps a single step's batch covering a
spread of prefix lengths -- with a deterministic ratio every row in the step gets the same
amount of help, and the group has no contrast between "almost solved it alone" and "needed
most of the answer".

Ported from the upstream recipe's scheduler/global_step.py. Kept as a standalone module with
no torch or framework imports so it can be unit-tested on CPU, which is also why the
controllers take plain kwargs instead of an OmegaConf node -- build_prefix_sampler does that
translation at the one call site.
"""

from __future__ import annotations

import math

import numpy as np


class ConstController:
    """A bound that does not move. Use for the LOW bound to pin the window's floor at 0."""

    def __init__(self, init: float = 0.0, **kwargs):
        self.c = float(init)

    def value(self, **kwargs) -> float:
        return self.c

    def __str__(self) -> str:
        return f"Const({self.c})"


class LinearDecayController:
    """Linear interpolation from ``init`` to ``target`` over ``n_steps``, after a warmup.

    The warmup ramps 0 -> init over ``warmup_ratio * n_steps`` steps. It exists because step 0
    is the worst moment to hand over a long prefix: the policy has not moved at all yet, so
    the off row would be pure imitation with no exploration to correct it.
    """

    def __init__(self, init: float = 1.0, target: float = 0.0, n_steps: int = 10000,
                 warmup_ratio: float = 0.0, **kwargs):
        if init <= target:
            raise ValueError(f"LinearDecay needs init > target, got {init} <= {target}")
        self.init = float(init)
        self.target = float(target)
        self.n_steps = max(1, int(n_steps))
        self.warmup_steps = int(self.n_steps * float(warmup_ratio))

    def value(self, *, global_step: int, **kwargs) -> float:
        if global_step < self.warmup_steps:
            return (global_step / self.warmup_steps) * self.init
        after = global_step - self.warmup_steps
        if after >= self.n_steps:
            return self.target
        decay_ratio = 1.0 - (after / self.n_steps)
        return self.target + decay_ratio * (self.init - self.target)

    def __str__(self) -> str:
        return (f"LinearDecay({self.init}->{self.target}, n_steps={self.n_steps}, "
                f"warmup={self.warmup_steps})")


class CosineDecayController:
    """Cosine interpolation between ``init`` and ``target``, in EITHER direction.

    ``init > target`` decays, ``init < target`` rises -- the upstream class supports both, and
    the rising form is what makes a LOW bound that starts at 0 and climbs (a window that
    narrows from below rather than from above). Equal values are rejected rather than treated
    as a constant, because that is almost always a copy-paste mistake and ConstController
    already says it clearly.
    """

    def __init__(self, init: float = 0.9, target: float = 0.0, n_steps: int = 10000,
                 warmup_ratio: float = 0.0, **kwargs):
        if init == target:
            raise ValueError(
                f"CosineDecay needs init != target (got {init}); use const for a fixed bound"
            )
        self.init = float(init)
        self.target = float(target)
        self.n_steps = max(1, int(n_steps))
        self.warmup_steps = int(float(warmup_ratio) * self.n_steps)
        self.mode = "decay" if init > target else "rise"

    def value(self, *, global_step: int, **kwargs) -> float:
        if global_step < self.warmup_steps:
            return (global_step / self.warmup_steps) * self.init
        after = global_step - self.warmup_steps
        if after > self.n_steps:
            return self.target
        # Always 1 -> 0 over the horizon; the branch below decides which end it drives.
        decay_ratio = 0.5 * (1.0 + math.cos(math.pi * after / self.n_steps))
        if self.mode == "decay":
            return self.target + decay_ratio * (self.init - self.target)
        return self.init + (1.0 - decay_ratio) * (self.target - self.init)

    def __str__(self) -> str:
        return (f"Cosine{self.mode.capitalize()}({self.init}->{self.target}, "
                f"n_steps={self.n_steps}, warmup={self.warmup_steps})")


class DelayWrapper:
    """Hold a bound at ``delay_val`` for the first ``delay_steps``, then start its schedule.

    The wrapped controller sees a SHIFTED step count, so its horizon still spans n_steps once
    it begins -- delaying does not compress the schedule into the remaining steps.
    """

    def __init__(self, ctrl, delay_steps: int = 0, delay_val: float = 1.0, **kwargs):
        self.ctrl = ctrl
        self.delay_steps = int(delay_steps)
        self.delay_val = float(delay_val)

    def value(self, *, global_step: int, **kwargs) -> float:
        if global_step < self.delay_steps:
            return self.delay_val
        return self.ctrl.value(global_step=global_step - self.delay_steps, **kwargs)

    def __str__(self) -> str:
        return f"Delay({self.ctrl}, steps={self.delay_steps}, val={self.delay_val})"


class IDWrapper:
    """Pass-through, so the wrapper slot can be left unused without a None check."""

    def __init__(self, ctrl, **kwargs):
        self.ctrl = ctrl

    def value(self, **kwargs) -> float:
        return self.ctrl.value(**kwargs)

    def __str__(self) -> str:
        return str(self.ctrl)


CTRL_MAPPING = {
    "const": ConstController,
    "linear_decay": LinearDecayController,
    "cosine_decay": CosineDecayController,
}

CTRL_WRAPPER_MAPPING = {
    "delay": DelayWrapper,
    "id": IDWrapper,
}


class BetaSampler:
    """Draw a prefix ratio from Beta(alpha, beta) inside the two controllers' window.

    Returns ``(ratio, low, high)`` -- the bounds come back alongside the draw so the caller
    can log where the window actually was, which is the only way to tell a schedule that is
    not moving from one that is moving but being clamped.

    ``min``/``max`` rather than assuming ``low <= high``: the two bounds are independently
    scheduled and may cross mid-run (a rising low bound passing a decaying high one). Ordering
    them keeps the draw inside the interval either way instead of producing a ratio outside
    both.

    Beta(1,1) is uniform on the window -- the neutral default. alpha>1 biases toward the high
    end (longer prefixes), beta>1 toward the low end.
    """

    def __init__(self, low_ctrl, high_ctrl, alpha: float = 1.0, beta: float = 1.0,
                 rng: np.random.Generator | None = None, **kwargs):
        self.low_ctrl = low_ctrl
        self.high_ctrl = high_ctrl
        self.alpha = float(alpha)
        self.beta = float(beta)
        # An injectable Generator keeps the unit tests deterministic; None uses numpy's global
        # stream, which is what the dataloader workers already seed.
        self._rng = rng

    def value(self, *, global_step: int, **kwargs) -> tuple[float, float, float]:
        low = self.low_ctrl.value(global_step=global_step, **kwargs)
        high = self.high_ctrl.value(global_step=global_step, **kwargs)
        draw = self._rng.beta(self.alpha, self.beta) if self._rng is not None \
            else np.random.beta(self.alpha, self.beta)
        lo, hi = min(low, high), max(low, high)
        return lo + (hi - lo) * draw, lo, hi

    def __str__(self) -> str:
        return f"Beta(a={self.alpha}, b={self.beta}) in [{self.low_ctrl}, {self.high_ctrl}]"


def build_prefix_sampler(cfg, rng: np.random.Generator | None = None) -> BetaSampler:
    """Build the sampler from a rollout config node.

    Reads FLAT keys via ``.get`` rather than the upstream's nested ``prefix_low_ctrl.kwargs``:
    a Hydra config is struct mode, so a nested node that no script defines raises on access.
    That is exactly why the upstream recipe cannot be launched without reconstructing its
    entire config by hand -- fifteen keys it reads by attribute are absent from its own yaml.

    ``prefix_steps`` is the shared horizon for both bounds. Set it near the run's total step
    count; leaving the default while training for a fraction of it means the window reaches
    its target long before the run ends (or never gets there).
    """
    def _ctrl(side: str, default_type: str, default_init: float, default_target: float):
        ctrl_type = str(cfg.get(f"prefix_{side}_ctrl_type", default_type))
        if ctrl_type not in CTRL_MAPPING:
            raise ValueError(
                f"unknown prefix_{side}_ctrl_type {ctrl_type!r}; "
                f"expected one of {sorted(CTRL_MAPPING)}"
            )
        ctrl = CTRL_MAPPING[ctrl_type](
            init=float(cfg.get(f"prefix_{side}_ctrl_init", default_init)),
            target=float(cfg.get(f"prefix_{side}_ctrl_target", default_target)),
            n_steps=int(cfg.get("prefix_steps", 1000)),
            warmup_ratio=float(cfg.get("prefix_ctrl_warmup_ratio", 0.0)),
        )
        wrapper_type = cfg.get(f"prefix_{side}_ctrl_wrapper_type", None)
        if wrapper_type:
            wrapper_type = str(wrapper_type)
            if wrapper_type not in CTRL_WRAPPER_MAPPING:
                raise ValueError(
                    f"unknown prefix_{side}_ctrl_wrapper_type {wrapper_type!r}; "
                    f"expected one of {sorted(CTRL_WRAPPER_MAPPING)}"
                )
            ctrl = CTRL_WRAPPER_MAPPING[wrapper_type](
                ctrl,
                delay_steps=int(cfg.get(f"prefix_{side}_ctrl_delay_steps", 0)),
                delay_val=float(cfg.get(f"prefix_{side}_ctrl_delay_val", 1.0)),
            )
        return ctrl

    # Defaults: low pinned at min_prefix_ratio, high decaying from max_prefix_ratio to it, so
    # the window collapses onto the floor by the end of prefix_steps.
    lo_default = float(cfg.get("min_prefix_ratio", 0.0))
    hi_default = float(cfg.get("max_prefix_ratio", 1.0))
    return BetaSampler(
        low_ctrl=_ctrl("low", "const", lo_default, lo_default),
        high_ctrl=_ctrl("high", "cosine_decay", hi_default, lo_default),
        alpha=float(cfg.get("prefix_ctrl_alpha", 1.0)),
        beta=float(cfg.get("prefix_ctrl_beta", 1.0)),
        rng=rng,
    )
