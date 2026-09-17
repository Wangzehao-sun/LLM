"""CPU-only tests for the Prefix-RFT prefix-length scheduler.

Runs without torch: the scheduler module deliberately imports only math and numpy so these
can be exercised on a laptop, which is where the arithmetic mistakes live (off-by-one at the
warmup boundary, a window that inverts when the two bounds cross, a controller that keeps
extrapolating past its horizon).
"""

import numpy as np
import pytest

from verl.custom.prefix_scheduler import (
    BetaSampler,
    ConstController,
    CosineDecayController,
    DelayWrapper,
    LinearDecayController,
    build_prefix_sampler,
)


def test_const_ignores_step():
    c = ConstController(init=0.25)
    assert c.value(global_step=0) == 0.25
    assert c.value(global_step=10_000) == 0.25


def test_linear_decay_endpoints_and_clamp():
    c = LinearDecayController(init=0.8, target=0.2, n_steps=100)
    assert c.value(global_step=0) == pytest.approx(0.8)
    assert c.value(global_step=50) == pytest.approx(0.5)
    assert c.value(global_step=100) == pytest.approx(0.2)
    # Past the horizon it must SIT at the target, not keep extrapolating below it.
    assert c.value(global_step=10_000) == pytest.approx(0.2)


def test_linear_warmup_ramps_from_zero():
    c = LinearDecayController(init=1.0, target=0.0, n_steps=100, warmup_ratio=0.1)
    assert c.warmup_steps == 10
    assert c.value(global_step=0) == pytest.approx(0.0)
    assert c.value(global_step=5) == pytest.approx(0.5)
    # First step past warmup is the schedule's own start, i.e. init again.
    assert c.value(global_step=10) == pytest.approx(1.0)


def test_linear_rejects_non_decreasing():
    with pytest.raises(ValueError):
        LinearDecayController(init=0.2, target=0.8)


def test_cosine_decay_monotone_and_endpoints():
    c = CosineDecayController(init=0.9, target=0.0, n_steps=100)
    assert c.mode == "decay"
    assert c.value(global_step=0) == pytest.approx(0.9)
    assert c.value(global_step=50) == pytest.approx(0.45)
    assert c.value(global_step=100) == pytest.approx(0.0, abs=1e-9)
    assert c.value(global_step=500) == pytest.approx(0.0)
    vals = [c.value(global_step=s) for s in range(0, 101, 5)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), "decay must be monotone"


def test_cosine_rise_is_supported():
    """init < target rises -- the form that gives a LOW bound climbing off the floor."""
    c = CosineDecayController(init=0.0, target=0.6, n_steps=100)
    assert c.mode == "rise"
    assert c.value(global_step=0) == pytest.approx(0.0)
    assert c.value(global_step=100) == pytest.approx(0.6)
    vals = [c.value(global_step=s) for s in range(0, 101, 5)]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), "rise must be monotone"


def test_cosine_rejects_equal_endpoints():
    with pytest.raises(ValueError):
        CosineDecayController(init=0.5, target=0.5)


def test_delay_holds_then_shifts_horizon():
    inner = LinearDecayController(init=1.0, target=0.0, n_steps=100)
    d = DelayWrapper(inner, delay_steps=20, delay_val=0.7)
    assert d.value(global_step=0) == pytest.approx(0.7)
    assert d.value(global_step=19) == pytest.approx(0.7)
    # At the boundary the inner schedule starts from ITS step 0, so init.
    assert d.value(global_step=20) == pytest.approx(1.0)
    # The horizon is shifted, not compressed: half of n_steps after the delay.
    assert d.value(global_step=70) == pytest.approx(0.5)


def test_beta_sampler_stays_in_window():
    rng = np.random.default_rng(0)
    s = BetaSampler(ConstController(0.2), ConstController(0.8), alpha=1.0, beta=1.0, rng=rng)
    for _ in range(200):
        ratio, lo, hi = s.value(global_step=5)
        assert (lo, hi) == (0.2, 0.8)
        assert lo <= ratio <= hi


def test_beta_sampler_orders_crossed_bounds():
    """Independently scheduled bounds can cross; the draw must still land inside."""
    rng = np.random.default_rng(1)
    s = BetaSampler(ConstController(0.9), ConstController(0.1), rng=rng)
    for _ in range(50):
        ratio, lo, hi = s.value(global_step=0)
        assert (lo, hi) == (0.1, 0.9), "bounds must come back ordered"
        assert lo <= ratio <= hi


def test_beta_sampler_degenerate_window_is_a_point():
    rng = np.random.default_rng(2)
    s = BetaSampler(ConstController(0.4), ConstController(0.4), rng=rng)
    ratio, lo, hi = s.value(global_step=3)
    assert (lo, hi, ratio) == (0.4, 0.4, 0.4)


def test_beta_alpha_biases_high():
    """alpha >> beta should concentrate near the window's top; this is the knob's whole point."""
    rng = np.random.default_rng(3)
    high = BetaSampler(ConstController(0.0), ConstController(1.0), alpha=8.0, beta=1.0, rng=rng)
    low = BetaSampler(ConstController(0.0), ConstController(1.0), alpha=1.0, beta=8.0, rng=rng)
    mean_high = np.mean([high.value(global_step=0)[0] for _ in range(500)])
    mean_low = np.mean([low.value(global_step=0)[0] for _ in range(500)])
    assert mean_high > 0.7 and mean_low < 0.3


class _Cfg(dict):
    """Stand-in for an OmegaConf node: only .get is used by build_prefix_sampler."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


def test_build_defaults_collapse_window_onto_the_floor():
    s = build_prefix_sampler(_Cfg(min_prefix_ratio=0.1, max_prefix_ratio=0.9, prefix_steps=100),
                             rng=np.random.default_rng(4))
    _, lo0, hi0 = s.value(global_step=0)
    _, lo_end, hi_end = s.value(global_step=100)
    assert (lo0, hi0) == pytest.approx((0.1, 0.9))
    # By the horizon the high bound has decayed to the floor: no prefix left.
    assert (lo_end, hi_end) == pytest.approx((0.1, 0.1))


def test_build_rejects_unknown_types():
    with pytest.raises(ValueError, match="prefix_high_ctrl_type"):
        build_prefix_sampler(_Cfg(prefix_high_ctrl_type="bogus"))
    with pytest.raises(ValueError, match="prefix_low_ctrl_wrapper_type"):
        build_prefix_sampler(_Cfg(prefix_low_ctrl_wrapper_type="bogus"))


def test_build_honours_wrapper():
    s = build_prefix_sampler(
        _Cfg(min_prefix_ratio=0.0, max_prefix_ratio=0.8, prefix_steps=100,
             prefix_high_ctrl_wrapper_type="delay",
             prefix_high_ctrl_delay_steps=10, prefix_high_ctrl_delay_val=0.8),
        rng=np.random.default_rng(5),
    )
    assert s.value(global_step=0)[2] == pytest.approx(0.8)
    assert s.value(global_step=9)[2] == pytest.approx(0.8)
    assert s.value(global_step=10)[2] == pytest.approx(0.8)  # inner starts at init
    assert s.value(global_step=110)[2] == pytest.approx(0.0)  # horizon shifted by 10
