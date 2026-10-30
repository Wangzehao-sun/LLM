"""PICO policy loss.

One function, ``compute_token_on_off_sft_loss``, for loss_mode ``"<on rows>-<off rows>"``
(rl-rl / rl-sft / sft-sft / none-sft). Rows with ``prefix_mask=1`` are the summarize-replacement
candidates ("off" rows); every other row is an on-policy rollout ("on" rows).

  on rows,  rl   -- PPO dual-clip, the same formula as core_algos.compute_policy_loss.
  on rows,  sft  -- -log pi on tokens with a positive advantage.
  on rows,  none -- no loss.
  off rows, rl   -- PPO dual-clip on an off-policy ratio chosen by ``off_policy_reshape``:
                      clip           -- r = pi_theta / p_t, plain PPO; Z_t is NOT applied
                      clip_without_z -- the same r, and the clipped term is multiplied by Z_t:
                                        -Z_t * min(r A, clip(r) A)
                      clip_with_z    -- r = pi_theta / mu_t = Z_t * r, clipped as is
                      p_div_p_0.1    -- r = p / (p + 0.1) with p = pi_theta (LUFFY); no Z_t
                    Z_t = exp(off_log_z) is the per-token cost of joint decoding's teacher
                    constraint. Older comments (joint_sr.py, joint_decode.yaml) call 'clip' the
                    same as 'clip_without_z'; the code never was -- 'clip' ignores Z_t.
  off rows, sft  -- -log pi.

Copied from verl/custom/new_core_alg.py and trimmed to the options above; for those the loss,
its gradient and the logged metrics are unchanged, except that ``on_pg_clipfrac`` is now
computed (it was hard-coded to 0).
"""

import torch

import verl.utils.torch_functional as verl_F

ON_LOSS_TYPES = ("rl", "sft", "none")
OFF_LOSS_TYPES = ("rl", "sft")
OFF_RL_RESHAPES = ("clip", "clip_without_z", "clip_with_z", "p_div_p_0.1")
# Off-row SFT variants of verl/custom that PICO dropped. Refused rather than silently trained
# as plain SFT.
REMOVED_SFT_RESHAPES = ("high_mask", "low_mask", "low_sft_other_rl", "reweight_f", "sft_clip_adv", "weighted_sft")

# Off-row tokens whose old probability is below this count as low-confidence (logging only).
LOW_CONF_PROB = 0.2


def _dual_clip_loss(ratio, advantages, cliprange_low, cliprange_high, clip_ratio_c):
    """Per-token PPO dual-clip loss and whether the clip bound: max(-rA, -clip(r)A), floored at
    -c A where A < 0."""
    losses = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
    pessimistic = torch.max(losses, clipped)
    floored = torch.min(-advantages * clip_ratio_c, pessimistic)
    return torch.where(advantages < 0, floored, pessimistic), torch.gt(clipped, losses)


def compute_token_on_off_sft_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    prefix_mask,
    reward_mask,
    cliprange,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=None,
    clip_ratio_c=3.0,
    off_policy_strategy="rl-rl",
    off_policy_reshape="clip",
    off_log_z=None,
    all_max_clip=None,
):
    """Token-level loss over on rows and replaced (off) rows.

    Args:
        old_log_prob, log_prob, advantages, response_mask: [B, T], as in compute_policy_loss.
        prefix_mask: [B, T], 1 on the response tokens of replaced rows.
        reward_mask: [B, T], 0 on rows the trainer drops from the loss.
        cliprange / cliprange_low / cliprange_high: PPO clip range; the upper bound is
            1 + max(cliprange_high, cliprange). off_cliprange_high replaces cliprange_high for
            the off rows. clip_ratio_c: the dual-clip floor for negative advantages.
        off_policy_strategy: ``"<on>-<off>"`` loss types, e.g. "rl-rl".
        off_policy_reshape: the off-row ratio for an rl off-loss (see module docstring).
        off_log_z: [B, T] per-token log Z_t of joint decoding, or None.
        all_max_clip: drop tokens whose probability exceeds it. Since a probability is at most 1,
            a value >= 1 drops nothing; it only switches the loss-split metrics to count
            response tokens alone (otherwise they also see padding).

    Returns:
        dict with ``pg_loss`` (for backward), ``ppo_kl`` and detached logging values.
    """
    on_type, off_type = off_policy_strategy.split("-")
    assert on_type in ON_LOSS_TYPES and off_type in OFF_LOSS_TYPES, f"invalid loss mode {off_policy_strategy!r}"
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_cliprange_high = max(cliprange_high, cliprange)
    off_cliprange_high = max(off_cliprange_high if off_cliprange_high is not None else cliprange_high, cliprange)

    prefix_mask = prefix_mask.float()
    reward_mask = reward_mask.float()
    off_region = prefix_mask * response_mask * reward_mask
    on_region = (1.0 - prefix_mask) * response_mask * reward_mask
    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))

    # ---------------------------
    # on-policy rows
    # ---------------------------
    on_pg_clipfrac = torch.tensor(0.0)
    if on_type == "rl":
        on_losses, on_clipped = _dual_clip_loss(ratio, advantages, cliprange_low, on_cliprange_high, clip_ratio_c)
        on_pg_clipfrac = verl_F.masked_mean(on_clipped.float(), on_region)
    elif on_type == "sft":
        on_losses = -log_prob * (advantages > 0).float()
    else:
        on_losses = torch.zeros_like(log_prob)

    # ---------------------------
    # off-policy (replaced) rows. Split into an SFT and an RL part so the logged loss
    # composition adds up; only one of them is non-zero.
    # ---------------------------
    off_sft_losses = torch.zeros_like(log_prob)
    off_rl_losses = torch.zeros_like(log_prob)
    off_pg_clipfrac = torch.tensor(0.0)
    off_ratio_mean = torch.tensor(0.0)
    # Mean Z_t over the off tokens; 1.0 when there is none.
    off_z_mean = torch.tensor(1.0)
    if off_type == "sft":
        assert off_policy_reshape not in REMOVED_SFT_RESHAPES, f"off_policy_reshape={off_policy_reshape!r} was removed from PICO; off rows under an sft loss are trained with plain -log pi"
        off_sft_losses = -log_prob
    else:
        assert off_policy_reshape in OFF_RL_RESHAPES, f"off_policy_reshape must be one of {OFF_RL_RESHAPES}, got {off_policy_reshape!r}"
        off_ratio = ratio
        if off_policy_reshape == "clip_with_z" and off_log_z is not None:
            # Z_t folded into the denominator, so the clip sees it. Summed in log space so the
            # clamp bounds the whole exponent. mu_t is renormalised over a subset, so Z_t <= 1
            # shrinks the ratio, and negative-advantage tokens hit the 1-eps bound sooner.
            off_ratio = torch.exp(torch.clamp(log_prob - old_log_prob + off_log_z.detach(), min=-20.0, max=20.0))
        elif off_policy_reshape == "p_div_p_0.1":
            # LUFFY-style weight in place of the importance ratio; p carries the gradient.
            prob = torch.exp(log_prob)
            off_ratio = prob / (prob + 0.1)
        off_ratio_mean = verl_F.masked_mean(off_ratio, off_region).detach()

        off_rl_losses, off_clipped = _dual_clip_loss(off_ratio, advantages, cliprange_low, off_cliprange_high, clip_ratio_c)
        off_pg_clipfrac = verl_F.masked_mean(off_clipped.float(), off_region)

        if off_log_z is not None:
            # Z_t is a fixed per-token coefficient recorded while sampling: detached.
            z_t = torch.exp(off_log_z).detach()
            if off_policy_reshape == "clip_without_z":
                # Multiplied in after the clip, so the clip polices the policy shift alone.
                off_rl_losses = off_rl_losses * z_t
            off_z_mean = verl_F.masked_mean(z_t, off_region).detach()
    off_losses = off_sft_losses + off_rl_losses

    # ---------------------------
    # combine
    # ---------------------------
    off_pg_losses = off_losses * prefix_mask * reward_mask
    on_pg_losses = on_losses * (1.0 - prefix_mask) * reward_mask
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_region)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, on_region)

    pg_losses = off_pg_losses + on_pg_losses
    if all_max_clip is not None:
        keep = (torch.exp(log_prob) <= all_max_clip).float()
        response_mask = response_mask * keep
        pg_losses = pg_losses * keep
    loss_mask = response_mask * reward_mask
    pg_loss = verl_F.masked_mean(pg_losses, loss_mask)

    ppo_kl = verl_F.masked_mean(-torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0), response_mask)

    # ---------------------------
    # loss composition, logging only. Same denominator as pg_loss, so
    # on_loss_contrib + off_loss_contrib == pg_loss.
    # ---------------------------
    with torch.no_grad():
        # all_max_clip has already folded its mask into response_mask; without it these region
        # sums also run over padding.
        region_mask = response_mask if all_max_clip is not None else 1.0
        on_region_losses = on_pg_losses * region_mask
        off_mask = prefix_mask * reward_mask * region_mask
        denom = loss_mask.sum().clamp(min=1.0)

        on_loss_contrib = on_region_losses.sum() / denom
        off_loss_contrib = (off_losses * off_mask).sum() / denom
        off_sft_loss_contrib = (off_sft_losses * off_mask).sum() / denom
        off_rl_loss_contrib = (off_rl_losses * off_mask).sum() / denom

        # on vs off share of |loss|, in [0, 1], summing to 1
        abs_on_off = (on_loss_contrib.abs() + off_loss_contrib.abs()).clamp(min=1e-8)
        # SFT vs RL share within the off loss; off_has_loss is False when there is none
        abs_off_parts = off_sft_loss_contrib.abs() + off_rl_loss_contrib.abs()
        off_has_loss = abs_off_parts > 1e-8

        off_tokens = prefix_mask * loss_mask
        off_low_conf_token_frac = ((torch.exp(old_log_prob) < LOW_CONF_PROB).float() * off_tokens).sum() / off_tokens.sum().clamp(min=1.0)

        # on-row loss split by the sign of the advantage
        on_pos_adv_loss_contrib = (on_region_losses * (advantages > 0).float()).sum() / denom
        on_neg_adv_loss_contrib = (on_region_losses * (advantages < 0).float()).sum() / denom
        abs_pos_neg = (on_pos_adv_loss_contrib.abs() + on_neg_adv_loss_contrib.abs()).clamp(min=1e-8)

    return {
        "pg_loss": pg_loss,
        "ppo_kl": ppo_kl,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "off_ratio_mean": off_ratio_mean,
        "off_z_mean": off_z_mean,
        "on_loss_contrib": on_loss_contrib,
        "off_loss_contrib": off_loss_contrib,
        "off_sft_loss_contrib": off_sft_loss_contrib,
        "off_rl_loss_contrib": off_rl_loss_contrib,
        "on_loss_ratio": on_loss_contrib.abs() / abs_on_off,
        "off_loss_ratio": off_loss_contrib.abs() / abs_on_off,
        "off_has_loss": off_has_loss,
        "off_sft_ratio_in_off": off_sft_loss_contrib.abs() / abs_off_parts.clamp(min=1e-8),
        "off_rl_ratio_in_off": off_rl_loss_contrib.abs() / abs_off_parts.clamp(min=1e-8),
        "off_low_conf_token_frac": off_low_conf_token_frac,
        "on_pos_adv_loss_contrib": on_pos_adv_loss_contrib,
        "on_neg_adv_loss_contrib": on_neg_adv_loss_contrib,
        "on_pos_adv_loss_ratio": on_pos_adv_loss_contrib.abs() / abs_pos_neg,
        "on_neg_adv_loss_ratio": on_neg_adv_loss_contrib.abs() / abs_pos_neg,
    }
