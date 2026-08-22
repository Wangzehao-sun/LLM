"""Drive the joint decoder and hand its output back to the summarize-replacement path.

This is the seam between two things that otherwise know nothing about each other:
``NewActorRolloutRefWorker.generate_joint`` (the LIVE actor as student, a frozen teacher
constraining it, fused per token) and ``_summarize_replace_normal_step`` (score, filter,
select, splice). Keeping the seam in its own module is what limits the edit to the existing
trainer to a single extra branch plus a skipped log-prob call -- the load-bearing steps 5-7
of that function are shared by all three candidate sources rather than copied per source.

The student is the model being TRAINED, not a frozen copy, which is why the decode lives on
the actor's own worker rather than in a worker of its own: only that process holds the
actor's FSDP module.

WHAT IS DIFFERENT ABOUT THIS SOURCE. The other two paths produce candidate TEXT and then
have to ask a model what density that text had:

  * online  -- ``logp_wg.generate_sequences`` then ``logp_wg.compute_log_prob``
  * offline -- a parquet column, re-tokenised by the dataset, then ``compute_log_prob``

Joint decoding measures the density AS IT SAMPLES, so there is nothing to recover
afterwards and no second forward pass. Three consequences, and they are the reason this
path is worth its cost:

  1. ALIGNMENT IS EXACT. Token i of ``responses`` is the token ``off_log_probs[i]`` was
     computed on, by construction -- both are written in the same statement from the same
     draw. The offline path stores text and re-tokenises it, where a per-token density
     could silently slip by a token and produce a plausible, wrong gradient.
  2. THE DENSITY IS THE REAL ONE. ``agree`` samples from a TRUNCATED, renormalised
     student distribution, mu_t(v) = p_a(v) * 1[v in F_t] / Z_t. Asking a model for
     ``log p_a(v)`` afterwards would miss the ``1/Z_t`` factor and understate the
     density by exactly ``log Z_t``, biasing ``off_ratio`` upward by ``1/Z_t`` -- a factor
     of ~1.8 at the keep ratio of 0.55 seen in the offline sweeps, and worse the harder
     the constraint bites. Only the sampler can know this number.
  3. THE RATIO IS CHECKABLE. Because the student is the actor itself, the loss's numerator
     and this denominator are the same model under the same prompt at the same temperature,
     so before any optimizer step the two differ ONLY by the truncation:

         off_ratio = exp(log pi_theta - log mu_t) = Z_t

     So ``actor/off_ratio`` on the off rows must equal ``batch/sr_joint_keep_ratio``. That
     identity holds in the real configuration rather than a degenerate one, and it catches a
     token misalignment, a temperature mismatch, and a wrong normaliser at once.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def generate(
    *,
    worker_group,
    joint_config,
    student_prompts: torch.Tensor,
    teacher_prompts: torch.Tensor,
    pad_token_id: int,
    max_response_length: int,
    attention_mask_dtype: torch.dtype,
    make_masks,
    meta_info: dict,
    metrics: dict,
):
    """Decode one candidate per question and return it in the SR path's own shape.

    Args:
        worker_group: the ACTOR's worker group. The student is the live actor, so the
            decode has to run in the process that owns its FSDP module -- there is no
            separate joint-decode worker. See
            ``NewActorRolloutRefWorker.generate_joint``.
        joint_config: the ``joint_decode`` config node. Sent over with the request rather
            than read on the far side, because the actor worker is constructed with
            ``config.actor_rollout_ref`` and cannot see the top-level node.
        student_prompts: [W, L_short] left-padded. The prompt the LOSS will use, and the
            one the actor decodes under.
        teacher_prompts: [W, L_long] left-padded. May differ in length; only the
            generated suffix has to align, and it does because both models are fed the
            same sampled token each step.
        max_response_length: the width every response tensor in the batch must have, so
            the spliced row lines up with the on-policy rows.
        make_masks: ``generate_masks_from_input_ids`` from the trainer, passed in rather
            than imported to avoid a circular import -- and so masks and position_ids are
            built by exactly the same code the other two candidate sources use.
        meta_info: copied onto the returned DataProto, matching what the other two
            candidate sources do.
        metrics: mutated in place with ``batch/sr_joint_*`` entries.

    Returns:
        ``(cand_out, off_log_probs)`` where ``cand_out`` carries the same five keys
        ``generate_sequences`` would return (``prompts`` / ``responses`` / ``input_ids`` /
        ``attention_mask`` / ``position_ids``) and ``off_log_probs`` is [W, w], the
        per-token log density of each generated token.
    """
    device = student_prompts.device
    n_questions = student_prompts.size(0)

    student_attn, _ = make_masks(student_prompts, pad_token_id, attention_mask_dtype)
    teacher_attn, _ = make_masks(teacher_prompts, pad_token_id, attention_mask_dtype)
    request = DataProto.from_dict(
        tensors={
            "student_input_ids": student_prompts,
            "student_attention_mask": student_attn,
            "teacher_input_ids": teacher_prompts,
            "teacher_attention_mask": teacher_attn,
        },
        meta_info={
            "max_response_length": max_response_length,
            # Resolved to a plain container: meta_info is pickled to the workers, and an
            # OmegaConf node carrying interpolations would not survive that cleanly.
            "joint_decode_config": OmegaConf.to_container(joint_config, resolve=True),
        },
    )

    # Pad to the worker's dp size before dispatch. DP_COMPUTE_PROTO chunks by world_size
    # and auto-padding is off by default (protocol.py:45-58), so a row count that does not
    # divide evenly asserts inside chunk(). Under summarize_replace='wrong_only' that
    # count is "how many questions failed this step" -- an arbitrary integer, which is
    # exactly the 11-rows-into-4-GPUs failure this repo already hit once (commit 4ba0f1d).
    padded, pad_size = pad_dataproto_to_divisor(request, worker_group.world_size)
    reply = unpad_dataproto(worker_group.generate_joint(padded), pad_size=pad_size)

    cand_resp = reply.batch["responses"].to(device)
    off_log_probs = reply.batch["off_log_probs"].to(device)
    lengths = reply.batch["response_lengths"].to(device)

    # The padded rows were real decodes of duplicated prompts; dropping them is what
    # unpad_dataproto just did. What is left must match the questions we asked about.
    assert cand_resp.size(0) == n_questions, (
        f"joint decoder returned {cand_resp.size(0)} rows for {n_questions} questions"
    )
    assert off_log_probs.shape == cand_resp.shape, (
        f"density {tuple(off_log_probs.shape)} does not align with tokens "
        f"{tuple(cand_resp.shape)}; the importance ratio would be computed on the wrong "
        f"positions"
    )

    # Same five keys generate_sequences produces, so steps 5-7 cannot tell the sources
    # apart. The prompt segment is the TEACHER's, matching what the offline branch does
    # with its long prompt: downstream only compute_reward reads cand_out, and it reads
    # `responses`. The loss-time prompt is built separately by
    # _build_hybrid_off_policy_output from the short prompt.
    #
    # Masks and position_ids are computed over the WHOLE concatenated sequence rather
    # than per segment, as _build_hybrid_off_policy_output does: a left-padded prompt
    # followed by a right-padded response needs one continuous cumsum, and computing the
    # halves separately misnumbers the boundary whenever the prompt has no padding.
    cand_input_ids = torch.cat([teacher_prompts, cand_resp], dim=-1)
    cand_attn, cand_pos = make_masks(cand_input_ids, pad_token_id, attention_mask_dtype)
    cand_out = DataProto.from_single_dict({
        "prompts": teacher_prompts,
        "responses": cand_resp,
        "input_ids": cand_input_ids,
        "attention_mask": cand_attn,
        "position_ids": cand_pos,
    })
    cand_out.meta_info = dict(meta_info)

    # Diagnostics, averaged over questions. These are the three numbers that say whether
    # the agreement did any work:
    #   fallback_frac -- share of tokens where the intersection came out EMPTY, so one
    #       model decided alone. Near 1.0 and the run is plain single-model decoding.
    #   keep_ratio    -- Z_t, how much of the student's mass the teacher permitted. This
    #       is the continuous version of the above: fallback only fires when Z_t hits 0,
    #       so a low keep_ratio with zero fallback means the constraint is biting hard
    #       everywhere without ever failing outright -- invisible to fallback alone.
    #   student_logp  -- what the constraint cost, in the student's own terms.
    metrics["batch/sr_joint_questions"] = n_questions
    metrics["batch/sr_joint_fallback_frac"] = float(reply.batch["fallback_frac"].mean().item())
    metrics["batch/sr_joint_keep_ratio"] = float(reply.batch["teacher_keep_ratio"].mean().item())
    metrics["batch/sr_joint_student_logp"] = float(reply.batch["student_mean_logp"].mean().item())
    metrics["batch/sr_joint_mean_len"] = float(lengths.float().mean().item())
    # Rows that produced nothing. They will fail the reward check in step 5 and be
    # counted as sr_no_candidate, but that lumps them in with wrong answers, so the
    # distinct failure of "decoded zero tokens" is surfaced here.
    metrics["batch/sr_joint_empty"] = int((lengths == 0).sum().item())
    # Rows that ran to the cap without emitting EOS -- truncated mid-reasoning, so almost
    # certainly not a usable candidate. If this is high, max_new_tokens is too low.
    metrics["batch/sr_joint_truncated"] = int((lengths >= max_response_length).sum().item())

    return cand_out, off_log_probs


def select_questions(all_q, max_questions_per_step, metrics):
    """Cap the question count, and SAY what was dropped.

    The per-step cost is ``ceil(W / (batch_size * n_gpus)) * max_new_tokens`` sequential
    two-model forwards, linear in W. W is data-dependent under ``wrong_only`` -- a step
    where most rollouts failed sends far more questions here than a step where few did --
    so an uncapped run has no bound on how long a single step takes.

    The cap is logged rather than applied quietly: a truncated set that reports nothing
    reads downstream exactly like full coverage, and ``sr_accepted`` would drop with no
    indication why.
    """
    if not max_questions_per_step or len(all_q) <= max_questions_per_step:
        metrics["batch/sr_joint_dropped"] = 0
        return all_q
    dropped = len(all_q) - max_questions_per_step
    metrics["batch/sr_joint_dropped"] = dropped
    print(
        f"[joint_decode] {len(all_q)} questions this step, decoding the first "
        f"{max_questions_per_step} and DROPPING {dropped}. Those questions get no "
        f"replacement rollout this step (raise joint_decode.max_questions_per_step to "
        f"cover them, at a proportional increase in step time).",
        flush=True,
    )
    return all_q[:max_questions_per_step]
