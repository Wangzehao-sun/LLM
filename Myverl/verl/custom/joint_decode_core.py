"""Two-model joint decoding: the compute core, shared by offline eval and training.

Every token is decided by BOTH models. Two families of rule, selected by ``fuse``:

* ``linear`` / ``contrastive`` / ``max`` -- blend the two next-token distributions in
  log space, then sample from the result (contrastive decoding / proxy tuning /
  DExperts).
* ``agree`` -- do not blend. The TEACHER (model B) constrains which tokens are allowed:
  a candidate must sit in both models' top-k and clear each side's probability floor.
  The STUDENT (model A) then samples among the survivors, so the teacher steers
  direction while the student picks the token it finds most natural. When nothing
  survives there is no agreement to honour and the step falls back to one model's full
  distribution.

This lives in the package rather than in ``Data/`` because it now has two callers:
``Data/joint_decode.py`` (offline evaluation, under torchrun) and
``verl/custom/joint_decode_worker.py`` (a Ray worker that produces the
summarize-replacement candidate during training). The rule that decided the split is
CLAUDE.md's: reusable package code belongs here, and ``Data/`` scripts may import from
``verl`` but never the reverse. Nothing in this module touches argparse, parquet, Ray,
or ``torch.distributed`` -- prompts arrive already tokenised, so both callers reach it
without an adapter.

It deliberately does NOT go through verl's rollout stack. vLLM is entered once per
request (``vllm_rollout_spmd.py:304`` calls ``LLM.generate()``), so there is no
per-token hook to fuse into, and the pinned ``vllm<=0.8.5`` (setup.py:51) runs the V1
engine, which dropped per-request ``logits_processors``. The one token-level loop that
does exist, ``naive_rollout.py:68-100``, re-forwards the whole prefix every step (no KV
cache, O(n^2)) -- unusable at these response lengths. So this is a plain HF decode loop
WITH a KV cache.

The price is throughput: no paged attention, no CUDA graph, and two forwards per token.
That is what confines the training-time path to a small number of questions per step;
see ``joint_sr.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
def fuse_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, mode: str, weight: float) -> torch.Tensor:
    """Combine two next-token logit rows into one, in LOG-PROBABILITY space.

    Normalising first is load-bearing for two of the three modes:

    * ``max`` needs it outright -- an elementwise max over raw logits compares two
      models' arbitrary offsets against each other, which is meaningless.
    * ``linear`` / ``contrastive`` are affine, so a per-row offset survives into the
      result as another per-row offset and softmax cancels it; normalising is
      mathematically a no-op there. It is still done, because it makes
      ``weight == 0`` return exactly A's log-probs (so the degenerate case is bit-
      comparable against a single-model run) and keeps the returned scores on one
      interpretable scale for all three modes.

    Returns log-probabilities, not logits.
    """
    log_pa = F.log_softmax(logits_a.float(), dim=-1)
    if weight == 0.0:
        return log_pa
    log_pb = F.log_softmax(logits_b.float(), dim=-1)

    if mode == "linear":
        # Geometric mixture of the two distributions (arithmetic mean in log space).
        return (1.0 - weight) * log_pa + weight * log_pb
    if mode == "contrastive":
        # Extrapolate along B-minus-A. weight>0 pushes AWAY from A towards B and
        # may leave the interval spanned by the two models -- that is the point
        # (proxy tuning / DExperts), but it also means the result can be sharper
        # than either input, so expect it to need a smaller weight than 'linear'.
        return log_pa + weight * (log_pb - log_pa)
    if mode == "max":
        # Ignores `weight` by construction: an elementwise max has no mixing knob.
        return torch.maximum(log_pa, log_pb)
    raise ValueError(f"unknown fuse mode {mode!r}")


def agree_select(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    student_top_k: int,
    teacher_top_k: int,
    student_min_prob: float,
    teacher_min_prob: float,
    temperature: float,
    top_p: float,
    fallback: str = "teacher",
    narrow_counter: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Let the teacher constrain the direction and the student pick within it.

    Model B is the teacher (it steers), model A is the student (it chooses). The
    asymmetry is the point, so the two roles use the mask and the scores
    differently:

      * CONSTRAINT (teacher-led) -- a candidate must be in BOTH models' top-k, and
        clear ``teacher_min_prob`` under B and ``student_min_prob`` under A. The two k's
        are separate knobs because they mean different things: the teacher's bounds how
        much of the vocabulary it permits at all, the student's bounds how far down its
        own ranking it will look for something permitted. The teacher's floor sets how
        wide the allowed region is; the student's floor only drops tokens it would
        itself be very reluctant to emit.
      * CHOICE (student-led) -- among the survivors, sample by the STUDENT's
        distribution. Ranking by the teacher instead would collapse this to the
        teacher's own argmax almost every step, leaving the student a veto and no
        say -- which is not the intended division of labour.
      * FALLBACK -- when nothing survives there is no agreement to honour, and
        ``fallback`` decides who breaks the tie. These are different experiments, not
        two spellings of one:

          'teacher' -- defer to the model steering direction. The teacher gets to
              intervene exactly where the student disagrees with it, which is the
              aggressive reading of "the teacher steers".
          'student' -- let the student continue on its own. The teacher then only ever
              CONSTRAINS, never overrides: agreement narrows the student's choices and
              nothing else. Text stays in the student's voice throughout, and a run
              whose fallback rate is high degenerates toward plain student decoding
              rather than plain teacher decoding -- which also flips how the fallback
              column should be read.

    Sampling, not argmax, in both branches: an argmax fallback would pin every
    disagreeing position to one fixed token, so a sequence's diversity would decay
    with each fallback and ``n_samples > 1`` would stop meaning anything.
    ``temperature == 0`` still gives greedy behaviour on both paths.

    Returns (token ids [B, 1], fell_back [B] bool, log_pa [B, vocab], z_t [B],
    log_q [B]) so the caller can report how often the agreement actually bound
    anything, reuse the student's log-probs, track how much student mass survived the
    constraint, and record the density each token was really drawn from. Under
    ``fallback='teacher'`` a fallback rate near 1.0 means the run is effectively plain
    teacher decoding; under ``'student'`` it means plain student decoding. Either way the
    intersection is doing no work.
    """
    log_pa = F.log_softmax(logits_a.float(), dim=-1)
    log_pb = F.log_softmax(logits_b.float(), dim=-1)
    vocab = log_pa.size(-1)

    # Membership masks over the FULL vocab, so the intersection is a plain AND
    # rather than a set-of-ids comparison (which would need a loop per row).
    # The two k's are separate because the roles are: the teacher's sets how much of
    # the vocabulary it is willing to permit at all, while the student's sets how far
    # down its own ranking it is willing to look for something permitted. Raising only
    # the student's therefore lets it reach further for an agreed token instead of
    # falling back, without widening what the teacher allows.
    in_a = torch.zeros_like(log_pa, dtype=torch.bool)
    in_b = torch.zeros_like(log_pb, dtype=torch.bool)
    in_a.scatter_(-1, log_pa.topk(min(student_top_k, vocab), dim=-1).indices, True)
    in_b.scatter_(-1, log_pb.topk(min(teacher_top_k, vocab), dim=-1).indices, True)

    # Compare in log space; a floor of 0 becomes -inf, which every finite log-prob
    # clears, so it disables that side's threshold.
    def _floor(value: float) -> torch.Tensor:
        return torch.log(torch.tensor(value, device=log_pa.device, dtype=log_pa.dtype))

    eligible = (
        in_a & in_b
        & (log_pa >= _floor(student_min_prob))
        & (log_pb >= _floor(teacher_min_prob))
    )
    fell_back = ~eligible.any(dim=-1)

    # Z_t: how much of the STUDENT's probability mass the teacher left standing, i.e.
    # sum of p_a over the eligible set. This is the renormaliser in
    # mu_t(v) = p_a(v) * 1[v in F_t] / Z_t, and the most direct measure of how hard the
    # constraint is biting -- 1.0 means the teacher permitted everything the student
    # cared about, near 0 means it kept only tokens the student thought unlikely.
    # Read off the mask rather than the sampled token, so it describes the constraint
    # itself and not the draw. Fallback rows get 0: nothing survived, so no mass did.
    z_t = torch.where(eligible, log_pa.exp(), torch.zeros_like(log_pa)).sum(dim=-1)
    z_t = torch.where(fell_back, torch.zeros_like(z_t), z_t)

    # Student chooses inside the allowed set. Renormalising is not needed --
    # sample_next softmaxes and the -inf entries drop out -- but a row masked to all
    # -inf would give softmax nothing to sample, which is why the rows with no survivor
    # are routed to a full, unmasked distribution instead.
    student_scores = log_pa.masked_fill(~eligible, float("-inf"))
    fallback_scores = log_pb if fallback == "teacher" else log_pa
    scores = torch.where(fell_back.unsqueeze(-1), fallback_scores, student_scores)

    # top_k is already enforced by the intersection above, so only top_p is left to
    # apply here; passing top_k again would re-cut the (already tiny) survivor set.
    chosen = sample_next(scores, temperature, top_p, -1, narrow_counter=narrow_counter)
    # The density the token was actually drawn from. Computed from the SAME `scores`
    # tensor sample_next just consumed, which is what makes one expression cover both
    # branches: on a constrained step `scores` is masked to F_t, so its log-normaliser
    # IS log Z_t and this reduces to log p_a(v) - log Z_t; on a fallback step `scores`
    # is the full fallback distribution, so it needs no special case (and in
    # particular no log(Z_t = 0) = -inf).
    log_q = sampled_log_prob(scores, chosen, temperature)
    # log_pa goes back to the caller so the per-token confidence stat can reuse it
    # instead of recomputing a [B, vocab] log_softmax every step.
    return chosen, fell_back, log_pa, z_t, log_q


def sample_next(scores: torch.Tensor, temperature: float, top_p: float, top_k: int,
                *, top_p_candidates: int = 2048,
                narrow_counter: torch.Tensor | None = None) -> torch.Tensor:
    """Pick the next token id per row. Mirrors naive_rollout.py:77-88, plus top-p.

    ``temperature == 0`` means greedy, which is what makes the output a
    deterministic fingerprint of the weights (the trick smoke_dual_vllm.py:21-24
    relies on).

    The top-p nucleus is taken from the top ``top_p_candidates`` logits instead of a
    full sort. Sorting the whole vocabulary is the single largest allocation in a step:
    the index tensor is int64, so at a 152k vocab it is 8 bytes per element -- more than
    both log_softmax results together -- and a peaked next-token distribution puts the
    0.95 nucleus in its first few dozen entries, so the rest is sorted only to be thrown
    away.

    This makes the cap an implicit ``top_k``: the sampled set is
    ``top_p AND top_p_candidates``, which is the standard way both flags compose.
    Whenever the candidates do not already cover ``top_p`` the effective nucleus is
    narrower than asked for, so pass ``narrow_counter`` (a scalar GPU tensor) to have
    those rows tallied -- accumulated on-device, without the host sync a branch here
    would cost, and reported once at the end.
    """
    if temperature == 0.0:
        return scores.argmax(dim=-1, keepdim=True)

    scores = scores / temperature
    if top_k > 0:
        kth = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        k = min(top_p_candidates, scores.size(-1))
        top_probs, top_idx = scores.softmax(dim=-1).topk(k, dim=-1)
        cumulative = top_probs.cumsum(dim=-1)
        if narrow_counter is not None:
            narrow_counter += (cumulative[:, -1] < top_p).sum()
        # Shift by one so the token that crosses the threshold is kept: otherwise a
        # top_p below the top token's probability would mask everything.
        keep = cumulative - top_probs <= top_p
        kept_scores = torch.where(keep, scores.gather(-1, top_idx),
                                  torch.full_like(top_probs, float("-inf")))
        scores = torch.full_like(scores, float("-inf")).scatter_(-1, top_idx, kept_scores)
    return torch.multinomial(scores.softmax(dim=-1), num_samples=1)


def sampled_log_prob(scores: torch.Tensor, chosen: torch.Tensor, temperature: float) -> torch.Tensor:
    """log q(chosen) under the distribution ``sample_next`` draws from.

    ``sample_next`` samples from ``softmax(scores / T)``, so

        log q(v) = scores[v]/T - logsumexp(scores/T)

    and that one line covers every decode mode, because each mode differs only in what
    it puts in ``scores``: the eligible-masked student log-probs, a full fallback
    distribution, or a fused blend. Rows masked to ``-inf`` contribute nothing to the
    logsumexp, which is exactly the renormalisation an explicit ``/ Z_t`` would do.

    Two conventions, both chosen to line up with how the trainer computes the ratio's
    NUMERATOR rather than to be locally elegant:

    * Temperature IS applied. ``compute_log_prob`` divides logits by
      ``rollout.temperature`` (fsdp_workers_new.py:772 -> new_dp_actor.py:122), so a
      T=1 denominator against a T-scaled numerator would leave a systematic,
      T-dependent bias in ``off_ratio``. At T=1 this reduces to the plain log-prob.
    * top_p is NOT applied, matching verl's convention everywhere else (nothing in the
      framework renormalises a density for nucleus truncation). The returned value is
      therefore slightly LOWER than the true post-truncation density for tokens inside
      the nucleus, and non-zero for tokens outside it -- which cannot happen, since
      those were never sampled.

    ``temperature == 0`` makes sampling a point mass at the argmax, so its log-density
    is 0 -- returned directly rather than falling out of the formula, which would
    divide by zero.
    """
    if temperature == 0.0:
        return torch.zeros(scores.size(0), device=scores.device, dtype=torch.float32)
    scaled = scores.float() / temperature
    return scaled.gather(-1, chosen).squeeze(-1) - torch.logsumexp(scaled, dim=-1)


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def last_logit_kwargs(model) -> dict:
    """Ask the model for the LAST position's logits only, if it supports it.

    Prefill computes logits for every prompt position, and only the last one is
    ever read. At a 152k vocab that waste dominates the memory profile:
    [8, 4096, 151936] in bf16 is ~10GB per model, ~20GB for the pair, versus 2.4MB
    for the single position actually used -- and it scales with batch size, so 16
    rows would throw away ~40GB. Dropping it moves the binding constraint to the KV
    cache (~2.4GB per row for a 4B pair at 8192), which lifts the usable batch on an
    80GB card from about 8 to about 20.

    The flag exists for exactly this, but was renamed mid-flight
    (num_logits_to_keep -> logits_to_keep), and older versions have neither. So
    inspect the signature instead of guessing, and return {} when it is absent --
    the maths is identical either way, only the footprint changes.
    """
    import inspect

    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    # **kwargs-only signatures cannot be probed; skipping is the safe answer.
    return {}


@torch.no_grad()
def joint_generate(model_a, model_b, batch_a, batch_b, *, eos_ids, pad_id, args):
    """Decode one batch under both models, fusing per token.

    ``batch_a`` / ``batch_b`` are the two tokenised prompt sides. They may differ
    in content AND length (A can see the bare question while B sees a richer
    prompt) -- only the GENERATED suffix has to stay aligned, which it does
    because both models are fed the same sampled token each step. This is the
    same split the trainer's sr_logprob_prompt uses.

    Returns (responses [B, T] padded with pad_id, lengths [B], fb_rows [B],
    logp_rows [B], z_rows [B], logq [B, T], n_narrow). fb_rows counts teacher-fallback
    tokens per row and stays 0 outside ``fuse='agree'``; logp_rows sums log p_student
    over the emitted tokens and z_rows sums the surviving student mass, so dividing
    either by lengths gives a per-row mean; logq holds the PER-TOKEN log density each
    token was drawn from (0 past a row's end), which is what the trainer needs as the
    importance ratio's denominator; n_narrow counts row-steps where top-p's nucleus ran
    past sample_next's candidate cap.

    Rows are dropped from the batch as they finish, so the returned tensors are indexed
    by the row's ORIGINAL position, not by its position in the shrinking batch.
    """
    device = next(model_a.parameters()).device
    ids_a = batch_a["input_ids"].to(device)
    mask_a = batch_a["attention_mask"].to(device)
    ids_b = batch_b["input_ids"].to(device)
    mask_b = batch_b["attention_mask"].to(device)
    n_rows = ids_a.size(0)

    # Left-padded prompts: position_ids must start from 0 at the first REAL token,
    # so build them from the mask rather than arange. A wrong offset here shifts
    # RoPE and corrupts generation silently.
    pos_a = (mask_a.cumsum(dim=-1) - 1).clamp(min=0)
    pos_b = (mask_b.cumsum(dim=-1) - 1).clamp(min=0)

    # Probed once per batch, not per step: inspect.signature is not free and the
    # answer cannot change mid-decode.
    keep_a = last_logit_kwargs(model_a)
    keep_b = last_logit_kwargs(model_b)

    out_a = model_a(input_ids=ids_a, attention_mask=mask_a, position_ids=pos_a,
                    use_cache=True, **keep_a)
    out_b = model_b(input_ids=ids_b, attention_mask=mask_b, position_ids=pos_b,
                    use_cache=True, **keep_b)
    kv_a, kv_b = out_a.past_key_values, out_b.past_key_values
    # [:, -1, :] regardless: with the flag the tensor is already [B, 1, V], without
    # it this is the slice that discards the unused positions.
    last_a, last_b = out_a.logits[:, -1, :], out_b.logits[:, -1, :]
    next_pos_a = pos_a[:, -1:] + 1
    next_pos_b = pos_b[:, -1:] + 1

    lengths = torch.zeros(n_rows, dtype=torch.long, device=device)
    eos_tensor = torch.tensor(eos_ids, device=device)
    # Per-ROW rather than per-batch, so the caller can put these next to each row in
    # the parquet and the summary can average over questions the same way it
    # averages scores.
    #   fb_rows      -- tokens that came from the teacher fallback (agree only)
    #   logp_rows    -- sum of log p_student for the tokens actually emitted
    # Both count live rows only: a finished row's tokens are trimmed off later, so
    # including them would dilute the averages with padding.
    fb_rows = torch.zeros(n_rows, dtype=torch.long, device=device)
    logp_rows = torch.zeros(n_rows, dtype=torch.float64, device=device)
    # Sum of Z_t over emitted tokens, so dividing by lengths gives the mean share of
    # student mass the teacher permitted. Fallback steps contribute 0, which is the
    # honest reading -- nothing survived there.
    z_rows = torch.zeros(n_rows, dtype=torch.float64, device=device)
    # Rows where top_p's nucleus ran past the candidate cap, so sampling was narrower
    # than requested. Kept on-device and read once at the end -- checking per step would
    # reintroduce exactly the host sync the EOS check was loosened to avoid.
    narrow = torch.zeros((), dtype=torch.long, device=device)

    # Finished rows are DROPPED from the batch instead of being carried along emitting
    # padding. Length varies enormously between rows -- a batch of 8 typically has ~33%
    # of its row-steps doing real work -- and every carried row still costs its share of
    # the KV read, which by 10k tokens is the largest term in a step. Dropping them
    # shrinks that read as the batch drains.
    #
    # Two things follow. Rows are written into a buffer indexed by ORIGINAL row, since
    # the batch no longer lines up with the caller's rows; and `alive_idx` maps current
    # batch position -> original row so the per-row stats stay attributable.
    out_tokens = torch.full((n_rows, args.max_new_tokens), pad_id,
                            dtype=torch.long, device=device)
    # Per-token densities, laid out like out_tokens so index i of one names index i of
    # the other. This is the alignment the trainer's importance ratio rests on, and it
    # holds by construction: both are written in the same statement from the same
    # sampled token. 0 past a row's end -- an unemitted token has no density, and 0
    # keeps it inert in the masked sums downstream. Cheap in absolute terms: 128 rows x
    # 14336 tokens in fp32 is 7.3MB, three orders below the KV cache.
    out_logq = torch.zeros((n_rows, args.max_new_tokens), dtype=torch.float32, device=device)
    alive_idx = torch.arange(n_rows, device=device)
    alive = torch.ones(n_rows, dtype=torch.bool, device=device)
    # batch_select_indices is a DynamicCache method (transformers uses it for
    # contrastive search); the base Cache class does not define it, so a model handing
    # back some other cache type must keep the full batch rather than crash.
    can_shrink = (args.shrink_batch
                  and hasattr(kv_a, "batch_select_indices")
                  and hasattr(kv_b, "batch_select_indices"))

    steps_run = 0
    for step in range(args.max_new_tokens):
        steps_run = step + 1
        if args.fuse == "agree":
            nxt, fell_back, log_pa, z_t, log_q = agree_select(
                last_a, last_b,
                student_top_k=args.agree_student_top_k,
                teacher_top_k=args.agree_teacher_top_k,
                student_min_prob=args.agree_student_min_prob,
                teacher_min_prob=args.agree_teacher_min_prob,
                temperature=args.temperature,
                top_p=args.top_p,
                fallback=args.agree_fallback,
                narrow_counter=narrow,
            )
            fb_rows.index_add_(0, alive_idx, (fell_back & alive).long())
            z_rows.index_add_(0, alive_idx,
                              torch.where(alive, z_t.double(),
                                          torch.zeros_like(z_t, dtype=torch.float64)))
        else:
            scores = fuse_logits(last_a, last_b, args.fuse, args.fuse_weight)
            nxt = sample_next(scores, args.temperature, args.top_p, args.top_k,
                              narrow_counter=narrow)
            # Same density expression as the agree branch, on the fused scores -- so
            # every fuse mode reports a usable denominator, not just 'agree'.
            log_q = sampled_log_prob(scores, nxt, args.temperature)
            log_pa = None

        # Written before `nxt` is overwritten with padding below, so it is the density
        # of the token actually chosen. Dead rows are zeroed rather than trusted: their
        # `nxt` is about to become pad_id, whose density means nothing.
        out_logq[alive_idx, step] = torch.where(alive, log_q, torch.zeros_like(log_q))

        # A row that finished since the last shrink is still in the batch, so it still
        # has to be masked out here; the shrink only removes it at the next checkpoint.
        nxt = torch.where(alive.unsqueeze(-1), nxt, torch.full_like(nxt, pad_id))
        # How confident the STUDENT was in the token that was actually emitted --
        # read off log p_a, not off the fused/masked scores, so the number means the
        # same thing in every fuse mode and stays comparable to a single-model run.
        # Taken before the row is marked finished, so the EOS token itself counts
        # (matching how `lengths` counts it). agree_select already normalised log p_a,
        # so reuse it rather than paying for a second [B, vocab] log_softmax per step.
        if log_pa is None:
            log_pa = F.log_softmax(last_a.float(), dim=-1)
        step_logp = log_pa.gather(-1, nxt).squeeze(-1)
        logp_rows.index_add_(0, alive_idx,
                             torch.where(alive, step_logp.double(),
                                         torch.zeros_like(step_logp, dtype=torch.float64)))
        out_tokens[alive_idx, step] = nxt.squeeze(-1)
        lengths.index_add_(0, alive_idx, alive.long())
        # nxt is [B, 1] and eos_tensor is [n_eos], so the comparison broadcasts to
        # [B, n_eos] and any(-1) collapses back to [B].
        alive = alive & ~(nxt == eos_tensor).any(dim=-1)

        # One host sync serves both the stop test and the shrink, at a cadence chosen so
        # neither pays for its own. `.any()` copies to the host, which drains the CUDA
        # queue and idles the GPU, so doing it every step costs more than the handful of
        # all-padding steps a coarser check may run before noticing.
        if step % args.eos_check_every == 0:
            n_alive = int(alive.sum().item())
            if n_alive == 0:
                break
            if can_shrink and n_alive < alive.numel():
                keep = alive.nonzero(as_tuple=True)[0]
                kv_a.batch_select_indices(keep)
                kv_b.batch_select_indices(keep)
                mask_a, mask_b = mask_a[keep], mask_b[keep]
                next_pos_a, next_pos_b = next_pos_a[keep], next_pos_b[keep]
                last_a, last_b = last_a[keep], last_b[keep]
                nxt = nxt[keep]
                alive_idx = alive_idx[keep]
                # Every survivor is alive by construction after the drop.
                alive = torch.ones(keep.numel(), dtype=torch.bool, device=device)

        mask_a = torch.cat([mask_a, torch.ones_like(nxt)], dim=-1)
        mask_b = torch.cat([mask_b, torch.ones_like(nxt)], dim=-1)
        # No keep-last kwarg here: these feed a single token, so the logits are
        # already [B, 1, V] and the flag would be a no-op.
        step_a = model_a(input_ids=nxt, attention_mask=mask_a, position_ids=next_pos_a,
                         past_key_values=kv_a, use_cache=True)
        step_b = model_b(input_ids=nxt, attention_mask=mask_b, position_ids=next_pos_b,
                         past_key_values=kv_b, use_cache=True)
        kv_a, kv_b = step_a.past_key_values, step_b.past_key_values
        last_a, last_b = step_a.logits[:, -1, :], step_b.logits[:, -1, :]
        next_pos_a = next_pos_a + 1
        next_pos_b = next_pos_b + 1

    return (out_tokens[:, :steps_run].cpu(), lengths.cpu(), fb_rows.cpu(),
            logp_rows.cpu(), z_rows.cpu(), out_logq[:, :steps_run].cpu(),
            int(narrow.item()))
