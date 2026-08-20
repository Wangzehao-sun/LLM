"""Joint decoding: combine two models' predictions at every generated token.

Two families of rule, chosen with ``--fuse``:

* ``linear`` / ``contrastive`` / ``max`` -- blend the two next-token distributions
  in log space, then sample from the result (contrastive-decoding / proxy-tuning /
  DExperts).
* ``agree`` -- do not blend. The TEACHER (model B) constrains which tokens are
  allowed: a candidate must sit in both models' top-k and clear each side's
  probability floor. The STUDENT (model A) then samples among the survivors, so
  the teacher steers direction while the student picks the token it finds most
  natural. When nothing survives there is no agreement to honour and the step
  samples from the teacher instead. Both branches respect ``--temperature`` /
  ``--top-p``.

This deliberately does NOT go through verl's rollout stack. vLLM is entered once
per request (``vllm_rollout_spmd.py:304`` calls ``LLM.generate()``), so there is
no per-token hook to fuse into, and the pinned ``vllm<=0.8.5`` (setup.py:51) runs
the V1 engine, which dropped per-request ``logits_processors``. The one token-level
loop that does exist, ``naive_rollout.py:68-100``, re-forwards the whole prefix
every step (no KV cache, O(n^2)) -- unusable at these response lengths. So this is
a plain HF decode loop WITH a KV cache, run outside Ray.

The cost of leaving vLLM behind is throughput: no paged attention, no CUDA graph,
and two forwards per token. Data parallelism claws back some of it -- launch under
torchrun and each rank decodes its own slice of the rows onto its own GPU. Ranks
never talk to each other, so there is no process group to initialise; each writes
its own shard, matching what ``main_generation.py:233`` does per batch. Every
downstream reader (``aggregate_sr_responses.py:119``,
``sweep_sft_checkpoints.sh``) globs ``*.parquet``, so no merge step is needed.

Because it is this much slower than vLLM, it is for OFFLINE evaluation only --
putting it in the GRPO rollout path would need the V0/logits_processors route
instead.

Usage:

    # 4-way data parallel
    torchrun --standalone --nproc_per_node=4 Data/joint_decode.py \\
        --model-a /home/data/shared/Qwen3-4b-base \\
        --model-b /home/data/shared/rephraser-ckpt \\
        --input   Data/eval_rephrase_flat.parquet \\
        --output-dir /tmp/joint_out \\
        --fuse linear --fuse-weight 0.5

    # single GPU, no torchrun needed (RANK defaults to 0, WORLD_SIZE to 1)
    python Data/joint_decode.py --model-a ... --model-b ... \\
        --input ... --output-dir ... --fuse-weight 0

    # top-k agreement: teacher B constrains, student A picks
    torchrun --standalone --nproc_per_node=4 Data/joint_decode.py \\
        --model-a ... --model-b ... --input ... --output-dir ... \\
        --fuse agree --agree-top-k 10 --agree-teacher-min-prob 0.05

``--fuse-weight 0`` reduces exactly to model A alone; with ``--temperature 0`` it
must reproduce A's greedy output token for token, which is the sanity check that
the KV cache and masks are wired correctly.

Prompt rendering follows ``main_generation.py:128-171``, including the prefill case
where the last message is an assistant turn the model must RESUME rather than answer
afresh. On such a set ``--teacher-drop-prefill`` hides that partial answer from model
B, so the teacher guides from the question alone while the student still continues
its own text.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    raise ValueError(f"unknown --fuse {mode!r}")


def agree_select(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    top_k: int,
    student_min_prob: float,
    teacher_min_prob: float,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Let the teacher constrain the direction and the student pick within it.

    Model B is the teacher (it steers), model A is the student (it chooses). The
    asymmetry is the point, so the two roles use the mask and the scores
    differently:

      * CONSTRAINT (teacher-led) -- a candidate must be in BOTH models' top-k, and
        clear ``teacher_min_prob`` under B and ``student_min_prob`` under A. The
        teacher's floor sets how wide the allowed region is; the student's floor
        only drops tokens it would itself be very reluctant to emit.
      * CHOICE (student-led) -- among the survivors, sample by the STUDENT's
        distribution. Ranking by the teacher instead would collapse this to the
        teacher's own argmax almost every step, leaving the student a veto and no
        say -- which is not the intended division of labour.
      * FALLBACK -- when nothing survives there is no agreement to honour, so the
        step defers to the model steering direction: sample from the TEACHER.

    Sampling, not argmax, in both branches: an argmax fallback would pin every
    disagreeing position to one fixed token, so a sequence's diversity would decay
    with each fallback and ``--n-samples > 1`` would stop meaning anything.
    ``temperature == 0`` still gives greedy behaviour on both paths.

    Returns (token ids [B, 1], fell_back [B] bool, log_pa [B, vocab]) so the caller can
    report how often the agreement actually bound anything, and reuse the student's
    log-probs. A fallback rate near 1.0 means the run is effectively plain teacher
    decoding and the intersection is doing no work.
    """
    log_pa = F.log_softmax(logits_a.float(), dim=-1)
    log_pb = F.log_softmax(logits_b.float(), dim=-1)
    k = min(top_k, log_pa.size(-1))

    # Membership masks over the FULL vocab, so the intersection is a plain AND
    # rather than a set-of-ids comparison (which would need a loop per row).
    in_a = torch.zeros_like(log_pa, dtype=torch.bool)
    in_b = torch.zeros_like(log_pb, dtype=torch.bool)
    in_a.scatter_(-1, log_pa.topk(k, dim=-1).indices, True)
    in_b.scatter_(-1, log_pb.topk(k, dim=-1).indices, True)

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

    # Student chooses inside the allowed set; teacher decides the fallback rows.
    # Renormalising is not needed -- sample_next softmaxes, and the -inf entries
    # drop out of it -- but the masked rows must not be all -inf, which is why
    # fallback rows are routed to the teacher's unmasked distribution instead.
    student_scores = log_pa.masked_fill(~eligible, float("-inf"))
    scores = torch.where(fell_back.unsqueeze(-1), log_pb, student_scores)

    # top_k is already enforced by the intersection above, so only top_p is left to
    # apply here; passing top_k again would re-cut the (already tiny) survivor set.
    chosen = sample_next(scores, temperature, top_p, -1)
    # log_pa goes back to the caller so the per-token confidence stat can reuse it
    # instead of recomputing a [B, vocab] log_softmax every step.
    return chosen, fell_back, log_pa


def sample_next(scores: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """Pick the next token id per row. Mirrors naive_rollout.py:77-88, plus top-p.

    ``temperature == 0`` means greedy, which is what makes the output a
    deterministic fingerprint of the weights (the trick smoke_dual_vllm.py:21-24
    relies on).
    """
    if temperature == 0.0:
        return scores.argmax(dim=-1, keepdim=True)

    scores = scores / temperature
    if top_k > 0:
        kth = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        ordered, order = torch.sort(scores, descending=True, dim=-1)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        # Shift by one so the token that crosses the threshold is kept: otherwise
        # top_p smaller than the top token's probability would mask everything.
        drop = cumulative - ordered.softmax(dim=-1) > top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        scores = torch.empty_like(scores).scatter_(-1, order, ordered)
    return torch.multinomial(scores.softmax(dim=-1), num_samples=1)


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def last_logit_kwargs(model) -> dict:
    """Ask the model for the LAST position's logits only, if it supports it.

    Prefill computes logits for every prompt position, and only the last one is
    ever read. At a 152k vocab that waste dominates the memory profile:
    [8, 4096, 151936] in bf16 is ~10GB per model, ~20GB for the pair, versus 2.4MB
    for the single position actually used -- and it scales with --batch-size, so 16
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
    logp_rows [B]). fb_rows counts teacher-fallback tokens per row and stays 0
    outside ``--fuse agree``; logp_rows sums log p_student over the emitted tokens,
    so dividing by lengths gives a per-row mean.
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

    collected = []
    lengths = torch.zeros(n_rows, dtype=torch.long, device=device)
    unfinished = torch.ones(n_rows, dtype=torch.bool, device=device)
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

    for step in range(args.max_new_tokens):
        if args.fuse == "agree":
            nxt, fell_back, log_pa = agree_select(
                last_a, last_b,
                top_k=args.agree_top_k,
                student_min_prob=args.agree_student_min_prob,
                teacher_min_prob=args.agree_teacher_min_prob,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            fb_rows += (fell_back & unfinished).long()
        else:
            scores = fuse_logits(last_a, last_b, args.fuse, args.fuse_weight)
            nxt = sample_next(scores, args.temperature, args.top_p, args.top_k)
            log_pa = None

        # Finished rows emit pad and stop counting, but keep stepping so the batch
        # stays rectangular -- their tokens are trimmed off at the end.
        nxt = torch.where(unfinished.unsqueeze(-1), nxt, torch.full_like(nxt, pad_id))
        # How confident the STUDENT was in the token that was actually emitted --
        # read off log p_a, not off the fused/masked scores, so the number means the
        # same thing in every --fuse mode and stays comparable to a single-model run.
        # Taken before the row is marked finished, so the EOS token itself counts
        # (matching how `lengths` counts it). agree_select already normalised log p_a,
        # so reuse it rather than paying for a second [B, vocab] log_softmax per step.
        if log_pa is None:
            log_pa = F.log_softmax(last_a.float(), dim=-1)
        step_logp = log_pa.gather(-1, nxt).squeeze(-1)
        logp_rows += torch.where(unfinished, step_logp.double(), torch.zeros_like(logp_rows))
        collected.append(nxt)
        lengths += unfinished.long()
        # nxt is [B, 1] and eos_tensor is [n_eos], so the comparison broadcasts to
        # [B, n_eos] and any(-1) collapses back to [B].
        unfinished = unfinished & ~(nxt == eos_tensor).any(dim=-1)
        # `.any()` copies to the host, which drains the CUDA queue and leaves the GPU
        # idle while the CPU catches up. Checking every step therefore costs more than
        # the handful of all-padding steps a coarser check may run before noticing.
        if step % args.eos_check_every == 0 and not unfinished.any():
            break

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

    responses = torch.cat(collected, dim=-1) if collected else torch.zeros((n_rows, 0), dtype=torch.long)
    return responses.cpu(), lengths.cpu(), fb_rows.cpu(), logp_rows.cpu()


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def ends_with_assistant(chat) -> bool:
    """True when the last message is an assistant turn, i.e. this row is a prefill."""
    return bool(len(chat)) and chat[-1].get("role") == "assistant"


def render(tokenizer, chats, max_length):
    """Tokenise a batch of chat prompts, matching main_generation.py:128-171.

    A prompt whose last message is an ASSISTANT turn is a prefill: the model must
    RESUME writing that message, not start a new one. The two template flags that
    control this are mutually exclusive in transformers (passing both raises):

      add_generation_prompt=True   -> ...assistant\\nPREFIX<|im_end|>\\n<|im_start|>assistant\\n
                                      the prefix is closed as a finished turn and the
                                      model answers again from scratch.
      continue_final_message=True  -> ...assistant\\nPREFIX
                                      no end-of-turn token, so generation resumes
                                      inside the prefix. This is what prefill means.

    Built by Data/prepare_prefill_continue.py. Getting it wrong does not raise -- the
    model just silently re-answers instead of continuing, which quietly voids the
    experiment.

    The flag is per-CALL but the decision is per-ROW, so a batch mixing prefilled and
    plain prompts cannot use one call. Each row is rendered to text with its own flag
    and the batch is tokenised afterwards -- same padding/truncation, same result for
    the all-plain case.
    """
    n_prefill = sum(ends_with_assistant(chat) for chat in chats)
    if n_prefill == 0:
        return tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
    rendered = [
        tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            **({"continue_final_message": True} if ends_with_assistant(chat)
               else {"add_generation_prompt": True}),
        )
        for chat in chats
    ]
    # add_special_tokens=False: the rendered text already carries every control token
    # the template needs, so letting the tokenizer add BOS again would duplicate it.
    return tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def strip_trailing_assistant(chats):
    """Drop a trailing assistant turn, so this side sees the question without the prefix.

    For prefill eval sets the prefix is a student attempt spliced in as the last
    message. Handing it to the teacher would let it read what the student already
    wrote -- fine for the student, which is supposed to continue its own text, but it
    makes the teacher's guidance conditional on the very thing being judged.

    Dropping the turn here rather than reading a different column keeps this working
    for any prefill parquet, including ones with no bare-question column, and keeps
    both sides on the same `--prompt-key`.
    """
    return [chat[:-1] if ends_with_assistant(chat) else chat for chat in chats]


def as_chat(cell):
    """Normalise one prompt cell to a list of {role, content} dicts."""
    items = [dict(m) for m in list(cell)]
    if items and "role" in items[0]:
        return items
    raise ValueError("prompt cell is not a flat [{role, content}] list; build the eval "
                     "parquet with Data/prepare_rephrase_eval.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", required=True, help="Model that is being steered (the 'base').")
    p.add_argument("--model-b", required=True, help="Model whose distribution is mixed in.")
    p.add_argument("--input", type=Path, required=True,
                   help="Eval parquet with a flat prompt column (Data/prepare_rephrase_eval.py).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the per-rank shards; read back with glob('*.parquet').")
    p.add_argument("--prompt-key", default="prompt", help="Prompt column for model A (default: %(default)s)")
    p.add_argument("--prompt-key-b", default=None,
                   help="Prompt column for model B. Defaults to --prompt-key, i.e. both models see the "
                        "same prompt and the fusion is purely a model difference.")
    p.add_argument("--teacher-drop-prefill", action="store_true",
                   help="Drop a trailing assistant turn from MODEL B's prompt only. On a "
                        "prefill eval set (Data/prepare_prefill_continue.py) that turn is the "
                        "student's own partial answer: the student should resume it, but giving "
                        "it to the teacher makes the teacher's guidance conditional on the text "
                        "being judged. No effect on prompts that do not end in an assistant turn.")
    p.add_argument("--fuse", default="linear", choices=("linear", "contrastive", "max", "agree"),
                   help="How to combine the two models. linear/contrastive/max blend the "
                        "distributions and then sample; 'agree' lets the teacher (model B) "
                        "constrain which tokens are allowed and the student (model A) sample "
                        "among them, deferring to the teacher when they do not overlap "
                        "(default: %(default)s)")
    p.add_argument("--fuse-weight", type=float, default=0.5,
                   help="Mixing weight on model B. 0 == model A alone. Unused by --fuse agree "
                        "(default: %(default)s)")
    p.add_argument("--agree-top-k", type=int, default=10,
                   help="--fuse agree: size of each model's candidate set before intersecting "
                        "(default: %(default)s)")
    p.add_argument("--agree-teacher-min-prob", type=float, default=0.05,
                   help="--fuse agree: a candidate needs p >= this under the TEACHER (model B). "
                        "This is the knob that sets how wide the allowed region is. 0 disables "
                        "it (default: %(default)s)")
    p.add_argument("--agree-student-min-prob", type=float, default=0.0,
                   help="--fuse agree: a candidate needs p >= this under the STUDENT (model A). "
                        "Defaults to 0 -- the student already ranks the survivors, so this only "
                        "needs raising to exclude tokens it is very reluctant to emit "
                        "(default: %(default)s)")
    p.add_argument("--n-samples", type=int, default=1, help="Samples per question (default: %(default)s)")
    p.add_argument("--temperature", type=float, default=0.6, help="0 == greedy (default: %(default)s)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=-1, help="<=0 disables top-k (default: %(default)s)")
    p.add_argument("--prompt-length", type=int, default=4096, help="Prompt truncation cap (default: %(default)s)")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Rows per decode batch. Much smaller than vLLM's because there is no "
                        "request-level scheduling here: one long row holds up its whole batch "
                        "(default: %(default)s)")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all).")
    p.add_argument("--reward-impl-version", type=int, default=4,
                   help="Passed to math_select_rm_score_fn; 4 = no-think math-verify, what the "
                        "summarize/grpo scripts use (default: %(default)s)")
    p.add_argument("--no-score", action="store_true",
                   help="Skip scoring (no test_score column). Use when the reward fn is unavailable.")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=("flash_attention_2", "sdpa", "eager", "auto"),
                   help="Attention kernel. Defaults to flash_attention_2 and FAILS if it is "
                        "unavailable rather than downgrading quietly -- a slower run that looks "
                        "identical is the outcome worth avoiding. 'auto' probes "
                        "flash_attention_2 -> sdpa -> eager instead. eager rebuilds the full "
                        "score matrix every step and is much slower; pin it only to compare "
                        "(default: %(default)s)")
    p.add_argument("--eos-check-every", type=int, default=16,
                   help="Steps between all-rows-finished checks. Each check syncs GPU->CPU, "
                        "which stalls the pipeline, so testing every step costs more than the "
                        "few extra steps a coarser check may run. Raise to sync less "
                        "(default: %(default)s)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.temperature == 0.0 and args.n_samples > 1:
        print(f"[warn] --temperature 0 is deterministic, so --n-samples {args.n_samples} "
              f"just repeats identical work", file=sys.stderr)
    if args.fuse == "agree":
        # --top-k is the only sampling flag agree really ignores: the top-k
        # intersection has already narrowed the candidates, so re-cutting the
        # survivor set would be both redundant and confusing. temperature and top_p
        # DO apply, on the student branch and the teacher fallback alike.
        if args.top_k > 0:
            print(f"[warn] --fuse agree ignores --top-k {args.top_k}; the candidate sets are "
                  f"already cut by --agree-top-k {args.agree_top_k}", file=sys.stderr)
        if args.fuse_weight != 0.5:
            print(f"[warn] --fuse agree ignores --fuse-weight {args.fuse_weight}: it constrains "
                  f"and selects rather than blending", file=sys.stderr)
        for name, value in (("--agree-teacher-min-prob", args.agree_teacher_min_prob),
                            ("--agree-student-min-prob", args.agree_student_min_prob)):
            if not 0.0 <= value <= 1.0:
                print(f"[fatal] {name} must be in [0, 1], got {value}", file=sys.stderr)
                return 2
        if args.agree_top_k < 1:
            print(f"[fatal] --agree-top-k must be >= 1, got {args.agree_top_k}", file=sys.stderr)
            return 2

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
    # Fusion is elementwise over the vocab axis, so a mismatched vocab silently
    # adds up unrelated tokens' scores. The trainer only warns about this
    # (new_ray_trainer.py:664-676); here it is fatal.
    if tok_a.get_vocab() != tok_b.get_vocab():
        print("[fatal] the two models have different vocabularies -- logits cannot be fused "
              "elementwise. They must come from the same tokenizer family.", file=sys.stderr)
        return 2
    for tok in (tok_a, tok_b):
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.iloc[: args.limit]
    prompt_key_b = args.prompt_key_b or args.prompt_key
    for key in {args.prompt_key, prompt_key_b}:
        if key not in df.columns:
            print(f"[fatal] column {key!r} not in {args.input} (have: {list(df.columns)})", file=sys.stderr)
            return 2

    # Strided, not contiguous: eval sets are often ordered by difficulty or source,
    # so a contiguous split would hand one rank all the long rows.
    shard = df.iloc[rank::world_size].copy()
    if rank == 0:
        print(f"[data] {len(df):,} rows -> {world_size} rank(s); this rank: {len(shard):,}", flush=True)
    if shard.empty:
        print(f"[rank {rank}] no rows in this shard, nothing to do")
        return 0

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # flash-attn is a CUDA kernel and requires fp16/bf16, so it cannot serve a CPU
    # debug run. Downgrade rather than fail there: on CPU the point is to exercise the
    # logic, not the throughput, so the substitution costs nothing that matters.
    attn_impl = args.attn_impl
    if device == "cpu" and attn_impl == "flash_attention_2":
        print("[warn] flash_attention_2 needs CUDA and fp16/bf16; using eager on CPU",
              file=sys.stderr, flush=True)
        attn_impl = "eager"

    # The attention kernel is the single biggest throughput lever here. "eager"
    # materialises a [B, heads, 1, kv_len] score matrix per layer per step, which at 36
    # layers across two models with an 8k context is ~0.6GB of extra traffic per step
    # against ~8GB for the weights -- the same order of magnitude, so it can cost close
    # to a factor of two. flash_attention_2 and sdpa use a fused kernel instead.
    #
    # Default to flash_attention_2 EXPLICITLY rather than probing, because a silent
    # downgrade is the bad outcome: the run still works and still looks right, just
    # slower, with nothing in the output saying why. Ask for --attn-impl auto to get the
    # probing behaviour, or pin sdpa/eager to compare.
    def load(path: str, tag: str):
        wanted = ([attn_impl] if attn_impl != "auto"
                  else ["flash_attention_2", "sdpa", "eager"])
        for impl in wanted:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=dtype, trust_remote_code=True,
                    attn_implementation=impl,
                )
                if rank == 0:
                    print(f"[rank {rank}] {tag} attn_implementation={impl}", flush=True)
                # Plain .to(device), NOT device_map="auto": each rank owns exactly one
                # GPU here, and letting accelerate shard a model across all of them
                # would collide with the other ranks' models.
                return model.to(device).eval()
            except (ValueError, ImportError, RuntimeError) as error:
                if impl == wanted[-1]:
                    if impl == "flash_attention_2":
                        print(
                            f"[fatal] {tag}: flash_attention_2 could not be used "
                            f"({type(error).__name__}: {error}).\n"
                            f"  install it with `pip install flash-attn --no-build-isolation` "
                            f"(it is already in Myverl/setup.py's GPU extra), or run with "
                            f"ATTN_IMPL=sdpa to use the fused PyTorch kernel instead.\n"
                            f"  Not falling back silently: sdpa is slower, and a run that "
                            f"quietly used it would look identical to one that did not.",
                            file=sys.stderr, flush=True,
                        )
                    raise
                if rank == 0:
                    print(f"[rank {rank}] {tag} attn_implementation={impl} unavailable "
                          f"({type(error).__name__}), trying next", flush=True)
        raise AssertionError("unreachable")

    print(f"[rank {rank}] loading A={args.model_a}", flush=True)
    model_a = load(args.model_a, "A")
    print(f"[rank {rank}] loading B={args.model_b}", flush=True)
    model_b = load(args.model_b, "B")

    # Say which way it went. Silently falling back is the dangerous case: the run
    # still produces correct output, just with ~20GB of prefill logits it never
    # reads, which shows up much later as an OOM the moment --batch-size is raised.
    if rank == 0:
        keep = last_logit_kwargs(model_a)
        if keep:
            print(f"[mem] prefill keeps last logit only ({next(iter(keep))}=1)", flush=True)
        else:
            v = getattr(model_a.config, "vocab_size", 0)
            waste = args.batch_size * args.prompt_length * v * 2 * 2 / 1e9
            print(f"[mem] this transformers has no logits_to_keep/num_logits_to_keep, so "
                  f"prefill materialises logits for every position: about {waste:.1f} GB "
                  f"across both models at --batch-size {args.batch_size}. Output is "
                  f"unaffected; lower --batch-size if it OOMs.", file=sys.stderr, flush=True)

    # Stop on real EOS only. pad_token_id is excluded on purpose: many chat models
    # set pad == eos, but for those that do not, treating pad as a stop signal would
    # end generation the first time the model emits a pad-ish token.
    if tok_a.eos_token_id is None:
        print("[fatal] model A's tokenizer has no eos_token_id, so generation would never "
              "stop early. Set one or lower --max-new-tokens deliberately.", file=sys.stderr)
        return 2
    eos_ids = [tok_a.eos_token_id]
    pad_id = tok_a.pad_token_id

    chats_a = [as_chat(c) for c in shard[args.prompt_key]]
    chats_b = [as_chat(c) for c in shard[prompt_key_b]]

    n_prefill = sum(ends_with_assistant(c) for c in chats_a)
    if args.teacher_drop_prefill:
        chats_b = strip_trailing_assistant(chats_b)
    if rank == 0 and n_prefill:
        # Worth stating either way: whether the teacher can see the student's partial
        # answer changes what the run measures, and neither choice raises.
        seen = "does NOT see" if args.teacher_drop_prefill else "SEES"
        print(f"[prefill] {n_prefill}/{len(chats_a)} rows end with an assistant turn; "
              f"student resumes it, teacher {seen} it", flush=True)

    # [n_rows][n_samples] -- transposed to match main_generation.py's per-row lists.
    texts: list[list[str]] = [[] for _ in range(len(shard))]
    tok_lens: list[list[int]] = [[] for _ in range(len(shard))]
    fb_frac: list[list[float]] = [[] for _ in range(len(shard))]
    mean_logp: list[list[float]] = [[] for _ in range(len(shard))]

    total_tokens = 0
    fallback_tokens = 0
    decided_tokens = 0
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start:
        start.record()

    # One bar, on rank 0 only: four ranks each drawing their own would interleave
    # into unreadable output, and `tee`ing that to a log file leaves it full of
    # cursor-movement escapes. The shards differ in length by at most one row, so
    # rank 0's progress tracks the whole job closely enough. Other ranks print a
    # line per batch instead (below), which is coarse but keeps their logs plain.
    n_batches = -(-len(shard) // args.batch_size) * args.n_samples
    bar = None
    if rank == 0:
        from tqdm import tqdm

        bar = tqdm(total=n_batches, desc=f"decode ({args.fuse})", unit="batch")

    for sample_idx in range(args.n_samples):
        for begin in range(0, len(shard), args.batch_size):
            stop = min(begin + args.batch_size, len(shard))
            batch_a = render(tok_a, chats_a[begin:stop], args.prompt_length)
            batch_b = render(tok_b, chats_b[begin:stop], args.prompt_length)
            responses, lengths, fb_row, logp_row = joint_generate(
                model_a, model_b, batch_a, batch_b, eos_ids=eos_ids, pad_id=pad_id, args=args,
            )
            for row in range(responses.size(0)):
                n_tok = int(lengths[row].item())
                ids = responses[row, :n_tok]
                texts[begin + row].append(tok_a.decode(ids, skip_special_tokens=True))
                tok_lens[begin + row].append(n_tok)
                # Guard the empty-response case: a row that emitted nothing has no
                # tokens to average over, so report NaN rather than 0/0. NaN also
                # keeps it out of the summary means instead of dragging them down.
                n_fb = int(fb_row[row].item())
                fb_frac[begin + row].append(n_fb / n_tok if n_tok else float("nan"))
                mean_logp[begin + row].append(
                    float(logp_row[row].item()) / n_tok if n_tok else float("nan")
                )
                fallback_tokens += n_fb
                decided_tokens += n_tok
                total_tokens += n_tok
            if bar is not None:
                # tok/s is the number that decides whether this path is fast enough
                # to be worth keeping, so surface it live rather than only in the
                # final summary. tqdm's own rate is batches/s, which hides the fact
                # that batches vary hugely in generated length.
                bar.update(1)
                bar.set_postfix(sample=f"{sample_idx + 1}/{args.n_samples}",
                                tok_s=f"{total_tokens / max(bar.format_dict['elapsed'], 1e-6):.0f}")
            else:
                print(f"[rank {rank}] sample {sample_idx + 1}/{args.n_samples} "
                      f"rows {begin}-{stop - 1} done", flush=True)

    if bar is not None:
        bar.close()
    if start:
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
        print(f"[rank {rank}] {total_tokens:,} tokens in {seconds:.1f}s "
              f"= {total_tokens / max(seconds, 1e-6):.1f} tok/s", flush=True)

    if decided_tokens:
        # Mean p_student over the tokens actually emitted -- exp of a per-token mean
        # log-prob, so it is a geometric mean and does not depend on length. Reported
        # for every mode: it says how natural the emitted text was to the student,
        # which is exactly what the constraint trades away.
        # NaN marks a response that emitted nothing, so it has no mean to contribute.
        finite = [v for row in mean_logp for v in row if not math.isnan(v)]
        if finite:
            mean = sum(finite) / len(finite)
            print(f"[rank {rank}] student mean logp={mean:.4f} "
                  f"(geometric-mean prob {math.exp(mean):.4f})", flush=True)

    if args.fuse == "agree" and decided_tokens:
        # A rate near 1.0 means the two models almost never overlapped under these
        # settings, so the student never got to choose and the run is really just
        # teacher sampling -- loosen the floors or raise --agree-top-k before
        # reading anything into the scores.
        rate = fallback_tokens / decided_tokens
        print(f"[rank {rank}] teacher fallback on {fallback_tokens:,}/{decided_tokens:,} "
              f"tokens = {rate:.1%} (top_k={args.agree_top_k}, "
              f"teacher_min_prob={args.agree_teacher_min_prob}, "
              f"student_min_prob={args.agree_student_min_prob})", flush=True)

    shard["responses"] = texts
    shard["fallback_frac"] = fb_frac
    shard["student_mean_logp"] = mean_logp
    shard["response_lengths"] = tok_lens

    if not args.no_score:
        # Same entry point main_generation.py:216 uses, so scores are comparable
        # with everything else in the repo rather than a second opinion.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Myverl"))
        from verl.custom.math_verify_reward import math_select_rm_score_fn

        scores = []
        for i in range(len(shard)):
            row = shard.iloc[i]
            score_fn = math_select_rm_score_fn(row["data_source"], reward_impl_version=args.reward_impl_version)
            ground_truth = row["reward_model"]["ground_truth"]
            per_response = [score_fn(solution_str=r, ground_truth=ground_truth) for r in row["responses"]]
            scores.append({
                "scores_per_response": per_response,
                "mean_score": float(np.mean(per_response)),
                "max_score": bool(np.max(per_response)),
            })
        shard["test_score"] = scores
        print(f"[rank {rank}] mean_score={np.mean([s['mean_score'] for s in scores]):.4f} "
              f"pass@{args.n_samples}={np.mean([s['max_score'] for s in scores]):.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"rank{rank}.parquet"
    shard.to_parquet(out_path, index=False)
    print(f"[rank {rank}] {len(shard):,} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
