"""Joint decoding: combine two models' predictions at every generated token.

This is the OFFLINE EVALUATION driver -- CLI, sharding, parquet I/O, prompt rendering,
scoring. The decoding itself lives in ``verl.custom.joint_decode_core``, which this
imports; see that module for what the fusion rules actually do and why the loop is a
hand-written HF one rather than vLLM.

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

Data parallelism: launch under torchrun and each rank decodes its own slice of the rows
onto its own GPU. Ranks never talk to each other, so there is no process group to
initialise; each writes its own shard, matching what ``main_generation.py:233`` does per
batch. Every downstream reader (``aggregate_sr_responses.py:119``,
``sweep_sft_checkpoints.sh``) globs ``*.parquet``, so no merge step is needed.

The same decode core also runs INSIDE training, on the actor's own worker, to produce the
summarize-replacement candidate (``NewActorRolloutRefWorker.generate_joint`` ->
``verl/custom/joint_sr.py``). There the student is the model being TRAINED rather than a
frozen checkpoint. That path is throughput-bound for the reasons the core module explains, so
it is confined to a small number of questions per step. This script stays the way to sweep
settings cheaply before committing one to a training run.

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


# ---------------------------------------------------------------------------
# The decode core -- fusion, agreement, sampling, and the KV-cache loop -- lives in the
# verl package rather than here, because it now has a SECOND caller:
# NewActorRolloutRefWorker.generate_joint runs the same loop on the actor's own worker to
# produce the summarize-replacement candidate during training. One copy is what makes the
# offline numbers and the training behaviour the same thing, instead of two
# implementations that agree until they drift.
#
# The direction of the import is fixed by CLAUDE.md: reusable package code belongs under
# Myverl/verl/, and Data/ scripts may import from verl but never the reverse. main()
# already extends sys.path this way to reach the reward function; hoisting it here keeps
# one rule for the whole file.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Myverl"))

from verl.custom.joint_decode_core import joint_generate, last_logit_kwargs  # noqa: E402


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
                   help="--fuse agree: default candidate-set size for BOTH models before "
                        "intersecting. Override either side with --agree-student-top-k / "
                        "--agree-teacher-top-k (default: %(default)s)")
    p.add_argument("--agree-student-top-k", type=int, default=None,
                   help="--fuse agree: how far down the STUDENT's own ranking it will look for "
                        "a permitted token. Raising this alone lets it reach further for an "
                        "agreed token instead of falling back, without widening what the "
                        "teacher allows. Defaults to --agree-top-k.")
    p.add_argument("--agree-teacher-top-k", type=int, default=None,
                   help="--fuse agree: how much of the vocabulary the TEACHER permits at all. "
                        "This is the width of the allowed region. Defaults to --agree-top-k.")
    p.add_argument("--agree-teacher-min-prob", type=float, default=0.05,
                   help="--fuse agree: a candidate needs p >= this under the TEACHER (model B). "
                        "This is the knob that sets how wide the allowed region is. 0 disables "
                        "it (default: %(default)s)")
    p.add_argument("--agree-student-min-prob", type=float, default=0.0,
                   help="--fuse agree: a candidate needs p >= this under the STUDENT (model A). "
                        "Defaults to 0 -- the student already ranks the survivors, so this only "
                        "needs raising to exclude tokens it is very reluctant to emit "
                        "(default: %(default)s)")
    p.add_argument("--agree-fallback", default="teacher", choices=("teacher", "student"),
                   help="--fuse agree: who decides a step where no candidate survives the "
                        "constraint. 'teacher' lets it override the student exactly where they "
                        "disagree; 'student' lets the student carry on, so the teacher only ever "
                        "narrows its choices and never overrides. This also flips what a high "
                        "fallback rate means -- teacher decoding in the first case, student "
                        "decoding in the second (default: %(default)s)")
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
                        "few extra steps a coarser check may run. Also the cadence at which "
                        "finished rows are dropped (default: %(default)s)")
    p.add_argument("--no-shrink-batch", dest="shrink_batch", action="store_false",
                   help="Keep finished rows in the batch instead of dropping them. They emit "
                        "padding that is thrown away, but still pay their share of the KV read "
                        "-- the largest cost per step at long contexts. Only useful to isolate "
                        "the shrink when comparing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.temperature == 0.0 and args.n_samples > 1:
        print(f"[warn] --temperature 0 is deterministic, so --n-samples {args.n_samples} "
              f"just repeats identical work", file=sys.stderr)
    if args.fuse == "agree":
        # Resolve the per-side k's here rather than at each use, so the rest of the run
        # (including the log lines) sees the values actually in force.
        if args.agree_student_top_k is None:
            args.agree_student_top_k = args.agree_top_k
        if args.agree_teacher_top_k is None:
            args.agree_teacher_top_k = args.agree_top_k
        # --top-k is the only sampling flag agree really ignores: the top-k
        # intersection has already narrowed the candidates, so re-cutting the
        # survivor set would be both redundant and confusing. temperature and top_p
        # DO apply, on the student branch and the teacher fallback alike.
        if args.top_k > 0:
            print(f"[warn] --fuse agree ignores --top-k {args.top_k}; the candidate sets are "
                  f"already cut by --agree-student-top-k {args.agree_student_top_k} / "
                  f"--agree-teacher-top-k {args.agree_teacher_top_k}", file=sys.stderr)
        if args.fuse_weight != 0.5:
            print(f"[warn] --fuse agree ignores --fuse-weight {args.fuse_weight}: it constrains "
                  f"and selects rather than blending", file=sys.stderr)
        for name, value in (("--agree-teacher-min-prob", args.agree_teacher_min_prob),
                            ("--agree-student-min-prob", args.agree_student_min_prob)):
            if not 0.0 <= value <= 1.0:
                print(f"[fatal] {name} must be in [0, 1], got {value}", file=sys.stderr)
                return 2
        for name, value in (("--agree-student-top-k", args.agree_student_top_k),
                            ("--agree-teacher-top-k", args.agree_teacher_top_k)):
            if value < 1:
                print(f"[fatal] {name} must be >= 1, got {value}", file=sys.stderr)
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
    mean_z: list[list[float]] = [[] for _ in range(len(shard))]

    total_tokens = 0
    fallback_tokens = 0
    decided_tokens = 0
    narrow_steps = 0
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
            # log_z is the per-token log Z_t. Offline evaluation has no use for it -- it
            # exists for the training path, where it weights the off-policy loss
            # (verl/custom/joint_sr.py). Discarded here rather than written out: a
            # [rows, 14336] float column would dominate the parquet, and nothing downstream
            # of a sweep reads it. The row-mean Z_t is still reported as teacher_keep_ratio.
            responses, lengths, fb_row, logp_row, z_row, _log_z, n_narrow = joint_generate(
                model_a, model_b, batch_a, batch_b, eos_ids=eos_ids, pad_id=pad_id, args=args,
            )
            narrow_steps += n_narrow
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
                mean_z[begin + row].append(
                    float(z_row[row].item()) / n_tok if n_tok else float("nan")
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

    if narrow_steps:
        # The top-p nucleus ran past the candidate cap on these row-steps, so sampling
        # was effectively top_p AND top-2048 rather than top_p alone. Normal in small
        # amounts on flat distributions; a large share means the cap, not top_p, is
        # setting the sampling width.
        print(f"[rank {rank}] top_p nucleus exceeded the {2048}-candidate cap on "
              f"{narrow_steps:,} row-steps ({narrow_steps / max(decided_tokens, 1):.2%} "
              f"of emitted tokens): those were sampled from top_p AND top-2048",
              file=sys.stderr, flush=True)

    if args.fuse == "agree" and decided_tokens:
        # A rate near 1.0 means the two models almost never overlapped under these
        # settings, so the constraint never bound and the run collapses to whichever
        # model owns the fallback. Raising --agree-student-top-k is usually the cheapest
        # response: it lets the student reach further down its own ranking for a token
        # the teacher already permits, without widening what the teacher permits.
        rate = fallback_tokens / decided_tokens
        print(f"[rank {rank}] fallback to {args.agree_fallback} on "
              f"{fallback_tokens:,}/{decided_tokens:,} tokens = {rate:.1%} "
              f"(student_top_k={args.agree_student_top_k}, "
              f"teacher_top_k={args.agree_teacher_top_k}, "
              f"teacher_min_prob={args.agree_teacher_min_prob}, "
              f"student_min_prob={args.agree_student_min_prob})", flush=True)
        # Z_t averaged over emitted tokens: the share of the student's probability mass
        # the teacher left standing. Near 1.0 the constraint is nominal; near 0 the
        # student is being pushed onto tokens it thought unlikely, which is what
        # student_prob then pays for.
        finite_z = [v for row in mean_z for v in row if not math.isnan(v)]
        if finite_z:
            print(f"[rank {rank}] teacher_keep_ratio (mean Z_t) = "
                  f"{sum(finite_z) / len(finite_z):.4f}", flush=True)

    shard["responses"] = texts
    shard["fallback_frac"] = fb_frac
    shard["student_mean_logp"] = mean_logp
    shard["teacher_keep_ratio"] = mean_z
    shard["response_lengths"] = tok_lens

    if not args.no_score:
        # Same entry point main_generation.py:216 uses, so scores are comparable
        # with everything else in the repo rather than a second opinion. sys.path
        # already reaches Myverl -- the module-level insert for the decode core did it.
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
