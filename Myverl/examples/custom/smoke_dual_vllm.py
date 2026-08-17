"""Smoke test: can ONE process host TWO sleeping vLLM engines?

This is the single high-risk unknown behind splitting reasoner / rephraser into two
models. `create_colocated_worker_cls` (verl/single_controller/ray/base.py:710) merges
every worker in a resource pool into ONE Ray actor process, so two worker groups on the
shared pool means two `LLM(...)` instances in one OS process. But vLLM's sleep mechanism
is built on a PROCESS-GLOBAL `CuMemAllocator` singleton, and the parallel state
(`_TP` / `_WORLD`) is global too. Nothing guarantees the second engine can be created,
or that engine A's `sleep()` will not release memory engine B is still holding.

Answer that here, in ~2 minutes of GPU time, before writing any trainer code.

What is checked, in order:

  1. both engines construct                 -> a global re-init assert fires here if ever
  2. B generates while A sleeps             -> A's sleep did not take B's weights with it
  3. A wakes and generates                  -> waking A did not evict B
  4. B's output is byte-identical to (2)    -> THE load-bearing check; see below
  5. a full alternating cycle x2            -> state does not degrade across rounds

Check 4 is the one that catches the failure mode that matters. If A's wake_up() quietly
corrupted B's weights, B would still generate happily -- just differently. Greedy
decoding (temperature=0) makes B's output a deterministic fingerprint of its weights,
so a mismatch means corruption. A crash would have been the lucky outcome.

Run it the way verl runs the real thing -- `external_launcher` needs torch.distributed,
which is what the Ray worker sets up and what torchrun gives us here:

    cd $HOME/LLM/Myverl
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        examples/custom/smoke_dual_vllm.py \\
        --model-a /home/data/shared/Qwen3-4b-base \\
        --model-b /home/data/shared/<sft-rephraser-ckpt>

Exit code 0 = share one 4-GPU pool as planned. Non-zero = fall back to a 2+2 split pool,
which additionally requires fixing four world_size-dependent sites (`_balance_batch`,
`_validate_summarize`'s `pad_dataproto_to_divisor`, the `ppo_mini_batch_size`
normalization divisor, and `_validate_config`'s divisibility checks).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def gpu_mem(tag: str, rank: int) -> None:
    """Print this rank's allocated / reserved GPU memory.

    Reserved is the number to watch: sleep(level=1) returns memory to the caching
    allocator, so a successful sleep shows reserved dropping while allocated may not.
    """
    if rank != 0:
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[mem] {tag:38s} allocated={allocated:6.2f}GB  reserved={reserved:6.2f}GB  total={total:6.2f}GB", flush=True)


def build_engine(name: str, model_path: str, args, rank: int):
    """Build one vLLM engine with the same arguments verl's rollout uses.

    Mirrors vllm_rollout_spmd.py:148-169 -- notably `enable_sleep_mode=True` and
    `distributed_executor_backend="external_launcher"`, the two settings that make the
    process-global allocator and parallel state relevant in the first place.
    """
    from vllm import LLM

    if rank == 0:
        print(f"\n[build] {name} <- {model_path}", flush=True)
    engine = LLM(
        model=model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=args.tensor_parallel,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem_util,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=args.max_model_len,
        disable_log_stats=True,
        max_num_batched_tokens=args.max_model_len,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=0,
    )
    gpu_mem(f"after building {name}", rank)
    return engine


def generate(engine, prompts: list[str], max_tokens: int) -> list[str]:
    """Greedy-decode so the output is a deterministic fingerprint of the weights."""
    from vllm import SamplingParams

    outputs = engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, n=1))
    return [o.outputs[0].text for o in outputs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True, help="Reasoner model path.")
    ap.add_argument("--model-b", required=True, help="Rephraser model path (may equal --model-a).")
    ap.add_argument("--gpu-mem-util", type=float, default=0.42,
                    help="Per-engine gpu_memory_utilization; two engines must fit (default: %(default)s).")
    ap.add_argument("--tensor-parallel", type=int, default=1, help="Per-engine TP size (default: %(default)s).")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="Kept small on purpose: this tests engine coexistence, not context length "
                         "(default: %(default)s).")
    ap.add_argument("--max-tokens", type=int, default=32, help="Tokens to generate per probe (default: %(default)s).")
    ap.add_argument("--rounds", type=int, default=2, help="Alternating sleep/wake cycles (default: %(default)s).")
    args = ap.parse_args()

    # external_launcher expects the ranks to already exist, exactly as they do inside the
    # Ray worker (fsdp_workers_new.py:105-110 does this init itself).
    rank = int(os.environ.get("RANK", -1))
    if rank < 0:
        print("must be launched with torchrun (external_launcher needs torch.distributed); see the "
              "module docstring", file=sys.stderr)
        return 2
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")
    if rank == 0:
        print(f"[init] world_size={world_size}  gpu_mem_util={args.gpu_mem_util} per engine "
              f"({2 * args.gpu_mem_util:.2f} total)", flush=True)
    gpu_mem("before any engine", rank)

    prompts = ["What is 2+2? Answer briefly.", "Name one prime number."]

    # --- 1. both engines construct -------------------------------------------------
    # verl builds the rollout engine then immediately sleeps it (vllm_rollout_spmd.py:172),
    # so A is asleep while B is being constructed. That is the ordering to reproduce: it is
    # the moment a process-global allocator would be re-entered.
    engine_a = build_engine("engine A (reasoner)", args.model_a, args, rank)
    engine_a.sleep(level=1)
    gpu_mem("after A.sleep(1)", rank)

    engine_b = build_engine("engine B (rephraser)", args.model_b, args, rank)
    gpu_mem("after building B (A asleep)", rank)

    # --- 2. B generates while A sleeps ---------------------------------------------
    if rank == 0:
        print("\n[check 2] B generates while A is asleep", flush=True)
    b_reference = generate(engine_b, prompts, args.max_tokens)
    if not all(text.strip() for text in b_reference):
        print(f"[FAIL] B produced empty output while A slept: {b_reference!r}", file=sys.stderr)
        return 1
    if rank == 0:
        print(f"[ok]    B: {b_reference[0][:70]!r}", flush=True)

    # --- 3/4. wake A, then re-probe B ----------------------------------------------
    # Swap which engine is resident. Only ONE holds GPU weights at a time -- that is the
    # whole point of sleep mode, and why 0.42 + 0.42 fits. What is being tested is not
    # simultaneous residency but whether the swap is CLEAN: two engines sharing one
    # process-global CuMemAllocator could free each other's memory on sleep, or hand back
    # a corrupted arena on wake.
    engine_b.sleep(level=1)
    engine_a.wake_up()
    gpu_mem("after B.sleep(1) + A.wake_up()", rank)

    if rank == 0:
        print("\n[check 3] A generates after waking", flush=True)
    a_reference = generate(engine_a, prompts, args.max_tokens)
    if not all(text.strip() for text in a_reference):
        print(f"[FAIL] A produced empty output after wake_up: {a_reference!r}", file=sys.stderr)
        return 1
    if rank == 0:
        print(f"[ok]    A: {a_reference[0][:70]!r}", flush=True)

    if rank == 0:
        print("\n[check 4] B still produces IDENTICAL output after A's wake/generate cycle", flush=True)
    engine_a.sleep(level=1)
    engine_b.wake_up()
    b_again = generate(engine_b, prompts, args.max_tokens)
    if b_again != b_reference:
        # Greedy decoding makes this deterministic, so a diff means A's cycle changed B's
        # weights or KV cache -- silent corruption, the failure a crash would have spared us.
        print("[FAIL] B's greedy output changed after A woke and generated -- the engines are "
              "corrupting each other", file=sys.stderr)
        for i, (before, after) in enumerate(zip(b_reference, b_again)):
            if before != after:
                print(f"        prompt {i} before: {before[:120]!r}", file=sys.stderr)
                print(f"        prompt {i} after : {after[:120]!r}", file=sys.stderr)
        return 1
    if rank == 0:
        print("[ok]    B unchanged", flush=True)

    # --- 5. alternating cycles -----------------------------------------------------
    # One successful swap could be luck. Repeat to catch state that degrades per cycle
    # (leaked allocator segments, a parallel-state counter drifting).
    if rank == 0:
        print(f"\n[check 5] {args.rounds} alternating cycle(s), both outputs must stay identical", flush=True)
    for round_idx in range(args.rounds):
        engine_b.sleep(level=1)
        engine_a.wake_up()
        if generate(engine_a, prompts, args.max_tokens) != a_reference:
            print(f"[FAIL] A's output changed in round {round_idx + 1}", file=sys.stderr)
            return 1
        engine_a.sleep(level=1)
        engine_b.wake_up()
        if generate(engine_b, prompts, args.max_tokens) != b_reference:
            print(f"[FAIL] B's output changed in round {round_idx + 1}", file=sys.stderr)
            return 1
        gpu_mem(f"after round {round_idx + 1}", rank)
        if rank == 0:
            print(f"[ok]    round {round_idx + 1}/{args.rounds}", flush=True)

    if rank == 0:
        print("\n=== PASS: two sleeping vLLM engines coexist in one process ===")
        print(f"    Share one {world_size}-GPU pool with gpu_memory_utilization={args.gpu_mem_util} "
              f"on both worker groups.", flush=True)
    torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
