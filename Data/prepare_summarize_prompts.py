"""Add a ``summarize_prompts`` column to a parquet that already carries
``token_split_points``.

For each row, given the K split points produced by ``add_token_split_points.py``
this script renders K user-turn messages of the form::

    {question}

    【已完成的部分推理】
    {prefix_text_at_split_k}

    请先用自己的话总结上面的思路，再继续推理给出最终答案。

These pre-rendered messages are consumed at training time by the
``prefix_mode='summarize'`` branch of the trainer (explain-style off-policy
loss). Doing the rendering offline avoids any decode/re-tokenize work in the
training loop.

Schema add:
    summarize_prompts : np.ndarray[object] of length K, each element being a
                        list[{role, content}] (system + user).

Usage:

    python prepare_summarize_prompts.py \
        --input  $HOME/LLM/Data/deepmath_dgt6_n10000_split.parquet \
        --output $HOME/LLM/Data/deepmath_dgt6_n10000_summarize.parquet \
        --tokenizer-path /home/shared/Qwen2.5-Math-7B-16k-think
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = (
    "{question}\n\n"
    "【已完成的部分推理】\n{prefix}\n\n"
    "请先用自己的话总结上面的思路，再继续推理给出最终答案。"
)

worker_tokenizer = None


def init_worker(tokenizer_path: str) -> None:
    global worker_tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_think_process(item: Dict[str, Any]) -> str:
    """Same convention as add_token_split_points.py:extract_think_process.

    Pull the text inside ``<think>...</think>`` from ``target[0].content``.
    Falls back to ``thought_process`` / ``thinking`` keys if absent.
    """
    target_val = item.get("target", "")
    text = ""

    if isinstance(target_val, (list, np.ndarray)) and len(target_val) > 0:
        first = target_val[0]
        if isinstance(first, dict):
            text = first.get("content", "") or ""
        elif isinstance(first, str):
            text = first
    elif isinstance(target_val, dict):
        text = target_val.get("content", "") or ""
    elif isinstance(target_val, str):
        text = target_val
    else:
        text = item.get("thought_process", "") or item.get("thinking", "") or ""

    think_process = text.split("</think>")[0].strip()
    if "<think>" in think_process:
        think_process = think_process.replace("<think>", "").strip()
    return think_process


def _extract_question(item: Dict[str, Any]) -> str:
    """Pull the user's original question out of ``prompt`` (chat-style list)."""
    prompt = item.get("prompt")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return ""
    # prompt = [{system, ...}, {user, content: question}, ...]
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    # Fallback: prompt[1] convention
    if len(prompt) >= 2 and isinstance(prompt[1], dict):
        return str(prompt[1].get("content", ""))
    return ""


def _system_message(item: Dict[str, Any]) -> Dict[str, str] | None:
    """Pull the system message out of ``prompt`` if present."""
    prompt = item.get("prompt")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return None
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return {"role": "system", "content": str(msg.get("content", ""))}
    # Fallback: prompt[0] convention
    if len(prompt) >= 1 and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        return {"role": "system", "content": str(prompt[0].get("content", ""))}
    return None


def _build_summarize_prompt(
    system_msg: Dict[str, str] | None,
    question: str,
    prefix_text: str,
    template: str,
) -> List[Dict[str, str]]:
    """Build a [system?, user] messages list. Empty prefix -> still wraps the
    question in the template (the user explicitly asked the model to summarize
    "what's been done so far"; with empty prefix the summary will just be
    trivial, training keeps working). The trainer / dataset can choose to
    short-circuit step_i==0 separately if desired."""
    user_content = template.format(question=question, prefix=prefix_text)
    messages: List[Dict[str, str]] = []
    if system_msg is not None:
        messages.append(system_msg)
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_single_item(
    item: Dict[str, Any],
    template: str,
) -> Dict[str, Any]:
    global worker_tokenizer

    question = _extract_question(item)
    system_msg = _system_message(item)
    think_process = _extract_think_process(item)
    split_points = item.get("token_split_points")

    # Defensive fallbacks: if anything is missing, emit an empty np.array so
    # downstream code can detect and skip these rows.
    if (
        worker_tokenizer is None
        or not question
        or not think_process
        or split_points is None
    ):
        item["summarize_prompts"] = np.array([], dtype=object)
        return item

    if isinstance(split_points, np.ndarray):
        split_points = split_points.tolist()

    # Encode the think_process once; we'll slice tokens by split_point and
    # decode each prefix back to text.
    tgt_tokens = worker_tokenizer(
        think_process, add_special_tokens=False
    )["input_ids"]
    n_tgt = len(tgt_tokens)

    prompts_list: List[List[Dict[str, str]]] = []
    for sp in split_points:
        sp = int(sp)
        sp = max(0, min(sp, n_tgt))  # clamp
        if sp == 0:
            prefix_text = ""
        else:
            prefix_text = worker_tokenizer.decode(
                tgt_tokens[:sp], skip_special_tokens=True
            )
        msgs = _build_summarize_prompt(system_msg, question, prefix_text, template)
        prompts_list.append(msgs)

    item["summarize_prompts"] = np.array(prompts_list, dtype=object)
    return item


def _worker_process(args: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    item, template = args
    return process_single_item(item, template)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input parquet (must have token_split_points column).")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument(
        "--tokenizer-path",
        default="/home/shared/Qwen2.5-Math-7B-16k-think",
        help="Tokenizer path. MUST match the tokenizer used by add_token_split_points.py "
             "or split-point token counts will misalign with text offsets.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Worker process count (1 = sync).")
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="User-turn template, must include {question} and {prefix} placeholders.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick smoke tests.")
    args = parser.parse_args()

    if "{question}" not in args.template or "{prefix}" not in args.template:
        raise SystemExit("--template must contain both {question} and {prefix}")

    print(f"Reading {args.input}")
    df = pd.read_parquet(args.input)
    if "token_split_points" not in df.columns:
        raise SystemExit(
            "Input parquet has no 'token_split_points' column. "
            "Run Data/add_token_split_points.py first."
        )

    if args.limit is not None:
        df = df.head(args.limit)
    records = df.to_dict(orient="records")
    print(f"  -> {len(records):,} rows. Tokenizer: {args.tokenizer_path}")

    workers = max(1, min(args.workers, mp.cpu_count()))
    if workers == 1:
        init_worker(args.tokenizer_path)
        processed = [
            process_single_item(item, args.template)
            for item in tqdm(records, total=len(records), desc="rendering")
        ]
    else:
        task_iter = ((item, args.template) for item in records)
        with mp.Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(args.tokenizer_path,),
        ) as pool:
            chunksize = max(1, len(records) // (workers * 4)) if records else 1
            processed = list(
                tqdm(
                    pool.imap(_worker_process, task_iter, chunksize=chunksize),
                    total=len(records),
                    desc="rendering",
                )
            )

    out_df = pd.DataFrame(processed)
    n_emitted = sum(
        1 for p in out_df["summarize_prompts"]
        if isinstance(p, np.ndarray) and len(p) > 0
    )
    print(f"  -> {n_emitted:,}/{len(out_df):,} rows have non-empty summarize_prompts.")

    # Print a sample for human-eyeball verification.
    sample = next(
        (
            row for _, row in out_df.iterrows()
            if isinstance(row["summarize_prompts"], np.ndarray)
            and len(row["summarize_prompts"]) >= 2
        ),
        None,
    )
    if sample is not None:
        sps = sample["summarize_prompts"]
        for idx in [0, len(sps) // 2, len(sps) - 1]:
            print(f"\n--- sample summarize_prompts[{idx}] (split_point={sample['token_split_points'][idx]}) ---")
            for msg in sps[idx]:
                content = msg.get("content", "")
                head = content[:300].replace("\n", " ")
                tail = content[-200:].replace("\n", " ") if len(content) > 500 else ""
                print(f"  [{msg.get('role')}] {head}{' ... ' + tail if tail else ''}")

    out_df.to_parquet(args.output, index=False)
    size_mb = pd.Series([0]).memory_usage()  # placeholder, real size below
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nWrote {args.output} ({size_mb:.1f} MB, {len(out_df):,} rows)")


if __name__ == "__main__":
    main()
