"""Add a ``summarize_prompts`` column to a parquet that already carries
``token_split_points``.

Two modes (``--mode``):

* ``multi`` (default): for each row, given the K split points produced by
  ``add_token_split_points.py``, render K user-turn messages -- one per prefix
  length -- so a single question yields a length-K ``summarize_prompts`` array.
* ``full``: ignore the split points and render a SINGLE user-turn message whose
  [Reasoning Draft] is a leading fraction of the (answer-truncated) reasoning,
  i.e. one prompt per question (length-1 ``summarize_prompts`` array). The
  fraction is controlled by ``--full-ratio`` (default ``1.0`` = the complete
  reasoning). With ``ratio < 1.0`` the prefix is cut at ``ratio * n_tgt``
  tokens and then backed up to the nearest sentence/newline boundary so it
  never ends mid-sentence. The downstream dataset replicates this single
  prompt up to its configured K.

In both modes each rendered user turn has the form::

    You are an expert mathematician. You are given a [Problem] and a
    [Reasoning Draft] ... re-author a single "Gold Standard" solution,
    entirely self-contained, as if you solved it using only your own
    mathematical intuition. ... final answer within \boxed{}.

    [Problem]
    {question}

    [Reasoning Draft]
    {prefix_text_at_split_k}

    [Your solution]

These pre-rendered messages are consumed at training time by the
``prefix_mode='summarize'`` branch of the trainer (explain-style off-policy
loss). Doing the rendering offline avoids any decode/re-tokenize work in the
training loop.

Schema add:
    summarize_prompts : np.ndarray[object] of length K (``multi``) or length 1
                        (``full``), each element being a list[{role, content}]
                        (system + user).

Usage:

    python prepare_summarize_prompts.py \
        --input  $HOME/LLM/Data/deepmath_dgt6_n10000_split.parquet \
        --output $HOME/LLM/Data/deepmath_dgt6_n10000_summarize.parquet \
        --tokenizer-path /home/shared/Qwen2.5-Math-7B-16k-think \
        --mode full          # single prefix per question
        --full-ratio 0.5     # use only the first ~50% of the reasoning as draft
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE1 = (
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "You are given a [Problem] and a [Reasoning Draft] — an in-progress derivation for it that stops before "
    "the final answer.\n\n"
    "Your task is to re-author a single \"Gold Standard\" solution that is "
    "entirely self-contained, as if you solved the problem using only your "
    "own mathematical intuition.\n\n"
    "Strict requirements:\n"
    "1. Total De-Reference: Do NOT mention, quote, or even subtly refer to the "
    "[Reasoning Draft]. The reader must have no idea that any draft was provided to you.\n"
    "2. Invisible Integration: Re-derive the full content of the draft in your "
    "own words — the overall approach, the conditions and formulas it relies "
    "on, and every key step and intermediate result — completely and without "
    "omission, woven into one seamless derivation rather than a labeled "
    "summary. Do not copy the draft verbatim.\n"
    "3. Consistency Check: As you re-derive, make sure each calculation and "
    "logical step is correct and follows cleanly from the previous one, so the "
    "final solution reads as one coherent, error-free argument with no "
    "meta-talk or conversational fillers.\n"
    "4. Continue & Conclude: After re-deriving that reasoning, carry it forward "
    "and solve the problem step by step, and put your final answer within "
    "\\boxed{{}}. The final answer must follow from your own reasoning.\n\n"
    "[Problem]\n{question}\n\n"
    "[Reasoning Draft]\n{prefix}\n\n"
    "[Your solution]"
)

DEFAULT_TEMPLATE = (
    "You are given a [Problem] and a [Partial Reasoning Draft].\n\n"
    "Your task is to write one complete, self-contained solution. "
    "First, faithfully rephrase the reasoning already contained in the draft in your own words. " 
    "Then continue from that point and finish the remaining derivation. " 
    "The final output should read as a single coherent solution for the problem, not as a commentary on the draft.\n\n"
    "## Strict requirements:\n"
    "1. Do not mention or refer to the Partial Reasoning Draft. " 
    "The reader should not be able to tell that any draft was provided.\n"
    "2. Rephrase the valid reasoning in the draft as part of a natural solution. " 
    "Preserve the main approach, necessary conditions, formulas, intermediate results, " 
    "and nontrivial logical steps, but express them in your own clear wording. " 
    "Do not merely summarize the draft, and do not copy long phrases verbatim.\n"
    "3. After rephrasing the draft, continue the derivation from that point and solve the remaining part of the problem. "
    "The continuation should follow naturally from the rephrased reasoning, fill in any missing steps, and lead to a complete final answer."
    "If the draft contains an obvious error, silently correct it and continue with a valid derivation.\n"
    "4. Please reason step by step, and put your final answer within \\boxed{{}}."
    "## Problem:\n{question}\n\n"
    "## Partial Reasoning Draft:\n{prefix}\n\n"
    "## Your solution:"
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


def _find_boxed_spans(text: str) -> List[Tuple[int, str]]:
    """Locate every ``\\boxed{...}`` with balance-aware brace matching.

    Returns a list of ``(start, content)`` tuples in document order, where
    ``start`` is the index of the leading backslash and ``content`` is the
    text inside the (possibly nested) braces, e.g. ``\\frac{m}{n}`` for
    ``\\boxed{\\frac{m}{n}}``. Unbalanced trailing braces fall back to "rest
    of string" so we never crash on malformed input.
    """
    spans: List[Tuple[int, str]] = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        brace_open = m.end() - 1  # index of the opening '{'
        depth = 0
        i = brace_open
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    spans.append((m.start(), text[brace_open + 1:i]))
                    break
            i += 1
        else:
            spans.append((m.start(), text[brace_open + 1:]))
    return spans


def _truncate_before_final_answer(think_process: str) -> str:
    """Cut off the answer-revealing tail of ``think_process``.

    The split points produced by ``add_token_split_points.py`` span the whole
    thought process, whose final sentences almost always reveal the result
    (typically via a ``\\boxed{...}``). Any prefix that reaches that far would
    let the model simply copy the answer instead of re-deriving it.

    Strategy: take the content of the LAST ``\\boxed{...}`` as the final
    answer, then cut at the FIRST ``\\boxed{...}`` whose content is identical
    to it. This drops the earliest point where the correct answer is revealed
    (covering "we get X ... let me verify ... yes, X" patterns) while keeping
    earlier exploratory boxes whose content differs (e.g. a wrong guess that
    is later discarded).

    Brace matching is balance-aware so nested braces such as
    ``\\boxed{\\frac{m}{n}}`` are handled correctly. If no ``\\boxed`` is
    present the text is returned unchanged (the downstream clamp still bounds
    prefixes to the available tokens). If the answer is revealed already in the
    first sentence, an empty string is returned so the caller skips the row
    (no answer-free prefix exists).
    """
    if not think_process:
        return think_process

    spans = _find_boxed_spans(think_process)
    if not spans:
        return think_process

    final_content = spans[-1][1].strip()
    # Earliest \boxed whose content matches the final answer.
    box_pos = spans[-1][0]
    for start, content in spans:
        if content.strip() == final_content:
            box_pos = start
            break

    # Back up to the start of the sentence/line that contains that \boxed.
    # We cut at the latest sentence-ish boundary before box_pos: sentence-final
    # punctuation followed by whitespace, or a newline. This keeps the cut at a
    # natural boundary instead of mid-sentence.
    boundary = 0
    for m in re.finditer(r"(?:[.!?]+[\)\]\"']?\s+|\n+)", think_process[:box_pos]):
        boundary = m.end()
    cut = boundary

    # NOTE: cut == 0 means the answer is revealed already in the very first
    # sentence -- there is no answer-free prefix to keep, so we return an empty
    # string. The caller treats falsy think_process as "skip this row" (its
    # summarize_prompts becomes an empty array), which is the desired behaviour:
    # such rows cannot produce a safe prefix. We must NOT fall back to the
    # original text here, otherwise the answer would leak.
    return think_process[:cut].rstrip()


def _backup_to_sentence_boundary(prefix: str) -> str:
    """Back ``prefix`` up to its last sentence/newline boundary so a draft cut
    at an arbitrary token offset never ends mid-sentence.

    Used by ``full`` mode with ``--full-ratio < 1.0``: the prefix is first
    sliced to ``round(ratio * n_tgt)`` tokens (token-space, consistent with the
    split points), decoded back to text, then handed here. We look for the last
    sentence-final punctuation (optionally followed by a closing quote/bracket)
    + whitespace, or a newline, anywhere in ``prefix`` -- mirroring the boundary
    regex in ``_truncate_before_final_answer``. If no such boundary exists (very
    short / punctuation-free draft) we return the prefix unchanged so it is not
    collapsed to an empty string.
    """
    if not prefix:
        return prefix
    boundary = 0
    for m in re.finditer(r"(?:[.!?]+[\)\]\"']?\s+|\n+)", prefix):
        boundary = m.end()
    return prefix[:boundary].rstrip() if boundary > 0 else prefix.rstrip()


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
    mode: str = "multi",
    full_ratio: float = 1.0,
) -> Dict[str, Any]:
    global worker_tokenizer

    question = _extract_question(item)
    system_msg = _system_message(item)
    think_process = _extract_think_process(item)
    # Drop the answer-revealing tail so no prefix can leak the final answer.
    # split_points were computed on the FULL think process; after truncation
    # n_tgt shrinks and the existing clamp(sp, n_tgt) below squashes any
    # split point past the cut down to the truncated end (i.e. just before
    # the final-answer sentence).
    think_process = _truncate_before_final_answer(think_process)
    split_points = item.get("token_split_points")

    # Defensive fallbacks: if anything is missing, emit an empty np.array so
    # downstream code can detect and skip these rows.
    #
    # NOTE: ``full`` mode does not consume split_points (it always uses the
    # whole truncated think_process as the single prefix), so a missing
    # token_split_points column is NOT a blocker there.
    if (
        worker_tokenizer is None
        or not question
        or not think_process
        or (mode != "full" and split_points is None)
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

    # ``full`` mode: a single prefix = a leading fraction of the
    # (answer-truncated) reasoning, i.e. one synthetic split point at
    # ``round(full_ratio * n_tgt)``. With ratio == 1.0 this is the whole
    # truncated reasoning (original behaviour); with ratio < 1.0 the decoded
    # prefix is backed up to the nearest sentence boundary so it never ends
    # mid-sentence. Yields a length-1 summarize_prompts array; the downstream
    # dataset replicates it up to K.
    if mode == "full":
        sp_full = int(round(full_ratio * n_tgt))
        effective_points: List[int] = [sp_full]
        apply_boundary_backup = full_ratio < 1.0
    else:
        effective_points = [int(sp) for sp in split_points]
        apply_boundary_backup = False

    prompts_list: List[List[Dict[str, str]]] = []
    for sp in effective_points:
        sp = int(sp)
        sp = max(0, min(sp, n_tgt))  # clamp
        if sp == 0:
            prefix_text = ""
        else:
            prefix_text = worker_tokenizer.decode(
                tgt_tokens[:sp], skip_special_tokens=True
            )
            if apply_boundary_backup:
                prefix_text = _backup_to_sentence_boundary(prefix_text)
        msgs = _build_summarize_prompt(system_msg, question, prefix_text, template)
        prompts_list.append(msgs)

    # IMPORTANT: 不能用 np.array(prompts_list, dtype=object)。
    # 每个 msgs 都是 [system, user] 长度 2 的 list，numpy 会自动推断成 2D
    # shape=(K, 2)，pyarrow 写 parquet 时只接受 1D object 列、会报
    # "Only 1D arrays accepted"。显式构造 1D object array 才能让每个 cell
    # 保持成 list of dict，让 pyarrow 序列化为 list[list[struct]]。
    arr = np.empty(len(prompts_list), dtype=object)
    for i, p in enumerate(prompts_list):
        arr[i] = p
    item["summarize_prompts"] = arr
    return item


def _worker_process(args: Tuple[Dict[str, Any], str, str, float]) -> Dict[str, Any]:
    item, template, mode, full_ratio = args
    return process_single_item(item, template, mode, full_ratio)


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
        "--mode",
        choices=["multi", "full"],
        default="multi",
        help="multi (default): one prompt per split point -> length-K array. "
             "full: a single prompt whose draft is a leading fraction of the "
             "(answer-truncated) reasoning -> length-1 array (split points ignored).",
    )
    parser.add_argument(
        "--full-ratio",
        type=float,
        default=1.0,
        help="Only used in --mode full. Fraction (0, 1] of the answer-truncated "
             "reasoning (in token space) to use as the [Reasoning Draft]. "
             "1.0 (default) = the complete reasoning; e.g. 0.5 = the first ~50%% "
             "of tokens, backed up to the nearest sentence boundary.",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="User-turn template, must include {question} and {prefix} placeholders.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick smoke tests.")
    args = parser.parse_args()

    if "{question}" not in args.template or "{prefix}" not in args.template:
        raise SystemExit("--template must contain both {question} and {prefix}")

    if not (0.0 < args.full_ratio <= 1.0):
        raise SystemExit("--full-ratio must be in the (0, 1] range")
    if args.mode != "full" and args.full_ratio != 1.0:
        print("  (--full-ratio is ignored unless --mode full)")

    print(f"Reading {args.input}")
    df = pd.read_parquet(args.input)
    if "token_split_points" not in df.columns:
        # ``full`` mode never reads split points, so a missing column is fine
        # there; ``multi`` mode requires them.
        if args.mode == "full":
            print(
                "  (no 'token_split_points' column; OK in --mode full, "
                "split points are not used)"
            )
        else:
            raise SystemExit(
                "Input parquet has no 'token_split_points' column. "
                "Run Data/add_token_split_points.py first."
            )

    if args.limit is not None:
        df = df.head(args.limit)
    records = df.to_dict(orient="records")
    print(f"  -> {len(records):,} rows. Tokenizer: {args.tokenizer_path}. Mode: {args.mode}")
    if args.mode == "full":
        print(f"     full-ratio: {args.full_ratio}")

    workers = max(1, min(args.workers, mp.cpu_count()))
    if workers == 1:
        init_worker(args.tokenizer_path)
        processed = [
            process_single_item(item, args.template, args.mode, args.full_ratio)
            for item in tqdm(records, total=len(records), desc="rendering")
        ]
    else:
        task_iter = ((item, args.template, args.mode, args.full_ratio) for item in records)
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

    # Print a sample for human-eyeball verification. ``full`` mode emits
    # length-1 arrays, so require >= 1 (not >= 2) and only index entries that
    # exist. token_split_points may be absent in ``full`` mode, so guard it.
    sample = next(
        (
            row for _, row in out_df.iterrows()
            if isinstance(row["summarize_prompts"], np.ndarray)
            and len(row["summarize_prompts"]) >= 1
        ),
        None,
    )
    if sample is not None:
        sps = sample["summarize_prompts"]
        sample_points = sample.get("token_split_points")
        idxs = sorted({0, len(sps) // 2, len(sps) - 1})
        for idx in idxs:
            sp_str = ""
            if isinstance(sample_points, (list, np.ndarray)) and idx < len(sample_points):
                sp_str = f" (split_point={sample_points[idx]})"
            print(f"\n--- sample summarize_prompts[{idx}]{sp_str} ---")
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
