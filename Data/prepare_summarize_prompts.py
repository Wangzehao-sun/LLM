"""Add TWO summarize prompt columns to a parquet that (for ``--list-mode multi``)
already carries ``token_split_points``.

This one run produces two physically-separated columns so the extra-step and
normal-step paths never share data:

* ``summarize_prompt`` (length-1 array): the SINGLE prompt consumed by the
  extra-step / recycle path (all off rollouts share it). Controlled by
  ``--single-mode``:
    - ``full`` (default): prefix = a fixed ``--full-ratio`` fraction of the
      answer-truncated reasoning, same for every row.
    - ``progressive``: the fraction DECREASES across training steps. Rows are
      grouped by ``--batch-size`` (row ``i`` -> step ``i // batch_size``) and the
      ratio falls linearly from ``--ratio-start`` to ``--ratio-end`` over the
      whole dataset. Assumes the trainer reads rows in order with shuffle off.
    - ``random``: each row draws its prefix ratio independently and uniformly
      from ``[--ratio-min, --ratio-max]`` (seeded by ``--seed`` for
      reproducibility). Use this to diversify the recycle draft length per
      question instead of a single fixed ratio.

* ``summarize_prompts`` (length-K array, prefix ASCENDING): the prompt list
  consumed by the normal-step summarize_replace path, which scans in order and
  picks the SHORTEST correct candidate. Controlled by ``--list-mode``:
    - ``multi`` (default): one prompt per ``token_split_points`` entry (the split
      points are produced ascending, so prefixes are ascending).
    - ``custom``: one prompt per ratio in ``--list-ratios`` (sorted ascending);
      ``token_split_points`` not required.

With ``ratio < 1.0`` a prefix is cut at ``round(ratio * n_tgt)`` tokens and then
backed up to the nearest sentence/newline boundary so it never ends mid-sentence.

Each rendered user turn has the form::

    You are given a [Problem] and a [Reasoning Draft] ...

    [Problem]
    {question}

    [Reasoning Draft]
    {prefix_text}

    [Your solution]

These pre-rendered messages are consumed at training time: ``summarize_prompts``
by the normal-step summarize_replace, ``summarize_prompt`` by the extra-step
``prefix_mode='summarize'`` branch. Rendering offline avoids any decode/
re-tokenize work in the training loop.

Schema add:
    summarize_prompt  : np.ndarray[object] of length 1
    summarize_prompts : np.ndarray[object] of length K (prefix ascending)
                        each element being a list[{role, content}] (system + user).
    teacher_prompts   : (OPTIONAL, with --teacher-prompts) np.ndarray[object] of
                        length 1, the online teacher-API's OWN single pre-rendered
                        prompt, built with a SEPARATE template (--teacher-template) and
                        a single prefix (--teacher-ratio), kept physically apart from
                        summarize_prompts. Column name is --teacher-col and must match
                        teacher_api.prompt_key at train time. The K summarize-replace
                        candidates all reuse this one prompt.

Usage:

    python prepare_summarize_prompts.py \
        --input  $HOME/LLM/Data/deepmath_dgt6_n10000_split.parquet \
        --output $HOME/LLM/Data/deepmath_dgt6_n10000_summarize.parquet \
        --tokenizer-path /home/shared/Qwen2.5-Math-7B-16k-think \
        --single-mode full --full-ratio 1.0 \
        --list-mode custom --list-ratios 0.2,0.4,0.6,0.8

    # also render the separate SINGLE teacher-API prompt column (teacher_prompts):
    python prepare_summarize_prompts.py \
        --input  ..._split.parquet --output ..._summarize.parquet \
        --tokenizer-path /home/shared/Qwen2.5-Math-7B-16k-think \
        --list-mode custom --list-ratios 0.2,0.4,0.6,0.8 \
        --teacher-prompts --teacher-ratio 0.5

    # teacher-ONLY: render just the teacher column, no token_split_points needed:
    python prepare_summarize_prompts.py \
        --input  ..._nosplit.parquet --output ..._teacher.parquet \
        --tokenizer-path /home/shared/Qwen2.5-Math-7B-16k-think \
        --teacher-only --teacher-ratio 0.5

    # teacher-ONLY, no tokenizer at all (char-space prefix; no transformers needed):
    python prepare_summarize_prompts.py \
        --input  ..._nosplit.parquet --output ..._teacher.parquet \
        --teacher-only --no-tokenizer --teacher-ratio 0.5
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
# NOTE: transformers is imported lazily inside init_worker so that --no-tokenizer /
# --teacher-only (char-space) runs need neither the library nor a model download.

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

DEFAULT_TEMPLATE2 = (
    "You are given a [Problem] and a [Partial Reasoning Draft].\n\n"
    "Your task is to write one complete, self-contained solution. "
    "Use the [Reference Reasoning Draft] as private mathematical guidance: understand its reasoning process, "
    "extract the useful and valid reasoning steps, and reconstruct them in your own step-by-step problem-solving style. "
    "Then continue the derivation naturally until the problem is fully solved. "
    "The final output should be a standard, well-organized solution to the problem, not a commentary on the draft.\n"
    "## Strict requirements:\n" 
    "1. Use the Reference Reasoning Draft as private mathematical guidance. " 
    "Understand its reasoning process, extract its useful and valid steps, and rewrite them in your own step-by-step problem-solving style, as if solving the problem directly.\n" 
    "Do not explicitly mention the draft, the prefix, or that any prior reasoning was provided. " 
    "2. Reconstruct the reasoning rather than merely paraphrasing it. " 
    "In the reconstructed part, preserve as many valid reasoning steps from the draft as possible, including important equations, intermediate conclusions, and useful verification or correction steps. " 
    "For each reconstructed step, explain the reasoning in your own words rather than merely copying the draft's wording."
    "The reasoning should be reorganized into a clear, coherent, and natural solution.\n" 
    "3. Do not summarize, compress, or skip the draft's valid reasoning steps. " 
    "For each step of the derivation, explain the underlying mathematical reasoning in your own words rather than only stating the result.\n" 
    "4. Continue naturally from the reconstructed reasoning. " 
    "If the draft contains an obvious mathematical mistake, correct it silently and continue with a valid derivation. " 
    "Add any necessary new valid steps to complete the derivation and reach the final answer.\n"
    "5. Please reason step by step, and put your final answer within \\boxed{{}}."
    "## Problem:\n{question}\n\n"
    "## Reference Reasoning Draft:\n{prefix}\n\n"
    "## Your solution:"
)

DEFAULT_TEMPLATE3 = (
    "You are given a [Problem] and a [Noisy Reasoning Draft].\n\n"
    "Your task is to write one complete, self-contained solution. "
    "Use the [Reference Reasoning Draft] as private mathematical guidance: understand its reasoning process, "
    "extract the useful and valid reasoning steps, and reconstruct them in your own step-by-step problem-solving style. "
    "Then continue the derivation naturally until the problem is fully solved. "
    "The final output should be a standard, well-organized solution to the problem, not a commentary on the draft.\n"
    "## Strict requirements:\n" 
    "1. Use the Reference Reasoning Draft as private mathematical guidance. " 
    "Understand its reasoning process, extract its useful and valid steps, and rewrite them in your own step-by-step problem-solving style, as if solving the problem directly.\n" 
    "Do not explicitly mention the draft, the prefix, or that any prior reasoning was provided. " 
    "2. Reconstruct the reasoning rather than merely paraphrasing it. " 
    "In the reconstructed part, preserve as many valid reasoning steps from the draft as possible, including important equations, intermediate conclusions, and useful verification or correction steps. " 
    "For each reconstructed step, explain the reasoning in your own words rather than merely copying the draft's wording."
    "The reasoning should be reorganized into a clear, coherent, and natural solution.\n" 
    "3. Do not summarize, compress, or skip the draft's valid reasoning steps. " 
    "For each step of the derivation, explain the underlying mathematical reasoning in your own words rather than only stating the result.\n" 
    "4. Continue naturally from the reconstructed reasoning. " 
    "Add any necessary new valid steps to complete the derivation and reach the final answer. "
    "If the draft contains an obvious mathematical mistake, correct it silently and continue with a valid derivation.\n" 
    "5. Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "## Problem:\n{question}\n\n"
    "## Reference Reasoning Draft:\n{prefix}\n\n"
    "## Your solution:"
)
DEFAULT_TEMPLATE = (
    "You are given a [Problem] and a [Noisy Reasoning Draft].\n\n"
    "Your task is to write one complete, self-contained solution. "
    "Use the [Reference Reasoning Draft] as private mathematical guidance: understand its reasoning process, "
    "extract the useful and valid reasoning steps, and reconstruct them in your own step-by-step problem-solving style. "
    "Then continue the derivation naturally until the problem is fully solved. "
    "The final output should be a standard, well-organized solution to the problem, not a commentary on the draft.\n"
    "## Strict requirements:\n" 
    "1. Use the Reference Reasoning Draft as private mathematical guidance. " 
    "Understand its reasoning process, extract its useful and valid steps, and rewrite them in your own step-by-step problem-solving style, as if solving the problem directly.\n" 
    "Do not explicitly mention the draft, the prefix, or that any prior reasoning was provided. " 
    "2. Reconstruct the reasoning rather than merely paraphrasing it. " 
    "In the reconstructed part, preserve as many valid reasoning steps from the draft as possible, including important equations, intermediate conclusions, and useful verification or correction steps. " 
    "For each reconstructed step, explain the reasoning in your own words rather than merely copying the draft's wording."
    "The reasoning should be reorganized into a clear, coherent, and natural solution.\n" 
    "3. Do not summarize, compress, or skip the draft's valid reasoning steps. " 
    "For each step of the derivation, explain the underlying mathematical reasoning in your own words rather than only stating the result.\n" 
    "4. Continue naturally from the reconstructed reasoning. " 
    "Add any necessary new valid steps to complete the derivation and reach the final answer. "
    "If the draft contains an obvious mathematical mistake, correct it silently and continue with a valid derivation.\n" 
    "5. Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "## Problem:\n{question}\n\n"
    "## Reference Reasoning Draft:\n{prefix}\n\n"
    "## Your solution:"
)

worker_tokenizer = None

# Sentinel meaning "run without a tokenizer": teacher-only mode then cuts the prefix
# in CHARACTER space instead of token space (no transformers / model download needed).
NO_TOKENIZER = "__no_tokenizer__"


# ---------------------------------------------------------------------------
# teacher-API prompts (independent column, separate template)
# ---------------------------------------------------------------------------
# Rendered into a SEPARATE column (default 'teacher_prompts') consumed ONLY by the
# online teacher-API rephraser (teacher_api.prompt_key). Kept physically apart from
# the policy's summarize_prompts so the two prompt designs never mix. Same prefix
# machinery (split points / ratios, ascending), but its own template below.
TEACHER_TEMPLATE_DEFAULT = (
    "You are given a mathematical problem, a detailed expert reasoning draft, and several "
    "outputs that illustrate the target model's natural reasoning style.\n\n"
    "Write one complete solution to the problem.\n\n"
    "Use the expert draft for mathematical guidance: reconstruct its valid reasoning path, "
    "preserve its useful intermediate steps, equations, checks, and corrections, and fill in "
    "any implicit transitions needed for a self-contained solution.\n\n"
    "Use the style examples only to learn the target model's reasoning tone, pacing, and way "
    "of connecting steps. Do not reuse their mathematical content, conclusions, or "
    "problem-specific wording.\n\n"
    "Requirements:\n"
    "1. Solve the problem directly without mentioning the expert draft or the style examples.\n"
    "2. Preserve the useful detail of the expert reasoning rather than summarizing it.\n"
    "3. Express the reasoning naturally in the target style, without artificial headings or "
    "deliberately imitated mistakes.\n"
    "4. Silently correct any clear error in the draft, while keeping its valid reasoning path "
    "whenever possible.\n"
    "5. Output only the solution and place the final answer in \\boxed{}.\n\n"
    "## Target Style Examples\n\n"
    "### Example 1\n{style_example_1}\n\n"
    "### Example 2\n{style_example_2}\n\n"
    "## Problem\n{question}\n\n"
    "## Expert Reasoning Draft\n{prefix}\n\n"
    "## Solution"
)

# Style-example placeholders left UNFILLED at offline render time and substituted at
# training time by the trainer with the policy's own recent correct rollouts. Offline we
# only fill {question} and {prefix}; these tokens pass through verbatim.
TEACHER_STYLE_PLACEHOLDERS = ("{style_example_1}", "{style_example_2}")


def _render_teacher_template(template: str, question: str, prefix_text: str) -> str:
    """Render the teacher template WITHOUT str.format.

    The teacher template intentionally contains (a) literal ``\\boxed{}`` braces and
    (b) style-example placeholders that must survive offline rendering to be filled at
    train time. str.format would choke on the literal braces / unknown fields, so we do
    targeted string replacement of only ``{question}`` and ``{prefix}``.
    """
    return template.replace("{question}", question).replace("{prefix}", prefix_text)


def init_worker(tokenizer_path: str) -> None:
    global worker_tokenizer
    # Empty / "none" path -> char-space mode: skip loading any tokenizer.
    if not tokenizer_path or str(tokenizer_path).lower() in ("none", "null", "no_tokenizer"):
        worker_tokenizer = NO_TOKENIZER
        return
    from transformers import AutoTokenizer  # lazy: only needed with a real tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def compute_progressive_ratios(
    n_rows: int,
    batch_size: int,
    ratio_start: float,
    ratio_end: float,
) -> List[float]:
    """``progressive`` 课程：每行一个随 step 递减的 prefix 比例。

    训练时 dataloader 按顺序消费数据（要求 shuffle 关闭），所以第 ``i`` 行落在
    step ``i // batch_size``。prefix 比例随 step 线性递减：第一个 step 用
    ``ratio_start``，最后一个 step 用 ``ratio_end``，在整个数据集上均匀分布。
    同一个 step（同一 batch）内的所有行共用一个比例。

    返回长度为 ``n_rows`` 的 float 列表（每行一个比例）。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0 for progressive mode")
    n_steps = (n_rows + batch_size - 1) // batch_size  # ceil
    if n_steps <= 1:
        step_ratio = [ratio_start]
    else:
        step_ratio = list(np.linspace(ratio_start, ratio_end, n_steps))
    return [float(step_ratio[i // batch_size]) for i in range(n_rows)]


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
    is_teacher: bool = False,
) -> List[Dict[str, str]]:
    """Build a [system?, user] messages list. Empty prefix -> still wraps the
    question in the template (the user explicitly asked the model to summarize
    "what's been done so far"; with empty prefix the summary will just be
    trivial, training keeps working). The trainer / dataset can choose to
    short-circuit step_i==0 separately if desired.

    ``is_teacher`` selects the teacher render path (str.replace, preserves literal
    ``\\boxed{}`` and the {style_example_*} placeholders) instead of str.format.
    """
    if is_teacher:
        user_content = _render_teacher_template(template, question, prefix_text)
    else:
        user_content = template.format(question=question, prefix=prefix_text)
    messages: List[Dict[str, str]] = []
    if system_msg is not None:
        messages.append(system_msg)
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def _build_prompt_array(
    system_msg: Dict[str, str] | None,
    question: str,
    template: str,
    tgt_tokens: List[int],
    n_tgt: int,
    sp_backup_pairs: List[Tuple[int, bool]],
    is_teacher: bool = False,
) -> np.ndarray:
    """Render a 1D object array of message-lists, one per (split_point, backup).

    ``sp_backup_pairs`` is a list of ``(sp, do_backup)`` where ``sp`` is an offset
    into the (answer-truncated) reasoning and ``do_backup`` says whether to back the
    prefix up to a sentence boundary (used when the prefix is a fractional cut rather
    than a sentence-aligned split point).

    With a real tokenizer ``tgt_tokens`` are token ids and ``sp`` is a token count
    (decoded back to text). In no-tokenizer mode ``tgt_tokens`` is the raw reasoning
    STRING and ``sp`` is a CHARACTER count -- the prefix is sliced directly in char
    space (no decode). Both paths then optionally back up to a sentence boundary.
    """
    no_tok = (worker_tokenizer == NO_TOKENIZER)
    prompts_list: List[List[Dict[str, str]]] = []
    for sp, do_backup in sp_backup_pairs:
        sp = max(0, min(int(sp), n_tgt))  # clamp
        if sp == 0:
            prefix_text = ""
        else:
            if no_tok:
                # char-space slice: tgt_tokens is the reasoning string here
                prefix_text = tgt_tokens[:sp]
            else:
                prefix_text = worker_tokenizer.decode(
                    tgt_tokens[:sp], skip_special_tokens=True
                )
            if do_backup:
                prefix_text = _backup_to_sentence_boundary(prefix_text)
        prompts_list.append(
            _build_summarize_prompt(system_msg, question, prefix_text, template, is_teacher)
        )

    # IMPORTANT: 不能用 np.array(prompts_list, dtype=object)。
    # 每个 msgs 都是 [system, user] 长度 2 的 list，numpy 会自动推断成 2D
    # shape=(K, 2)，pyarrow 写 parquet 时只接受 1D object 列、会报
    # "Only 1D arrays accepted"。显式构造 1D object array 才能让每个 cell
    # 保持成 list of dict，让 pyarrow 序列化为 list[list[struct]]。
    arr = np.empty(len(prompts_list), dtype=object)
    for i, p in enumerate(prompts_list):
        arr[i] = p
    return arr


def process_single_item(
    item: Dict[str, Any],
    template: str,
    single_mode: str = "full",
    single_ratio: float = 1.0,
    list_mode: str = "multi",
    list_ratios: List[float] | None = None,
    teacher_template: str | None = None,
    teacher_ratio: float = 1.0,
    teacher_col: str = "teacher_prompts",
    teacher_only: bool = False,
) -> Dict[str, Any]:
    """Render the summarize columns (and, if requested, the teacher column) for one row.

    * ``summarize_prompt`` (length-1 array): the SINGLE prompt consumed by the
      extra-step / recycle path. Its prefix is a leading fraction
      (``single_ratio``) of the answer-truncated reasoning. ``single_mode`` is
      "full" (fixed ratio for every row) or "progressive" (the caller passes a
      per-row ``single_ratio`` that decreases with the training step).
    * ``summarize_prompts`` (length-K array, prefix ASCENDING): the prompt list
      consumed by the normal-step summarize_replace path, which picks the
      shortest correct candidate by scanning in order. ``list_mode`` is "multi"
      (one prompt per ``token_split_points`` entry -- already ascending) or
      "custom" (one prompt per ratio in the sorted ``list_ratios``).
    * ``<teacher_col>`` (length-1 array): OPTIONAL, only when ``teacher_template``
      is not None. The teacher-API's OWN single pre-rendered prompt, built with
      ``teacher_template`` and a single prefix at ``teacher_ratio``. Kept in a
      separate column so the teacher prompt design never mixes with the policy's.
      (The K summarize-replace candidates all reuse this one prompt -- K stochastic
      teacher samples of the same request.)
    """
    global worker_tokenizer

    question = _extract_question(item)
    system_msg = _system_message(item)
    think_process = _extract_think_process(item)
    # Drop the answer-revealing tail so no prefix can leak the final answer.
    think_process = _truncate_before_final_answer(think_process)
    split_points = item.get("token_split_points")

    # ``multi`` list mode is the only consumer of token_split_points; the single
    # column, ``custom`` list mode and the teacher column derive prefixes from
    # ratios, so a missing split-points column only blocks ``multi``. In
    # ``teacher_only`` we render ONLY the teacher column and never need split points.
    need_split = (list_mode == "multi") and (not teacher_only)
    if (
        worker_tokenizer is None
        or not question
        or not think_process
        or (need_split and split_points is None)
    ):
        if teacher_only:
            item[teacher_col] = np.array([], dtype=object)
        else:
            item["summarize_prompt"] = np.array([], dtype=object)
            item["summarize_prompts"] = np.array([], dtype=object)
            if teacher_template is not None:
                item[teacher_col] = np.array([], dtype=object)
        return item

    if isinstance(split_points, np.ndarray):
        split_points = split_points.tolist()

    # Encode the think_process once; reused for all columns.
    # No-tokenizer mode: work in CHARACTER space -- tgt_tokens IS the string, n_tgt its
    # char length. Only teacher-only supports this (summarize split points are token-space).
    if worker_tokenizer == NO_TOKENIZER:
        tgt_tokens = think_process
        n_tgt = len(think_process)
    else:
        tgt_tokens = worker_tokenizer(
            think_process, add_special_tokens=False
        )["input_ids"]
        n_tgt = len(tgt_tokens)

    # --- teacher-only: render just the teacher column, skip summarize entirely ---
    if teacher_only:
        sp_teacher = int(round(teacher_ratio * n_tgt))
        item[teacher_col] = _build_prompt_array(
            system_msg, question, teacher_template, tgt_tokens, n_tgt,
            [(sp_teacher, teacher_ratio < 1.0)], is_teacher=True,
        )
        return item

    # --- single column (extra-step): one fractional prefix ---
    sp_single = int(round(single_ratio * n_tgt))
    item["summarize_prompt"] = _build_prompt_array(
        system_msg, question, template, tgt_tokens, n_tgt,
        [(sp_single, single_ratio < 1.0)],
    )

    # --- list column (normal-step): prefix ASCENDING ---
    if list_mode == "multi":
        # token_split_points are produced ascending by add_token_split_points.py.
        list_pairs = [(int(sp), False) for sp in split_points]
    else:  # custom: one prompt per (pre-sorted) ratio
        list_pairs = [(int(round(r * n_tgt)), r < 1.0) for r in (list_ratios or [])]
    item["summarize_prompts"] = _build_prompt_array(
        system_msg, question, template, tgt_tokens, n_tgt, list_pairs,
    )

    # --- teacher column (online teacher-API): its own template + a SINGLE prefix ---
    if teacher_template is not None:
        sp_teacher = int(round(teacher_ratio * n_tgt))
        item[teacher_col] = _build_prompt_array(
            system_msg, question, teacher_template, tgt_tokens, n_tgt,
            [(sp_teacher, teacher_ratio < 1.0)], is_teacher=True,
        )
    return item


def _worker_process(
    args: Tuple[Dict[str, Any], str, str, float, str, List[float] | None,
                str | None, float, str, bool],
) -> Dict[str, Any]:
    (item, template, single_mode, single_ratio, list_mode, list_ratios,
     teacher_template, teacher_ratio, teacher_col, teacher_only) = args
    return process_single_item(
        item, template, single_mode, single_ratio, list_mode, list_ratios,
        teacher_template, teacher_ratio, teacher_col, teacher_only,
    )


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
             "or split-point token counts will misalign with text offsets. Set to empty "
             "'' or 'none' (or pass --no-tokenizer) to run WITHOUT a tokenizer -- only "
             "valid with --teacher-only, where the prefix is cut in CHARACTER space.",
    )
    parser.add_argument(
        "--no-tokenizer",
        dest="no_tokenizer",
        action="store_true",
        default=False,
        help="Run without loading any tokenizer (teacher-only): the --teacher-ratio prefix "
             "is sliced in CHARACTER space instead of token space. Avoids the transformers "
             "dependency / model download. Ignored unless --teacher-only.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Worker process count (1 = sync).")
    # ---- single column (summarize_prompt, for extra-step / recycle) ----
    parser.add_argument(
        "--single-mode",
        choices=["full", "progressive", "random"],
        default="full",
        help="How to build the SINGLE prompt column 'summarize_prompt' (extra-step). "
             "full: prefix = a fixed --full-ratio fraction of the answer-truncated "
             "reasoning (same for every row). progressive: the fraction DECREASES "
             "per training step -- row i uses step i // --batch-size, ratio linearly "
             "from --ratio-start down to --ratio-end across the dataset (assumes the "
             "trainer reads rows in order with shuffle off). random: each row draws "
             "its ratio uniformly from [--ratio-min, --ratio-max] (seeded by --seed).",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.4,
        help="Only used in --single-mode random. Lower bound (inclusive) of the "
             "uniform prefix-ratio range. Must satisfy 0 < ratio-min <= ratio-max <= 1.",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=0.8,
        help="Only used in --single-mode random. Upper bound (inclusive) of the "
             "uniform prefix-ratio range.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Only used in --single-mode random. RNG seed for reproducible per-row "
             "ratio sampling.",
    )
    parser.add_argument(
        "--full-ratio",
        type=float,
        default=1.0,
        help="Only used in --single-mode full. Fraction (0, 1] of the answer-truncated "
             "reasoning (in token space) to use as the single [Reasoning Draft]. "
             "1.0 (default) = the complete reasoning; e.g. 0.5 = the first ~50%% "
             "of tokens, backed up to the nearest sentence boundary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Only used in --single-mode progressive. Training batch size, so rows "
             "can be grouped into steps (row i -> step i // batch_size).",
    )
    parser.add_argument(
        "--ratio-start",
        type=float,
        default=1.0,
        help="Only used in --single-mode progressive. Prefix ratio at the FIRST step "
             "(default 1.0 = longest draft).",
    )
    parser.add_argument(
        "--ratio-end",
        type=float,
        default=0.0,
        help="Only used in --single-mode progressive. Prefix ratio at the LAST step "
             "(default 0.0 = no draft, model derives alone).",
    )
    # ---- list column (summarize_prompts, for normal-step summarize_replace) ----
    parser.add_argument(
        "--list-mode",
        choices=["multi", "custom"],
        default="multi",
        help="How to build the LIST prompt column 'summarize_prompts' (normal-step). "
             "multi (default): one prompt per token_split_points entry (already prefix "
             "ASCENDING). custom: one prompt per ratio in --list-ratios (split points "
             "not needed). Both keep prefixes ascending so the trainer's "
             "summarize_replace picks the shortest correct candidate first.",
    )
    parser.add_argument(
        "--list-ratios",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
        help="Only used in --list-mode custom. Comma-separated fractions in (0, 1] "
             "for the list prompts, e.g. '0.2,0.4,0.6,0.8'. Sorted ascending before "
             "use so summarize_replace scans shortest prefix first.",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="User-turn template, must include {question} and {prefix} placeholders.",
    )
    # ---- teacher column (teacher_prompts, for the online teacher-API rephraser) ----
    parser.add_argument(
        "--teacher-prompts",
        dest="teacher_prompts",
        action="store_true",
        default=False,
        help="Also render a SEPARATE teacher-API prompt column (--teacher-col, "
             "default 'teacher_prompts'), kept apart from summarize_prompts. It is a "
             "SINGLE prompt (length-1) built with --teacher-template and a single "
             "--teacher-ratio prefix; the K summarize-replace candidates reuse it. The "
             "training-time teacher_api.prompt_key must match --teacher-col.",
    )
    parser.add_argument(
        "--teacher-col",
        default="teacher_prompts",
        help="Column name for the teacher-API prompts (must equal teacher_api.prompt_key).",
    )
    parser.add_argument(
        "--teacher-template",
        default=None,
        help="User-turn template for the teacher column, must include {question} and "
             "{prefix}. Defaults to the built-in TEACHER_TEMPLATE_DEFAULT when "
             "--teacher-prompts is set.",
    )
    parser.add_argument(
        "--teacher-ratio",
        type=float,
        default=1.0,
        help="Prefix fraction (0, 1] for the SINGLE teacher prompt. 1.0 (default) = the "
             "full answer-truncated reasoning as the [Reasoning Draft]; e.g. 0.5 = the "
             "first ~50%% of tokens, backed up to a sentence boundary. The teacher column "
             "is length-1; the K summarize-replace candidates reuse this one prompt.",
    )
    parser.add_argument(
        "--teacher-only",
        dest="teacher_only",
        action="store_true",
        default=False,
        help="Render ONLY the teacher column (implies --teacher-prompts); skip the "
             "summarize_prompt / summarize_prompts columns entirely. In this mode "
             "token_split_points is NOT required (the teacher prefix comes from "
             "--teacher-ratio). Use when the input parquet has no split points and you "
             "only need teacher prompts.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick smoke tests.")
    parser.add_argument(
        "--drop-empty",
        dest="drop_empty",
        action="store_true",
        default=True,
        help="Drop rows whose summarize_prompt / summarize_prompts came out empty "
             "(no question / no reasoning / answer leaks in first sentence). ON by "
             "default: such rows crash the failure-recycle np.stack collate in "
             "new_ray_trainer.py when they mix with normal rows. Use --keep-empty "
             "to disable.",
    )
    parser.add_argument(
        "--keep-empty",
        dest="drop_empty",
        action="store_false",
        help="Keep rows with empty summarize columns (disables the default cleanup).",
    )
    args = parser.parse_args()

    if "{question}" not in args.template or "{prefix}" not in args.template:
        raise SystemExit("--template must contain both {question} and {prefix}")

    # --teacher-only implies --teacher-prompts (render only the teacher column).
    if args.teacher_only:
        args.teacher_prompts = True

    # Resolve the effective tokenizer path. No-tokenizer (char-space) mode is triggered
    # by --no-tokenizer or an empty/none --tokenizer-path, and is ONLY valid with
    # --teacher-only (summarize split points are token-space and need a real tokenizer).
    tok_path = args.tokenizer_path
    no_tok = args.no_tokenizer or (not tok_path) or str(tok_path).lower() in ("none", "null", "no_tokenizer")
    if no_tok:
        if not args.teacher_only:
            raise SystemExit(
                "no-tokenizer / empty --tokenizer-path is only supported with --teacher-only "
                "(summarize columns need a real tokenizer for token-space split points)."
            )
        tok_path = ""  # init_worker treats empty as char-space mode

    # teacher-column setup (optional)
    teacher_template = None
    if args.teacher_prompts:
        teacher_template = args.teacher_template or TEACHER_TEMPLATE_DEFAULT
        if "{question}" not in teacher_template or "{prefix}" not in teacher_template:
            raise SystemExit("--teacher-template must contain both {question} and {prefix}")
        if not (0.0 < args.teacher_ratio <= 1.0):
            raise SystemExit("--teacher-ratio must be in the (0, 1] range")

    # single-column validation
    if not (0.0 < args.full_ratio <= 1.0):
        raise SystemExit("--full-ratio must be in the (0, 1] range")
    if args.single_mode == "progressive":
        if args.batch_size <= 0:
            raise SystemExit("--single-mode progressive requires --batch-size > 0")
        if not (0.0 <= args.ratio_end <= args.ratio_start <= 1.0):
            raise SystemExit(
                "--single-mode progressive expects 0 <= --ratio-end <= --ratio-start <= 1"
            )
    if args.single_mode == "random":
        if not (0.0 < args.ratio_min <= args.ratio_max <= 1.0):
            raise SystemExit(
                "--single-mode random expects 0 < --ratio-min <= --ratio-max <= 1"
            )

    # list-column validation: parse + sort ratios ascending
    list_ratios: List[float] = []
    if args.list_mode == "custom":
        try:
            list_ratios = sorted(
                float(x) for x in args.list_ratios.split(",") if x.strip() != ""
            )
        except ValueError:
            raise SystemExit("--list-ratios must be comma-separated floats, e.g. '0.2,0.5,1.0'")
        if not list_ratios:
            raise SystemExit("--list-mode custom requires a non-empty --list-ratios")
        if not all(0.0 < r <= 1.0 for r in list_ratios):
            raise SystemExit("each --list-ratios value must be in the (0, 1] range")

    print(f"Reading {args.input}")
    df = pd.read_parquet(args.input)
    if "token_split_points" not in df.columns:
        # Split points are only read by --list-mode multi. The single column,
        # --list-mode custom, and --teacher-only all derive prefixes from ratios.
        if args.teacher_only or args.list_mode != "multi":
            reason = "teacher-only" if args.teacher_only else f"--list-mode {args.list_mode}"
            print(f"  (no 'token_split_points' column; OK with {reason}, split points are not used)")
        else:
            raise SystemExit(
                "Input parquet has no 'token_split_points' column. "
                "Run Data/add_token_split_points.py first (or use --list-mode custom / --teacher-only)."
            )

    if args.limit is not None:
        df = df.head(args.limit)
    records = df.to_dict(orient="records")
    print(
        f"  -> {len(records):,} rows. Tokenizer: {tok_path or '(none / char-space)'}. "
        f"single-mode: {args.single_mode}, list-mode: {args.list_mode}"
    )
    if args.list_mode == "custom":
        print(f"     list-ratios (sorted): {list_ratios}")

    # Per-row single_ratio. For single-mode full every row shares args.full_ratio;
    # for progressive each row gets a step-decreasing ratio (row i -> step
    # i // batch_size, ratio linear from ratio_start to ratio_end);
    # for random each row draws uniformly from [ratio_min, ratio_max].
    if args.single_mode == "progressive":
        row_ratios = compute_progressive_ratios(
            len(records), args.batch_size, args.ratio_start, args.ratio_end
        )
        n_steps = (len(records) + args.batch_size - 1) // args.batch_size
        print(
            f"     progressive single: {len(records)} rows / batch_size "
            f"{args.batch_size} = {n_steps} steps; ratio {args.ratio_start} -> "
            f"{args.ratio_end}"
        )
    elif args.single_mode == "random":
        rng = np.random.default_rng(args.seed)
        row_ratios = rng.uniform(
            args.ratio_min, args.ratio_max, size=len(records)
        ).tolist()
        print(
            f"     random single: {len(records)} rows; ratio ~ U[{args.ratio_min}, "
            f"{args.ratio_max}] (seed={args.seed})"
        )
    else:
        row_ratios = [args.full_ratio] * len(records)

    workers = max(1, min(args.workers, mp.cpu_count()))
    if workers == 1:
        init_worker(tok_path)
        processed = [
            process_single_item(
                item, args.template, args.single_mode, row_ratios[i],
                args.list_mode, list_ratios,
                teacher_template, args.teacher_ratio, args.teacher_col, args.teacher_only,
            )
            for i, item in enumerate(tqdm(records, total=len(records), desc="rendering"))
        ]
    else:
        task_iter = (
            (item, args.template, args.single_mode, row_ratios[i],
             args.list_mode, list_ratios,
             teacher_template, args.teacher_ratio, args.teacher_col, args.teacher_only)
            for i, item in enumerate(records)
        )
        with mp.Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(tok_path,),
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
    if args.teacher_only:
        emitted_cols = [args.teacher_col]
    else:
        emitted_cols = ["summarize_prompt", "summarize_prompts"]
        if teacher_template is not None:
            emitted_cols.append(args.teacher_col)
    for col in emitted_cols:
        n_emitted = sum(
            1 for p in out_df[col]
            if isinstance(p, np.ndarray) and len(p) > 0
        )
        print(f"  -> {n_emitted:,}/{len(out_df):,} rows have non-empty {col}.")

    # Cleanup: drop rows whose emitted prompt columns came out empty. process_single_item
    # emits np.array([], dtype=object) when a row cannot be rendered (no question,
    # no reasoning, or the answer leaks in the first sentence). These empty (shape
    # (0,)) cells are harmless during normal steps (the dataset pads them) but crash
    # the failure-recycle np.stack collate in new_ray_trainer.py when they land in
    # the failure buffer next to normal rows (mismatched shapes). Removing them here
    # keeps the training data clean at the source. Disable with --keep-empty.
    if args.drop_empty:
        def _is_empty(v) -> bool:
            return not (isinstance(v, np.ndarray) and len(v) > 0)

        def _row_bad(r) -> bool:
            # Only check the columns this run actually emitted.
            return any(_is_empty(r[c]) for c in emitted_cols)

        bad = out_df.apply(_row_bad, axis=1)
        n_drop = int(bad.sum())
        if n_drop > 0:
            bad_pos = [int(i) for i in np.where(bad.to_numpy())[0][:20]]
            print(
                f"  -> dropping {n_drop:,} row(s) with empty prompt columns "
                f"(positions: {bad_pos}{' ...' if n_drop > 20 else ''})."
            )
            out_df = out_df[~bad].reset_index(drop=True)
        else:
            print("  -> no empty prompt rows to drop.")

    # Print a sample of BOTH columns for human-eyeball verification.
    # Print a sample of the emitted columns for human-eyeball verification.
    _seed_col = args.teacher_col if args.teacher_only else "summarize_prompts"
    sample = next(
        (
            row for _, row in out_df.iterrows()
            if isinstance(row[_seed_col], np.ndarray)
            and len(row[_seed_col]) >= 1
        ),
        None,
    )
    if sample is not None:
        sample_points = sample.get("token_split_points")
        sample_cols = list(emitted_cols)
        for col in sample_cols:
            arr = sample[col]
            if not (isinstance(arr, np.ndarray) and len(arr) >= 1):
                continue
            idxs = sorted({0, len(arr) // 2, len(arr) - 1})
            for idx in idxs:
                # 标注每条 prompt 的 prefix 来源，各列情形不同：
                #   - summarize_prompt（单条）：来自 single_ratio，与 split_points 无关。
                #   - summarize_prompts + list-mode multi：一条对一个 token_split_points。
                #   - summarize_prompts + list-mode custom：来自 list_ratios[idx]。
                #   - teacher 列：单条，prefix 来自 teacher_ratio。
                if col == "summarize_prompt":
                    tag = f" (single, mode={args.single_mode})"
                elif col == args.teacher_col and teacher_template is not None:
                    tag = f" (teacher, single, ratio={args.teacher_ratio})"
                elif args.list_mode == "multi":
                    tag = ""
                    if isinstance(sample_points, (list, np.ndarray)) and idx < len(sample_points):
                        tag = f" (split_point={sample_points[idx]})"
                else:  # custom
                    tag = f" (ratio={list_ratios[idx]})" if idx < len(list_ratios) else ""
                print(f"\n--- sample {col}[{idx}]{tag} ---")
                for msg in arr[idx]:
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
