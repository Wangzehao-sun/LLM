import argparse
import multiprocessing as mp
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

worker_tokenizer = None


def init_worker(tokenizer_path: str) -> None:
    global worker_tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# ---------------------------------------------------------------------------
# Final answer extraction & truncation (prevent answer leakage)
# ---------------------------------------------------------------------------

def _find_boxed_spans(text: str) -> List[Tuple[int, int, str]]:
    r"""Find all \boxed{...} spans with balanced brace matching.

    Returns list of (start, end, content) tuples where:
    - start: index of '\' in '\boxed{'
    - end: index after the closing '}'
    - content: the text inside the braces
    """
    spans = []
    pattern = re.compile(r"\\boxed\{")
    for m in pattern.finditer(text):
        start = m.start()
        depth = 1
        i = m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[m.end():i - 1]
            spans.append((start, i, content))
    return spans


def _truncate_answer_from_think(think_process: str, full_text: str) -> str:
    r"""Remove answer leakage from think_process.

    1. Extract the last \boxed{} content from full_text as the final answer.
    2. In think_process, find the FIRST \boxed{} with matching content.
    3. Truncate at the sentence boundary before that match.

    Returns truncated text, or original if no leakage found.
    """
    # Step 1: get final answer from the last \boxed{} in full text
    all_spans = _find_boxed_spans(full_text)
    if not all_spans:
        return think_process
    final_answer = all_spans[-1][2].strip()
    if not final_answer:
        return think_process

    # Step 2: find first matching \boxed{} in think_process
    think_spans = _find_boxed_spans(think_process)
    match_start = None
    for start, end, content in think_spans:
        if content.strip() == final_answer:
            match_start = start
            break

    if match_start is None:
        return think_process

    # Step 3: truncate at sentence boundary before the match
    search_region = think_process[:match_start]
    # Find the last clean boundary (newline or ". ")
    last_newline = search_region.rfind("\n")
    last_period = max(search_region.rfind(". "), search_region.rfind(".\n"))
    boundary = max(last_newline, last_period)

    if boundary > 0:
        if search_region[boundary] == "\n":
            cut_pos = boundary
        else:
            cut_pos = boundary + 2  # include ". "
    else:
        cut_pos = match_start

    truncated = think_process[:cut_pos].rstrip()
    return truncated if truncated else think_process


# ---------------------------------------------------------------------------
# Segmentation: paragraph or sentence
# ---------------------------------------------------------------------------

# LaTeX/math regions whose internal punctuation must NOT trigger sentence
# boundaries. Listed roughly in the order we want to match them so that
# longer delimiters (``$$``, ``\[``) are preferred over shorter ones (``$``).
_MATH_REGEX = re.compile(
    r"\$\$.*?\$\$"        # $$ ... $$
    r"|\\\[.*?\\\]"       # \[ ... \]
    r"|\\\(.*?\\\)"       # \( ... \)
    r"|\$[^\$\n]+?\$",    # $ ... $   (single-line only, to avoid catastrophic match)
    re.DOTALL,
)

# A sentence ends at one or more of ``.``/``!``/``?`` (optionally followed
# by a closing quote/bracket) and is followed by whitespace.
_SENTENCE_END_RE = re.compile(r"[.!?]+[\)\]\"']?\s+")

# Closing delimiters of a display- or inline-math region. When such a
# delimiter is immediately followed by whitespace + a likely sentence-starter,
# we also treat it as a sentence boundary -- this catches R1 lines like
# ``We compute \[ a = 1. \] Then we conclude.`` where the period is hidden
# inside the math block.
_MATH_CLOSE_END_RE = re.compile(
    r"(?:\$\$|\\\]|\\\)|\$)\s+(?=[A-Z\\$\d\(\[\"'])"
)

# Paragraph break.
_PARAGRAPH_BREAK_RE = re.compile(r"\n\n+")


def _find_paragraph_boundaries(text: str) -> List[int]:
    """Char positions at the start of every paragraph after the first one.

    A boundary is the index immediately AFTER the last newline of a
    ``\\n\\n+`` run, i.e. the offset where the next paragraph begins.
    The end of the text is always included as a final boundary.
    """
    boundaries: List[int] = []
    for m in _PARAGRAPH_BREAK_RE.finditer(text):
        boundaries.append(m.end())
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    return boundaries


def _find_sentence_boundaries(text: str) -> List[int]:
    """Char positions at the start of every sentence after the first one.

    Returns an ascending list of offsets where each sentence's successor
    begins (i.e. the index right after the trailing whitespace following the
    sentence-final punctuation). The end of the text is always included as a
    final boundary so consumers always have something to land on.

    Math regions delimited by ``$...$``, ``$$...$$``, ``\\(...\\)`` and
    ``\\[...\\]`` are masked before scanning so punctuation inside formulas
    does not produce spurious sentence boundaries. Hard ``\\n\\n+`` paragraph
    breaks are always treated as sentence boundaries as well, even when no
    terminating punctuation is present.
    """
    if not text:
        return []

    # Mask sentence-ending punctuation that lives INSIDE a math region.
    masked_chars = list(text)
    for m in _MATH_REGEX.finditer(text):
        for i in range(m.start(), m.end()):
            if masked_chars[i] in ".!?":
                masked_chars[i] = "_"
    masked = "".join(masked_chars)

    seen: set = set()
    boundaries: List[int] = []

    # Sentence-final punctuation followed by whitespace.
    for m in _SENTENCE_END_RE.finditer(masked):
        b = m.end()
        if b not in seen:
            seen.add(b)
            boundaries.append(b)

    # Closing math delimiter followed by whitespace + likely sentence-starter.
    # Run on the ORIGINAL text so the delimiter chars are still visible.
    for m in _MATH_CLOSE_END_RE.finditer(text):
        b = m.end()
        if b not in seen:
            seen.add(b)
            boundaries.append(b)

    # Paragraph breaks count as boundaries too (catches lists / displayed
    # equations that don't end with ``.``).
    for m in _PARAGRAPH_BREAK_RE.finditer(text):
        b = m.end()
        if b not in seen:
            seen.add(b)
            boundaries.append(b)

    boundaries.sort()

    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))

    return boundaries


def _segment_text(text: str, split_unit: str) -> List[int]:
    if split_unit == "sentence":
        return _find_sentence_boundaries(text)
    if split_unit == "paragraph":
        return _find_paragraph_boundaries(text)
    raise ValueError(f"Unknown split_unit: {split_unit!r}")


# ---------------------------------------------------------------------------
# Token-level split-point computation
# ---------------------------------------------------------------------------

def process_think_process(
    think_process: str,
    tokenizer,
    num_stages: int = 8,
    curve_power: float = 1.5,
    split_unit: str = "sentence",
) -> List[int]:
    """Compute ``num_stages`` token split points from a thought process.

    The thought process is segmented into sentences (default) or paragraphs.
    For each ratio ``r`` in ``linspace(0, 1, num_stages+1)[1:] ** curve_power``
    we find the first segment boundary whose char offset is at least
    ``r * len(text)`` and report the corresponding token count.
    """
    if not think_process:
        return [0] * num_stages

    boundaries = _segment_text(think_process, split_unit)
    if not boundaries:
        return [0] * num_stages

    total_len = len(think_process)
    linear_steps = np.linspace(0, 1, num_stages + 1)[1:]
    ratios = np.power(linear_steps, curve_power)

    split_counts: List[int] = []

    try:
        encoding = tokenizer(
            think_process, return_offsets_mapping=True, add_special_tokens=False
        )
        token_cnt = len(encoding.input_ids)
        offset_ends = np.array([end for _, end in encoding.offset_mapping])

        for r in ratios:
            if r == 0:
                split_counts.append(0)
                continue

            target = r * total_len
            cutoff_char = boundaries[-1]
            for b in boundaries:
                if b >= target:
                    cutoff_char = b
                    break

            if cutoff_char >= len(think_process):
                split_counts.append(token_cnt)
            else:
                idx = int(np.searchsorted(offset_ends, cutoff_char))
                if idx > token_cnt:
                    idx = token_cnt
                split_counts.append(idx)
    except Exception:
        # Fallback path when offset mapping is unavailable or fails.
        for r in ratios:
            if r == 0:
                split_counts.append(0)
                continue

            target = r * total_len
            cutoff_char = boundaries[-1]
            for b in boundaries:
                if b >= target:
                    cutoff_char = b
                    break
            partial_text = think_process[:cutoff_char]
            ids = tokenizer.encode(partial_text, add_special_tokens=False)
            split_counts.append(len(ids))

    return split_counts


def extract_think_process(item: Dict[str, Any]) -> Tuple[str, str]:
    """Extract think process from a row with different possible target formats.

    Returns (think_process, full_text) tuple. full_text is needed for
    final answer extraction.
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

    full_text = text
    think_process = text.split("</think>")[0].strip()
    if "<think>" in think_process:
        think_process = think_process.replace("<think>", "").strip()
    return think_process, full_text


def process_single_item(
    item: Dict[str, Any],
    num_stages: int,
    curve_power: float,
    split_unit: str,
) -> Dict[str, Any]:
    global worker_tokenizer
    think_process, full_text = extract_think_process(item)

    if worker_tokenizer is None:
        item["token_split_points"] = [0] * num_stages
        return item

    # Truncate think_process before any sentence that leaks the final answer
    think_process_safe = _truncate_answer_from_think(think_process, full_text)

    item["token_split_points"] = process_think_process(
        think_process_safe,
        worker_tokenizer,
        num_stages=num_stages,
        curve_power=curve_power,
        split_unit=split_unit,
    )
    return item


def _worker_process(
    args: Tuple[Dict[str, Any], int, float, str],
) -> Dict[str, Any]:
    item, num_stages, curve_power, split_unit = args
    return process_single_item(item, num_stages, curve_power, split_unit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add token_split_points to parquet rows.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument(
        "--tokenizer-path",
        default="/home/shared/Qwen2.5-Math-7B-16k-think",
        help="Tokenizer path",
    )
    parser.add_argument("--num-stages", type=int, default=8, help="Number of split stages")
    parser.add_argument(
        "--curve-power",
        type=float,
        default=1.0,
        help="Curve exponent for split ratios, e.g. 1.0 for linear, >1 to bias later splits",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument(
        "--split-unit",
        choices=["sentence", "paragraph"],
        default="sentence",
        help=(
            "Segmentation unit used to derive split-point boundaries. "
            "'sentence' (default) splits on .!? + whitespace while protecting "
            "LaTeX math regions; 'paragraph' uses the legacy \\n\\n behaviour."
        ),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    records = df.to_dict(orient="records")

    workers = max(1, min(args.workers, mp.cpu_count()))
    task_iter = (
        (item, args.num_stages, args.curve_power, args.split_unit) for item in records
    )

    if workers == 1:
        init_worker(args.tokenizer_path)
        processed = [
            process_single_item(item, args.num_stages, args.curve_power, args.split_unit)
            for item in tqdm(records, total=len(records))
        ]
    else:
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
                )
            )

    out_df = pd.DataFrame(processed)
    # 打印一条数据
    print(out_df.iloc[0])
    out_df.to_parquet(args.output, index=False)

    print(f"Done. Wrote {len(out_df)} rows to: {args.output}")


if __name__ == "__main__":
    main()
