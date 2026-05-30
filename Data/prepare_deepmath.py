"""Download zwhe99/DeepMath-103K and convert it into the same parquet schema
as ``openr1.parquet``.

Target schema (matches /Users/zenohaoz/LLM/Data/openr1.parquet):
    data_source   : str
    prompt        : list[{role, content}]   # system + user
    target        : list[{role, content}]   # assistant, with <think>...</think>
    ability       : str (empty)
    reward_model  : {ground_truth: str, style: 'rule'}
    extra_info    : {index: -1, split: 'default', topic: str, difficulty: float}

Each DeepMath row carries three reasoning paths (r1_solution_1/2/3); we pick
ONE at random per row to keep the row count roughly equal to the number of
unique problems (~103K), matching the per-row=single-target convention of
openr1.parquet.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("/Users/zenohaoz/LLM/Data/deepmath.parquet")
LOCAL_PATH = Path.home() / "LLM" / "DeepMath-103K"  # if exists, load from here
DATASET_NAME = "zwhe99/DeepMath-103K"
SPLIT = "train"
SEED = 42

# Same system prompt used in openr1.parquet so downstream training code can
# treat both files interchangeably.
SYSTEM_PROMPT = (
    "Your task is to follow a systematic, thorough reasoning process before "
    "providing the final solution. This involves analyzing, summarizing, "
    "exploring, reassessing, and refining your thought process through "
    "multiple iterations. Structure your response into two sections: Thought "
    "and Solution. In the Thought section, present your reasoning using the "
    'format: "<think>\n {thoughts} </think>\n". Each thought should include '
    "detailed analysis, brainstorming, verification, and refinement of "
    'ideas. After "</think>\n," in the Solution section, provide the final, '
    "logical, and accurate answer, clearly derived from the exploration in "
    "the Thought section. If applicable, include the answer in \\boxed{} for "
    "closed-form results like multiple choices or mathematical solutions."
)

SOLUTION_KEYS = ("r1_solution_1", "r1_solution_2", "r1_solution_3")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def normalize_solution(raw: str) -> str:
    """Ensure the assistant content is in ``<think>...</think>\n\n{answer}`` form.

    DeepSeek-R1 outputs in DeepMath-103K may or may not retain the literal
    ``<think>`` markers (some preprocessing strips them). We handle both:

    * If both opening and closing tags are present -> use as-is (stripped).
    * If only the closing tag is present -> prepend ``<think>\n``.
    * If neither tag is present -> wrap the entire string with
      ``<think>...</think>\n\n`` so the format still matches openr1.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    has_open = bool(_THINK_OPEN_RE.search(text))
    has_close = bool(_THINK_CLOSE_RE.search(text))

    if has_open and has_close:
        return text
    if has_close and not has_open:
        return "<think>\n" + text
    # No tags at all -> the string is purely the reasoning trace; we have no
    # separate "final solution" portion, so we leave the body inside <think>
    # and emit an empty solution section. Downstream code that strips the
    # think block will still get a usable example because the final answer
    # also lives in `reward_model.ground_truth`.
    return f"<think>\n{text}\n</think>\n\n"


def pick_solution(row: dict, rng: random.Random) -> str:
    """Pick one non-empty r1_solution_X uniformly at random."""
    candidates = [row[k] for k in SOLUTION_KEYS if row.get(k)]
    if not candidates:
        # Fall back to any of the three slots even if empty so we never crash;
        # normalize_solution will turn None/"" into "".
        return row.get(SOLUTION_KEYS[0], "") or ""
    return rng.choice(candidates)


def convert_row(row: dict, rng: random.Random) -> dict:
    chosen = pick_solution(row, rng)
    assistant_content = normalize_solution(chosen)

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(row.get("question", ""))},
    ]
    target = [
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "data_source": "DeepMath-103K",
        "prompt": prompt,
        "target": target,
        "ability": "",
        "reward_model": {
            "ground_truth": str(row.get("final_answer", "")),
            "style": "rule",
        },
        "extra_info": {
            "index": -1,
            "split": "default",
            "topic": str(row.get("topic", "")),
            "difficulty": float(row["difficulty"])
            if row.get("difficulty") is not None
            else None,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_files = sorted((LOCAL_PATH / "data").glob("*.parquet")) if (LOCAL_PATH / "data").is_dir() \
        else sorted(LOCAL_PATH.glob("*.parquet")) if LOCAL_PATH.is_dir() else []
    if local_files:
        print(f"Loading {len(local_files)} local parquet file(s) from {LOCAL_PATH}")
        dataset = load_dataset("parquet", data_files=[str(f) for f in local_files], split="train")
    else:
        print(f"Loading {DATASET_NAME} (split={SPLIT}) from Hugging Face Hub...")
        dataset = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"  -> {len(dataset):,} rows; columns: {dataset.column_names}")

    rng = random.Random(SEED)
    rows: list[dict] = []
    for row in tqdm(dataset, desc="Converting", unit="ex"):
        rows.append(convert_row(row, rng))

    df = pd.DataFrame(rows)
    print("Output dataframe:")
    print(f"  shape   : {df.shape}")
    print(f"  columns : {df.columns.tolist()}")
    print(f"  dtypes  :\n{df.dtypes}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
