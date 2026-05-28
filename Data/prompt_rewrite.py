import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd

# ==========================================
# Prompt Definitions
# ==========================================

SYSTEM_PROMPTS = {
    "system_instruction": "You are a helpful assistant.",
}

USER_PROMPTS = {
    "rewrite_problem_only": "{problem}",
    "qa_style": "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}",
}


# ==========================================
# Arguments and Main Logic
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite prompt field in openr1 parquet data."
    )

    default_input = "/home/zhwang/LLM/Train/data_0306/openr1.parquet"
    default_output = "/home/zhwang/LLM/Train/data_0306/openr1_prompt_rewritten_nothinking.parquet"

    parser.add_argument(
        "--input_file",
        type=str,
        default=default_input,
        help="Path to input parquet file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=default_output,
        help="Path to save output parquet file.",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="rewrite_problem_only",
        choices=USER_PROMPTS.keys(),
        help="User prompt template key.",
    )
    parser.add_argument(
        "--system_key",
        type=str,
        default="system_instruction",
        choices=SYSTEM_PROMPTS.keys(),
        help="System prompt key.",
    )
    parser.add_argument(
        "--backup_old_prompt",
        action="store_true",
        help="Store original prompt in field old_prompt.",
    )
    parser.add_argument(
        "--backup_old_target",
        action="store_true",
        help="Store original target in field old_target.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="think_only",
        choices=["think_only", "full", "solution_only"],
        help=(
            "How to rewrite target[0].content:\n"
            "  think_only   - keep only content INSIDE <think>...</think> (default)\n"
            "  full         - keep target unchanged\n"
            "  solution_only- keep only content OUTSIDE <think>...</think> "
            "(everything after </think>)"
        ),
    )
    parser.add_argument(
        "--thinking_mode",
        action="store_true",
        help="DEPRECATED alias for --target_mode full. Kept for backward compatibility.",
    )
    args = parser.parse_args()

    # Backward-compat: --thinking_mode overrides --target_mode to "full".
    if args.thinking_mode:
        args.target_mode = "full"

    return args


def _extract_problem_from_prompt(prompt):
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()

    if not isinstance(prompt, list) or len(prompt) < 2:
        return None

    try:
        return prompt[1]["content"]
    except (TypeError, KeyError, IndexError):
        return None


def _split_target_content(content: str):
    """Split a target content string into (think, solution).

    The expected format is ``<think>{think}</think>\n\n{solution}`` (with the
    opening tag optional). When ``</think>`` is missing we fall back to
    treating the whole string as the think portion with an empty solution,
    which keeps callers safe regardless of input shape.
    """
    if not isinstance(content, str):
        return "", ""

    if "</think>" in content:
        parts = content.split("</think>", 1)
        think = parts[0]
        solution = parts[1] if len(parts) > 1 else ""
    else:
        think = content
        solution = ""

    if "<think>" in think:
        think = think.replace("<think>", "")

    return think.strip(), solution.strip()


def _rewrite_target_content(target, mode: str):
    """Apply the requested ``target_mode`` transformation to target[0].content.

    ``mode`` is one of:
      * ``full``          - return target unchanged
      * ``think_only``    - keep only the reasoning between <think>...</think>
      * ``solution_only`` - keep only the solution after </think>
    """
    if mode == "full":
        return target

    if isinstance(target, np.ndarray):
        target = target.tolist()

    if not isinstance(target, list) or len(target) < 1:
        return target

    first_item = target[0]
    if not isinstance(first_item, dict):
        return target

    content = first_item.get("content")
    if not isinstance(content, str):
        return target

    think, solution = _split_target_content(content)

    if mode == "think_only":
        new_content = think
    elif mode == "solution_only":
        new_content = solution
    else:
        raise ValueError(f"Unknown target_mode: {mode!r}")

    new_target = deepcopy(target)
    new_target[0]["content"] = new_content
    return np.array(new_target)


def process_data(
    dataset,
    usr_prompt,
    system_prompt,
    backup_old_prompt=False,
    backup_old_target=False,
    target_mode="think_only",
):
    processed = []

    for item in dataset:
        prompt = item.get("prompt")
        problem = _extract_problem_from_prompt(prompt)

        if not problem:
            continue

        try:
            new_user_content = usr_prompt.format(problem=problem, question=problem)
        except (KeyError, IndexError, ValueError):
            continue

        # Keep role names consistent with existing data format.
        new_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": new_user_content},
        ]

        if backup_old_prompt:
            item["old_prompt"] = deepcopy(item.get("prompt"))

        if backup_old_target:
            item["old_target"] = deepcopy(item.get("target"))

        item["prompt"] = np.array(new_prompt)
        item["target"] = _rewrite_target_content(item.get("target"), target_mode)

        processed.append(item)

    return processed


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    print(f"Loading data from: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    dataset = df.to_dict(orient="records")

    selected_usr_prompt = USER_PROMPTS[args.prompt_key]
    selected_system_prompt = SYSTEM_PROMPTS[args.system_key]

    print(f"Using prompt key: {args.prompt_key}")
    print(f"Using system key: {args.system_key}")
    print(f"Target mode: {args.target_mode}")
    print(f"Processing {len(dataset)} items...")

    processed_data = process_data(
        dataset,
        selected_usr_prompt,
        selected_system_prompt,
        backup_old_prompt=args.backup_old_prompt,
        backup_old_target=args.backup_old_target,
        target_mode=args.target_mode,
    )
    #打印一条数据样例
    
    if processed_data:
        print("Sample processed item:")
        print(processed_data[0])
        print(f"Processed {len(processed_data)} items.")
    else:
        print("No items were processed.")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving output to: {args.output_file}")
    out_df = pd.DataFrame(processed_data)
    out_df.to_parquet(args.output_file)
    print("Done.")


if __name__ == "__main__":
    main()
