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
        "--thinking_mode",
        action="store_true",
        help="If set, keep target unchanged. If not set, keep only content inside <think>...</think> in target.",
    )
    return parser.parse_args()


def _extract_problem_from_prompt(prompt):
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()

    if not isinstance(prompt, list) or len(prompt) < 2:
        return None

    try:
        return prompt[1]["content"]
    except (TypeError, KeyError, IndexError):
        return None


def _target_keep_think_only(target):
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
    if "</think>" in content:
            parts = content.split("</think>")
            think_content = parts[0].strip()
            # If standard answer exists (len > 1), use it, otherwise empty string or handle gracefully
            if len(parts) > 1:
                standard_answer = parts[1].strip()
            else:
                standard_answer = ""
                
            if "<think>" in think_content:
                think_content = think_content.replace("<think>", "").strip()
    # start_tag = "<think>"
    # end_tag = "</think>"
    # start = content.find(start_tag)
    # end = content.find(end_tag)

    # if start != -1 and end != -1 and end >= start + len(start_tag):
    #     think_content = content[start + len(start_tag):end].strip()
    # else:
    #     # Fallback for malformed samples: keep original text to avoid data loss.
    #     think_content = content.strip()

    new_target = deepcopy(target)
    new_target[0]["content"] = think_content
    return np.array(new_target)


def process_data(
    dataset,
    usr_prompt,
    system_prompt,
    backup_old_prompt=False,
    backup_old_target=False,
    thinking_mode=False,
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
        if not thinking_mode:
            item["target"] = _target_keep_think_only(item.get("target"))

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
    print(f"Thinking mode: {args.thinking_mode}")
    print(f"Processing {len(dataset)} items...")

    processed_data = process_data(
        dataset,
        selected_usr_prompt,
        selected_system_prompt,
        backup_old_prompt=args.backup_old_prompt,
        backup_old_target=args.backup_old_target,
        thinking_mode=args.thinking_mode,
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
