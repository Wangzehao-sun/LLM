import argparse
import multiprocessing as mp
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

worker_tokenizer = None


def init_worker(tokenizer_path: str) -> None:
    global worker_tokenizer
    worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def process_think_process(
    think_process: str,
    tokenizer,
    num_stages: int = 8,
    curve_power: float = 1.5,
) -> List[int]:
    """Compute token split points from a thought process string."""
    if not think_process:
        return [0] * num_stages

    segments = [s for s in think_process.split("\n\n") if s]
    if not segments:
        return [0] * num_stages

    cum_lens = []
    curr = 0
    for s in segments:
        curr += len(s) + 2
        cum_lens.append(curr)

    total_len = curr
    linear_steps = np.linspace(0, 1, num_stages+1)[1:]
    ratios = np.power(linear_steps, curve_power)

    split_counts: List[int] = []
    full_text = "\n\n".join(segments)

    try:
        encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        token_cnt = len(encoding.input_ids)
        offset_ends = np.array([end for _, end in encoding.offset_mapping])

        for r in ratios:
            if r == 0:
                split_counts.append(0)
                continue

            target = r * total_len
            found_idx = len(segments) - 1
            for i, cl in enumerate(cum_lens):
                if cl >= target:
                    found_idx = i
                    break

            cutoff_char_len = cum_lens[found_idx]
            if found_idx == len(segments) - 1:
                split_counts.append(token_cnt)
            else:
                idx = np.searchsorted(offset_ends, cutoff_char_len)
                if idx > token_cnt:
                    idx = token_cnt
                split_counts.append(int(idx))
    except Exception:
        # Fallback path when offset mapping is unavailable or fails.
        for r in ratios:
            if r == 0:
                split_counts.append(0)
                continue

            target = r * total_len
            found_idx = len(segments) - 1
            for i, cl in enumerate(cum_lens):
                if cl >= target:
                    found_idx = i
                    break

            partial_text = "\n\n".join(segments[: found_idx + 1])
            ids = tokenizer.encode(partial_text, add_special_tokens=False)
            split_counts.append(len(ids))

    return split_counts


def extract_think_process(item: Dict[str, Any]) -> str:
    """Extract think process from a row with different possible target formats."""
    target_val = item.get("target", "")
    text = ""

    if isinstance(target_val, (list, np.ndarray)) and target_val:
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


def process_single_item(item: Dict[str, Any], num_stages: int, curve_power: float) -> Dict[str, Any]:
    global worker_tokenizer
    think_process = extract_think_process(item)

    if worker_tokenizer is None:
        item["token_split_points"] = [0] * num_stages
        return item

    item["token_split_points"] = process_think_process(
        think_process,
        worker_tokenizer,
        num_stages=num_stages,
        curve_power=curve_power,
    )
    return item


def _worker_process(args: Tuple[Dict[str, Any], int, float]) -> Dict[str, Any]:
    item, num_stages, curve_power = args
    return process_single_item(item, num_stages, curve_power)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add token_split_points to parquet rows.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--tokenizer-path", default="/home/shared/Qwen2.5-Math-7B-16k-think", help="Tokenizer path")
    parser.add_argument("--num-stages", type=int, default=8, help="Number of split stages")
    parser.add_argument(
        "--curve-power",
        type=float,
        default=1.0,
        help="Curve exponent for split ratios, e.g. 1.0 for linear, >1 to bias later splits",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    records = df.to_dict(orient="records")

    workers = max(1, min(args.workers, mp.cpu_count()))
    task_iter = ((item, args.num_stages, args.curve_power) for item in records)

    if workers == 1:
        init_worker(args.tokenizer_path)
        processed = [
            process_single_item(item, args.num_stages, args.curve_power)
            for item in tqdm(records, total=len(records))
        ]
    else:
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(args.tokenizer_path,)) as pool:
            chunksize = max(1, len(records) // (workers * 4)) if records else 1
            processed = list(tqdm(pool.imap(_worker_process, task_iter, chunksize=chunksize), total=len(records)))

    out_df = pd.DataFrame(processed)
    #打印一条数据
    print(out_df.iloc[0])
    out_df.to_parquet(args.output, index=False)

    print(f"Done. Wrote {len(out_df)} rows to: {args.output}")


if __name__ == "__main__":
    main()
