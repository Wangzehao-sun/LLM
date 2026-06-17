"""Convert a deepmath/openr1-style parquet (columns ``prompt`` = [system, user]
and ``target`` = [assistant]) into an SFT parquet with a single ``messages``
column that MultiTurnSFTDataset consumes.

MultiTurnSFTDataset masks loss to assistant tokens only and keeps the system
prompt, which is what we want for summarize/reasoning SFT. It expects each row
to be one conversation: [system, user, assistant, ...].

Usage:
    python prepare_sft.py \
        --in  ../data/deepmath_dgt6_n10000.parquet \
        --out ../data/deepmath_dgt6_n10000_sft.parquet \
        --limit 2000          # 可选：只保存前 2000 条
"""
import argparse

import pandas as pd


def _to_message_list(x):
    # parquet stores message arrays as numpy object arrays; normalise to list[dict]
    return [dict(m) for m in list(x)]


def build_messages(row, prompt_key, target_key):
    prompt_msgs = _to_message_list(row[prompt_key])
    target_msgs = _to_message_list(row[target_key])
    return prompt_msgs + target_msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--target-key", default="target")
    ap.add_argument("--messages-key", default="messages")
    ap.add_argument("--limit", type=int, default=None,
                    help="只保存前 N 条；默认 None 表示全部保存。")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    print(f"loaded {len(df)} rows from {args.in_path}; columns={list(df.columns)}")

    if args.limit is not None:
        df = df.head(args.limit)
        print(f"limiting to first {len(df)} rows (--limit={args.limit})")

    df[args.messages_key] = df.apply(
        lambda r: build_messages(r, args.prompt_key, args.target_key), axis=1
    )

    # quick sanity check on the first row
    sample = df.iloc[0][args.messages_key]
    roles = [m["role"] for m in sample]
    print(f"sample messages roles: {roles}")
    assert roles and roles[-1] == "assistant", (
        "last message must be assistant for SFT loss masking; got "
        f"{roles}"
    )

    df.to_parquet(args.out_path)
    print(f"wrote {len(df)} rows to {args.out_path} with '{args.messages_key}' column")


if __name__ == "__main__":
    main()
