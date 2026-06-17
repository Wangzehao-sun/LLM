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


def _normalize_msg(m):
    # one message -> plain dict, tolerant of numpy structured records.
    if isinstance(m, dict):
        return dict(m)
    if hasattr(m, "dtype") and getattr(m.dtype, "names", None):
        return {k: m[k] for k in m.dtype.names}
    return dict(m)


def _to_message_list(x, str_role=None):
    # 列里可能是：单个 message dict（{'role','content'}）、一组 message
    # （ndarray/list of dict），或一段纯文本字符串（如 output 列的答案）。
    # 统一成 list[dict]。注意：单个 dict 不能直接 list(x)，那样得到的是它的 keys。
    if isinstance(x, str):
        if str_role is None:
            raise ValueError(
                "got a plain string but no str_role to wrap it; "
                "pass str_role (e.g. 'assistant') for text-only columns"
            )
        return [{"role": str_role, "content": x}]
    if isinstance(x, dict):
        return [_normalize_msg(x)]
    return [_normalize_msg(m) for m in list(x)]


def build_messages(row, prompt_key, target_key):
    # prompt 通常是 [system, user] 消息数组；target 既可能是消息(数组/单 dict)，
    # 也可能是纯文本答案(如 output 列)——后者按 assistant 角色包装。
    prompt_msgs = _to_message_list(row[prompt_key], str_role="user")
    target_msgs = _to_message_list(row[target_key], str_role="assistant")
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
