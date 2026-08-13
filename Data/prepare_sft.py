"""Convert a deepmath/openr1-style parquet (columns ``prompt`` = [system, user]
and ``target`` = [assistant]) into an SFT parquet with a single ``messages``
column that MultiTurnSFTDataset consumes.

MultiTurnSFTDataset masks loss to assistant tokens only and keeps the system
prompt, which is what we want for summarize/reasoning SFT. It expects each row
to be one conversation: [system, user, assistant, ...].

``--prompt-key`` / ``--target-key`` accept any of the shapes these parquets use:
a flat ``[system, user]`` messages array (``prompt``), a length-1 array wrapping
one messages list (``summarize_prompt``, ``teacher_prompts``), a single message
dict, or plain text (``output`` / ``rollout``, wrapped as the assistant turn).
Every row is validated before writing, so a wrong ``--prompt-key`` fails here
rather than an hour later inside the training job.

Usage:
    python prepare_sft.py \
        --in  ../data/deepmath_dgt6_n10000.parquet \
        --out ../data/deepmath_dgt6_n10000_sft.parquet \
        --limit 2000          # 可选：只保存前 2000 条

    # rephraser SFT: rendered prompt + the student's own correct rollout
    python prepare_sft.py \
        --in  ../Data/rollouts_summarize.parquet \
        --out ../Data/rephraser_sft.parquet \
        --prompt-key summarize_prompt --target-key rollout
"""
import argparse

import pandas as pd

VALID_ROLES = ("system", "user", "assistant", "tool")


def _normalize_msg(m):
    """One message -> plain dict, tolerant of numpy structured records.

    Anything that is not recognisably a single message is REJECTED rather than
    coerced. ``dict()`` used to be the fallback here, which silently destroyed
    data: handed a list of messages, ``dict([{"role": "system", ...},
    {"role": "user", ...}])`` reads the two dicts as key/value pairs and returns
    ``{"role": "content"}`` -- one bogus message where two real ones were, with no
    error. That reached training as ``{"role": "content", "content": None}`` and
    surfaced only as an opaque "Unknown role: content" from the dataset loader,
    after the prompt had already been thrown away.
    """
    if isinstance(m, dict):
        return dict(m)
    if hasattr(m, "dtype") and getattr(m.dtype, "names", None):
        return {k: m[k] for k in m.dtype.names}
    if isinstance(m, (list, tuple)) or hasattr(m, "tolist"):
        raise ValueError(
            f"expected one message dict, got a sequence of {len(m)} item(s). The column "
            f"probably wraps its messages in an extra array -- pass the inner list, or "
            f"let _to_message_list unwrap it."
        )
    raise ValueError(f"cannot read a message from {type(m).__name__}: {str(m)[:120]}")


def _looks_like_message(x) -> bool:
    """True if ``x`` is a single ``{role, content}`` message rather than a list."""
    if isinstance(x, dict):
        return "role" in x
    return bool(hasattr(x, "dtype") and getattr(x.dtype, "names", None) and "role" in x.dtype.names)


def _validate_messages(messages, key: str) -> None:
    """Reject malformed messages at build time, not at training time.

    The dataset loader's error for a bad role names neither the column nor the row,
    so an hour of GPU startup can be spent before the real cause is visible.
    """
    for i, m in enumerate(messages):
        where = f"{key}[{i}]"
        if "role" not in m or "content" not in m:
            raise ValueError(f"{where} is missing 'role'/'content': {str(m)[:120]}")
        if m["role"] not in VALID_ROLES:
            raise ValueError(
                f"{where} has role {m['role']!r}, expected one of {VALID_ROLES}. "
                f"Role 'content' in particular means two messages were collapsed into one."
            )
        if not isinstance(m["content"], str) or not m["content"].strip():
            raise ValueError(f"{where} (role={m['role']}) has empty content")


def _to_message_list(x, str_role=None):
    # 列里可能是：单个 message dict（{'role','content'}）、一组 message
    # （ndarray/list of dict）、一段纯文本字符串（如 output 列的答案），或者
    # 「长度-1 数组里再套一份 messages」（summarize_prompt / teacher_prompts 就是
    # 这个形状）。统一成 list[dict]。
    # 注意：单个 dict 不能直接 list(x)，那样得到的是它的 keys。
    if isinstance(x, str):
        if str_role is None:
            raise ValueError(
                "got a plain string but no str_role to wrap it; "
                "pass str_role (e.g. 'assistant') for text-only columns"
            )
        return [{"role": str_role, "content": x}]
    if _looks_like_message(x):
        return [_normalize_msg(x)]
    items = list(x)
    if not items:
        raise ValueError("got an empty messages column")
    # Unwrap the extra nesting level: a length-1 array holding the real messages.
    # Doing it here rather than making every caller pre-flatten is what keeps
    # --prompt-key summarize_prompt from silently producing garbage.
    if len(items) == 1 and not _looks_like_message(items[0]):
        items = list(items[0])
    return [_normalize_msg(m) for m in items]


def build_messages(row, prompt_key, target_key):
    # prompt 通常是 [system, user] 消息数组；target 既可能是消息(数组/单 dict)，
    # 也可能是纯文本答案(如 output 列)——后者按 assistant 角色包装。
    prompt_msgs = _to_message_list(row[prompt_key], str_role="user")
    target_msgs = _to_message_list(row[target_key], str_role="assistant")
    _validate_messages(prompt_msgs, prompt_key)
    _validate_messages(target_msgs, target_key)
    return prompt_msgs + target_msgs


def report_lengths(messages_list, tokenizer_path, max_length, prompt_length):
    """Report prompt / response / total lengths and how many rows exceed the caps.

    Two separate caps bite at two different stages, and both fail SILENTLY:

    * ``data.max_length`` in training (16384 by default). With
      ``data.truncation=right`` a longer row is cut from the right, which removes
      the tail of the assistant turn -- including the ``<|im_end|>`` the model is
      supposed to learn as its stop signal.
    * ``rollout.prompt_length`` at evaluation (4096 by default). main_generation
      calls apply_chat_template with truncation=True, so a longer prompt loses its
      tail -- the end of the reasoning draft.

    Token counts need the real tokenizer; without ``--tokenizer-path`` this falls
    back to a chars/3.2 estimate, which is rough but enough to see whether a cap is
    anywhere near.
    """
    encode = None
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            encode = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
            print(f"[length] tokenizer: {tokenizer_path}")
        except Exception as error:
            print(f"[length] could not load {tokenizer_path} ({error}); falling back to chars/3.2")
    if encode is None:
        encode = lambda text: int(len(text) / 3.2)  # noqa: E731
        print("[length] no tokenizer: counts are chars/3.2 ESTIMATES, not exact")

    prompt_lens, response_lens = [], []
    for messages in messages_list:
        prompt_chars = "".join(m["content"] for m in messages if m["role"] != "assistant")
        response_chars = "".join(m["content"] for m in messages if m["role"] == "assistant")
        prompt_lens.append(encode(prompt_chars))
        response_lens.append(encode(response_chars))

    prompts = pd.Series(prompt_lens)
    responses = pd.Series(response_lens)
    totals = prompts + responses

    def describe(name, series):
        print(
            f"[length] {name:8s} p50={int(series.median()):>6,}  p90={int(series.quantile(0.9)):>6,}  "
            f"p99={int(series.quantile(0.99)):>6,}  max={int(series.max()):>6,}"
        )

    describe("prompt", prompts)
    describe("response", responses)
    describe("total", totals)

    # The chat template adds per-message overhead (role markers, <|im_start|> etc.)
    # that these raw content counts miss, so a row close to a cap may already cross
    # it. Flag that instead of reporting a clean pass.
    over_total = int((totals > max_length).sum())
    near_total = int((totals > max_length * 0.95).sum()) - over_total
    print(
        f"[length] vs training data.max_length={max_length:,}: "
        f"{over_total} row(s) over"
        + (f", {near_total} within 5%" if near_total else "")
    )
    if over_total:
        print(
            "         -> those rows get right-truncated, cutting the assistant tail "
            "(and its stop token) out of the loss. Raise MAX_LENGTH or drop them."
        )

    over_prompt = int((prompts > prompt_length).sum())
    near_prompt = int((prompts > prompt_length * 0.95).sum()) - over_prompt
    print(
        f"[length] vs eval rollout.prompt_length={prompt_length:,}: "
        f"{over_prompt} prompt(s) over"
        + (f", {near_prompt} within 5%" if near_prompt else "")
    )
    if over_prompt:
        print(
            "         -> those prompts get truncated at evaluation, losing the end of "
            "the reasoning draft. Raise PROMPT_LENGTH in sweep_sft_checkpoints.sh."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--target-key", default="target")
    ap.add_argument("--messages-key", default="messages")
    ap.add_argument("--limit", type=int, default=None,
                    help="只保存前 N 条；默认 None 表示全部保存。")
    ap.add_argument("--tokenizer-path", default=None,
                    help="Tokenizer for exact token counts (needs transformers). "
                         "Without it, lengths are chars/3.2 estimates.")
    ap.add_argument("--max-length", type=int, default=16384,
                    help="Training data.max_length to check against (default: %(default)s).")
    ap.add_argument("--prompt-length", type=int, default=4096,
                    help="Evaluation rollout.prompt_length to check against (default: %(default)s).")
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

    report_lengths(
        df[args.messages_key].tolist(),
        args.tokenizer_path,
        args.max_length,
        args.prompt_length,
    )

    df.to_parquet(args.out_path)
    print(f"wrote {len(df)} rows to {args.out_path} with '{args.messages_key}' column")


if __name__ == "__main__":
    main()
