"""PICO rule-based math reward. Merged copy of verl/custom/math_reward_manager.py,
verl/custom/math_verify_reward.py and the two deepscaler pieces they import.

Score per response is 1 (correct) or 0, written on its last valid token. reward_impl_version
3 expects a ``<think>...</think>`` block before the answer, 4 grades the whole response.

Not registered under ``"math"``: verl/workers/reward_manager/__init__.py already registers
verl.custom's MathRewardManager under that name at import time, and the registry rejects a
second class for it. verl/PICO/main_ppo_new.py constructs this class directly instead.
"""

import torch
from math_verify import parse, verify

from verl import DataProto
from verl.utils.reward_score import gsm8k, math

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def math_verify_is_correct(solution_str: str, ground_truth, think_format: bool) -> bool:
    """True when the answer in ``solution_str`` matches ``ground_truth`` under math_verify.

    With ``think_format`` the answer is what follows ``</think>``; a response without a
    complete think block is wrong.
    """
    if think_format:
        if THOUGHT_DELIMITER_START not in solution_str or THOUGHT_DELIMITER_END not in solution_str:
            return False
        solution_str = solution_str.split(THOUGHT_DELIMITER_END)[1]
    return verify(parse("$" + ground_truth + "$"), parse(solution_str)) is True


def reward_fn_math_verify(solution_str, ground_truth, enable_llm=False):
    return math_verify_is_correct(solution_str, ground_truth, think_format=True)


def reward_fn_math_verify_no_think(solution_str, ground_truth, enable_llm=False):
    return math_verify_is_correct(solution_str, ground_truth, think_format=False)


def math_select_rm_score_fn(data_source, reward_impl_version):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score
    if data_source == "lighteval/MATH":
        return math.compute_score
    if reward_impl_version == 3:
        return reward_fn_math_verify
    if reward_impl_version == 4:
        return reward_fn_math_verify_no_think
    raise NotImplementedError(f"reward_impl_version={reward_impl_version} is not supported (use 3 or 4)")


class PICOMathRewardManager:
    def __init__(self, tokenizer, num_examine, reward_impl_version) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_impl_version = reward_impl_version

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(data)):
            item = data[i]
            prompt_length = item.batch["prompts"].shape[-1]
            valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
            sequences_str = self.tokenizer.decode(item.batch["responses"][:valid_response_length])
            if self.reward_impl_version != 4:
                sequences_str = THOUGHT_DELIMITER_START + "\n" + sequences_str

            compute_score = math_select_rm_score_fn(item.non_tensor_batch["data_source"], self.reward_impl_version)
            score = compute_score(solution_str=sequences_str, ground_truth=item.non_tensor_batch["reward_model"]["ground_truth"])
            reward_tensor[i, valid_response_length - 1] = score

        return {"reward_tensor": reward_tensor} if return_dict else reward_tensor
