# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor

PICO actor. Only ``update_policy`` is overridden, and it follows
verl/workers/actor/dp_actor.py's line for line; PICO's additions are marked ``# PICO:``:
  * it also selects ``prefix_mask`` / ``reward_mask`` (and ``off_log_z`` when present),
  * the loss keeps the stock ``vanilla`` (plain GRPO / PPO) and registry branches, and adds an
    ``elif`` for PICO's loss modes: new_core_alg.compute_token_on_off_sft_loss, which trains
    rows with ``prefix_mask=1`` (the summarize-replacement candidates) with the off-policy term,
  * it returns ``(metrics, {"log_prob": ...})``: the trainer logs pi_new / pi_old per row.
"""

import itertools
import logging
import os

import torch

from verl import DataProto
from verl.PICO.new_core_alg import compute_token_on_off_sft_loss
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

__all__ = ["PICODataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# PICO: "<on rows>-<off rows>", see new_core_alg.compute_token_on_off_sft_loss.
PICO_LOSS_MODES = ("rl-rl", "rl-sft", "sft-sft", "none-sft")

# PICO: compute_token_on_off_sft_loss outputs logged as actor/<name>.
_LOGGED = (
    "off_pg_loss",
    "on_pg_loss",
    "off_pg_clipfrac",
    "on_pg_clipfrac",
    "off_ratio_mean",
    # Mean Z_t over the off tokens (1.0 when no row carries off_log_z).
    "off_z_mean",
    # Same denominator as pg_loss, so on_loss_contrib + off_loss_contrib == pg_loss.
    "on_loss_contrib",
    "off_loss_contrib",
    "off_sft_loss_contrib",
    "off_rl_loss_contrib",
    "on_loss_ratio",
    "off_loss_ratio",
    "off_low_conf_token_frac",
    "on_pos_adv_loss_contrib",
    "on_neg_adv_loss_contrib",
    "on_pos_adv_loss_ratio",
    "on_neg_adv_loss_ratio",
)
# Only meaningful when the off region has a loss at all; skipped otherwise so a batch without
# replaced rows does not drag the mean towards 0/0.
_LOGGED_IF_OFF = ("off_sft_ratio_in_off", "off_rl_ratio_in_off")


class PICODataParallelPPOActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        # PICO: prefix_mask marks the replaced rows, reward_mask the rows the loss sees.
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "prefix_mask", "reward_mask"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # PICO: joint decoding's per-token log Z_t, carried by batches with replaced rows.
        if "off_log_z" in data.batch.keys():
            select_keys.append("off_log_z")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        # PICO: options of the PICO loss modes.
        # all_max_clip=10 never masks a token (a probability is at most 1), but it is not inert:
        # it is what makes new_core_alg multiply the loss-split metrics (on_loss_contrib, ...) by
        # response_mask, so train_PICO.sh sets it. Gradients are the same either way.
        all_max_clip = self.config.get("all_max_clip", -1)
        pico_loss_kwargs = dict(
            all_max_clip=all_max_clip if all_max_clip != -1 else None,
            off_policy_reshape=self.config.policy_loss.get("off_policy_reshape", "clip"),
            off_cliprange_high=self.config.get("off_clip_ratio_high", None),
        )
        # PICO: the updated log-probs, returned so the trainer can log pi_new / pi_old per row.
        epoch_log_probs = []

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            epoch_log_probs = []
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                # PICO: rearrange_micro_batches' reordering, to put the log-probs back in row order.
                reorder_indices = None
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)
                        reorder_indices = textual_indices

                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, reorder_indices = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                mini_log_probs = []

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    mini_log_probs.append(log_prob.detach())

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    # PICO: the loss metrics, filled per branch below.
                    loss_metrics = {}
                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        loss_metrics["actor/pg_clipfrac"] = pg_clipfrac.detach().item()
                        loss_metrics["actor/pg_clipfrac_lower"] = pg_clipfrac_lower.detach().item()
                    elif loss_mode in PICO_LOSS_MODES:
                        # PICO: on rows (prefix_mask=0) get the on-policy term and replaced rows the
                        # off-policy one, as selected by loss_mode; reward_mask gates both.
                        ret = compute_token_on_off_sft_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            prefix_mask=data["prefix_mask"],
                            reward_mask=data["reward_mask"],
                            off_log_z=data["off_log_z"] if "off_log_z" in data.keys() else None,
                            off_policy_strategy=loss_mode,
                            **pico_loss_kwargs,
                        )
                        pg_loss = ret["pg_loss"]
                        ppo_kl = ret["ppo_kl"]
                        # PICO: the on / off split of the loss.
                        for name in _LOGGED:
                            if name in ret:
                                loss_metrics[f"actor/{name}"] = ret[name].detach().item()
                        if ret["off_has_loss"].item():
                            for name in _LOGGED_IF_OFF:
                                loss_metrics[f"actor/{name}"] = ret[name].detach().item()
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config)
                        loss_metrics["actor/pg_clipfrac"] = pg_clipfrac.detach().item()
                        loss_metrics["actor/pg_clipfrac_lower"] = pg_clipfrac_lower.detach().item()

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        **loss_metrics,
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)

                # PICO: this mini-batch's log-probs, back in its own row order.
                mini_log_prob = torch.cat(mini_log_probs, dim=0)
                if reorder_indices is not None:
                    order = list(itertools.chain.from_iterable(reorder_indices))
                    mini_log_prob = mini_log_prob[torch.tensor(get_reverse_idx(order), dtype=torch.long)]
                epoch_log_probs.append(mini_log_prob)
        self.actor_optimizer.zero_grad()

        # PICO: the last epoch's log-probs for this shard's rows.
        log_prob = torch.cat(epoch_log_probs, dim=0) if epoch_log_probs else None
        if log_prob is not None and log_prob.shape[0] != batch.batch_size[0]:
            logger.warning(f"Reconstructed log_prob has {log_prob.shape[0]} rows, expected {batch.batch_size[0]}; not returning it.")
            log_prob = None
        return metrics, {"log_prob": log_prob}
