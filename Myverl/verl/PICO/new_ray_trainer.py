# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""PICO trainer: GRPO with summarize-replacement (SR). Trimmed copy of
verl/custom/new_ray_trainer.py, keeping only what examples/h20/train_PICO.sh runs.

One training step:

  1. rollout       -- vLLM samples n responses per question (short prompt).
  2. SR            -- for the questions picked by ``summarize_replace`` (all / wrong_only), build
                      candidates (``_generate_candidates``; today: joint decoding of the live actor
                      under a frozen teacher), score them, and splice one correct candidate into a
                      wrong slot as an "off" row: [short prompt, candidate], ``prefix_mask=1``.
  3. reward        -- rule-based reward, optional trajectory filter on the off rows, group stats.
  4. old_log_prob  -- the actor's own log-probs. The off rows' importance-ratio denominator is the
                      actor itself (the student IS the actor), so nothing is swapped in; the
                      teacher's per-token cost travels separately as ``off_log_z``.
  5. advantage     -- GRPO.
  6. update_actor  -- ``policy_loss.loss_mode`` picks how on/off rows are trained:
                      rl-rl / rl-sft / sft-sft / none-sft (see new_core_alg.compute_token_on_off_sft_loss),
                      or the stock ``vanilla`` / registered losses, which treat every row alike.

Entry point: ``python -m verl.PICO.main_ppo_new``.
"""

import json
import os
import re
import uuid
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.PICO import joint_sr
from verl.PICO.new_dp_actor import PICO_LOSS_MODES
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

# Reward values of the math reward manager (verl/PICO/math_reward_manager.py).
SUCCESS, FAIL = 1, 0


# ---------------------------------------------------------------------------
# Trajectory filter for summarize-replacement candidates:
#   - reject responses that are too short (likely a continuation, not a re-solve)
#   - reject responses that still reference the draft / experience (keywords)
#   - reject responses that restate the summarize-template instructions
#   - reject noisy responses with long runs of repeated chars / substrings
# Defaults derive from DEFAULT_TEMPLATE in Data/prepare_summarize_prompts.py.
# ---------------------------------------------------------------------------
_DEFAULT_TRAJ_KEYWORDS = [
    "the draft",
    "this draft",
    "reasoning draft",
    "let's summarize the reasoning",
    "partial reasoning draft",
    "[reasoning draft]",
    "[partial reasoning draft]",
    "the provided draft",
    "the given draft",
    "the reference reasoning",
    "reference reasoning",
    "provided reasoning",
    "based on the reasoning above",
    "given reasoning",
    "based on the draft",
    "according to the draft",
    "as stated in the draft",
    "as shown in the draft",
    "from the draft",
    "the experience",
    "refer to experience",
    "based on the experience",
    "according to the experience",
]
_DEFAULT_TRAJ_INSTR_PHRASES = [
    "gold standard",
    "strict requirements",
    "total de-reference",
    "invisible integration",
    "re-author",
    "your task is",
    "do not mention",
    "do not quote",
    "do not summarize",
    "do not copy",
    "output only",
    "the reader must",
    "the reader should",
    "commentary on the draft",
    "single coherent solution",
    "self-contained solution",
    "no meta-talk",
    "conversational fillers",
    "final output should read",
]


def trajectory_filter_reject(text: str, cfg: dict):
    """Return (reject: bool, reason: str) for a rewritten-trajectory response.

    cfg keys (all optional):
      min_length_chars      : reject if the response is shorter than N chars (default 200)
      keywords              : reject if any appears (case-insensitive)
      instruction_phrases   : reject if any appears (restates the summarize template)
      max_repeated_char_run : reject if a single char repeats >= N times (default 50)
      max_repeated_substr   : reject if a 2-20 char block repeats >= K times (default 10)
    """
    n_min_len = cfg.get("min_length_chars", 200)
    if n_min_len and len(text) < int(n_min_len):
        return True, "too_short"
    low = text.lower()
    for kw in cfg.get("keywords", _DEFAULT_TRAJ_KEYWORDS):
        if kw.lower() in low:
            return True, "keyword"
    for ph in cfg.get("instruction_phrases", _DEFAULT_TRAJ_INSTR_PHRASES):
        if ph.lower() in low:
            return True, "instruction"
    n_char = cfg.get("max_repeated_char_run", 50)
    if n_char and re.search(r"(.)\1{" + str(int(n_char) - 1) + ",}", text):
        return True, "noise_char"
    n_sub = cfg.get("max_repeated_substr", 10)
    if n_sub and re.search(r"(.{2,20}?)\1{" + str(int(n_sub) - 1) + ",}", text):
        return True, "noise_substr"
    return False, ""


def index_non_tensor(nt: dict, idx: list) -> dict:
    """Row-index a non_tensor_batch dict by a list of positions (ndarray / list values)."""
    out = {}
    for k, v in nt.items():
        if isinstance(v, np.ndarray):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


def generate_masks_from_input_ids(input_ids: torch.Tensor, pad_token_id: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """attention_mask and position_ids from input_ids; works for left and right padding."""
    attention_mask = (input_ids != pad_token_id).to(dtype)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    return attention_mask, position_ids


def build_off_rows(prompts: torch.Tensor, responses: torch.Tensor, lengths: torch.Tensor, width: int, pad_token_id: int, mask_dtype) -> dict:
    """[prompt, response] rows trained as off-policy: ``prefix_mask`` covers the response.

    ``prompts`` is left-padded [R, Lp]; ``responses`` is right-padded [R, w] and is padded or
    cut to ``width`` so the rows line up with the on-policy batch. ``lengths`` [R] is each
    response's token count including its EOS.

    The response masks come from ``lengths`` rather than from the pad token, because the EOS and
    the pad can be the same id (Qwen3's are both <|endoftext|>): reading the pad token would
    drop the EOS from the mask, and the model would never be trained to stop on these rows.
    This matches the vLLM rows, whose mask (get_response_mask) keeps the first EOS. The prompt
    keeps the pad-token mask, as the vLLM rows do.
    """
    if responses.size(1) < width:
        fill = torch.full((responses.size(0), width - responses.size(1)), pad_token_id, dtype=responses.dtype, device=responses.device)
        responses = torch.cat([responses, fill], dim=1)
    responses = responses[:, :width].to(prompts.dtype)
    lengths = lengths.to(responses.device).clamp(max=width)
    prefix_mask = torch.arange(width, device=responses.device).unsqueeze(0) < lengths.unsqueeze(1)

    prompt_mask, prompt_position_ids = generate_masks_from_input_ids(prompts, pad_token_id, mask_dtype)
    attention_mask = torch.cat([prompt_mask, prefix_mask.to(mask_dtype)], dim=-1)
    # Continue the prompt's numbering through the response, as vLLM does.
    response_position_ids = prompt_position_ids[:, -1:] + torch.arange(1, width + 1, device=responses.device)
    return {
        "prompts": prompts,
        "responses": responses,
        "input_ids": torch.cat([prompts, responses], dim=-1),
        "attention_mask": attention_mask,
        "position_ids": torch.cat([prompt_position_ids, response_position_ids], dim=-1),
        "prefix_mask": prefix_mask,
    }


def _jsonable(obj):
    """Tensors / numpy values -> plain Python, recursively, so json.dumps accepts them."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class PICORayPPOTrainer(RayPPOTrainer):
    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _validate_config(self):
        super()._validate_config()
        config = self.config
        rollout = config.actor_rollout_ref.rollout
        joint = config.get("joint_decode", {})
        loss_mode = config.actor_rollout_ref.actor.policy_loss.loss_mode
        reshape = config.actor_rollout_ref.actor.policy_loss.get("off_policy_reshape", "p_div_p_0.1")

        if rollout.get("summarize_replace", None) not in ("all", "wrong_only"):
            raise ValueError("PICO needs rollout.summarize_replace='wrong_only' or 'all'.")
        if rollout.get("summarize_replace_select", "shortest") not in ("shortest", "longest"):
            raise ValueError("rollout.summarize_replace_select must be 'shortest' or 'longest'.")
        if not joint.get("enable", False) or not joint.get("teacher_model_path"):
            raise ValueError("PICO generates SR candidates by joint decoding: set +joint_decode.enable=True and +joint_decode.teacher_model_path.")
        # Two branches of the stock fit() that _train_step does not carry; see there.
        if config.reward_model.get("launch_reward_fn_async", False) or config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise ValueError("PICO does not support reward_model.launch_reward_fn_async or adv_estimator=remax.")

        if abs(float(joint.get("temperature", 1.0)) - float(rollout.temperature)) > 1e-6:
            print(f"[sr] WARNING: joint_decode.temperature={joint.get('temperature')} != rollout.temperature={rollout.temperature}; off_log_z and old_log_probs are then taken at different temperatures.")
        if loss_mode not in PICO_LOSS_MODES:
            # The actor's stock branches: vanilla (plain GRPO / PPO) or a registered policy loss.
            print(f"[sr] loss_mode={loss_mode!r} is a stock loss: replaced rows are trained like any other rollout, ignoring prefix_mask, reward_mask and off_log_z.")
        elif loss_mode.endswith("-sft"):
            print("[sr] off rows are trained with SFT, so off_log_z (Z_t) does not enter the loss.")
        print(f"[sr] summarize_replace={rollout.summarize_replace}, loss_mode={loss_mode}, off_policy_reshape={reshape}, teacher={joint.get('teacher_model_path')}")

    # ------------------------------------------------------------------
    # validation: stock metrics plus an exact pass@k
    # ------------------------------------------------------------------
    def _validate(self) -> dict:
        """verl reports best@k (a bootstrap estimate); this adds the exact share of questions
        with at least one hit, by intercepting process_validation_metrics for the call."""
        from verl.PICO.new_metrics import compute_exact_pass_at_k
        from verl.trainer.ppo import ray_trainer as _rt

        captured = {}
        original = _rt.process_validation_metrics

        def _capture(data_sources, sample_inputs, infos_dict, *args, **kwargs):
            scores = infos_dict.get("acc" if "acc" in infos_dict else "reward")
            if scores is not None:
                captured.update(compute_exact_pass_at_k(data_sources, sample_inputs, scores))
            return original(data_sources, sample_inputs, infos_dict, *args, **kwargs)

        _rt.process_validation_metrics = _capture
        try:
            metric_dict = super()._validate()
        finally:
            _rt.process_validation_metrics = original

        for data_source, metrics in captured.items():
            for name, value in metrics.items():
                section = "val-core" if name.startswith("pass@") else "val-aux"
                metric_dict[f"{section}/{data_source}/acc/{name}"] = value
        return metric_dict

    # ------------------------------------------------------------------
    # summarize replacement
    # ------------------------------------------------------------------
    def _generate_candidates(self, question_ids, batch, short_prompts, mask_dtype, timing_raw, metrics):
        """Produce SR candidates for ``question_ids``.

        Returns ``(question_ids, k, responses, lengths, columns)``:
          question_ids -- the questions actually covered (a source may cap them),
          k            -- candidates per question; row ``i * k + j`` is candidate j of question i,
          responses    -- [len(question_ids) * k, w] right-padded,
          lengths      -- [len(question_ids) * k] tokens per response, its EOS included,
          columns      -- extra per-token columns to carry into the off rows. Each must also be
                          allocated for the whole batch at the top of ``_summarize_replace``.

        The only source today is joint decoding. Another source (e.g. a trajectory generated by
        the teacher alone, or one read from the dataset) goes here as another branch.
        """
        joint = self.config.joint_decode
        world_size = self.actor_rollout_wg.world_size
        batch_size = int(joint.get("batch_size", 8))
        # K from the uncapped count, then cap the questions at K -- the other order would size K
        # from the already-capped count and overshoot the row budget.
        k = joint_sr.resolve_k(len(question_ids), batch_size, world_size, target_rows=int(joint.get("target_rows", 0)), max_k=int(joint.get("max_k", 1)))
        question_ids = joint_sr.select_questions(question_ids, batch_size, world_size, metrics, k_per_question=k)
        metrics["batch/sr_joint_k"] = k
        metrics["batch/sr_joint_rows"] = len(question_ids) * k

        # Teacher reads the summarize prompt, student the short one the loss is taken under.
        summarize_ids = batch.batch["summarize_input_ids"]  # [B, K_data, L_long]
        k_data = summarize_ids.size(1)
        long_prompts = torch.stack([summarize_ids[q, min(j, k_data - 1)] for q in question_ids for j in range(k)])
        short_rows = short_prompts[torch.tensor(question_ids)].repeat_interleave(k, dim=0)
        long_rows = long_prompts.to(short_rows.device)

        with marked_timer("sr_joint_gen", timing_raw, color="cyan"):
            cand_out, off_log_z, lengths = joint_sr.generate(
                worker_group=self.actor_rollout_wg,
                joint_config=joint,
                student_prompts=short_rows if joint.get("student_prompt", "short") == "short" else long_rows,
                teacher_prompts=long_rows if joint.get("teacher_prompt", "long") == "long" else short_rows,
                pad_token_id=self.tokenizer.pad_token_id,
                max_response_length=self.config.data.max_response_length,
                attention_mask_dtype=mask_dtype,
                make_masks=generate_masks_from_input_ids,
                meta_info={},
                metrics=metrics,
            )
        return question_ids, k, cand_out.batch["responses"], lengths, {"off_log_z": off_log_z}

    def _summarize_replace(self, gen_batch, gen_batch_output, batch, timing_raw, metrics):
        """Splice one correct candidate per selected question into ``gen_batch_output``.

        Rows are interleaved (question q's rollouts are rows q*n .. q*n+n-1). A replaced row
        becomes [short prompt, candidate] with ``prefix_mask=1``; the slot is the question's
        first wrong rollout when rewards are known, else its last.
        """
        rollout = self.config.actor_rollout_ref.rollout
        pad_token_id = self.tokenizer.pad_token_id
        responses = gen_batch_output.batch["responses"]
        n_questions = len(batch)
        n = responses.size(0) // n_questions

        # Allocated for every row: the splice below only writes keys that already exist.
        # 0 is neutral (exp(0) = 1), and on rows never reach the off branch anyway.
        gen_batch_output.batch["off_log_z"] = torch.zeros(responses.shape, dtype=torch.float32)

        # 1. which questions get a candidate
        on_scores = None
        if rollout.summarize_replace == "wrong_only":
            saved = gen_batch_output.non_tensor_batch
            gen_batch_output.non_tensor_batch = index_non_tensor(batch.non_tensor_batch, [q for q in range(n_questions) for _ in range(n)])
            on_reward, _ = compute_reward(gen_batch_output, self.reward_fn)
            gen_batch_output.non_tensor_batch = saved
            on_scores = on_reward.sum(-1).view(n_questions, n)
            threshold = rollout.get("accuracy_threshold", 0.0)
            if threshold and threshold > 0:
                picked = (on_scores == SUCCESS).float().mean(-1) <= threshold
            else:
                picked = ~(on_scores == SUCCESS).any(-1)
            question_ids = torch.nonzero(picked).squeeze(-1).tolist()
            metrics["batch/sr_wrong_questions"] = len(question_ids)
            if not question_ids:
                return gen_batch_output
        else:
            question_ids = list(range(n_questions))
        metrics["batch/sr_target_questions"] = len(question_ids)

        # 2. candidates
        short_prompts = gen_batch.batch["input_ids"][::n]  # [B, L_short], one per question
        mask_dtype = gen_batch_output.batch["attention_mask"].dtype
        question_ids, k, cand_responses, cand_lengths, columns = self._generate_candidates(question_ids, batch, short_prompts, mask_dtype, timing_raw, metrics)
        n_cand = len(question_ids) * k

        off_rows = build_off_rows(
            prompts=short_prompts[torch.tensor(question_ids)].repeat_interleave(k, dim=0),
            responses=cand_responses,
            lengths=cand_lengths,
            width=max(self.config.data.max_response_length, cand_responses.size(1)),
            pad_token_id=pad_token_id,
            mask_dtype=mask_dtype,
        )
        for key, value in columns.items():
            assert value.shape == off_rows["responses"].shape, f"{key} {tuple(value.shape)} does not align with the candidate tokens {tuple(off_rows['responses'].shape)}"
            off_rows[key] = value

        # 3. score candidates, optionally screen them with the trajectory filter
        scored = DataProto.from_dict(tensors={k_: off_rows[k_] for k_ in ("prompts", "responses", "input_ids", "attention_mask", "position_ids")})
        scored.non_tensor_batch = index_non_tensor(batch.non_tensor_batch, [q for q in question_ids for _ in range(k)])
        cand_scores = compute_reward(scored, self.reward_fn)[0].sum(-1)  # [n_cand]
        tf_cfg = self.config.algorithm.get("trajectory_filter", {})
        cand_texts = self.tokenizer.batch_decode(off_rows["responses"], skip_special_tokens=True) if tf_cfg.get("enable", False) else None

        # 4. pick one candidate per question and splice it in
        take_last = rollout.get("summarize_replace_select", "shortest") == "longest"
        n_accepted = n_no_candidate = n_rej_incorrect = n_rej_filter = 0
        for i, q in enumerate(question_ids):
            chosen = -1
            for j in range(k):
                r = i * k + j
                if cand_scores[r].item() != SUCCESS:
                    n_rej_incorrect += 1
                    continue
                if cand_texts is not None and trajectory_filter_reject(cand_texts[r], tf_cfg)[0]:
                    n_rej_filter += 1
                    continue
                chosen = r
                if not take_last:
                    break
            if chosen < 0:
                n_no_candidate += 1
                continue
            slot = n - 1
            if on_scores is not None:
                wrong = (on_scores[q] != SUCCESS).nonzero().squeeze(-1)
                if wrong.numel() > 0:
                    slot = int(wrong[0].item())
            target = q * n + slot
            for key, value in off_rows.items():
                if key in gen_batch_output.batch.keys():
                    gen_batch_output.batch[key][target] = value[chosen].to(gen_batch_output.batch[key].device)
            n_accepted += 1

        n_correct = int((cand_scores == SUCCESS).sum().item())
        n_solved = int((cand_scores.view(len(question_ids), k) == SUCCESS).any(-1).sum().item())
        metrics.update(
            {
                "batch/sr_rollout_acc": n_correct / max(1, n_cand),
                "batch/sr_rollout_correct": n_correct,
                "batch/sr_rollout_total": n_cand,
                "batch/sr_question_solve_rate": n_solved / max(1, len(question_ids)),
                "batch/sr_accepted": n_accepted,
                "batch/sr_no_candidate": n_no_candidate,
                "batch/sr_rej_incorrect": n_rej_incorrect,
                "batch/sr_rej_filter": n_rej_filter,
            }
        )
        print(f"[summarize_replace] questions={len(question_ids)}, K={k}, accepted={n_accepted}, no_candidate={n_no_candidate}, rej_incorrect={n_rej_incorrect}, rej_filter={n_rej_filter}, rollout_acc={n_correct}/{n_cand}")
        return gen_batch_output

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.

        Same as RayPPOTrainer.fit(), except that the loop body is _train_step() and each step's
        metrics are also appended to <trainer.metrics_data_dir>/training_metrics.jsonl.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics, last_val_metrics = self._train_step(batch_dict, epoch, is_last_step)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                metrics_data_dir = self.config.trainer.get("metrics_data_dir", None)
                if metrics_data_dir:
                    os.makedirs(metrics_data_dir, exist_ok=True)
                    with open(os.path.join(metrics_data_dir, "training_metrics.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(_jsonable(metrics)) + "\n")

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _train_step(self, batch_dict, epoch, is_last_step):
        """One iteration of RayPPOTrainer.fit()'s loop body, in the same order.

        PICO's additions are marked ``# PICO:``. Returns (metrics, last_val_metrics).

        Not carried over from the stock body: the REMAX baseline generation and the async
        reward (both rejected in _validate_config), and the critic update (no critic under GRPO).
        """
        do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
        if do_profile:
            self.actor_rollout_wg.start_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()

        metrics = {}
        timing_raw = {}
        last_val_metrics = None
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # PICO: position within the step's batch; Data/visualize_rollouts.py groups dumps by it.
        batch.non_tensor_batch["original_index"] = np.arange(len(batch))
        n = self.config.actor_rollout_ref.rollout.n

        # pop those keys for generation
        gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        # PICO: repeat here rather than inside vLLM (the vLLM rollout samples one response per
        # row), so _summarize_replace sees the interleaved [question * n] layout.
        gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)

        with marked_timer("step", timing_raw):
            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

            # PICO: summarize replacement -- splice one correct joint-decoded candidate into a
            # wrong slot of each selected question. prefix_mask marks those rows as off-policy.
            gen_batch_output.batch["prefix_mask"] = torch.zeros(gen_batch_output.batch["responses"].shape, dtype=torch.bool)
            disable_after = self.config.data.get("recycle_sr_disable_after", 0)
            if disable_after <= 0 or self.global_steps < disable_after:
                gen_batch_output = self._summarize_replace(gen_batch, gen_batch_output, batch, timing_raw, metrics)

            # PICO: the summarize prompts are only needed by SR; the reference solution only by
            # the rollout dump, so it travels as text instead of [B, L] ids repeated n times.
            batch.pop(batch_keys=["summarize_input_ids"])
            tgt_input_ids = batch.pop(batch_keys=["tgt_input_ids"]).batch["tgt_input_ids"]
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                batch.non_tensor_batch["target"] = np.array(self.tokenizer.batch_decode(tgt_input_ids, skip_special_tokens=True), dtype=object)

            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=n, interleave=True)
            batch = batch.union(gen_batch_output)

            batch.batch["response_mask"] = compute_response_mask(batch)
            # PICO: no SE rows; kept all-zero so save_tensors output keeps the layout
            # Data/load_saved_tensors.py and the plot_* scripts expect.
            batch.batch["se_mask"] = torch.zeros_like(batch.batch["prefix_mask"])
            # Balance the number of valid tokens across DP ranks.
            # NOTE: This usually changes the order of data in the `batch`,
            # which won't affect the advantage calculation (since it's based on uid),
            # but might affect the loss calculation (due to the change of mini-batching).
            # TODO: Decouple the DP balancing and mini-batching.
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            response_mask = batch.batch["response_mask"]
            off_mask = batch.batch["prefix_mask"].any(-1)
            on_mask = ~off_mask

            with marked_timer("reward", timing_raw, color="yellow"):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # PICO: group the rollouts by question (uid) for two things:
                #   * reward_mask -- which rows the loss sees. policy_loss.mask_off_inject_group
                #     acts on questions that received a replaced row: True drops the whole group,
                #     'on_only' drops its on-policy rows, False (default) keeps everything.
                #   * solve_none / solve_one over on-policy rows only (a replaced correct row would
                #     otherwise make an all-wrong question look solved once); solve_all over the
                #     whole group.
                scores = reward_tensor.sum(-1)
                uids = batch.non_tensor_batch["uid"]
                off_rows = off_mask.cpu().numpy()
                mask_off_inject_group = self.config.actor_rollout_ref.actor.policy_loss.get("mask_off_inject_group", False)
                row_keep = torch.ones(len(uids), dtype=torch.bool)
                solve_none = solve_one = solve_all = 0
                for uid in np.unique(uids):
                    in_group = uids == uid
                    on_group = in_group & ~off_rows
                    if on_group.any():
                        on_scores = scores[torch.from_numpy(on_group)]
                        if (on_scores == FAIL).all():
                            solve_none += 1
                        elif (on_scores == SUCCESS).sum() == 1:
                            solve_one += 1
                    if (scores[torch.from_numpy(in_group)] == SUCCESS).all():
                        solve_all += 1
                    if off_rows[in_group].any():
                        if mask_off_inject_group == "on_only":
                            row_keep[torch.from_numpy(on_group)] = False
                        elif mask_off_inject_group:
                            row_keep[torch.from_numpy(in_group)] = False
                batch.batch["reward_mask"] = row_keep.unsqueeze(-1).expand_as(response_mask).clone()
                batch.non_tensor_batch["reward_sum"] = scores.unsqueeze(1).cpu().numpy()

                # PICO: solve counts and solved fractions, plus token counts / lengths, for on rows
                # and replaced (off) rows.
                metrics["batch/solve_none"] = solve_none
                metrics["batch/solve_one"] = solve_one
                metrics["batch/solve_all"] = solve_all
                metrics["batch/solved"] = (scores == SUCCESS).sum().item() / len(scores)
                metrics["batch/failed"] = (scores == FAIL).sum().item() / len(scores)
                metrics["batch/on_solved"] = (scores[on_mask] == SUCCESS).sum().item() / (on_mask.sum().item() + 1e-6)
                metrics["batch/off_solved"] = (scores[off_mask] == SUCCESS).sum().item() / (off_mask.sum().item() + 1e-6)
                off_correct = off_mask & (scores == SUCCESS)
                for name, rows in (("on", on_mask), ("off_standard", off_mask), ("off_correct", off_correct)):
                    tokens = response_mask[rows].sum().item()
                    if name != "off_correct":
                        metrics[f"batch/{name}_tokens"] = int(tokens)
                    metrics[f"batch/{name}_avg_response_len"] = tokens / (rows.sum().item() + 1e-6)

            # recompute old_log_probs
            # PICO: the off rows' importance-ratio denominator is the actor's own log-prob (the
            # student IS the actor), so nothing is swapped in; the teacher's constraint reaches
            # the loss only through off_log_z.
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)
                # PICO: entropy / old prob split into on rows and replaced rows.
                old_probs = torch.exp(batch.batch["old_log_probs"])
                for name, rows in (("on", on_mask), ("off_standard", off_mask)):
                    if rows.any():
                        metrics[f"batch/entropy_{name}"] = agg_loss(loss_mat=entropys[rows], loss_mask=response_masks[rows], loss_agg_mode="token-mean").item()
                        metrics[f"batch/old_prob_{name}"] = verl_F.masked_mean(old_probs[rows], response_masks[rows]).item()

            if self.use_reference_policy:
                # compute reference log_prob
                with marked_timer("ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            with marked_timer("adv", timing_raw, color="brown"):
                # we combine with rule-based rm
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    config=self.config.algorithm,
                )

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)
                # PICO: pi_new / pi_old after the update, split into on rows and replaced rows.
                log_prob = actor_output.batch.get("log_prob", None)
                if log_prob is not None:
                    batch.batch["log_probs"] = log_prob
                    metrics.update(self._ratio_metrics(batch, on_mask, off_mask))

            # Log rollout generations if enabled
            if rollout_data_dir:
                with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                    self._dump_rollouts(batch, reward_extra_infos_dict, os.path.join(rollout_data_dir, "normal"))

            # validate
            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

        # training metrics
        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
            }
        )
        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        # PICO: per-step tensors for Data/load_saved_tensors.py and the Data/plot_* scripts.
        save_tensors_dir = self.config.trainer.get("save_tensors_dir", None)
        if save_tensors_dir:
            self._save_tensors(batch, save_tensors_dir, self.config.trainer.get("save_tensor_dtype", "fp32"))

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()

        return metrics, last_val_metrics

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ratio_metrics(batch, on_mask, off_mask) -> dict:
        """Mean / variance of pi_new and pi_new / pi_old after the update, for on and off rows."""
        log_prob = batch.batch["log_probs"]
        response_mask = batch.batch["response_mask"]
        probs = torch.exp(log_prob)
        ratio = torch.clamp(torch.exp(log_prob - batch.batch["old_log_probs"]), 1e-3, 10)
        out = {}
        for name, rows in (("on", on_mask), ("off_standard", off_mask)):
            if not rows.any():
                continue
            mask = response_mask[rows]
            out[f"batch/prob_{name}"] = verl_F.masked_mean(probs[rows], mask).item()
            out[f"batch/prob_{name}_var"] = verl_F.masked_var(probs[rows], mask, False).item()
            out[f"batch/ratio_{name}_mean"] = verl_F.masked_mean(ratio[rows], mask).item()
            out[f"batch/ratio_{name}_var"] = verl_F.masked_var(ratio[rows], mask, False).item()
        return out

    def _dump_rollouts(self, batch, reward_extra_infos_dict, dump_path):
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        infos = dict(reward_extra_infos_dict or {})
        keys = self.config.trainer.get("rollout_dump_extra_keys", ["reward_model", "data_source", "original_index", "uid", "extra_info", "target"])
        for key in keys:
            if key in batch.non_tensor_batch and len(batch.non_tensor_batch[key]) == len(inputs):
                infos[key] = [_jsonable(v) for v in batch.non_tensor_batch[key]]
        # True for rows spliced in by summarize replacement.
        infos["is_replaced"] = batch.batch["prefix_mask"].any(-1).cpu().tolist()
        self._dump_generations(inputs=inputs, outputs=outputs, scores=scores, reward_extra_infos_dict=infos, dump_path=dump_path)

    def _save_tensors(self, batch, save_dir, save_dtype):
        """One .pt per step, read by Data/load_saved_tensors.py and the Data/plot_* scripts."""
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(save_dtype, torch.float16)
        out = {"step": int(self.global_steps)}
        for key in ("old_log_probs", "log_probs", "prefix_mask", "se_mask", "response_mask"):
            if key in batch.batch.keys():
                out[key] = batch.batch[key].detach().to(dtype=dtype, device="cpu")
        for key in ("reward_sum", "uid"):
            out[key] = list(batch.non_tensor_batch[key])
        os.makedirs(save_dir, exist_ok=True)
        torch.save(out, os.path.join(save_dir, f"tensors_step_{self.global_steps:07d}.pt"))
