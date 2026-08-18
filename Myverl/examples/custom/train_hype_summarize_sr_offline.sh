set -x
#!/usr/bin/env bash
# GPU selection. The server has 8 GPUs; expose 4 by default.
# Override with, for example: GPU_DEVICES=4,5,6,7 bash train_hype_summarize_sr_offline.sh
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

# ---------------------------------------------------------------------------
# Offline candidates + an online logprob-only rephraser.
#
# Two problems solved by one split of labour:
#
# GENERATION is autoregressive, so it needs a vLLM engine -- and a SECOND engine in the
# same process is not possible: vLLM's sleep mode is built on a process-global
# CuMemAllocator with hardcoded "weights"/"kv_cache" tags, so one engine's sleep() frees
# the other's memory (SGLang has the identical problem via torch_memory_saver). So
# generation is moved OFFLINE.
#
# LOG-PROBS are a single forward pass, no engine, no KV cache -- the same path
# critic/reward scoring already takes, which is exactly why those can coexist today. So
# the rephraser stays ONLINE as a logprob-only worker.
#
# Why this matters: log pi_phi(y|x_long) is the DENOMINATOR of the importance ratio, and
# it must come from the model that actually produced the candidates. Computing it with
# the reasoner instead would silently use a distribution that never emitted y -- no
# error, no NaN, just a biased gradient. Storing it in the parquet would work too, but
# costs one float per token and freezes the choice of rephraser; computing it online
# keeps the candidates reusable when phi changes.
#
# Pipeline:
#   1. sweep_sft_checkpoints.sh        -> per-batch parquets (responses + test_score)
#   2. Data/aggregate_sr_responses.py  -> one correct, median-length response per
#                                         question, in an `sr_response` column
#   3. Data/prepare_summarize_prompts.py -> the long prompt (still needed: it is what
#                                         the proposal log-prob is taken under)
#   4. this script                     -> reads the column, scores it with phi
#
# What the old per-step generation cost: 128 questions x 8 candidates of up to 14k
# tokens, of which one per question survives -- ~97% discarded. And with a frozen
# rephraser that sampling bought nothing: weights fixed, prompts pre-rendered,
# temperature fixed, so pi_phi(.|x_long) is the SAME distribution at step 1 and step
# 500. Re-sampling it each step was re-rolling one die.
#
# The math is unchanged, only reinterpreted: candidates come from the long summarize
# prompt, the loss is computed under the short one, so
#     off_ratio = pi_theta(y|x_short) / pi_phi(y|x_long)
# stays a valid importance weight -- now correcting a prompt shift AND a model shift.
#
# NOT reported here: val_summarize/*. Measuring it needs generation under the summarize
# prompt, which the logprob-only worker cannot do; the trainer returns {} rather than
# quietly reporting the reasoner's accuracy under a rephraser's name. Evaluate the
# rephraser offline (sweep_sft_checkpoints.sh + Data/prepare_rephrase_eval.py).
#
# WATCH actor/off_ratio_ess FROM STEP 1. theta and phi no longer share weights, so the
# per-token ratio drifts and over ~10k tokens the sequence weight degrades
# exponentially. If ESS collapses, switch off_policy_reshape to batch_mean_norm before
# concluding anything about the method. Also compare batch/long_prob_off_standard (phi,
# long prompt) against batch/old_prob_off_standard (theta, short prompt): that gap IS
# the combined shift.
#
# Usage:
#   TRAIN_PATH=$HOME/LLM/Data/deepmath_hard_sr.parquet \
#   REPHRASER_PATH=/home/data/shared/<model-that-generated-sr_response> \
#       bash train_hype_summarize_sr_offline.sh Qwen3-4b-base
#
#   # no rephraser worker: the reasoner scores the candidates itself. Cheaper, but the
#   # denominator is then pi_theta(y|x_long), not the true proposal density.
#   TRAIN_PATH=... bash train_hype_summarize_sr_offline.sh Qwen3-4b-base
#
#   # low-variance comparison: score under the SAME short prompt as the on-policy rows.
#   # See the SR_LOGPROB_PROMPT block below -- this is a different estimator, not a knob.
#   TRAIN_PATH=... SR_LOGPROB_PROMPT=short bash train_hype_summarize_sr_offline.sh
# ---------------------------------------------------------------------------

WORKER_DIR=/home/zhwang
MODEL_DIR=/home/data/shared
# Experiment knobs.
name="rl-rl"
off_policy_strategy="rl-rl"

# Must consume old_log_probs for the importance ratio to survive: dynamic_clip and
# vanilla skip the off_old_log_probs -> old_log_probs swap, and p_div_p_0.1 overwrites
# off_ratio outright. Either way the proposal density never reaches the loss.
off_policy_reshape="clip"
suffix="hype_summarize_sr_offline_"${off_policy_strategy}"_"${off_policy_reshape}

# Data paths. The train parquet MUST carry an `sr_response` column -- build it with
# Data/aggregate_sr_responses.py from a sweep_sft_checkpoints.sh output dir, then render
# the summarize prompts on top with Data/prepare_summarize_prompts.py (the long prompt is
# still needed: it is what the proposal log-prob is taken under).
train_path=${TRAIN_PATH:-$WORKER_DIR/LLM/Data/deepmath/deepmath_hard_sr.parquet}
test3_path=$WORKER_DIR/LLM/Data/split_by_source/amc_nothink.parquet
train_files="['$train_path']"
val_files="['$test3_path']"

MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4b-base"}

# The frozen rephraser: the model that GENERATED sr_response, used online only to score
# it. Empty = no rephraser worker, and the reasoner computes the log-prob itself (cheaper,
# but then the denominator is pi_theta(y|x_long), not the true proposal density).
#
# MUST share the reasoner's tokenizer family: summarize_input_ids are pre-tokenized
# offline with the reasoner's tokenizer, and the candidate token ids are spliced straight
# into the reasoner's input_ids without re-tokenizing. A different vocabulary corrupts
# the data silently.
REPHRASER_PATH=${REPHRASER_PATH:-}
if [ -n "$REPHRASER_PATH" ] && [ ! -d "$REPHRASER_PATH" ]; then
    echo "REPHRASER_PATH is not a directory: $REPHRASER_PATH" >&2
    exit 1
fi

PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME=${EXP_NAME:-"sr_offline_$(date +%m%d_%H%M)"}
LOG_DIR=/home/data/zhwang_logs/train_4b/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")
TENSOR_PARALLEL=1

# Only ONE vLLM engine either way (the rephraser has none), so the single-model budget
# applies. The rephraser costs bf16 params only -- no Adam state, no KV cache -- and
# param_offload keeps them in host RAM between uses.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}

# Which prompt the importance-ratio DENOMINATOR is computed under. These are two
# different estimators, not a tuning knob:
#
#   long  (default) -- q = pi(y|x_long), the distribution the candidates actually came
#                      from. The only mathematically correct proposal density, so
#                      off_ratio is a consistent policy gradient. Risk is variance: long
#                      vs short prompts differ a lot and over ~10k tokens the sequence
#                      weight can degrade exponentially. Watch actor/off_ratio_ess.
#   short           -- q = pi(y|x_short), the SAME prompt the on-policy rows use, so the
#                      ratio reflects only the model gap. Much lower variance, but it is
#                      not the density y was sampled from, so the estimator is biased.
#                      With no REPHRASER_PATH both sides are the same model under the same
#                      prompt, off_ratio ~= 1, and the importance correction vanishes --
#                      that trains on the offline candidates as if they were on-policy
#                      (closer to SFT than to a policy gradient).
#
# Run long first and read off_ratio_ess; switch to short as the low-variance comparison.
SR_LOGPROB_PROMPT=${SR_LOGPROB_PROMPT:-long}
if [ "$SR_LOGPROB_PROMPT" != "long" ] && [ "$SR_LOGPROB_PROMPT" != "short" ]; then
    echo "SR_LOGPROB_PROMPT must be 'long' or 'short', got: $SR_LOGPROB_PROMPT" >&2
    exit 1
fi

# Assembled below so the rephraser overrides vanish entirely when REPHRASER_PATH is unset.
REPHRASER_ARGS=()
if [ -n "$REPHRASER_PATH" ]; then
    REPHRASER_ARGS=(
        rephraser.enable=True
        +actor_rollout_rephraser.model.path=$REPHRASER_PATH
        +actor_rollout_rephraser.actor.fsdp_config.param_offload=True
    )
    echo "=== reasoner  (trained): $MODEL_PATH"
    echo "=== rephraser (frozen, logprob-only): $REPHRASER_PATH"
else
    echo "=== reasoner (trained): $MODEL_PATH"
    echo "=== no rephraser worker: the reasoner will score the offline candidates itself"
fi

cd "$MODEL_DIR" || exit 1
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi

# Fail here rather than after the model loads. The trainer also checks this, but the
# parquet path is the thing most likely to be wrong.
python - "$train_path" <<'PYEOF' || exit 1
import sys

import pandas as pd

path = sys.argv[1]
try:
    df = pd.read_parquet(path, columns=["sr_response"])
except Exception as error:
    sys.exit(
        f"[preflight] {path} has no usable 'sr_response' column ({error}).\n"
        f"            Build it with:\n"
        f"              python Data/aggregate_sr_responses.py --input-dir <sweep>/save_data "
        f"--output {path}"
    )
empty = int((df["sr_response"].fillna("").astype(str).str.strip() == "").sum())
print(f"[preflight] sr_response: {len(df):,} rows, {empty:,} empty "
      f"({empty / max(1, len(df)):.1%} of questions get no replacement)")
PYEOF

python -m verl.trainer.main_ppo_new \
    "${REPHRASER_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_reward=False \
    +algorithm.trajectory_filter.enable=True \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=14336 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.shuffle=False \
    +data.warmup_steps=0 \
    +data.reward_impl_version=4 \
    +data.filter_targets=False \
    +data.use_se=False \
    +data.collect_failures=False \
    +data.failure_buffer_max_size=128 \
    +data.n_recycle_failure=1 \
    +data.retain_hard_in_buffer=False \
    +data.retain_accuracy_low=0.50 \
    +data.retain_accuracy_high=0.75 \
    +data.max_recycle_count=3 \
    +data.use_summarize=True \
    +data.summarize_prompts_key=summarize_prompts \
    +data.max_summarize_prompts=8 \
    +data.max_summarize_length=10240 \
    +data.sr_response_key=sr_response \
    +data.extra_step_start_after=0 \
    +data.collect_accuracy_threshold=0.5 \
    +data.collect_accuracy_low=0.1 \
    reward_model.reward_manager='math' \
    +se_model.enable=False \
    +actor_rollout_se.model.path=$MODEL_PATH \
    +actor_rollout_ref.rollout.sr_use_offline=True \
    +actor_rollout_ref.rollout.sr_logprob_prompt=$SR_LOGPROB_PROMPT \
    +actor_rollout_ref.rollout.summarize_replace=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25600 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.policy_loss.rephrase_kl_coef=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$name \
    +actor_rollout_ref.actor.policy_loss.recycle_loss_mode=$off_policy_strategy \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    +actor_rollout_ref.actor.policy_loss.off_policy_masking=True \
    +actor_rollout_ref.actor.policy_loss.off_policy_reshape=${off_policy_reshape} \
    +actor_rollout_ref.actor.use_sft_prefix_reward=False \
    +actor_rollout_ref.actor.use_off_policy_loss=True \
    +actor_rollout_ref.actor.off_policy_normalize=False \
    +actor_rollout_ref.actor.off_policy_strategy=$name \
    +actor_rollout_ref.actor.off_policy_loss_impl=token \
    +actor_rollout_ref.actor.off_policy_max_clip=-1 \
    +actor_rollout_ref.actor.off_policy_min_clip=-1 \
    +actor_rollout_ref.actor.all_max_clip=10 \
    +actor_rollout_ref.actor.use_off_policy_probs=False \
    +actor_rollout_ref.actor.loss_remove_token_mean=True \
    +actor_rollout_ref.actor.loss_remove_clip=False \
    +actor_rollout_ref.actor.on_loss_remove_clip=False \
    +actor_rollout_ref.actor.off_loss_remove_clip=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.max_num_batched_tokens=63568 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.extra_temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.prefix_mode=summarize \
    +actor_rollout_ref.rollout.se_top_k=-1 \
    +actor_rollout_ref.rollout.se_top_p=1 \
    +actor_rollout_ref.rollout.n_val=1 \
    +actor_rollout_ref.rollout.max_prefix_len=10240 \
    +actor_rollout_ref.rollout.n_off=0 \
    +actor_rollout_ref.rollout.n_prefix=8 \
    +actor_rollout_ref.rollout.n_se=0 \
    +actor_rollout_ref.rollout.prefix_ratio=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    +trainer.log_prob_dir=$LOG_DIR/log_probs \
    +trainer.save_tensors_dir=$LOG_DIR/save_tensors \
    +trainer.metrics_data_dir=$LOG_DIR \
    trainer.default_local_dir=$LOG_DIR/ckpt \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 $@ 2>&1 | tee ${LOG_PATH}
