set -x
#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Two-model rollout with the rephraser on its OWN GPU pool.
#
# This is the configuration that makes online two-model rollout possible at all.
# The blocker was never memory -- it was that vLLM's sleep mode is built on a
# PROCESS-GLOBAL CuMemAllocator whose tags are the hardcoded strings
# "weights"/"kv_cache", so two engines in one process free each other's memory.
# (SGLang has the identical problem via torch_memory_saver, and
# change_current_allocator can only be called once per process, so switching
# backends does not help.) verl puts every worker of a resource POOL into one Ray
# actor process, so the fix is a second pool -- which is also what vLLM upstream
# recommends: one engine per process.
#
# Consequence: the rephraser gets its own engine, so candidates are generated
# ONLINE. No offline pre-generation step, and val_summarize/* comes back and
# genuinely measures the rephraser.
#
#   reasoner  ($MODEL_PATH, N_REASONER_GPUS)  -- rolls out the short question
#       prompt and is the ONLY model that receives gradient updates.
#   rephraser ($REPHRASER_PATH, REPHRASER_GPUS) -- rolls out the long summarize
#       prompt to produce the rewrite candidates AND computes their log-probs.
#       Never updated: no optimizer is built for it, and update_actor /
#       save_checkpoint both `assert self._is_actor`, which it is not.
#
# GPU accounting -- the one thing that is easy to get wrong:
#   CUDA_VISIBLE_DEVICES must expose the GPUs of BOTH pools (set it via
#   GPU_DEVICES below), while trainer.n_gpus_per_node counts the REASONER's only.
#   The rephraser's share is passed separately as rephraser.n_gpus_per_node.
#   Ray's _check_resource_available validates each pool and fails clearly.
#
# The math is unchanged, only reinterpreted. Candidates are sampled under the long
# prompt while the loss is computed under the short one, so
#
#     off_ratio = pi_theta(y|x_short) / pi_phi(y|x_long)
#
# stays a valid importance weight -- now correcting a prompt shift AND a model
# shift. It holds only because the numerator and denominator come from the models
# that actually played those roles; see SR_LOGPROB_PROMPT below.
#
# WATCH actor/off_ratio_ess FROM STEP 1. theta and phi no longer share weights, so
# the per-token ratio drifts systematically and over ~10k tokens the sequence
# weight degrades exponentially. If ESS collapses (or off_ratio_max_clip_frac
# saturates), switch off_policy_reshape to batch_mean_norm BEFORE concluding
# anything about the method. Also compare batch/long_prob_off_standard (phi, long
# prompt) against batch/old_prob_off_standard (theta, short prompt): that gap IS
# the combined shift.
#
# Usage:
#   REPHRASER_PATH=/home/data/shared/<sft-rephraser> \
#       bash train_hype_summarize_2model_pool.sh Qwen3-4b-base
#
#   # 4 reasoner GPUs + 2 rephraser GPUs; GPU_DEVICES lists all six
#   GPU_DEVICES=0,1,2,3,4,5 REPHRASER_GPUS=2 REPHRASER_PATH=... \
#       bash train_hype_summarize_2model_pool.sh
#
# For the shared-pool variant (no second engine, candidates pre-generated offline)
# see train_hype_summarize_sr_offline.sh, which is kept as the A/B comparison.
# ---------------------------------------------------------------------------

# ALL GPUs across BOTH pools. The reasoner takes the first N_REASONER_GPUS of
# them and the rephraser the rest; Ray does the actual placement.
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

WORKER_DIR=/home/zhwang
MODEL_DIR=/home/data/shared
name="rl-rl"
off_policy_strategy="rl-rl"

# Must consume old_log_probs for the importance ratio to survive: dynamic_clip and
# vanilla skip the off_old_log_probs -> old_log_probs swap, and p_div_p_0.1
# overwrites off_ratio outright. Either way the proposal density never reaches the
# loss and the ratio degenerates to ~1. The trainer refuses to start on those.
off_policy_reshape="clip"
suffix="hype_summarize_2pool_"${off_policy_strategy}"_"${off_policy_reshape}

# Data. Needs the pre-rendered summarize_prompts column
# (Data/prepare_summarize_prompts.py) -- the long prompt is what the candidates are
# generated from. NO sr_response column needed: generation is online here.
train_path=${TRAIN_PATH:-$WORKER_DIR/LLM/Data/deepmath/deepmath_hard_thinkonly_split_summarize_new3.parquet}
test3_path=$WORKER_DIR/LLM/Data/split_by_source/amc_nothink.parquet
train_files="['$train_path']"
val_files="['$test3_path']"

MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4b-base"}

# The frozen rephraser -- typically the SFT checkpoint from train_sft.sh.
# MUST share the reasoner's tokenizer family: summarize_input_ids are pre-tokenized
# offline with the reasoner's tokenizer, and the candidate token ids are spliced
# straight into the reasoner's input_ids without re-tokenizing. A different
# vocabulary corrupts the data silently -- no error, just wrong training.
REPHRASER_PATH=${REPHRASER_PATH:-}
if [ -z "$REPHRASER_PATH" ]; then
    echo "set REPHRASER_PATH to the frozen rephraser (e.g. an SFT checkpoint dir)" >&2
    echo "  REPHRASER_PATH=/home/data/shared/<sft-rephraser> bash $0" >&2
    exit 1
fi
if [ ! -d "$REPHRASER_PATH" ]; then
    echo "REPHRASER_PATH is not a directory: $REPHRASER_PATH" >&2
    exit 1
fi

TOTAL_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")
REPHRASER_GPUS=${REPHRASER_GPUS:-1}
# The reasoner keeps whatever is left. Override N_REASONER_GPUS to hold the
# reasoner's count fixed and add the rephraser's GPUs on top instead.
N_REASONER_GPUS=${N_REASONER_GPUS:-$((TOTAL_GPUS - REPHRASER_GPUS))}

if [ "$N_REASONER_GPUS" -lt 1 ]; then
    echo "no GPUs left for the reasoner: $TOTAL_GPUS visible - $REPHRASER_GPUS for the rephraser" >&2
    exit 1
fi
if [ $((N_REASONER_GPUS + REPHRASER_GPUS)) -gt "$TOTAL_GPUS" ]; then
    echo "pools want $((N_REASONER_GPUS + REPHRASER_GPUS)) GPUs but only $TOTAL_GPUS are visible" >&2
    echo "  expose more via GPU_DEVICES, or lower REPHRASER_GPUS / N_REASONER_GPUS" >&2
    exit 1
fi

# train_batch_size * rollout.n must be divisible by the REASONER's GPU count
# (_validate_config). The rephraser's batches are padded to its own world_size in
# the trainer, so its count is unconstrained.
TRAIN_BSZ=${TRAIN_BSZ:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
if [ $((TRAIN_BSZ * ROLLOUT_N % N_REASONER_GPUS)) -ne 0 ]; then
    echo "train_batch_size*rollout.n ($((TRAIN_BSZ * ROLLOUT_N))) is not divisible by the" >&2
    echo "  reasoner's $N_REASONER_GPUS GPU(s); adjust TRAIN_BSZ or the split" >&2
    exit 1
fi

PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME=${EXP_NAME:-"2pool_$(basename $REPHRASER_PATH)"}
LOG_DIR=/home/data/zhwang_logs/train_4b/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

TENSOR_PARALLEL=1

# Each model owns its GPUs outright, so neither has to leave room for the other --
# no reason to shrink these the way a shared pool would require.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}
REPHRASER_GPU_MEM_UTIL=${REPHRASER_GPU_MEM_UTIL:-0.85}

# Which prompt the importance-ratio DENOMINATOR is computed under. Two different
# estimators, not a tuning knob:
#   long  (default) -- q = pi_phi(y|x_long), the distribution the candidates
#                      actually came from. The only mathematically correct proposal
#                      density, so off_ratio is a consistent policy gradient. Risk
#                      is variance; watch actor/off_ratio_ess.
#   short           -- q = pi_phi(y|x_short), the SAME prompt the on-policy rows
#                      use, so the ratio reflects only the model gap. Much lower
#                      variance, but it is not the density y was sampled from, so
#                      the estimator is biased.
SR_LOGPROB_PROMPT=${SR_LOGPROB_PROMPT:-long}
if [ "$SR_LOGPROB_PROMPT" != "long" ] && [ "$SR_LOGPROB_PROMPT" != "short" ]; then
    echo "SR_LOGPROB_PROMPT must be 'long' or 'short', got: $SR_LOGPROB_PROMPT" >&2
    exit 1
fi

# 'all' generates candidates for every question; 'wrong_only' scores the on-policy
# rollouts first and only rewrites the ones the reasoner got entirely wrong --
# cheaper, and the case where the candidate batch size is an arbitrary integer
# (hence the pad/unpad around the rephraser's dispatch).
SUMMARIZE_REPLACE=${SUMMARIZE_REPLACE:-True}

cd "$MODEL_DIR" || exit 1
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi

echo "=== visible GPUs        : $GPU_DEVICES ($TOTAL_GPUS total)"
echo "=== reasoner  (trained) : $MODEL_PATH  [$N_REASONER_GPUS GPU(s), mem_util=$GPU_MEM_UTIL]"
echo "=== rephraser (frozen)  : $REPHRASER_PATH  [$REPHRASER_GPUS GPU(s), mem_util=$REPHRASER_GPU_MEM_UTIL]"
echo "=== sr_logprob_prompt   : $SR_LOGPROB_PROMPT"

python -m verl.trainer.main_ppo_new \
    rephraser.enable=True \
    +rephraser.n_gpus_per_node=$REPHRASER_GPUS \
    +actor_rollout_rephraser.model.path=$REPHRASER_PATH \
    +actor_rollout_rephraser.rollout.gpu_memory_utilization=$REPHRASER_GPU_MEM_UTIL \
    +actor_rollout_rephraser.actor.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_reward=False \
    +algorithm.trajectory_filter.enable=True \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=$TRAIN_BSZ \
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
    +data.extra_step_start_after=0 \
    +data.collect_accuracy_threshold=0.5 \
    +data.collect_accuracy_low=0.1 \
    reward_model.reward_manager='math' \
    +se_model.enable=False \
    +actor_rollout_se.model.path=$MODEL_PATH \
    +actor_rollout_ref.rollout.sr_logprob_prompt=$SR_LOGPROB_PROMPT \
    +actor_rollout_ref.rollout.summarize_replace=$SUMMARIZE_REPLACE \
    +actor_rollout_ref.rollout.summarize_replace_k=8 \
    +actor_rollout_ref.rollout.summarize_replace_select=shortest \
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
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
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
    trainer.n_gpus_per_node=$N_REASONER_GPUS \
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
