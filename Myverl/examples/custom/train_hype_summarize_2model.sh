set -x
#!/usr/bin/env bash
# GPU selection. The server has 8 GPUs; expose 4 by default.
# Override with, for example: GPU_DEVICES=4,5,6,7 bash train_hype_summarize_2model.sh
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

# ---------------------------------------------------------------------------
# Two-model variant of train_hype_summarize.sh: the rephraser is a SEPARATE,
# FROZEN model. train_hype_summarize.sh is left untouched as the single-model
# A/B baseline.
#
# Who does what:
#   reasoner  ($MODEL_PATH)      -- rolls out the short question prompt, and is the
#                                   ONLY model that receives gradient updates
#   rephraser ($REPHRASER_PATH)  -- rolls out the long summarize prompt to produce
#                                   the rewrite candidates, and supplies the proposal
#                                   density for the importance ratio. Never updated:
#                                   no optimizer is built for it, and update_actor /
#                                   save_checkpoint assert _is_actor, which it is not.
#
# The math is unchanged from the single-model case, only reinterpreted. Candidates are
# sampled under the long prompt while the loss is computed under the short one, so
#
#     off_ratio = pi_theta(y|x_short) / pi_phi(y|x_long)
#
# which is a valid importance weight -- now correcting for a prompt shift AND a model
# shift. What this buys: the rewrites come from a model actually trained to rewrite,
# instead of from the reasoner pretending to.
#
# WATCH actor/off_ratio_ess FROM STEP 1. Two densities that no longer share weights
# drift apart per token, and over a ~10k-token sequence the weight degrades
# exponentially. If ESS collapses (or off_ratio_max_clip_frac saturates), switch
# off_policy_reshape to batch_mean_norm before drawing any conclusion about the method.
# Also compare batch/long_prob_off_standard (phi, long prompt) against
# batch/old_prob_off_standard (theta, short prompt): that gap IS the combined shift.
#
# Run examples/custom/smoke_dual_vllm.py FIRST. Two sleeping vLLM engines in one
# process is the one assumption here that is not verified by any existing code path.
#
# Usage:
#   REPHRASER_PATH=/home/data/shared/<sft-rephraser> bash train_hype_summarize_2model.sh
#   REPHRASER_PATH=... bash train_hype_summarize_2model.sh Qwen3-4b-base
# ---------------------------------------------------------------------------

WORKER_DIR=/home/zhwang
MODEL_DIR=/home/data/shared # /home/data/shared /home/zhwang/LLM/Myverl
# Experiment knobs.
name="rl-rl"
off_policy_strategy="rl-rl"

# Must stay in {clip, batch_mean_norm, group_ess_weight}. dynamic_clip and vanilla skip
# the off_old_log_probs -> old_log_probs swap, and p_div_p_0.1 overwrites off_ratio
# outright -- either way the rephraser's proposal density never reaches the loss and the
# ratio degenerates to ~1. The trainer refuses to start on those.
off_policy_reshape="clip"
suffix="hype_summarize_2model_"${off_policy_strategy}"_"${off_policy_reshape}

# Data paths.
train_path=$WORKER_DIR/LLM/Data/deepmath/deepmath_hard_thinkonly_split_summarize_new3.parquet
test_path=$WORKER_DIR/LLM/Data/valid_with_aime25_new.parquet
test1_path=$WORKER_DIR/LLM/Data/split_by_source_new/aime.parquet
test2_path=$WORKER_DIR/LLM/Data/split_by_source_new/aime25.parquet
test3_path=$WORKER_DIR/LLM/Data/split_by_source/amc_nothink.parquet
test4_path=$WORKER_DIR/LLM/Data/split_by_source_new/olympiad_bench.parquet
train_files="['$train_path']"
val_files="['$test3_path']"

# Reasoner (trained). First positional arg picks the subdir, as in every other script.
MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4b-base"}

# Rephraser (frozen). Point this at the SFT checkpoint from train_sft.sh, e.g.
#   /home/data/zhwang_logs/sft_4b/<project>/<exp>/ckpt/global_step_N
# MUST share the reasoner's tokenizer family: summarize_input_ids are pre-tokenized
# offline with the reasoner's tokenizer, and raw token ids from this model's rollout are
# spliced straight into the reasoner's input_ids with no re-tokenization. A different
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

PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME=${EXP_NAME:-"2model_$(basename $REPHRASER_PATH)"}
LOG_DIR=/home/data/zhwang_logs/train_4b/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")
TENSOR_PARALLEL=1

# Two vLLM engines share the same GPUs, so each gets roughly half of what the
# single-model script (0.7) uses. Lower both if the KV cache cannot be allocated.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.42}

cd "$MODEL_DIR" || exit 1
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi

echo "=== reasoner  (trained): $MODEL_PATH"
echo "=== rephraser (frozen) : $REPHRASER_PATH"
echo "=== gpu_memory_utilization: $GPU_MEM_UTIL per engine, 2 engines on $GPU_NUM GPU(s)"

# Train over a single node using the GPUs exposed by *_VISIBLE_DEVICES.
python -m verl.trainer.main_ppo_new \
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
    +data.extra_step_start_after=0 \
    +data.collect_accuracy_threshold=0.5 \
    +data.collect_accuracy_low=0.1 \
    reward_model.reward_manager='math' \
    +se_model.enable=False \
    +actor_rollout_se.model.path=$MODEL_PATH \
    rephraser.enable=True \
    +actor_rollout_rephraser.model.path=$REPHRASER_PATH \
    +actor_rollout_rephraser.actor.fsdp_config.param_offload=True \
    +actor_rollout_rephraser.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    +actor_rollout_ref.rollout.summarize_replace=True \
    +actor_rollout_ref.rollout.summarize_replace_k=8 \
    +actor_rollout_ref.rollout.summarize_replace_select=shortest \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
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
