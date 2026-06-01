set -x

unset ROCR_VISIBLE_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

# ---------------------------------------------------------------------------
# Variant of train_grpo.sh that uses the explain-style summarize-then-continue
# off-policy mode. Two key differences from train_grpo.sh:
#   1. Rollout prompt = system + user(question + prefix + reason_prompt), pre-
#      rendered offline by Data/prepare_summarize_prompts.py and shipped via the
#      `summarize_prompts` column in the parquet.
#   2. Loss prompt = original question prompt only. Trainer hands the actor an
#      input_ids of [original_short_prompt, model_output] so log_prob is
#      computed under the short prompt — this is what makes it explain-style.
#      IS ratio = exp(log_prob_under_short - rollout_log_prob_under_long)
#      automatically does the prompt-shift correction.
#
# Required upstream:
#   1. Data/prepare_deepmath.py     -> deepmath.parquet
#   2. Data/filter_deepmath.py      -> deepmath_dgt6_n10000.parquet
#   3. Data/add_token_split_points.py -> deepmath_dgt6_n10000_split.parquet
#   4. Data/prepare_summarize_prompts.py -> deepmath_dgt6_n10000_summarize.parquet  <-- this is what we train on
# ---------------------------------------------------------------------------

train_path=$HOME/LLM/Data/deepmath_dgt6_n10000_summarize.parquet
test_path=$HOME/LLM/Train/data/valid_with_aime25_new.parquet
test1_path=$HOME/LLM/Train/data/split_by_source_new/aime.parquet
test2_path=$HOME/LLM/Train/data/split_by_source_new/aime25.parquet
test3_path=$HOME/LLM/Train/data/split_by_source_new/amc.parquet
test4_path=$HOME/LLM/Train/data/split_by_source_new/olympiad_bench.parquet
train_files="['$train_path']"
val_files="['$test_path']"

name="vanilla"

MODEL_DIR=/home/shared
MODEL_PATH=$MODEL_DIR/${1:-"Qwen2.5-Math-7B-16k-think"}

suffix="summarize_explain"
PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME="training"
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

GPU_NUM=4
TENSOR_PARALLEL=1

DATA_DIR=$HOME/LLM/Train/data/

cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi
# Train over a single node, 4 A100-80GB GPUs.
python -m verl.trainer.main_ppo_new \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_reward=False \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=10240 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.shuffle=False \
    +data.reward_impl_version=4 \
    +data.filter_targets=False \
    +data.use_se=False \
    +data.use_summarize=True \
    +data.summarize_prompts_key=summarize_prompts \
    +data.max_summarize_prompts=8 \
    +data.max_summarize_length=8192 \
    +data.collect_failures=True \
    +data.failure_buffer_max_size=128 \
    +data.n_recycle_failure=3 \
    reward_model.reward_manager='math' \
    +se_model.enable=False \
    +actor_rollout_se.model.path=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$name \
    +actor_rollout_ref.actor.policy_loss.recycle_loss_mode='luffy' \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    +actor_rollout_ref.actor.policy_loss.off_policy_masking=True \
    +actor_rollout_ref.actor.policy_loss.off_policy_reshape="vanilla" \
    +actor_rollout_ref.actor.use_sft_prefix_reward=False \
    +actor_rollout_ref.actor.use_off_policy_loss=True \
    +actor_rollout_ref.actor.off_policy_normalize=False \
    +actor_rollout_ref.actor.off_policy_strategy=$name \
    +actor_rollout_ref.actor.off_policy_loss_impl=token \
    +actor_rollout_ref.actor.off_policy_max_clip=1.28 \
    +actor_rollout_ref.actor.off_policy_min_clip=-1 \
    +actor_rollout_ref.actor.all_max_clip=10 \
    +actor_rollout_ref.actor.use_off_policy_probs=False \
    +actor_rollout_ref.actor.loss_remove_token_mean=True \
    +actor_rollout_ref.actor.loss_remove_clip=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.max_num_batched_tokens=24576 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.prompt_length=8192 \
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
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    +trainer.log_prob_dir=$LOG_DIR/log_probs \
    +trainer.save_tensors_dir=$LOG_DIR/save_tensors \
    +trainer.metrics_data_dir=$LOG_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=5 $@ 2>&1 | tee ${LOG_PATH}
