#!/usr/bin/env bash
# PICO: GRPO + summarize-replacement with joint decoding (verl/PICO, entry verl.PICO.main_ppo_new).
#
# Same run as train_hype_summarize.sh, without the parameters PICO never reads.
#
# GPU selection. The server has 8 GPUs; expose 4 by default.
# Override with, for example: GPU_DEVICES=4,5,6,7 bash train_PICO.sh
set -x

CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME:-verl}

GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
export RAY_ADDRESS=local
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

# Experiment knobs.
loss_mode="rl-rl"               # on rows - off rows: rl-rl / rl-sft / sft-sft / none-sft
off_policy_reshape="clip_with_z" # rl off rows: clip_with_z / clip_without_z / ...
suffix="PICO_${loss_mode}_${off_policy_reshape}"

# Data paths.
WORKER_DIR=${WORKER_DIR:-/apdcephfs_qy3/share_301372554/share_info/zenohaowang}
train_path=$WORKER_DIR/LLM/Data/dapo_math/dapo_en_math_solution_9k_random_summarize_train.parquet
test_path=$WORKER_DIR/LLM/Data/test/aime25_nothink_repeat16.parquet
train_files="['$train_path']"
val_files="['$test_path']"

# Model path.
MODEL_DIR=$WORKER_DIR/Model
MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4B-Base"}
TEACHER_PATH=$MODEL_DIR/Qwen3-4B-Instruct-2507

PROJECT_NAME="train_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME="training_128_64_k5"
LOG_DIR=$WORKER_DIR/LLM/zhwang_logs/train_4b/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$LOG_DIR/tensorboard}

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")

cd "$MODEL_DIR" || exit 1
if [ -n "$1" ]; then
    shift
fi

python -m verl.PICO.main_ppo_new \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    +data.reward_impl_version=4 \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=10240 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.shuffle=False \
    +data.summarize_prompts_key=summarize_prompt \
    +data.max_summarize_length=2048 \
    +actor_rollout_ref.rollout.summarize_replace=wrong_only \
    +actor_rollout_ref.rollout.accuracy_threshold=0.1 \
    +actor_rollout_ref.rollout.summarize_replace_select=shortest \
    +algorithm.trajectory_filter.enable=False \
    +joint_decode.enable=True \
    +joint_decode.teacher_model_path=$TEACHER_PATH \
    +joint_decode.agree_dynamic_top_k=False \
    +joint_decode.agree_teacher_top_k=5 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51200 \
    actor_rollout_ref.actor.clip_ratio=0.28 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    +actor_rollout_ref.actor.policy_loss.off_policy_reshape=$off_policy_reshape \
    +actor_rollout_ref.actor.all_max_clip=10 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    +trainer.save_tensors_dir=$LOG_DIR/save_tensors \
    +trainer.metrics_data_dir=$LOG_DIR \
    trainer.default_local_dir=$LOG_DIR/ckpt \
    trainer.total_epochs=3 "$@" 2>&1 | tee ${LOG_PATH}
