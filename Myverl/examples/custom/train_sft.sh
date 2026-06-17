set -x

unset ROCR_VISIBLE_DEVICES
echo $HOME
export WANDB_MODE=offline

# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT) launcher, adapted to this repo's data layout.
#
# Pipeline differs entirely from the RL trainer (no Ray / rollout / reward):
#   torchrun -> verl.trainer.fsdp_sft_trainer  (FSDP, plain next-token CE loss).
#
# Data: this repo's parquet stores the conversation split across two columns,
#   prompt = [system, user]  and  target = [assistant].
# MultiTurnSFTDataset needs a single `messages` column ([system,user,assistant])
# and masks the loss to assistant tokens only. So first merge the columns:
#
#   python Data/prepare_sft.py \
#       --in  Data/deepmath_dgt6_n10000.parquet \
#       --out Data/deepmath_dgt6_n10000_sft.parquet
#
# Usage:
#   bash train_sft.sh <nproc_per_node> [model_subdir] [extra hydra overrides...]
#   e.g. bash train_sft.sh 4 Qwen2.5-Math-7B-16k-think
# ---------------------------------------------------------------------------

nproc_per_node=${1:-4}
if [ -n "$1" ]; then shift; fi

MODEL_DIR=/home/shared
MODEL_PATH=$MODEL_DIR/${1:-"Qwen2.5-Math-7B-16k-think"}
if [ -n "$1" ]; then shift; fi

train_path=$HOME/LLM/Data/deepmath_dgt6_n10000_sft.parquet
val_path=$HOME/LLM/Data/deepmath_dgt6_n1000_sft.parquet
train_files="['$train_path']"
val_files="['$val_path']"

PROJECT_NAME="sft_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME="sft"
SAVE_DIR=$HOME/LLM/Train/verl/checkpoints/${PROJECT_NAME}
LOG_DIR=$HOME/LLM/Train/verl/logs/${PROJECT_NAME}
mkdir -p ${LOG_DIR} ${SAVE_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

cd $HOME/LLM/Train/verl/
echo "change to dir: $PWD"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.multiturn.enable=True \
    data.multiturn.messages_key=messages \
    data.max_length=16384 \
    data.truncation=right \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.fsdp_config.cpu_offload=False \
    use_remove_padding=True \
    ulysses_sequence_parallel_size=1 \
    optim.lr=1e-6 \
    optim.lr_scheduler=cosine \
    optim.warmup_steps_ratio=0.03 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.total_epochs=3 \
    trainer.logger=['console','tensorboard'] \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    +trainer.save_per_epoch=True \
    trainer.test_freq=200 \
    trainer.default_hdfs_dir=null $@ 2>&1 | tee ${LOG_PATH}
