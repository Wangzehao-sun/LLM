set -x
#!/usr/bin/env bash
# GPU selection. The server has 8 GPUs; expose 4 by default.
# Override with, for example: GPU_DEVICES=4,5,6,7 bash train_sft.sh
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export WANDB_MODE=offline
# NOTE: no RAY_DEDUP_LOGS here -- unlike the RL scripts, SFT does not use Ray.

# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT) launcher.
#
# Pipeline differs entirely from the RL trainers (no Ray / rollout / reward):
#   torchrun -> verl.trainer.fsdp_sft_trainer  (FSDP, plain next-token CE loss).
#
# DATA: needs a single `messages` column ([system, user, assistant]);
# MultiTurnSFTDataset masks the loss to the assistant turn only. Two producers:
#
#   # rephraser SFT -- target is the student's OWN correct rollout
#   python Data/prepare_rephraser_sft.py \
#       --rollout $HOME/LLM/Data/rollout_data/80.jsonl \
#       --parquet $HOME/LLM/Data/deepmath_hard_solonly_split_summarize_teacher.parquet \
#       --output  $HOME/LLM/Data/deepmath_hard_rephraser_sft.parquet
#
#   # plain SFT -- merges this repo's prompt=[system,user] + target=[assistant]
#   python Data/prepare_sft.py --in <parquet> --out <parquet>_sft.parquet
#
# VALIDATION: trainer.test_freq only computes teacher-forcing `val/loss` -- this
# pipeline has no generation, so it CANNOT report rollout accuracy. For accuracy,
# evaluate the saved checkpoints offline with verl.trainer.main_generation (see
# examples/custom/generation_deepmath.sh); save_per_epoch below gives one
# checkpoint per epoch to sweep. Set TEST_FREQ=-1 to skip val/loss entirely --
# each validation pass walks the whole val set at micro-batch granularity.
#
# Usage:
#   bash train_sft.sh [model_subdir] [extra hydra overrides...]
#   bash train_sft.sh Qwen3-4b-base
#   GPU_DEVICES=4,5,6,7 EPOCHS=5 bash train_sft.sh Qwen3-4b-base optim.lr=2e-6
# ---------------------------------------------------------------------------

WORKER_DIR=${WORKER_DIR:-/home/zhwang}
MODEL_DIR=${MODEL_DIR:-/home/data/shared}
CODE_DIR=${CODE_DIR:-$WORKER_DIR/LLM/Myverl}   # dir from which `verl` imports

# Experiment knobs.
name="sft"
suffix="rephraser"

# Data paths. train/val must both carry the `messages` column.
train_path=${TRAIN_PATH:-$WORKER_DIR/LLM/Data/sft/self_rollouts_summarize_sft.parquet}
val_path=${VAL_PATH:-$train_path}

# Model path.
MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4b-base"}

PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME=${EXP_NAME:-"sft_0807"}
LOG_ROOT=${LOG_ROOT:-/home/data/zhwang_logs/sft_4b}
LOG_DIR=${LOG_ROOT}/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")

# Training knobs. TRAIN_BSZ is the GLOBAL batch. The trainer asserts
# TRAIN_BSZ % GPU_NUM == 0 and then (TRAIN_BSZ / GPU_NUM) % MICRO_BSZ == 0;
# grad accumulation is TRAIN_BSZ / (MICRO_BSZ * GPU_NUM).
#
# WATCH THE STEP COUNT on small sets. Both dataloaders use drop_last=True, so
# steps/epoch = floor(rows / TRAIN_BSZ) -- a 92-row rephraser set with
# TRAIN_BSZ=64 gives 1 step/epoch (3 steps total at EPOCHS=3), and any
# TRAIN_BSZ > rows silently trains on NOTHING. Keep TRAIN_BSZ well below the row
# count: 16 over 92 rows is 5 steps/epoch. The launcher prints both below.
TRAIN_BSZ=${TRAIN_BSZ:-64}
MICRO_BSZ=${MICRO_BSZ:-1}
MAX_LENGTH=${MAX_LENGTH:-16384}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}
SAVE_FREQ=${SAVE_FREQ:--1}       # -1 = rely on save_per_epoch below
TEST_FREQ=${TEST_FREQ:-200}      # val/loss only; -1 disables

cd "$CODE_DIR" || exit 1
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi

# Preflight: report rows and the resulting step count. Both dataloaders use
# drop_last=True, so floor(rows / TRAIN_BSZ) == 0 means the run would train on
# nothing at all -- fail loudly here instead of after model load.
python - "$train_path" "$val_path" "$TRAIN_BSZ" "$EPOCHS" <<'PYEOF' || exit 1
import sys
import pandas as pd

train_path, val_path, bsz, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for tag, path in (("train", train_path), ("val", val_path)):
    df = pd.read_parquet(path, columns=None)
    if "messages" not in df.columns:
        sys.exit(f"[preflight] {tag} parquet {path} has no 'messages' column "
                 f"(got {list(df.columns)}); build it with Data/prepare_rephraser_sft.py "
                 f"or Data/prepare_sft.py")
    print(f"[preflight] {tag}: {len(df)} rows  {path}")
    if tag == "train":
        spe = len(df) // bsz
        print(f"[preflight] train_batch_size={bsz} -> {spe} steps/epoch, "
              f"{spe * epochs} steps over {epochs} epoch(s)")
        if spe == 0:
            sys.exit(f"[preflight] train_batch_size={bsz} exceeds the {len(df)} available rows; "
                     f"drop_last=True would yield 0 steps. Lower TRAIN_BSZ.")
PYEOF

# Train over a single node using the GPUs exposed by *_VISIBLE_DEVICES.
torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_NUM \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$train_path" \
    data.val_files="$val_path" \
    data.multiturn.enable=True \
    data.multiturn.messages_key=messages \
    data.max_length=$MAX_LENGTH \
    data.truncation=right \
    data.train_batch_size=$TRAIN_BSZ \
    data.micro_batch_size_per_gpu=$MICRO_BSZ \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.fsdp_config.cpu_offload=False \
    model.trust_remote_code=True \
    use_remove_padding=True \
    ulysses_sequence_parallel_size=1 \
    optim.lr=$LR \
    optim.lr_scheduler=cosine \
    optim.warmup_steps_ratio=0.03 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    +trainer.save_per_epoch=True \
    trainer.default_local_dir=$LOG_DIR/ckpt \
    trainer.default_hdfs_dir=null $@ 2>&1 | tee ${LOG_PATH}
