#!/usr/bin/env bash
# ============================================================================
# Runnable launcher for the UPSTREAM Prefix-RFT recipe
#   github.com/ZeroYuHuang/prefix_rft  (arXiv:2507.01679)
#
# WHY THIS FILE EXISTS. Upstream's own run_single_node.sh is:
#
#     cd $HOME/train-verl-updated
#     python3 -m recipe.prefix_rft.main $TRAIN_CONFIG
#
# and $TRAIN_CONFIG is never defined anywhere in that repository. Its
# config/prefix_rft_trainer.yaml is unmodified stock verl -- none of the prefix keys the
# recipe reads are in it. So the overrides below are not tuning: they are the MISSING HALF
# of the config, without which the run dies in the dataset constructor.
#
# Every value is annotated with where it came from:
#   [code]  a default read straight out of the recipe's own source
#   [paper] stated in arXiv:2507.01679
#   [ours]  chosen here, because upstream left no value to copy
#
# HOW THE METHOD WORKS (recipe/prefix_rft/rl_dataset.py:335-380). Per question, the dataset
# builds NUM_PREFIX rollout prompts. NUM_EMPTY_PREFIX of them get ratio 0 (pure on-policy);
# for the rest it draws
#
#     ratio ~ Beta(alpha, beta) mapped into [low_ctrl(step), high_ctrl(step)]
#
# takes that fraction of a RANDOMLY CHOSEN demo's tokens, and appends it as an UNFINISHED
# assistant turn (`continue_final_message=True`). So the model continues a partial
# derivation rather than being shown one. The two window bounds are separately scheduled
# controllers, which is what makes the prefix shrink as training proceeds.
#
# Usage (run from the prefix_rft repo root, NOT from this repo):
#     bash /path/to/run_prefix_rft.sh
#     MODEL_PATH=... TRAIN_PARQUET=... bash /path/to/run_prefix_rft.sh
# ============================================================================
set -e

REPO_DIR=${REPO_DIR:-/tmp/repro/prefix_rft}
cd "$REPO_DIR" || { echo "REPO_DIR not found: $REPO_DIR" >&2; exit 1; }
[ -f recipe/prefix_rft/main.py ] || { echo "not the prefix_rft repo: $REPO_DIR" >&2; exit 1; }

export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}

GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")

# ---- model + data ---------------------------------------------------------
WORKER_DIR=${WORKER_DIR:-/apdcephfs_qy3/share_301372554/share_info/zenohaowang}
MODEL_PATH=${MODEL_PATH:-$WORKER_DIR/Model/Qwen3-4B-Base}
# Built by Data/prepare_prefix_rft.py -- needs the `demos` / `demos_corr` columns.
DATA_DIR=${DATA_DIR:-/tmp/repro/prefix_rft_data}
TRAIN_PARQUET=${TRAIN_PARQUET:-$DATA_DIR/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-$DATA_DIR/val.parquet}

for f in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
  [ -f "$f" ] || { echo "missing data: $f -- run Data/prepare_prefix_rft.py first" >&2; exit 1; }
done

EXP_NAME=${EXP_NAME:-prefix_rft_repro}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs/$EXP_NAME}
mkdir -p "$LOG_DIR"

# ---- the prefix window ----------------------------------------------------
# Beta(1,1) is uniform on the window: every prefix length in [low, high] equally likely.
# [code] BetaSampler's own defaults are alpha=beta=1.0.
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-1.0}

# The window's UPPER bound decays 0.9 -> 0.0 over N steps (cosine), so late in training the
# window collapses toward pure on-policy. [ours] -- upstream ships no values; cosine_decay is
# the controller its CTRL_MAPPING registers and whose docstring describes this shape.
HIGH_INIT=${HIGH_INIT:-0.9}
HIGH_TARGET=${HIGH_TARGET:-0.0}
# The LOWER bound stays at 0, so a short prefix is always reachable. [ours]
LOW_CONST=${LOW_CONST:-0.0}
# Decay horizon in optimizer steps. Set it near your total step count or the window will
# reach 0 long before training ends. [ours]
CTRL_STEPS=${CTRL_STEPS:-500}

# Rollouts per question, and how many of them are pure on-policy (ratio 0).
# n = NUM_PREFIX is the GRPO group size on this path: the dataset emits input_ids_0..
# input_ids_{n-1} and each is one rollout. [code] ray_trainer reads data.num_prefix.
NUM_PREFIX=${NUM_PREFIX:-8}
NUM_EMPTY_PREFIX=${NUM_EMPTY_PREFIX:-4}   # [paper] half prefixed / half not
MIN_PREFIX_LEN=${MIN_PREFIX_LEN:-16}      # [code] floor applied "for training stability"
TOTAL_DEMO_N=${TOTAL_DEMO_N:-1}           # our parquet has exactly 1 demo per question
DEMO_RATIO=${DEMO_RATIO:-1.0}             # [code] default: every row keeps its demo

MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}
# The prefix lives inside the PROMPT (it is an unfinished assistant turn), so a long prefix
# needs prompt room. Upstream truncates the prefix when prompt+prefix overflows -- and
# asserts prefix_len > min_prefix_len afterwards, so too small a budget aborts the run.
MAX_PREFIX_LEN=${MAX_PREFIX_LEN:-8192}

TRAIN_BSZ=${TRAIN_BSZ:-128}
MINI_BSZ=${MINI_BSZ:-64}
MICRO_BSZ_PER_GPU=${MICRO_BSZ_PER_GPU:-1}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-24576}
TP=${TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}

echo "=== repo        : $REPO_DIR"
echo "=== model       : $MODEL_PATH"
echo "=== data        : $TRAIN_PARQUET"
echo "=== GPUs        : $GPU_DEVICES ($GPU_NUM)"
echo "=== prefix      : Beta($ALPHA,$BETA) in [$LOW_CONST, cosine $HIGH_INIT->$HIGH_TARGET over $CTRL_STEPS]"
echo "=== rollouts    : $NUM_PREFIX/question, $NUM_EMPTY_PREFIX of them empty-prefix"

python3 -m recipe.prefix_rft.main \
    --config-name=prefix_rft_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="['$TRAIN_PARQUET']" \
    data.val_files="['$VAL_PARQUET']" \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    +data.seed=42 \
    `# ---- the 15 keys the recipe reads by ATTRIBUTE, so a missing one raises ----` \
    +data.num_prefix=$NUM_PREFIX \
    +data.num_empty_prefix=$NUM_EMPTY_PREFIX \
    +data.min_prefix_len=$MIN_PREFIX_LEN \
    +data.max_prefix_len=$MAX_PREFIX_LEN \
    +data.total_demo_n=$TOTAL_DEMO_N \
    +data.demo_ratio=$DEMO_RATIO \
    +data.demo_key=demos \
    +data.demo_corr_key=demos_corr \
    +data.only_keep_dp_with_demo=True \
    `# lower bound of the Beta window: constant 0` \
    +data.prefix_low_ctrl_type=const \
    +data.prefix_low_ctrl.kwargs.init=$LOW_CONST \
    `# upper bound: cosine decay, so the window shrinks as training proceeds` \
    +data.prefix_high_ctrl_type=cosine_decay \
    +data.prefix_high_ctrl.kwargs.init=$HIGH_INIT \
    +data.prefix_high_ctrl.kwargs.target=$HIGH_TARGET \
    +data.prefix_high_ctrl.kwargs.n_steps=$CTRL_STEPS \
    +data.prefix_high_ctrl.kwargs.warmup_ratio=0.0 \
    +data.prefix_sampler.kwargs.alpha=$ALPHA \
    +data.prefix_sampler.kwargs.beta=$BETA \
    `# CTRL_WRAPPER is optional (.get), but prefix_*_ctrl_wrapper itself is attribute-read` \
    `# inside the wrapper branch only -- pass empty kwargs so the branch stays skippable.` \
    +data.prefix_low_ctrl_wrapper.kwargs.delay_steps=0 \
    +data.prefix_high_ctrl_wrapper.kwargs.delay_steps=0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    `# ---- actor-side keys the recipe reads by attribute but the yaml lacks ----` \
    `# enable_clip=False routes PREFIX tokens to the unclipped off-policy loss` \
    `# (core_algos.py:634) -- that split IS the method, so it must be False.` \
    +actor_rollout_ref.actor.enable_clip=False \
    +actor_rollout_ref.actor.entropy_mode=response_only \
    +actor_rollout_ref.actor.loss_agg_max_tokens=$MAX_RESPONSE_LEN \
    `# 'identity' = no advantage reshaping. .get() returns None otherwise and the` \
    `# constructor then calls .split() on it -> AttributeError before step 1.` \
    +actor_rollout_ref.actor.off_adv_reshaper=identity \
    +actor_rollout_ref.actor.off_ent_mask_ratio_ctrl_type=const \
    +actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.init=0.0 \
    +actor_rollout_ref.actor.reshape_kwargs.placeholder=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    `# read at ray_trainer.py:1114 as truncate_response_len; absent from the yaml` \
    +actor_rollout_ref.rollout.truncate_length=$MAX_RESPONSE_LEN \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=prefix_rft \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$LOG_DIR/ckpt \
    "$@" 2>&1 | tee "$LOG_DIR/train.log"
