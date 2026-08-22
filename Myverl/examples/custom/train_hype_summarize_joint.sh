set -x
#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SR candidates from JOINT DECODING: the trained actor and a frozen teacher decide every
# token together.
#
# The teacher constrains and the actor chooses. At each step a token must sit in BOTH models'
# top-k and clear both probability floors; among the survivors the ACTOR's own distribution
# decides. So the teacher steers direction while the text stays in the actor's voice. The
# resulting response replaces one of the n GRPO rollouts for that question, exactly as the
# other two SR paths do.
#
#   actor / student ($MODEL_PATH)  -- the model being TRAINED. It rolls out the short
#       question prompt as usual, and then acts as the student in joint decoding, decoding
#       under that same short prompt.
#   teacher ($TEACHER_PATH)        -- frozen, plain HF, never updated. Constrains the token
#       set while reading the LONG summarize prompt, so it steers using information the
#       actor does not have.
#
# ONE POOL, NO EXTRA GPUs. Joint decoding is a method on the actor's own worker
# (NewActorRolloutRefWorker.generate_joint), so it runs on the reasoner's GPUs after the
# rollout, once vLLM has gone to sleep. There is nothing to split and no CUDA_VISIBLE_DEVICES
# arithmetic to get wrong: trainer.n_gpus_per_node is the whole story. The student must be
# the live actor, and only the process holding its FSDP module can read its logits per token.
#
# HOW IT REACHES THE LOSS. Because the student IS the actor, the density the candidate was
# sampled from is the actor's own, which compute_log_prob already produces. So nothing is
# substituted for old_log_probs and the importance ratio r_t = pi_theta(y_t)/p_t(y_t) comes out
# of the framework untouched. What the sampler contributes is the one quantity no later forward
# pass can recover -- Z_t = p_t/mu_t, the token-level cost of the teacher's constraint, where
#
#     mu_t(v) = p_actor(v) * 1[v in F_t] / Z_t
#
# is the truncated, renormalised distribution actually sampled from. It multiplies the
# off-policy term AFTER clipping:
#
#     L_off = -Z_t * min( r_t * A_t , clip(r_t, 1-eps, 1+eps) * A_t )
#
# Keeping Z_t outside the ratio is the point: folded in, a Z_t of 0.2 inflates the ratio
# fivefold and saturates the clip on nearly every token, discarding gradient for reasons that
# have nothing to do with how far the policy has moved. Token alignment is exact -- the ids and
# their Z_t are written in the same statement from the same draw, unlike the offline path which
# stores text and re-tokenises.
#
# WHAT IT COSTS. There is no paged attention and no CUDA graph here, and every token costs
# two forward passes. Per step:
#
#     ceil(W / (JOINT_BATCH * N_GPUS)) * MAX_NEW_TOKENS    sequential two-model forwards
#
# W is data-dependent under 'wrong_only' -- it is how many questions the actor got entirely
# wrong this step. That is why SUMMARIZE_REPLACE defaults to wrong_only here and why
# MAX_QUESTIONS caps W: an uncapped step has no bound on its own duration. Whatever the cap
# drops is logged (batch/sr_joint_dropped), never silently skipped.
#
# MEMORY -- this design assumes each GPU can hold a whole model TWICE. The actor's weights are
# gathered once per step into an unsharded plain-HF mirror, and the decode runs against that;
# without it every layer would all-gather per token and the interconnect, not the GPU, would be
# the bottleneck. So a joint step holds: the actor's FSDP shard, the full mirror, the teacher,
# and both decoding models' KV caches. All of it comes out of what is left after vLLM's
# gpu_memory_utilization reservation. vLLM is asleep by then so its arena is back with the
# caching allocator, but how much is really reusable has NOT been measured -- if the first
# joint step OOMs, lower GPU_MEM_UTIL.
#
# THE THREE METRICS TO WATCH, in this order:
#   batch/sr_joint_fallback_frac -- share of tokens where the intersection was EMPTY, so
#       one model decided alone. Near 1.0 means the agreement never bound and the run is
#       effectively plain single-model decoding; raise AGREE_STUDENT_TOP_K first (it lets
#       the actor reach further down its own ranking without widening what the teacher
#       permits).
#   batch/sr_joint_keep_ratio    -- Z_t as measured at sampling time from the eligibility mask:
#       the share of the actor's mass the teacher left standing. The continuous version of the
#       above, and it sees what fallback cannot: keep_ratio 0.2 with zero fallback means the
#       constraint bites hard at every single step without ever failing outright.
#   actor/off_z_mean             -- the Z_t actually multiplied into the loss. Differs from
#       keep_ratio on fallback steps (keep_ratio records 0 there by its own diagnostic
#       convention; the loss uses the real p_student/p_teacher), so the two separate as the
#       fallback rate rises. This is the one that affects the gradient.
#   actor/off_ratio_ess          -- as always. Watch it from step 1.
#
# THE CHECK THAT VALIDATES EVERYTHING, and it needs no special configuration. Because the
# student IS the actor and no optimizer step has intervened, the loss's numerator and
# denominator are the same model, same prompt, same temperature -- so on step 1
#
#     actor/off_ratio  must sit at 1.0  on the off rows
#
# with Z_t appearing separately in actor/off_z_mean rather than inside the ratio. A ratio that
# is not ~1 there means a token misalignment or a temperature mismatch; nothing else in the
# pipeline reports either.
#
# Usage:
#   TEACHER_PATH=/home/data/shared/<sft-teacher> \
#       bash train_hype_summarize_joint.sh Qwen3-4b-base
#
#   # all four GPUs go to the reasoner; joint decoding shares them
#   GPU_DEVICES=0,1,2,3 TEACHER_PATH=... bash train_hype_summarize_joint.sh
#
# Sweep the agreement settings OFFLINE first -- examples/custom/run_joint_decode.sh scores
# them against a val set for a fraction of a training run's cost. Bring one setting here.
# ---------------------------------------------------------------------------

# The GPUs the reasoner gets. Joint decoding shares them -- there is no second pool.
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline

WORKER_DIR=/home/zhwang
MODEL_DIR=/home/data/shared
name="rl-rl"
off_policy_strategy="rl-rl"

# Must consume old_log_probs for the importance ratio to survive: dynamic_clip and vanilla
# skip the off_old_log_probs -> old_log_probs swap. The trainer refuses to start otherwise.
off_policy_reshape="clip"
suffix="hype_summarize_joint_"${off_policy_strategy}"_"${off_policy_reshape}

# Data. Needs the pre-rendered summarize_prompts column
# (Data/prepare_summarize_prompts.py) -- that is the long prompt the TEACHER reads. NO
# sr_response column: candidates are decoded online here.
train_path=${TRAIN_PATH:-$WORKER_DIR/LLM/Data/deepmath/deepmath_hard_thinkonly_split_summarize_new3.parquet}
test3_path=$WORKER_DIR/LLM/Data/split_by_source/amc_nothink.parquet
train_files="['$train_path']"
val_files="['$test3_path']"

MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4b-base"}

# The frozen teacher. The student needs no path -- it IS the actor at $MODEL_PATH.
#
# It MUST share the actor's tokenizer, and the requirement is STRICTER than elsewhere in this
# repo. The rephraser path only needs matching token IDs, because it exchanges ids. Here the
# two models' logits are intersected POSITION BY POSITION, so index i must mean the same token
# to both. A mismatch produces fluent nonsense with no error anywhere; generate_joint compares
# the vocabularies against the actor's tokenizer and refuses to start.
TEACHER_PATH=${TEACHER_PATH:-}
if [ -z "$TEACHER_PATH" ]; then
    echo "set TEACHER_PATH to the frozen model that should steer (e.g. an SFT checkpoint)" >&2
    echo "  TEACHER_PATH=/home/data/shared/<sft-teacher> bash $0" >&2
    exit 1
fi
if [ ! -d "$TEACHER_PATH" ]; then
    echo "TEACHER_PATH is not a directory: $TEACHER_PATH" >&2
    exit 1
fi

N_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")

# train_batch_size * rollout.n must be divisible by the GPU count (_validate_config). Joint
# decoding adds no constraint here: its batch is padded to the same world_size in the trainer.
TRAIN_BSZ=${TRAIN_BSZ:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
if [ $((TRAIN_BSZ * ROLLOUT_N % N_GPUS)) -ne 0 ]; then
    echo "train_batch_size*rollout.n ($((TRAIN_BSZ * ROLLOUT_N))) is not divisible by the" >&2
    echo "  $N_GPUS visible GPU(s); adjust TRAIN_BSZ or ROLLOUT_N" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# joint decoding knobs. Defaults live in verl/custom/config/joint_decode.yaml; these
# override them. Sweep with run_joint_decode.sh before changing them here.
# ---------------------------------------------------------------------------

# How much of the vocabulary the TEACHER permits at all -- the knob that sets how tightly
# the constraint binds.
AGREE_TEACHER_TOP_K=${AGREE_TEACHER_TOP_K:-10}
# How far down its OWN ranking the student may look for a permitted token. Raising this is
# the cheapest cure for a high fallback rate: it does not widen what the teacher allows.
AGREE_STUDENT_TOP_K=${AGREE_STUDENT_TOP_K:-10}
AGREE_TEACHER_MIN_PROB=${AGREE_TEACHER_MIN_PROB:-0.05}
AGREE_STUDENT_MIN_PROB=${AGREE_STUDENT_MIN_PROB:-0.0}

# Who decides when the intersection is empty. Two different experiments:
#   teacher -- it overrides exactly where the student disagrees. High fallback => the run
#              is effectively plain teacher decoding.
#   student -- the teacher only ever CONSTRAINS, never overrides. High fallback => plain
#              student decoding. Also flips how fallback_frac should be read.
AGREE_FALLBACK=${AGREE_FALLBACK:-teacher}

# Rows decoded concurrently per rank. Bounded by memory, not by the question count: during
# the decode the actor's params are unsharded and the teacher is resident, and both models'
# KV caches grow with this times MAX_NEW_TOKENS.
JOINT_BATCH=${JOINT_BATCH:-8}

# Keep the teacher on CPU between steps and move it to GPU only while decoding. ~8GB over
# PCIe per step for a 4B model, negligible against thousands of sequential decode steps, and
# it leaves GPU_MEM_UTIL alone. Set False to keep it resident (no transfer, but then lower
# GPU_MEM_UTIL to make room).
TEACHER_OFFLOAD=${TEACHER_OFFLOAD:-True}

# Hard cap on questions per step, applied AFTER wrong_only narrows the set. See the cost
# formula in the header. 0 disables it, which is only safe if you have measured a step.
MAX_QUESTIONS=${MAX_QUESTIONS:-16}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}

# MUST match actor_rollout_ref.rollout.temperature. The loss scales its NUMERATOR's logits
# by the rollout temperature while the denominator is recorded at this one, so a gap puts a
# systematic factor into off_ratio that nothing else reports. The trainer warns; it cannot
# know which of the two you meant.
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
JOINT_TEMPERATURE=${JOINT_TEMPERATURE:-$ROLLOUT_TEMPERATURE}

# 'wrong_only' scores the on-policy rollouts first and only replaces questions the reasoner
# got entirely wrong. That is what keeps W -- and therefore the step time -- bounded;
# 'all' sends every question to a decoder with no paged attention and is not viable at
# these lengths.
SUMMARIZE_REPLACE=${SUMMARIZE_REPLACE:-wrong_only}

PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME=${EXP_NAME:-"joint_$(basename $TEACHER_PATH)_k${AGREE_STUDENT_TOP_K}.${AGREE_TEACHER_TOP_K}_tmp${AGREE_TEACHER_MIN_PROB}"}
LOG_DIR=/home/data/zhwang_logs/train_4b/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log

TENSOR_PARALLEL=1

# vLLM's arena, reserved up front. Joint decoding runs AFTER the rollout with vLLM asleep, so
# it works out of what this leaves plus whatever the sleeping arena releases back to the
# caching allocator. How much of that is really reusable has not been measured -- lower this
# first if the joint step OOMs.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}

cd "$MODEL_DIR" || exit 1
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi

echo "=== GPUs                : $GPU_DEVICES ($N_GPUS, shared by rollout/train/joint)"
echo "=== actor  = student    : $MODEL_PATH  [trained; chooses, short prompt, mem_util=$GPU_MEM_UTIL]"
echo "=== teacher (frozen)    : $TEACHER_PATH  [constrains, long prompt, offload=$TEACHER_OFFLOAD]"
echo "=== joint decode        : batch=$JOINT_BATCH, max_new=$MAX_NEW_TOKENS"
echo "=== agreement           : student_top_k=$AGREE_STUDENT_TOP_K teacher_top_k=$AGREE_TEACHER_TOP_K"
echo "                          teacher_min_prob=$AGREE_TEACHER_MIN_PROB fallback=$AGREE_FALLBACK"
echo "=== questions/step       : <= $MAX_QUESTIONS ($SUMMARIZE_REPLACE)"

python -m verl.trainer.main_ppo_new \
    +joint_decode.enable=True \
    +joint_decode.teacher_model_path=$TEACHER_PATH \
    +joint_decode.teacher_offload=$TEACHER_OFFLOAD \
    +joint_decode.fuse=agree \
    +joint_decode.agree_student_top_k=$AGREE_STUDENT_TOP_K \
    +joint_decode.agree_teacher_top_k=$AGREE_TEACHER_TOP_K \
    +joint_decode.agree_student_min_prob=$AGREE_STUDENT_MIN_PROB \
    +joint_decode.agree_teacher_min_prob=$AGREE_TEACHER_MIN_PROB \
    +joint_decode.agree_fallback=$AGREE_FALLBACK \
    +joint_decode.temperature=$JOINT_TEMPERATURE \
    +joint_decode.max_new_tokens=$MAX_NEW_TOKENS \
    +joint_decode.batch_size=$JOINT_BATCH \
    +joint_decode.max_questions_per_step=$MAX_QUESTIONS \
    +joint_decode.student_prompt=short \
    +joint_decode.teacher_prompt=long \
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
    +actor_rollout_ref.rollout.summarize_replace=$SUMMARIZE_REPLACE \
    +actor_rollout_ref.rollout.summarize_replace_k=1 \
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
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
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
    trainer.n_gpus_per_node=$N_GPUS \
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
