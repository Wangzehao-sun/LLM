set -x
#!/usr/bin/env bash
# Prefix-RFT (arXiv:2507.01679) ported onto this framework.
#
# WHAT IT DOES. Each question gets n_prefix rollout prompts. The first num_empty_prefix are
# the bare question (pure on-policy); the rest each draw their OWN prefix ratio, per sample,
# from Beta(alpha, beta) inside a scheduled window:
#
#     ratio ~ Beta(a, b)  mapped into  [low_ctrl(step), high_ctrl(step)]
#
# That fraction of the reference solution is spliced in front of the model's continuation, and
# prefix tokens are marked in prefix_mask. The window's upper bound decays (cosine by default),
# so the prefix is withdrawn as training proceeds -- early on it is the only thing producing a
# usable trajectory on a hard question, later it does work the policy should be doing itself.
#
# WHY THE SPREAD MATTERS. A single fixed ratio gives every row in a step the same amount of
# help, so the group has no contrast between "almost solved it alone" and "needed most of the
# answer". Sampling per sample is what creates that contrast.
#
# HOW IT REACHES THE LOSS. Two pieces, both already in this framework:
#   * prefix tokens go through the off-policy branch of compute_token_on_off_sft_loss
#     (new_core_alg.py), gated by prefix_mask -- with off_loss_remove_clip=True they are
#     NOT PPO-clipped, matching upstream's enable_clip=False. That split IS the method.
#   * the advantage baseline is taken per (question, prefix slot) rather than per question,
#     because rows conditioned on different prefix lengths are not comparable samples.
#     See compute_grpo_prefix_outcome_advantage; it engages on the presence of prefix_uid.
#
# DIFFERENCES FROM UPSTREAM, both deliberate:
#   * the prefix is spliced into the RESPONSE segment, not appended as an unfinished assistant
#     turn in the prompt. Mathematically the same conditioning; the token stream differs by one
#     chat-template marker.
#   * one reference solution per question (our `target`), where upstream samples from a list
#     of `demos`. Costs a diversity source, not correctness.
#
# Usage (first arg = model subdir under $MODEL_DIR, rest passes through to Hydra):
#   bash train_prefix_rft_h20.sh Qwen3-4B-Base
#   PREFIX_STEPS=210 NUM_EMPTY_PREFIX=4 bash train_prefix_rft_h20.sh Qwen3-4B-Base

CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME:-verl}


GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline


# Experiment knobs.
# rl-rl routes to compute_token_on_off_sft_loss (new_dp_actor.py:373), whose off branch is
# the PPO ratio gated by prefix_mask -- the structure upstream's compute_policy_loss has.
name="rl-rl"   # -> compute_token_on_off_sft_loss; off branch = PPO ratio

# 'clip' keeps off_ratio as the plain PPO ratio exp(logp - old_logp), which is what upstream
# uses for prefix tokens. NOT p_div_p_0.1: that is LUFFY's p/(p+0.1) shaping, which discards
# the ratio entirely and would make this a different algorithm.
off_policy_reshape="clip"

# ---------------------------------------------------------------------------
# Prefix-RFT knobs
# ---------------------------------------------------------------------------
# Rollouts per question. This IS the GRPO group size on this path.
N_PREFIX=${N_PREFIX:-8}
# How many of those are bare-question (ratio 0). The paper: "8 rollouts per prompt, with one
# trajectory initiated using a sampled prefix... and the remaining 7 generated through standard
# online policy sampling" -- so 7 of 8.
NUM_EMPTY_PREFIX=${NUM_EMPTY_PREFIX:-7}
# Decay horizon. The paper trains 500 steps and decays over all of them; match it to YOUR
# run or the window never reaches its target (9004 rows / batch 128 * 3 epochs ~= 210 steps).
PREFIX_STEPS=${PREFIX_STEPS:-500}
# Window bounds. NOTE the direction: the paper moves the LOW bound and pins the HIGH one --
# "l is randomly sampled from U(low, high), where high is a constant and low decreases from
# high to near zero throughout entire training". So the window OPENS UP over training: early on
# low~=high~=0.95 (nearly always a long prefix), later low~=0.05 with high still 0.95 (the full
# range from short to long). That is the opposite of collapsing the high bound onto the floor.
PREFIX_LOW_INIT=${PREFIX_LOW_INIT:-0.95}
PREFIX_LOW_TARGET=${PREFIX_LOW_TARGET:-0.05}
PREFIX_HIGH_CONST=${PREFIX_HIGH_CONST:-0.95}
# Beta(1,1) = uniform in the window. alpha>1 favours long prefixes, beta>1 short ones.
PREFIX_ALPHA=${PREFIX_ALPHA:-1.0}
PREFIX_BETA=${PREFIX_BETA:-1.0}
# Token floor, applied per row against that row's own target length.
MIN_PREFIX_LEN=${MIN_PREFIX_LEN:-16}
# Fraction of prefix tokens that keep their advantage, ranked by entropy. Paper: top 20%.
PREFIX_ENT_KEEP=${PREFIX_ENT_KEEP:-0.2}

suffix="prefix_rft_"${off_policy_reshape}



# Data paths.
WORKER_DIR=${WORKER_DIR:-/apdcephfs_qy3/share_301372554/share_info/zenohaowang}

# Needs the `target` column: it is the reference solution prefixes are cut from. The
# summarize columns are unread here. Use the _thinkonly parquet if you want reasoning rather
# than a written-up solution as the demonstration.
train_path=$WORKER_DIR/LLM/Data/dapo_math/dapo_en_math_solution_9k_random_summarize_train.parquet
test1_path=$WORKER_DIR/LLM/Data/test/aime24_nothink_repeat16.parquet
test2_path=$WORKER_DIR/LLM/Data/test/aime25_nothink_repeat16.parquet
test3_path=$WORKER_DIR/LLM/Data/test/amc_nothink.parquet
test4_path=$WORKER_DIR/LLM/Data/test/math500_nothink.parquet
test5_path=$WORKER_DIR/LLM/Data/test/hmmt26_nothink_repeat16.parquet
train_files="['$train_path']"
#val_files="['$test1_path', '$test2_path', '$test3_path', '$test4_path']"
val_files="['$test2_path', '$test3_path','$test5_path']"
# Model path.
MODEL_DIR=$WORKER_DIR/Model
MODEL_PATH=$MODEL_DIR/${1:-"Qwen3-4B-Base"}

PROJECT_NAME="train_${name}_${suffix}_$(basename $MODEL_PATH)_$(basename $train_path .parquet)"
EXP_NAME="training_128_64"
LOG_DIR=$WORKER_DIR/LLM/zhwang_logs/train_4b/${PROJECT_NAME}/$EXP_NAME
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${PROJECT_NAME}.log
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$LOG_DIR/tensorboard}



GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")
TENSOR_PARALLEL=1

cd "$MODEL_DIR" || exit 1
echo "change to dir: $PWD"
if [ -n "$1" ]; then
    shift
fi

echo "=== GPUs          : $GPU_DEVICES ($GPU_NUM)"
echo "=== actor         : $MODEL_PATH"
echo "=== rollouts/step : $N_PREFIX per question ($NUM_EMPTY_PREFIX bare + $((N_PREFIX-NUM_EMPTY_PREFIX)) prefixed)"
echo "=== prefix window : U(low, $PREFIX_HIGH_CONST), low: cosine $PREFIX_LOW_INIT->$PREFIX_LOW_TARGET over $PREFIX_STEPS"
echo "=== entropy keep  : top $PREFIX_ENT_KEEP of prefix tokens carry gradient"
echo "=== off shaping   : $off_policy_reshape"

# Train over a single node using the GPUs exposed by *_VISIBLE_DEVICES.
python -m verl.trainer.main_ppo_new \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.filter_reward=False \
    +algorithm.trajectory_filter.enable=False \
    data.train_files=$train_files \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.shuffle=False \
    +data.warmup_steps=0 \
    +data.reward_impl_version=4 \
    +data.target_key=target \
    +data.filter_targets=True \
    +data.use_se=False \
    +data.collect_failures=False \
    +data.failure_buffer_max_size=128 \
    +data.n_recycle_failure=1 \
    +data.retain_hard_in_buffer=False \
    +data.retain_accuracy_low=0.50 \
    +data.retain_accuracy_high=0.75 \
    +data.max_recycle_count=3 \
    +data.use_summarize=False \
    +data.extra_step_start_after=0 \
    +data.recycle_sr_disable_after=1000 \
    +data.max_rounds_per_trigger=3 \
    +data.extra_step_interval=1 \
    +data.collect_accuracy_threshold=0.5 \
    +data.collect_accuracy_low=0.1 \
    reward_model.reward_manager='math' \
    +se_model.enable=False \
    rephraser.enable=False \
    +joint_decode.enable=False \
    +actor_rollout_ref.rollout.summarize_replace=off \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.policy_loss.rephrase_kl_coef=0.0 \
    +actor_rollout_ref.actor.policy_loss.reasoner_affinity_coef=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$name \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    +actor_rollout_ref.actor.policy_loss.off_policy_masking=False \
    +actor_rollout_ref.actor.policy_loss.off_policy_reshape=${off_policy_reshape} \
    +actor_rollout_ref.actor.use_sft_prefix_reward=False \
    +actor_rollout_ref.actor.use_off_policy_loss=True \
    +actor_rollout_ref.actor.off_policy_normalize=False \
    +actor_rollout_ref.actor.off_policy_strategy=$name \
    +actor_rollout_ref.actor.off_policy_loss_impl=token \
    +actor_rollout_ref.actor.off_policy_max_clip=-1 \
    +actor_rollout_ref.actor.off_policy_min_clip=-1 \
    +actor_rollout_ref.actor.all_max_clip=-1 \
    +actor_rollout_ref.actor.use_off_policy_probs=False \
    +actor_rollout_ref.actor.loss_remove_token_mean=False \
    +actor_rollout_ref.actor.loss_remove_clip=False \
    +actor_rollout_ref.actor.on_loss_remove_clip=False \
    `# enable_clip=False upstream: PREFIX tokens take the UNCLIPPED off-policy loss while` \
    `# generated tokens keep standard PPO clipping. That split is the method` \
    `# (core_algos.py:634 there, new_core_alg.py:358 here).` \
    +actor_rollout_ref.actor.off_loss_remove_clip=True \
    `# Keep only the top-20% highest-entropy PREFIX tokens in the gradient; the rest get` \
    `# advantage 0. The paper's entropy clipping.` \
    +actor_rollout_ref.actor.prefix_entropy_keep_ratio=$PREFIX_ENT_KEEP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.extra_temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$N_PREFIX \
    +actor_rollout_ref.rollout.prefix_mode=beta \
    +actor_rollout_ref.rollout.se_top_k=-1 \
    +actor_rollout_ref.rollout.se_top_p=1 \
    +actor_rollout_ref.rollout.n_val=1 \
    +actor_rollout_ref.rollout.max_prefix_len=10240 \
    +actor_rollout_ref.rollout.n_off=0 \
    +actor_rollout_ref.rollout.n_prefix=$N_PREFIX \
    +actor_rollout_ref.rollout.num_empty_prefix=$NUM_EMPTY_PREFIX \
    +actor_rollout_ref.rollout.min_prefix_len=$MIN_PREFIX_LEN \
    +actor_rollout_ref.rollout.prefix_steps=$PREFIX_STEPS \
    `# low: cosine 0.95 -> 0.05 over PREFIX_STEPS; high: constant. Window opens up.` \
    +actor_rollout_ref.rollout.prefix_low_ctrl_type=cosine_decay \
    +actor_rollout_ref.rollout.prefix_low_ctrl_init=$PREFIX_LOW_INIT \
    +actor_rollout_ref.rollout.prefix_low_ctrl_target=$PREFIX_LOW_TARGET \
    +actor_rollout_ref.rollout.prefix_high_ctrl_type=const \
    +actor_rollout_ref.rollout.prefix_high_ctrl_init=$PREFIX_HIGH_CONST \
    +actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
    +actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    +actor_rollout_ref.rollout.prefix_ctrl_alpha=$PREFIX_ALPHA \
    +actor_rollout_ref.rollout.prefix_ctrl_beta=$PREFIX_BETA \
    +actor_rollout_ref.rollout.n_se=0 \
    +actor_rollout_ref.rollout.prefix_ratio=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.balance_batch=False \
    trainer.rollout_data_dir=$LOG_DIR/rollout_data \
    +trainer.log_prob_dir=$LOG_DIR/log_probs \
    +trainer.save_tensors_dir=$LOG_DIR/save_tensors \
    +trainer.metrics_data_dir=$LOG_DIR \
    trainer.default_local_dir=$LOG_DIR/ckpt \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 $@ 2>&1 | tee ${LOG_PATH}
