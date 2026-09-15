set -x

CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME:-verl}

unset ROCR_VISIBLE_DEVICES

GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export RAY_DEDUP_LOGS=0
export WANDB_MODE=offline



# ---- data paths (mirrors train_hype_summarize's WORKER_DIR layout) ----
WORKER_DIR=${WORKER_DIR:-/apdcephfs_qy3/share_301372554/share_info/zenohaowang}

# ---- the checkpoints to evaluate: "name:path", one per line ----
# The name is what labels this checkpoint everywhere -- summary column, log file,
# output dirs, tensorboard run. An optional third field ":/path/to/base_model"
# overrides MODEL_PATH for that entry. A path of "base" evaluates the untrained
# model instead. Setting CKPTS in the environment replaces this list.
CKPT_ROOT=${CKPT_ROOT:-$WORKER_DIR/LLM/zhwang_logs/train_4b}
CKPTS=${CKPTS:-"
  step20:$CKPT_ROOT/<project>/training_128_64/ckpt/global_step_20
  step40:$CKPT_ROOT/<project>/training_128_64/ckpt/global_step_40
  base:base
"}

# The train dataloader is built even for a val-only run, and RLHFDatasetWithTarget
# needs the summarize columns, so this has to stay a real training parquet.
# train_batch_size cannot be shrunk to speed up that construction either --
# ray_trainer.py:439 asserts train_batch_size >= ppo_mini_batch_size.
TRAIN_FILES=${TRAIN_FILES:-$WORKER_DIR/LLM/Data/dapo_math/dapo_en_math_solution_9k_random_summarize_train.parquet}

test1_path=$WORKER_DIR/LLM/Data/test/aime24_nothink_repeat16.parquet
test2_path=$WORKER_DIR/LLM/Data/test/aime25_nothink_repeat16.parquet
test3_path=$WORKER_DIR/LLM/Data/test/amc_nothink.parquet
test4_path=$WORKER_DIR/LLM/Data/test/math500_nothink.parquet
test5_path=$WORKER_DIR/LLM/Data/test/hmmt26_nothink_repeat16.parquet
VAL_FILES=${VAL_FILES:-"$test2_path,$test3_path,$test5_path"}

# Optional: leave empty to skip the rephraser metric entirely. _validate_summarize
# returns {} when data.summarize_val_files is unset (new_ray_trainer.py:2282).
SUMMARIZE_VAL_PATH=${SUMMARIZE_VAL_PATH:-}

# Comma-separated in, python-list out.
val_files="['$(echo "$VAL_FILES" | sed "s/,/','/g")']"
train_files="['$TRAIN_FILES']"

# Irrelevant to a val-only run -- loss_mode only drives the actor update, which we
# return before reaching. Left at a mode that needs no extra +actor... keys.
name="vanilla"

# The base model the checkpoints were trained from. NOT a script parameter -- the
# script's parameters are the checkpoints -- but it cannot be dropped either: the
# actor is built with from_pretrained (fsdp_workers_new.py:288) before the shards
# can be loaded into it, and <ckpt>/actor/ holds only model_world_size_*_rank_*.pt
# while its huggingface/ subdir holds only config and tokenizer, no weights
# (fsdp_checkpoint_manager.py:236-248, since save_contents has no 'hf_model').
# Override here, via the environment, or per entry with an entry's third field.
MODEL_DIR=${MODEL_DIR:-$WORKER_DIR/Model}
MODEL_PATH=${MODEL_PATH:-$MODEL_DIR/Qwen3-4B-Base}

PROJECT_NAME=${PROJECT_NAME:-"eval_$(date +%m%d_%H%M)"}
LOG_DIR=${LOG_DIR:-$WORKER_DIR/LLM/zhwang_logs/eval_4b/${PROJECT_NAME}}
mkdir -p ${LOG_DIR}

CODE_DIR=${CODE_DIR:-$WORKER_DIR/LLM/Myverl}

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}

# val_batch_size only controls how many rows go to vLLM per generate call -- the
# val dataloader is drop_last=False, so nothing is dropped whatever the value and
# it may exceed the row count. It cannot be left unset though: with
# data.val_batch_size=null, _create_dataloader reads len(self.val_dataset) one
# line before that attribute is assigned (new_ray_trainer.py:913-916) and dies.
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}

# ---------------------------------------------------------------------------
# Parse the checkpoint map into LABELS / CKPT_PATHS / CKPT_MODELS.
# ---------------------------------------------------------------------------
if [ -n "$CKPT_LIST" ]; then
    [ -f "$CKPT_LIST" ] || { echo "CKPT_LIST not found: $CKPT_LIST" >&2; exit 1; }
    CKPTS="$CKPTS
$(cat "$CKPT_LIST")"
fi
[ -n "$CKPTS" ] || { echo "set CKPTS=\"name:/path/global_step_N,...\" or CKPT_LIST=<file>" >&2; exit 1; }

LABELS=(); CKPT_PATHS=(); CKPT_MODELS=()
while IFS= read -r entry; do
    entry="$(echo "$entry" | xargs)"           # trim whitespace; drops empties
    [ -n "$entry" ] || continue

    lbl="${entry%%:*}"                         # before the first colon
    rest="${entry#*:}"                         # after the first colon
    if [ "$lbl" = "$entry" ]; then             # no colon at all
        lbl="$entry"; rest=""
    fi
    mdl=""
    case "$rest" in
        *:*) mdl="${rest#*:}"; rest="${rest%%:*}" ;;   # optional model override
    esac

    # Filesystem- and hydra-safe label; it becomes a filename and an experiment
    # name. printf, not echo: echo's trailing newline would become a '_' too.
    safe=$(printf '%s' "$lbl" | tr -c 'A-Za-z0-9._-' '_')
    [ "$safe" = "$lbl" ] || echo "[note] label '$lbl' -> '$safe'"

    if [ -z "$rest" ] || [ "$rest" = "base" ] || [ "$rest" = "none" ] \
       || [ "$lbl" = "base" ] || [ "$lbl" = "none" ]; then
        LABELS+=("$safe"); CKPT_PATHS+=("BASE"); CKPT_MODELS+=("$mdl")
        continue
    fi

    case "$rest" in
        *global_step_*) ;;
        *)
            # ray_trainer.py:865 asserts on this, and :872 parses the step out of it,
            # so point at the global_step_N folder itself, not the run directory.
            echo "[skip] $safe: path has no 'global_step_' in it: $rest" >&2
            if compgen -G "$rest/global_step_*" >/dev/null; then
                echo "        did you mean one of:" >&2
                ls -d "$rest"/global_step_* 2>/dev/null | sed 's/^/          /' >&2
            fi
            continue ;;
    esac
    if [ ! -d "$rest/actor" ]; then
        echo "[skip] $safe: $rest/actor not found" >&2
        continue
    fi
    LABELS+=("$safe"); CKPT_PATHS+=("$rest"); CKPT_MODELS+=("$mdl")
# Comments are stripped BEFORE the comma split, not after: a comment containing a
# comma would otherwise be cut in two and its tail survive as a bogus entry.
done < <(echo "$CKPTS" | sed 's/#.*//' | tr ',' '\n')

[ ${#LABELS[@]} -gt 0 ] || { echo "no usable checkpoints" >&2; exit 1; }
echo "will evaluate ${#LABELS[@]}: ${LABELS[*]}"
echo "val_batch_size=$VAL_BATCH_SIZE over: $VAL_FILES"

cd $CODE_DIR
echo "change to dir: $PWD"

if [ -n "$SUMMARIZE_VAL_PATH" ]; then
    SUMMARIZE_ARGS="+data.summarize_val_files=['$SUMMARIZE_VAL_PATH'] +data.summarize_val_k=8 +data.summarize_val_batch_size=128"
else
    SUMMARIZE_ARGS=""
fi

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    CKPT_PATH="${CKPT_PATHS[$i]}"
    THIS_MODEL="${CKPT_MODELS[$i]:-$MODEL_PATH}"

    # Per checkpoint, not once for the run: TENSORBOARD_DIR takes precedence over
    # the per-experiment default (tracking.py:205), so exporting it globally would
    # funnel every checkpoint into one directory and let two of them collide at the
    # same x -- global_steps comes from the path's global_step_N (ray_trainer.py:872),
    # and different runs both have a global_step_100.
    export TENSORBOARD_DIR=$LOG_DIR/tensorboard/$LABEL

    if [ "$CKPT_PATH" = "BASE" ]; then
        # Nothing to load: resume_mode=disable leaves the freshly-initialised base
        # weights in place, which is the pre-training baseline.
        RESUME_ARGS="trainer.resume_mode=disable"
    else
        RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=$CKPT_PATH"
    fi

    LOG_PATH=${LOG_DIR}/eval_${LABEL}.log
    echo "=== evaluating $LABEL ($CKPT_PATH, model $THIS_MODEL) -> $LOG_PATH ==="

    python -m verl.trainer.main_ppo_new \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.000 \
        algorithm.norm_adv_by_std_in_grpo=False \
        +algorithm.filter_reward=False \
        data.train_files=$train_files \
        data.val_files="$val_files" \
        data.train_batch_size=128 \
        data.val_batch_size=$VAL_BATCH_SIZE \
        data.max_prompt_length=2048 \
        data.max_response_length=20480 \
        data.return_full_prompt=True \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=4 \
        data.shuffle=False \
        +data.reward_impl_version=4 \
        +data.filter_targets=False \
        +data.use_se=False \
        +data.use_summarize=True \
        +data.summarize_prompts_key=summarize_prompts \
        +data.summarize_prompt_key=summarize_prompt \
        +data.max_summarize_prompts=8 \
        +data.max_summarize_length=8192 \
        reward_model.reward_manager='math' \
        +se_model.enable=False \
        +actor_rollout_se.model.path=$THIS_MODEL \
        actor_rollout_ref.model.path=$THIS_MODEL \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size=64 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.policy_loss.loss_mode=$name \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
        actor_rollout_ref.rollout.max_num_batched_tokens=81920 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.prompt_length=8192 \
        +actor_rollout_ref.rollout.prefix_mode=summarize \
        +actor_rollout_ref.rollout.se_top_k=-1 \
        +actor_rollout_ref.rollout.se_top_p=1 \
        +actor_rollout_ref.rollout.max_prefix_len=10240 \
        +actor_rollout_ref.rollout.n_off=0 \
        +actor_rollout_ref.rollout.n_prefix=8 \
        +actor_rollout_ref.rollout.n_se=0 \
        +actor_rollout_ref.rollout.prefix_ratio=1 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="$LABEL" \
        trainer.val_only=True \
        trainer.val_before_train=True \
        $RESUME_ARGS \
        trainer.n_gpus_per_node=$GPU_NUM \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.balance_batch=False \
        trainer.validation_data_dir=$LOG_DIR/val_data_${LABEL} \
        trainer.rollout_data_dir=$LOG_DIR/rollout_data_${LABEL} \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=1 $@ 2>&1 | tee ${LOG_PATH}
done

# ---------------------------------------------------------------------------
# Summary: one row per metric, one column per checkpoint name.
#
# Only val-core is tabulated, minus the spread metrics. process_validation_metrics
# emits standard deviations in two shapes (metric_utils.py:396,407-417): "std@N" as
# a segment of its own, and "best@N/std" / "worst@N/std" / "maj@N/std" as a
# trailing segment -- testing the last path segment catches both. Everything else
# (val-aux/*, timing/*) stays in the per-checkpoint log.
#
# new_ray_trainer.py:2486 prints the metric dict through pprint, which wraps the
# resulting long string across lines. A metric straddling a seam loses its value
# to a plain grep, so rejoin the seams before parsing.
# ---------------------------------------------------------------------------
echo
echo "================ summary (val-core) ================"
python - "$LOG_DIR" "${LABELS[@]}" <<'PYEOF'
import os
import re
import sys

log_dir, labels = sys.argv[1], sys.argv[2:]


def keep(key):
    if not key.startswith("val-core/"):
        return False
    last = key.rsplit("/", 1)[-1]
    return last != "std" and not last.startswith("std@")


rows, keys = {}, []
for lab in labels:
    path = os.path.join(log_dir, f"eval_{lab}.log")
    if not os.path.exists(path):
        print(f"  [warn] missing log for {lab}")
        continue
    text = open(path, errors="replace").read()
    text = re.sub(r'"\s*\n\s*"', "", text)
    found = {
        k: float(v)
        for k, v in re.findall(r"'(val-[^']*)': ([0-9.eE+-]+)", text)
        if keep(k)
    }
    if not found:
        print(f"  [warn] no val-core metrics in {path}")
        continue
    rows[lab] = found
    for k in found:
        if k not in keys:
            keys.append(k)

if not rows:
    print("  no results")
    sys.exit(0)

keys.sort()
have = [l for l in labels if l in rows]
w = max(len(k) for k in keys) + 2
cw = max(12, max(len(l) for l in have) + 2)
print("  " + "metric".ljust(w) + "".join(l.rjust(cw) for l in have))
for k in keys:
    print("  " + k.ljust(w) + "".join(
        (f"{rows[l][k]:.4f}" if k in rows[l] else "-").rjust(cw) for l in have))
PYEOF
echo
echo "logs: $LOG_DIR"
