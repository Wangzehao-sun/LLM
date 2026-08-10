set -x
#!/usr/bin/env bash
# GPU selection. Override with, for example: GPU_DEVICES=4,5,6,7
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export WANDB_MODE=offline

# ---------------------------------------------------------------------------
# Sweep every SFT checkpoint on a fixed rephrase eval set.
#
# fsdp_sft_trainer has no generation path -- trainer.test_freq only computes a
# teacher-forcing val/loss -- so rephrase accuracy has to be measured offline.
# train_sft.sh saves one checkpoint per epoch (save_per_epoch=True) in HF format,
# which vLLM loads directly, so this walks <CKPT_DIR>/global_step_* and runs
# verl.trainer.main_generation on each, then prints a label -> accuracy table.
#
# The eval parquet must be built by Data/prepare_rephrase_eval.py: main_generation
# reads data.prompt_key=prompt and needs a flat [system, user] list, while the
# renderer nests it one level. That script also refuses to run when the eval
# prompt differs from the one training used.
#
# Usage:
#   CKPT_DIR=/path/to/sft_run/ckpt \
#   EVAL_PATH=$HOME/LLM/Data/eval_rephrase_flat.parquet \
#   bash sweep_sft_checkpoints.sh
#
#   # only some steps, or the untrained model as a baseline
#   CKPT_DIR=... EVAL_PATH=... STEPS=15,30 bash sweep_sft_checkpoints.sh
#   BASE_MODEL=/home/data/shared/Qwen3-4b-base EVAL_PATH=... bash sweep_sft_checkpoints.sh
# ---------------------------------------------------------------------------

# Where train_sft.sh wrote the checkpoints (its trainer.default_local_dir).
CKPT_DIR=${CKPT_DIR:-}
# Evaluate this model too -- use it for the untrained baseline, which is what makes
# the SFT numbers interpretable.
BASE_MODEL=${BASE_MODEL:-}
EVAL_PATH=${EVAL_PATH:-$HOME/LLM/Data/eval_rephrase_flat.parquet}

# Comma-separated global_step numbers to evaluate; empty = every checkpoint found.
STEPS=${STEPS:-}

N_SAMPLES=${N_SAMPLES:-4}          # samples per question; >1 to see sampling variance
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
PROMPT_LENGTH=${PROMPT_LENGTH:-4096}     # rephrase prompts carry a draft, so longer than a bare question
RESPONSE_LENGTH=${RESPONSE_LENGTH:-14336}
BATCH_SIZE=${BATCH_SIZE:-256}
MAX_STEPS=${MAX_STEPS:-1000}       # batch cap inside main_generation
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}

CODE_DIR=${CODE_DIR:-$HOME/LLM/Myverl}
LOG_ROOT=${LOG_ROOT:-$HOME/LLM/Train/verl/logs}
EXP_NAME=${EXP_NAME:-sweep_$(date +%m%d_%H%M)}

if [ -z "$CKPT_DIR" ] && [ -z "$BASE_MODEL" ]; then
    echo "set CKPT_DIR (to sweep checkpoints) or BASE_MODEL (to evaluate one model)" >&2
    exit 1
fi
if [ ! -f "$EVAL_PATH" ]; then
    echo "eval parquet not found: $EVAL_PATH" >&2
    echo "build it with: python3 Data/prepare_rephrase_eval.py --input <rendered>.parquet --output $EVAL_PATH" >&2
    exit 1
fi

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")

# Collect the models to evaluate as "<label>:<path>" pairs.
TARGETS=()
if [ -n "$BASE_MODEL" ]; then
    TARGETS+=("base:$BASE_MODEL")
fi
if [ -n "$CKPT_DIR" ]; then
    if [ ! -d "$CKPT_DIR" ]; then
        echo "CKPT_DIR is not a directory: $CKPT_DIR" >&2
        exit 1
    fi
    for path in $(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
        step=$(basename "$path" | sed 's/global_step_//')
        if [ -n "$STEPS" ] && ! grep -qx "$step" <<< "$(tr ',' '\n' <<< "$STEPS")"; then
            continue
        fi
        # A checkpoint still being written has no weights yet; skip it rather than
        # failing after the model-load attempt.
        if ! ls "$path"/*.safetensors "$path"/pytorch_model*.bin >/dev/null 2>&1; then
            echo "[skip] $path has no weight files yet"
            continue
        fi
        TARGETS+=("step$step:$path")
    done
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "nothing to evaluate: no global_step_* under $CKPT_DIR matching STEPS=$STEPS" >&2
    exit 1
fi

SWEEP_DIR=${LOG_ROOT}/${EXP_NAME}
mkdir -p "$SWEEP_DIR"
SUMMARY="${SWEEP_DIR}/summary.tsv"
printf 'label\tavg_score\toutput_dir\n' > "$SUMMARY"

echo "=== sweeping ${#TARGETS[@]} model(s) on $(basename "$EVAL_PATH") ==="
for target in "${TARGETS[@]}"; do
    echo "  ${target%%:*}  <-  ${target#*:}"
done

cd "$CODE_DIR" || exit 1
echo "change to dir: $PWD"

for target in "${TARGETS[@]}"; do
    label="${target%%:*}"
    model_path="${target#*:}"
    out_dir="${SWEEP_DIR}/${label}/save_data"
    log_path="${SWEEP_DIR}/${label}.log"
    mkdir -p "$out_dir"

    echo "=== [$label] $model_path ==="
    # main_generation treats data.output_path as a directory and writes one parquet
    # per batch into it, each carrying a test_score column.
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=$GPU_NUM \
        data.path="$EVAL_PATH" \
        data.prompt_key=prompt \
        +data.reward_model_key=reward_model \
        +data.data_source_key=data_source \
        data.n_samples=$N_SAMPLES \
        data.batch_size=$BATCH_SIZE \
        data.output_path="$out_dir/" \
        model.path="$model_path" \
        +model.trust_remote_code=True \
        rollout.temperature=$TEMPERATURE \
        rollout.top_k=-1 \
        rollout.top_p=$TOP_P \
        rollout.prompt_length=$PROMPT_LENGTH \
        rollout.response_length=$RESPONSE_LENGTH \
        rollout.max_num_batched_tokens=32768 \
        rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
        rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
        +is_eval=True \
        +max_steps=$MAX_STEPS \
        +reward_model.reward_impl_version=4 2>&1 | tee "$log_path"

    # Recompute the mean from the written parquets rather than scraping the log:
    # main_generation averages over batches, so a short final batch would be
    # weighted the same as a full one.
    score=$(python - "$out_dir" <<'PYEOF'
import glob
import sys

import pandas as pd

files = sorted(glob.glob(f"{sys.argv[1]}/*.parquet"))
if not files:
    print("NA")
    sys.exit()
means = []
for path in files:
    df = pd.read_parquet(path, columns=["test_score"])
    means.extend(float(s["mean_score"]) for s in df["test_score"])
print(f"{sum(means) / len(means):.4f}" if means else "NA")
PYEOF
)
    printf '%s\t%s\t%s\n' "$label" "$score" "$out_dir" >> "$SUMMARY"
    echo "=== [$label] mean score: $score ==="
done

echo
echo "=== sweep summary ($SUMMARY) ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
