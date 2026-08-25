set -x
#!/usr/bin/env bash
# GPU selection. Override with, for example: GPU_DEVICES=4,5,6,7
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}
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
# reads data.prompt_key and needs a flat [system, user] list, while the renderer
# nests it one level. That script also refuses to run when the eval prompt differs
# from the one training used.
#
# Usage:
#   CKPT_DIR=/path/to/sft_run/ckpt \
#   EVAL_PATH=$HOME/LLM/Data/eval_rephrase_flat.parquet \
#   bash sweep_sft_checkpoints.sh
#
#   # evaluate plain problem-solving instead of the rephrase task
#   CKPT_DIR=... EVAL_PATH=... PROMPT_KEY=question_prompt bash sweep_sft_checkpoints.sh
#
#   # override the model name that prefixes every label
#   CKPT_DIR=... EVAL_PATH=... MODEL_NAME=qwen3-4b-run2 bash sweep_sft_checkpoints.sh
#
#   # only some steps, or the untrained model as a baseline
#   CKPT_DIR=... EVAL_PATH=... STEPS=15,30 bash sweep_sft_checkpoints.sh
#   BASE_MODEL=/home/data/shared/Qwen3-4b-base EVAL_PATH=... bash sweep_sft_checkpoints.sh
# ---------------------------------------------------------------------------

# Where train_sft.sh wrote the checkpoints (its trainer.default_local_dir).
CKPT_DIR=${CKPT_DIR:-}
# Evaluate this model too -- use it for the untrained baseline, which is what makes
# the SFT numbers interpretable.
BASE_MODEL=${BASE_MODEL:-/home/data/shared/Qwen3-4B-Instruct}
EVAL_PATH=${EVAL_PATH:-$HOME/LLM/Data/eval_rephrase_flat.parquet}

# Comma-separated global_step numbers to evaluate; empty = every checkpoint found.
STEPS=${STEPS:-}

# Name used in the result labels. Empty = derive it from CKPT_DIR / BASE_MODEL.
MODEL_NAME=${MODEL_NAME:-}

# Which prompt column to evaluate. prepare_rephrase_eval.py writes both:
#   prompt          -> the rephrase task (question + expert-reasoning draft)
#   question_prompt -> the bare question, i.e. plain problem-solving ability
PROMPT_KEY=${PROMPT_KEY:-prompt}

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

# Format screening on top of math-verify's answer check. The rule is the trainer's
# (_trajectory_filter_reject); these word lists are the generation side's own and do not
# affect training. Hydra list syntax: no spaces after the commas.
TRAJ_FILTER=${TRAJ_FILTER:-False}
TRAJ_KEYWORDS=${TRAJ_KEYWORDS:-'["the draft","based on the draft","according to the draft","from the draft","the experience","based on the experience","according to the experience","the provided reasoning","based on the reasoning above"]'}
TRAJ_INSTR_PHRASES=${TRAJ_INSTR_PHRASES:-'["your task is","output only","do not mention","self-contained solution","no meta-talk"]'}

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

# Model name for the labels. train_sft.sh names its run directory
# train_sft_<suffix>_<model>_<dataset>, so the model is in there but so is
# everything else; pull out the recognisable model part rather than using the whole
# string, since the label also becomes a directory name. Set MODEL_NAME to override.
if [ -z "$MODEL_NAME" ]; then
    if [ -n "$CKPT_DIR" ]; then
        run_name=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
        # e.g. train_sft_rephraser_Qwen3-4b-base_self_rollouts... -> Qwen3-4b-base
        MODEL_NAME=$(grep -oE '''[Qq]wen[A-Za-z0-9._-]*''' <<< "$run_name" | head -1)
        MODEL_NAME=${MODEL_NAME:-$run_name}
    else
        MODEL_NAME=$(basename "$BASE_MODEL")
    fi
fi
echo "model name for labels: $MODEL_NAME"

# Collect the models to evaluate as "<label>:<path>" pairs. The label names the
# model as well as the step, so summaries from different runs stay distinguishable
# when compared side by side.
TARGETS=()
if [ -n "$BASE_MODEL" ]; then
    TARGETS+=("${MODEL_NAME}-base:$BASE_MODEL")
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
        # failing after the model-load attempt. Check each pattern separately: one
        # `ls` with both patterns exits non-zero when EITHER finds nothing, so a
        # complete safetensors checkpoint would look empty just because there is no
        # pytorch_model*.bin.
        if ! compgen -G "$path/*.safetensors" >/dev/null && \
           ! compgen -G "$path/pytorch_model*.bin" >/dev/null; then
            echo "[skip] $path has no weight files yet"
            continue
        fi
        TARGETS+=("${MODEL_NAME}-step${step}:$path")
    done
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "nothing to evaluate: no global_step_* under $CKPT_DIR matching STEPS=$STEPS" >&2
    exit 1
fi

SWEEP_DIR=${LOG_ROOT}/${EXP_NAME}
mkdir -p "$SWEEP_DIR"
SUMMARY="${SWEEP_DIR}/summary.tsv"
printf 'label\tavg_score\tmax_score\tavg_len_tokens\tans_err\tfmt_err\tfmt_ok_ans\toutput_dir\n' > "$SUMMARY"

echo "=== sweeping ${#TARGETS[@]} model(s) on $(basename "$EVAL_PATH"), prompt_key=$PROMPT_KEY ==="
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
        data.prompt_key=$PROMPT_KEY \
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
        +algorithm.trajectory_filter.enable=$TRAJ_FILTER \
        +algorithm.trajectory_filter.keywords="$TRAJ_KEYWORDS" \
        +algorithm.trajectory_filter.instruction_phrases="$TRAJ_INSTR_PHRASES" \
        +reward_model.reward_impl_version=4 2>&1 | tee "$log_path"

    # Recompute the metrics from the written parquets rather than scraping the log:
    # main_generation averages over batches, so a short final batch would be
    # weighted the same as a full one.
    #   avg_score -- mean of per-question mean_score, i.e. overall pass rate
    #   max_score -- mean of per-question max_score, i.e. pass@N
    #   avg_len   -- mean response length in TOKENS, to spot a model that started
    #                rambling or truncating rather than answering. Compare it
    #                against RESPONSE_LENGTH to see whether generations are
    #                hitting the cap.
    #   ans_err / fmt_err -- share of responses failing on the ANSWER vs on FORMAT.
    #                Only present when TRAJ_FILTER=True wrote the raw_* columns; NA
    #                otherwise. fmt_ok_ans is the subset rejected on format DESPITE a
    #                correct answer, i.e. how much avg_score would rise without the
    #                format check.
    metrics=$(python - "$out_dir" <<'PYEOF'
import glob
import sys

import pandas as pd
import pyarrow.parquet as pq

files = sorted(glob.glob(f"{sys.argv[1]}/*.parquet"))
means, maxes, lengths = [], [], []
missing_lengths = False
n_resp = n_answer_err = n_format_err = n_fmt_ok_ans = 0
have_breakdown = False


def answer_ok(score):
    # reward_impl_version=4 returns numpy bools, others return floats. float() covers
    # both (float(np.True_) == 1.0); the guard is for a missing/odd value, which must
    # read as "not correct" rather than raise.
    try:
        return float(score) == 1.0
    except (TypeError, ValueError):
        return False


for path in files:
    # response_lengths is the per-response valid token count recorded by
    # main_generation; parquets written before it existed have to be reported as
    # NA rather than silently measured in characters, which reads ~3x larger.
    columns = ["test_score"]
    if "response_lengths" in pq.read_schema(path).names:
        columns.append("response_lengths")
    else:
        missing_lengths = True
    df = pd.read_parquet(path, columns=columns)
    for score in df["test_score"]:
        means.append(float(score["mean_score"]))
        maxes.append(float(score["max_score"]))
        # format_rejected / raw_scores_per_response only exist when the filter ran.
        rejected = score.get("format_rejected") if hasattr(score, "get") else None
        raw = score.get("raw_scores_per_response") if hasattr(score, "get") else None
        if rejected is None or raw is None:
            continue
        have_breakdown = True
        for rej, raw_score in zip(rejected, raw):
            n_resp += 1
            if rej:
                n_format_err += 1
                if answer_ok(raw_score):
                    n_fmt_ok_ans += 1
            elif not answer_ok(raw_score):
                n_answer_err += 1
    if "response_lengths" in df:
        for row in df["response_lengths"]:
            lengths.extend(int(n) for n in row)

if not means:
    print("NA\tNA\tNA\tNA\tNA\tNA")
else:
    avg_len = f"{sum(lengths) / len(lengths):.0f}" if lengths and not missing_lengths else "NA"
    if have_breakdown and n_resp:
        ans_err = f"{n_answer_err / n_resp:.4f}"
        fmt_err = f"{n_format_err / n_resp:.4f}"
        fmt_ok = f"{n_fmt_ok_ans / n_resp:.4f}"
    else:
        ans_err = fmt_err = fmt_ok = "NA"
    print(
        f"{sum(means) / len(means):.4f}\t{sum(maxes) / len(maxes):.4f}\t{avg_len}\t"
        f"{ans_err}\t{fmt_err}\t{fmt_ok}"
    )
PYEOF
)
    printf '%s\t%s\t%s\n' "$label" "$metrics" "$out_dir" >> "$SUMMARY"
    echo "=== [$label] avg_score / max_score / avg_len_tokens / ans_err / fmt_err / fmt_ok_ans: $metrics ==="
done

echo
echo "=== sweep summary ($SUMMARY) ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
