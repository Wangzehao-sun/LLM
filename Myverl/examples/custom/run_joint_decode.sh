set -x
#!/usr/bin/env bash
# GPU selection. Override with, for example: GPU_DEVICES=4,5,6,7
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

echo $HOME
export TOKENIZERS_PARALLELISM=true

# ---------------------------------------------------------------------------
# Sweep two-model joint decoding on a fixed eval set.
#
# Data/joint_decode.py combines two models' predictions at every step, which
# verl's rollout cannot do: vLLM is entered once per request
# (vllm_rollout_spmd.py:304) so there is no per-token hook, and the pinned
# vllm<=0.8.5 runs the V1 engine, which dropped per-request logits_processors.
# So this drives a plain HF decode loop instead of verl.trainer.main_generation.
#
# Two families of rule, selected with FUSE:
#   linear / contrastive / max -- blend the distributions, then sample.
#                                 Swept over WEIGHTS.
#   agree                      -- the TEACHER (MODEL_B) constrains which tokens are
#                                 allowed; the STUDENT (MODEL_A) samples among them.
#                                 Empty overlap defers to the teacher. Both branches
#                                 sample at TEMPERATURE, so n_samples still varies.
#                                 Swept over AGREE_TEACHER_MIN_PROBS.
#
# The structure mirrors sweep_sft_checkpoints.sh -- one run per swept value rather
# than one per checkpoint -- and the results table is the same shape, because
# joint_decode.py writes the same parquet schema (responses / test_score /
# response_lengths). The metrics block below is byte-identical to that script's.
#
# WEIGHT=0 is not just a data point, it is the CORRECTNESS CHECK: at weight 0 the
# fusion reduces to model A alone, so with TEMPERATURE=0 its output must match a
# plain single-model greedy run token for token. Keep 0 in the sweep and compare
# it against main_generation.py on the same eval set before trusting any other row.
#
# For FUSE=agree the analogous check is the per-rank "teacher fallback" rate that
# joint_decode.py prints: near 100% means the overlap was always empty, the student
# never chose anything, and the run is just teacher sampling -- so the scores say
# nothing about the method. Near 0% with a floor of 0 means the allowed set covers
# everything, which makes it student sampling instead. The method only does work in
# between.
#
# Parallelism: torchrun starts one process per GPU and each decodes its own stride
# of the rows (Data/joint_decode.py shards with iloc[rank::world_size]). The ranks
# never talk, so this is pure data parallelism with no process group.
#
# Expect it to be MUCH slower than the vLLM path -- no paged attention, no CUDA
# graph, two forwards per token. That is why this is offline-eval only.
#
# The eval parquet must be built by Data/prepare_rephrase_eval.py: joint_decode.py
# needs a flat [system, user] list, while the renderer nests it one level.
#
# Usage:
#   MODEL_A=/home/data/shared/Qwen3-4b-base \
#   MODEL_B=/home/data/shared/<sft-rephraser-ckpt> \
#   EVAL_PATH=$HOME/LLM/Data/eval_rephrase_flat.parquet \
#   bash run_joint_decode.sh
#
#   # teacher B constrains, student A picks
#   MODEL_A=... MODEL_B=... FUSE=agree bash run_joint_decode.sh
#   MODEL_A=... MODEL_B=... FUSE=agree AGREE_TOP_K=20 \
#       AGREE_TEACHER_MIN_PROBS=0.05 bash run_joint_decode.sh
#
#   # a single weight, greedy -- the degenerate-case check
#   MODEL_A=... MODEL_B=... WEIGHTS=0 TEMPERATURE=0 LIMIT=8 bash run_joint_decode.sh
#
#   # contrastive extrapolation instead of interpolation (needs smaller weights)
#   MODEL_A=... MODEL_B=... FUSE=contrastive WEIGHTS=0,0.1,0.2 bash run_joint_decode.sh
#
#   # score the bare question instead of the rephrase task
#   MODEL_A=... MODEL_B=... PROMPT_KEY=question_prompt bash run_joint_decode.sh
# ---------------------------------------------------------------------------

# The model being steered, and the model mixed into it.
MODEL_A=${MODEL_A:-}
MODEL_B=${MODEL_B:-}
EVAL_PATH=${EVAL_PATH:-$HOME/LLM/Data/eval_rephrase_flat.parquet}

# Fusion weights on model B, one run each. 0 must stay in the list (see header).
# Ignored when FUSE=agree, which sweeps AGREE_TEACHER_MIN_PROBS instead.
WEIGHTS=${WEIGHTS:-0,0.25,0.5,0.75,1.0}
FUSE=${FUSE:-linear}               # linear | contrastive | max | agree

# --- FUSE=agree knobs ------------------------------------------------------
# The TEACHER (model B) constrains which tokens are allowed -- both models' top-k
# must contain the candidate and it must clear each side's probability floor. The
# STUDENT (model A) then samples among the survivors, so the teacher steers
# direction while the student picks what it finds natural. Empty overlap = no
# agreement to honour, so the step samples from the teacher instead.
#
# TEMPERATURE / TOP_P apply to both branches (WEIGHTS and TOP_K do not).
#
# The teacher floor is the knob that actually bites: on peaked, same-family
# distributions the top-k sets overlap almost completely, so k stops mattering
# above ~10 while the floor still moves both the fallback rate and how often the
# student's pick differs from the teacher's. So the sweep runs over it.
AGREE_TEACHER_MIN_PROBS=${AGREE_TEACHER_MIN_PROBS:-0,0.02,0.05,0.1}
AGREE_TOP_K=${AGREE_TOP_K:-10}
# Only raise this to veto tokens the student is very reluctant to emit -- it
# already ranks the survivors, so 0 is the natural default.
AGREE_STUDENT_MIN_PROB=${AGREE_STUDENT_MIN_PROB:-0}

# Which prompt column each model sees. prepare_rephrase_eval.py writes both:
#   prompt          -> the rephrase task (question + expert-reasoning draft)
#   question_prompt -> the bare question, i.e. plain problem-solving ability
# PROMPT_KEY_B empty = model B reads the same column as A, so the fusion is purely
# a model difference; set it to feed B a richer prompt than A instead.
PROMPT_KEY=${PROMPT_KEY:-prompt}
PROMPT_KEY_B=${PROMPT_KEY_B:-}

N_SAMPLES=${N_SAMPLES:-4}          # samples per question; >1 to see sampling variance
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
PROMPT_LENGTH=${PROMPT_LENGTH:-4096}      # rephrase prompts carry a draft, so longer than a bare question
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
# Far smaller than the vLLM path's 256: there is no request-level scheduling here,
# so the longest row in a batch holds up every other row in it.
#
# Memory, not that long tail, is the hard ceiling. Both models' weights are fixed
# (~16GB for a 4B pair) but KV cache grows linearly with this -- roughly 2.4GB per
# row for a 4B pair at prompt+response 8192. On an 80GB card that puts the limit
# around 20; past that expect OOM. Raise it while watching nvidia-smi, and halve it
# for a 7B pair.
BATCH_SIZE=${BATCH_SIZE:-8}
LIMIT=${LIMIT:-0}                  # 0 = all rows; small values for a smoke run

CODE_DIR=${CODE_DIR:-$HOME/LLM}
LOG_ROOT=${LOG_ROOT:-$HOME/LLM/Train/verl/logs}
EXP_NAME=${EXP_NAME:-joint_$(date +%m%d_%H%M)}

if [ -z "$MODEL_A" ] || [ -z "$MODEL_B" ]; then
    echo "set MODEL_A and MODEL_B (pass the same path twice to fuse a model with itself)" >&2
    exit 1
fi
if [ ! -f "$EVAL_PATH" ]; then
    echo "eval parquet not found: $EVAL_PATH" >&2
    echo "build it with: python3 Data/prepare_rephrase_eval.py --input <rendered>.parquet --output $EVAL_PATH" >&2
    exit 1
fi

GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")

# Label the models the way sweep_sft_checkpoints.sh does -- the label also becomes a
# directory name, so keep it to the recognisable part of the path.
NAME_A=$(basename "$MODEL_A")
NAME_B=$(basename "$MODEL_B")
echo "model names for labels: A=$NAME_A  B=$NAME_B"

SWEEP_DIR=${LOG_ROOT}/${EXP_NAME}
mkdir -p "$SWEEP_DIR"
SUMMARY="${SWEEP_DIR}/summary.tsv"
printf 'label\tavg_score\tmax_score\tavg_len_tokens\toutput_dir\n' > "$SUMMARY"

# The swept variable depends on the mode: a mixing weight for the blending modes,
# a min-prob floor for the selection mode.
if [ "$FUSE" = "agree" ]; then
    IFS=',' read -r -a SWEEP_LIST <<< "$AGREE_TEACHER_MIN_PROBS"
    SWEEP_LABEL="teacher_min_prob"
else
    IFS=',' read -r -a SWEEP_LIST <<< "$WEIGHTS"
    SWEEP_LABEL="weight"
fi
echo "=== joint decoding: ${#SWEEP_LIST[@]} ${SWEEP_LABEL}(s) on $(basename "$EVAL_PATH"), fuse=$FUSE, prompt_key=$PROMPT_KEY ==="
echo "  A: $MODEL_A  (student: picks within the allowed set)"
echo "  B: $MODEL_B  (teacher: constrains the allowed set)"
echo "  ${SWEEP_LABEL}s: ${SWEEP_LIST[*]}   (${GPU_NUM}-way data parallel)"
if [ "$FUSE" = "agree" ]; then
    echo "  agree_top_k=$AGREE_TOP_K, student_min_prob=$AGREE_STUDENT_MIN_PROB, temperature=$TEMPERATURE"
fi

cd "$CODE_DIR" || exit 1
echo "change to dir: $PWD"

for value in "${SWEEP_LIST[@]}"; do
    # 0.5 -> w0.5 / mp0.5; keeps the label filesystem-safe and sortable.
    if [ "$FUSE" = "agree" ]; then
        label="${NAME_A}-x-${NAME_B}-agree-k${AGREE_TOP_K}-tmp${value}"
    else
        label="${NAME_A}-x-${NAME_B}-${FUSE}-w${value}"
    fi
    out_dir="${SWEEP_DIR}/${label}/save_data"
    log_path="${SWEEP_DIR}/${label}.log"
    mkdir -p "$out_dir"

    extra_args=()
    if [ -n "$PROMPT_KEY_B" ]; then
        extra_args+=(--prompt-key-b "$PROMPT_KEY_B")
    fi
    if [ "$LIMIT" -gt 0 ]; then
        extra_args+=(--limit "$LIMIT")
    fi
    # Pass only the knobs the chosen mode actually reads, so joint_decode.py's
    # "these flags are ignored" warning stays meaningful. TEMPERATURE/TOP_P go to
    # both modes: agree samples too, on the student branch and the fallback alike.
    if [ "$FUSE" = "agree" ]; then
        extra_args+=(--agree-top-k "$AGREE_TOP_K"
                     --agree-teacher-min-prob "$value"
                     --agree-student-min-prob "$AGREE_STUDENT_MIN_PROB"
                     --temperature "$TEMPERATURE" --top-p "$TOP_P")
    else
        extra_args+=(--fuse-weight "$value"
                     --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
    fi

    echo "=== [$label] ${SWEEP_LABEL}=$value ==="
    # One process per GPU, each writing rank<N>.parquet into out_dir. The metrics
    # block below globs the directory, so no merge step is needed -- same as
    # main_generation writing one parquet per batch.
    torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_NUM" \
        Data/joint_decode.py \
        --model-a "$MODEL_A" \
        --model-b "$MODEL_B" \
        --input "$EVAL_PATH" \
        --output-dir "$out_dir" \
        --prompt-key "$PROMPT_KEY" \
        --fuse "$FUSE" \
        --n-samples "$N_SAMPLES" \
        --prompt-length "$PROMPT_LENGTH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --batch-size "$BATCH_SIZE" \
        "${extra_args[@]}" 2>&1 | tee "$log_path"

    # Recompute the metrics from the written parquets rather than scraping the log:
    # each rank prints only its own shard's mean, so the printed numbers are
    # per-rank and a short final shard would be weighted like a full one.
    #   avg_score -- mean of per-question mean_score, i.e. overall pass rate
    #   max_score -- mean of per-question max_score, i.e. pass@N
    #   avg_len   -- mean response length in TOKENS. Compare it against
    #                MAX_NEW_TOKENS to see whether generations are hitting the cap.
    metrics=$(python - "$out_dir" <<'PYEOF'
import glob
import sys

import pandas as pd
import pyarrow.parquet as pq

files = sorted(glob.glob(f"{sys.argv[1]}/*.parquet"))
means, maxes, lengths = [], [], []
missing_lengths = False
for path in files:
    # response_lengths is the per-response valid token count; a parquet written
    # without it has to be reported as NA rather than silently measured in
    # characters, which reads ~3x larger.
    columns = ["test_score"]
    if "response_lengths" in pq.read_schema(path).names:
        columns.append("response_lengths")
    else:
        missing_lengths = True
    df = pd.read_parquet(path, columns=columns)
    for score in df["test_score"]:
        means.append(float(score["mean_score"]))
        maxes.append(float(score["max_score"]))
    if "response_lengths" in df:
        for row in df["response_lengths"]:
            lengths.extend(int(n) for n in row)

if not means:
    print("NA\tNA\tNA")
else:
    avg_len = f"{sum(lengths) / len(lengths):.0f}" if lengths and not missing_lengths else "NA"
    print(f"{sum(means) / len(means):.4f}\t{sum(maxes) / len(maxes):.4f}\t{avg_len}")
PYEOF
)
    printf '%s\t%s\t%s\n' "$label" "$metrics" "$out_dir" >> "$SUMMARY"
    echo "=== [$label] avg_score / max_score / avg_len_tokens: $metrics ==="
done

echo
echo "=== joint-decode summary ($SUMMARY) ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
if [ "$FUSE" = "agree" ]; then
    echo "reminder: check the 'teacher fallback' rate in the per-run logs"
    echo "  ($SWEEP_DIR/*.log). Near 100% = the overlap was always empty, so the student"
    echo "  never chose and this is plain teacher sampling; near 0% at tmp0 = the allowed"
    echo "  set covers everything, so it is plain student sampling. Only the middle range"
    echo "  actually tests 'teacher steers, student picks' -- tune AGREE_TEACHER_MIN_PROBS"
    echo "  and AGREE_TOP_K until the rate lands there."
else
    echo "reminder: the w0 row must match a single-model run of A. With TEMPERATURE=0,"
    echo "compare it against sweep_sft_checkpoints.sh on the same EVAL_PATH -- if they"
    echo "differ, the decode loop is wrong and every other row is meaningless."
fi
