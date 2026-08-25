set -x
#!/usr/bin/env bash
# GPU selection. Override with, for example: GPU_DEVICES=4,5,6,7
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
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
# For FUSE=agree the analogous check is the summary's "fallback" column: the share of
# tokens where the constraint left no candidate, so AGREE_FALLBACK decided instead. Near
# 1.0 means the overlap was almost always empty and the run collapsed to plain decoding
# by whoever owns the fallback -- so the scores say nothing about the method. Near 0.0
# with a floor of 0 means the allowed set covers everything, which makes it plain student
# decoding. The method only does work in between. The "student_prob" column is the other
# half of the picture: it falls as the constraint pushes the student off its own
# preferences, so it prices what the constraint cost.
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
#   # let the student reach further for an agreed token without widening the teacher's
#   # allowed set -- the cheapest way to cut a high fallback rate
#   MODEL_A=... MODEL_B=... FUSE=agree \
#       AGREE_STUDENT_TOP_K=50 AGREE_TEACHER_TOP_K=10 bash run_joint_decode.sh
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
MODEL_A=${MODEL_A:-"/home/data/shared/Qwen3-4b-base"}
MODEL_B=${MODEL_B:-"/home/data/shared/Qwen3-4B-Instruct"}
EVAL_PATH=${EVAL_PATH:-$HOME/LLM/Data/openr1/openr1_hard_mix64_restore_solonly_train_val128_flat.parquet}

# Fusion weights on model B, one run each. 0 must stay in the list (see header).
# Ignored when FUSE=agree, which sweeps AGREE_TEACHER_MIN_PROBS instead.
WEIGHTS=${WEIGHTS:-0,0.25,0.5,0.75,1.0}
FUSE=${FUSE:-agree}               # linear | contrastive | max | agree

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
AGREE_TEACHER_MIN_PROBS=${AGREE_TEACHER_MIN_PROBS:-0.05}
AGREE_TOP_K=${AGREE_TOP_K:-10}
# Per-side overrides, both defaulting to AGREE_TOP_K. They mean different things: the
# teacher's bounds how much of the vocabulary it permits at all, the student's bounds how
# far down its own ranking it will look for something permitted. So raising only the
# student's is the cheapest way to cut the fallback rate -- it reaches further for an
# agreed token without widening what the teacher allows.
AGREE_STUDENT_TOP_K=${AGREE_STUDENT_TOP_K:-50}
AGREE_TEACHER_TOP_K=${AGREE_TEACHER_TOP_K:-8}
# Only raise this to veto tokens the student is very reluctant to emit -- it
# already ranks the survivors, so 0 is the natural default.
AGREE_STUDENT_MIN_PROB=${AGREE_STUDENT_MIN_PROB:-0}
# Who decides a step where the constraint leaves no candidate. These are two different
# experiments:
#   teacher -- it overrides the student exactly where they disagree. The aggressive
#              reading of "the teacher steers".
#   student -- the student carries on alone, so the teacher only ever NARROWS its
#              choices and never overrides. Output stays in the student's voice.
# It also flips how the 'fallback' column reads: a high rate means the run collapsed
# to teacher decoding in the first case, to student decoding in the second.
AGREE_FALLBACK=${AGREE_FALLBACK:-student}

# Which prompt column each model sees. prepare_rephrase_eval.py writes both:
#   prompt          -> the rephrase task (question + expert-reasoning draft)
#   question_prompt -> the bare question, i.e. plain problem-solving ability
# PROMPT_KEY_B empty = model B reads the same column as A, so the fusion is purely
# a model difference; set it to feed B a richer prompt than A instead.
PROMPT_KEY=${PROMPT_KEY:-question_prompt}
PROMPT_KEY_B=${PROMPT_KEY_B:-prompt}

N_SAMPLES=${N_SAMPLES:-4}          # samples per question; >1 to see sampling variance
TEMPERATURE=${TEMPERATURE:-1}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
PROMPT_LENGTH=${PROMPT_LENGTH:-4096}      # rephrase prompts carry a draft, so longer than a bare question
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
# Far smaller than the vLLM path's 256: there is no request-level scheduling here,
# so the longest row in a batch holds up every other row in it.
#
# Memory, not that long tail, is the hard ceiling. Both models' weights are fixed
# (~16GB for a 4B pair) but KV cache grows linearly with this -- roughly 2.4GB per
# row for a 4B pair at prompt+response 8192. On an 80GB card that puts the limit
# around 20; past that expect OOM. Raise it while watching nvidia-smi, and halve it
# for a 7B pair.
BATCH_SIZE=${BATCH_SIZE:-32}
LIMIT=${LIMIT:-0}                  # 0 = all rows; small values for a smoke run

# --- throughput knobs ------------------------------------------------------
# Attention kernel, the single biggest lever here: "eager" rebuilds the whole
# [B, heads, 1, kv_len] score matrix every step and layer, which at an 8k context
# across two models is the same order of traffic as the weights themselves.
#
# Defaults to flash_attention_2 and FAILS if it is unavailable, rather than quietly
# using something slower -- flash-attn is already in Myverl/setup.py's GPU extra, so
# an environment set up by scripts/setup_env.sh has it. Set ATTN_IMPL=sdpa for the
# fused PyTorch kernel (no extra install), or =auto to probe and take what loads.
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
# Steps between all-rows-finished checks. Each one synchronises with the host, so
# polling every step stalls the GPU more than the <=N-1 all-padding steps a coarser
# check may run (those tokens get trimmed anyway). Also the cadence at which finished
# rows leave the batch.
EOS_CHECK_EVERY=${EOS_CHECK_EVERY:-16}
# Drop rows from the batch once they hit EOS. Generation lengths vary enormously, so a
# batch of 8 typically has only ~a third of its row-steps doing real work, and every
# carried row still pays its share of the KV read -- the biggest per-step cost at a 10k
# context. Set to 0 to keep the full batch, which is only useful for isolating this.
SHRINK_BATCH=${SHRINK_BATCH:-1}

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
printf 'label\tavg_score\tmax_score\tavg_len_tokens\tlen_ok\tlen_bad\tfallback\tkeep_ratio\tstudent_prob\toutput_dir\n' > "$SUMMARY"

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
    echo "  student_top_k=${AGREE_STUDENT_TOP_K:-$AGREE_TOP_K}, teacher_top_k=${AGREE_TEACHER_TOP_K:-$AGREE_TOP_K}, student_min_prob=$AGREE_STUDENT_MIN_PROB, temperature=$TEMPERATURE, fallback=$AGREE_FALLBACK"
fi

cd "$CODE_DIR" || exit 1
echo "change to dir: $PWD"

for value in "${SWEEP_LIST[@]}"; do
    # 0.5 -> w0.5 / mp0.5; keeps the label filesystem-safe and sortable. The fallback
    # owner is in there because the two settings are different experiments on identical
    # knobs -- without it, comparing them in one EXP_NAME would overwrite the first run.
    if [ "$FUSE" = "agree" ]; then
        # The per-side k's go in the label because two runs differing only in them would
        # otherwise share an output directory and the second would overwrite the first.
        k_tag="k${AGREE_STUDENT_TOP_K:-$AGREE_TOP_K}.${AGREE_TEACHER_TOP_K:-$AGREE_TOP_K}"
        label="${NAME_A}-x-${NAME_B}-agree-${k_tag}-tmp${value}-fb${AGREE_FALLBACK}"
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
    if [ "$TEACHER_DROP_PREFILL" != "0" ]; then
        extra_args+=(--teacher-drop-prefill)
    fi
    if [ "$SHRINK_BATCH" = "0" ]; then
        extra_args+=(--no-shrink-batch)
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
                     --agree-fallback "$AGREE_FALLBACK"
                     --temperature "$TEMPERATURE" --top-p "$TOP_P")
        # Empty means "inherit AGREE_TOP_K"; joint_decode.py resolves that, so only pass
        # the flag when it was actually set.
        if [ -n "$AGREE_STUDENT_TOP_K" ]; then
            extra_args+=(--agree-student-top-k "$AGREE_STUDENT_TOP_K")
        fi
        if [ -n "$AGREE_TEACHER_TOP_K" ]; then
            extra_args+=(--agree-teacher-top-k "$AGREE_TEACHER_TOP_K")
        fi
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
        --attn-impl "$ATTN_IMPL" \
        --eos-check-every "$EOS_CHECK_EVERY" \
        "${extra_args[@]}" 2>&1 | tee "$log_path"

    # Recompute the metrics from the written parquets rather than scraping the log:
    # each rank prints only its own shard's mean, so the printed numbers are
    # per-rank and a short final shard would be weighted like a full one.
    #   avg_score     -- mean of per-question mean_score, i.e. overall pass rate
    #   max_score     -- mean of per-question max_score, i.e. pass@N
    #   avg_len       -- mean response length in TOKENS. Compare it against
    #                    MAX_NEW_TOKENS to see whether generations are hitting the cap.
    #   len_ok        -- mean length of CORRECT responses
    #   len_bad       -- mean length of WRONG responses. Split out because the two
    #                    usually differ sharply and the combined mean hides it: wrong
    #                    answers are often the ones that ran to the cap without
    #                    converging, so len_bad near MAX_NEW_TOKENS with len_ok well
    #                    below it says the failures are non-termination rather than bad
    #                    reasoning -- a different problem with a different fix.
    #   fallback      -- share of emitted tokens where the constraint left no candidate,
    #                    so AGREE_FALLBACK decided instead. 0 outside --fuse agree. This
    #                    is the column that says whether the method did anything: ~1.0 is
    #                    plain decoding by the fallback owner, ~0.0 at a zero floor is
    #                    plain student decoding.
    #   keep_ratio    -- mean Z_t, the share of the STUDENT's probability mass the
    #                    teacher left standing at each step (sum of p_student over the
    #                    eligible set). The most direct measure of how hard the
    #                    constraint bites: 1.0 means the teacher permitted everything the
    #                    student cared about, near 0 means the student is being pushed
    #                    onto tokens it thought unlikely. Fallback steps count as 0.
    #   student_prob  -- geometric-mean p_student of the emitted tokens. Falls as the
    #                    constraint pushes the student off its own preferences, so read
    #                    it against avg_score to see what the accuracy cost.
    metrics=$(python - "$out_dir" <<'PYEOF'
import glob
import math
import sys

import pandas as pd
import pyarrow.parquet as pq

files = sorted(glob.glob(f"{sys.argv[1]}/*.parquet"))
means, maxes, lengths, fbs, logps, keeps = [], [], [], [], [], []
len_ok, len_bad = [], []
missing_lengths = False


def _usable(v):
    return v is not None and not math.isnan(v)


def _correct(score):
    """Match main_generation's notion of a passing score: truthy, or >= 1 numerically."""
    if isinstance(score, bool):
        return score
    try:
        return float(score) >= 1.0
    except (TypeError, ValueError):
        return bool(score)


for path in files:
    # These columns are all optional: a parquet written before they existed has to
    # report NA rather than a silently wrong number (response_lengths in particular
    # would otherwise fall back to characters, which reads ~3x larger).
    present = set(pq.read_schema(path).names)
    columns = ["test_score"] + [c for c in ("response_lengths", "fallback_frac",
                                            "student_mean_logp",
                                            "teacher_keep_ratio") if c in present]
    if "response_lengths" not in present:
        missing_lengths = True
    df = pd.read_parquet(path, columns=columns)
    for score in df["test_score"]:
        means.append(float(score["mean_score"]))
        maxes.append(float(score["max_score"]))
    if "response_lengths" in df:
        for row in df["response_lengths"]:
            lengths.extend(int(n) for n in row)
        # scores_per_response and response_lengths are per-sample lists in the same
        # order, so they pair by index -- no extra column needed. zip() also guards the
        # case where one is shorter, which would otherwise misalign silently.
        for score, row in zip(df["test_score"], df["response_lengths"]):
            for ok, n in zip(score["scores_per_response"], row):
                (len_ok if _correct(ok) else len_bad).append(int(n))
    # NaN marks a response that emitted no tokens, so there was nothing to average
    # over. Arrow stores those as nulls, which read back as None rather than NaN, so
    # both have to be filtered -- math.isnan(None) is a TypeError, not False.
    if "fallback_frac" in df:
        fbs.extend(v for row in df["fallback_frac"] for v in row if _usable(v))
    if "student_mean_logp" in df:
        logps.extend(v for row in df["student_mean_logp"] for v in row if _usable(v))
    if "teacher_keep_ratio" in df:
        keeps.extend(v for row in df["teacher_keep_ratio"] for v in row if _usable(v))

if not means:
    print("NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
else:
    def _mean_len(vals):
        # NA rather than 0 when a bucket is empty: an all-correct run has no wrong
        # responses to measure, and 0 would read as "the wrong ones were empty".
        return f"{sum(vals) / len(vals):.0f}" if vals and not missing_lengths else "NA"

    avg_len = _mean_len(lengths)
    ok_len = _mean_len(len_ok)
    bad_len = _mean_len(len_bad)
    fb = f"{sum(fbs) / len(fbs):.4f}" if fbs else "NA"
    kr = f"{sum(keeps) / len(keeps):.4f}" if keeps else "NA"
    # exp of the mean per-token log-prob: a geometric mean, so it is not skewed by
    # response length the way a plain mean of probabilities would be.
    sp = f"{math.exp(sum(logps) / len(logps)):.4f}" if logps else "NA"
    print(f"{sum(means) / len(means):.4f}\t{sum(maxes) / len(maxes):.4f}\t{avg_len}\t"
          f"{ok_len}\t{bad_len}\t{fb}\t{kr}\t{sp}")
PYEOF
)
    printf '%s\t%s\t%s\n' "$label" "$metrics" "$out_dir" >> "$SUMMARY"
    echo "=== [$label] score/max/len/len_ok/len_bad/fallback/keep_ratio/student_prob: $metrics ==="
done

echo
echo "=== joint-decode summary ($SUMMARY) ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
if [ "$FUSE" = "agree" ]; then
    echo "reminder: read the 'fallback' column above -- it is the share of tokens where the"
    echo "  constraint left no candidate and AGREE_FALLBACK=$AGREE_FALLBACK decided instead."
    echo "  Near 1.0 = the overlap was almost always empty, so the run collapsed to plain"
    echo "  $AGREE_FALLBACK decoding; near 0.0 at tmp0 = the allowed set covers everything,"
    echo "  so it is plain student decoding. Only the middle range actually tests"
    echo "  'teacher steers, student picks' -- tune AGREE_TEACHER_MIN_PROBS, or raise"
    echo "  AGREE_STUDENT_TOP_K to let the student reach further for an agreed token"
    echo "  without widening what the teacher permits."
    echo "  until it lands there."
    echo "  'student_prob' is the geometric-mean p_student of the emitted tokens: it falls"
    echo "  as the constraint pushes the student off its own preferences, so read it next"
    echo "  to avg_score to see what that constraint bought or cost."
else
    echo "reminder: the w0 row must match a single-model run of A. With TEMPERATURE=0,"
    echo "compare it against sweep_sft_checkpoints.sh on the same EVAL_PATH -- if they"
    echo "differ, the decode loop is wrong and every other row is meaningless."
fi
