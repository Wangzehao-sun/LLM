set -x
#!/usr/bin/env bash
# GPU selection. Override with, for example: GPU_DEVICES=4,5,6,7

CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME:-verl}



GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
echo $HOME
export TOKENIZERS_PARALLELISM=true



# The model being steered, and the model mixed into it.
MODEL_A=${MODEL_A:-"/apdcephfs_qy3/share_301372554/share_info/zenohaowang/Model/Qwen3-4B-Base"}
MODEL_B=${MODEL_B:-"/apdcephfs_qy3/share_301372554/share_info/zenohaowang/Model/Qwen3-4B-Instruct-2507"}
EVAL_PATH=${EVAL_PATH:-"/apdcephfs_qy3/share_301372554/share_info/zenohaowang/LLM/Data/dapo_math/dapo_en_math_solution_9k_summarize_train_val128_flat.parquet"}

# Fusion weights on model B, one run each. 0 must stay in the list (see header).
# Ignored when FUSE=agree, which sweeps AGREE_TEACHER_MIN_PROBS instead.
WEIGHTS=${WEIGHTS:-0,0.25,0.5,0.75,1.0}
FUSE=${FUSE:-agree}               # linear | contrastive | max | agree

AGREE_TEACHER_MIN_PROBS=${AGREE_TEACHER_MIN_PROBS:-0.01}
AGREE_TOP_K=${AGREE_TOP_K:-10}
# Per-side overrides, both defaulting to AGREE_TOP_K. They mean different things: the
# teacher's bounds how much of the vocabulary it permits at all, the student's bounds how
# far down its own ranking it will look for something permitted. So raising only the
# student's is the cheapest way to cut the fallback rate -- it reaches further for an
# agreed token without widening what the teacher allows.
AGREE_STUDENT_TOP_K=${AGREE_STUDENT_TOP_K:-50}
AGREE_TEACHER_TOP_K=${AGREE_TEACHER_TOP_K:-10}
# Widen the teacher's k PER TOKEN where the fixed set keeps too little of the student's mass:
#   1. intersect as usual (AGREE_TEACHER_TOP_K, teacher min-prob) and compute Z_t;
#   2. tokens with Z_t below AGREE_TARGET_Z widen the teacher's k (no min-prob floor) up to
#      AGREE_DYNAMIC_MAX_TOP_K, taking the smallest k that reaches the target;
#   3. tokens still short at the cap fall back (AGREE_FALLBACK).
# Every non-fallback token then keeps at least the target of the student's mass. The fallback
# rate can be HIGHER than at the fixed width: step 3 falls back non-empty sets below target.
#
# 1 also SWITCHES WHAT THIS SCRIPT SWEEPS: the sweep runs over AGREE_TARGET_Z, and the teacher
# min-prob of the fixed set is the FIRST value of AGREE_TEACHER_MIN_PROBS.
AGREE_DYNAMIC_TOP_K=${AGREE_DYNAMIC_TOP_K:-0}
# Comma-separated: the sweep list when AGREE_DYNAMIC_TOP_K=1. Read the summary's fallback
# column against it: it counts the tokens the cap could not bring up to the target.
AGREE_TARGET_Z=${AGREE_TARGET_Z:-0.5}
# Ceiling on how far the teacher's k may widen; a token still below target here falls back.
# Also the cost knob: the search works over a [rows, this] slice.
AGREE_DYNAMIC_MAX_TOP_K=${AGREE_DYNAMIC_MAX_TOP_K:-20}
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

N_SAMPLES=${N_SAMPLES:-1}          # samples per question; >1 to see sampling variance
TEMPERATURE=${TEMPERATURE:-1}
TOP_P=${TOP_P:-1}
TOP_K=${TOP_K:--1}
PROMPT_LENGTH=${PROMPT_LENGTH:-2048}      # rephrase prompts carry a draft, so longer than a bare question
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
# Far smaller than the vLLM path's 256: there is no request-level scheduling here,
# so the longest row in a batch holds up every other row in it.
#
# Memory, not that long tail, is the hard ceiling. Both models' weights are fixed
# (~16GB for a 4B pair) but KV cache grows linearly with this -- roughly 2.4GB per
# row for a 4B pair at prompt+response 8192. On an 80GB card that puts the limit
# around 20; past that expect OOM. Raise it while watching nvidia-smi, and halve it
# for a 7B pair.
BATCH_SIZE=${BATCH_SIZE:-64}
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

# Screen each response's FORMAT on top of math-verify's answer check. The rule is the
# trainer's; these word lists are the generation side's own and do not affect training.
# Comma-separated; unset a variable to fall back to the trainer's default list, or set it
# empty to disable that category.
TRAJ_FILTER=${TRAJ_FILTER:-0}
TRAJ_KEYWORDS=${TRAJ_KEYWORDS-"the draft,based on the draft,according to the draft,from the draft,the experience,based on the experience,according to the experience,the provided reasoning,based on the reasoning above"}
TRAJ_INSTR_PHRASES=${TRAJ_INSTR_PHRASES-"your task is,output only,do not mention,self-contained solution,no meta-talk"}


WORKER_DIR="/apdcephfs_qy3/share_301372554/share_info/zenohaowang"
# Myverl/, so that `-m verl.PICO.joint_decode` resolves the package.
CODE_DIR=${CODE_DIR:-$WORKER_DIR/LLM/Myverl}
LOG_ROOT=${LOG_ROOT:-$WORKER_DIR/LLM/logs/joint_decoding}
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
printf 'label\tavg_score\tmax_score\tavg_len_tokens\tlen_ok\tlen_bad\tfallback\tkeep_ratio\tkeep_low20\tstudent_prob\tans_err\tfmt_err\tfmt_ok_ans\toutput_dir\n' > "$SUMMARY"

# The swept variable depends on the mode: a mixing weight for the blending modes, a
# min-prob floor for the selection mode -- or the Z_t target under the dynamic path, with
# the floor of its fixed first set held at the first AGREE_TEACHER_MIN_PROBS value.
if [ "$FUSE" = "agree" ] && [ "$AGREE_DYNAMIC_TOP_K" = "1" ]; then
    IFS=',' read -r -a SWEEP_LIST <<< "$AGREE_TARGET_Z"
    SWEEP_LABEL="target_z"
    DYN_TEACHER_MIN_PROB="${AGREE_TEACHER_MIN_PROBS%%,*}"
elif [ "$FUSE" = "agree" ]; then
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
    if [ "$AGREE_DYNAMIC_TOP_K" = "1" ]; then
        echo "  student_top_k=${AGREE_STUDENT_TOP_K:-$AGREE_TOP_K}, teacher_top_k=${AGREE_TEACHER_TOP_K:-$AGREE_TOP_K} (teacher_min_prob=$DYN_TEACHER_MIN_PROB), widened to <=$AGREE_DYNAMIC_MAX_TOP_K where Z_t<target, student_min_prob=$AGREE_STUDENT_MIN_PROB, temperature=$TEMPERATURE, fallback=$AGREE_FALLBACK"
    else
        echo "  student_top_k=${AGREE_STUDENT_TOP_K:-$AGREE_TOP_K}, teacher_top_k=${AGREE_TEACHER_TOP_K:-$AGREE_TOP_K}, student_min_prob=$AGREE_STUDENT_MIN_PROB, temperature=$TEMPERATURE, fallback=$AGREE_FALLBACK"
    fi
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
        if [ "$AGREE_DYNAMIC_TOP_K" = "1" ]; then
            # tz, not tmp: the swept value is a Z_t target here, and a fixed-width run and a
            # dynamic one at the same numeric value are different experiments that must not
            # collide in one EXP_NAME. The fixed set's floor goes in the label as well.
            label="${NAME_A}-x-${NAME_B}-agree-dyn${AGREE_DYNAMIC_MAX_TOP_K}-${k_tag}-tmp${DYN_TEACHER_MIN_PROB}-tz${value}-fb${AGREE_FALLBACK}"
        else
            label="${NAME_A}-x-${NAME_B}-agree-${k_tag}-tmp${value}-fb${AGREE_FALLBACK}"
        fi
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
    # Pass only the knobs the chosen mode actually reads, so verl/PICO/joint_decode.py's
    # "these flags are ignored" warning stays meaningful. TEMPERATURE/TOP_P go to
    # both modes: agree samples too, on the student branch and the fallback alike.
    if [ "$FUSE" = "agree" ]; then
        extra_args+=(--agree-top-k "$AGREE_TOP_K"
                     --agree-student-min-prob "$AGREE_STUDENT_MIN_PROB"
                     --agree-fallback "$AGREE_FALLBACK"
                     --temperature "$TEMPERATURE" --top-p "$TOP_P")
        if [ "$AGREE_DYNAMIC_TOP_K" = "1" ]; then
            # The swept value is the Z_t target in this mode; the fixed first set keeps the
            # teacher min-prob floor (the first AGREE_TEACHER_MIN_PROBS value).
            extra_args+=(--agree-dynamic-top-k
                         --agree-target-z "$value"
                         --agree-dynamic-max-top-k "$AGREE_DYNAMIC_MAX_TOP_K"
                         --agree-teacher-min-prob "$DYN_TEACHER_MIN_PROB")
        else
            extra_args+=(--agree-teacher-min-prob "$value")
        fi
        # Empty means "inherit AGREE_TOP_K"; verl/PICO/joint_decode.py resolves that, so only pass
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

    if [ "$TRAJ_FILTER" = "1" ]; then
        extra_args+=(--traj-filter)
        # Passed only when set, so an unset variable inherits the trainer's default list
        # rather than pinning a copy of it here.
        if [ -n "${TRAJ_KEYWORDS+x}" ]; then
            extra_args+=(--traj-keywords "$TRAJ_KEYWORDS")
        fi
        if [ -n "${TRAJ_INSTR_PHRASES+x}" ]; then
            extra_args+=(--traj-instruction-phrases "$TRAJ_INSTR_PHRASES")
        fi
    fi

    echo "=== [$label] ${SWEEP_LABEL}=$value ==="
    # One process per GPU, each writing rank<N>.parquet into out_dir. The metrics
    # block below globs the directory, so no merge step is needed -- same as
    # main_generation writing one parquet per batch.
    torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_NUM" \
        -m verl.PICO.joint_decode \
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
    #   keep_low20    -- mean Z_t over each row's WORST 20% of tokens, then averaged over
    #                    rows. The row mean above hides exactly the steps the constraint is
    #                    doing its damage at: a run can average 0.6 while a fifth of its
    #                    steps sit near 0, and those are the ones that drive the off-policy
    #                    term to zero. Read the gap between the two columns -- a wide one
    #                    means the constraint is concentrated on a few tokens rather than
    #                    spread evenly, which is what dynamic top-k is meant to fix.
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
means, maxes, lengths, fbs, logps, keeps, keeps_low = [], [], [], [], [], [], []
len_ok, len_bad = [], []
missing_lengths = False
n_resp = n_answer_err = n_format_err = n_fmt_ok_ans = 0
have_breakdown = False


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
                                            "teacher_keep_ratio",
                                            "teacher_keep_ratio_low") if c in present]
    if "response_lengths" not in present:
        missing_lengths = True
    df = pd.read_parquet(path, columns=columns)
    for score in df["test_score"]:
        means.append(float(score["mean_score"]))
        maxes.append(float(score["max_score"]))
        # Only present when --traj-filter ran; keyed on the data so an older parquet
        # reports NA instead of failing.
        rejected = score.get("format_rejected") if hasattr(score, "get") else None
        raw = score.get("raw_scores_per_response") if hasattr(score, "get") else None
        if rejected is None or raw is None:
            continue
        have_breakdown = True
        for rej, raw_score in zip(rejected, raw):
            n_resp += 1
            if rej:
                n_format_err += 1
                if _correct(raw_score):
                    n_fmt_ok_ans += 1
            elif not _correct(raw_score):
                n_answer_err += 1
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
    if "teacher_keep_ratio_low" in df:
        keeps_low.extend(v for row in df["teacher_keep_ratio_low"] for v in row if _usable(v))

if not means:
    print("NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
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
    krl = f"{sum(keeps_low) / len(keeps_low):.4f}" if keeps_low else "NA"
    # exp of the mean per-token log-prob: a geometric mean, so it is not skewed by
    # response length the way a plain mean of probabilities would be.
    sp = f"{math.exp(sum(logps) / len(logps)):.4f}" if logps else "NA"
    if have_breakdown and n_resp:
        ans_err = f"{n_answer_err / n_resp:.4f}"
        fmt_err = f"{n_format_err / n_resp:.4f}"
        fmt_ok = f"{n_fmt_ok_ans / n_resp:.4f}"
    else:
        ans_err = fmt_err = fmt_ok = "NA"
    print(f"{sum(means) / len(means):.4f}\t{sum(maxes) / len(maxes):.4f}\t{avg_len}\t"
          f"{ok_len}\t{bad_len}\t{fb}\t{kr}\t{krl}\t{sp}\t{ans_err}\t{fmt_err}\t{fmt_ok}")
PYEOF
)
    printf '%s\t%s\t%s\n' "$label" "$metrics" "$out_dir" >> "$SUMMARY"
    echo "=== [$label] score/max/len/len_ok/len_bad/fallback/keep_ratio/keep_low20/student_prob/ans_err/fmt_err/fmt_ok_ans: $metrics ==="
done

echo
echo "=== joint-decode summary ($SUMMARY) ==="
column -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
if [ "$FUSE" = "agree" ] && [ "$AGREE_DYNAMIC_TOP_K" = "1" ]; then
    echo "reminder: read the 'fallback' column above -- under AGREE_DYNAMIC_TOP_K it counts the"
    echo "  tokens whose set stayed below each row's tz target even at"
    echo "  AGREE_DYNAMIC_MAX_TOP_K=$AGREE_DYNAMIC_MAX_TOP_K (plus empty intersections), which"
    echo "  AGREE_FALLBACK=$AGREE_FALLBACK then decided. Every other token keeps at least tz of"
    echo "  the student's mass. A high fallback means the cap is too low for the target: raise"
    echo "  the cap or lower tz. It can sit ABOVE a fixed-width run's fallback at the same"
    echo "  teacher_top_k, since sets that were non-empty but below target now fall back."
    echo "  'keep_ratio' is the mean Z_t with fallback tokens counted as 0, so it sits below tz"
    echo "  whenever anything fell back. 'keep_low20' is the mean Z_t over each row's worst 20%"
    echo "  of tokens. 'student_prob' is the geometric-mean p_student of the emitted tokens:"
    echo "  read it next to avg_score to see what the constraint bought or cost."
elif [ "$FUSE" = "agree" ]; then
    echo "reminder: read the 'fallback' column above -- it is the share of tokens where the"
    echo "  constraint left no candidate and AGREE_FALLBACK=$AGREE_FALLBACK decided instead."
    echo "  Near 1.0 = the overlap was almost always empty, so the run collapsed to plain"
    echo "  $AGREE_FALLBACK decoding; near 0.0 at tmp0 = the allowed set covers everything,"
    echo "  so it is plain student decoding. Only the middle range actually tests"
    echo "  'teacher steers, student picks' -- tune AGREE_TEACHER_MIN_PROBS, or raise"
    echo "  AGREE_STUDENT_TOP_K to let the student reach further for an agreed token"
    echo "  without widening what the teacher permits."
    echo "  until it lands there."
    echo "  'keep_low20' is the mean Z_t over each row's worst 20% of tokens. A wide gap"
    echo "  between it and keep_ratio means the constraint is concentrated on a few tokens"
    echo "  rather than spread evenly -- try AGREE_DYNAMIC_TOP_K=1, which widens the"
    echo "  teacher's set only at those tokens."
    echo "  'student_prob' is the geometric-mean p_student of the emitted tokens: it falls"
    echo "  as the constraint pushes the student off its own preferences, so read it next"
    echo "  to avg_score to see what that constraint bought or cost."
else
    echo "reminder: the w0 row must match a single-model run of A. With TEMPERATURE=0,"
    echo "compare it against sweep_sft_checkpoints.sh on the same EVAL_PATH -- if they"
    echo "differ, the decode loop is wrong and every other row is meaningless."
fi

