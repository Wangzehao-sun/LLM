#!/usr/bin/env bash
#
# Idempotent end-to-end data pipeline for this repo.
#
# Each step writes one parquet and skips itself if that parquet already
# exists, so re-running this script is safe.
#
# Pipeline:
#   1. Data/prepare_deepmath.py            -> deepmath.parquet           (~720 MB)
#   2. Data/filter_deepmath.py             -> deepmath_dgt6_n10000.parquet
#   3. Data/add_token_split_points.py      -> deepmath_dgt6_n10000_split.parquet
#   4. Data/prepare_summarize_prompts.py   -> deepmath_dgt6_n10000_summarize.parquet
#
# Usage:
#   bash scripts/prepare_data.sh
#   TOKENIZER_PATH=/path/to/model bash scripts/prepare_data.sh
#   WORKERS=8 bash scripts/prepare_data.sh
#   FORCE=1 bash scripts/prepare_data.sh        # re-run all steps even if outputs exist
#
# Pre-conditions:
#   - conda env with this repo's deps active (run scripts/setup_env.sh first)
#   - $TOKENIZER_PATH points to a tokenizer matching the training model
#     (defaults to /home/shared/Qwen2.5-Math-7B-16k-think)
#   - HuggingFace access for zwhe99/DeepMath-103K (or pre-cached at
#     ~/LLM/DeepMath-103K/data/*.parquet, which prepare_deepmath.py auto-detects)

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=${DATA_DIR:-$REPO_ROOT/Data}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/shared/Qwen2.5-Math-7B-16k-think}
WORKERS=${WORKERS:-16}
FORCE=${FORCE:-0}

DEEPMATH_RAW=$DATA_DIR/deepmath.parquet
DEEPMATH_HARD=$DATA_DIR/deepmath_dgt6_n10000.parquet
DEEPMATH_SPLIT=$DATA_DIR/deepmath_dgt6_n10000_split.parquet
DEEPMATH_SUMMARIZE=$DATA_DIR/deepmath_dgt6_n10000_summarize.parquet

mkdir -p "$DATA_DIR"
cd "$REPO_ROOT"

echo "==> data dir       : $DATA_DIR"
echo "==> tokenizer path : $TOKENIZER_PATH"
echo "==> workers        : $WORKERS"
echo "==> force rerun    : $FORCE"
echo

skip_or_run() {
    # $1 = output path, $2 = step label, rest = command (passed via "$@")
    local out=$1
    local label=$2
    shift 2
    if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
        echo "[$label] $out exists, skip (set FORCE=1 to rerun)"
    else
        echo "[$label] running: $*"
        "$@"
    fi
}

skip_or_run "$DEEPMATH_RAW" "1/4" \
    python Data/prepare_deepmath.py

skip_or_run "$DEEPMATH_HARD" "2/4" \
    python Data/filter_deepmath.py

skip_or_run "$DEEPMATH_SPLIT" "3/4" \
    python Data/add_token_split_points.py \
        --input          "$DEEPMATH_HARD" \
        --output         "$DEEPMATH_SPLIT" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --workers        "$WORKERS"

skip_or_run "$DEEPMATH_SUMMARIZE" "4/4" \
    python Data/prepare_summarize_prompts.py \
        --input          "$DEEPMATH_SPLIT" \
        --output         "$DEEPMATH_SUMMARIZE" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --workers        "$WORKERS"

echo
echo "==> done. Outputs:"
ls -lh "$DEEPMATH_RAW" "$DEEPMATH_HARD" "$DEEPMATH_SPLIT" "$DEEPMATH_SUMMARIZE" 2>/dev/null || true
echo
echo "==> next: kick off training, e.g."
echo "    bash Myverl/examples/custom/train_grpo_summarize.sh"
