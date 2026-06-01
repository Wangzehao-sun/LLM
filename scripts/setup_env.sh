#!/usr/bin/env bash
#
# One-shot environment setup for this repo on a fresh GPU server.
#
# Wraps Myverl/scripts/install_vllm_sglang_mcore.sh by adding what it
# does NOT do:
#   1. create / activate a conda env
#   2. download the two prebuilt wheels (flash-attn, flashinfer)
#      that the verl install script `pip install`s but does not fetch
#   3. install verl itself in editable mode (so verl/custom/ takes effect)
#   4. install math-verify (verl[math] extra, used by the reward manager)
#   5. final sanity-import check
#
# Usage:
#   bash scripts/setup_env.sh                 # default: env=verl, no megatron / no sglang
#   ENV_NAME=verl-mine bash scripts/setup_env.sh
#   USE_MEGATRON=1 USE_SGLANG=1 bash scripts/setup_env.sh
#
# Pre-conditions (NOT installed by this script):
#   - conda is on PATH and `nvidia-smi` works (CUDA driver visible)
#   - your CUDA driver supports CUDA 12.x runtime (the prebuilt wheels are cu12)
#   - target Python version is 3.10 (wheels below assume cp310)
#
# If your server has a different CUDA / Python combo, override the wheel
# URLs via FLASH_ATTN_WHEEL_URL / FLASHINFER_WHEEL_URL env vars.

set -euo pipefail

ENV_NAME=${ENV_NAME:-verl}
PY_VERSION=${PY_VERSION:-3.10}
USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}

# Default wheels match install_vllm_sglang_mcore.sh's hard-coded filenames.
# Override these if your server isn't cu124 + torch2.6 + py310.
FLASH_ATTN_WHEEL_URL=${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}
FLASHINFER_WHEEL_URL=${FLASHINFER_WHEEL_URL:-https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MYVERL_DIR=$REPO_ROOT/Myverl
INSTALL_SCRIPT=$MYVERL_DIR/scripts/install_vllm_sglang_mcore.sh

echo "==> repo root        : $REPO_ROOT"
echo "==> conda env        : $ENV_NAME (python $PY_VERSION)"
echo "==> USE_MEGATRON     : $USE_MEGATRON"
echo "==> USE_SGLANG       : $USE_SGLANG"
echo

if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "ERROR: $INSTALL_SCRIPT not found." >&2
    echo "       Are you on the right branch? Did you clone with submodules?" >&2
    exit 1
fi

# ---- 1. conda env --------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[1/5] env '$ENV_NAME' already exists, reusing"
else
    echo "[1/5] creating conda env '$ENV_NAME' (python $PY_VERSION)"
    conda create -n "$ENV_NAME" python="$PY_VERSION" -y
fi
# Make `conda activate` work in non-interactive bash:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Sanity: confirm we're in the right interpreter
ACTIVE_PY=$(python -c 'import sys;print(sys.executable)')
echo "       active python: $ACTIVE_PY"

pip install -U pip setuptools wheel

# ---- 2. download prebuilt wheels into $MYVERL_DIR -----------------------
# install_vllm_sglang_mcore.sh references these by bare filename, so they
# must sit in its CWD when it runs (we'll cd into $MYVERL_DIR for step 3).
cd "$MYVERL_DIR"

FLASH_ATTN_WHEEL=$(basename "$FLASH_ATTN_WHEEL_URL")
FLASHINFER_WHEEL=$(basename "$FLASHINFER_WHEEL_URL")

echo "[2/5] fetching prebuilt wheels into $MYVERL_DIR"
for url in "$FLASH_ATTN_WHEEL_URL" "$FLASHINFER_WHEEL_URL"; do
    fname=$(basename "$url")
    if [ -f "$fname" ]; then
        echo "       [skip] $fname already present"
    else
        echo "       [get ] $fname"
        wget -nv "$url"
    fi
done

# Cross-check filenames the install script expects:
EXPECTED_FA=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
EXPECTED_FI=flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
if [ "$FLASH_ATTN_WHEEL" != "$EXPECTED_FA" ] || [ "$FLASHINFER_WHEEL" != "$EXPECTED_FI" ]; then
    echo "WARNING: install_vllm_sglang_mcore.sh has hard-coded wheel filenames."
    echo "         You overrode the URLs to non-default filenames; the install"
    echo "         script will fail unless you also patch the .sh."
    echo "         expected: $EXPECTED_FA"
    echo "                   $EXPECTED_FI"
fi

# ---- 3. run verl's official install script ------------------------------
echo "[3/5] running install_vllm_sglang_mcore.sh"
echo "      (this installs vllm, torch, transformers, ray, flash-attn, flashinfer, ...)"
USE_MEGATRON="$USE_MEGATRON" USE_SGLANG="$USE_SGLANG" \
    bash scripts/install_vllm_sglang_mcore.sh

# ---- 4. install verl itself + math extra --------------------------------
echo "[4/5] installing verl in editable mode + math-verify"
# --no-deps: deps were just pinned by step 3, don't let setup.py re-resolve
pip install -e ".[math]" --no-deps
# math-verify is in [math] extras, but --no-deps skips extras too on some
# pip versions, so make sure it's actually there:
pip install math-verify

# ---- 5. verify ----------------------------------------------------------
echo "[5/5] verifying imports"
python - <<'PY'
import importlib, sys
mods = ["torch", "vllm", "flash_attn", "flashinfer",
        "transformers", "verl", "math_verify"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "<no __version__>")
        print(f"  {m:14s} {ver}")
    except Exception as e:
        ok = False
        print(f"  {m:14s} FAILED -> {e}")
import torch
print(f"  cuda available  {torch.cuda.is_available()}  ngpu={torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
PY

echo
echo "==> environment ready. Activate later with:"
echo "    conda activate $ENV_NAME"
echo
echo "==> next: generate training data"
echo "    bash scripts/prepare_data.sh"
