#!/bin/bash
# Setup XSA-LongVA on a fresh PyTorch container (Ubuntu, Python 3.12, torch 2.8+cu128)
#
# Usage: bash scripts/setup_pod.sh [workspace_dir]
#
# This is the H100 pod setup. It assumes:
#   - PyTorch is already installed at the system level
#   - /workspace exists and has plenty of free space
#   - Python 3.10+ is available
#
# Steps:
#   1. Create a venv with --system-site-packages so we inherit the container's torch
#   2. Install transformers + the rest of our ML stack
#   3. Clone LongVA reference code (without re-installing torch)
#   4. Install the LongVA package without its torch pin
#   5. Run the local pytest suite as a smoke test
#   6. (Optional) Install flash-attn — slow, run separately

set -euo pipefail

WORKSPACE="${1:-/workspace}"
REPO_DIR="$WORKSPACE/xsa-longva"
VENV_DIR="$WORKSPACE/venv"

mkdir -p "$WORKSPACE"

echo "============================================"
echo "[1/7] System info"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 --version
python3 -c "import torch; print(f'torch {torch.__version__} cuda_available={torch.cuda.is_available()}'); print(f'device cap={torch.cuda.get_device_capability(0)}')"

echo ""
echo "============================================"
echo "[2/7] Creating venv at $VENV_DIR (inheriting system torch)"
echo "============================================"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -U pip setuptools wheel

echo ""
echo "============================================"
echo "[3/7] Cloning xsa-longva repo"
echo "============================================"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/ZoreAnuj/xsa-longva.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull origin main || true

echo ""
echo "============================================"
echo "[4/7] Installing core ML deps (transformers + peft + accelerate + ...)"
echo "============================================"
# Compatible versions for torch 2.8 + LongVA's LlavaQwen architecture.
# transformers 4.43.x is the stable line that still ships LlavaQwen-compatible
# CLIPAttention without breaking changes that newer 4.50+ introduced for some
# vision dispatchers.
pip install \
    "transformers==4.43.4" \
    "accelerate==0.33.0" \
    "peft==0.12.0" \
    "deepspeed==0.14.4" \
    "bitsandbytes>=0.43.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "einops>=0.7.0" \
    "shortuuid" \
    "decord>=0.6.0" \
    "av>=12.0.0" \
    "Pillow>=10.0.0" \
    "numpy<2.0" \
    "pandas>=2.0.0" \
    "pyarrow>=14.0.0" \
    "matplotlib>=3.7.0" \
    "scipy>=1.10.0" \
    "tqdm>=4.65.0" \
    "huggingface_hub>=0.24.0" \
    "datasets>=2.18.0" \
    "timm>=0.9.12" \
    "pytest>=8.0.0"

echo ""
echo "============================================"
echo "[5/7] Cloning + installing LongVA reference package"
echo "============================================"
cd "$REPO_DIR"
if [ ! -d "LongVA" ]; then
    git clone https://github.com/EvolvingLMMs-Lab/LongVA.git
fi
cd LongVA

# Strip torch/transformers pins so we can use the container's newer versions.
if [ -f pyproject.toml ]; then
    sed -i.bak \
        -e 's/"torch==2\.1\.2"[,]\?//g' \
        -e 's/"torchvision==0\.16\.2"[,]\?//g' \
        -e 's/"transformers@.*"[,]\?//g' \
        -e 's/"transformers==4\.40[^"]*"[,]\?//g' \
        pyproject.toml
fi
if [ -f setup.py ]; then
    sed -i.bak \
        -e 's/torch==2\.1\.2//g' \
        -e 's/torchvision==0\.16\.2//g' \
        -e 's/transformers@[^"]*//g' \
        setup.py
fi

# Install LongVA without re-resolving deps (we already installed compatible versions)
pip install -e . --no-deps --no-build-isolation || \
    echo "WARN: 'pip install -e .' failed; LongVA still importable via PYTHONPATH"
cd "$REPO_DIR"

echo ""
echo "============================================"
echo "[6/7] Running local test suite (XSA module + patcher)"
echo "============================================"
# These tests don't need a checkpoint or flash-attn - just torch + transformers
python -m pytest tests/ -v --tb=short

echo ""
echo "============================================"
echo "[7/7] Done."
echo "============================================"
echo "Repo: $REPO_DIR"
echo "Venv: $VENV_DIR  (activate with: source $VENV_DIR/bin/activate)"
echo ""
echo "Optional next steps (run separately, slow):"
echo "  pip install flash-attn --no-build-isolation     # 10-20 min compile"
echo "  pip install lmms-eval                            # benchmark harness"
echo "  bash scripts/download_longva.sh ./checkpoints/LongVA-7B-DPO"
echo "  bash scripts/download_eval.sh ./data/eval"
