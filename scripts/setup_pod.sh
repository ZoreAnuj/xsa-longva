#!/bin/bash
# Setup XSA-LongVA on a fresh PyTorch container.
#
# Usage: bash scripts/setup_pod.sh
#
# Strategy: install everything into the system Python directly
# (the container is ephemeral — no point isolating in a venv).
# This is much faster than building a venv with --system-site-packages
# because we avoid re-unpacking large wheels like transformers.

set -euo pipefail

echo "============================================"
echo "[1/5] System info"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 --version
python3 -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')"

echo ""
echo "============================================"
echo "[2/5] Installing core ML deps into system Python"
echo "============================================"
# Drop the numpy pin — torch 2.8 already brought numpy 2.x and we don't
# need a specific version. --break-system-packages overrides PEP 668 (the
# container is ephemeral, system Python is fine to modify).
pip install --no-cache-dir --break-system-packages \
    "transformers==4.43.4" \
    "tokenizers>=0.19,<0.20" \
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
echo "[3/5] Cloning LongVA reference code"
echo "============================================"
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
echo "Repo: $REPO_DIR"

if [ ! -d "LongVA" ]; then
    git clone --depth 1 https://github.com/EvolvingLMMs-Lab/LongVA.git
fi

# We DON'T pip install the LongVA package (avoids dependency conflicts).
# Instead we add LongVA's longva/ subdirectory to PYTHONPATH at runtime
# in our scripts. The model code lives in LongVA/longva/.

echo ""
echo "============================================"
echo "[4/5] Sanity check imports"
echo "============================================"
python3 -c "
import torch
import transformers
import accelerate
import peft
import deepspeed
import decord
print(f'torch       {torch.__version__}')
print(f'transformers {transformers.__version__}')
print(f'accelerate  {accelerate.__version__}')
print(f'peft        {peft.__version__}')
print(f'deepspeed   {deepspeed.__version__}')
print(f'decord      {decord.__version__}')
print(f'cuda avail  {torch.cuda.is_available()}')
print(f'gpu         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
"

echo ""
echo "============================================"
echo "[5/5] Running local test suite"
echo "============================================"
cd "$REPO_DIR"
python3 -m pytest tests/ -v --tb=short

echo ""
echo "============================================"
echo "Setup complete."
echo "============================================"
echo "Repo: $REPO_DIR"
echo "LongVA: $REPO_DIR/LongVA"
echo ""
echo "Optional next steps:"
echo "  pip install flash-attn --no-build-isolation       # 10-20 min compile"
echo "  pip install lmms-eval                              # benchmark harness"
echo "  bash scripts/download_longva.sh ./checkpoints/LongVA-7B-DPO"
