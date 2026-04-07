#!/bin/bash
# Setup environment for XSA-LongVA training
# Usage: bash scripts/setup_env.sh

set -e

ENV_NAME="${ENV_NAME:-xsa-longva}"

echo "[1/6] Creating conda env: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y
# shellcheck disable=SC1091
source activate "$ENV_NAME"

echo "[2/6] Installing PyTorch 2.1.2 + CUDA 12.1"
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[3/6] Cloning LongVA reference code"
if [ ! -d "LongVA" ]; then
    git clone https://github.com/EvolvingLMMs-Lab/LongVA.git
fi

echo "[4/6] Installing LongVA package (editable)"
cd LongVA
pip install -e .
cd ..

echo "[5/6] Installing xsa-longva requirements"
pip install -r requirements.txt

echo "[6/6] Building flash-attn 2.5.0 (this can take 10-20 minutes)"
pip install flash-attn==2.5.0 --no-build-isolation

# lmms-eval for benchmark harness
pip install lmms-eval

echo ""
echo "==========================================="
echo "Environment '$ENV_NAME' ready."
echo "Activate with: conda activate $ENV_NAME"
echo "==========================================="
