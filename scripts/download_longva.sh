#!/bin/bash
# Download LongVA-7B-DPO from HuggingFace.
#
# Usage: bash scripts/download_longva.sh [local_dir]
#
# Default local_dir: ./checkpoints/LongVA-7B-DPO
# Size: ~16 GB (Qwen2-7B + CLIP ViT-L + projector, all in bf16 safetensors)

set -e

MODEL_ID="lmms-lab/LongVA-7B-DPO"
LOCAL_DIR="${1:-./checkpoints/LongVA-7B-DPO}"

mkdir -p "$LOCAL_DIR"

echo "Downloading $MODEL_ID -> $LOCAL_DIR"
echo "(Skipping non-essential files like .gguf and .pickle)"

huggingface-cli download "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False \
    --exclude "*.gguf" "*.pickle" "*.pkl" "*.bin"

echo ""
echo "Done. Checkpoint at: $LOCAL_DIR"
ls -lh "$LOCAL_DIR" | head -20
echo ""
echo "Total size:"
du -sh "$LOCAL_DIR"
