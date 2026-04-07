#!/bin/bash
# Download LLaVA-Video-178K training data (a length-bucket subset).
#
# Usage: bash scripts/download_data.sh [data_dir] [bucket]
#
# Default data_dir: /workspace/data/train
# Default bucket: 30_60_s   (other options: 0_30_s, 1_2_m, 2_3_m)
#
# Sizes (approximate):
#   0_30_s_*  ~120 GB
#   30_60_s_* ~250 GB
#   1_2_m_*   ~400 GB
#   2_3_m_*   ~150 GB

set -e

DATA_DIR="${1:-/workspace/data/train}"
BUCKET="${2:-30_60_s}"

mkdir -p "$DATA_DIR"

echo "Downloading LLaVA-Video-178K bucket: $BUCKET"
echo "Target: $DATA_DIR/LLaVA-Video-178K"

huggingface-cli download lmms-lab/LLaVA-Video-178K \
    --repo-type dataset \
    --include "${BUCKET}_*/*" \
    --local-dir "$DATA_DIR/LLaVA-Video-178K" \
    --local-dir-use-symlinks False

echo ""
echo "Done. Total size:"
du -sh "$DATA_DIR/LLaVA-Video-178K"
