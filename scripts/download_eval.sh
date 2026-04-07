#!/bin/bash
# Download LongVideoBench evaluation dataset.
#
# Usage: bash scripts/download_eval.sh [eval_dir]
#
# Default eval_dir: ./data/eval
# Size: ~140GB (3,763 videos + subtitles)
#
# Requires the dataset access to be approved on HF first:
#   https://huggingface.co/datasets/longvideobench/LongVideoBench

set -e

EVAL_DIR="${1:-./data/eval}"
mkdir -p "$EVAL_DIR/LongVideoBench"

echo "Downloading LongVideoBench from HuggingFace..."
huggingface-cli download longvideobench/LongVideoBench \
    --repo-type dataset \
    --local-dir "$EVAL_DIR/LongVideoBench" \
    --local-dir-use-symlinks False

cd "$EVAL_DIR/LongVideoBench"

# Reassemble video archive (split into 26 parts)
if ls videos.tar.part.* >/dev/null 2>&1; then
    echo "Reassembling videos.tar from parts..."
    cat videos.tar.part.* > videos.tar
    rm videos.tar.part.*
fi

# Extract videos and subtitles
if [ -f videos.tar ]; then
    echo "Extracting videos.tar (~140GB)..."
    tar -xf videos.tar
    rm videos.tar
fi
if [ -f subtitles.tar ]; then
    echo "Extracting subtitles.tar..."
    tar -xf subtitles.tar
    rm subtitles.tar
fi

echo ""
echo "Done. Dataset at: $EVAL_DIR/LongVideoBench"
ls -lh "$EVAL_DIR/LongVideoBench" | head -10
echo ""
echo "Total size:"
du -sh "$EVAL_DIR/LongVideoBench"
