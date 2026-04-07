#!/bin/bash
# Wait for the training process to finish, then automatically:
#   1. Run XSA-tuned eval on LongVideoBench val
#   2. Generate the loss curve plot
#   3. Generate the SA vs XSA comparison plot
#   4. Generate the attention viz on a sample video
#
# Usage:
#   bash scripts/auto_pipeline.sh <train_pid> <output_dir>
#
# Example:
#   bash scripts/auto_pipeline.sh 78616 /checkpoints/xsa-longva-run1

set -e

TRAIN_PID="${1:?need training PID as first arg}"
OUTPUT_DIR="${2:-/checkpoints/xsa-longva-run1}"
RESULTS_DIR="/results"
DATA_PATH="/data/eval/LongVideoBench"
MODEL_PATH="/checkpoints/LongVA-7B-DPO"

mkdir -p "$RESULTS_DIR"

echo "[auto] waiting for train PID $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 30
done
echo "[auto] training process gone, proceeding"

# Sanity check that final checkpoint was saved
FINAL_CKPT="$OUTPUT_DIR/final"
if [ ! -d "$FINAL_CKPT" ]; then
    # Fall back to the most recent step_* checkpoint
    FINAL_CKPT=$(ls -d "$OUTPUT_DIR"/step_* 2>/dev/null | sort | tail -1)
    echo "[auto] no /final dir, using $FINAL_CKPT"
fi
if [ ! -d "$FINAL_CKPT" ]; then
    echo "[auto] ERROR: no checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo "[auto] checkpoint: $FINAL_CKPT"
ls "$FINAL_CKPT"

echo ""
echo "[auto 1/4] Running XSA-tuned eval on LongVideoBench val"
echo "============================================================"
cd /workspace/xsa-longva
python3 -u eval_longvideobench.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output "$RESULTS_DIR/lvb_xsa_tuned.json" \
    --max-frames 32 \
    --mode xsa-tuned \
    --xsa-ckpt "$FINAL_CKPT" \
    > "$RESULTS_DIR/xsa_eval.log" 2>&1
echo "[auto 1/4] eval done -> $RESULTS_DIR/lvb_xsa_tuned.json"

echo ""
echo "[auto 2/4] Plotting training curve"
echo "============================================================"
python3 analysis/plot_training_curve.py \
    --log "$OUTPUT_DIR/training_log.jsonl" \
    --output "$RESULTS_DIR/training_curve.png"

echo ""
echo "[auto 3/4] Comparing SA vs XSA on LongVideoBench"
echo "============================================================"
python3 analysis/compare_eval.py \
    --baseline "$RESULTS_DIR/lvb_baseline.json" \
    --xsa "$RESULTS_DIR/lvb_xsa_tuned.json" \
    --output "$RESULTS_DIR/eval_comparison.png" \
    > "$RESULTS_DIR/eval_comparison.txt"
cat "$RESULTS_DIR/eval_comparison.txt"

echo ""
echo "[auto 4/4] Generating attention viz on a sample video"
echo "============================================================"
SAMPLE_VIDEO=$(ls "$DATA_PATH"/videos/*.mp4 2>/dev/null | head -1)
if [ -n "$SAMPLE_VIDEO" ]; then
    python3 analysis/attention_viz.py \
        --model-path "$MODEL_PATH" \
        --video "$SAMPLE_VIDEO" \
        --num-frames 8 \
        --layer 22 \
        --output "$RESULTS_DIR/attention_viz.png" \
        > "$RESULTS_DIR/attention_viz.log" 2>&1 || \
        echo "[auto] attention viz failed, see $RESULTS_DIR/attention_viz.log"
fi

echo ""
echo "============================================================"
echo "[auto] DONE. Outputs in $RESULTS_DIR:"
ls -la "$RESULTS_DIR"
