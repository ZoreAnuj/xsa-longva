"""
Plot training loss + learning rate curves from train_xsa.py's training_log.jsonl.

Usage:
    python analysis/plot_training_curve.py \
        --log /checkpoints/xsa-longva-run1/training_log.jsonl \
        --output /results/training_curve.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def smooth(values, window: int):
    """Simple moving average. Returns the smoothed array (length-(window-1))."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main(args):
    rows = load_log(args.log)
    if not rows:
        raise SystemExit(f"empty log: {args.log}")

    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    lr_other = np.array([r["lr_other"] for r in rows])
    lr_vision = np.array([r["lr_vision"] for r in rows])
    elapsed = np.array([r.get("elapsed_h", 0) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: loss curve
    ax = axes[0]
    ax.plot(steps, loss, alpha=0.25, color="tab:orange", label="raw")
    if len(loss) > args.smooth_window:
        smoothed = smooth(loss, args.smooth_window)
        ax.plot(
            steps[args.smooth_window - 1:],
            smoothed,
            color="tab:orange",
            linewidth=2,
            label=f"smoothed (window={args.smooth_window})",
        )
    ax.set_xlabel("Optimizer step", fontsize=12)
    ax.set_ylabel("Training loss", fontsize=12)
    ax.set_title("XSA-LongVA training loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Annotate start, min, end
    if len(loss):
        ax.annotate(
            f"start: {loss[0]:.3f}",
            xy=(steps[0], loss[0]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=9,
        )
        min_idx = int(np.argmin(loss))
        ax.annotate(
            f"min: {loss[min_idx]:.3f} @ step {steps[min_idx]}",
            xy=(steps[min_idx], loss[min_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
        )

    # Right: learning rate schedule
    ax = axes[1]
    ax.plot(steps, lr_other, color="tab:blue", linewidth=2,
            label="LoRA + projector")
    ax.plot(steps, lr_vision, color="tab:red", linewidth=2,
            label="vision tower (XSA)")
    ax.set_xlabel("Optimizer step", fontsize=12)
    ax.set_ylabel("Learning rate", fontsize=12)
    ax.set_title("Learning rate schedule", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig.suptitle(
        f"XSA-LongVA — {len(rows)} log points, "
        f"final loss {loss[-1]:.3f}, "
        f"elapsed {elapsed[-1]:.2f}h",
        fontsize=12,
    )
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")
    print(f"  steps: {len(rows)}, range {steps[0]}..{steps[-1]}")
    print(f"  loss: start {loss[0]:.4f}, min {loss.min():.4f}, end {loss[-1]:.4f}")
    print(f"  elapsed: {elapsed[-1]:.2f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="training_log.jsonl from train_xsa.py")
    parser.add_argument("--output", default="results/training_curve.png")
    parser.add_argument("--smooth-window", type=int, default=10)
    args = parser.parse_args()
    main(args)
