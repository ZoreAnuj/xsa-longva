"""
Compare baseline (SA) vs XSA-tuned eval results on LongVideoBench.

Reads two JSON files written by eval_longvideobench.py and produces:
  1. A printed comparison table (markdown-friendly)
  2. A bar chart PNG (overall + per-duration-group if available)

Usage:
    python analysis/compare_eval.py \
        --baseline /results/lvb_baseline.json \
        --xsa /results/lvb_xsa_tuned.json \
        --output /results/eval_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: str):
    with open(path) as f:
        return json.load(f)


def main(args):
    baseline = load(args.baseline)
    xsa = load(args.xsa)

    base_acc = baseline["overall_accuracy"]
    xsa_acc = xsa["overall_accuracy"]
    delta = xsa_acc - base_acc

    print()
    print("=" * 60)
    print("XSA-LongVA — LongVideoBench val comparison")
    print("=" * 60)
    print()
    print("| Model                       | Accuracy | Δ vs SA |")
    print("|-----------------------------|---------:|--------:|")
    print(f"| LongVA-7B-DPO (SA baseline) | {base_acc:>7.2f}% |    --   |")
    print(f"| LongVA-7B + XSA (ours)      | {xsa_acc:>7.2f}% | {delta:+6.2f}% |")
    print()
    print(f"Correct (baseline): {baseline['correct']}/{baseline['total']}")
    print(f"Correct (xsa):      {xsa['correct']}/{xsa['total']}")
    print()

    # Per-duration-group if both runs have meaningful groups
    base_groups = baseline.get("per_group", {})
    xsa_groups = xsa.get("per_group", {})
    interesting_groups = [
        g for g in sorted(set(base_groups) | set(xsa_groups))
        if g != "unknown"
    ]
    if interesting_groups:
        print("Per duration group:")
        print()
        print("| Group                | SA      | XSA     | Δ      |")
        print("|----------------------|--------:|--------:|-------:|")
        for g in interesting_groups:
            b = base_groups.get(g, {}).get("acc", 0)
            x = xsa_groups.get(g, {}).get("acc", 0)
            d = x - b
            print(f"| {g:<20s} | {b:>6.2f}% | {x:>6.2f}% | {d:+5.2f}% |")
        print()

    # Bar chart
    fig, axes = plt.subplots(
        1,
        1 if not interesting_groups else 2,
        figsize=(12 if interesting_groups else 6, 5),
    )
    if not interesting_groups:
        axes = [axes]

    # Overall comparison
    ax = axes[0]
    bars = ax.bar(
        ["SA baseline", "XSA (ours)"],
        [base_acc, xsa_acc],
        color=["#4c72b0", "#dd8452"],
        edgecolor="black",
        linewidth=1,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"LongVideoBench val — overall\n"
        f"Δ = {delta:+.2f}%",
        fontsize=13,
    )
    ax.set_ylim([0, max(base_acc, xsa_acc) * 1.18])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, [base_acc, xsa_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Per-group comparison
    if interesting_groups:
        ax = axes[1]
        x = np.arange(len(interesting_groups))
        width = 0.35
        base_vals = [base_groups.get(g, {}).get("acc", 0) for g in interesting_groups]
        xsa_vals = [xsa_groups.get(g, {}).get("acc", 0) for g in interesting_groups]
        ax.bar(x - width / 2, base_vals, width, label="SA",
               color="#4c72b0", edgecolor="black")
        ax.bar(x + width / 2, xsa_vals, width, label="XSA",
               color="#dd8452", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(interesting_groups, rotation=20, ha="right")
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("By video duration", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--xsa", required=True)
    parser.add_argument("--output", default="results/eval_comparison.png")
    args = parser.parse_args()
    main(args)
