"""
Measure the attention similarity bias <y_i, v_i> in LongVA's CLIP vision tower.

This reproduces Figure 1 from the XSA paper but on a real video-LM vision
encoder. For each of CLIP's 24 layers, we compute the mean cosine similarity
between the standard attention output y_i and the token's own value v_i.

A high similarity means the layer is wasting capacity replicating self-info
that's already on the residual path — which is exactly what XSA fixes.

Usage:
    python analysis/cosine_similarity.py \
        --model-path /checkpoints/LongVA-7B-DPO \
        --video /data/eval/LongVideoBench/videos/--mUOD9Tok4.mp4 \
        --output /results/cosine_similarity.png
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LongVA" / "longva"))

from longva_helpers import load_longva  # noqa: E402


def sample_frames(video_path: str, num_frames: int):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = np.linspace(0, total - 1, num=num_frames).astype(np.int64)
    frames = vr.get_batch(indices).asnumpy()
    return [frames[i] for i in range(frames.shape[0])]


def measure_layer_similarity(model, frames, image_processor, num_frames, device):
    """For each CLIPAttention layer in the vision tower, hook the forward
    pass and measure mean cosine(y_i, v_i)."""
    from transformers.models.clip.modeling_clip import CLIPAttention

    vt = model.get_vision_tower().vision_tower
    layer_sims: dict[int, float] = {}
    layer_counts: dict[int, int] = {}

    layer_idx_by_module = {}
    for i, layer in enumerate(vt.vision_model.encoder.layers):
        layer_idx_by_module[id(layer.self_attn)] = i

    def make_hook(idx):
        def hook(module, inputs, output):
            x = inputs[0]
            B, N, C = x.shape
            # Recompute q/k/v the same way HF does
            q = module.q_proj(x).view(B, N, module.num_heads, -1).transpose(1, 2)
            k = module.k_proj(x).view(B, N, module.num_heads, -1).transpose(1, 2)
            v = module.v_proj(x).view(B, N, module.num_heads, -1).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, head_dim)
            cos = F.cosine_similarity(
                y.float().reshape(-1, y.shape[-1]),
                v.float().reshape(-1, v.shape[-1]),
                dim=-1,
            )
            layer_sims[idx] = layer_sims.get(idx, 0.0) + float(cos.abs().mean().item())
            layer_counts[idx] = layer_counts.get(idx, 0) + 1
        return hook

    handles = []
    for i, layer in enumerate(vt.vision_model.encoder.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i))
        handles.append(h)

    # Forward pass on the sampled frames
    pixel_values = image_processor.preprocess(
        frames, return_tensors="pt"
    )["pixel_values"].to(device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        _ = model.encode_images(pixel_values)

    for h in handles:
        h.remove()

    return {
        i: layer_sims[i] / layer_counts[i]
        for i in layer_sims
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model_path}")
    tokenizer, model, image_processor, _ = load_longva(
        args.model_path,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()

    print(f"sampling {args.num_frames} frames from {args.video}")
    frames = sample_frames(args.video, args.num_frames)

    print("measuring per-layer cosine similarity <y_i, v_i>")
    sims = measure_layer_similarity(
        model, frames, image_processor, args.num_frames, device,
    )

    layers = sorted(sims.keys())
    values = [sims[l] for l in layers]

    print()
    print("Layer | mean |cos(y, v)|")
    print("------|---------------------")
    for l, v in zip(layers, values):
        bar = "#" * int(v * 50)
        print(f"  {l:>3d} | {v:.4f} {bar}")
    print()
    print(f"mean across layers: {np.mean(values):.4f}")
    print(f"max:                {np.max(values):.4f} (layer {layers[int(np.argmax(values))]})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(layers, values, color="#c44e52", edgecolor="black", linewidth=1)
    ax.set_xlabel("CLIP vision encoder layer", fontsize=12)
    ax.set_ylabel("mean |cos(y_i, v_i)|", fontsize=12)
    ax.set_title(
        "Attention similarity bias in LongVA's CLIP vision tower\n"
        "(higher = more capacity wasted on self-information)",
        fontsize=13,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(layers)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--output", default="results/cosine_similarity.png")
    args = parser.parse_args()
    main(args)
