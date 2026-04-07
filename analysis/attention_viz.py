"""
Visualise CLIP vision attention maps for SA vs XSA on the same video frame.

Picks one query token (default: center patch of the first frame) and renders
the attention weights it places on all other frames as heat maps. Two grids
are produced side by side:
  - SA: vanilla CLIPAttention from the unmodified LongVA-7B-DPO checkpoint
  - XSA: XSACLIPAttention swapped in (use_xsa=True)

The visual difference is the viral content for the X thread:
  - SA tends to put a lot of mass on the same patch in temporally adjacent frames
    (the self-similarity bias)
  - XSA strips that out and the heat map shifts toward parts of the scene that
    are actually moving / changing

Usage:
    python analysis/attention_viz.py \
        --model-path /checkpoints/LongVA-7B-DPO \
        --video /data/eval/LongVideoBench/videos/--mUOD9Tok4.mp4 \
        --layer 22 \
        --output /results/attention_viz.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LongVA" / "longva"))

from longva_helpers import load_longva  # noqa: E402
from patch_longva import patch_clip_model_with_xsa  # noqa: E402
from xsa_clip_attention import XSACLIPAttention  # noqa: E402


def sample_frames(video_path: str, num_frames: int):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = np.linspace(0, total - 1, num=num_frames).astype(np.int64)
    frames = vr.get_batch(indices).asnumpy()
    return [frames[i] for i in range(frames.shape[0])]


def grab_attention_maps(model, image_processor, frames, layer_idx, device):
    """Run vision encoder on `frames` and return per-frame attention maps
    from layer `layer_idx`. Returns shape (num_frames, num_heads, N, N).
    """
    pixel_values = image_processor.preprocess(
        frames, return_tensors="pt"
    )["pixel_values"].to(device, dtype=next(model.parameters()).dtype)

    captured = {}

    def hook(module, inputs, output):
        x = inputs[0]
        B, N, C = x.shape
        q = module.q_proj(x).view(B, N, module.num_heads, -1).transpose(1, 2)
        k = module.k_proj(x).view(B, N, module.num_heads, -1).transpose(1, 2)
        # Manual softmax(QK^T / sqrt(d)) — what SDPA does internally
        scale = q.shape[-1] ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        captured["attn"] = attn.detach().cpu().float()

    vt = model.get_vision_tower().vision_tower
    target_layer = vt.vision_model.encoder.layers[layer_idx].self_attn
    h = target_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model.encode_images(pixel_values)
    finally:
        h.remove()

    return captured["attn"]  # (num_frames, num_heads, N, N)


def query_to_patch_grid(attn_for_query: np.ndarray, grid_size: int):
    """Take a (N,) attention vector over CLIP tokens for one query.
    Strip CLS, reshape patch portion to (grid, grid)."""
    patch_attn = attn_for_query[1:]  # drop CLS at index 0
    if patch_attn.shape[0] != grid_size * grid_size:
        # In case of slight mismatch, pad/truncate
        n = grid_size * grid_size
        if patch_attn.shape[0] < n:
            patch_attn = np.pad(patch_attn, (0, n - patch_attn.shape[0]))
        else:
            patch_attn = patch_attn[:n]
    return patch_attn.reshape(grid_size, grid_size)


def render(frames, attn_maps, query_frame, query_patch, grid_size, title, ax_row):
    """Overlay attention heat maps on frames in one row of subplots."""
    # attn_maps: (num_frames, num_heads, N, N)
    num_frames = len(frames)
    # Average heads
    avg = attn_maps.mean(dim=1)  # (num_frames, N, N)
    # The query is in `query_frame` at token index `query_patch + 1` (1=skip CLS)
    query_token = query_patch + 1
    for i, ax in enumerate(ax_row):
        if i >= num_frames:
            ax.axis("off")
            continue
        ax.imshow(frames[i])
        # Attention vector from this frame's encoder pass for the query token
        attn_vec = avg[i, query_token].numpy()
        heat = query_to_patch_grid(attn_vec, grid_size)
        # Resize heat to image dim
        H, W = frames[i].shape[:2]
        heat_resized = np.kron(heat, np.ones((H // grid_size, W // grid_size)))
        # Pad if not divisible
        ph, pw = H - heat_resized.shape[0], W - heat_resized.shape[1]
        if ph > 0 or pw > 0:
            heat_resized = np.pad(heat_resized, ((0, max(ph, 0)), (0, max(pw, 0))))
        heat_resized = heat_resized / (heat_resized.max() + 1e-9)
        ax.imshow(heat_resized, cmap="hot", alpha=0.55)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(title, fontsize=11, rotation=90, labelpad=10)


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

    grid_size = 24  # CLIP ViT-L/14-336 produces 24x24 patches

    print(f"capturing SA attention from layer {args.layer}")
    sa_maps = grab_attention_maps(
        model, image_processor, frames, args.layer, device,
    )

    print("patching vision tower with XSA")
    vt = model.get_vision_tower().vision_tower
    patch_clip_model_with_xsa(vt, use_xsa=True)

    print(f"capturing XSA attention from layer {args.layer}")
    xsa_maps = grab_attention_maps(
        model, image_processor, frames, args.layer, device,
    )

    # Plot
    print("rendering side-by-side")
    nf = len(frames)
    fig, axes = plt.subplots(2, nf, figsize=(2.0 * nf, 4.5))
    if nf == 1:
        axes = axes.reshape(2, 1)

    qp = (grid_size // 2) * grid_size + (grid_size // 2)  # center patch
    render(frames, sa_maps, args.query_frame, qp, grid_size,
           "SA (vanilla)", axes[0])
    render(frames, xsa_maps, args.query_frame, qp, grid_size,
           "XSA (ours)", axes[1])

    fig.suptitle(
        f"CLIP layer {args.layer} attention from center patch of frame "
        f"{args.query_frame}\n"
        "(Hot = high attention)",
        fontsize=12,
    )
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--layer", type=int, default=22,
                        help="Which CLIP layer to visualise (later layers have more bias)")
    parser.add_argument("--query-frame", type=int, default=0,
                        help="Which frame the query patch comes from")
    parser.add_argument("--output", default="results/attention_viz.png")
    args = parser.parse_args()
    main(args)
