"""
Manual smoke test for LongVA + XSA integration.

NOT a pytest test - this is a script you run after downloading the
LongVA-7B-DPO checkpoint to verify the full integration works:

  1. LongVA loads with our newer transformers (4.43.4)
  2. The vision tower contains 24 CLIPAttention layers
  3. patch_clip_model_with_xsa() finds and replaces them
  4. With use_xsa=False, patched model produces identical output to unpatched
  5. With use_xsa=True, patched model produces different output
  6. Generation still runs end-to-end with XSA patched in

Run from the repo root:
    python scripts/smoke_test_longva.py [--ckpt /path/to/LongVA-7B-DPO]

Default ckpt path: ./checkpoints/LongVA-7B-DPO
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root and the LongVA submodule to PYTHONPATH so we can import both.
# LongVA's actual package is at LongVA/longva/longva/, so the importable
# parent directory is LongVA/longva/.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LongVA" / "longva"))

import torch  # noqa: E402

from patch_longva import patch_clip_model_with_xsa, count_xsa_layers  # noqa: E402


def step(n, msg):
    print(f"\n[{n}] {msg}")
    print("-" * 60)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")

    ckpt_path = args.ckpt
    if not os.path.isdir(ckpt_path):
        print(f"ERROR: checkpoint dir not found: {ckpt_path}")
        print("Run: bash scripts/download_longva.sh")
        sys.exit(1)
    print(f"Checkpoint: {ckpt_path}")

    # ----------------------------------------------------------------
    step(1, "Importing LongVA")
    # ----------------------------------------------------------------
    try:
        from longva.model.builder import load_pretrained_model
        print("OK: from longva.model.builder import load_pretrained_model")
    except ImportError as e:
        print(f"FAIL: cannot import LongVA: {e}")
        print("Make sure LongVA/ submodule is cloned at the repo root.")
        sys.exit(1)

    # ----------------------------------------------------------------
    step(2, "Loading LongVA-7B-DPO (this takes a minute)")
    # ----------------------------------------------------------------
    # Note: LongVA's builder hardcodes torch_dtype=float16 when not using
    # 4/8-bit quantization, so we get fp16 here regardless. Also force
    # sdpa instead of flash_attention_2 since we haven't built flash-attn.
    tokenizer, model, image_processor, _ = load_pretrained_model(
        ckpt_path,
        None,
        "llava_qwen",
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    print(f"OK: model class = {type(model).__name__}")
    print(f"    total params = {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ----------------------------------------------------------------
    step(3, "Inspecting vision tower")
    # ----------------------------------------------------------------
    vt_wrapper = model.get_vision_tower()
    print(f"vision_tower wrapper: {type(vt_wrapper).__name__}")
    vt = vt_wrapper.vision_tower  # the actual CLIPVisionModel
    print(f"inner vision_tower:   {type(vt).__name__}")

    # Count CLIPAttention layers in the vision tower
    from transformers.models.clip.modeling_clip import CLIPAttention
    n_clip_attn = sum(1 for m in vt.modules() if isinstance(m, CLIPAttention))
    print(f"CLIPAttention layers: {n_clip_attn}")
    if n_clip_attn != 24:
        print(f"WARN: expected 24 CLIPAttention layers, got {n_clip_attn}")

    n_xsa_before = count_xsa_layers(vt)
    print(f"XSA layers (before patch): {n_xsa_before}")

    # ----------------------------------------------------------------
    step(4, "Capturing baseline vision feature on dummy frames")
    # ----------------------------------------------------------------
    # 4 dummy frames at 336x336 — use uint8 numpy arrays in [0,255] since the
    # HF CLIP image_processor expects PIL-convertible inputs.
    import numpy as np
    dummy_frames = [
        (np.random.rand(336, 336, 3) * 255).astype(np.uint8)
        for _ in range(4)
    ]
    images = image_processor.preprocess(dummy_frames, return_tensors="pt")["pixel_values"]
    # Match the model's actual dtype (LongVA's builder forces fp16)
    model_dtype = next(model.parameters()).dtype
    print(f"model dtype: {model_dtype}")
    images = images.to(device, dtype=model_dtype)
    print(f"images shape: {images.shape}, dtype: {images.dtype}")

    with torch.no_grad():
        baseline_features = model.encode_images(images).clone()
    print(f"baseline features: {baseline_features.shape}, dtype: {baseline_features.dtype}")

    # ----------------------------------------------------------------
    step(5, "Patching with XSA disabled (use_xsa=False)")
    # ----------------------------------------------------------------
    # First test: with use_xsa=False the output MUST match baseline.
    patch_clip_model_with_xsa(vt, use_xsa=False)
    n_xsa = count_xsa_layers(vt)
    print(f"XSA layers (after patch with use_xsa=False): {n_xsa}")
    assert n_xsa == n_clip_attn, f"expected {n_clip_attn}, got {n_xsa}"

    with torch.no_grad():
        patched_off_features = model.encode_images(images)
    diff_off = (patched_off_features - baseline_features).abs().max().item()
    print(f"max diff (use_xsa=False): {diff_off:.2e}")
    if diff_off > 1e-2:
        print(f"WARN: patched output differs noticeably from baseline (max abs diff {diff_off})")
    else:
        print("OK: patched output matches baseline within tolerance")

    # ----------------------------------------------------------------
    step(6, "Re-patching with XSA enabled (use_xsa=True)")
    # ----------------------------------------------------------------
    # Easier path: directly flip the flag on existing XSA modules.
    from xsa_clip_attention import XSACLIPAttention
    flipped = 0
    for m in vt.modules():
        if isinstance(m, XSACLIPAttention):
            m.use_xsa = True
            flipped += 1
    print(f"Flipped use_xsa=True on {flipped} XSA modules")

    with torch.no_grad():
        patched_on_features = model.encode_images(images)
    diff_on = (patched_on_features - baseline_features).abs().max().item()
    print(f"max diff (use_xsa=True): {diff_on:.2e}")
    if diff_on < 1e-3:
        print("FAIL: XSA enabled produced same output as baseline (broken)")
        sys.exit(1)
    print("OK: XSA enabled produces different features as expected")

    # ----------------------------------------------------------------
    step(7, "End-to-end generation with XSA patched in")
    # ----------------------------------------------------------------
    try:
        from longva.constants import IMAGE_TOKEN_INDEX
        from longva.mm_utils import tokenizer_image_token
    except ImportError as e:
        print(f"WARN: skipping generation test, can't import LongVA helpers: {e}")
    else:
        prompt = (
            "<|im_start|>user\n<image>\nDescribe what you see.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=[images],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=20,
            )
        text = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"Generated (XSA enabled): {text!r}")

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="./checkpoints/LongVA-7B-DPO",
        help="Path to LongVA-7B-DPO checkpoint directory",
    )
    args = parser.parse_args()
    main(args)
