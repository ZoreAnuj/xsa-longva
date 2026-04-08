"""
Merge PEFT LoRA adapters into the base vision tower weights for an XSA
training checkpoint, and write a clean state_dict that eval_longvideobench.py
can load directly.

Background: train_xsa.py used inject_adapter_in_model on model.model, which
walked into model.model.vision_tower (LongVA nests the CLIP vision tower
under the Qwen2Model wrapper). LoRA matched the CLIPAttention's q/k/v_proj
names and added adapter layers there too. The saved vision_tower_xsa.safetensors
therefore contains LoRA-wrapped key paths (e.g.
`...self_attn.q_proj.base_layer.weight` and
`...self_attn.q_proj.lora_A.default.weight`) instead of the plain
`...self_attn.q_proj.weight` paths the eval expects.

This script:
  1. Re-loads LongVA the same way train_xsa.py did (load_longva, bf16, XSA patch)
  2. Re-injects LoRA on model.model with the same config
  3. Loads the saved state_dict from `<ckpt>/vision_tower_xsa.safetensors`
     INTO the vision tower (keys now match because LoRA layers exist)
  4. Calls PEFT's merge functionality on every LoRA layer in the vision tower,
     folding the adapters into the base weights
  5. Walks the vision tower one more time and saves a CLEAN state_dict
     containing only the merged base parameters (q_proj.weight not
     q_proj.base_layer.weight)

Output: <ckpt>/vision_tower_xsa_merged.safetensors

Usage:
    python scripts/merge_lora_into_vision_tower.py \
        --model-path /checkpoints/LongVA-7B-DPO \
        --xsa-ckpt /checkpoints/xsa-longva-run1/final \
        --lora-r 16 \
        --lora-alpha 32 \
        --lora-dropout 0.05
"""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "LongVA" / "longva"))

from longva_helpers import load_longva  # noqa: E402
from patch_longva import patch_clip_model_with_xsa  # noqa: E402


def main(args):
    print(f"[merge] loading {args.model_path}")
    tokenizer, model, image_processor, _ = load_longva(
        args.model_path,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )

    # Match training: convert to bf16, patch vision tower with XSA
    print("[merge] converting fp16 -> bf16")
    model = model.to(dtype=torch.bfloat16)

    vt = model.get_vision_tower().vision_tower
    patch_clip_model_with_xsa(vt, use_xsa=True)

    # Re-inject the same LoRA configuration the training used
    print("[merge] injecting LoRA adapters with same config as training")
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    inject_adapter_in_model(lora_config, model.model)

    # Load the trained vision tower state_dict
    from safetensors.torch import load_file, save_file
    st_path = Path(args.xsa_ckpt) / "vision_tower_xsa.safetensors"
    print(f"[merge] loading {st_path}")
    sd = load_file(str(st_path), device="cuda:0")
    sd = {k: v.to(dtype=torch.bfloat16) for k, v in sd.items()}
    missing, unexpected = vt.load_state_dict(sd, strict=False, assign=True)
    print(f"[merge] loaded into vt (missing={len(missing)}, unexpected={len(unexpected)})")
    if len(unexpected) > 0:
        print(f"[merge] WARN: still {len(unexpected)} unexpected keys")
        print(f"[merge]       first 3: {unexpected[:3]}")

    # Walk the vision tower and merge any LoRA layers found
    print("[merge] merging LoRA into base weights")
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        from peft.tuners.lora.layer import LoraLayer  # newer peft

    merged_count = 0
    for name, module in vt.named_modules():
        if isinstance(module, LoraLayer):
            try:
                module.merge()
                merged_count += 1
            except Exception as e:
                print(f"  failed to merge {name}: {e}")
    print(f"[merge] merged {merged_count} LoRA layers")

    # Now extract the underlying base weights with their original key names.
    # After merge(), each LoraLayer holds the merged weight in
    # module.base_layer.weight. We need to walk the tree and emit a state_dict
    # using the LOGICAL key names (replacing 'q_proj' instead of
    # 'q_proj.base_layer'), so that a non-PEFT model can load them.
    print("[merge] extracting clean state_dict")
    clean_sd: dict[str, torch.Tensor] = {}
    for name, param in vt.named_parameters():
        # Strip the '.base_layer' marker from any LoRA-wrapped path,
        # and skip any pure-LoRA params that are still in the model
        if ".lora_A" in name or ".lora_B" in name or ".lora_embedding" in name:
            continue
        clean_name = name.replace(".base_layer.", ".")
        clean_sd[clean_name] = param.detach().cpu().contiguous()

    print(f"[merge] clean state_dict has {len(clean_sd)} keys")

    out_path = Path(args.xsa_ckpt) / "vision_tower_xsa_merged.safetensors"
    save_file(clean_sd, str(out_path))
    print(f"[merge] saved {out_path}")
    print()
    print("Eval with:")
    print(f"  python eval_longvideobench.py \\")
    print(f"    --model-path {args.model_path} \\")
    print(f"    --xsa-ckpt {args.xsa_ckpt} \\")
    print(f"    --mode xsa-tuned \\")
    print(f"    ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--xsa-ckpt", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
