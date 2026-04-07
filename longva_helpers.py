"""
Helpers for loading LongVA correctly with newer transformers versions.

LongVA's `load_pretrained_model` was written for transformers==4.40.0.dev0
and hardcodes `low_cpu_mem_usage=True`. With transformers >= 4.43, some
weights in the vision tower (specifically the layers that LongVA fine-tuned
on top of stock OpenAI CLIP) end up as meta tensors after loading because
the in-place .copy_() into a meta destination is a no-op.

This module provides:
  - load_longva(): wraps load_pretrained_model and post-fixes meta tensors
  - fix_meta_tensors(): scans a loaded model for meta tensors and reloads
    only those specific keys from the safetensors shards using assign=True
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import torch

# Make LongVA's longva package importable as `longva.*`.
# Layout is: <repo>/LongVA/longva/longva/  (the inner `longva` is the package)
_REPO_ROOT = Path(__file__).resolve().parent
_LONGVA_PARENT = _REPO_ROOT / "LongVA" / "longva"
if _LONGVA_PARENT.is_dir() and str(_LONGVA_PARENT) not in sys.path:
    sys.path.insert(0, str(_LONGVA_PARENT))


def fix_meta_tensors(model: torch.nn.Module, ckpt_path: str) -> int:
    """
    Reload any parameters that are still on the meta device after
    `from_pretrained(..., low_cpu_mem_usage=True)` returned. Only the
    affected keys are read from the safetensors shards (cheap).

    Returns the number of parameters that were materialized.
    """
    from safetensors import safe_open

    meta_keys = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if not meta_keys:
        return 0

    # Pick a destination device by finding any param that's already on a
    # real device. Fall back to cuda:0 if literally everything is meta.
    target_device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cuda:0"),
    )

    # Use the safetensors index to find which shard contains each key.
    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for key in meta_keys:
            shard = weight_map.get(key)
            if shard is not None:
                shards.setdefault(shard, []).append(key)
    else:
        # Single-file checkpoint
        single = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.isfile(single):
            print(f"[fix_meta_tensors] No safetensors found in {ckpt_path}")
            return 0
        shards = {"model.safetensors": meta_keys}

    # Pull the meta-keyed tensors directly from disk into the target device.
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, keys in shards.items():
        shard_path = os.path.join(ckpt_path, shard_name)
        with safe_open(shard_path, framework="pt", device=str(target_device)) as f:
            available = set(f.keys())
            for key in keys:
                if key in available:
                    state_dict[key] = f.get_tensor(key)

    if not state_dict:
        print(f"[fix_meta_tensors] {len(meta_keys)} meta keys but none found in shards")
        return 0

    # `assign=True` REPLACES the meta tensor with the loaded one instead of
    # trying to .copy_() into it (which is a no-op for meta destinations).
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    remaining = sum(1 for p in model.parameters() if p.device.type == "meta")
    fixed = len(meta_keys) - remaining
    print(
        f"[fix_meta_tensors] Materialized {fixed}/{len(meta_keys)} meta params "
        f"({remaining} still meta, {len(missing)} missing, {len(unexpected)} unexpected)"
    )
    return fixed


def load_longva(
    ckpt_path: str,
    device_map: str = "cuda:0",
    attn_implementation: str = "sdpa",
) -> Tuple:
    """
    Load LongVA-7B-DPO and post-fix any meta tensor issues introduced by
    LongVA's hardcoded low_cpu_mem_usage=True with newer transformers.

    Returns: (tokenizer, model, image_processor, context_len)
    """
    from longva.model.builder import load_pretrained_model  # noqa: WPS433

    tokenizer, model, image_processor, ctx_len = load_pretrained_model(
        ckpt_path,
        None,
        "llava_qwen",
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    fix_meta_tensors(model, ckpt_path)
    return tokenizer, model, image_processor, ctx_len
