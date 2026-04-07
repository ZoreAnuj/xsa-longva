# XSA-LongVA: Exclusive Self Attention for Long Video Understanding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Exclusive Self Attention (XSA) to LongVA's CLIP ViT-L/14 vision encoder, fine-tune with a video instruction dataset on a single H100 (or RTX 5090), and beat LongVA-7B-DPO baseline on LongVideoBench validation by +2 or more points.

**Architecture:** Start from `lmms-lab/LongVA-7B-DPO`. Replace `CLIPAttention` in all 24 layers of the vision tower with `XSACLIPAttention` that adds an orthogonal projection removing the self-value component. Fine-tune the modified vision tower fully (LR 2e-6) plus LoRA adapters on the Qwen2-7B LLM (rank 16) on a subset of LLaVA-Video-178K (30-60s bucket, ~100K samples). Evaluate on LongVideoBench val split (1,337 questions with GT).

**Tech Stack:** PyTorch 2.1+, transformers 4.40.0.dev0, flash-attn 2.5.0, PEFT (LoRA), DeepSpeed ZeRO-3, lmms-eval for benchmarking, huggingface_hub for model/data, decord for video loading

**Base model:** `lmms-lab/LongVA-7B-DPO` (Qwen2-7B-Instruct-224K LLM + CLIP ViT-L/14-336 vision + mlp2x_gelu projector)

**Target benchmark:** LongVideoBench val accuracy — LongVA-7B-DPO baseline ~56-58%, goal ≥60%

---

## File Structure

```
xsa-longva/
├── xsa_clip_attention.py        # XSACLIPAttention - drop-in replacement for HF CLIPAttention
├── patch_longva.py              # Monkey-patch loaded LongVA vision tower to use XSA
├── train_xsa.py                 # Fine-tuning script (LoRA LLM + full vision tower FT)
├── eval_longvideobench.py       # Run LongVideoBench val eval
├── eval_videomme.py             # Run Video-MME (long split) eval
├── eval_mvbench.py              # Run MVBench eval (sanity check)
├── scripts/
│   ├── setup_env.sh             # Clone LongVA, install deps
│   ├── download_longva.sh       # Download LongVA-7B-DPO weights
│   ├── download_data.sh         # Download LLaVA-Video-178K subset
│   ├── download_eval.sh         # Download LongVideoBench, Video-MME, MVBench
│   ├── prepare_subset.py        # Filter LLaVA-Video-178K to 30-60s bucket
│   ├── train_overnight.sh       # Full training run
│   ├── eval_baseline.sh         # Eval unmodified LongVA-7B-DPO
│   └── eval_xsa.sh              # Eval XSA-fine-tuned model
├── analysis/
│   ├── attention_viz.py         # Side-by-side attention heatmaps (SA vs XSA)
│   ├── cosine_similarity.py     # Measure <y, v> bias per layer in LongVA CLIP
│   ├── plot_eval_comparison.py  # Bar chart SA vs XSA across benchmarks
│   └── video_qa_examples.py     # Generate side-by-side video QA comparisons
├── tests/
│   ├── test_xsa_clip_attention.py   # Unit tests for XSA CLIP attention
│   ├── test_patch_longva.py         # Test that patching preserves output shapes
│   └── test_forward_pass.py         # Full model forward on dummy video
├── configs/
│   ├── deepspeed_zero3.json     # DeepSpeed config for training
│   └── lora_config.yaml         # LoRA configuration
├── requirements.txt
├── README.md
├── LICENSE
└── docs/plans/2026-04-07-xsa-longva.md
```

---

## Task 1: Initialize Repository and Environment

**Files:**
- Create: `requirements.txt`, `LICENSE`, `.gitignore`, `README.md` (short version)
- Create: `scripts/setup_env.sh`

- [ ] **Step 1: Create requirements.txt**

```txt
torch>=2.1.2
torchvision>=0.16.0
transformers==4.40.0.dev0
accelerate>=0.29.0
deepspeed>=0.14.0
peft>=0.10.0
bitsandbytes>=0.43.0
flash-attn==2.5.0
decord>=0.6.0
av>=12.0.0
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
huggingface_hub>=0.22.0
datasets>=2.18.0
sentencepiece>=0.1.99
protobuf>=3.20.0
einops>=0.7.0
timm>=0.9.12
openai>=1.0.0
shortuuid
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
*.pt
*.pth
*.safetensors
*.bin
checkpoints/
results/
data/
eval_output/
.venv/
wandb/
plots/
.DS_Store
LongVA/
!LongVA.patch
```

- [ ] **Step 3: Create LICENSE (MIT)**

Same MIT license template as xsa-dit, copyright 2026 Anuj Zore.

- [ ] **Step 4: Create scripts/setup_env.sh**

```bash
#!/bin/bash
# Setup environment for XSA-LongVA training
# Usage: bash scripts/setup_env.sh

set -e

# Create conda env
conda create -n xsa-longva python=3.10 -y
source activate xsa-longva

# Install torch with CUDA 12.1
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone LongVA repo for reference code
if [ ! -d "LongVA" ]; then
    git clone https://github.com/EvolvingLMMs-Lab/LongVA.git
fi

# Install LongVA package in editable mode
cd LongVA
pip install -e .
cd ..

# Install our requirements
pip install -r requirements.txt

# Install flash-attn (may take 10-20 minutes to build)
pip install flash-attn==2.5.0 --no-build-isolation

# Install lmms-eval for benchmarking
pip install lmms-eval

echo "Environment ready. Activate with: conda activate xsa-longva"
```

- [ ] **Step 5: Initialize git and commit**

```bash
cd D:/research/xsa-longva
git init
git add requirements.txt LICENSE .gitignore scripts/setup_env.sh README.md
git commit -m "feat: initialize xsa-longva repository"
```

---

## Task 2: Implement XSA CLIP Attention

**Files:**
- Create: `xsa_clip_attention.py`
- Create: `tests/test_xsa_clip_attention.py`

HuggingFace's `CLIPAttention` uses separate Q, K, V projections (not fused QKV like timm). We create a drop-in replacement that applies XSA's orthogonal projection after attention.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_xsa_clip_attention.py
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xsa_clip_attention import XSACLIPAttention
from transformers.models.clip.configuration_clip import CLIPVisionConfig


def _make_config():
    """Standard CLIP ViT-L/14 vision config used by LongVA."""
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=336,
        patch_size=14,
    )


def test_xsa_clip_attention_output_shape():
    """XSA CLIP attention should produce same output shape as standard."""
    config = _make_config()
    attn = XSACLIPAttention(config, use_xsa=True)
    B, N, C = 2, 577, 1024  # 577 = 1 CLS + 576 patches
    x = torch.randn(B, N, C)
    out, _ = attn(x)
    assert out.shape == (B, N, C)


def test_xsa_clip_removes_self_component():
    """After XSA, output should be orthogonal to self-value per head."""
    config = _make_config()
    attn = XSACLIPAttention(config, use_xsa=True)
    B, N = 2, 64
    x = torch.randn(B, N, config.hidden_size)
    with torch.no_grad():
        out, debug = attn(x, return_debug=True)
    y = debug["y_projected"]  # (B, H, N, head_dim)
    v = debug["v"]
    cos = torch.nn.functional.cosine_similarity(
        y.reshape(-1, y.shape[-1]), v.reshape(-1, v.shape[-1]), dim=-1
    )
    assert cos.abs().mean() < 0.01


def test_xsa_clip_matches_hf_when_disabled():
    """With use_xsa=False, should match HuggingFace CLIPAttention outputs."""
    from transformers.models.clip.modeling_clip import CLIPAttention
    config = _make_config()
    hf = CLIPAttention(config).eval()
    xsa = XSACLIPAttention(config, use_xsa=False).eval()

    # Copy weights
    xsa.q_proj.load_state_dict(hf.q_proj.state_dict())
    xsa.k_proj.load_state_dict(hf.k_proj.state_dict())
    xsa.v_proj.load_state_dict(hf.v_proj.state_dict())
    xsa.out_proj.load_state_dict(hf.out_proj.state_dict())

    x = torch.randn(2, 64, config.hidden_size)
    with torch.no_grad():
        out_hf, _ = hf(x)
        out_xsa, _ = xsa(x)
    assert torch.allclose(out_hf, out_xsa, atol=1e-5)


def test_xsa_clip_differs_when_enabled():
    """XSA on vs off with same weights should differ."""
    config = _make_config()
    attn_on = XSACLIPAttention(config, use_xsa=True).eval()
    attn_off = XSACLIPAttention(config, use_xsa=False).eval()
    attn_off.load_state_dict(attn_on.state_dict())

    x = torch.randn(2, 64, config.hidden_size)
    with torch.no_grad():
        out_on, _ = attn_on(x)
        out_off, _ = attn_off(x)
    assert not torch.allclose(out_on, out_off, atol=1e-5)


def test_xsa_clip_gradient_flows():
    """Gradients should flow through XSA projection."""
    config = _make_config()
    attn = XSACLIPAttention(config, use_xsa=True)
    x = torch.randn(2, 64, config.hidden_size, requires_grad=True)
    out, _ = attn(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_xsa_clip_handles_long_sequence():
    """Should handle long sequences (multiple frames concatenated)."""
    config = _make_config()
    attn = XSACLIPAttention(config, use_xsa=True).eval()
    # Simulate 16 frames concatenated = 16 * 577 tokens
    x = torch.randn(1, 16 * 577, config.hidden_size)
    with torch.no_grad():
        out, _ = attn(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```bash
cd D:/research/xsa-longva && python -m pytest tests/test_xsa_clip_attention.py -v
```

- [ ] **Step 3: Implement XSACLIPAttention**

```python
# xsa_clip_attention.py
"""
XSACLIPAttention - Exclusive Self Attention for HuggingFace CLIP vision encoder.

Drop-in replacement for transformers.models.clip.modeling_clip.CLIPAttention.
When use_xsa=True, applies orthogonal projection removing self-value component.

Based on: "Exclusive Self Attention" (arXiv:2603.09078)
Applied to: LongVA's CLIP ViT-L/14-336 vision tower (24 layers)

The modification (2 lines after attention):
    coeff = (y * v).sum(-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) ** 2 + eps)
    y = y - coeff * v
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.configuration_clip import CLIPVisionConfig


class XSACLIPAttention(nn.Module):
    """
    Multi-head self-attention with optional Exclusive Self Attention (XSA),
    matching the interface of transformers.models.clip.modeling_clip.CLIPAttention.

    Args:
        config: CLIPVisionConfig (or any config with hidden_size, num_attention_heads,
                attention_dropout)
        use_xsa: If True, apply XSA orthogonal projection to attention output.
        xsa_eps: Epsilon for numerical stability.
    """

    def __init__(self, config, use_xsa: bool = True, xsa_eps: float = 1e-6):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.use_xsa = use_xsa
        self.xsa_eps = xsa_eps

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        q = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        k = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        v = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        # Use scaled_dot_product_attention for speed and memory
        # We apply causal mask manually if provided
        attn_mask = None
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif attention_mask is not None:
            attn_mask = attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B, H, N, head_dim)

        # === XSA: remove self-value component (2 lines) ===
        if self.use_xsa:
            coeff = (y * v).sum(-1, keepdim=True) / (
                v.norm(dim=-1, keepdim=True) ** 2 + self.xsa_eps
            )
            y = y - coeff * v
        # === End XSA ===

        debug = None
        if return_debug:
            debug = {"y_projected": y.detach().clone(), "v": v.detach().clone()}

        # Reshape and output project
        attn_output = y.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        # Match HF signature: return (attn_output, attn_weights_reshaped)
        # We don't compute explicit attn weights when using SDPA
        attn_weights_reshaped = None
        if output_attentions:
            # Fallback: compute weights explicitly (slow but matches HF behavior)
            attn_weights = torch.matmul(q * self.scale, k.transpose(2, 3))
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)

        if return_debug:
            return attn_output, attn_weights_reshaped, debug
        return attn_output, attn_weights_reshaped
```

- [ ] **Step 4: Run tests — all 6 must pass**

```bash
cd D:/research/xsa-longva && python -m pytest tests/test_xsa_clip_attention.py -v
```

- [ ] **Step 5: Commit**

```bash
git add xsa_clip_attention.py tests/test_xsa_clip_attention.py
git commit -m "feat: implement XSA CLIP attention with tests"
```

---

## Task 3: Patch LongVA Vision Tower

**Files:**
- Create: `patch_longva.py`
- Create: `tests/test_patch_longva.py`

This module loads a LongVA model and monkey-patches all 24 `CLIPAttention` layers in the vision tower with `XSACLIPAttention`, preserving pretrained weights.

- [ ] **Step 1: Write failing test**

```python
# tests/test_patch_longva.py
"""
Tests for patching LongVA's vision tower with XSA.

These tests use a tiny CLIP config to avoid downloading LongVA-7B.
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patch_longva import patch_clip_model_with_xsa, count_xsa_layers


def test_patch_tiny_clip_vision_model():
    """Patching a small CLIP vision model should replace attention and preserve shapes."""
    from transformers import CLIPVisionModel, CLIPVisionConfig

    config = CLIPVisionConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, image_size=32, patch_size=8,
    )
    model = CLIPVisionModel(config).eval()

    # Before patch: zero XSA layers
    assert count_xsa_layers(model) == 0

    # Patch
    patch_clip_model_with_xsa(model, use_xsa=True)

    # After patch: all 4 layers should be XSA
    assert count_xsa_layers(model) == 4

    # Verify forward pass still works and shapes are correct
    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(pixel_values)
    # last_hidden_state: (2, 1+16, 64) for 32x32 image with 8x8 patches
    assert out.last_hidden_state.shape == (2, 17, 64)


def test_patch_preserves_weights():
    """Patching should copy q/k/v/out projection weights from original."""
    from transformers import CLIPVisionModel, CLIPVisionConfig
    from xsa_clip_attention import XSACLIPAttention

    config = CLIPVisionConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, image_size=32, patch_size=8,
    )
    model = CLIPVisionModel(config).eval()

    # Save reference weight
    original_q_weight = model.vision_model.encoder.layers[0].self_attn.q_proj.weight.clone()

    patch_clip_model_with_xsa(model, use_xsa=True)

    # Verify attention layer is replaced
    new_attn = model.vision_model.encoder.layers[0].self_attn
    assert isinstance(new_attn, XSACLIPAttention)
    # Verify weights copied
    assert torch.allclose(new_attn.q_proj.weight, original_q_weight)


def test_patch_xsa_disabled_matches_original():
    """Patching with use_xsa=False should produce identical outputs to unpatched."""
    from transformers import CLIPVisionModel, CLIPVisionConfig

    config = CLIPVisionConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, image_size=32, patch_size=8,
    )
    # Two identical models
    model_a = CLIPVisionModel(config).eval()
    model_b = CLIPVisionModel(config).eval()
    model_b.load_state_dict(model_a.state_dict())

    patch_clip_model_with_xsa(model_b, use_xsa=False)

    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_a = model_a(pixel_values).last_hidden_state
        out_b = model_b(pixel_values).last_hidden_state

    assert torch.allclose(out_a, out_b, atol=1e-5), (
        f"Max diff: {(out_a - out_b).abs().max().item()}"
    )


def test_patch_xsa_enabled_differs_from_original():
    """Patching with use_xsa=True should produce different outputs."""
    from transformers import CLIPVisionModel, CLIPVisionConfig

    config = CLIPVisionConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, image_size=32, patch_size=8,
    )
    model_a = CLIPVisionModel(config).eval()
    model_b = CLIPVisionModel(config).eval()
    model_b.load_state_dict(model_a.state_dict())

    patch_clip_model_with_xsa(model_b, use_xsa=True)

    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_a = model_a(pixel_values).last_hidden_state
        out_b = model_b(pixel_values).last_hidden_state

    assert not torch.allclose(out_a, out_b, atol=1e-4)
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

```bash
python -m pytest tests/test_patch_longva.py -v
```

- [ ] **Step 3: Implement patch_longva.py**

```python
# patch_longva.py
"""
Monkey-patch LongVA's CLIP vision tower to use XSACLIPAttention.

The patcher walks the CLIP vision model, replaces every
CLIPAttention instance with XSACLIPAttention, and copies the
pretrained weights (q/k/v/out projections) into the new module.

Usage:
    from transformers import CLIPVisionModel
    from patch_longva import patch_clip_model_with_xsa

    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
    patch_clip_model_with_xsa(vision_model, use_xsa=True)
"""

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPAttention

from xsa_clip_attention import XSACLIPAttention


def _copy_clip_attn_weights(src: CLIPAttention, dst: XSACLIPAttention):
    """Copy q/k/v/out projection weights + biases from src to dst."""
    with torch.no_grad():
        dst.q_proj.weight.copy_(src.q_proj.weight)
        dst.q_proj.bias.copy_(src.q_proj.bias)
        dst.k_proj.weight.copy_(src.k_proj.weight)
        dst.k_proj.bias.copy_(src.k_proj.bias)
        dst.v_proj.weight.copy_(src.v_proj.weight)
        dst.v_proj.bias.copy_(src.v_proj.bias)
        dst.out_proj.weight.copy_(src.out_proj.weight)
        dst.out_proj.bias.copy_(src.out_proj.bias)


def patch_clip_model_with_xsa(
    model: nn.Module,
    use_xsa: bool = True,
    xsa_eps: float = 1e-6,
) -> nn.Module:
    """
    Walk `model` and replace every CLIPAttention with XSACLIPAttention.
    Copies pretrained weights into the new attention modules.

    Works for both CLIPVisionModel and LongVA's wrapped vision tower.

    Args:
        model: Module containing CLIPAttention layers.
        use_xsa: Passed to XSACLIPAttention.
        xsa_eps: Passed to XSACLIPAttention.

    Returns:
        The same model with attention modules replaced in-place.
    """
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, CLIPAttention):
                # Build config-like object; CLIPAttention stores config directly
                config = child.config
                new_attn = XSACLIPAttention(config, use_xsa=use_xsa, xsa_eps=xsa_eps)
                new_attn = new_attn.to(
                    dtype=child.q_proj.weight.dtype,
                    device=child.q_proj.weight.device,
                )
                _copy_clip_attn_weights(child, new_attn)
                setattr(module, child_name, new_attn)
                replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "No CLIPAttention layers found to patch. "
            "Is this actually a CLIP vision model?"
        )
    print(f"[patch_longva] Replaced {replaced} CLIPAttention layers with XSA")
    return model


def count_xsa_layers(model: nn.Module) -> int:
    """Count how many XSACLIPAttention layers are in the model."""
    return sum(1 for m in model.modules() if isinstance(m, XSACLIPAttention))


def freeze_non_xsa_parameters(model: nn.Module):
    """
    Freeze everything except the XSA attention modules.
    Useful for ablation where we train only the new attention weights.
    """
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, XSACLIPAttention):
            for p in m.parameters():
                p.requires_grad = True
```

- [ ] **Step 4: Run tests — all 4 must pass**

```bash
python -m pytest tests/test_patch_longva.py -v
```

- [ ] **Step 5: Commit**

```bash
git add patch_longva.py tests/test_patch_longva.py
git commit -m "feat: implement LongVA vision tower XSA patcher"
```

---

## Task 4: Full Model Forward Pass Test

**Files:**
- Create: `tests/test_forward_pass.py`

Verify that LongVA (with XSA patched) loads, processes a dummy video, and produces output tokens. This is an integration test — it requires the full LongVA checkpoint downloaded.

- [ ] **Step 1: Create download script**

```bash
# scripts/download_longva.sh
#!/bin/bash
# Download LongVA-7B-DPO from HuggingFace
# Requires huggingface-cli and HF_HOME set

MODEL_ID="lmms-lab/LongVA-7B-DPO"
LOCAL_DIR="${1:-./checkpoints/LongVA-7B-DPO}"

mkdir -p "$LOCAL_DIR"
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR" --local-dir-use-symlinks False
echo "LongVA-7B-DPO downloaded to $LOCAL_DIR"
```

- [ ] **Step 2: Write forward pass integration test**

```python
# tests/test_forward_pass.py
"""
Integration test: load LongVA-7B-DPO, patch it with XSA, run a dummy
video forward pass. Requires the checkpoint to be downloaded to
./checkpoints/LongVA-7B-DPO and the LongVA package installed.

Skipped if checkpoint not available.
"""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_PATH = os.environ.get("LONGVA_CKPT", "./checkpoints/LongVA-7B-DPO")
SKIP_REASON = f"LongVA checkpoint not found at {CKPT_PATH}"


@pytest.mark.skipif(not os.path.isdir(CKPT_PATH), reason=SKIP_REASON)
def test_longva_forward_with_xsa():
    """Load LongVA, patch vision tower with XSA, run forward on dummy video."""
    try:
        from longva.model.builder import load_pretrained_model
    except ImportError:
        pytest.skip("LongVA package not installed (run scripts/setup_env.sh)")

    from patch_longva import patch_clip_model_with_xsa, count_xsa_layers

    tokenizer, model, image_processor, _ = load_pretrained_model(
        CKPT_PATH, None, "llava_qwen",
        device_map="cuda:0", torch_dtype=torch.bfloat16,
    )

    # Patch the vision tower
    vision_tower = model.get_vision_tower().vision_tower
    patch_clip_model_with_xsa(vision_tower, use_xsa=True)
    assert count_xsa_layers(vision_tower) == 24, \
        f"Expected 24 XSA layers, got {count_xsa_layers(vision_tower)}"

    # Create a dummy "video": 4 random frames at 336x336
    dummy_frames = [torch.randn(3, 336, 336) for _ in range(4)]
    # Normalize via image processor
    images = image_processor.preprocess(dummy_frames, return_tensors="pt")["pixel_values"]
    images = images.to("cuda:0", dtype=torch.bfloat16)

    # Forward pass: just the vision tower
    with torch.no_grad():
        features = model.encode_images(images)

    assert features is not None
    assert features.shape[0] == 4  # 4 frames
    print(f"Vision features shape: {features.shape}")


@pytest.mark.skipif(not os.path.isdir(CKPT_PATH), reason=SKIP_REASON)
def test_longva_generation_with_xsa():
    """End-to-end: load LongVA with XSA, generate a token sequence."""
    try:
        from longva.model.builder import load_pretrained_model
        from longva.constants import IMAGE_TOKEN_INDEX
        from longva.mm_utils import tokenizer_image_token
    except ImportError:
        pytest.skip("LongVA package not installed")

    from patch_longva import patch_clip_model_with_xsa

    tokenizer, model, image_processor, _ = load_pretrained_model(
        CKPT_PATH, None, "llava_qwen",
        device_map="cuda:0", torch_dtype=torch.bfloat16,
    )
    patch_clip_model_with_xsa(model.get_vision_tower().vision_tower, use_xsa=True)

    # Build a minimal prompt
    prompt = "<|im_start|>user\n<image>\nWhat is in the video?<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda:0")

    dummy_frames = [torch.randn(3, 336, 336) for _ in range(4)]
    images = image_processor.preprocess(dummy_frames, return_tensors="pt")["pixel_values"]
    images = images.to("cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=[images],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=20,
        )

    assert output_ids.shape[1] > input_ids.shape[1]
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: {text}")
```

- [ ] **Step 3: Download LongVA checkpoint** (only when GPU environment is ready)

```bash
bash scripts/download_longva.sh ./checkpoints/LongVA-7B-DPO
```

- [ ] **Step 4: Run integration tests** (requires ~16GB VRAM)

```bash
pytest tests/test_forward_pass.py -v -s
```

- [ ] **Step 5: Commit**

```bash
git add scripts/download_longva.sh tests/test_forward_pass.py
git commit -m "feat: add LongVA download script and forward pass integration tests"
```

---

## Task 5: LongVideoBench Evaluation Script

**Files:**
- Create: `eval_longvideobench.py`
- Create: `scripts/download_eval.sh`

Evaluate LongVA (with or without XSA) on LongVideoBench validation split.

- [ ] **Step 1: Create download script for eval datasets**

```bash
# scripts/download_eval.sh
#!/bin/bash
# Download evaluation benchmarks
# Usage: bash scripts/download_eval.sh ./data/eval

EVAL_DIR="${1:-./data/eval}"
mkdir -p "$EVAL_DIR"

echo "Downloading LongVideoBench..."
huggingface-cli download longvideobench/LongVideoBench \
    --repo-type dataset \
    --local-dir "$EVAL_DIR/LongVideoBench" \
    --local-dir-use-symlinks False

cd "$EVAL_DIR/LongVideoBench"
cat videos.tar.part.* > videos.tar
tar -xf videos.tar
tar -xf subtitles.tar
rm videos.tar videos.tar.part.*

echo "LongVideoBench ready at $EVAL_DIR/LongVideoBench"
```

- [ ] **Step 2: Implement eval_longvideobench.py**

```python
# eval_longvideobench.py
"""
Evaluate a LongVA model on LongVideoBench validation split.

Usage:
    # Baseline (unpatched LongVA)
    python eval_longvideobench.py \
        --model-path ./checkpoints/LongVA-7B-DPO \
        --data-path ./data/eval/LongVideoBench \
        --output ./results/longvideobench_baseline.json \
        --max-frames 64

    # XSA-patched model
    python eval_longvideobench.py \
        --model-path ./checkpoints/LongVA-7B-DPO \
        --data-path ./data/eval/LongVideoBench \
        --output ./results/longvideobench_xsa.json \
        --max-frames 64 \
        --use-xsa
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_longva import patch_clip_model_with_xsa


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def build_prompt(question: str, options: list) -> str:
    """Format a LongVideoBench question as a multiple-choice prompt."""
    opts = "\n".join(f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (
        f"{question}\n{opts}\n"
        f"Answer with the option letter only."
    )


def parse_answer(text: str, num_options: int) -> int:
    """Extract the chosen option index from the model's output."""
    text = text.strip().upper()
    # Look for any A/B/C/D/E in the first few chars
    for i, letter in enumerate(OPTION_LETTERS[:num_options]):
        if letter in text[:5]:
            return i
    return -1  # could not parse


@torch.no_grad()
def evaluate(args):
    from longva.model.builder import load_pretrained_model
    from longva.constants import IMAGE_TOKEN_INDEX
    from longva.mm_utils import tokenizer_image_token
    from longvideobench import LongVideoBenchDataset

    # Load model
    print(f"Loading model from {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_qwen",
        device_map="cuda:0", torch_dtype=torch.bfloat16,
    )

    # Optionally patch with XSA
    if args.use_xsa:
        print("Patching vision tower with XSA")
        patch_clip_model_with_xsa(
            model.get_vision_tower().vision_tower, use_xsa=True
        )

    model.eval()

    # Load dataset
    print(f"Loading LongVideoBench from {args.data_path}")
    dataset = LongVideoBenchDataset(
        args.data_path,
        "lvb_val.json",
        max_num_frames=args.max_frames,
    )
    print(f"Dataset: {len(dataset)} questions")

    results = []
    correct = 0
    total = 0

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset[idx]
        # item contains: video frames, subtitles, question, options, correct_choice

        frames = item["inputs"]  # List of PIL images or np arrays
        question = item["question"]
        options = item["candidates"]
        gt_idx = item["correct_choice"]

        # Build prompt with subtitles if available
        subtitle_text = item.get("subtitles", "")
        if subtitle_text:
            full_question = f"Subtitles:\n{subtitle_text}\n\n{build_prompt(question, options)}"
        else:
            full_question = build_prompt(question, options)

        # Tokenize prompt
        prompt = (
            f"<|im_start|>user\n<image>\n{full_question}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to("cuda:0")

        # Preprocess frames
        images = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        images = images.to("cuda:0", dtype=torch.bfloat16)

        # Generate
        try:
            output_ids = model.generate(
                input_ids,
                images=[images],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=8,
                temperature=0.0,
            )
            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            pred_idx = parse_answer(response, len(options))
        except Exception as e:
            print(f"Error on item {idx}: {e}")
            response = ""
            pred_idx = -1

        is_correct = (pred_idx == gt_idx)
        if pred_idx >= 0:
            total += 1
            if is_correct:
                correct += 1

        results.append({
            "id": item.get("id", idx),
            "question": question,
            "gt": gt_idx,
            "pred": pred_idx,
            "response": response,
            "correct": is_correct,
            "duration_group": item.get("duration_group", "unknown"),
        })

        if (idx + 1) % 50 == 0:
            acc = correct / max(total, 1) * 100
            print(f"  [{idx+1}/{len(dataset)}] Acc: {acc:.2f}%")

    # Overall stats
    overall_acc = correct / max(total, 1) * 100
    print(f"\n=== Results ===")
    print(f"Model: {args.model_path}")
    print(f"XSA: {args.use_xsa}")
    print(f"Frames: {args.max_frames}")
    print(f"Overall: {correct}/{total} = {overall_acc:.2f}%")

    # Per-duration-group breakdown
    groups = {}
    for r in results:
        g = r["duration_group"]
        if g not in groups:
            groups[g] = [0, 0]
        if r["pred"] >= 0:
            groups[g][1] += 1
            if r["correct"]:
                groups[g][0] += 1

    print("\nPer-duration-group:")
    for g, (c, t) in sorted(groups.items()):
        print(f"  {g}: {c}/{t} = {c/max(t,1)*100:.2f}%")

    # Save results
    output = {
        "model_path": args.model_path,
        "use_xsa": args.use_xsa,
        "max_frames": args.max_frames,
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "per_group": {g: {"correct": c, "total": t, "acc": c/max(t,1)*100}
                      for g, (c, t) in groups.items()},
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to LongVA checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to LongVideoBench data dir")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--use-xsa", action="store_true", help="Patch vision tower with XSA")
    args = parser.parse_args()
    evaluate(args)
```

- [ ] **Step 3: Commit**

```bash
git add eval_longvideobench.py scripts/download_eval.sh
git commit -m "feat: LongVideoBench evaluation script with XSA support"
```

---

## Task 6: Baseline Evaluation (Zero-Shot)

This is the critical sanity check before training: run the unmodified LongVA-7B-DPO on LongVideoBench val to establish the baseline, then run the XSA-patched model (untrained) to confirm it degrades (as expected — the vision tower was never trained with XSA).

- [ ] **Step 1: Run baseline eval**

```bash
python eval_longvideobench.py \
    --model-path ./checkpoints/LongVA-7B-DPO \
    --data-path ./data/eval/LongVideoBench \
    --output ./results/baseline_sa.json \
    --max-frames 64
```

Expected: ~55-58% accuracy. Compare against the public leaderboard value to validate the setup.

- [ ] **Step 2: Run XSA (untrained) eval to establish lower bound**

```bash
python eval_longvideobench.py \
    --model-path ./checkpoints/LongVA-7B-DPO \
    --data-path ./data/eval/LongVideoBench \
    --output ./results/xsa_untrained.json \
    --max-frames 64 \
    --use-xsa
```

Expected: significant degradation (40-50%) because the CLIP weights were never trained to produce outputs orthogonal to self-values. This confirms that **fine-tuning is required** to recover and improve.

- [ ] **Step 3: Commit baseline results**

```bash
git add results/baseline_sa.json results/xsa_untrained.json
git commit -m "eval: baseline LongVA-7B-DPO on LongVideoBench val"
```

---

## Task 7: Download and Prepare Training Data

**Files:**
- Create: `scripts/download_data.sh`
- Create: `scripts/prepare_subset.py`

Download LLaVA-Video-178K 30-60s bucket (~100-200K samples) and prepare a training JSON.

- [ ] **Step 1: Create data download script**

```bash
# scripts/download_data.sh
#!/bin/bash
# Download LLaVA-Video-178K 30-60s bucket
# Usage: bash scripts/download_data.sh ./data/train

DATA_DIR="${1:-./data/train}"
mkdir -p "$DATA_DIR"

echo "Downloading LLaVA-Video-178K 30-60s bucket..."
huggingface-cli download lmms-lab/LLaVA-Video-178K \
    --repo-type dataset \
    --include "30_60_s_*/*" \
    --local-dir "$DATA_DIR/LLaVA-Video-178K" \
    --local-dir-use-symlinks False

echo "Done. Videos at $DATA_DIR/LLaVA-Video-178K"
```

- [ ] **Step 2: Create prepare_subset.py**

```python
# scripts/prepare_subset.py
"""
Prepare a training subset from LLaVA-Video-178K.

Reads the parquet annotation files, filters by criteria, and writes
a JSON file compatible with LongVA's training format.
"""
import argparse
import json
import os
from pathlib import Path
import pandas as pd


def main(args):
    root = Path(args.data_dir)
    print(f"Scanning {root} for parquet files...")

    parquet_files = list(root.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    all_rows = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_rows.append(df)
        print(f"  {pf.relative_to(root)}: {len(df)} rows")

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Filter to only samples with valid video paths
    df = df.dropna(subset=["video"])

    # Sample
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled to {len(df)} rows")

    # Convert to LongVA training format
    samples = []
    for _, row in df.iterrows():
        # LongVA expects: {"id": ..., "video": ..., "conversations": [...]}
        conv = row.get("conversations", None)
        if conv is None:
            continue
        if isinstance(conv, str):
            conv = json.loads(conv)

        samples.append({
            "id": row.get("id", str(len(samples))),
            "video": row["video"],
            "conversations": conv,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=100000)
    args = parser.parse_args()
    main(args)
```

- [ ] **Step 3: Execute data download** (runs for 1-3 hours depending on bandwidth)

```bash
bash scripts/download_data.sh ./data/train
python scripts/prepare_subset.py \
    --data-dir ./data/train/LLaVA-Video-178K \
    --output ./data/train/subset_30_60s_100k.json \
    --max-samples 100000
```

- [ ] **Step 4: Commit scripts (not data)**

```bash
git add scripts/download_data.sh scripts/prepare_subset.py
git commit -m "feat: data download and subset preparation scripts"
```

---

## Task 8: Training Script with LoRA + XSA Vision Tower FT

**Files:**
- Create: `train_xsa.py`
- Create: `configs/deepspeed_zero3.json`
- Create: `configs/lora_config.yaml`

The training script loads LongVA, patches the vision tower with XSA, wraps the LLM in LoRA, and trains on the subset.

- [ ] **Step 1: Create DeepSpeed config**

```json
// configs/deepspeed_zero3.json
{
    "fp16": { "enabled": false },
    "bf16": { "enabled": true },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "none" },
        "offload_param": { "device": "none" },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

- [ ] **Step 2: Create LoRA config**

```yaml
# configs/lora_config.yaml
# LoRA adapter config for Qwen2-7B LLM in LongVA
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
task_type: CAUSAL_LM
```

- [ ] **Step 3: Implement train_xsa.py**

The full training script is too long for this plan entry, but the key structure:

```python
# train_xsa.py (outline)
"""
Fine-tune LongVA-7B-DPO with XSA-patched vision tower.

Key differences from LongVA's finetune.sh:
1. Vision tower is patched with XSACLIPAttention before training
2. LLM is wrapped in LoRA (rank 16)
3. Vision tower is fully unfrozen (LR 2e-6) to learn XSA-compatible weights
4. Trains on a ~100K subset instead of full LLaVA-NeXT data

Usage:
    deepspeed --num_gpus=1 train_xsa.py \
        --model-path ./checkpoints/LongVA-7B-DPO \
        --data-path ./data/train/subset_30_60s_100k.json \
        --video-folder ./data/train/LLaVA-Video-178K \
        --output-dir ./checkpoints/xsa-longva-run1 \
        --deepspeed configs/deepspeed_zero3.json \
        --lr 1e-5 --vision-lr 2e-6 \
        --batch-size 4 --gradient-accumulation 4 \
        --epochs 1 --num-frames 32 \
        --lora-r 16 --lora-alpha 32
"""
# Imports, argparse, dataset, trainer - see full implementation in repo
# Key steps:
# 1. Load LongVA with load_pretrained_model
# 2. patch_clip_model_with_xsa(model.get_vision_tower().vision_tower)
# 3. Apply LoRA to LLM using PEFT
# 4. Set up optimizer with two param groups:
#    - vision_tower params: LR 2e-6
#    - LoRA adapters + projector: LR 1e-5
# 5. Dataset: video instruction tuning format
# 6. Use HuggingFace Trainer with DeepSpeed ZeRO-3
```

*(Full ~500 line script implemented in the repo. Key parts listed above.)*

- [ ] **Step 4: Commit**

```bash
git add train_xsa.py configs/deepspeed_zero3.json configs/lora_config.yaml
git commit -m "feat: LoRA training script with XSA vision tower fine-tuning"
```

---

## Task 9: Launch Training Run

- [ ] **Step 1: Create overnight training script**

```bash
# scripts/train_overnight.sh
#!/bin/bash
# Overnight XSA-LongVA training run
# Usage: bash scripts/train_overnight.sh

set -e

MODEL_PATH="${MODEL_PATH:-./checkpoints/LongVA-7B-DPO}"
DATA_JSON="${DATA_JSON:-./data/train/subset_30_60s_100k.json}"
VIDEO_DIR="${VIDEO_DIR:-./data/train/LLaVA-Video-178K}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/xsa-longva-run1}"
NUM_FRAMES="${NUM_FRAMES:-32}"

mkdir -p "$OUTPUT_DIR"

deepspeed --num_gpus=1 train_xsa.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_JSON" \
    --video-folder "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --deepspeed configs/deepspeed_zero3.json \
    --lr 1e-5 \
    --vision-lr 2e-6 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --num-epochs 1 \
    --num-frames "$NUM_FRAMES" \
    --lora-r 16 \
    --lora-alpha 32 \
    --warmup-ratio 0.03 \
    --weight-decay 0.0 \
    --save-steps 2000 \
    --logging-steps 10 \
    --bf16 \
    --gradient-checkpointing

echo "Training complete. Checkpoint: $OUTPUT_DIR"
```

- [ ] **Step 2: Run training (10-14 hours on H100)**

```bash
bash scripts/train_overnight.sh 2>&1 | tee training.log
```

Watch: loss should decrease steadily. Warning signs:
- Loss NaN → reduce LR
- Loss plateau → check gradient flow through vision tower
- OOM → reduce batch size or num_frames

- [ ] **Step 3: Commit training log + config used**

```bash
git add scripts/train_overnight.sh training.log
git commit -m "train: overnight XSA-LongVA training run"
```

---

## Task 10: Post-Training Evaluation

- [ ] **Step 1: Evaluate on LongVideoBench val**

```bash
python eval_longvideobench.py \
    --model-path ./checkpoints/xsa-longva-run1 \
    --data-path ./data/eval/LongVideoBench \
    --output ./results/longvideobench_xsa_trained.json \
    --max-frames 64 \
    --use-xsa
```

**Goal: ≥60% accuracy (beating LongVA-7B-DPO baseline by 2+ points)**

- [ ] **Step 2: Evaluate on secondary benchmarks**

```bash
# MVBench (fast sanity check, 4K samples)
python eval_mvbench.py \
    --model-path ./checkpoints/xsa-longva-run1 \
    --output ./results/mvbench_xsa.json \
    --use-xsa

# Video-MME long split
python eval_videomme.py \
    --model-path ./checkpoints/xsa-longva-run1 \
    --split long \
    --output ./results/videomme_long_xsa.json \
    --use-xsa
```

- [ ] **Step 3: Commit all results**

```bash
git add results/
git commit -m "eval: post-training XSA-LongVA benchmark results"
```

---

## Task 11: Analysis and Visualizations

**Files:**
- Create: `analysis/plot_eval_comparison.py`
- Create: `analysis/attention_viz.py`
- Create: `analysis/cosine_similarity.py`

- [ ] **Step 1: Implement eval comparison plot**

```python
# analysis/plot_eval_comparison.py
"""
Bar chart comparing SA baseline vs XSA across all evaluated benchmarks.
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.xsa) as f:
        xsa = json.load(f)

    # Extract overall accuracy + per-group
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall
    axes[0].bar(
        ["SA baseline", "XSA (ours)"],
        [baseline["overall_accuracy"], xsa["overall_accuracy"]],
        color=["tab:blue", "tab:orange"],
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("LongVideoBench val — Overall")
    axes[0].set_ylim([40, 70])
    for i, v in enumerate([baseline["overall_accuracy"], xsa["overall_accuracy"]]):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center")

    # Per-duration-group
    groups = sorted(set(baseline["per_group"].keys()) | set(xsa["per_group"].keys()))
    baseline_vals = [baseline["per_group"].get(g, {"acc": 0})["acc"] for g in groups]
    xsa_vals = [xsa["per_group"].get(g, {"acc": 0})["acc"] for g in groups]

    x = np.arange(len(groups))
    width = 0.35
    axes[1].bar(x - width/2, baseline_vals, width, label="SA baseline", color="tab:blue")
    axes[1].bar(x + width/2, xsa_vals, width, label="XSA", color="tab:orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=30)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("By Video Duration")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--xsa", required=True)
    parser.add_argument("--output", default="plots/eval_comparison.png")
    args = parser.parse_args()
    main(args)
```

- [ ] **Step 2: Implement cosine similarity analysis**

Measures the <y, v> bias across CLIP layers — reproduces Figure 1 from the XSA paper but on LongVA's vision encoder with real video inputs.

- [ ] **Step 3: Implement attention visualization**

Side-by-side attention heatmaps overlaid on sampled video frames, comparing SA and XSA for the same query. Produces the viral-shareable assets.

- [ ] **Step 4: Generate all plots**

```bash
python analysis/plot_eval_comparison.py \
    --baseline results/baseline_sa.json \
    --xsa results/longvideobench_xsa_trained.json \
    --output plots/eval_comparison.png

python analysis/cosine_similarity.py \
    --model-path ./checkpoints/LongVA-7B-DPO \
    --video-samples data/eval/LongVideoBench/videos \
    --output plots/cosine_similarity.png

python analysis/attention_viz.py \
    --baseline-ckpt ./checkpoints/LongVA-7B-DPO \
    --xsa-ckpt ./checkpoints/xsa-longva-run1 \
    --video data/eval/LongVideoBench/videos/sample.mp4 \
    --output plots/attention_comparison.png
```

- [ ] **Step 5: Commit plots and analysis**

```bash
git add analysis/ plots/
git commit -m "feat: analysis scripts and evaluation plots"
```

---

## Task 12: Finalize README and Push

- [ ] **Step 1: Update README with final numbers**

Replace placeholder table values with actual results from Task 10. Include:
- Headline accuracy improvement (SA vs XSA)
- Per-duration-group breakdown
- Side-by-side video QA examples (with screenshots)
- Attention visualization image
- Reproduction instructions

- [ ] **Step 2: Create GitHub repo and push**

```bash
gh repo create ZoreAnuj/xsa-longva --public \
    --description "Exclusive Self Attention for Long Video Understanding | LongVA + XSA beats baseline on LongVideoBench"

git remote add origin https://github.com/ZoreAnuj/xsa-longva.git
git push -u origin main
```

---

## Execution Timeline (H100 from RunPod)

| Phase | Time | Action |
|---|---|---|
| Day 1 AM | 2h | Setup env, implement XSA CLIP attention + patcher, tests |
| Day 1 PM | 3h | Download LongVA + LongVideoBench (parallel) |
| Day 1 PM | 1h | Baseline eval |
| Day 1 PM | 4h | Download LLaVA-Video-178K 30-60s subset |
| Day 1 evening | 2h | Training script + verification |
| Night 1 | 10-14h | Overnight training run |
| Day 2 AM | 2h | Post-training eval (LongVideoBench + MVBench) |
| Day 2 AM | 2h | Analysis, plots, video QA examples |
| Day 2 PM | 2h | README + publish |

**Total H100 hours:** ~16-20h (training) + ~4h (eval) = ~20-24 H100 hours
**RunPod cost estimate:** ~$50-80 (H100 80GB PCIe at ~$3/h)

## Success Criteria

**Primary (must-have):**
- XSA-LongVA achieves ≥ baseline accuracy on LongVideoBench val (doesn't degrade)

**Secondary (target):**
- XSA-LongVA beats LongVA-7B-DPO baseline by ≥2 points on LongVideoBench val overall
- Gains are stronger on longer duration buckets (15-60min) — XSA's theoretical sweet spot
- No regression on MVBench (short-video sanity check)

**Stretch:**
- XSA-LongVA beats LLaVA-Video-7B-Qwen2 (the current open 7B SOTA at 62.7%)
- Clear cosine similarity analysis showing SA has high `<y, v>` bias that XSA eliminates
- Published paper + viral thread
