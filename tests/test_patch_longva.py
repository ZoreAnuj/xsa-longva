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
