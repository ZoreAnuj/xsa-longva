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
    config = CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=336,
        patch_size=14,
    )
    # HF CLIPAttention (>=4.51) reads config._attn_implementation at forward
    # time to dispatch to an attention backend. A bare config has this as
    # None, which raises KeyError: None. Set it here so the side-by-side
    # parity test against HF's CLIPAttention works. XSACLIPAttention itself
    # does not read this field.
    config._attn_implementation = "eager"
    return config


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
        out, _, debug = attn(x, return_debug=True)
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
