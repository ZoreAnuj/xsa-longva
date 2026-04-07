"""Exclusive Self Attention (XSA) drop-in replacement for HuggingFace CLIPAttention.

Implements the XSA modification from "Exclusive Self Attention" (arXiv:2603.09078):
after standard scaled-dot-product attention, project out the component of each
token's attention output that is aligned with its own value vector. This forces
context-only information to flow through attention.

The class is interface-compatible with
``transformers.models.clip.modeling_clip.CLIPAttention`` so it can be
monkey-patched into a pretrained CLIP vision model without further changes.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.clip.configuration_clip import CLIPTextConfig, CLIPVisionConfig


class XSACLIPAttention(nn.Module):
    """Exclusive Self Attention variant of HuggingFace ``CLIPAttention``.

    Matches the HF interface (constructor, forward signature, return types) so
    instances can replace ``CLIPAttention`` modules in a pretrained model. The
    projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) are kept as
    separate ``nn.Linear`` modules with the exact attribute names HF uses, so
    weights can be copied directly via ``load_state_dict``.

    Args:
        config: A ``CLIPVisionConfig`` or ``CLIPTextConfig``.
        use_xsa: If True (default), apply the XSA self-component projection.
        xsa_eps: Numerical stabilizer for the XSA denominator.
    """

    def __init__(
        self,
        config: Union[CLIPVisionConfig, CLIPTextConfig],
        use_xsa: bool = True,
        xsa_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        # Same attribute names as HF CLIPAttention so state_dicts are
        # interchangeable.
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # XSA-specific options.
        self.use_xsa = use_xsa
        self.xsa_eps = xsa_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: ``(batch, seq_len, embed_dim)`` input.
            attention_mask: Optional additive attention mask.
            causal_attention_mask: Optional additive causal mask (CLIP text).
            output_attentions: If True, also return attention weights computed
                via an explicit eager path (SDPA does not expose them).
            return_debug: If True, return a third element with intermediate
                tensors used by the unit tests.

        Returns:
            ``(attn_output, attn_weights)`` by default, or
            ``(attn_output, attn_weights, debug)`` when ``return_debug=True``.
        """

        bsz, tgt_len, embed_dim = hidden_states.shape

        # Project to Q, K, V and reshape to (B, H, N, head_dim).
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine masks the same way HF CLIPAttention does so behavior matches
        # when XSA is disabled.
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        dropout_p = self.dropout if self.training else 0.0

        # Fast path: scaled-dot-product attention. Returns (B, H, N, head_dim).
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scale,
        )

        # XSA: project out the component of y aligned with each token's own v.
        if self.use_xsa:
            coeff = (y * v).sum(-1, keepdim=True) / (
                v.norm(dim=-1, keepdim=True) ** 2 + self.xsa_eps
            )
            y = y - coeff * v

        # (B, H, N, D) -> (B, N, H, D) -> (B, N, embed_dim)
        attn_output = y.transpose(1, 2).contiguous().reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        # SDPA does not expose attention weights; recompute eagerly only when
        # the caller asks for them. This keeps the fast path fast.
        attn_weights: Optional[torch.Tensor] = None
        if output_attentions:
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        if return_debug:
            debug = {"y_projected": y, "v": v, "q": q, "k": k}
            return attn_output, attn_weights, debug

        return attn_output, attn_weights
