"""Monkey-patch LongVA's CLIP vision tower to use Exclusive Self Attention (XSA).

Provides utilities that walk a module tree and replace every HuggingFace
``CLIPAttention`` with :class:`XSACLIPAttention` while preserving pretrained
weights. This allows XSA to be applied to LongVA's vision tower at runtime
without modifying the HuggingFace library or retraining the model.

Typical usage::

    from transformers import CLIPVisionModel
    from patch_longva import patch_clip_model_with_xsa

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
    patch_clip_model_with_xsa(model, use_xsa=True)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from transformers.models.clip.modeling_clip import CLIPAttention

from xsa_clip_attention import XSACLIPAttention


__all__ = [
    "patch_clip_model_with_xsa",
    "count_xsa_layers",
    "freeze_non_xsa_parameters",
]


def _copy_clip_attn_weights(src: CLIPAttention, dst: XSACLIPAttention) -> None:
    """Copy q/k/v/out projection weights and biases from ``src`` into ``dst``.

    Both modules must have the same ``embed_dim``. The copy is performed under
    ``torch.no_grad()`` so it does not interfere with autograd.
    """
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
    """Replace every ``CLIPAttention`` in ``model`` with ``XSACLIPAttention``.

    Walks ``model``'s module tree, constructs a new :class:`XSACLIPAttention`
    for each HF ``CLIPAttention`` child it finds, copies over the pretrained
    projection weights and biases, and swaps the module in place on its parent.

    Args:
        model: Any PyTorch module containing one or more ``CLIPAttention``
            layers (e.g. a ``CLIPVisionModel`` or the top-level LongVA model).
        use_xsa: If ``True`` (default), the new attention layers apply the XSA
            self-component projection. If ``False``, they behave like vanilla
            attention -- useful as an A/B ablation control.
        xsa_eps: Numerical stabilizer passed to each new ``XSACLIPAttention``.

    Returns:
        The same ``model`` instance (modified in place).

    Raises:
        RuntimeError: If no ``CLIPAttention`` layers were found in ``model``.
    """
    # Collect replacement targets first so we don't mutate the module tree
    # while iterating over it.
    to_replace: List[Tuple[nn.Module, str, CLIPAttention]] = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, CLIPAttention):
                to_replace.append((parent, child_name, child))

    if not to_replace:
        raise RuntimeError(
            "patch_clip_model_with_xsa: no CLIPAttention layers were found in "
            "the given model. Check that you passed a CLIPVisionModel (or a "
            "wrapper containing one) and not, e.g., the vision tower object."
        )

    for parent, child_name, child in to_replace:
        ref_weight = child.q_proj.weight
        new_attn = XSACLIPAttention(
            child.config,
            use_xsa=use_xsa,
            xsa_eps=xsa_eps,
        ).to(dtype=ref_weight.dtype, device=ref_weight.device)
        _copy_clip_attn_weights(child, new_attn)
        setattr(parent, child_name, new_attn)

    print(
        f"[patch_longva] Replaced {len(to_replace)} CLIPAttention layers with XSA"
    )
    return model


def count_xsa_layers(model: nn.Module) -> int:
    """Return the number of :class:`XSACLIPAttention` modules inside ``model``."""
    return sum(1 for m in model.modules() if isinstance(m, XSACLIPAttention))


def freeze_non_xsa_parameters(model: nn.Module) -> nn.Module:
    """Freeze everything in ``model`` except parameters inside XSA attention.

    Sets ``requires_grad=False`` on every parameter in the model, then
    re-enables gradients only for parameters that live inside an
    :class:`XSACLIPAttention` module. Useful for ablation experiments where
    only the new attention weights should be trained.

    Returns:
        The same ``model`` instance (modified in place).
    """
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, XSACLIPAttention):
            for param in module.parameters():
                param.requires_grad = True

    return model
