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

import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from transformers.models.clip.modeling_clip import CLIPAttention

from xsa_clip_attention import XSACLIPAttention


__all__ = [
    "patch_clip_model_with_xsa",
    "count_xsa_layers",
    "freeze_non_xsa_parameters",
    "set_xsa_alpha",
]


# CLIP layer paths look like:
#   vision_model.encoder.layers.22.self_attn
# We extract the integer layer index so callers can select a subset.
_CLIP_LAYER_IDX_RE = re.compile(r"encoder\.layers\.(\d+)\.self_attn")


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
    xsa_alpha: float = 1.0,
    only_layer_indices: Optional[Iterable[int]] = None,
) -> nn.Module:
    """Replace ``CLIPAttention`` layers in ``model`` with ``XSACLIPAttention``.

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
        xsa_alpha: Initial value of the projection scaling factor. 0 means
            the XSA modules start as plain attention; 1 means full XSA from
            the first forward pass.
        only_layer_indices: If provided, only CLIP encoder layers whose index
            is in this set will be patched. Layer index is parsed from the
            module name ``...encoder.layers.<i>.self_attn``. This lets callers
            run experiments with XSA applied to, e.g., just the last 12 of 24
            layers.

    Returns:
        The same ``model`` instance (modified in place).

    Raises:
        RuntimeError: If no ``CLIPAttention`` layers were found in ``model``,
            or if a layer filter was provided and no layer matched.
    """
    if only_layer_indices is not None:
        selected = set(int(i) for i in only_layer_indices)
    else:
        selected = None

    # Collect replacement targets first so we don't mutate the module tree
    # while iterating over it. We need the full dotted name of each parent
    # so we can parse out the encoder layer index and apply the filter.
    to_replace: List[Tuple[nn.Module, str, CLIPAttention, Optional[int]]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if not isinstance(child, CLIPAttention):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            m = _CLIP_LAYER_IDX_RE.search(full_name)
            layer_idx = int(m.group(1)) if m else None
            if selected is not None and (layer_idx is None or layer_idx not in selected):
                continue
            to_replace.append((parent, child_name, child, layer_idx))

    if not to_replace:
        raise RuntimeError(
            "patch_clip_model_with_xsa: no CLIPAttention layers were matched. "
            "Check that you passed a CLIPVisionModel (or a wrapper containing "
            "one) and that any `only_layer_indices` filter actually matches "
            "the layer numbering."
        )

    for parent, child_name, child, _layer_idx in to_replace:
        ref_weight = child.q_proj.weight
        new_attn = XSACLIPAttention(
            child.config,
            use_xsa=use_xsa,
            xsa_eps=xsa_eps,
            xsa_alpha=xsa_alpha,
        ).to(dtype=ref_weight.dtype, device=ref_weight.device)
        _copy_clip_attn_weights(child, new_attn)
        setattr(parent, child_name, new_attn)

    if selected is None:
        print(
            f"[patch_longva] Replaced {len(to_replace)} CLIPAttention layers with XSA"
        )
    else:
        patched_indices = sorted(
            {int(_CLIP_LAYER_IDX_RE.search(n).group(1))
             for n, _ in model.named_modules() if _CLIP_LAYER_IDX_RE.search(n) and
             isinstance(dict(model.named_modules())[n], XSACLIPAttention)}
        )
        print(
            f"[patch_longva] Replaced {len(to_replace)} CLIPAttention layers "
            f"with XSA (layer subset; indices={patched_indices})"
        )
    return model


def set_xsa_alpha(model: nn.Module, alpha: float) -> int:
    """Set the ``xsa_alpha`` buffer on every :class:`XSACLIPAttention` in ``model``.

    Returns the number of modules that were updated.
    """
    n = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, XSACLIPAttention):
                module.xsa_alpha.fill_(float(alpha))
                n += 1
    return n


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
