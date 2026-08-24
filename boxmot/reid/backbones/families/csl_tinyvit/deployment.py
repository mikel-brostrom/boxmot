# BoxMOT AGPL-3.0 license

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.blocks import fuse_conv2d_bn_eval_
from boxmot.reid.backbones.heads.bnneck import BNNeck3

__all__ = ["FoldedBNNeck", "optimize_csl_tinyvit_for_inference"]


class FoldedBNNeck(nn.Module):
    """Minimal inference-only replacement for a projection plus BatchNorm."""

    def __init__(self, source: BNNeck3) -> None:
        super().__init__()
        source.prepare_for_inference()
        self.register_buffer("weight", source._inference_weight.detach().clone())
        self.register_buffer("bias", source._inference_bias.detach().clone())
        self.stride = source.reduction.stride
        self.padding = source.reduction.padding
        self.dilation = source.reduction.dilation
        self.groups = source.reduction.groups

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Return the folded post-BN descriptor."""
        projected = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return projected.flatten(1)


def _can_replace_multibranch_necks(head: nn.Module) -> bool:
    """Return whether eval is guaranteed to use only folded BNNeck outputs."""
    return bool(
        getattr(head, "inference_feature", None) == "norm_concat_bn"
        and getattr(head, "part_pooling", None) in {"stripes", "overlap_stripes"}
        and getattr(head, "branch_attention", None) is None
        and getattr(head, "branch_set_attention", None) is None
        and getattr(head, "multiscale_query_decoder", None) is None
        and not getattr(head, "anatomical_deployment_enabled", False)
        and not getattr(head, "emit_late_interaction_packet", False)
        and getattr(head, "visibility_gate", None) is None
        and not getattr(head, "has_specialists", False)
        and hasattr(head, "branch_specs")
        and hasattr(head, "_bn_attr")
    )


def _has_shared_multibranch_head(model: nn.Module) -> bool:
    """Return whether a model uses the deployable hierarchical ReID head."""
    head = getattr(model, "head", None)
    return bool(
        head is not None
        and hasattr(head, "branch_specs")
        and hasattr(head, "_bn_attr")
    )


def _replace_multibranch_necks(head: nn.Module) -> int:
    if not _can_replace_multibranch_necks(head):
        return 0
    replacements = 0
    for key, _, _ in head.branch_specs:
        attr = head._bn_attr(key)
        source = getattr(head, attr)
        if isinstance(source, BNNeck3):
            setattr(head, attr, FoldedBNNeck(source))
            replacements += 1
    return replacements


def _prune_training_only_modules(model: nn.Module) -> int:
    """Remove classifiers and nondeployed privileged ReID branches in place."""
    pruned = 0
    head = getattr(model, "head", None)
    if head is not None and not getattr(head, "anatomical_deployment_enabled", False):
        if getattr(head, "anatomical_auxiliary_pool", None) is not None:
            head.anatomical_auxiliary_pool = None
            head.anatomical_auxiliary_enabled = False
            pruned += 1
        if getattr(head, "jpm", None) is not None:
            head.jpm = None
            head.jpm_enabled = False
            pruned += 1

    for module in tuple(model.modules()):
        for name, child in tuple(module.named_children()):
            if name == "classifier" or name.endswith("_classifier"):
                if not isinstance(child, nn.Identity):
                    setattr(module, name, nn.Identity())
                    pruned += 1
    return pruned


def optimize_csl_tinyvit_for_inference(model: nn.Module) -> nn.Module:
    """Apply equivalent hierarchical-ReID inference fusions in place.

    Plain convolutions replace any backbone ``Conv2d_BN`` containers.
    Standard fixed-stripe retrieval heads are replaced with genuinely compact
    folded BNNecks, and unreachable classifiers / privileged training branches
    are removed for both CSL-TinyViT and MobileNetV4. Specialized heads retain
    their source BNNecks and use the nonpersistent folded cache because their
    alternate descriptors still need pre-BN features.
    """
    if model.training:
        raise RuntimeError("Hierarchical ReID deployment optimization requires eval mode")

    fused_count = fuse_conv2d_bn_eval_(model)
    supports_deployment_optimization = bool(
        fused_count > 0
        or getattr(model, "_csl_tinyvit_inference_optimized", False)
        or _has_shared_multibranch_head(model)
    )
    if not supports_deployment_optimization:
        return model

    # An eager forward under inference_mode may have populated BNNeck caches
    # with inference tensors, which cannot subsequently be captured by a
    # grad-enabled tracer. Rebuild every cache from authoritative source state.
    for module in tuple(model.modules()):
        if isinstance(module, BNNeck3):
            module._invalidate_inference_cache()
    pruned_count = _prune_training_only_modules(model)
    replaced_necks = _replace_multibranch_necks(getattr(model, "head", nn.Identity()))
    for module in tuple(model.modules()):
        if isinstance(module, BNNeck3):
            module.prepare_for_inference()
    model._csl_tinyvit_inference_optimized = True
    model._csl_tinyvit_pruned_module_count = pruned_count
    model._csl_tinyvit_replaced_bnneck_count = replaced_necks
    return model
