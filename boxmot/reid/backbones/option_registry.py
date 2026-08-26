"""Canonical categorical options shared by ReID configuration consumers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectorSpec:
    """One exclusive configuration selector and its canonical choices."""

    name: str
    choices: tuple[str, ...]

    def normalize(self, value: str) -> str:
        """Normalize and validate one selected mode."""
        normalized = str(value).lower()
        if normalized not in self.choices:
            raise ValueError(
                f"Unsupported {self.name}={value!r}; expected one of "
                f"{list(self.choices)}"
            )
        return normalized


CSL_FEATURE_FUSION_MODES = (
    "final",
    "last2",
    "last3",
    "last4_layer0_target",
    "last3_stage2_target",
    "last3_stage1_concat",
    "global_final_parts_stage1_concat",
    "global_final_parts_fpn_layer0",
    "last3_fpn_stage1_add",
    "last3_fpn_stage1_split",
    "last3_panet_stage1_split",
    "last3_panet_stage1_shared",
    "last3_panet_stage1_scale_aware",
    "last3_bifpn_stage1_split",
    "last3_bifpn_stage1_branch_aware",
    "global_final_parts_hierarchical_fpn",
    "last3_fpn_stage2",
    "last3_pafpn_stage2",
    "last4_fpn_layer0_target",
    "global_final_parts_stage2",
    "global_final_parts_stage2_semantic_residual",
    "global_final_parts_stage2_hierarchical_control",
    "global_final_parts_stage0_semantic_fine_reference",
    "global_final_parts_stage0_semantic_fine",
    "global_final_parts_stage0_fine_lite",
    "global_final_parts_stage0_panet_lite",
    "global_final_parts_stage0_bifpn_lite",
    "global_final_parts_stage0_native_pyramid",
    "global_final_parts_stage0_pool_first",
    "late_concat_stage2",
    "weighted_last2",
    "weighted_last3",
    "normpres_last2",
    "normpres_last3",
    "dynamic_last3",
    "dynamic_last3_scale_token",
)

REID_SELECTOR_SPECS: dict[str, SelectorSpec] = {
    "metric_feature": SelectorSpec(
        "metric_feature",
        (
            "auto",
            "global",
            "coarse_concat",
            "raw_mean",
            "raw_concat",
            "concat_bn",
            "dse_weighted",
            "dse_mix",
        ),
    ),
    "inference_feature": SelectorSpec(
        "inference_feature",
        (
            "concat_bn",
            "norm_concat_bn",
            "global",
            "raw_mean",
            "raw_concat",
            "visibility_weighted_parts",
            "evidence_sinkhorn",
            "dse_weighted",
            "dse_mix",
        ),
    ),
    "feature_fusion": SelectorSpec(
        "feature_fusion",
        (*CSL_FEATURE_FUSION_MODES, "dpt_fpn"),
    ),
    "pyramid_resize_mode": SelectorSpec(
        "pyramid_resize_mode",
        ("bilinear", "pool_nearest", "pool_bilinear"),
    ),
    "spatial_conv_mode": SelectorSpec(
        "spatial_conv_mode",
        ("standard", "depthwise_separable", "bottleneck_depthwise"),
    ),
    "head_pool": SelectorSpec(
        "head_pool",
        ("avg", "gem", "dse", "gelu_gem", "relu_gem", "softplus_gem"),
    ),
    "part_pooling": SelectorSpec(
        "part_pooling",
        ("stripes", "overlap_stripes", "tokens", "semantic_parts"),
    ),
}


def selector_choices(name: str) -> tuple[str, ...]:
    """Return the choices for a registered categorical option."""
    try:
        return REID_SELECTOR_SPECS[name].choices
    except KeyError as exc:
        raise KeyError(
            f"Unknown ReID selector {name!r}; expected one of "
            f"{sorted(REID_SELECTOR_SPECS)}"
        ) from exc


def normalize_selector(name: str, value: str) -> str:
    """Normalize one registered categorical option."""
    try:
        spec = REID_SELECTOR_SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown ReID selector {name!r}; expected one of "
            f"{sorted(REID_SELECTOR_SPECS)}"
        ) from exc
    return spec.normalize(value)


__all__ = [
    "CSL_FEATURE_FUSION_MODES",
    "REID_SELECTOR_SPECS",
    "SelectorSpec",
    "normalize_selector",
    "selector_choices",
]
