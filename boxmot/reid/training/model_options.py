"""Declarative projection from trainer settings to ReID model kwargs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OptionTransform(str, Enum):
    """Small, serializable transforms for derived model switches."""

    IDENTITY = "identity"
    POSITIVE = "positive"

    def __str__(self) -> str:
        """Return the string value, matching ``StrEnum`` semantics."""
        return self.value


@dataclass(frozen=True)
class ModelOption:
    """Map one flat training option to one model constructor argument."""

    kwarg: str
    source: str | None = None
    transform: OptionTransform = OptionTransform.IDENTITY

    def resolve(self, options: object) -> Any:
        """Read and transform the effective option value."""
        value = getattr(options, self.source or self.kwarg)
        if self.transform == OptionTransform.POSITIVE:
            return float(value) > 0
        return value


@dataclass(frozen=True)
class ModelOptionGroup:
    """A cohesive model surface that can evolve as one ablation axis."""

    name: str
    options: tuple[ModelOption, ...]


def _options(*names: str) -> tuple[ModelOption, ...]:
    return tuple(ModelOption(name) for name in names)


REID_MODEL_OPTION_GROUPS: tuple[ModelOptionGroup, ...] = (
    ModelOptionGroup(
        "input_and_descriptor",
        _options(
            "img_size",
            "inference_feature",
            "feat_dim",
            "neck_dim",
            "drop_path_rate",
        ),
    ),
    ModelOptionGroup(
        "feature_fusion",
        _options(
            "feature_fusion",
            "pyramid_resize_mode",
            "spatial_conv_mode",
            "post_fusion_mixer",
            "post_fusion_mixer_reduction",
            "post_fusion_mixer_kernel",
            "post_fusion_mixer_gamma_init",
            "native_branch_widths",
            "fine_map_dim",
        ),
    ),
    ModelOptionGroup(
        "attention_backbone",
        _options(
            "attention_window_layout",
            "attention_bias",
            "interpolate_pretrained_attention_bias",
            "attention_mask",
            "attention_shift",
            "stage3_global",
            "stage3_downsample",
            "stage2_width_merge_after",
            "stage2_mlp_ratio",
            "stage3_mlp_ratio",
            "stage2_depth",
            "stage3_depth",
            "width_first_hierarchy",
        ),
    ),
    ModelOptionGroup(
        "identity_registers",
        _options(
            "identity_registers",
            "identity_register_count",
            "identity_register_dim",
            "identity_register_num_heads",
            "identity_register_dropout",
            "identity_register_gate_init",
        ),
    ),
    ModelOptionGroup(
        "reid_adapters",
        _options(
            "reid_adapter_stages",
            "reid_adapter_reduction",
            "reid_adapter_suppression_tau",
        ),
    ),
    ModelOptionGroup(
        "retrieval_head",
        (
            *_options(
                "compact_deployment_head",
                "head_pool",
                "head_parts",
                "head_type",
                "multiscale_channel_alpha",
                "body_slot_mode",
                "body_slot_alpha",
                "body_slot_visibility_floor",
                "part_pooling",
                "num_part_tokens",
                "decouple_patterns",
                "pattern_adapter_dim",
                "stripe_visibility",
                "drop_global_aux",
                "drop_global_aux_ratio",
                "evidence_num_roles",
                "scale_balanced_branches",
                "multilevel_suppression",
                "multilevel_suppression_ratio",
                "mcpt_mode",
                "mcpt_hidden_dim",
                "mcpt_max_displacement",
                "mcpt_start_epoch",
                "mcpt_ramp_end_epoch",
                "jpm",
                "jpm_num_groups",
                "jpm_shift",
                "jpm_token_dim",
                "jpm_num_heads",
                "jpm_mlp_ratio",
                "jpm_dropout",
            ),
            ModelOption("branch_metric", source="branch_aware_metric"),
        ),
    ),
    ModelOptionGroup(
        "privileged_anatomy",
        (
            *_options(
                "anatomical_auxiliary",
                "anatomical_token_dim",
                "anatomical_multiscale",
                "anatomical_target_type",
                "anatomical_accessory_query",
                "anatomical_deployment",
                "anatomical_deployment_dim",
                "anatomical_deployment_alpha",
            ),
            ModelOption(
                "anatomical_descriptor_distill",
                source="anatomical_descriptor_distill_weight",
                transform=OptionTransform.POSITIVE,
            ),
            ModelOption(
                "anatomical_branch_distill",
                source="anatomical_branch_distill_weight",
                transform=OptionTransform.POSITIVE,
            ),
        ),
    ),
    ModelOptionGroup(
        "branch_communication",
        _options(
            "hierarchical_branch_attention",
            "branch_attention_token_dim",
            "branch_attention_num_heads",
            "branch_attention_num_layers",
            "branch_attention_mlp_ratio",
            "branch_attention_dropout",
            "branch_set_attention",
            "branch_set_attention_token_dim",
            "branch_set_attention_num_heads",
            "branch_set_attention_num_layers",
            "branch_set_attention_mlp_ratio",
            "branch_set_attention_dropout",
            "multiscale_query_decoder",
            "query_decoder_dim",
            "query_decoder_num_heads",
            "query_decoder_num_layers",
            "query_decoder_mlp_ratio",
            "query_decoder_dropout",
            "hierarchical_late_interaction",
            "late_interaction_dim",
            "late_interaction_num_heads",
            "late_interaction_num_layers",
            "late_interaction_sinkhorn_iters",
            "late_interaction_null_tokens",
            "late_interaction_base_score_init",
        ),
    ),
    ModelOptionGroup(
        "training_only_outputs",
        (
            ModelOption(
                "return_auxiliary_features",
                source="return_auxiliary_features",
            ),
            ModelOption(
                "return_cross_scale_features",
                source="csmm_loss_weight",
                transform=OptionTransform.POSITIVE,
            ),
            ModelOption(
                "return_treeboost_features",
                source="treeboost_loss_weight",
                transform=OptionTransform.POSITIVE,
            ),
        ),
    ),
)


def build_reid_model_kwargs(options: object) -> dict[str, Any]:
    """Build model kwargs from the canonical grouped option registry."""
    kwargs: dict[str, Any] = {}
    for group in REID_MODEL_OPTION_GROUPS:
        for option in group.options:
            if option.kwarg in kwargs:
                raise RuntimeError(
                    f"Duplicate model kwarg {option.kwarg!r} in option registry"
                )
            kwargs[option.kwarg] = option.resolve(options)
    return kwargs


__all__ = [
    "ModelOption",
    "ModelOptionGroup",
    "OptionTransform",
    "REID_MODEL_OPTION_GROUPS",
    "build_reid_model_kwargs",
]
