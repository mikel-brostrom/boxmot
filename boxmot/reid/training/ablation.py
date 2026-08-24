"""Resolved, serializable component plans for hierarchical ReID ablations.

The command-line surface intentionally remains flat, but experiment treatments
are resolved here into named components. This gives validation, logging,
checkpoints, and reports one canonical view of what differs from the baseline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from boxmot.reid.backbones.head_registry import ReIDHeadSpec, get_reid_head_spec


class AddonCategory(StrEnum):
    """Independent axes used when reviewing an ablation table."""

    ARCHITECTURE = "architecture"
    HEAD = "head"
    AUGMENTATION = "augmentation"
    SUPERVISION = "supervision"
    OBJECTIVE = "objective"


class ActivationKind(StrEnum):
    """Supported declarative activation predicates."""

    TRUTHY = "truthy"
    POSITIVE = "positive"
    NONEMPTY = "nonempty"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"


@dataclass(frozen=True)
class Activation:
    """Describe when an add-on is active without executable predicates."""

    field: str
    kind: ActivationKind = ActivationKind.TRUTHY
    value: Any = None

    def matches(self, options: object | Mapping[str, Any]) -> bool:
        """Evaluate this activation against a trainer or a plain mapping."""
        current = _option(options, self.field)
        if self.kind == ActivationKind.TRUTHY:
            return bool(current)
        if self.kind == ActivationKind.POSITIVE:
            return current is not None and float(current) > 0
        if self.kind == ActivationKind.NONEMPTY:
            return bool(current)
        if self.kind == ActivationKind.EQUALS:
            return current == self.value
        if self.kind == ActivationKind.NOT_EQUALS:
            return current != self.value
        raise AssertionError(f"Unhandled activation kind: {self.kind}")


@dataclass(frozen=True)
class AddonSpec:
    """One optional experimental treatment and the settings that define it."""

    name: str
    category: AddonCategory
    activation: Activation
    settings: tuple[str, ...]
    description: str
    exclusive_group: str | None = None
    requires: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedAddon:
    """An enabled add-on with its effective hyperparameters."""

    spec: AddonSpec
    settings: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "name": self.spec.name,
            "category": self.spec.category.value,
            "description": self.spec.description,
            "settings": {
                key: _json_value(value) for key, value in self.settings.items()
            },
        }


@dataclass(frozen=True)
class CSLTinyViTAblationPlan:
    """Fully resolved head and optional treatments for one training run."""

    head: ReIDHeadSpec
    head_settings: Mapping[str, Any]
    addons: tuple[ResolvedAddon, ...]

    @property
    def active_names(self) -> tuple[str, ...]:
        """Names suitable for concise logs and ablation reports."""
        return (f"head.{self.head.name}", *(addon.spec.name for addon in self.addons))

    def by_category(self, category: AddonCategory) -> tuple[ResolvedAddon, ...]:
        """Return active treatments on one ablation axis."""
        return tuple(
            addon for addon in self.addons if addon.spec.category == category
        )

    def validate_dependencies(self) -> None:
        """Reject mutually exclusive or incomplete component selections."""
        active = {addon.spec.name for addon in self.addons}
        exclusive_groups: dict[str, list[str]] = defaultdict(list)
        for addon in self.addons:
            spec = addon.spec
            if spec.exclusive_group:
                exclusive_groups[spec.exclusive_group].append(spec.name)
            missing = sorted(set(spec.requires) - active)
            if missing:
                raise ValueError(
                    f"{spec.name} requires active add-on(s): {', '.join(missing)}"
                )
            conflicts = sorted(set(spec.conflicts_with) & active)
            if conflicts:
                raise ValueError(
                    f"{spec.name} conflicts with: {', '.join(conflicts)}"
                )
        for group, names in exclusive_groups.items():
            if len(names) > 1:
                raise ValueError(
                    f"Ablation group {group!r} accepts one treatment, got "
                    f"{', '.join(names)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical hparams/report representation."""
        grouped: dict[str, list[str]] = {
            category.value: [
                addon.spec.name for addon in self.by_category(category)
            ]
            for category in AddonCategory
        }
        return {
            "schema_version": 1,
            "head": {
                "name": self.head.name,
                "implementation": self.head.implementation.value,
                "specialist": self.head.specialist,
                "channel_control": self.head.channel_control,
                "settings": {
                    key: _json_value(value)
                    for key, value in self.head_settings.items()
                },
            },
            "active": list(self.active_names),
            "by_category": grouped,
            "addons": [addon.to_dict() for addon in self.addons],
        }


def _bool_addon(
    name: str,
    category: AddonCategory,
    field: str,
    settings: tuple[str, ...],
    description: str,
    **kwargs: Any,
) -> AddonSpec:
    return AddonSpec(
        name=name,
        category=category,
        activation=Activation(field),
        settings=settings,
        description=description,
        **kwargs,
    )


CSL_TINYVIT_ADDONS: tuple[AddonSpec, ...] = (
    _bool_addon(
        "architecture.width_first_hierarchy",
        AddonCategory.ARCHITECTURE,
        "width_first_hierarchy",
        (
            "width_first_hierarchy",
            "stage2_depth",
            "stage3_depth",
        ),
        "Merge width before the deeper hierarchy with an explicit depth allocation.",
    ),
    _bool_addon(
        "architecture.identity_registers",
        AddonCategory.ARCHITECTURE,
        "identity_registers",
        (
            "identity_registers",
            "identity_register_count",
            "identity_register_dim",
            "identity_register_num_heads",
            "identity_register_dropout",
            "identity_register_gate_init",
        ),
        "Communicate identity evidence through bottleneck registers.",
    ),
    _bool_addon(
        "architecture.stage3_downsample",
        AddonCategory.ARCHITECTURE,
        "stage3_downsample",
        ("stage3_downsample",),
        "Downsample before the final attention stage.",
    ),
    AddonSpec(
        name="architecture.stage2_width_merge",
        category=AddonCategory.ARCHITECTURE,
        activation=Activation(
            "stage2_width_merge_after", ActivationKind.POSITIVE
        ),
        settings=("stage2_width_merge_after",),
        description="Merge Stage-2 width after a selected block.",
    ),
    _bool_addon(
        "architecture.native_branch_widths",
        AddonCategory.ARCHITECTURE,
        "native_branch_widths",
        ("native_branch_widths",),
        "Keep global, coarse, and fine branches at native widths.",
    ),
    AddonSpec(
        name="architecture.fine_map_bottleneck",
        category=AddonCategory.ARCHITECTURE,
        activation=Activation("fine_map_dim", ActivationKind.POSITIVE),
        settings=("fine_map_dim",),
        description="Bottleneck the high-resolution fine feature map.",
    ),
    AddonSpec(
        name="architecture.post_fusion_mixer",
        category=AddonCategory.ARCHITECTURE,
        activation=Activation(
            "post_fusion_mixer", ActivationKind.NOT_EQUALS, "none"
        ),
        settings=(
            "post_fusion_mixer",
            "post_fusion_mixer_reduction",
            "post_fusion_mixer_kernel",
            "post_fusion_mixer_gamma_init",
        ),
        description="Apply a residual local mixer after feature fusion.",
    ),
    AddonSpec(
        name="architecture.reid_adapters",
        category=AddonCategory.ARCHITECTURE,
        activation=Activation("reid_adapter_stages", ActivationKind.NONEMPTY),
        settings=(
            "reid_adapter_stages",
            "reid_adapter_reduction",
            "reid_adapter_suppression_tau",
        ),
        description="Insert zero-gated residual ReID adapters.",
    ),
    AddonSpec(
        name="architecture.mcpt",
        category=AddonCategory.ARCHITECTURE,
        activation=Activation("mcpt_mode", ActivationKind.NOT_EQUALS, "none"),
        settings=(
            "mcpt_mode",
            "mcpt_hidden_dim",
            "mcpt_max_displacement",
            "mcpt_smoothness_weight",
            "mcpt_identity_weight",
            "mcpt_identity_decay_epoch",
            "mcpt_lr_multiplier",
            "mcpt_start_epoch",
            "mcpt_ramp_end_epoch",
            "mcpt_disabled_eval",
        ),
        description=(
            "Warp ordered local maps into one monotonic RGB-conditioned "
            "canonical vertical coordinate."
        ),
        conflicts_with=(
            "head.compact_deployment",
            "head.hierarchical_branch_attention",
            "head.branch_set_attention",
            "head.multiscale_query_decoder",
            "head.hierarchical_late_interaction",
        ),
    ),
    _bool_addon(
        "architecture.jpm",
        AddonCategory.ARCHITECTURE,
        "jpm",
        (
            "jpm",
            "jpm_num_groups",
            "jpm_shift",
            "jpm_token_dim",
            "jpm_num_heads",
            "jpm_mlp_ratio",
            "jpm_dropout",
            "jpm_id_loss_weight",
            "jpm_metric_loss_weight",
        ),
        "Train shuffled patch groups through one shared local transformer and discard the branch for deployment.",
        conflicts_with=(
            "architecture.mcpt",
            "head.compact_deployment",
            "head.hierarchical_branch_attention",
            "head.branch_set_attention",
            "head.multiscale_query_decoder",
            "head.hierarchical_late_interaction",
            "supervision.anatomical_teacher",
            "objective.part_relation",
        ),
    ),
    _bool_addon(
        "head.scale_balanced_branches",
        AddonCategory.HEAD,
        "scale_balanced_branches",
        ("scale_balanced_branches", "branch_loss_agg"),
        "Allocate equal descriptor and ID-loss power to each spatial scale.",
    ),
    _bool_addon(
        "supervision.multilevel_suppression",
        AddonCategory.SUPERVISION,
        "multilevel_suppression",
        (
            "multilevel_suppression",
            "multilevel_suppression_ratio",
            "multilevel_suppression_loss_weight",
            "multilevel_suppression_start_epoch",
            "multilevel_suppression_ramp_end_epoch",
            "multilevel_suppression_decay_start_epoch",
            "multilevel_suppression_decay_end_epoch",
        ),
        "Classify coarse and fine features after suppressing evidence selected by the preceding scale.",
    ),
    _bool_addon(
        "head.pattern_decoupling",
        AddonCategory.HEAD,
        "decouple_patterns",
        ("decouple_patterns", "pattern_adapter_dim"),
        "Use separate residual adapters for global and local patterns.",
    ),
    _bool_addon(
        "head.stripe_visibility",
        AddonCategory.HEAD,
        "stripe_visibility",
        ("stripe_visibility",),
        "Predict reliability for fixed stripe descriptors.",
    ),
    _bool_addon(
        "head.drop_global_aux",
        AddonCategory.HEAD,
        "drop_global_aux",
        ("drop_global_aux", "drop_global_aux_ratio"),
        "Regularize training by dropping global auxiliary features.",
    ),
    _bool_addon(
        "head.compact_deployment",
        AddonCategory.HEAD,
        "compact_deployment_head",
        (
            "compact_deployment_head",
            "compact_metric_loss_weight",
            "compact_cosine_distill_weight",
            "compact_pairwise_distill_weight",
        ),
        "Distill multi-branch evidence into one compact descriptor.",
        conflicts_with=(
            "head.hierarchical_branch_attention",
            "head.branch_set_attention",
            "head.multiscale_query_decoder",
            "head.hierarchical_late_interaction",
        ),
    ),
    _bool_addon(
        "head.hierarchical_branch_attention",
        AddonCategory.HEAD,
        "hierarchical_branch_attention",
        (
            "hierarchical_branch_attention",
            "branch_attention_token_dim",
            "branch_attention_num_heads",
            "branch_attention_num_layers",
            "branch_attention_mlp_ratio",
            "branch_attention_dropout",
        ),
        "Communicate over the global-to-coarse-to-fine branch tree.",
        exclusive_group="branch_communication",
    ),
    _bool_addon(
        "head.branch_set_attention",
        AddonCategory.HEAD,
        "branch_set_attention",
        (
            "branch_set_attention",
            "branch_set_attention_token_dim",
            "branch_set_attention_num_heads",
            "branch_set_attention_num_layers",
            "branch_set_attention_mlp_ratio",
            "branch_set_attention_dropout",
        ),
        "Communicate over the unmasked set of seven pooled branches.",
        exclusive_group="branch_communication",
    ),
    _bool_addon(
        "head.multiscale_query_decoder",
        AddonCategory.HEAD,
        "multiscale_query_decoder",
        (
            "multiscale_query_decoder",
            "query_decoder_dim",
            "query_decoder_num_heads",
            "query_decoder_num_layers",
            "query_decoder_mlp_ratio",
            "query_decoder_dropout",
        ),
        "Decode pooled branch queries against multi-scale spatial memory.",
        exclusive_group="branch_communication",
    ),
    _bool_addon(
        "head.hierarchical_late_interaction",
        AddonCategory.HEAD,
        "hierarchical_late_interaction",
        (
            "hierarchical_late_interaction",
            "late_interaction_dim",
            "late_interaction_num_heads",
            "late_interaction_num_layers",
            "late_interaction_sinkhorn_iters",
            "late_interaction_null_tokens",
            "late_interaction_base_score_init",
        ),
        "Learn a hierarchy-aware training matcher and retrieval reranker.",
        exclusive_group="branch_communication",
    ),
    _bool_addon(
        "augmentation.gaussian_blur",
        AddonCategory.AUGMENTATION,
        "gaussian_blur",
        ("gaussian_blur",),
        "Apply random Gaussian blur in the image transform.",
    ),
    AddonSpec(
        name="augmentation.random_grayscale",
        category=AddonCategory.AUGMENTATION,
        activation=Activation(
            "random_grayscale", ActivationKind.POSITIVE
        ),
        settings=("random_grayscale",),
        description="Randomly remove image color.",
    ),
    AddonSpec(
        name="augmentation.random_erasing",
        category=AddonCategory.AUGMENTATION,
        activation=Activation("random_erasing", ActivationKind.POSITIVE),
        settings=("random_erasing",),
        description="Erase random image regions after normalization.",
    ),
    _bool_addon(
        "augmentation.random_patch",
        AddonCategory.AUGMENTATION,
        "random_patch",
        ("random_patch",),
        "Paste sampled image patches as appearance perturbations.",
    ),
    AddonSpec(
        name="augmentation.random_crop",
        category=AddonCategory.AUGMENTATION,
        activation=Activation(
            "random_crop_scale", ActivationKind.NOT_EQUALS, 1.0
        ),
        settings=("random_crop_scale",),
        description="Crop from an enlarged canvas before resizing.",
    ),
    _bool_addon(
        "augmentation.color_jitter",
        AddonCategory.AUGMENTATION,
        "color_jitter",
        ("color_jitter",),
        "Apply the standard color-jitter transform.",
    ),
    _bool_addon(
        "augmentation.color_augmentation",
        AddonCategory.AUGMENTATION,
        "color_augmentation",
        ("color_augmentation",),
        "Apply ReID-specific color perturbations.",
    ),
    _bool_addon(
        "augmentation.background_mosaic",
        AddonCategory.AUGMENTATION,
        "background_mosaic",
        (
            "background_mosaic",
            "background_mosaic_mask_dir",
            "background_mosaic_probability",
            "background_mosaic_start_epoch",
            "background_mosaic_ramp_end_epoch",
            "background_mosaic_min_foreground_ratio",
            "background_mosaic_max_foreground_ratio",
            "background_mosaic_feather",
            "background_mosaic_dilation",
            "background_mosaic_occluder_probability",
            "background_mosaic_occluder_min_area",
            "background_mosaic_occluder_max_area",
        ),
        "Replace context while preserving the anchor identity foreground.",
    ),
    _bool_addon(
        "augmentation.same_id_part_mosaic",
        AddonCategory.AUGMENTATION,
        "same_id_part_mosaic",
        (
            "same_id_part_mosaic",
            "same_id_part_mosaic_probability",
            "same_id_part_mosaic_max_regions",
            "same_id_part_mosaic_min_area",
            "same_id_part_mosaic_max_area",
            "same_id_part_mosaic_boundary_jitter",
            "same_id_part_mosaic_cross_camera_rate",
            "same_id_part_mosaic_min_unaltered",
        ),
        "Exchange image regions with cross-camera samples of the same ID.",
    ),
    _bool_addon(
        "augmentation.pose_aligned_view_mosaic",
        AddonCategory.AUGMENTATION,
        "pav_mosaic",
        (
            "pav_mosaic",
            "pav_metadata_dir",
            "pav_mosaic_probability",
            "pav_mosaic_max_parts",
            "pav_mosaic_max_foreground_replacement",
            "pav_mosaic_cross_camera_rate",
            "pav_mosaic_different_pose_rate",
            "pav_mosaic_min_keypoint_confidence",
            "pav_mosaic_min_unaltered",
            "pav_mosaic_warmup_epochs",
            "pav_mosaic_decay_start_epoch",
            "pav_mosaic_final_probability_scale",
        ),
        "Replace pose-aligned body parts using same-ID donor views.",
    ),
    _bool_addon(
        "supervision.anatomical_teacher",
        AddonCategory.SUPERVISION,
        "anatomical_auxiliary",
        (
            "anatomical_auxiliary",
            "anatomical_target_type",
            "anatomical_metadata_dir",
            "anatomical_person_mask_dir",
            "anatomical_token_dim",
            "anatomical_multiscale",
            "anatomical_accessory_query",
            "anatomical_min_keypoint_confidence",
            "anatomical_student_start_epoch",
            "anatomical_student_ramp_end_epoch",
            "anatomical_fine_start_epoch",
            "anatomical_fine_ramp_end_epoch",
            "anatomical_decay_start_epoch",
            "anatomical_decay_end_epoch",
            "anatomical_distill_weight",
            "anatomical_attention_weight",
            "anatomical_foreground_weight",
            "anatomical_semantic_part_weight",
            "anatomical_visibility_weight",
            "anatomical_contrastive_weight",
            "anatomical_descriptor_distill_weight",
            "anatomical_branch_distill_weight",
            "anatomical_branch_global_coefficient",
            "anatomical_branch_coarse_coefficient",
            "anatomical_branch_fine_coefficient",
            "anatomical_pose_teacher_weight",
            "anatomical_query_distill_weight",
            "anatomical_query_relational_distill_weight",
            "anatomical_query_diversity_weight",
            "anatomical_query_diversity_margin",
            "anatomical_part_triplet_weight",
            "clean_student_consistency_weight",
            "anatomical_local_scale_weight",
            "anatomical_fine_scale_weight",
            "anatomical_cross_scale_weight",
            "anatomical_pose_only_reliability",
            "anatomical_min_effective_coverage",
            "anatomical_query_start_epoch",
            "anatomical_query_ramp_end_epoch",
            "anatomical_temperature",
            "anatomical_teacher_momentum",
        ),
        "Use pose and optional person masks as privileged training targets.",
    ),
    _bool_addon(
        "supervision.anatomical_deployment",
        AddonCategory.SUPERVISION,
        "anatomical_deployment",
        (
            "anatomical_deployment",
            "anatomical_deployment_dim",
            "anatomical_deployment_alpha",
            "anatomical_deployment_id_weight",
            "anatomical_deployment_metric_weight",
        ),
        "Retain pose-supervised RGB semantic tokens in the descriptor.",
        requires=("supervision.anatomical_teacher",),
    ),
    AddonSpec(
        name="objective.adasp",
        category=AddonCategory.OBJECTIVE,
        activation=Activation("adasp_loss_weight", ActivationKind.POSITIVE),
        settings=("adasp_loss_weight", "adasp_temperature", "adasp_scale"),
        description="Apply adaptive sparse pairwise learning to the full descriptor.",
    ),
    AddonSpec(
        name="objective.part_relation",
        category=AddonCategory.OBJECTIVE,
        activation=Activation("part_relation_weight", ActivationKind.POSITIVE),
        settings=(
            "coarse_branch_ce_weight",
            "fine_branch_ce_weight",
            "part_relation_weight",
            "part_to_global_weight",
            "part_relation_teacher_momentum",
            "part_relation_temperature",
        ),
        description="Preserve EMA-teacher cross-ID fine-part neighborhoods and distill them globally.",
    ),
    AddonSpec(
        name="objective.csmm",
        category=AddonCategory.OBJECTIVE,
        activation=Activation("csmm_loss_weight", ActivationKind.POSITIVE),
        settings=(
            "csmm_loss_weight",
            "csmm_margin",
            "csmm_temperature",
            "csmm_topk_negatives",
            "csmm_start_epoch",
            "csmm_ramp_end_epoch",
        ),
        description="Apply cross-scale majority-margin supervision.",
    ),
    AddonSpec(
        name="objective.treeboost_ap",
        category=AddonCategory.OBJECTIVE,
        activation=Activation(
            "treeboost_loss_weight", ActivationKind.POSITIVE
        ),
        settings=(
            "treeboost_loss_weight",
            "treeboost_coarse_coefficient",
            "treeboost_fine_coefficient",
            "treeboost_node_coefficient",
            "treeboost_regression_coefficient",
            "treeboost_difficulty_floor",
            "treeboost_regression_tolerance",
            "treeboost_temperature",
            "treeboost_start_epoch",
            "treeboost_ramp_end_epoch",
        ),
        description="Optimize hierarchy-aware differentiable AP surrogates.",
    ),
    AddonSpec(
        name="objective.identity_register_diversity",
        category=AddonCategory.OBJECTIVE,
        activation=Activation(
            "identity_register_diversity_weight", ActivationKind.POSITIVE
        ),
        settings=(
            "identity_register_diversity_weight",
            "identity_register_diversity_margin",
        ),
        description="Prevent identity registers from collapsing to one role.",
        requires=("architecture.identity_registers",),
    ),
    AddonSpec(
        name="objective.pav_consistency",
        category=AddonCategory.OBJECTIVE,
        activation=Activation("pav_consistency_weight", ActivationKind.POSITIVE),
        settings=("pav_consistency_weight",),
        description="Align descriptors from clean and pose-mosaic views.",
        requires=("augmentation.pose_aligned_view_mosaic",),
    ),
    AddonSpec(
        name="objective.anatomical_query_relational_distill",
        category=AddonCategory.OBJECTIVE,
        activation=Activation(
            "anatomical_query_relational_distill_weight",
            ActivationKind.POSITIVE,
        ),
        settings=("anatomical_query_relational_distill_weight",),
        description=(
            "Match per-part teacher/student identity-relation geometry."
        ),
        requires=("supervision.anatomical_teacher",),
    ),
    AddonSpec(
        name="objective.clean_student_consistency",
        category=AddonCategory.OBJECTIVE,
        activation=Activation(
            "clean_student_consistency_weight",
            ActivationKind.POSITIVE,
        ),
        settings=("clean_student_consistency_weight",),
        description=(
            "Distil clean masked queries and descriptors into augmented RGB."
        ),
        requires=("supervision.anatomical_teacher",),
    ),
)


def validate_addon_registry(
    addons: tuple[AddonSpec, ...] = CSL_TINYVIT_ADDONS,
) -> None:
    """Validate declarative add-on metadata independently of a run."""
    names = [addon.name for addon in addons]
    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError(
            "Duplicate CSL-TinyViT add-on names: "
            + ", ".join(duplicate_names)
        )

    known_names = set(names)
    for addon in addons:
        duplicate_settings = sorted(
            setting
            for setting in set(addon.settings)
            if addon.settings.count(setting) > 1
        )
        if duplicate_settings:
            raise ValueError(
                f"{addon.name} repeats setting(s): "
                + ", ".join(duplicate_settings)
            )
        unknown_dependencies = sorted(
            (set(addon.requires) | set(addon.conflicts_with))
            - known_names
        )
        if unknown_dependencies:
            raise ValueError(
                f"{addon.name} references unknown add-on(s): "
                + ", ".join(unknown_dependencies)
            )
        if addon.name in addon.requires or addon.name in addon.conflicts_with:
            raise ValueError(
                f"{addon.name} cannot depend on or conflict with itself"
            )


validate_addon_registry()


_HEAD_SETTINGS = (
    "head_type",
    "head_pool",
    "head_parts",
    "part_pooling",
    "num_part_tokens",
    "multiscale_channel_alpha",
    "body_slot_mode",
    "body_slot_alpha",
    "body_slot_visibility_floor",
    "metric_feature",
    "inference_feature",
)


def resolve_csl_tinyvit_ablation(
    options: object | Mapping[str, Any],
) -> CSLTinyViTAblationPlan:
    """Resolve a supported hierarchical ReID model into a component plan."""
    model_name = str(_option(options, "model_name", ""))
    if model_name.startswith("csl_tinyvit"):
        family = "csl_tinyvit"
    elif model_name.startswith("mobilenetv4"):
        family = "mobilenetv4"
    else:
        raise ValueError(
            "Hierarchical ReID ablation plans require a CSL-TinyViT or "
            "MobileNetV4 model, got "
            f"{model_name!r}"
        )
    head = get_reid_head_spec(
        str(_option(options, "head_type", "standard")),
        family=family,
    )
    resolved = tuple(
        ResolvedAddon(
            spec=spec,
            settings={
                field: _option(options, field) for field in spec.settings
            },
        )
        for spec in CSL_TINYVIT_ADDONS
        if spec.activation.matches(options)
    )
    plan = CSLTinyViTAblationPlan(
        head=head,
        head_settings={
            field: _option(options, field) for field in _HEAD_SETTINGS
        },
        addons=resolved,
    )
    plan.validate_dependencies()
    return plan


def _option(
    options: object | Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


__all__ = [
    "ActivationKind",
    "AddonCategory",
    "AddonSpec",
    "CSL_TINYVIT_ADDONS",
    "CSLTinyViTAblationPlan",
    "ResolvedAddon",
    "resolve_csl_tinyvit_ablation",
    "validate_addon_registry",
]
