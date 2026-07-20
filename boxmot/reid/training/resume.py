"""Canonical compatibility contracts for resumable ReID training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

RESUME_CONTRACT_VERSION = 1
_MISSING = object()

# Every field that changes learned weights, optimizer evolution, sampling, or
# validation model selection belongs here. Runtime placement and output paths
# are deliberately absent so a run can move machines and directories safely.
RESUME_CONTRACT_FIELDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "identity": (
        ("model", ("model_name", "model")),
        ("dataset", ("dataset_name", "dataset")),
        ("pretrained", ("pretrained",)),
    ),
    "data": (
        ("preprocess", ("preprocess",)),
        ("img_size", ("img_size", "imgsz")),
        ("p", ("p", "p_ids")),
        ("k", ("k", "k_instances")),
        ("source_balance", ("source_balance",)),
        ("data_specs", ("data_specs",)),
    ),
    "model": (
        ("metric_feature", ("metric_feature",)),
        ("inference_feature", ("inference_feature",)),
        ("feature_fusion", ("feature_fusion",)),
        ("pyramid_resize_mode", ("pyramid_resize_mode",)),
        ("spatial_conv_mode", ("spatial_conv_mode",)),
        ("post_fusion_mixer", ("post_fusion_mixer",)),
        ("post_fusion_mixer_reduction", ("post_fusion_mixer_reduction",)),
        ("post_fusion_mixer_kernel", ("post_fusion_mixer_kernel",)),
        ("post_fusion_mixer_gamma_init", ("post_fusion_mixer_gamma_init",)),
        ("feat_dim", ("feat_dim",)),
        ("neck_dim", ("neck_dim",)),
        ("drop_path_rate", ("drop_path_rate",)),
        ("attention_window_layout", ("attention_window_layout",)),
        ("attention_bias", ("attention_bias",)),
        ("interpolate_pretrained_attention_bias", ("interpolate_pretrained_attention_bias",)),
        ("attention_mask", ("attention_mask",)),
        ("attention_shift", ("attention_shift",)),
        ("stage3_global", ("stage3_global",)),
        ("stage3_downsample", ("stage3_downsample",)),
        ("stage2_width_merge_after", ("stage2_width_merge_after",)),
        ("stage3_mlp_ratio", ("stage3_mlp_ratio",)),
        ("stage3_depth", ("stage3_depth",)),
        ("native_branch_widths", ("native_branch_widths",)),
        ("compact_deployment_head", ("compact_deployment_head",)),
        ("reid_adapter_stages", ("reid_adapter_stages",)),
        ("reid_adapter_reduction", ("reid_adapter_reduction",)),
        ("head_pool", ("head_pool",)),
        ("head_parts", ("head_parts",)),
        ("head_type", ("head_type",)),
        ("part_pooling", ("part_pooling",)),
        ("num_part_tokens", ("num_part_tokens",)),
        ("evidence_num_roles", ("evidence_num_roles",)),
        ("decouple_patterns", ("decouple_patterns",)),
        ("pattern_adapter_dim", ("pattern_adapter_dim",)),
        ("stripe_visibility", ("stripe_visibility",)),
        ("drop_global_aux", ("drop_global_aux",)),
        ("drop_global_aux_ratio", ("drop_global_aux_ratio",)),
        ("branch_aware_metric", ("branch_aware_metric",)),
        ("branch_metric_part_weight", ("branch_metric_part_weight",)),
        ("branch_loss_agg", ("branch_loss_agg",)),
        ("scale_balanced_branches", ("scale_balanced_branches",)),
        ("head_warmup_epochs", ("head_warmup_epochs",)),
        ("head_warmup_lr_mult", ("head_warmup_lr_mult",)),
    ),
    "loss": (
        ("loss_type", ("loss_type", "loss")),
        ("classifier_loss", ("classifier_loss",)),
        ("margin", ("margin",)),
        ("triplet_soft_margin", ("triplet_soft_margin", "soft_margin_triplet")),
        ("label_smooth", ("label_smooth",)),
        ("arcface_scale", ("arcface_scale",)),
        ("arcface_margin", ("arcface_margin",)),
        ("cosface_scale", ("cosface_scale",)),
        ("cosface_margin", ("cosface_margin",)),
        ("center_loss_weight", ("center_loss_weight",)),
        ("id_loss_weight", ("id_loss_weight",)),
        ("metric_loss_weight", ("metric_loss_weight",)),
        ("compact_metric_loss_weight", ("compact_metric_loss_weight",)),
        ("compact_cosine_distill_weight", ("compact_cosine_distill_weight",)),
        ("compact_pairwise_distill_weight", ("compact_pairwise_distill_weight",)),
        ("early_id_loss_weight", ("early_id_loss_weight",)),
        ("early_id_loss_epochs", ("early_id_loss_epochs",)),
        ("center_loss_ramp_start_epoch", ("center_loss_ramp_start_epoch",)),
        ("center_loss_ramp_end_epoch", ("center_loss_ramp_end_epoch",)),
        ("aux_ce_weight", ("aux_ce_weight",)),
        ("aux_ce_drop_epoch", ("aux_ce_drop_epoch",)),
        ("evidence_alignment_loss_weight", ("evidence_alignment_loss_weight",)),
        ("evidence_alignment_margin", ("evidence_alignment_margin",)),
        ("evidence_sinkhorn_iters", ("evidence_sinkhorn_iters",)),
        ("evidence_sinkhorn_temperature", ("evidence_sinkhorn_temperature",)),
        ("evidence_rerank_topk", ("evidence_rerank_topk",)),
        ("evidence_null_loss_weight", ("evidence_null_loss_weight",)),
        ("evidence_diversity_loss_weight", ("evidence_diversity_loss_weight",)),
    ),
    "optimization": (
        ("training_recipe", ("training_recipe",)),
        ("optimizer", ("optimizer", "optimizer_name")),
        ("lr", ("lr",)),
        ("weight_decay", ("weight_decay",)),
        ("layer_decay", ("layer_decay",)),
        ("grad_clip", ("grad_clip",)),
        ("warmup_epochs", ("warmup_epochs",)),
        ("eta_min", ("eta_min",)),
        ("ema_decay", ("ema_decay",)),
        ("vit_lr_profile", ("vit_lr_profile",)),
        ("backbone_freeze_epochs", ("backbone_freeze_epochs",)),
        ("gradual_unfreeze", ("gradual_unfreeze",)),
        ("gradual_unfreeze_head_epochs", ("gradual_unfreeze_head_epochs",)),
        ("gradual_unfreeze_stage_epochs", ("gradual_unfreeze_stage_epochs",)),
        ("gradual_unfreeze_backbone_lr_mult", ("gradual_unfreeze_backbone_lr_mult",)),
        ("gradual_unfreeze_backbone_lr_epochs", ("gradual_unfreeze_backbone_lr_epochs",)),
    ),
    "augmentation": (
        ("gaussian_blur", ("gaussian_blur",)),
        ("random_grayscale", ("random_grayscale",)),
        ("color_jitter", ("color_jitter",)),
        ("random_erasing", ("random_erasing",)),
        ("random_patch", ("random_patch",)),
        ("random_crop_scale", ("random_crop_scale",)),
        ("color_augmentation", ("color_augmentation",)),
    ),
    "evaluation": (
        ("eval_interval", ("eval_interval",)),
        ("eval_datasets", ("eval_datasets",)),
        ("flip_tta", ("flip_tta",)),
    ),
    "reproducibility": (
        ("seed", ("seed",)),
        ("deterministic", ("deterministic",)),
    ),
}


def _mapping_value(values: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in values:
            return values[alias]
    return _MISSING


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _normalize_data_specs(value: Any) -> list[dict[str, Any]]:
    """Keep dataset identity/sampling fields while allowing roots to move."""
    normalized = []
    for raw in value or ():
        if not isinstance(raw, Mapping):
            continue
        normalized.append(
            {
                str(key): _json_value(item)
                for key, item in sorted(raw.items())
                if key not in {"root", "data_dir", "path"}
            }
        )
    return normalized


def build_resume_contract(values: Mapping[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Build a stable compatibility contract from trainer args or flat hparams."""
    contract: dict[str, Any] = {"version": RESUME_CONTRACT_VERSION}
    for group_name, fields in RESUME_CONTRACT_FIELDS.items():
        group: dict[str, Any] = {}
        for field_name, aliases in fields:
            value = _mapping_value(values, aliases)
            if value is _MISSING:
                if partial:
                    continue
                raise KeyError(f"Missing resume-contract value: {aliases[0]}")
            if field_name == "data_specs":
                value = _normalize_data_specs(value)
            group[field_name] = _json_value(value)
        if group or not partial:
            contract[group_name] = group
    return contract


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for a resume contract."""
    payload = json.dumps(_json_value(contract), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_fingerprint(contract: Mapping[str, Any], target_epochs: int) -> str:
    """Fingerprint a complete ablation specification, including its target."""
    return contract_fingerprint({"contract": contract, "target_epochs": int(target_epochs)})


def contract_differences(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    compare_common_only: bool = False,
) -> list[str]:
    """Describe value differences between saved and requested contracts."""

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {prefix: value}
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten(item, path))
        return flattened

    left = flatten(_json_value(expected))
    right = flatten(_json_value(actual))
    keys = left.keys() & right.keys() if compare_common_only else left.keys() | right.keys()
    differences = []
    for key in sorted(keys):
        saved_value = left.get(key, _MISSING)
        requested_value = right.get(key, _MISSING)
        numerically_equal = (
            isinstance(saved_value, Real)
            and not isinstance(saved_value, bool)
            and isinstance(requested_value, Real)
            and not isinstance(requested_value, bool)
            and math.isclose(float(saved_value), float(requested_value), rel_tol=1e-7, abs_tol=1e-12)
        )
        if saved_value != requested_value and not numerically_equal:
            differences.append(
                f"{key}: saved={left.get(key, '<missing>')!r}, "
                f"requested={right.get(key, '<missing>')!r}"
            )
    return differences
