# model_registry.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from boxmot.reid.backbones import BACKBONE_REGISTRY, get_backbone_spec, registered_backbone_names
from boxmot.reid.core.artifacts import MODEL_KWARGS_SCHEMA_VERSION, read_artifact_metadata
from boxmot.reid.core.catalog import NR_CLASSES_DICT, TRAINED_URLS
from boxmot.utils import logger as LOGGER

MODEL_TYPES_BY_SPECIFICITY = tuple(sorted(registered_backbone_names(), key=len, reverse=True))


def _identity(value: Any) -> Any:
    return value


def _int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def _int_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    values = tuple(int(item) for item in value)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) != 2:
        raise ValueError(f"Expected one or two integers, got {value!r}")
    return values


CHECKPOINT_MODEL_KWARG_CONVERTERS: Mapping[str, Callable[[Any], Any]] = {
    "img_size": _int_pair,
    "feat_dim": int,
    "neck_dim": int,
    "attention_reduction": int,
    "attention_groups": int,
    "attention_gamma": float,
    "suppression_tau": float,
    "dropblock_prob": float,
    "dropblock_size": int,
    "timm_model_name": _identity,
    "use_timm_head": bool,
    "timm_head_mode": _identity,
    "mobilenetv4_last_stride": int,
    "mobilenetv4_neck_mode": _identity,
    "post_fusion_mixer": _identity,
    "post_fusion_mixer_reduction": int,
    "post_fusion_mixer_kernel": _int_pair,
    "post_fusion_mixer_gamma_init": float,
    "head_pool": _identity,
    "head_parts": _int_tuple,
    "head_type": _identity,
    "multiscale_channel_alpha": float,
    "body_slot_mode": _identity,
    "body_slot_alpha": float,
    "body_slot_visibility_floor": float,
    "part_pooling": _identity,
    "num_part_tokens": int,
    "evidence_num_roles": int,
    "decouple_patterns": bool,
    "pattern_adapter_dim": int,
    "stripe_visibility": bool,
    "drop_global_aux": bool,
    "drop_global_aux_ratio": float,
    "scale_balanced_branches": bool,
    "inference_feature": _identity,
    "feature_fusion": _identity,
    "pyramid_resize_mode": _identity,
    "spatial_conv_mode": _identity,
    "drop_path_rate": float,
    "patch_stride": int,
    "dpt_out_indices": _int_tuple,
    "dpt_fpn_fusion": _identity,
    "dpt_fpn_target_index": int,
    "attention_window_layout": _identity,
    "attention_bias": _identity,
    "interpolate_pretrained_attention_bias": bool,
    "attention_mask": bool,
    "attention_shift": bool,
    "stage3_global": bool,
    "stage3_downsample": bool,
    "stage2_width_merge_after": int,
    "stage2_mlp_ratio": float,
    "stage3_mlp_ratio": float,
    "stage2_depth": int,
    "stage3_depth": int,
    "width_first_hierarchy": bool,
    "identity_registers": bool,
    "identity_register_count": int,
    "identity_register_dim": int,
    "identity_register_num_heads": int,
    "identity_register_dropout": float,
    "identity_register_gate_init": float,
    "native_branch_widths": bool,
    "fine_map_dim": int,
    "compact_deployment_head": bool,
    "anatomical_auxiliary": bool,
    "anatomical_token_dim": int,
    "anatomical_multiscale": bool,
    "anatomical_accessory_query": bool,
    "anatomical_target_type": _identity,
    "anatomical_deployment": bool,
    "anatomical_deployment_dim": int,
    "anatomical_deployment_alpha": float,
    "hierarchical_branch_attention": bool,
    "branch_attention_token_dim": int,
    "branch_attention_num_heads": int,
    "branch_attention_num_layers": int,
    "branch_attention_mlp_ratio": float,
    "branch_attention_dropout": float,
    "branch_set_attention": bool,
    "branch_set_attention_token_dim": int,
    "branch_set_attention_num_heads": int,
    "branch_set_attention_num_layers": int,
    "branch_set_attention_mlp_ratio": float,
    "branch_set_attention_dropout": float,
    "multiscale_query_decoder": bool,
    "query_decoder_dim": int,
    "query_decoder_num_heads": int,
    "query_decoder_num_layers": int,
    "query_decoder_mlp_ratio": float,
    "query_decoder_dropout": float,
    "hierarchical_late_interaction": bool,
    "late_interaction_dim": int,
    "late_interaction_num_heads": int,
    "late_interaction_num_layers": int,
    "late_interaction_sinkhorn_iters": int,
    "late_interaction_null_tokens": int,
    "late_interaction_base_score_init": float,
    "mcpt_mode": _identity,
    "mcpt_hidden_dim": int,
    "mcpt_max_displacement": float,
    "mcpt_start_epoch": int,
    "mcpt_ramp_end_epoch": int,
    "jpm": bool,
    "jpm_num_groups": int,
    "jpm_shift": int,
    "jpm_token_dim": int,
    "jpm_num_heads": int,
    "jpm_mlp_ratio": float,
    "jpm_dropout": float,
    "multilevel_suppression": bool,
    "multilevel_suppression_ratio": float,
    "branch_metric": bool,
    "return_auxiliary_features": bool,
    "return_cross_scale_features": bool,
    "return_treeboost_features": bool,
    "anatomical_descriptor_distill": bool,
    "anatomical_branch_distill": bool,
    "reid_adapter_stages": _int_tuple,
    "reid_adapter_reduction": int,
    "reid_adapter_suppression_tau": float,
}


@dataclass(frozen=True)
class StateDictLoadReport:
    """Auditable result of loading one checkpoint into a deployed model."""

    matched_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]
    allowed_missing_keys: tuple[str, ...]
    allowed_unexpected_keys: tuple[str, ...]
    allowed_mismatched_keys: tuple[str, ...]
    tensor_coverage: float
    numel_coverage: float


def _convert_checkpoint_model_kwargs(
    model_name: Any,
    raw_values: Mapping[str, Any],
    *,
    schema_version: Any = None,
) -> dict[str, Any]:
    """Validate and type one versioned model-construction contract."""
    if schema_version is not None:
        version = int(schema_version)
        if version != MODEL_KWARGS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported model_kwargs schema version {version}; "
                f"supported={MODEL_KWARGS_SCHEMA_VERSION}"
            )
        unknown = sorted(set(raw_values).difference(CHECKPOINT_MODEL_KWARG_CONVERTERS))
        if unknown:
            raise ValueError(f"Unknown model_kwargs in schema v{version}: {unknown}")
    values = {
        key: converter(raw_values[key])
        for key, converter in CHECKPOINT_MODEL_KWARG_CONVERTERS.items()
        if raw_values.get(key) is not None
    }
    return _filter_checkpoint_model_kwargs(model_name, values)


def _is_deployment_optional_state_key(model: Any, key: str) -> bool:
    """Return whether a state tensor is intentionally absent at inference."""
    module_parts = key.split(".")[:-1]
    if any(part == "classifier" or part.endswith("_classifier") for part in module_parts):
        return True
    if ".jpm." in f".{key}":
        return True
    if ".anatomical_auxiliary_pool." in f".{key}":
        head = getattr(model, "head", None)
        return not bool(getattr(head, "anatomical_deployment_enabled", False))
    return False

_CSL_TINYVIT_INCOMPATIBLE_CHECKPOINT_KWARGS = frozenset(
    {
        "attention_gamma",
        "attention_groups",
        "attention_reduction",
        "dropblock_prob",
        "dropblock_size",
        "mobilenetv4_last_stride",
        "mobilenetv4_neck_mode",
        "suppression_tau",
        "timm_head_mode",
        "timm_model_name",
        "use_timm_head",
    }
)


def _filter_checkpoint_model_kwargs(model_name: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Remove family-specific kwargs that the resolved backbone cannot consume."""
    if str(model_name or "").startswith("csl_tinyvit"):
        return {
            key: value
            for key, value in values.items()
            if key not in _CSL_TINYVIT_INCOMPATIBLE_CHECKPOINT_KWARGS
        }
    return values


class ReIDModelRegistry:
    """Encapsulates model registration and related utilities."""

    @staticmethod
    def _load_checkpoint(weight_path: str | Path, *, strict: bool = False) -> Any | None:
        try:
            return torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
        except Exception:
            if strict:
                raise
            return None

    @staticmethod
    def _checkpoint_dict(weight_path: str | Path) -> dict[str, Any] | None:
        checkpoint = ReIDModelRegistry._load_checkpoint(weight_path)
        return checkpoint if isinstance(checkpoint, dict) else None

    @staticmethod
    def show_downloadable_models():
        LOGGER.info("Available .pt ReID models for automatic download")
        LOGGER.info(list(TRAINED_URLS.keys()))

    @staticmethod
    def get_model_name(model):
        path = Path(model)
        metadata = read_artifact_metadata(path)
        if metadata.get("model_name"):
            return str(metadata["model_name"])
        if path.is_file():
            checkpoint = ReIDModelRegistry._checkpoint_dict(path)
            if checkpoint and checkpoint.get("model_name"):
                return checkpoint["model_name"]
        model_name = path.name.lower()
        for name in MODEL_TYPES_BY_SPECIFICITY:
            if name in model_name:
                return name
        return None

    @staticmethod
    def get_model_url(model):
        return TRAINED_URLS.get(Path(model).name, None)

    @staticmethod
    def get_checkpoint_preprocess(weight_path) -> str | None:
        """Return the preprocessing method stored in a checkpoint, or None."""
        metadata = read_artifact_metadata(weight_path)
        if metadata.get("preprocess"):
            return str(metadata["preprocess"])
        checkpoint = ReIDModelRegistry._checkpoint_dict(weight_path)
        return checkpoint.get("preprocess") if checkpoint else None

    @staticmethod
    def get_checkpoint_model_kwargs(weight_path) -> dict:
        """Return optional architecture kwargs stored in a checkpoint."""
        metadata = read_artifact_metadata(weight_path)
        artifact_kwargs = metadata.get("model_kwargs")
        if isinstance(artifact_kwargs, dict):
            try:
                return _convert_checkpoint_model_kwargs(
                    metadata.get("model_name"),
                    artifact_kwargs,
                    schema_version=metadata.get("model_kwargs_schema_version"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid model metadata for {weight_path}: {exc}") from exc

        checkpoint = ReIDModelRegistry._checkpoint_dict(weight_path)
        if not checkpoint:
            return {}

        try:
            checkpoint_kwargs = checkpoint.get("model_kwargs")
            if isinstance(checkpoint_kwargs, dict):
                return _convert_checkpoint_model_kwargs(
                    checkpoint.get("model_name"),
                    checkpoint_kwargs,
                    schema_version=checkpoint.get("model_kwargs_schema_version"),
                )
            values = {
                key: converter(checkpoint[key])
                for key, converter in CHECKPOINT_MODEL_KWARG_CONVERTERS.items()
                if checkpoint.get(key) is not None
            }
            model_metadata = checkpoint.get("model")
            reproduction_contract = (
                model_metadata.get("reproduction_contract")
                if isinstance(model_metadata, dict)
                else None
            )
            if isinstance(reproduction_contract, dict):
                contract_kwargs = reproduction_contract.get("model_kwargs")
                if not isinstance(contract_kwargs, dict):
                    architecture = reproduction_contract.get("architecture", {})
                    attention = architecture.get("attention", {})
                    dropblock = architecture.get("dropblock", {})
                    contract_kwargs = {
                        "img_size": architecture.get("img_size"),
                        "feat_dim": 512,
                        "attention_reduction": attention.get("reduction"),
                        "attention_groups": attention.get("groups"),
                        "attention_gamma": attention.get("gamma_init"),
                        "suppression_tau": architecture.get("suppression_tau"),
                        "dropblock_prob": dropblock.get("probability"),
                        "dropblock_size": dropblock.get("block_size"),
                    }
                values.update(
                    {
                        key: converter(contract_kwargs[key])
                        for key, converter in CHECKPOINT_MODEL_KWARG_CONVERTERS.items()
                        if contract_kwargs.get(key) is not None
                    }
                )
            state_dict = checkpoint.get("state_dict", checkpoint)
            if isinstance(state_dict, dict):
                if "feat_dim" not in values and "head.bn_global.reduction.weight" in state_dict:
                    values["feat_dim"] = int(state_dict["head.bn_global.reduction.weight"].shape[0])
                if "neck_dim" not in values and "neck.0.weight" in state_dict:
                    values["neck_dim"] = int(state_dict["neck.0.weight"].shape[0])
            return _filter_checkpoint_model_kwargs(checkpoint.get("model_name"), values)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid model metadata for {weight_path}: {exc}") from exc

    @staticmethod
    def deployment_model_kwargs(model_name: str | None, values: Mapping[str, Any]) -> dict[str, Any]:
        """Remove training-only CSL modules from a deployed construction contract."""
        deployed = dict(values)
        if not str(model_name or "").startswith("csl_tinyvit"):
            return deployed

        deployed.update(
            {
                "jpm": False,
                "multilevel_suppression": False,
                "return_auxiliary_features": False,
                "return_cross_scale_features": False,
                "return_treeboost_features": False,
            }
        )
        if not bool(deployed.get("anatomical_deployment", False)):
            deployed.update(
                {
                    "anatomical_auxiliary": False,
                    "anatomical_multiscale": False,
                    "anatomical_accessory_query": False,
                    "anatomical_descriptor_distill": False,
                    "anatomical_branch_distill": False,
                }
            )
        return deployed

    @staticmethod
    def _match_checkpoint_state(model, state_dict: Mapping[str, Any]):
        """Normalize a checkpoint and partition matching, missing, and invalid keys."""
        model_dict = model.state_dict()
        matched: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        mismatched: list[str] = []

        for original_key, original_value in state_dict.items():
            key = str(original_key)
            if key.startswith("module."):
                key = key[len("module.") :]
            key = ReIDModelRegistry._normalize_checkpoint_key(model, key)
            value = original_value
            if isinstance(value, torch.Tensor):
                key, value = ReIDModelRegistry._normalize_checkpoint_tensor(model_dict, key, value)
            if key not in model_dict or not isinstance(value, torch.Tensor):
                unexpected.append(key)
            elif model_dict[key].shape != value.shape:
                mismatched.append(key)
            else:
                matched[key] = value

        missing = sorted(set(model_dict).difference(matched))
        return model_dict, matched, tuple(missing), tuple(sorted(set(unexpected))), tuple(sorted(set(mismatched)))

    @staticmethod
    def load_partial_weights(model, weight_path) -> StateDictLoadReport:
        """Explicitly load compatible tensors for transfer learning.

        This API intentionally permits partial coverage.  Runtime deployment
        uses :meth:`load_deployment_weights`, which enforces all inference
        tensors instead.
        """
        weight_path = Path(weight_path)
        checkpoint = ReIDModelRegistry._load_checkpoint(weight_path, strict=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected checkpoint state mapping, got {type(state_dict).__name__}")
        model_dict, matched, missing, unexpected, mismatched = ReIDModelRegistry._match_checkpoint_state(
            model,
            state_dict,
        )
        model.load_state_dict(matched, strict=False)
        target_numel = sum(value.numel() for value in model_dict.values())
        matched_numel = sum(model_dict[key].numel() for key in matched)
        report = StateDictLoadReport(
            matched_keys=tuple(sorted(matched)),
            missing_keys=missing,
            unexpected_keys=unexpected,
            mismatched_keys=mismatched,
            allowed_missing_keys=(),
            allowed_unexpected_keys=(),
            allowed_mismatched_keys=(),
            tensor_coverage=len(matched) / max(len(model_dict), 1),
            numel_coverage=matched_numel / max(target_numel, 1),
        )

        if not matched:
            LOGGER.debug(
                f"Pretrained weights from {weight_path} cannot be loaded. Check key names manually."
            )
        else:
            LOGGER.info(
                f"Loaded transfer weights from {weight_path}: "
                f"{len(matched)}/{len(model_dict)} tensors, {report.numel_coverage:.2%} elements"
            )

        discarded_layers = (*unexpected, *mismatched)
        if discarded_layers:
            LOGGER.debug(
                f"Discarded layers due to unmatched keys or size: {discarded_layers}"
            )
        return report

    @staticmethod
    def load_pretrained_weights(model, weight_path) -> StateDictLoadReport:
        """Load partial transfer weights; deployment callers must use the strict API."""
        return ReIDModelRegistry.load_partial_weights(model, weight_path)

    @staticmethod
    def load_deployment_weights(model, weight_path) -> StateDictLoadReport:
        """Load a checkpoint only when every deployed tensor is covered."""
        weight_path = Path(weight_path)
        checkpoint = ReIDModelRegistry._load_checkpoint(weight_path, strict=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected checkpoint state mapping, got {type(state_dict).__name__}")

        model_dict, matched, missing, unexpected, mismatched = ReIDModelRegistry._match_checkpoint_state(
            model,
            state_dict,
        )
        allowed_missing = tuple(key for key in missing if _is_deployment_optional_state_key(model, key))
        allowed_unexpected = tuple(key for key in unexpected if _is_deployment_optional_state_key(model, key))
        allowed_mismatched = tuple(key for key in mismatched if _is_deployment_optional_state_key(model, key))
        required_missing = sorted(set(missing).difference(allowed_missing))
        required_unexpected = sorted(set(unexpected).difference(allowed_unexpected))
        required_mismatched = sorted(set(mismatched).difference(allowed_mismatched))
        if required_missing or required_unexpected or required_mismatched:
            raise RuntimeError(
                f"Checkpoint {weight_path} does not match the deployed model: "
                f"missing={required_missing[:12]}, unexpected={required_unexpected[:12]}, "
                f"shape_mismatch={required_mismatched[:12]}"
            )

        required_target = {
            key
            for key in model_dict
            if not _is_deployment_optional_state_key(model, key)
        }
        matched_required = required_target.intersection(matched)
        required_numel = sum(model_dict[key].numel() for key in required_target)
        matched_numel = sum(model_dict[key].numel() for key in matched_required)
        report = StateDictLoadReport(
            matched_keys=tuple(sorted(matched)),
            missing_keys=missing,
            unexpected_keys=unexpected,
            mismatched_keys=mismatched,
            allowed_missing_keys=allowed_missing,
            allowed_unexpected_keys=allowed_unexpected,
            allowed_mismatched_keys=allowed_mismatched,
            tensor_coverage=len(matched_required) / max(len(required_target), 1),
            numel_coverage=matched_numel / max(required_numel, 1),
        )
        model.load_state_dict(matched, strict=False)
        model.checkpoint_load_report = report
        LOGGER.info(
            f"Loaded deployed weights from {weight_path}: "
            f"{len(matched_required)}/{len(required_target)} required tensors, "
            f"{report.numel_coverage:.2%} required elements"
        )
        return report

    @staticmethod
    def _normalize_checkpoint_key(model, key: str) -> str:
        """Map checkpoint parameter names onto the current model."""
        if key.startswith("blocks.") and hasattr(model, "layers"):
            key = key.replace("blocks.", "layers.", 1)
        if hasattr(model, "feature_fusion_module"):
            if key.startswith("fusion_projections."):
                return key.replace(
                    "fusion_projections.",
                    "feature_fusion_module.projections.",
                    1,
                )
            if key.startswith("fusion_scales."):
                return key.replace(
                    "fusion_scales.",
                    "feature_fusion_module.residual_scales.",
                    1,
                )
            if key == "fusion_weights":
                return "feature_fusion_module.fusion_weights"
        return key

    @staticmethod
    def _normalize_checkpoint_tensor(model_dict: dict, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Map legacy checkpoint tensor values onto current parameter semantics."""
        if key.endswith(".p"):
            raw_key = f"{key[:-2]}.raw_p"
            if raw_key in model_dict and model_dict[raw_key].shape == value.shape:
                p = value.to(dtype=model_dict[raw_key].dtype).clamp(min=1.0 + 1e-6, max=8.0)
                return raw_key, torch.log(torch.expm1(p - 1.0))
        return key, value

    @staticmethod
    def show_available_models():
        LOGGER.info("Available models:")
        LOGGER.info(list(BACKBONE_REGISTRY.keys()))

    @staticmethod
    def get_nr_classes(weights):
        metadata = read_artifact_metadata(weights)
        if metadata.get("num_classes") is not None:
            return int(metadata["num_classes"])
        checkpoint = ReIDModelRegistry._checkpoint_dict(weights)
        if checkpoint and checkpoint.get("num_classes") is not None:
            return int(checkpoint["num_classes"])

        weights_name = Path(weights).stem.lower()
        for dataset_key in sorted(NR_CLASSES_DICT, key=len, reverse=True):
            if dataset_key in weights_name:
                return NR_CLASSES_DICT[dataset_key]
        return 1

    @staticmethod
    def build_model(
        name,
        weights,
        num_classes,
        loss="softmax",
        pretrained=True,
        use_gpu=True,
        **model_kwargs,
    ):
        if name not in BACKBONE_REGISTRY:
            available = list(BACKBONE_REGISTRY.keys())
            raise KeyError(f"Unknown model '{name}'. Must be one of {available}")

        try:
            spec = get_backbone_spec(name)
        except KeyError:
            spec = None
        accepts_model_kwargs = spec is not None and (
            spec.accepts_model_kwargs
            or spec.family in {"transformer", "hybrid"}
            or spec.supports_drop_path
            or spec.supports_layer_decay
        )
        if not accepts_model_kwargs:
            model_kwargs = {}

        return BACKBONE_REGISTRY[name](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **model_kwargs,
        )
