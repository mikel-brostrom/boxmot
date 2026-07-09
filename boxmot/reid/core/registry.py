# model_registry.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from boxmot.reid.backbones import get_backbone_spec
from boxmot.reid.backbones.common import load_partial_state_dict
from boxmot.reid.core.config import MODEL_TYPES, NR_CLASSES_DICT, TRAINED_URLS
from boxmot.reid.core.factory import MODEL_FACTORY
from boxmot.utils import logger as LOGGER

MODEL_TYPES_BY_SPECIFICITY = tuple(sorted(MODEL_TYPES, key=len, reverse=True))


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
    "feat_dim": int,
    "neck_dim": int,
    "timm_model_name": _identity,
    "use_timm_head": bool,
    "post_fusion_mixer": _identity,
    "post_fusion_mixer_reduction": int,
    "post_fusion_mixer_kernel": _int_pair,
    "post_fusion_mixer_gamma_init": float,
    "head_pool": _identity,
    "head_parts": _int_tuple,
    "head_type": _identity,
    "part_pooling": _identity,
    "num_part_tokens": int,
    "evidence_num_roles": int,
    "decouple_patterns": bool,
    "pattern_adapter_dim": int,
    "stripe_visibility": bool,
    "drop_global_aux": bool,
    "drop_global_aux_ratio": float,
    "inference_feature": _identity,
    "feature_fusion": _identity,
    "drop_path_rate": float,
    "patch_stride": int,
    "dpt_out_indices": _int_tuple,
    "dpt_fpn_fusion": _identity,
    "dpt_fpn_target_index": int,
    "attention_window_layout": _identity,
    "attention_bias": _identity,
    "attention_mask": bool,
    "attention_shift": bool,
    "stage3_global": bool,
    "reid_adapter_stages": _int_tuple,
    "reid_adapter_reduction": int,
}


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
        checkpoint = ReIDModelRegistry._checkpoint_dict(weight_path)
        return checkpoint.get("preprocess") if checkpoint else None

    @staticmethod
    def get_checkpoint_model_kwargs(weight_path) -> dict:
        """Return optional architecture kwargs stored in a checkpoint."""
        checkpoint = ReIDModelRegistry._checkpoint_dict(weight_path)
        if not checkpoint:
            return {}

        try:
            values = {
                key: converter(checkpoint[key])
                for key, converter in CHECKPOINT_MODEL_KWARG_CONVERTERS.items()
                if checkpoint.get(key) is not None
            }
            state_dict = checkpoint.get("state_dict", checkpoint)
            if isinstance(state_dict, dict):
                if "feat_dim" not in values and "head.bn_global.reduction.weight" in state_dict:
                    values["feat_dim"] = int(state_dict["head.bn_global.reduction.weight"].shape[0])
                if "neck_dim" not in values and "neck.0.weight" in state_dict:
                    values["neck_dim"] = int(state_dict["neck.0.weight"].shape[0])
            return values
        except Exception:
            return {}

    @staticmethod
    def load_pretrained_weights(model, weight_path):
        """
        Loads pretrained weights into a model.
        Chooses the proper map_location based on CUDA availability.
        """
        weight_path = Path(weight_path)
        checkpoint = ReIDModelRegistry._load_checkpoint(weight_path, strict=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model_dict = model.state_dict()

        matched_layers, discarded_layers = load_partial_state_dict(
            model,
            state_dict,
            key_transform=lambda key: ReIDModelRegistry._normalize_checkpoint_key(model, key),
            tensor_transform=lambda key, value: ReIDModelRegistry._normalize_checkpoint_tensor(
                model_dict,
                key,
                value,
            ),
        )

        if not matched_layers:
            LOGGER.debug(
                f"Pretrained weights from {weight_path} cannot be loaded. Check key names manually."
            )
        else:
            LOGGER.info(f"Loaded pretrained weights from {weight_path}")

        if discarded_layers:
            LOGGER.debug(
                f"Discarded layers due to unmatched keys or size: {discarded_layers}"
            )

    @staticmethod
    def _normalize_checkpoint_key(model, key: str) -> str:
        """Map checkpoint parameter names onto the current model."""
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
        LOGGER.info(list(MODEL_FACTORY.keys()))

    @staticmethod
    def get_nr_classes(weights):
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
        if name not in MODEL_FACTORY:
            available = list(MODEL_FACTORY.keys())
            raise KeyError(f"Unknown model '{name}'. Must be one of {available}")

        try:
            spec = get_backbone_spec(name)
        except KeyError:
            spec = None
        accepts_model_kwargs = spec is not None and (
            spec.family in {"transformer", "hybrid"}
            or spec.supports_drop_path
            or spec.supports_layer_decay
        )
        if not accepts_model_kwargs:
            model_kwargs = {}

        return MODEL_FACTORY[name](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **model_kwargs,
        )
