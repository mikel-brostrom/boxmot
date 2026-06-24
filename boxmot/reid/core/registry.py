# model_registry.py
from __future__ import annotations

from collections import OrderedDict

import torch

from boxmot.reid.core.config import MODEL_TYPES, NR_CLASSES_DICT, TRAINED_URLS
from boxmot.reid.core.factory import MODEL_FACTORY
from boxmot.utils import logger as LOGGER


class ReIDModelRegistry:
    """Encapsulates model registration and related utilities."""

    @staticmethod
    def show_downloadable_models():
        LOGGER.info("Available .pt ReID models for automatic download")
        LOGGER.info(list(TRAINED_URLS.keys()))

    @staticmethod
    def get_model_name(model):
        if hasattr(model, "is_file") and model.is_file():
            try:
                checkpoint = torch.load(
                    model,
                    map_location="cpu",
                    weights_only=False,
                    encoding="latin1",
                )
                if isinstance(checkpoint, dict) and checkpoint.get("model_name"):
                    return checkpoint["model_name"]
            except Exception:
                pass
        for name in MODEL_TYPES:
            if name in model.name:
                return name
        return None

    @staticmethod
    def get_model_url(model):
        return TRAINED_URLS.get(model.name, None)

    @staticmethod
    def get_checkpoint_preprocess(weight_path) -> str | None:
        """Return the preprocessing method stored in a checkpoint, or None."""
        try:
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict):
                return checkpoint.get("preprocess")
        except Exception:
            pass
        return None

    @staticmethod
    def get_checkpoint_model_kwargs(weight_path) -> dict:
        """Return optional architecture kwargs stored in a checkpoint."""
        try:
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict):
                values = {}
                if checkpoint.get("feat_dim") is not None:
                    values["feat_dim"] = int(checkpoint["feat_dim"])
                if checkpoint.get("neck_dim") is not None:
                    values["neck_dim"] = int(checkpoint["neck_dim"])
                if checkpoint.get("head_pool") is not None:
                    values["head_pool"] = checkpoint["head_pool"]
                if checkpoint.get("head_parts") is not None:
                    values["head_parts"] = tuple(int(part) for part in checkpoint["head_parts"])
                if checkpoint.get("head_type") is not None:
                    values["head_type"] = checkpoint["head_type"]
                if checkpoint.get("part_pooling") is not None:
                    values["part_pooling"] = checkpoint["part_pooling"]
                if checkpoint.get("num_part_tokens") is not None:
                    values["num_part_tokens"] = int(checkpoint["num_part_tokens"])
                if checkpoint.get("decouple_patterns") is not None:
                    values["decouple_patterns"] = bool(checkpoint["decouple_patterns"])
                if checkpoint.get("pattern_adapter_dim") is not None:
                    values["pattern_adapter_dim"] = int(checkpoint["pattern_adapter_dim"])
                if checkpoint.get("stripe_visibility") is not None:
                    values["stripe_visibility"] = bool(checkpoint["stripe_visibility"])
                if checkpoint.get("inference_feature") is not None:
                    values["inference_feature"] = checkpoint["inference_feature"]
                if checkpoint.get("feature_fusion") is not None:
                    values["feature_fusion"] = checkpoint["feature_fusion"]
                if checkpoint.get("drop_path_rate") is not None:
                    values["drop_path_rate"] = float(checkpoint["drop_path_rate"])
                if checkpoint.get("attention_window_layout") is not None:
                    values["attention_window_layout"] = checkpoint["attention_window_layout"]
                if checkpoint.get("attention_bias") is not None:
                    values["attention_bias"] = checkpoint["attention_bias"]
                if checkpoint.get("attention_mask") is not None:
                    values["attention_mask"] = bool(checkpoint["attention_mask"])
                if checkpoint.get("attention_shift") is not None:
                    values["attention_shift"] = bool(checkpoint["attention_shift"])
                if checkpoint.get("stage3_global") is not None:
                    values["stage3_global"] = bool(checkpoint["stage3_global"])
                if checkpoint.get("reid_adapter_stages") is not None:
                    values["reid_adapter_stages"] = tuple(
                        int(stage) for stage in checkpoint["reid_adapter_stages"]
                    )
                if checkpoint.get("reid_adapter_reduction") is not None:
                    values["reid_adapter_reduction"] = int(checkpoint["reid_adapter_reduction"])
                state_dict = checkpoint.get("state_dict", checkpoint)
                if isinstance(state_dict, dict):
                    if "feat_dim" not in values and "head.bn_global.reduction.weight" in state_dict:
                        values["feat_dim"] = int(state_dict["head.bn_global.reduction.weight"].shape[0])
                    if "neck_dim" not in values and "neck.0.weight" in state_dict:
                        values["neck_dim"] = int(state_dict["neck.0.weight"].shape[0])
                return values
        except Exception:
            pass
        return {}

    @staticmethod
    def load_pretrained_weights(model, weight_path):
        """
        Loads pretrained weights into a model.
        Chooses the proper map_location based on CUDA availability.
        """
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=False,
            encoding='latin1',
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()

        if "lmbn" in weight_path.parts:
            model.load_state_dict(model_dict, strict=True)
        else:
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present
                key = k[7:] if k.startswith("module.") else k
                key = ReIDModelRegistry._normalize_checkpoint_key(model, key)
                key, v = ReIDModelRegistry._normalize_checkpoint_tensor(model_dict, key, v)
                if key in model_dict and model_dict[key].size() == v.size():
                    new_state_dict[key] = v
                    matched_layers.append(key)
                else:
                    discarded_layers.append(key)
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)

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
        """Map legacy checkpoint parameter names onto the current model."""
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
        try:
            checkpoint = torch.load(
                weights,
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
            if isinstance(checkpoint, dict) and checkpoint.get("num_classes") is not None:
                return int(checkpoint["num_classes"])
        except Exception:
            pass
        # Extract dataset name from weights name, then look up in the class dictionary
        parts = weights.name.split("_")
        dataset_key = parts[1] if len(parts) > 1 else ""
        return NR_CLASSES_DICT.get(dataset_key, 1)

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

        # Special case handling for clip model
        if "clip" in name:
            from boxmot.reid.backbones.clip.config.defaults import _C as cfg

            if "vehicleid" in weights.name or "veri" in weights.name:
                cfg.INPUT.SIZE_TRAIN = [256, 256]
                cfg.INPUT.SIZE_TEST = [256, 256]

            return MODEL_FACTORY[name](
                cfg, num_class=num_classes, camera_num=2, view_num=1
            )

        if not str(name).startswith("csl_tinyvit"):
            model_kwargs = {}

        return MODEL_FACTORY[name](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **model_kwargs,
        )
