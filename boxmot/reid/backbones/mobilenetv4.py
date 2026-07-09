# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""MobileNetV4 ReID backbones using timm ImageNet weights.

The backbone comes from Hugging Face's pytorch-image-models (``timm``). BoxMOT
adds the ReID-specific CSL-TinyViT head path on top: multi-scale feature fusion,
the 512-channel neck, global/stripe BNNeck branches, optional post-fusion local
mixer, and optional dropped-global CE auxiliary supervision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from boxmot.reid.backbones.base import ReIDBackbone
from boxmot.reid.backbones.families.csl_tinyvit.fusion import (
    CSLTinyViTFeatureFusion,
    PostFusionLocalMixer,
)
from boxmot.reid.backbones.families.csl_tinyvit.heads import (
    GPCLiteMultiBranchHead,
    MultiBranchHead,
)
from boxmot.reid.backbones.registry import BackboneVariant, register_variant
from boxmot.utils import logger as LOGGER

_MOBILENETV4_PUBLIC_NAMES = (
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large",
    "mobilenetv4_hybrid_medium",
    "mobilenetv4_hybrid_large",
)

__all__ = ["TimmMobileNetV4ReID", *_MOBILENETV4_PUBLIC_NAMES]


_TIMM_MODEL_CANDIDATES = {
    "mobilenetv4_conv_small": (
        "mobilenetv4_conv_small.e2400_r224_in1k",
        "mobilenetv4_conv_small",
    ),
    "mobilenetv4_conv_medium": (
        "mobilenetv4_conv_medium.e500_r256_in1k",
        "mobilenetv4_conv_medium",
    ),
    "mobilenetv4_conv_large": (
        "mobilenetv4_conv_large.e600_r384_in1k",
        "mobilenetv4_conv_large",
    ),
    "mobilenetv4_hybrid_medium": (
        "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
        "mobilenetv4_hybrid_medium",
    ),
    "mobilenetv4_hybrid_large": (
        "mobilenetv4_hybrid_large.e600_r384_in1k",
        "mobilenetv4_hybrid_large",
    ),
}


def _import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "MobileNetV4 ReID backbones require timm. Install project dependencies "
            "with `uv sync --all-extras --all-groups` or install `timm>=1.0.15`."
        ) from exc
    return timm


def _resolve_timm_model_name(timm, alias: str, candidates: Sequence[str], pretrained: bool) -> str:
    """Resolve a stable timm MobileNetV4 model name across timm releases."""
    available = set(timm.list_models("mobilenetv4*", pretrained=pretrained))
    if not available and pretrained:
        available = set(timm.list_models("mobilenetv4*", pretrained=False))
    for candidate in candidates:
        if candidate in available:
            return candidate
    for candidate in candidates:
        matches = sorted(name for name in available if name.startswith(candidate))
        if matches:
            return matches[0]
    available_text = ", ".join(sorted(available)) or "(none)"
    raise RuntimeError(
        f"timm does not expose a MobileNetV4 model for '{alias}'. "
        f"Tried {tuple(candidates)}. Available MobileNetV4 models: {available_text}"
    )


def _feature_channels(backbone: nn.Module) -> list[int]:
    feature_info = getattr(backbone, "feature_info", None)
    if feature_info is None:
        raise RuntimeError("timm MobileNetV4 features_only model did not expose feature_info")
    if hasattr(feature_info, "channels"):
        return [int(value) for value in feature_info.channels()]
    return [int(item["num_chs"]) for item in feature_info]


def _fusion_path_channels(feature_fusion: str, channels: Sequence[int]) -> dict[int, int]:
    stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(feature_fusion)
    path_channels: dict[int, int] = {}
    for stage_index in stage_indices:
        if stage_index == 0:
            source_index = -4
        elif stage_index == 1:
            source_index = -3
        elif stage_index == 2:
            source_index = -2
        else:
            raise ValueError(f"Unsupported MobileNetV4 fusion stage index: {stage_index}")
        try:
            path_channels[stage_index] = int(channels[source_index])
        except IndexError as exc:
            raise RuntimeError(
                f"MobileNetV4 feature_fusion={feature_fusion!r} requires at least "
                f"{abs(source_index)} feature maps, got {len(channels)}"
            ) from exc
    return path_channels


def _cnn_projection(in_channels: int, out_channels: int) -> nn.Module:
    """Project CNN feature maps with CNN-native normalization."""
    if int(in_channels) == int(out_channels):
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False),
        nn.BatchNorm2d(int(out_channels)),
        nn.ReLU(inplace=True),
    )


def _timm_head_channels(backbone: nn.Module, fallback: int) -> int:
    conv_head = getattr(backbone, "conv_head", None)
    if isinstance(conv_head, nn.Conv2d):
        return int(conv_head.out_channels)
    return int(fallback)


class TimmMobileNetV4ReID(ReIDBackbone):
    """MobileNetV4 feature extractor with the CSL-TinyViT ReID head stack."""

    def __init__(
        self,
        num_classes: int,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        *,
        timm_model_name: str,
        timm_model_candidates: Sequence[str] = (),
        img_size: tuple[int, int] = (256, 128),
        feat_dim: int = 512,
        neck_dim: int = 512,
        metric_feature: str = "auto",
        inference_feature: str = "concat_bn",
        feature_fusion: str = "final",
        post_fusion_mixer: str = "none",
        post_fusion_mixer_reduction: int = 4,
        post_fusion_mixer_kernel: tuple[int, int] = (5, 3),
        post_fusion_mixer_gamma_init: float = 0.0,
        head_pool: str = "avg",
        head_parts: tuple[int, ...] = (1,),
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        evidence_num_roles: int = 8,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        head_type: str = "standard",
        stripe_visibility: bool = False,
        drop_global_aux: bool = False,
        drop_global_aux_ratio: float = 0.25,
        branch_metric: bool = False,
        drop_path_rate: float = 0.0,
        use_timm_head: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        del use_gpu
        if kwargs:
            LOGGER.debug(f"Ignoring unsupported MobileNetV4 ReID kwargs: {sorted(kwargs)}")
        self.loss = loss
        self.img_size = tuple(int(value) for value in img_size)
        self.feature_fusion = CSLTinyViTFeatureFusion.normalize_mode(feature_fusion)
        self.post_fusion_mixer = self._normalize_post_fusion_mixer(post_fusion_mixer)
        self.post_fusion_mixer_reduction = int(post_fusion_mixer_reduction)
        self.post_fusion_mixer_kernel = self._normalize_pair(post_fusion_mixer_kernel)
        self.post_fusion_mixer_gamma_init = float(post_fusion_mixer_gamma_init)
        self.head_type = str(head_type).lower()
        if self.head_type not in {"standard", "gpc_lite"}:
            raise ValueError("MobileNetV4 ReID head_type must be one of: standard, gpc_lite")
        if drop_global_aux and self.head_type != "standard":
            raise ValueError("drop_global_aux requires MobileNetV4 head_type='standard'")

        timm = _import_timm()
        self.timm_model_name = _resolve_timm_model_name(
            timm,
            timm_model_name,
            tuple(timm_model_candidates) or (timm_model_name,),
            pretrained=pretrained,
        )
        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_path_rate": float(drop_path_rate),
        }
        try:
            self.backbone = timm.create_model(self.timm_model_name, **create_kwargs)
        except TypeError:
            create_kwargs.pop("drop_path_rate")
            self.backbone = timm.create_model(self.timm_model_name, **create_kwargs)

        channels = _feature_channels(self.backbone)
        if len(channels) < 2:
            raise RuntimeError(f"Expected multiple MobileNetV4 feature maps, got channels={channels}")
        self.feature_channels = tuple(channels)
        final_channels = channels[-1]
        self.use_timm_head = bool(use_timm_head)
        self.timm_head_channels = _timm_head_channels(self.backbone, final_channels)

        global_input_channels = self.timm_head_channels if self.use_timm_head else final_channels
        self.neck = _cnn_projection(global_input_channels, neck_dim)
        self.spatial_neck = _cnn_projection(final_channels, neck_dim)
        fusion_path_channels = _fusion_path_channels(self.feature_fusion, channels)
        self.feature_fusion_module = CSLTinyViTFeatureFusion.from_mode(
            mode=self.feature_fusion,
            path_channels=fusion_path_channels,
            out_channels=neck_dim,
        )
        self._fusion_stage_indices = self.feature_fusion_module.stage_indices
        self._fusion_source_indices = {0: -4, 1: -3, 2: -2}

        if self.post_fusion_mixer == "dwconv":
            self.post_fusion_mixer_module = PostFusionLocalMixer(
                channels=neck_dim,
                reduction=self.post_fusion_mixer_reduction,
                kernel_size=self.post_fusion_mixer_kernel,
                gamma_init=self.post_fusion_mixer_gamma_init,
            )
        else:
            self.post_fusion_mixer_module = nn.Identity()

        metric_feature = str(metric_feature).lower()
        if metric_feature == "auto":
            metric_feature = "concat_bn" if loss == "ms" else "raw_mean"
        if self.head_type == "gpc_lite":
            self.head = GPCLiteMultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                branch_metric=branch_metric,
            )
        else:
            self.head = MultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                part_pooling=part_pooling,
                num_part_tokens=num_part_tokens,
                evidence_num_roles=evidence_num_roles,
                decouple_patterns=decouple_patterns,
                pattern_adapter_dim=pattern_adapter_dim,
                stripe_visibility=stripe_visibility,
                drop_global_aux=drop_global_aux,
                drop_global_aux_ratio=drop_global_aux_ratio,
                branch_metric=branch_metric,
            )
        self.pretrained_source = "huggingface/pytorch-image-models (timm)"
        LOGGER.info(
            f"MobileNetV4 ReID backbone: timm_model={self.timm_model_name}, "
            f"pretrained={pretrained}, source={self.pretrained_source}"
        )

    @staticmethod
    def _normalize_pair(value) -> tuple[int, int]:
        if isinstance(value, int):
            return (int(value), int(value))
        values = tuple(int(part) for part in value)
        if len(values) == 1:
            return (values[0], values[0])
        if len(values) != 2:
            raise ValueError(f"Expected one or two integers, got {value!r}")
        return values

    @staticmethod
    def _normalize_post_fusion_mixer(mixer: str) -> str:
        normalized = str(mixer).lower()
        if normalized in {"", "none", "off", "identity"}:
            return "none"
        if normalized in {"dwconv", "local", "dwconv5x3"}:
            return "dwconv"
        raise ValueError("post_fusion_mixer must be one of: none, dwconv")

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def _forward_intermediates(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if hasattr(self.backbone, "forward_intermediates"):
            final, intermediates = self.backbone.forward_intermediates(x)
            return final, list(intermediates)
        intermediates = list(self.backbone(x))
        return intermediates[-1], intermediates

    def _forward_timm_head(self, final_feature: torch.Tensor) -> torch.Tensor:
        if not self.use_timm_head:
            return final_feature
        required = ("global_pool", "conv_head", "norm_head")
        if not all(hasattr(self.backbone, name) for name in required):
            return final_feature
        feature = self.backbone.global_pool(final_feature)
        feature = self.backbone.conv_head(feature)
        feature = self.backbone.norm_head(feature)
        act2 = getattr(self.backbone, "act2", None)
        if act2 is not None:
            feature = act2(feature)
        if feature.ndim == 2:
            feature = feature[:, :, None, None]
        return feature

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        final_raw, features = self._forward_intermediates(x)
        final = self.neck(self._forward_timm_head(final_raw))
        spatial_final = self.spatial_neck(final_raw)
        path_features = {
            stage_index: features[self._fusion_source_indices[stage_index]]
            for stage_index in self._fusion_stage_indices
        }
        if not self._fusion_stage_indices:
            fused = final
        else:
            fusion_final = final if self.feature_fusion_module.split_global_local else spatial_final
            fused = self.feature_fusion_module(fusion_final, path_features)
        if isinstance(fused, tuple):
            return tuple(self.post_fusion_mixer_module(feature) for feature in fused)
        return self.post_fusion_mixer_module(fused)

    def forward_head(self, features):
        return self.head(features)


def _build_mobilenetv4_variant(
    *,
    alias: str,
    num_classes: int,
    loss: str,
    pretrained: bool,
    use_gpu: bool,
    **kwargs,
) -> TimmMobileNetV4ReID:
    candidates = _TIMM_MODEL_CANDIDATES[alias]
    timm_model_name = kwargs.pop("timm_model_name", candidates[0])
    return TimmMobileNetV4ReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        timm_model_name=timm_model_name,
        timm_model_candidates=candidates,
        **kwargs,
    )


def make_mobilenetv4_builder(alias: str):
    def builder(num_classes, loss="softmax", pretrained=True, use_gpu=True, **kwargs):
        return _build_mobilenetv4_variant(
            alias=alias,
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **kwargs,
        )

    builder.__name__ = alias
    builder.__qualname__ = alias
    builder.__module__ = __name__
    return builder


for _variant_name in _TIMM_MODEL_CANDIDATES:
    globals()[_variant_name] = register_variant(
        BackboneVariant(
            name=_variant_name,
            family="hybrid",
            default_recipe="hybrid_reid",
            default_img_size=(384, 128),
            supports_drop_path=True,
            pretrained_source="timm",
        )
    )(make_mobilenetv4_builder(_variant_name))

del _variant_name
