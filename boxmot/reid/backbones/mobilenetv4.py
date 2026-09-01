# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""MobileNetV4 ReID backbones using timm ImageNet weights.

The backbone comes from Hugging Face's pytorch-image-models (``timm``). BoxMOT
adds a configurable multi-scale ReID neck, global/stripe BNNeck branches,
optional post-fusion mixing, and training-only privileged anatomy supervision.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from boxmot.reid.backbones.anatomical_registry import (
    DEFAULT_ANATOMICAL_TARGET_TYPE,
)
from boxmot.reid.backbones.base import ReIDBackbone
from boxmot.reid.backbones.families.csl_tinyvit.blocks import LayerNorm2d
from boxmot.reid.backbones.families.csl_tinyvit.fusion import (
    CSLTinyViTFeatureFusion,
    PostFusionLocalMixer,
    make_spatial_conv,
)
from boxmot.reid.backbones.families.csl_tinyvit.heads import (
    GPCLiteMultiBranchHead,
    MultiBranchHead,
)
from boxmot.reid.backbones.families.csl_tinyvit.transport import MCPT_MODES
from boxmot.reid.backbones.head_registry import (
    HeadImplementation,
    get_reid_head_spec,
)
from boxmot.reid.backbones.registry import BackboneVariant, register_variant
from boxmot.utils import logger as LOGGER

_MOBILENETV4_PUBLIC_NAMES = (
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large",
    "mobilenetv4_hybrid_medium",
    "mobilenetv4_hybrid_large",
    "mobilenetv4_conv_medium_v20",
    "mobilenetv4_hybrid_medium_v20",
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
    "mobilenetv4_conv_medium_v20": (
        "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        "mobilenetv4_conv_medium.e500_r256_in1k",
        "mobilenetv4_conv_medium",
    ),
    "mobilenetv4_hybrid_medium_v20": (
        "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
        "mobilenetv4_hybrid_medium",
    ),
}

_TIMM_HEAD_MODES = frozenset(
    {
        "pooled",
        "spatial",
        "spatial_adapt_norm",
        "spatial_linear",
        "off",
    }
)
_MOBILENETV4_NECK_MODES = frozenset({"cnn", "spatial_ln"})


def _import_timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "MobileNetV4 ReID backbones require timm. Install project dependencies "
            "with `uv sync --extra cpu` (or `--extra cu130`) or install "
            "`timm>=1.0.15`."
        ) from exc
    return timm


def _resolve_timm_model_name(timm, alias: str, candidates: Sequence[str], pretrained: bool) -> str:
    """Resolve a stable timm MobileNetV4 model name across timm releases."""
    available = set(timm.list_models("mobilenetv4*", pretrained=pretrained))
    if not available and pretrained:
        available = set(timm.list_models("mobilenetv4*", pretrained=False))
    ordered_candidates = tuple(dict.fromkeys((str(alias), *candidates)))
    for candidate in ordered_candidates:
        if candidate in available:
            return candidate
    for candidate in ordered_candidates:
        matches = sorted(name for name in available if name.startswith(candidate))
        if matches:
            return matches[0]
    available_text = ", ".join(sorted(available)) or "(none)"
    raise RuntimeError(
        f"timm does not expose a MobileNetV4 model for '{alias}'. "
        f"Tried {ordered_candidates}. Available MobileNetV4 models: {available_text}"
    )


def _feature_channels(backbone: nn.Module) -> list[int]:
    feature_info = getattr(backbone, "feature_info", None)
    if feature_info is None:
        raise RuntimeError("timm MobileNetV4 features_only model did not expose feature_info")
    if hasattr(feature_info, "channels"):
        return [int(value) for value in feature_info.channels()]
    return [int(item["num_chs"]) for item in feature_info]


def _fusion_source_indices() -> dict[int, int]:
    """Map shared fusion roles onto MobileNetV4 pyramid endpoints.

    MobileNetV4 exposes stride-8 C3, stride-16 C4, and stride-32 C5 maps.
    The Stage-0 semantic-fine ReID head uses C3 for its 48x16 fine branch,
    C4 for its 24x8 coarse branch, and the timm C5 head for global semantics.
    Stage 1 also uses C3 as the intermediate semantic residual because timm
    exposes one endpoint per stride, unlike CSL-TinyViT's two stride-16 stages.
    """
    return {0: -3, 1: -3, 2: -2}


def _fusion_path_channels(
    feature_fusion: str,
    channels: Sequence[int],
    source_indices: dict[int, int],
) -> dict[int, int]:
    stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(feature_fusion)
    path_channels: dict[int, int] = {}
    for stage_index in stage_indices:
        if stage_index not in source_indices:
            raise ValueError(f"Unsupported MobileNetV4 fusion stage index: {stage_index}")
        source_index = source_indices[stage_index]
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


def _mobilenetv4_reid_neck(
    in_channels: int,
    out_channels: int,
    *,
    mode: str,
    spatial_conv_mode: str,
) -> nn.Module:
    """Build either the current CNN projection or TinyViT-matched ReID neck."""
    normalized = str(mode).lower()
    if normalized == "cnn":
        return _cnn_projection(in_channels, out_channels)
    if normalized == "spatial_ln":
        return nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False),
            LayerNorm2d(int(out_channels)),
            make_spatial_conv(int(out_channels), mode=spatial_conv_mode),
            LayerNorm2d(int(out_channels)),
        )
    raise ValueError(
        "mobilenetv4_neck_mode must be one of: "
        + ", ".join(sorted(_MOBILENETV4_NECK_MODES))
    )


def _set_mobilenetv4_last_stride(backbone: nn.Module, last_stride: int) -> None:
    """Optionally retain stride-16 C5 maps by removing the final stride-2 conv."""
    normalized = int(last_stride)
    if normalized == 2:
        return
    if normalized != 1:
        raise ValueError("mobilenetv4_last_stride must be 1 or 2")
    for module in reversed(list(backbone.modules())):
        if isinstance(module, nn.Conv2d) and tuple(module.stride) == (2, 2):
            module.stride = (1, 1)
            return
    raise RuntimeError("Could not locate MobileNetV4's final stride-2 convolution")


def _timm_head_channels(backbone: nn.Module, fallback: int) -> int:
    conv_head = getattr(backbone, "conv_head", None)
    if isinstance(conv_head, nn.Conv2d):
        return int(conv_head.out_channels)
    return int(fallback)


def _timm_pretrained_url(backbone: nn.Module, model_name: str) -> str:
    """Return the concrete timm/Hugging Face source selected by the model."""
    config = getattr(backbone, "pretrained_cfg", None)
    if isinstance(config, dict):
        if config.get("url"):
            return str(config["url"])
        if config.get("hf_hub_id"):
            return f"https://huggingface.co/{config['hf_hub_id']}"
    return f"https://huggingface.co/timm/{model_name}"


def _module_state_sha256(module: nn.Module) -> str:
    """Fingerprint the exact pretrained tensors materialized by timm."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_timm_head_mode(mode: str) -> str:
    """Normalize how the pretrained timm classification head handles C5."""
    normalized = str(mode).lower()
    if normalized not in _TIMM_HEAD_MODES:
        raise ValueError(
            "timm_head_mode must be one of: "
            + ", ".join(sorted(_TIMM_HEAD_MODES))
        )
    return normalized


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
        pyramid_resize_mode: str = "bilinear",
        spatial_conv_mode: str = "standard",
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
        scale_balanced_branches: bool = False,
        drop_path_rate: float = 0.0,
        use_timm_head: bool = True,
        timm_head_mode: str | None = None,
        mobilenetv4_last_stride: int = 2,
        mobilenetv4_neck_mode: str = "cnn",
        anatomical_auxiliary: bool = False,
        anatomical_token_dim: int = 128,
        anatomical_descriptor_distill: bool = False,
        anatomical_branch_distill: bool = False,
        anatomical_multiscale: bool = False,
        anatomical_target_type: str = DEFAULT_ANATOMICAL_TARGET_TYPE,
        anatomical_accessory_query: bool = False,
        anatomical_deployment: bool = False,
        anatomical_deployment_dim: int = 64,
        anatomical_deployment_alpha: float = 0.25,
        mcpt_mode: str = "none",
        mcpt_hidden_dim: int = 64,
        mcpt_max_displacement: float = 0.15,
        mcpt_start_epoch: int = 10,
        mcpt_ramp_end_epoch: int = 40,
        return_cross_scale_features: bool = False,
        return_treeboost_features: bool = False,
        return_auxiliary_features: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        del use_gpu
        if kwargs:
            LOGGER.debug(f"Ignoring unsupported MobileNetV4 ReID kwargs: {sorted(kwargs)}")
        self.loss = loss
        self.img_size = tuple(int(value) for value in img_size)
        self.feature_fusion = CSLTinyViTFeatureFusion.normalize_mode(feature_fusion)
        self.pyramid_resize_mode = CSLTinyViTFeatureFusion.normalize_resize_mode(pyramid_resize_mode)
        self.spatial_conv_mode = CSLTinyViTFeatureFusion.normalize_spatial_conv_mode(spatial_conv_mode)
        self.post_fusion_mixer = self._normalize_post_fusion_mixer(post_fusion_mixer)
        self.post_fusion_mixer_reduction = int(post_fusion_mixer_reduction)
        self.post_fusion_mixer_kernel = self._normalize_pair(post_fusion_mixer_kernel)
        self.post_fusion_mixer_gamma_init = float(post_fusion_mixer_gamma_init)
        self.head_type = str(head_type).lower()
        self.head_spec = get_reid_head_spec(
            self.head_type,
            family="mobilenetv4",
        )
        self.mcpt_mode = str(mcpt_mode).lower()
        self.mcpt_hidden_dim = int(mcpt_hidden_dim)
        self.mcpt_max_displacement = float(mcpt_max_displacement)
        self.mcpt_start_epoch = int(mcpt_start_epoch)
        self.mcpt_ramp_end_epoch = int(mcpt_ramp_end_epoch)
        if self.mcpt_mode not in MCPT_MODES:
            raise ValueError(
                f"mcpt_mode must be one of {sorted(MCPT_MODES)}, "
                f"got {mcpt_mode!r}"
            )
        if self.mcpt_mode != "none" and self.head_type != "standard":
            raise ValueError("MobileNetV4 MCPT requires head_type='standard'")
        self.scale_balanced_branches = bool(scale_balanced_branches)
        if self.scale_balanced_branches and self.head_type != "standard":
            raise ValueError("MobileNetV4 scale-balanced branches require head_type='standard'")
        if self.scale_balanced_branches and branch_metric:
            raise ValueError("MobileNetV4 scale-balanced branches use one selected metric descriptor")
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
        self.pretrained_url = (
            _timm_pretrained_url(self.backbone, self.timm_model_name)
            if pretrained
            else None
        )
        self.pretrained_sha256 = (
            _module_state_sha256(self.backbone) if pretrained else None
        )

        self.mobilenetv4_last_stride = int(mobilenetv4_last_stride)
        _set_mobilenetv4_last_stride(
            self.backbone,
            self.mobilenetv4_last_stride,
        )
        self.mobilenetv4_neck_mode = str(mobilenetv4_neck_mode).lower()
        if self.mobilenetv4_neck_mode not in _MOBILENETV4_NECK_MODES:
            raise ValueError(
                "mobilenetv4_neck_mode must be one of: "
                + ", ".join(sorted(_MOBILENETV4_NECK_MODES))
            )

        channels = _feature_channels(self.backbone)
        if len(channels) < 2:
            raise RuntimeError(f"Expected multiple MobileNetV4 feature maps, got channels={channels}")
        self.feature_channels = tuple(channels)
        final_channels = channels[-1]
        if timm_head_mode is None:
            timm_head_mode = "pooled" if use_timm_head else "off"
        self.timm_head_mode = _normalize_timm_head_mode(timm_head_mode)
        if not use_timm_head and self.timm_head_mode != "off":
            raise ValueError(
                "use_timm_head=False is only compatible with timm_head_mode='off'"
            )
        self.use_timm_head = self.timm_head_mode != "off"
        self.timm_head_channels = _timm_head_channels(self.backbone, final_channels)

        global_input_channels = self.timm_head_channels if self.use_timm_head else final_channels
        self.neck = _mobilenetv4_reid_neck(
            global_input_channels,
            neck_dim,
            mode=self.mobilenetv4_neck_mode,
            spatial_conv_mode=self.spatial_conv_mode,
        )
        self.spatial_neck = _mobilenetv4_reid_neck(
            final_channels,
            neck_dim,
            mode=self.mobilenetv4_neck_mode,
            spatial_conv_mode=self.spatial_conv_mode,
        )
        self._fusion_source_indices = _fusion_source_indices()
        fusion_path_channels = _fusion_path_channels(
            self.feature_fusion,
            channels,
            self._fusion_source_indices,
        )
        # Fusion arms contain different numbers of randomly initialized layers.
        # Keep their construction from shifting the RNG state seen by the
        # shared post-fusion mixer and ReID head so same-seed ablations retain
        # identical initialization outside the treatment module.
        with torch.random.fork_rng(devices=[]):
            self.feature_fusion_module = CSLTinyViTFeatureFusion.from_mode(
                mode=self.feature_fusion,
                path_channels=fusion_path_channels,
                out_channels=neck_dim,
                resize_mode=self.pyramid_resize_mode,
                spatial_conv_mode=self.spatial_conv_mode,
            )
        self._fusion_stage_indices = self.feature_fusion_module.stage_indices

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
        if self.head_spec.implementation == HeadImplementation.GPC_LITE:
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
                scale_balanced_branches=self.scale_balanced_branches,
                hierarchical_scales=(
                    CSLTinyViTFeatureFusion.uses_hierarchical_scales(
                        self.feature_fusion
                    )
                ),
                anatomical_auxiliary=anatomical_auxiliary,
                anatomical_token_dim=anatomical_token_dim,
                anatomical_descriptor_distill=anatomical_descriptor_distill,
                anatomical_branch_distill=anatomical_branch_distill,
                anatomical_multiscale=anatomical_multiscale,
                anatomical_target_type=anatomical_target_type,
                anatomical_accessory_query=anatomical_accessory_query,
                anatomical_deployment=anatomical_deployment,
                anatomical_deployment_dim=anatomical_deployment_dim,
                anatomical_deployment_alpha=anatomical_deployment_alpha,
                mcpt_mode=self.mcpt_mode,
                mcpt_hidden_dim=self.mcpt_hidden_dim,
                mcpt_max_displacement=self.mcpt_max_displacement,
                mcpt_start_epoch=self.mcpt_start_epoch,
                mcpt_ramp_end_epoch=self.mcpt_ramp_end_epoch,
                return_cross_scale_features=return_cross_scale_features,
                return_treeboost_features=return_treeboost_features,
                return_auxiliary_features=return_auxiliary_features,
            )
        self.pretrained_source = "huggingface/pytorch-image-models (timm)"
        LOGGER.info(
            f"MobileNetV4 ReID backbone: timm_model={self.timm_model_name}, "
            f"pretrained={pretrained}, timm_head_mode={self.timm_head_mode}, "
            f"last_stride={self.mobilenetv4_last_stride}, "
            f"neck_mode={self.mobilenetv4_neck_mode}, "
            f"source={self.pretrained_source}"
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
        if self.timm_head_mode == "off":
            return final_feature
        required = ("conv_head",)
        if self.timm_head_mode != "spatial_linear":
            required += ("norm_head",)
        if self.timm_head_mode == "pooled":
            required = ("global_pool", *required)
        if not all(hasattr(self.backbone, name) for name in required):
            return final_feature
        feature = final_feature
        if self.timm_head_mode == "pooled":
            feature = self.backbone.global_pool(feature)
        feature = self.backbone.conv_head(feature)
        if self.timm_head_mode != "spatial_linear":
            feature = self.backbone.norm_head(feature)
            act2 = getattr(self.backbone, "act2", None)
            if act2 is not None:
                feature = act2(feature)
        if feature.ndim == 2:
            feature = feature[:, :, None, None]
        return feature

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        final_raw, features = self._forward_intermediates(x)
        path_features = {
            stage_index: features[self._fusion_source_indices[stage_index]]
            for stage_index in self._fusion_stage_indices
        }

        # A MobileNet checkpoint retains both projections so older runs remain
        # strictly resumable, but each fixed fusion mode consumes only one of
        # them. Avoid executing the other 1x1 projection on every image.
        uses_final_global = (
            not self._fusion_stage_indices
            or CSLTinyViTFeatureFusion.uses_final_global_branch(self.feature_fusion)
        )
        if uses_final_global:
            fusion_final = self.neck(self._forward_timm_head(final_raw))
        else:
            fusion_final = self.spatial_neck(final_raw)

        if not self._fusion_stage_indices:
            fused = fusion_final
        else:
            fused = self.feature_fusion_module(fusion_final, path_features)
        if isinstance(fused, tuple):
            return tuple(self.post_fusion_mixer_module(feature) for feature in fused)
        return self.post_fusion_mixer_module(fused)

    def forward_head(
        self,
        features,
        *,
        anatomical_pose: torch.Tensor | None = None,
        anatomical_query_masks: torch.Tensor | None = None,
    ):
        """Convert pyramid features into ReID outputs and optional pose targets."""
        if anatomical_pose is None and anatomical_query_masks is None:
            return self.head(features)
        return self.head(
            features,
            anatomical_pose=anatomical_pose,
            anatomical_query_masks=anatomical_query_masks,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        anatomical_pose: torch.Tensor | None = None,
        anatomical_query_masks: torch.Tensor | None = None,
    ):
        """Run RGB inference with optional privileged inputs during training."""
        return self.forward_head(
            self.forward_features(x),
            anatomical_pose=anatomical_pose,
            anatomical_query_masks=anatomical_query_masks,
        )


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
