# BoxMOT AGPL-3.0 license

from __future__ import annotations

from dataclasses import dataclass

from boxmot.reid.backbones.families.csl_tinyvit.model import CSLTinyViT
from boxmot.reid.backbones.families.csl_tinyvit.pretrained import (
    _TINYVIT_5M_SHA256,
    _TINYVIT_5M_URL,
    _TINYVIT_11M_SHA256,
    _TINYVIT_11M_URL,
    _TINYVIT_21M_SHA256,
    _TINYVIT_21M_URL,
    load_pretrained_tinyvit,
)
from boxmot.reid.backbones.registry import BackboneVariant, register_variant

_CSL_TINYVIT_PUBLIC_NAMES = (
    "csl_tinyvit_7m",
    "csl_tinyvit_7m_v20",
    "csl_tinyvit_7m_lmbn",
    "csl_tinyvit_11m",
    "csl_tinyvit_11m_v20",
    "csl_tinyvit_11m_lmbn",
    "csl_tinyvit_23m",
    "csl_tinyvit_23m_lmbn",
    "csl_tinyvit_large",
    "csl_tinyvit_lmbn",
    "csl_tinyvit_normal",
    "csl_tinyvit_small",
)

__all__ = list(_CSL_TINYVIT_PUBLIC_NAMES)


def _build_csl_tinyvit_variant(
    *,
    num_classes: int,
    loss: str,
    pretrained: bool,
    use_gpu: bool,
    embed_dims: list[int],
    num_heads: list[int],
    drop_path_rate: float,
    pretrained_url: str,
    pretrained_sha256: str,
    **kwargs,
) -> CSLTinyViT:
    """Build one CSL-TinyViT size variant with shared ReID head defaults."""
    drop_path_rate = float(kwargs.pop("drop_path_rate", drop_path_rate))
    metric_feature = kwargs.pop("metric_feature", None)
    model = CSLTinyViT(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        img_size=kwargs.pop("img_size", (384, 128)),
        embed_dims=embed_dims,
        depths=[2, 2, 6, 2],
        num_heads=num_heads,
        window_sizes=kwargs.pop("window_sizes", None),
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        feat_dim=kwargs.pop("feat_dim", 512),
        neck_dim=kwargs.pop("neck_dim", 512),
        **kwargs,
    )
    if metric_feature is not None and hasattr(model.head, "metric_feature"):
        model.head.metric_feature = str(metric_feature)
    if pretrained:
        load_pretrained_tinyvit(model, pretrained_url, sha256=pretrained_sha256)
    return model


@dataclass(frozen=True)
class CSLTinyViTVariant:
    name: str
    embed_dims: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    drop_path_rate: float
    pretrained_url: str
    pretrained_sha256: str


@dataclass(frozen=True)
class CSLTinyViTAlias:
    name: str
    target: str
    lmbn_style_head: bool = False


def _register_csl_variant(name: str):
    return register_variant(
        BackboneVariant(
            name=name,
            family="transformer",
            default_recipe="transformer_reid",
            default_img_size=(384, 128),
            pretrained_source="TinyViT model zoo",
            supports_layer_decay=True,
            supports_drop_path=True,
        )
    )


def _set_builder_identity(builder, name: str):
    builder.__name__ = name
    builder.__qualname__ = name
    builder.__module__ = __name__
    return builder


def make_csl_tinyvit_builder(spec: CSLTinyViTVariant):
    def builder(
        num_classes: int = 1000,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        **kwargs,
    ) -> CSLTinyViT:
        drop_path_rate = kwargs.pop("drop_path_rate", spec.drop_path_rate)
        return _build_csl_tinyvit_variant(
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            embed_dims=list(spec.embed_dims),
            num_heads=list(spec.num_heads),
            drop_path_rate=drop_path_rate,
            pretrained_url=spec.pretrained_url,
            pretrained_sha256=spec.pretrained_sha256,
            **kwargs,
        )

    return _set_builder_identity(builder, spec.name)


def make_csl_tinyvit_alias_builder(spec: CSLTinyViTAlias):
    def builder(
        num_classes: int = 1000,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        **kwargs,
    ) -> CSLTinyViT:
        if spec.lmbn_style_head:
            kwargs["lmbn_style_head"] = True
        return globals()[spec.target](
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **kwargs,
        )

    return _set_builder_identity(builder, spec.name)


_CSL_TINYVIT_VARIANTS = (
    CSLTinyViTVariant(
        "csl_tinyvit_7m",
        embed_dims=(64, 128, 160, 320),
        num_heads=(2, 4, 5, 10),
        drop_path_rate=0.0,
        pretrained_url=_TINYVIT_5M_URL,
        pretrained_sha256=_TINYVIT_5M_SHA256,
    ),
    CSLTinyViTVariant(
        "csl_tinyvit_11m",
        embed_dims=(64, 128, 256, 448),
        num_heads=(2, 4, 8, 14),
        drop_path_rate=0.1,
        pretrained_url=_TINYVIT_11M_URL,
        pretrained_sha256=_TINYVIT_11M_SHA256,
    ),
    CSLTinyViTVariant(
        "csl_tinyvit_23m",
        embed_dims=(96, 192, 384, 576),
        num_heads=(3, 6, 12, 18),
        drop_path_rate=0.2,
        pretrained_url=_TINYVIT_21M_URL,
        pretrained_sha256=_TINYVIT_21M_SHA256,
    ),
)

_CSL_TINYVIT_ALIASES = (
    CSLTinyViTAlias("csl_tinyvit_small", target="csl_tinyvit_7m"),
    CSLTinyViTAlias("csl_tinyvit_normal", target="csl_tinyvit_11m"),
    CSLTinyViTAlias("csl_tinyvit_large", target="csl_tinyvit_23m"),
    CSLTinyViTAlias("csl_tinyvit_7m_lmbn", target="csl_tinyvit_7m", lmbn_style_head=True),
    CSLTinyViTAlias("csl_tinyvit_11m_lmbn", target="csl_tinyvit_11m", lmbn_style_head=True),
    CSLTinyViTAlias("csl_tinyvit_23m_lmbn", target="csl_tinyvit_23m", lmbn_style_head=True),
    CSLTinyViTAlias("csl_tinyvit_lmbn", target="csl_tinyvit_11m_lmbn"),
)


for _variant in _CSL_TINYVIT_VARIANTS:
    globals()[_variant.name] = _register_csl_variant(_variant.name)(make_csl_tinyvit_builder(_variant))


@_register_csl_variant("csl_tinyvit_7m_v20")
def csl_tinyvit_7m_v20(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """Build the promoted 7M V20 deployed topology.

    The preset deliberately excludes privileged anatomy by default: masks and
    pose metadata are a recipe-level training treatment, not an inference
    dependency.  The V20 training recipe opts that auxiliary branch back in.
    Explicit keyword arguments remain authoritative for controlled ablations
    and exact checkpoint reconstruction.
    """
    defaults = {
        "img_size": (384, 128),
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "pyramid_resize_mode": "bilinear",
        "spatial_conv_mode": "depthwise_separable",
        "feat_dim": 384,
        "neck_dim": 384,
        "head_pool": "gelu_gem",
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "scale_balanced_branches": True,
        "attention_window_layout": "rect",
        "attention_bias": "absolute",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "attention_shift": False,
        "stage3_global": False,
        "stage3_downsample": False,
        "stage2_width_merge_after": 0,
        "stage2_mlp_ratio": 4.0,
        "stage3_mlp_ratio": 4.0,
        "stage2_depth": 6,
        "stage3_depth": 2,
        "native_branch_widths": False,
        "fine_map_dim": 0,
        "drop_path_rate": 0.1,
        "anatomical_auxiliary": False,
        "anatomical_token_dim": 96,
        "anatomical_multiscale": False,
    }
    defaults.update(kwargs)
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[64, 128, 160, 320],
        num_heads=[2, 4, 5, 10],
        drop_path_rate=float(defaults.pop("drop_path_rate")),
        pretrained_url=_TINYVIT_5M_URL,
        pretrained_sha256=_TINYVIT_5M_SHA256,
        **defaults,
    )


@_register_csl_variant("csl_tinyvit_11m_v20")
def csl_tinyvit_11m_v20(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """Build the promoted 11M V20 Market-1501 topology.

    The preset makes the deployed architecture self-describing instead of
    relying on a training recipe to replace the generic 11M defaults.  Keyword
    arguments remain authoritative so checkpoint metadata and controlled
    ablations can reconstruct an exact derivative of the preset.
    """
    defaults = {
        "img_size": (384, 128),
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "pyramid_resize_mode": "bilinear",
        "spatial_conv_mode": "depthwise_separable",
        "post_fusion_mixer": "none",
        "post_fusion_mixer_reduction": 4,
        "post_fusion_mixer_kernel": (5, 3),
        "post_fusion_mixer_gamma_init": 0.0,
        "feat_dim": 512,
        "neck_dim": 512,
        "head_pool": "gelu_gem",
        "head_parts": (1, 2, 4),
        "head_type": "standard",
        "part_pooling": "stripes",
        "num_part_tokens": 4,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "branch_metric": False,
        "scale_balanced_branches": True,
        "attention_window_layout": "rect",
        "attention_bias": "absolute",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "attention_shift": False,
        "stage3_global": False,
        "stage3_downsample": False,
        "stage2_width_merge_after": 0,
        "stage2_mlp_ratio": 4.0,
        "stage3_mlp_ratio": 4.0,
        "stage2_depth": 6,
        "stage3_depth": 2,
        "native_branch_widths": False,
        "fine_map_dim": 0,
        "drop_path_rate": 0.1,
        "anatomical_auxiliary": False,
        "anatomical_token_dim": 128,
        "anatomical_multiscale": False,
    }
    defaults.update(kwargs)
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[64, 128, 256, 448],
        num_heads=[2, 4, 8, 14],
        drop_path_rate=float(defaults.pop("drop_path_rate")),
        pretrained_url=_TINYVIT_11M_URL,
        pretrained_sha256=_TINYVIT_11M_SHA256,
        **defaults,
    )


for _alias in _CSL_TINYVIT_ALIASES:
    globals()[_alias.name] = _register_csl_variant(_alias.name)(make_csl_tinyvit_alias_builder(_alias))

del _alias, _variant
