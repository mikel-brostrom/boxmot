# BoxMOT AGPL-3.0 license

"""Lazy ReID backbone registry exports."""

from boxmot.reid.backbones.registry import (
    BACKBONE_REGISTRY,
    BACKBONE_SPECS,
    BackboneSpec,
    BackboneVariant,
    build_backbone,
    get_backbone_builder,
    get_backbone_spec,
    register_variant,
    registered_backbone_names,
)

__all__ = [
    "BACKBONE_REGISTRY",
    "BACKBONE_SPECS",
    "BackboneSpec",
    "BackboneVariant",
    "build_backbone",
    "get_backbone_builder",
    "get_backbone_spec",
    "register_variant",
    "registered_backbone_names",
]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
