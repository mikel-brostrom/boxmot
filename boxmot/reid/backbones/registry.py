# BoxMOT AGPL-3.0 license

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from boxmot.reid.backbones.common.typing import BackboneFamily, ImageSize, RecipeName

BackboneBuilder = Callable[..., Any]


@dataclass(frozen=True)
class BackboneSpec:
    """Metadata used by training/config layers without coupling it to models."""

    name: str
    family: BackboneFamily
    default_recipe: RecipeName
    default_img_size: ImageSize
    supports_layer_decay: bool = False
    supports_drop_path: bool = False
    pretrained_source: str | None = None


@dataclass(frozen=True)
class BackboneVariant:
    """Declarative registration metadata for a backbone builder."""

    name: str
    family: BackboneFamily
    default_recipe: RecipeName
    default_img_size: ImageSize = (256, 128)
    pretrained_source: str | None = None
    supports_layer_decay: bool = False
    supports_drop_path: bool = False
    aliases: tuple[str, ...] = ()

    def spec(self, name: str | None = None) -> BackboneSpec:
        return BackboneSpec(
            name=name or self.name,
            family=self.family,
            default_recipe=self.default_recipe,
            default_img_size=self.default_img_size,
            supports_layer_decay=self.supports_layer_decay,
            supports_drop_path=self.supports_drop_path,
            pretrained_source=self.pretrained_source,
        )


BACKBONE_REGISTRY: dict[str, BackboneBuilder] = {}
BACKBONE_SPECS: dict[str, BackboneSpec] = {}


@dataclass(frozen=True)
class LazyBackboneImport:
    """Import target and static metadata for a lazily registered backbone."""

    variant: BackboneVariant
    module: str
    attr: str

    @property
    def name(self) -> str:
        return self.variant.name

    def spec(self) -> BackboneSpec:
        return self.variant.spec()


class LazyBackboneBuilder:
    """Callable proxy that imports a backbone implementation only when used."""

    def __init__(self, import_spec: LazyBackboneImport) -> None:
        self.name = import_spec.name
        self.module = import_spec.module
        self.attr = import_spec.attr
        self._resolved: BackboneBuilder | None = None

    @property
    def __name__(self) -> str:
        return self.attr

    @property
    def __module__(self) -> str:
        return self.module

    def _matches(self, fn: BackboneBuilder) -> bool:
        return getattr(fn, "__module__", None) == self.module and getattr(fn, "__name__", None) == self.attr

    def resolve(self) -> BackboneBuilder:
        if self._resolved is None:
            module = import_module(self.module)
            self._resolved = getattr(module, self.attr)
        return self._resolved

    def __call__(self, *args, **kwargs):
        return self.resolve()(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<lazy backbone {self.name!r} -> {self.module}:{self.attr}>"


def register_backbone(
    name: str,
    *,
    family: BackboneFamily,
    default_recipe: RecipeName,
    default_img_size: ImageSize = (256, 128),
    supports_layer_decay: bool = False,
    supports_drop_path: bool = False,
    pretrained_source: str | None = None,
    aliases: tuple[str, ...] = (),
) -> Callable[[BackboneBuilder], BackboneBuilder]:
    """Register a ReID backbone builder and its training-facing metadata."""

    variant = BackboneVariant(
        name=name,
        family=family,
        default_recipe=default_recipe,
        default_img_size=default_img_size,
        pretrained_source=pretrained_source,
        supports_layer_decay=supports_layer_decay,
        supports_drop_path=supports_drop_path,
        aliases=aliases,
    )

    def decorator(fn: BackboneBuilder) -> BackboneBuilder:
        for registered_name in (variant.name, *variant.aliases):
            existing = BACKBONE_REGISTRY.get(registered_name)
            if existing is not None and not (isinstance(existing, LazyBackboneBuilder) and existing._matches(fn)):
                raise KeyError(f"Backbone already registered: {registered_name}")
            BACKBONE_REGISTRY[registered_name] = fn
            BACKBONE_SPECS[registered_name] = variant.spec(registered_name)
        return fn

    return decorator


def register_variant(spec: BackboneVariant) -> Callable[[BackboneBuilder], BackboneBuilder]:
    """Register a builder from declarative variant metadata."""
    return register_backbone(
        spec.name,
        family=spec.family,
        default_recipe=spec.default_recipe,
        default_img_size=spec.default_img_size,
        supports_layer_decay=spec.supports_layer_decay,
        supports_drop_path=spec.supports_drop_path,
        pretrained_source=spec.pretrained_source,
        aliases=spec.aliases,
    )


def get_backbone_builder(name: str) -> BackboneBuilder:
    try:
        builder = BACKBONE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(registered_backbone_names())
        raise KeyError(f"Unknown backbone '{name}'. Available: {available}") from exc
    if isinstance(builder, LazyBackboneBuilder):
        builder = builder.resolve()
        BACKBONE_REGISTRY[name] = builder
    return builder


def build_backbone(name: str, **kwargs) -> Any:
    builder = get_backbone_builder(name)
    return builder(**kwargs)


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name]
    except KeyError as exc:
        available = ", ".join(registered_backbone_names())
        raise KeyError(f"Unknown backbone '{name}'. Available: {available}") from exc


def registered_backbone_names() -> tuple[str, ...]:
    return tuple(sorted(BACKBONE_REGISTRY))


def _register_lazy_backbone(import_spec: LazyBackboneImport) -> None:
    existing = BACKBONE_REGISTRY.get(import_spec.name)
    if existing is None:
        BACKBONE_REGISTRY[import_spec.name] = LazyBackboneBuilder(import_spec)
    elif not isinstance(existing, LazyBackboneBuilder):
        return
    BACKBONE_SPECS[import_spec.name] = import_spec.spec()


def _lazy_variant(
    name: str,
    module: str,
    *,
    attr: str | None = None,
    family: BackboneFamily,
    default_recipe: RecipeName,
    default_img_size: ImageSize = (256, 128),
    supports_layer_decay: bool = False,
    supports_drop_path: bool = False,
    pretrained_source: str | None = None,
) -> LazyBackboneImport:
    return LazyBackboneImport(
        variant=BackboneVariant(
            name=name,
            family=family,
            default_recipe=default_recipe,
            default_img_size=default_img_size,
            pretrained_source=pretrained_source,
            supports_layer_decay=supports_layer_decay,
            supports_drop_path=supports_drop_path,
        ),
        module=module,
        attr=attr or name,
    )


def _lazy_variants(
    names: tuple[str, ...],
    module: str,
    *,
    family: BackboneFamily,
    default_recipe: RecipeName,
    default_img_size: ImageSize = (256, 128),
    supports_layer_decay: bool = False,
    supports_drop_path: bool = False,
    pretrained_source: str | None = None,
) -> tuple[LazyBackboneImport, ...]:
    return tuple(
        _lazy_variant(
            name,
            module,
            family=family,
            default_recipe=default_recipe,
            default_img_size=default_img_size,
            supports_layer_decay=supports_layer_decay,
            supports_drop_path=supports_drop_path,
            pretrained_source=pretrained_source,
        )
        for name in names
    )


def _register_builtin_backbones() -> None:
    resnet_module = "boxmot.reid.backbones.resnet"
    csl_module = "boxmot.reid.backbones.families.csl_tinyvit.variants"
    osnet_module = "boxmot.reid.backbones.families.osnet.variants"
    mobilenetv4_module = "boxmot.reid.backbones.mobilenetv4"

    for import_spec in (
        # ResNet / ResNeXt
        *_lazy_variants(
            (
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
                "resnext50_32x4d",
                "resnext101_32x8d",
                "resnet50_fc512",
            ),
            resnet_module,
            family="cnn",
            default_recipe="cnn_reid",
            pretrained_source="torchvision",
        ),
        # OSNet family
        *_lazy_variants(
            (
                "osnet_x1_0",
                "osnet_x0_75",
                "osnet_x0_5",
                "osnet_x0_25",
                "osnet_ibn_x1_0",
                "osnet_ain_x1_0",
                "osnet_ain_x0_75",
                "osnet_ain_x0_5",
                "osnet_ain_x0_25",
            ),
            osnet_module,
            family="cnn",
            default_recipe="cnn_reid",
            pretrained_source="imagenet",
        ),
        # Lightweight CNN / legacy backbones
        _lazy_variant(
            "mobilenetv2_x1_0",
            "boxmot.reid.backbones.mobilenetv2",
            family="cnn",
            default_recipe="cnn_reid",
            pretrained_source="imagenet",
        ),
        _lazy_variant(
            "mobilenetv2_x1_4",
            "boxmot.reid.backbones.mobilenetv2",
            family="cnn",
            default_recipe="cnn_reid",
            pretrained_source="imagenet",
        ),
        _lazy_variant(
            "mlfn",
            "boxmot.reid.backbones.mlfn",
            family="legacy",
            default_recipe="legacy_reid",
            pretrained_source="imagenet",
        ),
        _lazy_variant(
            "hacnn",
            "boxmot.reid.backbones.hacnn",
            attr="HACNN",
            family="legacy",
            default_recipe="legacy_reid",
            default_img_size=(160, 64),
        ),
        _lazy_variant(
            "lmbn_n",
            "boxmot.reid.backbones.lmbn_n",
            attr="LMBN_n",
            family="cnn",
            default_recipe="lmbn_reid",
            default_img_size=(384, 128),
        ),
        _lazy_variant(
            "lmbn_ain_n",
            "boxmot.reid.backbones.lmbn_ain_n",
            attr="LMBN_ain_n",
            family="cnn",
            default_recipe="lmbn_reid",
            default_img_size=(384, 128),
        ),
        # CSL-TinyViT family
        *_lazy_variants(
            (
                "csl_tinyvit_7m",
                "csl_tinyvit_11m",
                "csl_tinyvit_23m",
                "csl_tinyvit_small",
                "csl_tinyvit_normal",
                "csl_tinyvit_large",
                "csl_tinyvit_7m_lmbn",
                "csl_tinyvit_11m_lmbn",
                "csl_tinyvit_23m_lmbn",
                "csl_tinyvit_lmbn",
            ),
            csl_module,
            family="transformer",
            default_recipe="transformer_reid",
            default_img_size=(384, 128),
            supports_layer_decay=True,
            supports_drop_path=True,
            pretrained_source="TinyViT model zoo",
        ),
        # MobileNetV4 family
        *_lazy_variants(
            (
                "mobilenetv4_conv_small",
                "mobilenetv4_conv_medium",
                "mobilenetv4_conv_large",
                "mobilenetv4_hybrid_medium",
                "mobilenetv4_hybrid_large",
            ),
            mobilenetv4_module,
            family="hybrid",
            default_recipe="hybrid_reid",
            default_img_size=(384, 128),
            supports_drop_path=True,
            pretrained_source="timm",
        ),
    ):
        _register_lazy_backbone(import_spec)


_register_builtin_backbones()
