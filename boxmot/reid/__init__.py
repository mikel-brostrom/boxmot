"""Public appearance-encoder contracts, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.reid.protocols import AppearanceEncoder, EncoderRequirements
    from boxmot.reid.specs import ReIDEncoderSpec

__all__ = ("AppearanceEncoder", "EncoderRequirements", "ReIDEncoderSpec", "create_reid_encoder")

_EXPORTS = {
    "AppearanceEncoder": ("boxmot.reid.protocols", "AppearanceEncoder"),
    "EncoderRequirements": ("boxmot.reid.protocols", "EncoderRequirements"),
    "ReIDEncoderSpec": ("boxmot.reid.specs", "ReIDEncoderSpec"),
    "create_reid_encoder": ("boxmot.reid.factory", "create_reid_encoder"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
