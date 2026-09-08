"""Independent object-detection contracts and factory, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .factory import create_detector
    from .protocols import Detector, DetectorCapabilities
    from .specs import DetectorSpec

__all__ = ("Detector", "DetectorCapabilities", "DetectorSpec", "create_detector")

_EXPORTS = {
    "Detector": ("boxmot.detectors.protocols", "Detector"),
    "DetectorCapabilities": ("boxmot.detectors.protocols", "DetectorCapabilities"),
    "DetectorSpec": ("boxmot.detectors.specs", "DetectorSpec"),
    "create_detector": ("boxmot.detectors.factory", "create_detector"),
}


def __getattr__(name: str) -> Any:
    """Resolve detector exports without loading runtimes for profile selection."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose public names without importing their implementations."""

    return sorted((*globals(), *__all__))
