"""BoxMOT package metadata and lazy public API exports."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("boxmot")
except PackageNotFoundError:
    # Source trees can be imported before the project is installed. Release
    # versions live exclusively in pyproject.toml and are exposed through the
    # installed distribution metadata.
    __version__ = "0+unknown"

_EXPORTS = {
    "BoxMOT": ("boxmot.pipeline", "BoxMOT"),
    "Detector": ("boxmot.models.detector", "Detector"),
    "ReIDModel": ("boxmot.models.reid", "ReIDModel"),
}

__all__ = tuple(_EXPORTS)


if TYPE_CHECKING:
    from boxmot.models.detector import Detector as Detector
    from boxmot.models.reid import ReIDModel as ReIDModel
    from boxmot.pipeline import BoxMOT as BoxMOT


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'boxmot' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
