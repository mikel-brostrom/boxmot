"""BoxMOT package metadata and lazy public API exports."""

from importlib import import_module
from typing import TYPE_CHECKING

from boxmot._tracker_exports import _TRACKER_EXPORTS

__version__ = "25.0.0"

_EXPORTS = {
    "create_tracker": ("boxmot.trackers.factory", "create_tracker"),
    **_TRACKER_EXPORTS,
}

__all__ = ("__version__", *_EXPORTS)


if TYPE_CHECKING:
    from boxmot.trackers.box.boosttrack.tracker import BoostTrack as BoostTrack
    from boxmot.trackers.box.botsort.tracker import BotSort as BotSort
    from boxmot.trackers.box.bytetrack.tracker import ByteTrack as ByteTrack
    from boxmot.trackers.box.deepocsort.tracker import DeepOcSort as DeepOcSort
    from boxmot.trackers.box.hybridsort.tracker import HybridSort as HybridSort
    from boxmot.trackers.box.occluboost.tracker import OccluBoost as OccluBoost
    from boxmot.trackers.box.ocsort.tracker import OcSort as OcSort
    from boxmot.trackers.box.sfsort.tracker import SFSORT as SFSORT
    from boxmot.trackers.box.strongsort.tracker import StrongSort as StrongSort
    from boxmot.trackers.factory import create_tracker as create_tracker
    from boxmot.trackers.multimodal.sam2mot.tracker import Sam2Mot as Sam2Mot


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'boxmot' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
