"""BoxMOT package metadata and lazy public API exports."""

from importlib import import_module
from typing import TYPE_CHECKING

from boxmot.trackers.common.manifest import _TRACKER_MANIFEST

__version__ = "25.0.0"

_EXPORTS = {
    "create_tracker": ("boxmot.trackers.common.factory", "create_tracker"),
    **{
        entry.class_path.rsplit(".", 1)[-1]: tuple(entry.class_path.rsplit(".", 1))
        for entry in _TRACKER_MANIFEST.values()
    },
}

__all__ = ("__version__", *_EXPORTS)


if TYPE_CHECKING:
    from boxmot.trackers.boosttrack.tracker import BoostTrack as BoostTrack
    from boxmot.trackers.botsort.tracker import BotSort as BotSort
    from boxmot.trackers.bytetrack.tracker import ByteTrack as ByteTrack
    from boxmot.trackers.common.factory import create_tracker as create_tracker
    from boxmot.trackers.deepocsort.tracker import DeepOcSort as DeepOcSort
    from boxmot.trackers.eagermot.tracker import EagerMot as EagerMot
    from boxmot.trackers.hybridsort.tracker import HybridSort as HybridSort
    from boxmot.trackers.maf_hda.tracker import MafHda as MafHda
    from boxmot.trackers.occluboost.tracker import OccluBoost as OccluBoost
    from boxmot.trackers.ocsort.tracker import OcSort as OcSort
    from boxmot.trackers.sfsort.tracker import SFSORT as SFSORT
    from boxmot.trackers.strongsort.tracker import StrongSort as StrongSort


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'boxmot' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
