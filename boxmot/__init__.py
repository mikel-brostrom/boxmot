# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

__version__ = '17.0.0'

from boxmot.engine.results import track
from boxmot.model import (
    BoxMOT,
    ExportResults,
    TrackEvalMetrics,
    TrackResults,
    TuneResults,
    TuneTrialResult,
    boxmot,
)
from boxmot.reid.core import ReID
from boxmot.trackers.boosttrack.boosttrack import BoostTrack
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.hybridsort.hybridsort import HybridSort
from boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.trackers.sfsort.sfsort import SFSORT
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

TRACKERS = [
    "bytetrack",
    "botsort",
    "strongsort",
    "ocsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "sfsort",
]

__all__ = (
    "__version__",
    "boxmot",
    "BoxMOT",
    "TrackResults",
    "ExportResults",
    "TrackEvalMetrics",
    "TuneResults",
    "TuneTrialResult",
    "StrongSort",
    "OcSort",
    "ByteTrack",
    "BotSort",
    "DeepOcSort",
    "HybridSort",
    "BoostTrack",
    "SFSORT",
    "ReID",
    "track",
    "create_tracker",
    "get_tracker_config",
    "gsi",
)
