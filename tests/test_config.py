from boxmot.trackers import OccluBoost
from boxmot.trackers.bbox.boosttrack import BoostTrack
from boxmot.trackers.bbox.botsort import BotSort
from boxmot.trackers.bbox.bytetrack import ByteTrack
from boxmot.trackers.bbox.deepocsort import DeepOcSort
from boxmot.trackers.bbox.hybridsort import HybridSort
from boxmot.trackers.bbox.ocsort import OcSort
from boxmot.trackers.bbox.sfsort import SFSORT
from boxmot.trackers.bbox.strongsort import StrongSort

MOTION_N_APPEARANCE_TRACKING_NAMES = [
    "botsort",
    "deepocsort",
    "strongsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
MOTION_ONLY_TRACKING_NAMES = ["ocsort", "bytetrack", "sfsort"]

MOTION_N_APPEARANCE_TRACKING_METHODS = [StrongSort, BotSort, DeepOcSort, BoostTrack, OccluBoost, HybridSort]
MOTION_ONLY_TRACKING_METHODS = [OcSort, ByteTrack, SFSORT]

ALL_TRACKERS = [
    "botsort",
    "deepocsort",
    "ocsort",
    "bytetrack",
    "sfsort",
    "strongsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
PER_CLASS_TRACKERS = [
    "botsort",
    "deepocsort",
    "ocsort",
    "bytetrack",
    "sfsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
