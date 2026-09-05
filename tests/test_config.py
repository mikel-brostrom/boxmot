from boxmot import (
    SFSORT,
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    HybridSort,
    OccluBoost,
    OcSort,
    StrongSort,
)

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
