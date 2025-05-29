from boxmot import (
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    OcSort,
    StrongSort,
)

MOTION_N_APPEARANCE_TRACKING_NAMES = [
    "botsort",
    "deepocsort",
    "strongsort",
    "boosttrack",
]
MOTION_ONLY_TRACKING_NAMES = ["ocsort", "bytetrack"]

MOTION_N_APPEARANCE_TRACKING_METHODS = [StrongSort, BotSort, DeepOcSort, BoostTrack]
MOTION_ONLY_TRACKING_METHODS = [OcSort, ByteTrack]

ALL_TRACKERS = [
    "botsort",
    "deepocsort",
    "ocsort",
    "bytetrack",
    "strongsort",
    "boosttrack",
]
PER_CLASS_TRACKERS = ["botsort", "deepocsort", "ocsort", "bytetrack"]
