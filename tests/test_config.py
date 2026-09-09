from boxmot import (
    SFSORT,
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    EagerMot,
    HybridSort,
    MafHda,
    OccluBoost,
    OcSort,
    Sam2Mot,
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
MULTIMODAL_TRACKING_NAMES = ["sam2mot", "maf_hda", "eagermot"]
MULTIMODAL_TRACKING_METHODS = [Sam2Mot, MafHda, EagerMot]

ALL_TRACKERS = [
    *MULTIMODAL_TRACKING_NAMES,
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
    *MULTIMODAL_TRACKING_NAMES,
    "botsort",
    "deepocsort",
    "ocsort",
    "bytetrack",
    "sfsort",
    "boosttrack",
    "occluboost",
    "hybridsort",
]
