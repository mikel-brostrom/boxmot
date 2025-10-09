from boxmot import (
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    OcSort,
    StrongSort,
    HybridSort,
)

# Define tracker metadata as a single source of truth
TRACKERS = {
    "botsort":    {"class": BotSort,    "motion_only": False, "per_class": True},
    "deepocsort": {"class": DeepOcSort, "motion_only": False, "per_class": True},
    "strongsort": {"class": StrongSort, "motion_only": False, "per_class": False},
    "boosttrack": {"class": BoostTrack, "motion_only": False, "per_class": True},
    "hybridsort": {"class": HybridSort, "motion_only": False, "per_class": True},
    "ocsort":     {"class": OcSort,     "motion_only": True,  "per_class": True},
    "bytetrack":  {"class": ByteTrack,  "motion_only": True,  "per_class": True},
}

# Derive all lists from the metadata
MOTION_N_APPEARANCE_TRACKING_NAMES = [k for k, v in TRACKERS.items() if not v["motion_only"]]
MOTION_ONLY_TRACKING_NAMES = [k for k, v in TRACKERS.items() if v["motion_only"]]

MOTION_N_APPEARANCE_TRACKING_METHODS = [v["class"] for v in TRACKERS.values() if not v["motion_only"]]
MOTION_ONLY_TRACKING_METHODS = [v["class"] for v in TRACKERS.values() if v["motion_only"]]

ALL_TRACKERS = list(TRACKERS.keys())
PER_CLASS_TRACKERS = [k for k, v in TRACKERS.items() if v["per_class"]]