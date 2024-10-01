from boxmot import (
    StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, ImprAssocTrack, get_tracker_config, create_tracker,
)

MOTION_N_APPEARANCE_TRACKING_NAMES = ['botsort', 'deepocsort', 'strongsort', 'imprassoc']
MOTION_ONLY_TRACKING_NAMES = ['ocsort', 'bytetrack']

MOTION_N_APPEARANCE_TRACKING_METHODS=[StrongSort, BotSort, DeepOcSort, ImprAssocTrack]
MOTION_ONLY_TRACKING_METHODS=[OcSort, ByteTrack]

ALL_TRACKERS = ['botsort', 'deepocsort', 'ocsort', 'bytetrack', 'strongsort', 'imprassoc']
PER_CLASS_TRACKERS = ['botsort', 'deepocsort', 'ocsort', 'bytetrack', 'imprassoc']