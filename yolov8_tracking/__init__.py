__version__ = '10.0'

from yolov8_tracking.trackers.strongsort.strong_sort import StrongSORT
from yolov8_tracking.trackers.ocsort.ocsort import OCSort as OCSORT
from yolov8_tracking.trackers.bytetrack.byte_tracker import BYTETracker
from yolov8_tracking.trackers.botsort.bot_sort import BoTSORT
from yolov8_tracking.trackers.deepocsort.ocsort import OCSort as DeepOCSORT

from yolov8_tracking.trackers.deep.reid_multibackend import ReIDDetectMultiBackend


__all__ = '__version__', 'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT',\
          'DeepOCSORT'  # allow simpler import