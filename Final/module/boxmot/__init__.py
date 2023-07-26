__version__ = "10.0.16"

from module.boxmot.tracker_zoo import create_tracker, get_tracker_config
from module.boxmot.trackers.botsort.bot_sort import BoTSORT
from module.boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from module.boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from module.boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
from module.boxmot.trackers.strongsort.strong_sort import StrongSORT

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT",
           "create_tracker", "get_tracker_config")
