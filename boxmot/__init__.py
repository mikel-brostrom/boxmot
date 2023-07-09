__version__ = '10.0.15'

from pathlib import Path

from boxmot.trackers import StrongSORT
from boxmot.trackers import OCSort as OCSORT
from boxmot.trackers import BYTETracker
from boxmot.trackers import BoTSORT
from boxmot.trackers import DeepOCSort as DeepOCSORT
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

from boxmot.tracker_zoo import create_tracker, get_tracker_config


__all__ = '__version__',\
          'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT', 'DeepOCSORT'
