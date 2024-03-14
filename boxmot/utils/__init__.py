# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
DATA = ROOT / 'data'
BOXMOT = ROOT / "boxmot"
EXAMPLES = ROOT / "tracking"
TRACKER_CONFIGS = ROOT / "boxmot" / "configs"
WEIGHTS = ROOT / "tracking" / "weights"
REQUIREMENTS = ROOT / "requirements.txt"

# global logger
from loguru import logger

logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")


class PerClassDecorator:
    def __init__(self, method):
        # Store the method that will be decorated
        self.update = method
        self.nr_classes = 80
        self.per_class_active_tracks = {}
        for i in range(self.nr_classes):
            self.per_class_active_tracks[i] = []

    def __get__(self, instance, owner):
        # This makes PerClassDecorator a non-data descriptor that binds the method to the instance
        def wrapper(*args, **kwargs):
            # Unpack arguments for clarity
            args = list(args)
            dets = args[0]
            im = args[1]
            
            if instance.per_class is True:

                # Initialize an array to store the tracks for each class
                per_class_tracks = []

                for cls_id in range(self.nr_classes):
                    if dets.size > 0:
                        class_dets = dets[dets[:, 5] == cls_id]
                    else:
                        class_dets = np.empty((0, 6))
                    logger.debug(f"Processing class {int(cls_id)}: {class_dets.shape}")

                    # activate the specific active tracks for this class id
                    instance.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # Update detections using the decorated method
                    tracks = self.update(instance, class_dets, im)

                    # save the updated active tracks
                    self.per_class_active_tracks[cls_id] = instance.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                
                # when all active tracks lists have been updated
                instance.per_class_active_tracks = self.per_class_active_tracks

                tracks = np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False or detections are empty
                tracks = self.update(instance, dets, im)
            
            return tracks

        return wrapper