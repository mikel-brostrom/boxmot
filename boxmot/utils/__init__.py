# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
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

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of BoxMOT multiprocessing threads


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
            embs = args[2] if len(args) > 2 else None
            
            if instance.per_class is True:

                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = instance.frame_count

                for i, cls_id in enumerate(range(self.nr_classes)):
 
                    if dets.size > 0:
                        class_dets = dets[dets[:, 5] == cls_id]
                    else:
                        class_dets = np.empty((0, 6))
                    logger.debug(f"Processing class {int(cls_id)}: {class_dets.shape}")

                    # activate the specific active tracks for this class id
                    instance.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # reset frame count for every class
                    instance.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = self.update(instance, dets=class_dets, img=im, embs=embs)

                    # save the updated active tracks
                    self.per_class_active_tracks[cls_id] = instance.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                                        
                # when all active tracks lists have been updated
                instance.per_class_active_tracks = self.per_class_active_tracks
                
                # increase frame count by 1
                instance.frame_count = frame_count + 1

                tracks = np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False or detections are empty
                tracks = self.update(instance, dets=dets, img=im, embs=embs)
            
            return tracks

        return wrapper