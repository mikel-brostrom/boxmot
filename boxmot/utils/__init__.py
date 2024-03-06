# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
BOXMOT = ROOT / "boxmot"
EXAMPLES = ROOT / "tracking"
EXPERIMENTATION = ROOT / "experimentation"
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
        self.update_method = method

    def __get__(self, instance, owner):
        # This makes PerClassDecorator a non-data descriptor that binds the method to the instance
        def wrapper(*args, **kwargs):
            # Unpack arguments for clarity
            detections, image = args
            
            if instance.per_class is True and detections.size != 0:
                # Organize detections by class ID for per-class processing
                detections_by_class = {
                    class_id: np.array([det for det in detections if det[5] == class_id])
                    for class_id in set(det[5] for det in detections)
                }

                # Detect classes in the current frame and active trackers
                detected_classes = set(detections_by_class.keys())
                active_classes = set(tracker.cls for tracker in instance.active_tracks)
                
                # Determine relevant classes for processing
                relevant_classes = active_classes.union(detected_classes)

                # Initialize an array to store modified detections
                modified_detections = np.empty(shape=(0, 8))
                for class_id in relevant_classes:
                    # Process detections class-by-class
                    current_class_detections = detections_by_class.get(int(class_id), np.empty((0, 6)))
                    logger.debug(f"Processing class {int(class_id)}: {current_class_detections.shape}")
                    
                    # Update detections using the decorated method
                    updated_dets = self.update_method(instance, current_class_detections, image)
                    if updated_dets.size != 0:
                        modified_detections = np.append(modified_detections, updated_dets, axis=0)

                logger.debug(f"Per-class update result: {modified_detections.shape}")
            else:
                # Process all detections at once if per_class is False or detections are empty
                modified_detections = self.update_method(instance, detections, image)

            return modified_detections

        return wrapper