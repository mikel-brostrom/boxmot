"""Hybrid trackers (bounding box + segmentation mask).

Trackers in this subpackage use both bounding boxes and segmentation masks
for association and state estimation (e.g., SAM2MOT-style approaches).
"""

from boxmot.trackers.hybrid.base import HybridBaseTracker
from boxmot.trackers.hybrid.sam2mot.sam2mot import Sam2Mot

__all__ = ["HybridBaseTracker", "Sam2Mot"]
