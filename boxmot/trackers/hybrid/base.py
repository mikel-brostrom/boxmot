"""Base class for hybrid trackers (bounding box + segmentation mask).

Hybrid trackers use both bounding boxes and segmentation masks for
association and state estimation (e.g., SAM2MOT-style approaches).
Masks are optional — the tracker can fall back to bbox-only mode.
"""

import numpy as np

from boxmot.trackers.base import BaseTracker


class HybridBaseTracker(BaseTracker):
    """Base class for trackers that use both bboxes and masks.

    Unlike MaskBaseTracker, masks are optional here. When provided,
    the tracker can use mask IoU for association alongside bbox IoU.
    Subclasses should return a tuple (tracks_array, output_masks) from
    ``_update_impl`` when masks are available.
    """

    supports_masks = True
