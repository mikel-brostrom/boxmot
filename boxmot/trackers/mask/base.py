"""Base class for mask-native trackers.

Mask trackers use segmentation masks as their primary representation.
They require masks to be provided in the update() call and produce
tracked masks as output.
"""

import numpy as np

from boxmot.trackers.base import BaseTracker


class MaskBaseTracker(BaseTracker):
    """Base class for trackers that use segmentation masks as primary input.

    Subclasses must implement ``_update_impl`` and should return a tuple
    of (tracks_array, output_masks) where output_masks has shape (M, H, W).
    """

    supports_masks = True

    def _preprocess_masks(self, dets: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """Override to require masks for mask-native trackers."""
        if masks is None:
            raise ValueError(
                f"{self.__class__.__name__} requires segmentation masks. "
                "Pass masks=<ndarray of shape (N, H, W)> to update()."
            )
        return super()._preprocess_masks(dets, masks)
