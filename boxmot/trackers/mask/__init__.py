"""Mask-based trackers.

Trackers in this subpackage use segmentation masks as their primary
representation for tracking targets.
"""

from boxmot.trackers.mask.base import MaskBaseTracker

__all__ = ["MaskBaseTracker"]
