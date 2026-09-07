"""Independent segmentation component contracts and factories."""

from .factory import create_segmentor
from .protocols import Segmentor
from .specs import SegmentorSpec

__all__ = ("Segmentor", "SegmentorSpec", "create_segmentor")
