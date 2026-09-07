"""Independent object-detection component contracts and factory."""

from .factory import create_detector
from .protocols import Detector, DetectorCapabilities
from .specs import DetectorSpec

__all__ = ("Detector", "DetectorCapabilities", "DetectorSpec", "create_detector")
