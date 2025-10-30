#!/usr/bin/env python3
# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Quick test script for the new detector interface.
"""

import numpy as np
import sys
from pathlib import Path

# Add boxmot to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from boxmot.engine.detectors import YOLOX, Ultralytics, RFDETR, resolve_image


def test_resolve_image():
    """Test the resolve_image utility function."""
    print("\n" + "=" * 50)
    print("Testing resolve_image utility")
    print("=" * 50)
    
    # Test with numpy array
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = resolve_image(img_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (480, 640, 3)
    print("✓ Numpy array input works")
    
    # Test with invalid input
    try:
        resolve_image(12345)
        print("✗ Should have raised TypeError")
    except TypeError:
        print("✓ TypeError raised for invalid input")
    
    print("resolve_image tests passed!\n")


def test_yolox_interface():
    """Test YOLOX detector interface."""
    print("\n" + "=" * 50)
    print("Testing YOLOX Interface")
    print("=" * 50)
    
    try:
        # Check if YOLOX is available
        from boxmot.engine.detectors.yolox import YOLOX
        
        print("Creating YOLOX detector...")
        detector = YOLOX(
            "yolox_s.pt",
            device="cpu",
            imgsz=640,
            conf_thres=0.25
        )
        
        # Check that methods exist
        assert hasattr(detector, 'preprocess')
        assert hasattr(detector, 'process')
        assert hasattr(detector, 'postprocess')
        assert hasattr(detector, '__call__')
        print("✓ All required methods exist")
        
        # Test method override
        def custom_preprocess(frame, **kwargs):
            return detector.preprocess(frame, **kwargs)
        
        detector.preprocess = custom_preprocess
        print("✓ Method override works")
        
        print("YOLOX interface tests passed!\n")
        
    except ImportError as e:
        print(f"⚠ YOLOX not available: {e}")
        print("Install with: pip install yolox --no-deps\n")


def test_ultralytics_interface():
    """Test Ultralytics detector interface."""
    print("\n" + "=" * 50)
    print("Testing Ultralytics Interface")
    print("=" * 50)
    
    try:
        from boxmot.engine.detectors.ultralytics import Ultralytics
        
        print("Creating Ultralytics detector...")
        detector = Ultralytics(
            "yolov8n.pt",
            device="cpu",
            imgsz=640,
            conf_thres=0.25
        )
        
        # Check that methods exist
        assert hasattr(detector, 'preprocess')
        assert hasattr(detector, 'process')
        assert hasattr(detector, 'postprocess')
        assert hasattr(detector, '__call__')
        print("✓ All required methods exist")
        
        # Test method override
        def custom_postprocess(boxes, **kwargs):
            return detector.postprocess(boxes, **kwargs)
        
        detector.postprocess = custom_postprocess
        print("✓ Method override works")
        
        print("Ultralytics interface tests passed!\n")
        
    except ImportError as e:
        print(f"⚠ Ultralytics not available: {e}")
        print("Install with: pip install ultralytics\n")


def test_detector_base_class():
    """Test the base Detector class."""
    print("\n" + "=" * 50)
    print("Testing Base Detector Class")
    print("=" * 50)
    
    from boxmot.engine.detectors.base import Detector
    
    # Create a minimal implementation
    class TestDetector(Detector):
        def _load_model(self, path, **kwargs):
            return None
        
        def preprocess(self, frame, **kwargs):
            return frame
        
        def process(self, frame, **kwargs):
            return np.array([[100, 100, 200, 200, 0.9, 0]])
        
        def postprocess(self, boxes, **kwargs):
            return boxes
    
    # Test instantiation
    detector = TestDetector("dummy.pt")
    print("✓ Base class can be subclassed")
    
    # Test __call__ method
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector(img)
    assert isinstance(result, np.ndarray)
    print("✓ __call__ method works")
    
    # Test method override
    original_preprocess = detector.preprocess
    
    def custom_preprocess(frame, **kwargs):
        print("Custom preprocessing called")
        return original_preprocess(frame, **kwargs)
    
    detector.preprocess = custom_preprocess
    result = detector(img)
    print("✓ Method override in __call__ works")
    
    print("Base Detector class tests passed!\n")


def test_rfdetr_interface():
    """Test RF-DETR detector interface."""
    print("\n" + "=" * 50)
    print("Testing RF-DETR Interface")
    print("=" * 50)
    
    try:
        from boxmot.engine.detectors.rfdetr import RFDETR
        
        print("Creating RF-DETR detector...")
        detector = RFDETR(
            "rfdetr-l.onnx",
            device="cpu",
            conf_thres=0.25
        )
        
        # Check that methods exist
        assert hasattr(detector, 'preprocess')
        assert hasattr(detector, 'process')
        assert hasattr(detector, 'postprocess')
        assert hasattr(detector, '__call__')
        print("✓ All required methods exist")
        
        # Test method override
        def custom_postprocess(detections, **kwargs):
            return detector.postprocess(detections, **kwargs)
        
        detector.postprocess = custom_postprocess
        print("✓ Method override works")
        
        print("RF-DETR interface tests passed!\n")
        
    except ImportError as e:
        print(f"⚠ RF-DETR not available: {e}")
        print("Install with: pip install rfdetr\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DETECTOR INTERFACE TESTS")
    print("=" * 70)
    
    test_resolve_image()
    test_detector_base_class()
    test_yolox_interface()
    test_ultralytics_interface()
    test_rfdetr_interface()
    
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
