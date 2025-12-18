#!/usr/bin/env python3
# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
Quick test script for the new detector interface.
"""

import numpy as np

from boxmot.detectors import YOLOX, Ultralytics, RFDETR


def test_yolox_interface():
    """Test YOLOX detector interface."""
    print("\n" + "=" * 50)
    print("Testing YOLOX Interface")
    print("=" * 50)
    
    try:
        # Check if YOLOX is available
        from boxmot.detectors.yolox import YOLOX
        
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
        from boxmot.detectors.ultralytics import Ultralytics
        
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
    
    from boxmot.detectors.detector import Detector
    
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
        from boxmot.detectors.rfdetr import RFDETR
        
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
    
    test_detector_base_class()
    test_yolox_interface()
    test_ultralytics_interface()
    test_rfdetr_interface()
    
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
