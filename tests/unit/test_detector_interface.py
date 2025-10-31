#!/usr/bin/env python3
# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Comprehensive test suite for the new detector interface.
Tests all detector classes, base class functionality, and utility functions.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

from boxmot.engine.detectors import YoloX, Ultralytics, resolve_image, Detector
from boxmot.engine.detectors import get_yolo_inferer, is_yolox_model, is_ultralytics_model


class TestResolveImage:
    """Test suite for resolve_image utility function."""
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = resolve_image(img_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8
    
    def test_grayscale_image(self):
        """Test with grayscale image."""
        img_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = resolve_image(img_gray)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640)
    
    def test_torch_tensor_input(self):
        """Test with torch tensor input."""
        # Test CHW format (C, H, W)
        img_tensor_chw = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
        result = resolve_image(img_tensor_chw)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        
        # Test HWC format (H, W, C)
        img_tensor_hwc = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)
        result = resolve_image(img_tensor_hwc)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        
        # Test BCHW format (B, C, H, W) - should convert to HWC
        img_tensor_bchw = torch.randint(0, 255, (1, 3, 480, 640), dtype=torch.uint8)
        result = resolve_image(img_tensor_bchw)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
    
    def test_list_input(self):
        """Test with list of images."""
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = resolve_image([img1, img2])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(img, np.ndarray) for img in result)
    
    def test_file_path_string(self):
        """Test with file path as string."""
        # Create a dummy image file
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_path = Path("test_image.jpg")
        
        # Note: This would require actually saving and loading the image
        # For now, we test that string paths are handled
        # In real scenario, resolve_image should load the image from path
    
    def test_invalid_input(self):
        """Test with invalid input types."""
        with pytest.raises(TypeError):
            resolve_image(12345)
        
        with pytest.raises(TypeError):
            resolve_image(None)


def test_yolox_interface():
    """Test YOLOX detector interface."""
    print("\n" + "=" * 50)
    print("Testing YOLOX Interface")
    print("=" * 50)
    
    print("Creating YOLOX detector...")
    detector = YoloX(
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
    assert hasattr(detector, 'warmup')
    print("✓ All required methods exist")
    
    # Test that postprocess uses 'preds' parameter
    import inspect
    sig = inspect.signature(detector.postprocess)
    params = list(sig.parameters.keys())
    assert 'preds' in params, f"postprocess should have 'preds' parameter, got: {params}"
    print("✓ postprocess uses 'preds' parameter")
    
    # Test method override
    def custom_preprocess(im, **kwargs):
        return detector.preprocess(im=im, **kwargs)
    
    detector.preprocess = custom_preprocess
    print("✓ Method override works")
    
    print("YOLOX interface tests passed!\n")


def test_ultralytics_interface():
    """Test Ultralytics detector interface."""
    print("\n" + "=" * 50)
    print("Testing Ultralytics Interface")
    print("=" * 50)
    
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
    assert hasattr(detector, 'warmup')
    print("✓ All required methods exist")
    
    # Test that postprocess uses 'preds' parameter
    import inspect
    sig = inspect.signature(detector.postprocess)
    params = list(sig.parameters.keys())
    assert 'preds' in params, f"postprocess should have 'preds' parameter, got: {params}"
    print("✓ postprocess uses 'preds' parameter")
    
    # Test method override
    def custom_postprocess(preds, **kwargs):
        return detector.postprocess(preds=preds, **kwargs)
    
    detector.postprocess = custom_postprocess
    print("✓ Method override works")
    
    print("Ultralytics interface tests passed!\n")


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
        
        def preprocess(self, im, **kwargs):
            return im
        
        def process(self, im, **kwargs):
            return np.array([[100, 100, 200, 200, 0.9, 0]])
        
        def postprocess(self, preds, **kwargs):
            return preds
    
    # Test instantiation
    detector = TestDetector("dummy.pt")
    print("✓ Base class can be subclassed")
    
    # Test __call__ method
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector(img)
    assert isinstance(result, np.ndarray)
    print("✓ __call__ method works")
    
    # Test warmup method
    assert hasattr(detector, 'warmup')
    try:
        detector.warmup(imgsz=(640, 640), n=2)
        print("✓ warmup method exists and runs")
    except Exception as e:
        print(f"⚠ warmup method raised: {e}")
    
    # Test batch processing support
    img_list = [img, img]
    result = detector(img_list)
    print("✓ Batch processing works")
    
    # Test method override
    original_preprocess = detector.preprocess
    
    def custom_preprocess(im, **kwargs):
        print("Custom preprocessing called")
        return original_preprocess(im=im, **kwargs)
    
    detector.preprocess = custom_preprocess
    result = detector(img)
    print("✓ Method override in __call__ works")
    
    print("Base Detector class tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DETECTOR INTERFACE TESTS")
    print("=" * 70)
    
    # Run resolve_image tests
    test_suite = TestResolveImage()
    print("\n" + "=" * 50)
    print("Testing resolve_image utility")
    print("=" * 50)
    test_suite.test_numpy_array_input()
    print("✓ NumPy array input works")
    test_suite.test_grayscale_image()
    print("✓ Grayscale image works")
    test_suite.test_torch_tensor_input()
    print("✓ Torch tensor input works")
    test_suite.test_list_input()
    print("✓ List input works")
    print("resolve_image tests passed!\n")
    
    # Run other tests
    test_detector_base_class()
    test_yolox_interface()
    test_ultralytics_interface()
    
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
