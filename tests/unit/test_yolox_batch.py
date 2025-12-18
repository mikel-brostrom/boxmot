#!/usr/bin/env python3
"""
Test YOLOX batch processing functionality.
"""

import numpy as np
from pathlib import Path

print("=" * 70)
print("YOLOX BATCH PROCESSING TEST")
print("=" * 70)
print()

# Test 1: Single image processing (baseline)
print("Test 1: Single image processing...")
try:
    from boxmot.detectors import YoloX
    
    model_path = "yolox_s.pt"
    if not Path(model_path).exists():
        print(f"⚠ Model file {model_path} not found, skipping test")
    else:
        detector = YoloX(model=model_path, device="cpu")
        
        # Create a dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process single image
        result = detector(image)
        print(f"  - Single image input shape: {image.shape}")
        print(f"  - Single image output shape: {result.shape}")
        print(f"  - Output format: [N, 6] = [{result.shape[0]}, {result.shape[1]}]")
        print("✓ Single image processing works")
except Exception as e:
    print(f"✗ Single image test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Batch processing
print("Test 2: Batch processing (list of images)...")
try:
    from boxmot.detectors import YoloX
    
    model_path = "yolox_s.pt"
    if not Path(model_path).exists():
        print(f"⚠ Model file {model_path} not found, skipping test")
    else:
        detector = YoloX(model=model_path, device="cpu")
        
        # Create batch of dummy images (different sizes to test flexibility)
        batch_size = 3
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8),
        ]
        
        print(f"  - Batch size: {batch_size}")
        print(f"  - Image shapes: {[img.shape for img in images]}")
        
        # Process batch
        results = detector(images)
        
        if isinstance(results, list):
            print(f"  - Batch output: list of {len(results)} arrays")
            for i, result in enumerate(results):
                print(f"    - Image {i+1} detections: {result.shape}")
            print("✓ Batch processing works (returns list)")
        else:
            print(f"  - Unexpected output type: {type(results)}")
            print("⚠ Expected list output for batch processing")
            
except Exception as e:
    print(f"✗ Batch processing test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Verify preprocessing handles batches correctly
print("Test 3: Testing preprocess method directly...")
try:
    from boxmot.detectors import YoloX
    
    model_path = "yolox_s.pt"
    if not Path(model_path).exists():
        print(f"⚠ Model file {model_path} not found, skipping test")
    else:
        detector = YoloX(model=model_path, device="cpu")
        
        # Test single image preprocessing
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor_single = detector.preprocess(image)
        print(f"  - Single image tensor shape: {tensor_single.shape}")
        print(f"    Expected: [3, H, W], Got: {list(tensor_single.shape)}")
        
        # Test batch preprocessing
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        ]
        tensor_batch = detector.preprocess(images)
        print(f"  - Batch tensor shape: {tensor_batch.shape}")
        print(f"    Expected: [B, 3, H, W], Got: {list(tensor_batch.shape)}")
        
        if len(tensor_batch.shape) == 4 and tensor_batch.shape[0] == 2:
            print("✓ Preprocess handles batches correctly")
        else:
            print("⚠ Unexpected batch tensor shape")
            
except Exception as e:
    print(f"✗ Preprocess test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("BATCH PROCESSING TEST COMPLETED")
print("=" * 70)
