# Detector Interface Documentation

## Overview

The BoxMOT detector interface provides a standardized, modular way to use object detection models. The interface follows a simple three-step pipeline: **preprocess → process → postprocess**, with the ability to override any step for custom behavior.

## Key Features

- 🔧 **Modular Design**: Override preprocessing, processing, or postprocessing independently
- 🎯 **Unified Interface**: Same API across all detector types (YOLOX, Ultralytics YOLO, etc.)
- 🖼️ **Flexible Input**: Accept both image file paths and numpy arrays
- 📦 **Easy Integration**: Drop-in replacement with minimal code changes

## Architecture

```python
class Detector:
    def __init__(self, path: str):
        self.path = path
        self.model = self._load_model(path)
    
    def _load_model(self, path: str):
        # Load model weights
        return model
    
    def preprocess(self, frame: np.ndarray, **kwargs):
        # Prepare image for inference
        raise NotImplementedError()
    
    def process(self, frame, **kwargs):
        # Run model inference
        raise NotImplementedError()
    
    def postprocess(self, boxes, **kwargs):
        # Process raw detections
        raise NotImplementedError()
    
    def __call__(self, image: Union[np.ndarray, str], **kwargs):
        image = resolve_image(image)
        frame = self.preprocess(image, **kwargs)
        boxes = self.process(frame, **kwargs)
        boxes = self.postprocess(boxes, **kwargs)
        return boxes
```

## Installation

The detector interface is part of BoxMOT. Install the required dependencies:

```bash
# For YOLOX
pip install yolox --no-deps
pip install tabulate thop

# For Ultralytics (YOLOv8, v9, v10, v11)
pip install ultralytics
```

## Basic Usage

### YOLOX Example

```python
from boxmot.engine.detectors import YOLOX

# Initialize detector
detector = YOLOX("yolox_s.pt", device="cpu", conf_thres=0.5)

# Detect from file path
boxes = detector("path/to/image.jpg")

# Detect from numpy array
import cv2
image = cv2.imread("path/to/image.jpg")
boxes = detector(image)

# Output format: [N, 6] array with [x1, y1, x2, y2, confidence, class]
print(f"Detected {len(boxes)} objects")
```

### Ultralytics Example

```python
from boxmot.engine.detectors import Ultralytics

# Initialize detector
detector = Ultralytics("yolov8n.pt", device="cuda", imgsz=1280)

# Run detection
boxes = detector("image.jpg")
```

### RF-DETR Example

```python
from boxmot.engine.detectors import RFDETR

# Initialize detector
detector = RFDETR("rfdetr-l.onnx", device="cpu", conf_thres=0.5)

# Run detection
boxes = detector("image.jpg")
```

## Advanced Usage

### Override Preprocessing

```python
from boxmot.engine.detectors import YOLOX
import cv2

detector = YOLOX("yolox_s.pt")

# Define custom preprocessing
def custom_preprocess(frame, **kwargs):
    """Apply custom image preprocessing."""
    # Apply Gaussian blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Call original preprocessing
    # Note: For YOLOX, this handles resizing, normalization, etc.
    return detector.preprocess.__wrapped__(detector, frame, **kwargs)

# Replace preprocessing method
detector.preprocess = custom_preprocess

# Detection now uses custom preprocessing
boxes = detector("image.jpg")
```

### Override Postprocessing

```python
detector = YOLOX("yolox_s.pt")

# Store original postprocessing
original_postprocess = detector.postprocess

# Define custom postprocessing
def custom_postprocess(boxes, **kwargs):
    """Filter detections by minimum area."""
    # Apply original postprocessing
    boxes = original_postprocess(boxes, **kwargs)
    
    # Additional filtering
    if len(boxes) > 0:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        boxes = boxes[areas > 500]  # Keep only large boxes
    
    return boxes

detector.postprocess = custom_postprocess
boxes = detector("image.jpg")
```

### Override Inference

```python
import time

detector = Ultralytics("yolov8n.pt")

# Store original process method
original_process = detector.process

# Add timing to inference
def timed_process(frame, **kwargs):
    """Run inference with timing."""
    start = time.time()
    result = original_process(frame, **kwargs)
    elapsed = time.time() - start
    print(f"Inference took {elapsed:.4f}s")
    return result

detector.process = timed_process
boxes = detector("image.jpg")
```

### Complete Custom Pipeline

```python
detector = YOLOX("yolox_s.pt")

# Save originals
orig_pre = detector.preprocess
orig_proc = detector.process
orig_post = detector.postprocess

# Custom preprocess
def my_preprocess(frame, **kwargs):
    print("Step 1: Custom preprocessing")
    # Your custom logic here
    return orig_pre(frame, **kwargs)

# Custom process
def my_process(frame, **kwargs):
    print("Step 2: Custom inference")
    # Your custom logic here
    return orig_proc(frame, **kwargs)

# Custom postprocess
def my_postprocess(boxes, **kwargs):
    print("Step 3: Custom postprocessing")
    # Your custom logic here
    processed = orig_post(boxes, **kwargs)
    
    # Filter high-confidence only
    if len(processed) > 0:
        processed = processed[processed[:, 4] > 0.7]
    
    return processed

# Replace all methods
detector.preprocess = my_preprocess
detector.process = my_process
detector.postprocess = my_postprocess

# Run with custom pipeline
boxes = detector("image.jpg")
```

## Supported Detectors

### YOLOX

```python
from boxmot.engine.detectors import YOLOX

detector = YOLOX(
    path="yolox_s.pt",           # Model weights path
    device="cpu",                 # Device: 'cpu', 'cuda', 'mps'
    imgsz=640,                    # Input size: int or [w, h]
    conf_thres=0.25,              # Confidence threshold
    iou_thres=0.45,               # NMS IoU threshold
    agnostic_nms=False,           # Class-agnostic NMS
    classes=None,                 # Filter by class IDs
)
```

**Supported Models:**
- yolox_n.pt
- yolox_s.pt
- yolox_m.pt
- yolox_l.pt
- yolox_x.pt
- yolox_x_MOT17_ablation.pt
- yolox_x_MOT20_ablation.pt
- yolox_x_dancetrack_ablation.pt

### Ultralytics

```python
from boxmot.engine.detectors import Ultralytics

detector = Ultralytics(
    path="yolov8n.pt",            # Model weights path
    device="cpu",                 # Device: 'cpu', 'cuda', 'mps'
    imgsz=640,                    # Input size
    conf_thres=0.25,              # Confidence threshold
    iou_thres=0.45,               # NMS IoU threshold
    agnostic_nms=False,           # Class-agnostic NMS
    classes=None,                 # Filter by class IDs
    verbose=False,                # Print verbose output
)
```

**Supported Models:**
- YOLOv8 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- YOLOv9
- YOLOv10
- YOLO11 (yolo11n.pt, etc.)
- RT-DETR

### RF-DETR

```python
from boxmot.engine.detectors import RFDETR

detector = RFDETR(
    path="rfdetr-l.onnx",         # Model ONNX path
    device="cpu",                 # Device: 'cpu' or 'cuda'
    conf_thres=0.25,              # Confidence threshold
    classes=None,                 # Filter by class IDs
)
```

**Supported Models:**
- rfdetr-l.onnx
- rfdetr-x.onnx
- Any RF-DETR ONNX model

## Output Format

All detectors return detections in a consistent format:

```python
boxes = detector("image.jpg")
# Shape: [N, 6]
# Format: [x1, y1, x2, y2, confidence, class_id]
```

Where:
- `N`: Number of detected objects
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- `confidence`: Detection confidence score (0-1)
- `class_id`: Class index (0-79 for COCO classes)

## Utility Functions

### resolve_image

Convert image input to numpy array:

```python
from boxmot.engine.detectors import resolve_image

# From file path
image = resolve_image("path/to/image.jpg")

# From numpy array (returns as-is)
import cv2
img = cv2.imread("image.jpg")
image = resolve_image(img)  # Returns img unchanged
```

## Migration Guide

### From Old Interface

**Old way:**
```python
from boxmot.engine.detectors import get_yolo_inferer

m = get_yolo_inferer("yolox_s.pt")
yolo_model = m(
    model="yolox_s.pt",
    device=device,
    args=args,
)
```

**New way:**
```python
from boxmot.engine.detectors import YOLOX

detector = YOLOX(
    "yolox_s.pt",
    device="cpu",
    conf_thres=0.25,
    iou_thres=0.45
)
boxes = detector("image.jpg")
```

## Best Practices

1. **Store Original Methods**: When overriding, save the original method if you need to call it
   ```python
   original = detector.preprocess
   detector.preprocess = lambda x, **kw: custom_logic(original(x, **kw))
   ```

2. **Use **kwargs**: Always include `**kwargs` in custom methods for forward compatibility
   ```python
   def my_preprocess(frame, **kwargs):  # ✓ Good
       ...
   
   def my_preprocess(frame):  # ✗ May break with updates
       ...
   ```

3. **Check Array Shapes**: Verify input/output shapes match expected formats
   ```python
   def my_postprocess(boxes, **kwargs):
       result = original_postprocess(boxes, **kwargs)
       assert result.shape[1] == 6, "Expected [N, 6] format"
       return result
   ```

4. **Handle Empty Detections**: Always check for empty results
   ```python
   def my_postprocess(boxes, **kwargs):
       boxes = original_postprocess(boxes, **kwargs)
       if len(boxes) == 0:
           return boxes  # No detections, return early
       # Process boxes...
       return boxes
   ```

## Examples

See `examples/detector_usage_example.py` for complete working examples including:
- Basic usage
- Custom preprocessing
- Custom postprocessing
- Custom inference
- Complete pipeline override
- Multiple detector types

## Testing

Run the test suite:

```bash
python test_detector_interface.py
```

This validates:
- Base class functionality
- Method override capability
- YOLOX interface
- Ultralytics interface
- Utility functions

## License

AGPL-3.0 License - See LICENSE file for details
