# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from pathlib import Path
from typing import Union

from boxmot.utils import logger as LOGGER
from boxmot.engine.detectors.base import Detector

# Check if ultralytics is available
try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class Ultralytics(Detector):
    """
    Ultralytics YOLO detector (YOLOv8, YOLOv9, YOLOv10, YOLO11, etc.).
    
    Example:
        >>> from boxmot.engine.detectors import Ultralytics
        >>> detector = Ultralytics("yolov8n.pt")
        >>> boxes = detector("image.jpg")
        >>> 
        >>> # With custom parameters
        >>> detector = Ultralytics("yolov8n.pt", conf_thres=0.5, imgsz=1280)
        >>> boxes = detector("image.jpg")
    """
    
    def __init__(
        self,
        path: str,
        device: str = "cpu",
        imgsz: Union[int, list, tuple] = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        agnostic_nms: bool = False,
        classes: list = None,
        verbose: bool = False,
    ):
        """
        Initialize Ultralytics YOLO detector.
        
        Args:
            path: Path to YOLO model weights
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            imgsz: Input image size (int or [width, height])
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            agnostic_nms: Whether to use class-agnostic NMS
            classes: List of class indices to filter detections
            verbose: Whether to print verbose output
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is not installed. Install it with: pip install ultralytics"
            )
        
        self.device = device
        
        # Parse image size
        if isinstance(imgsz, int):
            self.imgsz = imgsz
        else:
            vals = imgsz if isinstance(imgsz, (list, tuple)) else (imgsz,)
            self.imgsz = vals[0]  # Ultralytics uses single value
        
        # Detection parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.verbose = verbose
        
        super().__init__(path)
    
    def _load_model(self, path: Path, **kwargs):
        """Load Ultralytics YOLO model."""
        LOGGER.info(f"Loading Ultralytics model from {str(path)}")
        model = UltralyticsYOLO(str(path))
        model.to(self.device)
        return model
    
    def preprocess(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess frame for Ultralytics YOLO.
        
        Ultralytics handles preprocessing internally, so we just return the frame.
        
        Args:
            frame: Input image as BGR numpy array
            **kwargs: Additional arguments (unused)
            
        Returns:
            Original frame (preprocessing done internally by Ultralytics)
        """
        return frame
    
    def process(self, frame: np.ndarray, **kwargs):
        """
        Run Ultralytics YOLO inference.
        
        Args:
            frame: Input frame
            **kwargs: Additional arguments passed to model
            
        Returns:
            Ultralytics Results object
        """
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            agnostic_nms=self.agnostic_nms,
            classes=self.classes,
            verbose=self.verbose,
            **kwargs
        )
        return results
    
    def postprocess(self, boxes, **kwargs) -> np.ndarray:
        """
        Postprocess Ultralytics predictions.
        
        Args:
            boxes: Ultralytics Results object(s)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Processed boxes as numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        # Handle single or multiple results
        if not isinstance(boxes, list):
            boxes = [boxes]
        
        # Extract boxes from first result
        result = boxes[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6))
        
        # Get boxes in xyxy format with confidence and class
        # Ultralytics format: [x1, y1, x2, y2, conf, cls]
        boxes_data = result.boxes.data.cpu().numpy()
        
        return boxes_data
