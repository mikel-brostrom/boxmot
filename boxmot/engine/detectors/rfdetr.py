# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
from pathlib import Path
from typing import Union

from boxmot.utils import logger as LOGGER
from boxmot.engine.detectors.base import Detector

# Check if rfdetr is available
try:
    from rfdetr import RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    COCO_CLASSES = {}


class RFDETR(Detector):
    """
    RF-DETR (Real-time DEtection TRansformer) object detector.
    
    RF-DETR is an ONNX-based detector that provides fast inference.
    
    Example:
        >>> from boxmot.engine.detectors import RFDETR
        >>> detector = RFDETR(model="rfdetr-l.onnx")
        >>> boxes = detector("image.jpg")
        >>> 
        >>> # With custom confidence threshold
        >>> detector = RFDETR(model="rfdetr-l.onnx", conf_thres=0.5)
        >>> boxes = detector("image.jpg")
        
        >>> # Called from track.py via get_yolo_inferer()
        >>> detector = RFDETR(model="rfdetr-l.onnx", device="cpu", args=args)
    """
    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = COCO_CLASSES
    ch = 3
    
    def __init__(
        self,
        model: str,
        device: str = "cpu",
        conf_thres: float = 0.25,
        classes: list = None,
        args = None,
    ):
        """
        Initialize RF-DETR detector.
        
        Args:
            model: Path to RF-DETR ONNX model
            device: Device to run inference on ('cpu', 'cuda', or torch.device object)
                   Note: RF-DETR uses ONNX Runtime, device handling is internal
            conf_thres: Confidence threshold for detections
            classes: List of class indices to filter detections
            args: Legacy args object (ignored, for compatibility)
        """
        if not RFDETR_AVAILABLE:
            raise ImportError(
                "RF-DETR is not installed. Install it with: pip install rfdetr"
            )
        
        # Convert torch.device to string if needed
        import torch
        if isinstance(device, torch.device):
            device = str(device.type)
        
        self.device = device
        self.conf_thres = conf_thres
        self.classes = classes
        
        super().__init__(model)
    
    def _load_model(self, path: Path, **kwargs):
        """Load RF-DETR model."""
        LOGGER.info(f"Loading RF-DETR model from {str(path)}")
        
        # RF-DETR uses ONNX Runtime internally
        # Device handling is done by the library
        model = RFDETRBase(device=self.device)
        
        return model
    
    def preprocess(self, im, **kwargs):
        """
        Preprocess image(s) for RF-DETR.
        
        RF-DETR handles preprocessing internally, but we need to convert
        BGR to RGB as it expects RGB input.
        
        Args:
            im: Input image as BGR numpy array or list of images
            **kwargs: Additional arguments (unused)
            
        Returns:
            RGB image(s) ready for RF-DETR inference
        """
        from boxmot.engine.detectors.base import resolve_image
        
        # Handle list of images (batch processing)
        if isinstance(im, list):
            return [self.preprocess(img, **kwargs) for img in im]
        
        # Resolve image to numpy array
        im = resolve_image(im)
        
        # Convert BGR to RGB (RF-DETR expects RGB)
        if len(im.shape) == 3 and im.shape[2] == 3:
            frame_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = im
        
        return frame_rgb
    
    def process(self, im: np.ndarray, **kwargs):
        """
        Run RF-DETR inference.
        
        Args:
            im: Preprocessed RGB im
            **kwargs: Additional arguments (unused)
            
        Returns:
            RF-DETR detections object
        """
        # RF-DETR handles inference internally
        detections = self.model.predict(im, threshold=self.conf_thres)
        return detections
    
    def postprocess(self, detections, **kwargs) -> np.ndarray:
        """
        Postprocess RF-DETR predictions.
        
        Args:
            detections: RF-DETR detections object
            **kwargs: Additional arguments (unused)
            
        Returns:
            Processed boxes as numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        # Check if we have any detections
        if detections is None or len(detections.xyxy) == 0:
            return np.empty((0, 6))
        
        # RF-DETR returns detections with xyxy, confidence, and class_id
        # Combine them into the standard [N, 6] format
        boxes = np.column_stack([
            detections.xyxy,                          # [N, 4] - x1, y1, x2, y2
            detections.confidence[:, np.newaxis],     # [N, 1] - confidence
            detections.class_id[:, np.newaxis]        # [N, 1] - class_id
        ])
        
        # Filter by class if specified
        if self.classes is not None:
            class_mask = np.isin(boxes[:, 5].astype(int), self.classes)
            boxes = boxes[class_mask]
        
        return boxes
