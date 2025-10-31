# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
import torch
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
    
    def __call__(self, im, **kwargs):
        """
        Run inference (for Ultralytics predictor compatibility).
        
        This method is called by the Ultralytics predictor as `self.model(preprocessed)`.
        
        Args:
            im: Preprocessed RGB image(s)
            **kwargs: Additional arguments
            
        Returns:
            RF-DETR detections object (passed to postprocess)
        """
        return self.process(im, **kwargs)
    
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
    
    def postprocess(self, preds, **kwargs) -> Union[np.ndarray, list]:
        """
        Postprocess RF-DETR predictions.
        
        Args:
            preds: RF-DETR detections object (from __call__ or process method)
            **kwargs: Additional arguments including:
                - im: preprocessed images (for Ultralytics pipeline)
                - im0s: original images (for Ultralytics pipeline)
                - predictor: Ultralytics predictor instance (for path extraction)
            
        Returns:
            Processed boxes as numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
            When used in Ultralytics pipeline (im0s provided), returns list of Results
        """
        # preds is the detections object from __call__
        detections = preds
        
        # Check if we're in Ultralytics pipeline (has im0s parameter)
        im0s = kwargs.get('im0s', None)
        in_ultralytics_pipeline = im0s is not None
        
        # Check if we have any detections
        if detections is None or len(detections.xyxy) == 0:
            boxes = np.empty((0, 6))
            boxes_tensor = torch.empty((0, 6))
        else:
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
            
            # Create tensor version for Results
            boxes_tensor = torch.from_numpy(boxes)
        
        # If in Ultralytics pipeline, create Results object
        if in_ultralytics_pipeline:
            try:
                from ultralytics.engine.results import Results
                # Get the original image (RFDETR processes single images)
                orig_img = im0s[0] if isinstance(im0s, list) else im0s
                
                # Try to get path from predictor batch if available
                path = ''
                try:
                    if isinstance(kwargs, dict):
                        predictor = kwargs.get('predictor', None)
                        if predictor and hasattr(predictor, 'batch') and predictor.batch:
                            im_files = predictor.batch.get('im_file', None)
                            if im_files:
                                path = im_files[0] if isinstance(im_files, list) else im_files
                except Exception:
                    pass
                
                # Create Results object with torch tensor (not numpy array)
                result = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.names,
                    boxes=boxes_tensor,  # Pass torch.Tensor, not numpy
                )
                return [result]
            except Exception as e:
                LOGGER.warning(f"Failed to create Results object: {e}")
                return boxes
        else:
            return boxes
