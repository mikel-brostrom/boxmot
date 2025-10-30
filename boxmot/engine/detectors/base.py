# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
from pathlib import Path
from typing import Union


def resolve_image(image: Union[np.ndarray, str]) -> np.ndarray:
    """
    Resolve image input to numpy array.
    
    Args:
        image: Either a numpy array or a path to an image file
        
    Returns:
        np.ndarray: Image as numpy array in BGR format
    """
    if isinstance(image, str):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")
    elif isinstance(image, np.ndarray):
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D image array, got shape: {image.shape}")
    else:
        raise TypeError(f"Expected str or np.ndarray, got {type(image)}")
    
    return image


class Detector:
    """
    Base class for object detectors.
    
    This class provides a standardized interface for all detector implementations.
    Subclasses should implement preprocess, process, and postprocess methods.
    
    Example:
        >>> from boxmot.engine.detectors import YOLOX
        >>> detector = YOLOX("model.pt")
        >>> 
        >>> # Override methods if needed
        >>> def custom_preprocess(frame, **kwargs):
        >>>     # Custom preprocessing logic
        >>>     return processed_frame
        >>> 
        >>> detector.preprocess = custom_preprocess
        >>> 
        >>> # Use detector
        >>> boxes = detector(image)
        >>> # or
        >>> boxes = detector("path/to/image.jpg")
    """
    
    def __init__(self, path: str, **kwargs):
        """
        Initialize detector.
        
        Args:
            path: Path to model weights
            **kwargs: Additional arguments for model initialization
        """
        self.path = Path(path)
        self.model = self._load_model(self.path, **kwargs)
    
    def _load_model(self, path: Path, **kwargs):
        """
        Load model weights.
        
        Args:
            path: Path to model weights
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model object
        """
        raise NotImplementedError("Subclasses must implement _load_model")
    
    def preprocess(self, frame: np.ndarray, **kwargs):
        """
        Preprocess input frame before detection.
        
        Args:
            frame: Input image as numpy array
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed frame ready for model inference
        """
        raise NotImplementedError("Subclasses must implement preprocess")
    
    def process(self, frame, **kwargs):
        """
        Run model inference on preprocessed frame.
        
        Args:
            frame: Preprocessed frame
            **kwargs: Additional inference arguments
            
        Returns:
            Raw model predictions
        """
        raise NotImplementedError("Subclasses must implement process")
    
    def postprocess(self, boxes, **kwargs):
        """
        Postprocess raw model predictions.
        
        Args:
            boxes: Raw predictions from model
            **kwargs: Additional postprocessing arguments
            
        Returns:
            Processed detection results
        """
        raise NotImplementedError("Subclasses must implement postprocess")
    
    def warmup(self, imgsz=(640, 640), n=3):
        """
        Warm up the model by running dummy inferences.
        
        This is useful for:
        - GPU initialization and memory allocation
        - JIT compilation (e.g., TorchScript)
        - Cache warming
        - Getting accurate timing for subsequent inferences
        
        Args:
            imgsz: Image size as (height, width) tuple or single int
            n: Number of warmup iterations (default: 3)
        
        Example:
            >>> detector = YOLOX("model.pt", device="cuda")
            >>> detector.warmup(imgsz=640, n=5)  # Warm up with 5 iterations
            >>> # Now subsequent detections will be faster
            >>> boxes = detector("image.jpg")
        """
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        
        # Create dummy image
        dummy_image = np.zeros((imgsz[0], imgsz[1], 3), dtype=np.uint8)
        
        # Run n warmup iterations
        for _ in range(n):
            try:
                _ = self(dummy_image)
            except Exception:
                # If warmup fails, it's not critical - just continue
                pass
    
    def __call__(self, image: Union[np.ndarray, str], **kwargs):
        """
        Run detection on image.
        
        Args:
            image: Either a numpy array or path to image file
            **kwargs: Additional arguments passed to preprocess, process, and postprocess
            
        Returns:
            Detection results after postprocessing
        """
        image = resolve_image(image)
        frame = self.preprocess(image, **kwargs)
        boxes = self.process(frame, **kwargs)
        boxes = self.postprocess(boxes, **kwargs)
        return boxes
