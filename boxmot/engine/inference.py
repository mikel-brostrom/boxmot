# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
Unified inference module for BoxMOT.

This module provides a consistent interface for running YOLO inference across
both real-time tracking (tracker.py) and batch evaluation (evaluator.py) workflows.
It handles model initialization, callbacks, and timing instrumentation.
"""

import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Union

import numpy as np
import torch

from boxmot.detectors import (
    default_imgsz,
    get_yolo_inferer,
    is_ultralytics_model,
    is_rtdetr_model,
    is_yolox_model,
)
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.timing import TimingStats

checker = RequirementsChecker()
checker.check_packages(("ultralytics",))

from ultralytics import YOLO


class TimedReIDModel:
    """
    Wrapper around ReID model to track timing.
    
    This wrapper intercepts calls to get_features() and records the time
    spent in ReID feature extraction.
    """
    
    def __init__(self, model, timing_stats: Optional[TimingStats] = None):
        """
        Initialize the timed ReID model wrapper.
        
        Args:
            model: The underlying ReID model with get_features() method.
            timing_stats: Optional TimingStats to record ReID timing.
        """
        self._model = model
        self._timing_stats = timing_stats
    
    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Extract ReID features with timing instrumentation.
        
        Args:
            xyxys: Bounding boxes as numpy array of shape (N, 4) with [x1, y1, x2, y2].
            img: The image as numpy array (BGR).
        
        Returns:
            Feature embeddings as numpy array of shape (N, feature_dim).
        """
        t0 = time.perf_counter()
        features = self._model.get_features(xyxys, img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if self._timing_stats is not None:
            self._timing_stats.add_reid_time(elapsed_ms)
        
        return features
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self._model, name)


class DetectorReIDPipeline:
    """
    Unified pipeline for detection and ReID inference with timing instrumentation.
    
    This class provides a consistent interface for running YOLO detection and 
    ReID feature extraction across both real-time tracking and batch evaluation.
    It handles:
    - YOLO model initialization (Ultralytics and non-Ultralytics models)
    - ReID model initialization with timing wrappers
    - Batch and streaming inference
    - Comprehensive timing statistics
    
    Attributes:
        yolo_model_path: Path to the YOLO model weights.
        reid_model_paths: List of paths to ReID model weights.
        device: Device to run inference on.
        timing_stats: TimingStats instance for timing instrumentation.
    """
    
    def __init__(
        self,
        yolo_model_path: Union[str, Path],
        reid_model_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        device: str = "",
        imgsz: Optional[Union[int, List[int]]] = None,
        half: bool = False,
        timing_stats: Optional[TimingStats] = None,
    ):
        """
        Initialize the detector and ReID pipeline.
        
        Args:
            yolo_model_path: Path to the YOLO model weights.
            reid_model_paths: Path(s) to ReID model weights. Can be single path or list.
            device: Device to run inference on (e.g., 'cuda:0', 'cpu', 'mps').
            imgsz: Image size for inference. If None, uses model-specific defaults.
            half: Whether to use half precision (FP16).
            timing_stats: Optional TimingStats instance. If None, creates a new one.
        """
        self.yolo_model_path = Path(yolo_model_path)
        self.device = device
        self.half = half
        
        # Initialize or use provided timing stats
        self.timing_stats = timing_stats if timing_stats is not None else TimingStats()
        
        # Set default image size based on model type
        if imgsz is None:
            imgsz = default_imgsz(yolo_model_path)
        self.imgsz = imgsz
        
        # Determine model type
        self.is_ultralytics = is_ultralytics_model(yolo_model_path)
        self.is_yolox = is_yolox_model(yolo_model_path)
        self.is_rtdetr = is_rtdetr_model(yolo_model_path)
        
        # Initialize the base YOLO model
        placeholder = yolo_model_path if self.is_ultralytics else "yolov8n.pt"
        self.yolo = YOLO(placeholder)
        
        # Custom model instance for non-ultralytics models
        self._custom_model = None
        
        # Store user callbacks
        self._user_callbacks = {
            "on_predict_start": [],
            "on_predict_batch_start": [],
            "on_predict_postprocess_end": [],
            "on_predict_batch_end": [],
        }
        
        # Setup internal callbacks for YOLO
        self._setup_yolo_callbacks()
        
        # Initialize ReID models with timing wrappers
        self.reid_models: List[TimedReIDModel] = []
        self.reid_model_names: List[str] = []
        if reid_model_paths is not None:
            self._init_reid_models(reid_model_paths)
    
    def _init_reid_models(self, reid_model_paths: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize ReID models with timing wrappers.
        
        Args:
            reid_model_paths: Path(s) to ReID model weights.
        """
        # Import here to avoid circular imports
        from boxmot.reid.core.auto_backend import ReidAutoBackend
        
        # Normalize to list
        if isinstance(reid_model_paths, (str, Path)):
            reid_model_paths = [reid_model_paths]
        
        for reid_path in reid_model_paths:
            reid_path = Path(reid_path)
            reid_backend = ReidAutoBackend(
                weights=reid_path,
                device=self.device,
                half=self.half,
            )
            # Wrap with timing
            timed_model = TimedReIDModel(reid_backend.model, self.timing_stats)
            self.reid_models.append(timed_model)
            self.reid_model_names.append(reid_path.stem)
    
    def _setup_yolo_callbacks(self):
        """Setup internal callbacks for model initialization and timing."""
        
        # Add callback to setup custom models (YOLOX, RT-DETR)
        if not self.is_ultralytics:
            self.yolo.add_callback("on_predict_start", self._setup_custom_model)
            
            # Add callback to update image paths for YOLOX/RT-DETR
            if self.is_yolox or self.is_rtdetr:
                self.yolo.add_callback("on_predict_batch_start", self._update_custom_paths)
        
        # Add timing callback for frame start
        self.yolo.add_callback("on_predict_batch_start", self._on_frame_start)
    
    def _setup_custom_model(self, predictor):
        """Setup callback for non-ultralytics models (YOLOX, RT-DETR)."""
        if self._custom_model is not None:
            return  # Already initialized
        
        # Get the appropriate model class
        model_class = get_yolo_inferer(self.yolo_model_path)
        
        # Instantiate the model
        self._custom_model = model_class(
            model=self.yolo_model_path,
            device=predictor.device,
            args=predictor.args,
        )
        
        # Replace predictor's model
        predictor.model = self._custom_model
        
        # Override preprocess and postprocess for YOLOX/RT-DETR
        if self.is_yolox or self.is_rtdetr:
            predictor.preprocess = lambda im: self._custom_model.preprocess(im=im)
            predictor.postprocess = lambda preds, im, im0s: self._custom_model.postprocess(
                preds=preds, im=im, im0s=im0s
            )
    
    def _update_custom_paths(self, predictor):
        """Update image paths for custom models."""
        if self._custom_model is not None:
            self._custom_model.update_im_paths(predictor)
    
    def _on_frame_start(self, predictor):
        """Callback to mark start of frame processing for timing."""
        self.timing_stats.start_frame()
    
    def add_callback(self, event: str, callback: Callable):
        """
        Add a callback for a specific event.
        
        Args:
            event: Event name (e.g., 'on_predict_start', 'on_predict_postprocess_end').
            callback: Callback function to call.
        """
        # Register with the underlying YOLO model
        self.yolo.add_callback(event, callback)
        
        # Also track user callbacks
        if event in self._user_callbacks:
            self._user_callbacks[event].append(callback)
    
    def get_reid_features(
        self,
        xyxys: np.ndarray,
        img: np.ndarray,
        reid_index: int = 0,
    ) -> np.ndarray:
        """
        Extract ReID features for detections with timing.
        
        Args:
            xyxys: Bounding boxes as numpy array of shape (N, 4) with [x1, y1, x2, y2].
            img: The image as numpy array (BGR).
            reid_index: Index of the ReID model to use (default 0).
        
        Returns:
            Feature embeddings as numpy array of shape (N, feature_dim).
        """
        if not self.reid_models:
            raise ValueError("No ReID models initialized. Pass reid_model_paths to constructor.")
        
        if reid_index >= len(self.reid_models):
            raise ValueError(f"ReID index {reid_index} out of range. Have {len(self.reid_models)} models.")
        
        return self.reid_models[reid_index].get_features(xyxys, img)
    
    def get_all_reid_features(
        self,
        xyxys: np.ndarray,
        img: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Extract ReID features from all loaded ReID models.
        
        Args:
            xyxys: Bounding boxes as numpy array of shape (N, 4) with [x1, y1, x2, y2].
            img: The image as numpy array (BGR).
        
        Returns:
            Dictionary mapping model name to feature embeddings.
        """
        results = {}
        for model, name in zip(self.reid_models, self.reid_model_names):
            results[name] = model.get_features(xyxys, img)
        return results
    
    def predict(
        self,
        source: Union[str, Path, List, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.7,
        agnostic_nms: bool = False,
        classes: Optional[List[int]] = None,
        stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Generator:
        """
        Run YOLO inference on the source.
        
        Args:
            source: Source for inference (path, list of images, or numpy array).
            conf: Confidence threshold for detections.
            iou: IoU threshold for NMS.
            agnostic_nms: Whether to use class-agnostic NMS.
            classes: List of class indices to filter.
            stream: Whether to return a generator (for memory efficiency).
            verbose: Whether to print verbose output.
            **kwargs: Additional arguments passed to YOLO.predict().
        
        Yields:
            Results from inference.
        """
        results = self.yolo.predict(
            source=source,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms,
            device=self.device,
            classes=classes,
            imgsz=self.imgsz,
            stream=stream,
            verbose=verbose,
            **kwargs,
        )
        
        # Wrap results to record timing from Ultralytics
        for result in results:
            # Record Ultralytics timing
            if hasattr(result, 'speed') and result.speed:
                self.timing_stats.totals['preprocess'] += result.speed.get('preprocess', 0) or 0
                self.timing_stats.totals['inference'] += result.speed.get('inference', 0) or 0
                self.timing_stats.totals['postprocess'] += result.speed.get('postprocess', 0) or 0
            
            yield result
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.7,
        agnostic_nms: bool = False,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> List:
        """
        Run batch inference on a list of images.
        
        This is more efficient for batch processing in evaluation workflows.
        
        Args:
            images: List of numpy arrays (BGR images).
            conf: Confidence threshold for detections.
            iou: IoU threshold for NMS.
            agnostic_nms: Whether to use class-agnostic NMS.
            classes: List of class indices to filter.
            verbose: Whether to print verbose output.
            **kwargs: Additional arguments passed to YOLO.predict().
        
        Returns:
            List of Results from inference.
        """
        results = self.yolo.predict(
            source=images,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms,
            device=self.device,
            classes=classes,
            imgsz=self.imgsz,
            stream=False,
            verbose=verbose,
            **kwargs,
        )
        
        # Record timing for batch
        for result in results:
            if hasattr(result, 'speed') and result.speed:
                self.timing_stats.totals['preprocess'] += result.speed.get('preprocess', 0) or 0
                self.timing_stats.totals['inference'] += result.speed.get('inference', 0) or 0
                self.timing_stats.totals['postprocess'] += result.speed.get('postprocess', 0) or 0
        
        return results
    
    def warmup(self):
        """
        Warmup the model with a dummy inference.
        
        This helps ensure consistent timing for the first real inference.
        """
        try:
            if isinstance(self.imgsz, (list, tuple)):
                h, w = int(self.imgsz[0]), int(self.imgsz[1])
            else:
                h = w = int(self.imgsz)
            
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            _ = self.yolo.predict(
                source=[dummy],
                device=self.device,
                verbose=False,
                imgsz=self.imgsz,
            )
        except Exception as e:
            LOGGER.warning(f"Warmup failed: {e}")
    
    def autotune_batch_size(self, requested_batch_size: int) -> int:
        """
        Autotune batch size to find the maximum that fits in memory.
        
        Args:
            requested_batch_size: Initial batch size to try.
        
        Returns:
            The maximum batch size that fits in memory.
        """
        dev_lower = str(self.device).lower()
        use_accel = dev_lower.startswith(("cuda", "0", "1", "2", "3", "4", "5", "6", "7", "mps", "metal"))
        if not use_accel:
            return max(1, requested_batch_size)
        
        def _empty_cache():
            if dev_lower.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        
        if isinstance(self.imgsz, (list, tuple)):
            h, w = int(self.imgsz[0]), int(self.imgsz[1])
        else:
            h = w = int(self.imgsz)
        
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        
        bs = max(1, int(requested_batch_size))
        while bs >= 1:
            try:
                self.yolo.predict(
                    source=[dummy] * bs,
                    device=self.device,
                    verbose=False,
                    imgsz=self.imgsz,
                )
                if bs < requested_batch_size:
                    LOGGER.info(f"Auto-tuned batch size: {bs} (requested: {requested_batch_size})")
                return bs
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                _empty_cache()
                next_bs = max(1, bs // 2)
                LOGGER.warning(f"Batch size {bs} OOM; trying {next_bs}.")
                if next_bs == bs:
                    break
                bs = next_bs
        
        raise RuntimeError("Unable to run even batch size 1; reduce image size or move to CPU.")
    
    def print_timing_summary(self):
        """Print the timing summary."""
        self.timing_stats.print_summary()
    
    @property
    def custom_model(self):
        """Get the custom model instance (for YOLOX/RT-DETR)."""
        return self._custom_model


# Backwards compatibility alias
YOLOInference = DetectorReIDPipeline


def extract_detections(result) -> np.ndarray:
    """
    Extract detections from a YOLO result in a consistent format.
    
    Args:
        result: A YOLO result object.
    
    Returns:
        numpy array of shape (N, 6) with columns [x1, y1, x2, y2, conf, cls].
        Returns empty array (0, 6) if no detections.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6))
    
    return result.boxes.data.cpu().numpy()


def filter_detections(
    dets: np.ndarray,
    min_area: float = 10.0,
    remove_degenerate: bool = True,
) -> np.ndarray:
    """
    Filter detections to remove invalid boxes.
    
    Args:
        dets: Detection array of shape (N, 6+) with [x1, y1, x2, y2, conf, cls, ...].
        min_area: Minimum box area to keep.
        remove_degenerate: Whether to remove boxes with non-positive dimensions.
    
    Returns:
        Filtered detection array.
    """
    if len(dets) == 0:
        return dets
    
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    
    # Remove degenerate boxes
    if remove_degenerate:
        valid = (x2 > x1) & (y2 > y1)
        dets = dets[valid]
        if len(dets) == 0:
            return dets
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    
    # Remove tiny boxes
    areas = (x2 - x1) * (y2 - y1)
    valid_area = areas >= min_area
    
    return dets[valid_area]
