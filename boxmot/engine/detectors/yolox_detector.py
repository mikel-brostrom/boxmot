# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import fnmatch
import gdown
import numpy as np
import torch
from pathlib import Path
from typing import Union

from boxmot.utils import logger as LOGGER
from boxmot.engine.detectors.base import Detector

# Check if yolox is available
try:
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    from yolox.utils.model_utils import fuse_model
    YOLOX_AVAILABLE = True
except ImportError:
    YOLOX_AVAILABLE = False

# default model weights for these model names
YOLOX_ZOO = {
    "yolox_n.pt": "https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
    "yolox_s.pt": "https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
    "yolox_m.pt": "https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
    "yolox_l.pt": "https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
    "yolox_x.pt": "https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
    "yolox_x_MOT17_ablation.pt": "https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob",
    "yolox_x_MOT20_ablation.pt": "https://drive.google.com/uc?id=1H1BxOfinONCSdQKnjGq0XlRxVUo_4M8o",
    "yolox_x_dancetrack_ablation.pt": "https://drive.google.com/uc?id=1ZKpYmFYCsRdXuOL60NRuc7VXAFYRskXB",
}

# COCO class names
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}


class YOLOX(Detector):
    """
    YOLOX object detector.
    
    Example:
        >>> from boxmot.engine.detectors import YOLOX
        >>> detector = YOLOX("yolox_s.pt")
        >>> boxes = detector("image.jpg")
        >>> 
        >>> # With custom preprocessing
        >>> def my_preprocess(frame, **kwargs):
        >>>     # Custom logic here
        >>>     return processed_frame
        >>> detector.preprocess = my_preprocess
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
    ):
        """
        Initialize YOLOX detector.
        
        Args:
            path: Path to YOLOX model weights
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            imgsz: Input image size (int or [width, height])
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            agnostic_nms: Whether to use class-agnostic NMS
            classes: List of class indices to filter detections
        """
        if not YOLOX_AVAILABLE:
            raise ImportError(
                "YOLOX is not installed. Install it with: pip install yolox --no-deps"
            )
        
        self.device = torch.device(device)
        
        # Parse image size
        if isinstance(imgsz, int):
            self.imgsz = [imgsz, imgsz]
        else:
            vals = imgsz if isinstance(imgsz, (list, tuple)) else (imgsz,)
            w, h = (vals * 2)[:2]
            self.imgsz = [w, h]
        
        # Detection parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        
        # For storing preprocessing info
        self._preproc_ratio = None
        
        super().__init__(path)
    
    def _load_model(self, path: Path, **kwargs):
        """Load YOLOX model."""
        # Determine model type
        model_type = self._get_model_type(path)
        
        if model_type == "yolox_n":
            exp = get_exp(None, "yolox_nano")
        else:
            exp = get_exp(None, model_type)
        
        LOGGER.info(f"Loading {model_type} with {str(path)}")
        
        # Download model if needed
        if not path.exists() and (
            path.stem == model_type or fnmatch.fnmatch(path.stem, "yolox_x_*_ablation")
        ):
            LOGGER.info("Downloading pretrained weights...")
            gdown.download(
                url=YOLOX_ZOO[path.stem + ".pt"], output=str(path), quiet=False
            )
            # needed for bytetrack yolox people models
            exp.num_classes = 1
        elif path.stem.startswith(model_type):
            exp.num_classes = 1
        
        # Load checkpoint
        ckpt = torch.load(str(path), map_location=torch.device("cpu"))
        
        # Build and load model
        model = exp.get_model()
        model.eval()
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_model_type(self, path: Path):
        """Determine YOLOX model type from filename."""
        for key in YOLOX_ZOO.keys():
            if Path(key).stem in str(path.name):
                return str(Path(key).with_suffix(""))
        # Default to yolox_s if can't determine
        return "yolox_s"
    
    def preprocess(self, frame: np.ndarray, **kwargs) -> torch.Tensor:
        """
        Preprocess frame for YOLOX inference.
        
        This follows ByteTrack's preprocessing approach.
        
        Args:
            frame: Input image as BGR numpy array
            **kwargs: Additional arguments (unused)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Preprocessing parameters
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        swap = (2, 0, 1)  # HWC to CHW
        
        # Create padded image
        if len(frame.shape) == 3:
            padded_img = np.ones((self.imgsz[0], self.imgsz[1], 3)) * 114.0
        else:
            padded_img = np.ones(self.imgsz) * 114.0
        
        # Calculate resize ratio
        r = min(self.imgsz[0] / frame.shape[0], self.imgsz[1] / frame.shape[1])
        self._preproc_ratio = r
        
        # Resize image
        resized_img = cv2.resize(
            frame,
            (int(frame.shape[1] * r), int(frame.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        
        # Place resized image in padded canvas
        padded_img[: int(frame.shape[0] * r), : int(frame.shape[1] * r)] = resized_img
        
        # Normalize
        padded_img = padded_img[:, :, ::-1]  # BGR to RGB
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        
        # Transpose to CHW
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        
        # Convert to tensor
        return torch.from_numpy(padded_img).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def process(self, frame: torch.Tensor, **kwargs):
        """
        Run YOLOX inference.
        
        Args:
            frame: Preprocessed frame tensor
            **kwargs: Additional arguments (unused)
            
        Returns:
            Raw model predictions
        """
        return self.model(frame)
    
    def postprocess(self, boxes, **kwargs) -> np.ndarray:
        """
        Postprocess YOLOX predictions.
        
        Args:
            boxes: Raw predictions from model
            **kwargs: Additional arguments (unused)
            
        Returns:
            Processed boxes as numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        # Apply NMS
        pred = postprocess(
            boxes,
            1,
            conf_thre=self.conf_thres,
            nms_thre=self.iou_thres,
            class_agnostic=self.agnostic_nms,
        )[0]
        
        if pred is None:
            return np.empty((0, 6))
        
        # Scale boxes back to original image size
        if self._preproc_ratio is not None:
            pred[:, 0] = pred[:, 0] / self._preproc_ratio
            pred[:, 1] = pred[:, 1] / self._preproc_ratio
            pred[:, 2] = pred[:, 2] / self._preproc_ratio
            pred[:, 3] = pred[:, 3] / self._preproc_ratio
        
        # Combine objectness and class confidence
        pred[:, 4] *= pred[:, 5]
        pred = pred[:, [0, 1, 2, 3, 4, 6]]
        
        # Filter by class if specified
        if self.classes is not None:
            pred = pred[np.isin(pred[:, 5].cpu().numpy(), self.classes)]
        
        return pred.cpu().numpy()
