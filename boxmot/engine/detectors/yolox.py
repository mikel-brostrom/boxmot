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


class YoloX(Detector):
    """
    YOLOX object detector.
    
    YOLOX provides efficient object detection with models ranging from nano to x-large.
    
    Example:
        >>> from boxmot.engine.detectors import YoloXStrategy
        >>> detector = YoloXStrategy(model="yolox_s.pt", device="cpu")
        >>> boxes = detector("image.jpg")
        >>> 
        >>> # With custom parameters
        >>> detector = YoloXStrategy(model="yolox_s.pt", device="cuda", args=args)
        >>> boxes = detector("image.jpg")
    """
    
    # Class attributes for compatibility
    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = COCO_CLASSES
    
    def __init__(
        self,
        model: str,
        device: str = "cpu",
        imgsz: Union[int, list, tuple] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        agnostic_nms: bool = False,
        classes: list = None,
        args = None,
    ):
        """
        Initialize YOLOX detector.
        
        Args:
            model: Path to YOLOX model weights
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            imgsz: Input image size (int or [width, height])
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            agnostic_nms: Whether to use class-agnostic NMS
            classes: List of class indices to filter detections
            args: Legacy args object (can override parameters if provided)
        """
        if not YOLOX_AVAILABLE:
            raise ImportError(
                "YOLOX is not installed. Install it with: pip install yolox --no-deps"
            )
        
        self.device = torch.device(device)
        
        # Extract parameters from args if available (legacy interface)
        if args is not None:
            if hasattr(args, 'imgsz') and imgsz is None:
                imgsz = args.imgsz
            if hasattr(args, 'conf'):
                conf_thres = args.conf
            if hasattr(args, 'iou'):
                iou_thres = args.iou
            if hasattr(args, 'agnostic_nms'):
                agnostic_nms = args.agnostic_nms
            if hasattr(args, 'classes') and args.classes is not None:
                classes = args.classes
        
        # Parse image size (default for YOLOX is 640x640, but can be 800x1440 for ByteTrack)
        if imgsz is None:
            self.imgsz = [640, 640]
        elif isinstance(imgsz, int):
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
        
        # For storing preprocessing info (ratio for each image in batch)
        self._preproc_ratios = []
        
        super().__init__(model)

        super().__init__(model)
    
    def _get_model_type(self, path: Path) -> str:
        """Determine YOLOX model type from path."""
        for model_name in YOLOX_ZOO.keys():
            if Path(model_name).stem in str(path.name):
                return Path(model_name).stem
        # Default to yolox_s if can't determine
        return "yolox_s"
    
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
    
    def yolox_preprocess(
        self,
        image: np.ndarray,
        input_size: tuple,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        swap=(2, 0, 1),
    ) -> tuple:
        """
        YOLOX preprocessing (ByteTrack style).
        
        This matches the preprocessing from ByteTrack:
        https://github.com/ifzhang/ByteTrack/blob/main/yolox/data/data_augment.py#L189
        
        Args:
            image: Input image as numpy array
            input_size: Target size (height, width)
            mean: Mean values for normalization
            std: Std values for normalization
            swap: Channel swap order
            
        Returns:
            Tuple of (preprocessed_image, resize_ratio)
        """
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def preprocess(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess frame for YOLOX.
        
        Args:
            frame: Input image as BGR numpy array
            **kwargs: Additional arguments (unused)
            
        Returns:
            Preprocessed tensor ready for YOLOX inference
        """
        # Apply YOLOX preprocessing
        img_pre, ratio = self.yolox_preprocess(frame, input_size=self.imgsz)
        
        # Store ratio for postprocessing
        self._preproc_ratios.append(ratio)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_pre).to(self.device)
        
        return img_tensor
    
    def process(self, frame: np.ndarray, **kwargs):
        """
        Run YOLOX inference.
        
        Args:
            frame: Preprocessed tensor from preprocess()
            **kwargs: Additional arguments (unused)
            
        Returns:
            YOLOX predictions
        """
        # Ensure batch dimension
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
        
        with torch.no_grad():
            preds = self.model(frame)
        
        return preds
    
    def postprocess(self, preds, **kwargs) -> np.ndarray:
        """
        Postprocess YOLOX predictions.
        
        Args:
            preds: Raw predictions from YOLOX model
            **kwargs: Additional arguments (unused)
            
        Returns:
            Processed boxes as numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        # Get ratio from preprocessing
        if self._preproc_ratios:
            ratio = self._preproc_ratios.pop(0)
        else:
            ratio = 1.0
        
        # Apply YOLOX NMS postprocessing
        pred = postprocess(
            preds[0].unsqueeze(0) if len(preds[0].shape) == 2 else preds,
            1,
            conf_thre=self.conf_thres,
            nms_thre=self.iou_thres,
            class_agnostic=self.agnostic_nms,
        )[0]
        
        if pred is None or len(pred) == 0:
            return np.empty((0, 6))
        
        # Scale boxes back to original image size
        pred[:, 0] = pred[:, 0] / ratio
        pred[:, 1] = pred[:, 1] / ratio
        pred[:, 2] = pred[:, 2] / ratio
        pred[:, 3] = pred[:, 3] / ratio
        
        # Combine object confidence and class confidence
        pred[:, 4] *= pred[:, 5]
        
        # Reorder to [x1, y1, x2, y2, conf, cls]
        pred = pred[:, [0, 1, 2, 3, 4, 6]]
        
        # Filter by classes if specified
        if self.classes is not None:
            mask = torch.isin(pred[:, 5].cpu(), torch.as_tensor(self.classes))
            pred = pred[mask]
        
        return pred.cpu().numpy()
