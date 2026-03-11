import os
from abc import abstractmethod
from pathlib import Path

import cv2
import gdown
import numpy as np
import torch
from filelock import SoftFileLock

from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.utils import WEIGHTS, logger as LOGGER
from boxmot.utils.checks import RequirementsChecker


class BaseModelBackend:
    def __init__(self, weights, device, half):
        self.weights = weights[0] if isinstance(weights, list) else weights
        if isinstance(self.weights, str):
            self.weights = Path(self.weights)
        self.weights = WEIGHTS / self.weights.name
        LOGGER.info(self.weights)
        self.device = device
        self.half = half
        self.model = None
        self.cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.download_model(self.weights)
        self.model_name = ReIDModelRegistry.get_model_name(self.weights)

        self.model = ReIDModelRegistry.build_model(
            self.model_name,
            self.weights,
            num_classes=ReIDModelRegistry.get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=device,
        )
        self.checker = RequirementsChecker()
        self.load_model(self.weights)

        self.mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if "clip" in self.model_name:
            self.mean_array = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
            self.std_array = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

        # Determine input shape, depending on dataset and model name
        if "vehicleid" in self.weights.name or "veri" in self.weights.name:
            input_shape = (256, 256)
        elif "lmbn" in self.model_name:
            input_shape = (384, 128)
        elif "hacnn" in self.model_name:
            input_shape = (160, 64)
        else: 
            input_shape = (256, 128)
        self.input_shape = input_shape

    @staticmethod
    def _obb_to_xyxy(box: np.ndarray) -> np.ndarray:
        """Convert a single OBB `[cx, cy, w, h, angle]` to its enclosing AABB."""
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        cx, cy, bw, bh, angle = box[:5]
        rect = ((float(cx), float(cy)), (max(float(bw), 1e-4), max(float(bh), 1e-4)), float(np.degrees(angle)))
        corners = cv2.boxPoints(rect)
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""
        corners = np.asarray(corners, dtype=np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1).reshape(-1)
        ordered[0] = corners[np.argmin(s)]
        ordered[2] = corners[np.argmax(s)]
        ordered[1] = corners[np.argmin(d)]
        ordered[3] = corners[np.argmax(d)]
        return ordered

    @staticmethod
    def _crop_obb(box: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Extract a rectified crop from an oriented box `[cx, cy, w, h, angle]`."""
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        cx, cy, bw, bh, angle = box[:5]
        bw = max(float(bw), 1.0)
        bh = max(float(bh), 1.0)
        rect = ((float(cx), float(cy)), (bw, bh), float(np.degrees(angle)))
        src = BaseModelBackend._order_corners(cv2.boxPoints(rect))
        dst = np.array(
            [[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            img,
            matrix,
            (max(int(round(bw)), 1), max(int(round(bh)), 1)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    @staticmethod
    def _is_obb_box(box: np.ndarray) -> bool:
        """Return whether a single row is in one of the supported OBB layouts."""
        return np.asarray(box).reshape(-1).shape[0] in (5, 7, 9)

    @classmethod
    def _boxes_to_xyxy(cls, boxes: np.ndarray) -> np.ndarray:
        """
        Normalize AABB/OBB detections to `[x1, y1, x2, y2]` for ReID cropping.

        Accepted layouts:
        - AABB: `[x1, y1, x2, y2]` or rows with at least 4 leading AABB coordinates
        - OBB: `[cx, cy, w, h, angle]`, `[cx, cy, w, h, angle, conf, cls]`,
          or track outputs with 9 leading OBB fields.
        """
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return boxes.reshape(0, 4)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        if boxes.shape[1] in (5, 7, 9):
            return np.vstack([cls._obb_to_xyxy(box[:5]) for box in boxes]).astype(np.float32)

        if boxes.shape[1] < 4:
            raise ValueError("Expected detections with at least 4 coordinates")

        return boxes[:, :4].astype(np.float32, copy=False)

    def get_crops(self, xyxys, img):
        h, w = img.shape[:2]
        interpolation_method = cv2.INTER_LINEAR
        xyxys = np.asarray(xyxys, dtype=np.float32)
        if xyxys.size == 0:
            xyxys = xyxys.reshape(0, 4)
        elif xyxys.ndim == 1:
            xyxys = xyxys.reshape(1, -1)
        
        # Preallocate tensor for crops
        num_crops = len(xyxys)
        crops = torch.empty(
            (num_crops, 3, *self.input_shape),
            dtype=torch.half if self.half else torch.float,
            device=self.device,
        )

        for i, box in enumerate(xyxys):
            if self._is_obb_box(box):
                crop = self._crop_obb(box[:5], img)
            else:
                x1, y1, x2, y2 = box[:4].round().astype("int")
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                if x2 <= x1:
                    x1 = min(max(0, x1), max(0, w - 1))
                    x2 = min(w, x1 + 1)
                if y2 <= y1:
                    y1 = min(max(0, y1), max(0, h - 1))
                    y2 = min(h, y1 + 1)

                crop = img[y1:y2, x1:x2]

            # Resize and convert color in one step
            crop = cv2.resize(
                crop,
                (self.input_shape[1], self.input_shape[0]),
                interpolation=interpolation_method,
            )
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Convert to tensor and normalize (convert to [0, 1] by dividing by 255 in batch later)
            crop = torch.from_numpy(crop).to(
                self.device, dtype=torch.half if self.half else torch.float
            )
            crops[i] = torch.permute(crop, (2, 0, 1))  # Change to (C, H, W)

        # Normalize the entire batch in one go
        crops = crops / 255.0

        # Standardize the batch
        crops = (crops - self.mean_array) / self.std_array

        return crops

    @torch.no_grad()
    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(
                xyxys=np.array([[0, 0, 64, 64], [0, 0, 128, 128]]), img=im
            )
            crops = self.inference_preprocess(crops)
            self.forward(crops)  # warmup

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def inference_preprocess(self, x):
        if self.half:
            if isinstance(x, torch.Tensor):
                if x.dtype != torch.float16:
                    x = x.half()
            elif isinstance(x, np.ndarray):
                if x.dtype != np.float16:
                    x = x.astype(np.float16)

        if self.nhwc:
            if isinstance(x, torch.Tensor):
                x = x.permute(0, 2, 3, 1)  # Convert from NCHW to NHWC
            elif isinstance(x, np.ndarray):
                x = np.transpose(x, (0, 2, 3, 1))  # Convert from NCHW to NHWC
        return x

    def inference_postprocess(self, features):
        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    @abstractmethod
    def forward(self, im_batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def load_model(self, w):
        raise NotImplementedError("This method should be implemented by subclasses.")


    def download_model(self, w):
        if isinstance(w, str): 
            w = Path(w)
        w = WEIGHTS / w.name

        if w.suffix != ".pt":
            return

        w.parent.mkdir(parents=True, exist_ok=True)

        model_url = ReIDModelRegistry.get_model_url(w)
        lock = SoftFileLock(str(w) + ".lock", timeout=300)  # Wait up to 5 minutes

        with lock:
            if w.exists() or "openvino" in w.name:
                LOGGER.info(f"[PID {os.getpid()}] Found existing ReID weights at {w}; skipping download.")
                return

            if model_url:
                LOGGER.info(f"[PID {os.getpid()}] Downloading ReID weights from {model_url} → {w}")
                gdown.download(model_url, str(w), quiet=False)
            else:
                LOGGER.error(
                    f"No URL associated with the chosen ReID weights ({w}).\n"
                    f"Choose one of the following:"
                )
                ReIDModelRegistry.show_downloadable_models()