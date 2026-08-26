# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import fuse_model

from boxmot.detectors.base import BaseDetectorBackend, Detections, filter_detections
from boxmot.detectors.registry import YOLOX_MODELS, get_detector_url, load_detector_cfg
from boxmot.utils import logger as LOGGER
from boxmot.utils.misc import resolve_model_path


def _coerce_torch_dtype(dtype, fallback: torch.Tensor) -> torch.dtype:
    """Map YOLOX's dtype strings (e.g., 'torch.mps.FloatTensor') to real torch dtypes."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        lowered = dtype.lower()
        if "bfloat16" in lowered:
            return torch.bfloat16
        if "float16" in lowered or "half" in lowered:
            return torch.float16
    return fallback.dtype if isinstance(fallback, torch.Tensor) else torch.float32


def _patch_yolox_head_decode_outputs_for_mps() -> None:
    """Monkeypatch YOLOXHead.decode_outputs to work on MPS (avoids .type with dtype strings)."""
    try:
        from yolox.models.yolo_head import YOLOXHead
        from yolox.utils import meshgrid
    except Exception:
        return

    if getattr(YOLOXHead, "_boxmot_mps_patched", False):
        return

    def decode_outputs(self, outputs, dtype):
        dtype = _coerce_torch_dtype(dtype, outputs)
        device = outputs.device
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid(
                [
                    torch.arange(hsize, device=device),
                    torch.arange(wsize, device=device),
                ]
            )
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, device=device, dtype=grid.dtype))

        grids = torch.cat(grids, dim=1).to(device=device, dtype=dtype)
        strides = torch.cat(strides, dim=1).to(device=device, dtype=dtype)

        outputs = outputs.clone()
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    YOLOXHead.decode_outputs = decode_outputs
    YOLOXHead._boxmot_mps_patched = True


_patch_yolox_head_decode_outputs_for_mps()


class YoloXDetector(BaseDetectorBackend):
    """YOLOX detector with standalone preprocess/process/postprocess pipeline."""

    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = {0: "person"}

    def __init__(self, model: str | Path, device, imgsz=None) -> None:
        model_path = resolve_model_path(model)
        raw_size = imgsz or 640
        values = list(raw_size) if isinstance(raw_size, (list, tuple)) else [raw_size]
        if len(values) == 1:
            values *= 2
        self.imgsz = [int(values[0]), int(values[1])]

        detector_cfg = load_detector_cfg(model_path)
        configured_names = detector_cfg.get("classes", {})
        self.num_classes = len(configured_names) if configured_names else 1
        self.names = dict(configured_names) if configured_names else {0: "person"}
        model_type = self._get_model_type(YOLOX_MODELS, model_path)

        if model_type == "yolox_n":
            exp_name = "yolox_nano"
        else:
            exp_name = model_type
        exp = get_exp(None, exp_name)
        exp.num_classes = self.num_classes

        LOGGER.info(f"Loading {model_type} with {model_path}")

        if not model_path.exists():
            from boxmot.utils.download import download_file

            configured_url = get_detector_url(model_path)
            if not configured_url:
                raise FileNotFoundError(
                    f"Detector weights not found: {model_path}. No download URL in detector configs."
                )
            LOGGER.info("Downloading detector weights from config...")
            download_file(url=configured_url, dest=model_path, overwrite=False)

        checkpoint = torch.load(str(model_path), map_location=torch.device("cpu"))

        self.device = device
        self.model = exp.get_model()
        self.model.eval()
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model = fuse_model(self.model)
        self._preproc_data = []
        self._im0s = []

    @staticmethod
    def _get_model_type(model_names, weight_path):
        weight_name = Path(str(weight_path)).name.lower()
        for name in model_names:
            if name.lower() in weight_name:
                return name
        return "yolox_s"

    # This preprocess matches ByteTrack's implementation:
    # https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/data/data_augment.py#L189
    def _letterbox(
        self,
        image,
        input_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        if len(image.shape) == 3:
            padded = np.full((input_size[0], input_size[1], 3), 114.0, dtype=np.float32)
        else:
            padded = np.full(input_size, 114.0, dtype=np.float32)
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized
        padded = padded[:, :, ::-1] / 255.0
        padded = (padded - mean) / std
        padded = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
        return padded, r

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        if not isinstance(images, list):
            raise TypeError("YOLOX preprocess expects a list of images.")
        self._im0s = images
        self._preproc_data = []
        tensors = []
        for img in images:
            t, ratio = self._letterbox(img, input_size=self.imgsz)
            tensors.append(torch.from_numpy(t).unsqueeze(0).to(self.device))
            self._preproc_data.append(ratio)
        return torch.vstack(tensors)

    @torch.no_grad()
    def process(self, preprocessed: torch.Tensor) -> torch.Tensor:
        if preprocessed.ndim == 3:
            preprocessed = preprocessed.unsqueeze(0)
        return self.model(preprocessed)

    def postprocess(
        self,
        detections,
        conf=0.25,
        iou=0.7,
        classes=None,
        agnostic_nms=False,
        **kwargs,
    ) -> list[Detections]:
        results: list[Detections] = []
        for i, det in enumerate(detections):
            orig_img = self._im0s[i] if i < len(self._im0s) else None

            filtered = postprocess(
                det.unsqueeze(0),
                getattr(self, "num_classes", 1),
                conf_thre=conf,
                nms_thre=iou,
                class_agnostic=agnostic_nms,
            )[0]

            if filtered is None:
                boxes = np.empty((0, 6), dtype=np.float32)
            else:
                ratio = self._preproc_data[i]
                filtered[:, :4] /= ratio
                filtered[:, 4] *= filtered[:, 5]  # obj_conf * class_conf → final conf
                filtered = filtered[:, [0, 1, 2, 3, 4, 6]]  # drop class_conf column
                boxes = filter_detections(
                    filtered.detach().cpu().numpy(),
                    confidence=None,
                    classes=classes,
                )

            results.append(Detections(dets=boxes, orig_img=orig_img, names=self.names))

        return results
